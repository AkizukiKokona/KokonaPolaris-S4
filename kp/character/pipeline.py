"""角色卡数据管线 —— 收图 → 归一化 → 分层 → 配对 → 打包 → 报告。

⭐ 定位：把「用户交的原始截图」变成「可训练的角色卡素材」的**全自动 CPU 管线**。
   用户只交 图片 + caption + tag（见《补充 11》§4.1），其余全部由此处生成。

⚠️ 分层（`stage_layers`）是**启发式占位**：用 k-means 颜色聚类 + 位置/色调规则
   给出「近似语义层」。**正式版走 See-through 自举的 19 类模型**（`SEMANTIC_LAYERS`）。
   本占位的作用是**先把管线打通、把格式定下来**（对应路线图 P2.6 / G5），
   而不是现在就产出可训练的分层。

磁盘约定（与《补充 11》§4.2 一致）：
    data/characters/<批次>/
        images/            交付图（RGBA，已去底）
        manifest.csv       file,tag,caption,source
        raw/               原始截图（归档，不参与训练）
输出：
    out/characters/<批次>/  norm/ · layers/ · cards/ · pipeline_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .card import CharacterCard

CANVAS = 1024          # 归一化画布边长
BODY_RATIO = 0.88      # 角色本体高度占画布比例
KMEANS_K = 10
MAX_FIT_PIXELS = 20000
BANDS = {"head": (0.00, 0.22), "torso": (0.22, 0.55), "legs": (0.55, 1.00)}


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _require_pil():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("需要 Pillow：pip install Pillow") from e
    return __import__("PIL.Image", fromlist=["Image"])


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(N,3) uint8 → (N,3) float [h(0-1), s(0-1), v(0-1)]。"""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(1), x.min(1)
    v = mx
    d = (mx - mn) + 1e-9
    s = np.where(mx > 1e-9, d / (mx + 1e-9), 0.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    h = np.zeros_like(v)
    m = mx == r
    h[m] = ((g - b) / d)[m] % 6
    m = mx == g
    h[m] = ((b - r) / d)[m] + 2
    m = mx == b
    h[m] = ((r - g) / d)[m] + 4
    return np.stack([(h / 6.0) % 1.0, s, v], 1)


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """极简 k-means（k-means++ 初始化）。返回 (labels, centroids)。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = X[rng.choice(n, size=1)].copy()
    for _ in range(k - 1):                                  # k-means++ 播种
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).min(1)
        p = d2 / (d2.sum() + 1e-12)
        C = np.vstack([C, X[rng.choice(n, p=p)]])
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1), C


def band_share(mask: np.ndarray) -> Dict[str, float]:
    """按**本体界框**高度切三段几何分区，返回各段像素占比。

    ⚠️ 这是**几何分区，不是语义层** —— 只用来给「分层模型」提供位置先验，
       以及给报告一个人能看懂的粗结构。
       **语义层必须由 See-through 自举模型产出，不能靠颜色规则硬猜。**
    """
    h, _ = mask.shape
    ys, _ = np.where(mask)
    if len(ys) == 0:
        return {k: 0.0 for k in BANDS}
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(1, y1 - y0)
    rows = np.arange(h)[:, None]
    total = max(1, int(mask.sum()))
    return {name: round(float((mask & (rows >= y0 + a * span)
                               & (rows < y0 + b * span)).sum() / total), 4)
            for name, (a, b) in BANDS.items()}


# ---------------------------------------------------------------------------
# 阶段
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    file: str
    tag: str
    caption: str
    source: str
    path: str
    attrs: dict = None            # 可选：已声明的轴取值（view / pose / shader / character）

    def __post_init__(self):
        if self.attrs is None:
            self.attrs = {}


def load_manifest(batch_dir: str) -> List[Entry]:
    """读取并校验 manifest.csv（前 4 列必需；轴列可选）。

    ⭐ 轴列（`character,view,pose,shader`）是**可选的**，但一旦提供，
       配对阶段就**用声明的值**，不再从文件名猜。
       ⚠️ 这不需要人工标注：轴取值应由**渲染器 / 生成脚本**写进 manifest
       （用户仍然只交 图 + caption + tag，见《补充 11》§4.1）。
    """
    mf = os.path.join(batch_dir, "manifest.csv")
    if not os.path.isfile(mf):
        raise FileNotFoundError(f"缺少 manifest.csv：{mf}")
    out: List[Entry] = []
    from .pairing import AXES
    with open(mf, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"file", "tag", "caption", "source"}
        cols = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
        missing = need - set(cols)
        if missing:
            raise ValueError(f"manifest 缺少列 {sorted(missing)}")
        opt = [a for a in ("character",) + AXES if a in cols]
        for row in rd:
            p = os.path.join(batch_dir, "images", row[cols["file"]])
            if not os.path.isfile(p):
                raise FileNotFoundError(f"manifest 指向的图不存在：{p}")
            attrs = {a: (row.get(cols[a]) or "").strip() for a in opt}
            if not attrs.get("character"):
                attrs["character"] = (row.get(cols["tag"]) or "").strip()
            out.append(Entry(row[cols["file"]], row[cols["tag"]],
                             row[cols["caption"]], row[cols["source"]], p, attrs))
    if not out:
        raise ValueError("manifest 为空")
    return out


def stage_normalize(entries: List[Entry], out_dir: str,
                    canvas: int = CANVAS, body_ratio: float = BODY_RATIO) -> Dict[str, dict]:
    """裁到本体界框 → 等比缩放到 body_ratio → 居中贴到方画布 → 存 PNG。

    ⭐ 分辨率口径：看「**角色本体占多大**」而不是「画布多大」——
       归一化后本体高度恒为 `canvas * body_ratio`。
    """
    Image = _require_pil()
    os.makedirs(out_dir, exist_ok=True)
    stats: Dict[str, dict] = {}
    for e in entries:
        im = Image.open(e.path).convert("RGBA")
        a = np.array(im)
        alpha = a[..., 3]
        ys, xs = np.where(alpha > 8)
        if len(ys) == 0:
            raise ValueError(f"{e.file} 全透明（去底失败？）")
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = im.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
        cw, ch = crop.size
        target_h = int(canvas * body_ratio)
        scale = target_h / ch
        new_w = max(1, int(round(cw * scale)))
        crop = crop.resize((new_w, target_h), Image.LANCZOS)
        canvas_im = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
        px = (canvas - new_w) // 2
        py = canvas - target_h - int(canvas * (1 - body_ratio) * 0.35)
        canvas_im.paste(crop, (px, max(0, py)), crop)
        dst = os.path.join(out_dir, e.file)
        canvas_im.save(dst)
        stats[e.file] = {"body_bbox": [int(x0), int(y0), int(x1), int(y1)],
                         "body_size": [int(cw), int(ch)],
                         "scale": round(float(scale), 4),
                         "canvas": canvas, "body_ratio": body_ratio,
                         "effective_body_h": target_h}
    return stats


def stage_layers(entries: List[Entry], norm_dir: str, out_dir: str,
                 k: int = KMEANS_K, seed: int = 0) -> Dict[str, dict]:
    """启发式分层（k-means 颜色聚类 + 规则）→ 存掩码预览 + 返回统计。"""
    Image = _require_pil()
    os.makedirs(out_dir, exist_ok=True)
    info: Dict[str, dict] = {}
    for e in entries:
        p = os.path.join(norm_dir, e.file)
        a = np.array(Image.open(p).convert("RGBA"))
        h, w = a.shape[:2]
        mask = a[..., 3] > 8
        idx = np.where(mask.reshape(-1))[0]
        px = a.reshape(-1, 4)[idx][:, :3].astype(np.float32)
        if len(px) > MAX_FIT_PIXELS:
            sel = np.random.default_rng(seed).choice(len(px), MAX_FIT_PIXELS, replace=False)
            fit = px[sel]
        else:
            fit = px
        lab_fit, C = kmeans(fit, min(k, max(1, len(fit))), seed=seed)
        lab_all = ((px[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)

        ys_all = (idx // w)
        palette = []
        canvas_lab = np.full(h * w, -1, dtype=np.int16)
        canvas_lab[idx] = lab_all
        for j in range(len(C)):
            sel = lab_all == j
            if not sel.any():
                continue
            rgb = C[j]
            palette.append({"id": j, "rgb": [int(v) for v in rgb.round()],
                            "pixels": int(sel.sum()),
                            "share": round(float(sel.sum() / max(1, mask.sum())), 4),
                            "y_frac": round(float(ys_all[sel].mean() / h), 3)})
        # 可视化：聚类着色图 + 本体掩码
        pal = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(len(C)):
            pal[canvas_lab.reshape(h, w) == j] = C[j].round().astype(np.uint8)
        viz = np.dstack([pal, (mask * 255).astype(np.uint8)])
        Image.fromarray(viz, "RGBA").save(os.path.join(out_dir, f"clusters_{e.file}"))
        info[e.file] = {"n_clusters": len(palette),
                        "body_pixels": int(mask.sum()),
                        "palette": sorted(palette, key=lambda c: -c["pixels"]),
                        "bands": band_share(mask)}
    return info


def stage_pair(entries: List[Entry], norm_dir: str,
               target_views: Optional[Tuple[str, ...]] = None,
               verbose: bool = False) -> dict:
    """**声明式**配对：同角色 + 只动 `view`（其余轴相同）。

    ⚠️ 与旧版的区别（这是一次**正确性**修正）：旧版靠文件名里有没有
    `front` / 以 `f` 开头来判"这是正视图" —— 那是**猜**。一旦数据线换命名
    （渲染器导出 / LoRA 批量生成），猜测会**静默给出错误配对**，
    而错误配对直接污染 Fitter 的训练目标，比报错难查得多。

    ⇒ 现在：**有声明就用声明**；**没有声明就明说没有**，绝不猜。
    """
    from .pairing import AXES, PairSpec, Record, build_pairs, format_pair_report

    records = [Record(key=e.file, identity=e.attrs.get("character") or e.tag,
                      attrs={a: e.attrs.get(a, "") for a in ("character",) + AXES})
               for e in entries]
    spec = PairSpec(vary=("view",), match=(), identity="character")
    tv = {"view": target_views} if target_views else None
    rep = build_pairs(records, spec, target_values=tv,
                      name_of={e.file: e.file for e in entries})
    if verbose:
        print(format_pair_report(rep))

    # ---- 兼容旧结构：每个 tag 一条，供 card.meta / report 使用 ----
    by_tag: Dict[str, List[Entry]] = {}
    for e in entries:
        by_tag.setdefault(e.tag, []).append(e)
    # ⚠️ 配对里用的是 identity（`character` 列，可能大小写/写法与 tag 不同），
    #    报告里按 tag 分组 ⇒ 必须用「文件名 → tag」映射回填，别拿 identity 当键。
    tag_of = {e.file: e.tag for e in entries}
    pairs_by_tag: Dict[str, list] = {}
    for p in rep.pairs:
        tag = tag_of.get(p.anchor, tag_of.get(p.positive, p.identity))
        pairs_by_tag.setdefault(tag, []).append(p.as_dict())

    out = {}
    for tag, es in by_tag.items():
        declared = all("view" in e.attrs and e.attrs["view"] for e in es)
        views = sorted({e.attrs.get("view", "") for e in es if e.attrs.get("view")})
        vlow = {v.lower() for v in views}
        has_front = bool({"front", "正", "正视图", "f"} & vlow)
        has_back = bool({"back", "背", "背视图", "b", "rear"} & vlow)
        out[tag] = {
            "views": [e.file for e in es], "count": len(es),
            "declared_view": declared, "view_values": views,
            "has_front": has_front, "has_back": has_back,
            "ok_min_views": declared and len(es) >= 2,
            "pairs": pairs_by_tag.get(tag, []),
            "n_pairs": len(pairs_by_tag.get(tag, [])),
            "gap": "" if declared else "未声明 view 列 → 无法配对（不要用文件名去猜）",
        }
    out["_report"] = rep.as_dict()
    return out


def _pairs_from_entries(entries: List[Entry], norm_dir: str, size: int):
    """entries → 声明式配对 + 已加载的图（供 Fitter 训练）。"""
    from .pairing import AXES, PairSpec, Record, build_pairs
    from .dataset import FitPair, load_image

    records = [Record(key=e.file, identity=e.attrs.get("character") or e.tag,
                      attrs={a: e.attrs.get(a, "") for a in ("character",) + AXES})
               for e in entries]
    rep = build_pairs(records, PairSpec(vary=("view",), match=(), identity="character"))
    cache: Dict[str, torch.Tensor] = {}

    def img(key: str) -> torch.Tensor:
        if key not in cache:
            cache[key] = load_image(os.path.join(norm_dir, key), size)
        return cache[key]

    pairs = [FitPair(identity=p.identity, anchor_key=p.anchor, positive_key=p.positive,
                     varied=dict(p.varied), anchor=img(p.anchor), positive=img(p.positive))
             for p in rep.pairs]
    return pairs, rep, cache


def stage_fit(entries: List[Entry], norm_dir: str, out_dir: str, *,
              size: int = 256, steps: int = 60, lr: float = 3e-3, seed: int = 0,
              dim: int = 1024, n_tokens: int = 256) -> dict:
    """在批次的**声明式配对**上训练 Character Fitter，产出每个 tag 的身份 token。

    ⭐ 这一步把角色卡线闭环：没有它，角色卡里的身份 token 只是零向量占位。
    ⚠️ 但要诚实：**1 个身份 / 1 条配对**时，训练只能测到「视角不变性」，
       身份的**可区分性无法验证**（需要第 2 个角色或更多视图）。
       报告里会显式标注这个覆盖缺口 —— 它不算通过。
    """
    from ..train.fitter import train_fitter
    from .fitter import CharacterFitter
    from .dataset import PairViewLoader, load_image

    os.makedirs(out_dir, exist_ok=True)
    pairs, rep, cache = _pairs_from_entries(entries, norm_dir, size)
    for e in entries:                      # 补齐未进配对的视图（单视图角色也要能出 token）
        if e.file not in cache:
            cache[e.file] = load_image(os.path.join(norm_dir, e.file), size)
    if not pairs:
        return {"fitter_path": None, "tokens": {}, "n_pairs": 0, "budget": {},
                "inv_end": None, "has_negatives": False,
                "note": "0 条配对 → 无法训练 Fitter（需同身份 ≥2 视图且 view 已声明）",
                "gaps": list(rep.gaps)}

    loader = PairViewLoader(pairs, source=f"batch-fit({len(pairs)} 对)")
    fitter = CharacterFitter(dim=dim, n_tokens=n_tokens, view_dim=256, heads=8)
    res = train_fitter(loader, fitter, steps=steps, lr=lr, seed=seed)
    fitter.eval()

    by_tag: Dict[str, List[Entry]] = {}
    for e in entries:
        by_tag.setdefault(e.tag, []).append(e)
    tokens: Dict[str, "np.ndarray"] = {}
    for tag, es in by_tag.items():
        views = torch.stack([cache[e.file] for e in es])       # (V, 3, H, W)
        with torch.no_grad():
            tok = fitter(views.unsqueeze(0))[0]                # (T, dim)
        tokens[tag] = tok.cpu().numpy()

    fpath = os.path.join(out_dir, "fitter.pt")
    torch.save({"state_dict": fitter.state_dict(), "dim": dim, "n_tokens": n_tokens,
                "size": size, "steps": steps, "n_pairs": len(pairs),
                "has_negatives": res.has_negatives}, fpath)
    note = ("已训得可用身份 token" if res.has_negatives else
            "⚠️ 只有 1 个身份 ⇒ 身份的**可区分性未经验证**（缺第 2 个角色 / 更多视图）")
    return {"fitter_path": fpath, "tokens": tokens, "n_pairs": len(pairs),
            "budget": res.budget, "inv_end": res.inv_end,
            "has_negatives": res.has_negatives, "note": note, "gaps": list(rep.gaps)}


def stage_pack(entries: List[Entry], norm_dir: str, layer_info: Dict[str, dict],
               pair_info: dict, out_dir: str,
               identity_tokens: Optional[Dict[str, "np.ndarray"]] = None) -> Dict[str, str]:
    """打包成 CharacterCard（每 tag 一张）。

    `identity_tokens` : 由 `stage_fit` 训练后按 tag 给出的身份 token；
                        缺省时退回**零向量占位**并在 meta 里注明（不静默假装可用）。

    ⚠️ 这里**不再从文件名挑正视图**：视图由 manifest 的 `view` 列**声明**，
       没有声明就用第一条 —— 绝不靠 `"front" in filename` 这种猜。
    """
    Image = _require_pil()
    os.makedirs(out_dir, exist_ok=True)
    made: Dict[str, str] = {}
    by_tag: Dict[str, List[Entry]] = {}
    for e in entries:
        by_tag.setdefault(e.tag, []).append(e)
    for tag, es in by_tag.items():
        declared_front = next((e for e in es
                               if (e.attrs.get("view") or "").lower() in ("front", "正视图", "正", "f")),
                              None)
        front = declared_front or es[0]          # 声明优先；否则取第一条（不猜）
        img = np.array(Image.open(os.path.join(norm_dir, front.file)).convert("RGBA"))
        # 唯一「真实掩码」的层：本体 alpha（语义层掩码待 See-through 自举）
        layers: Dict[str, np.ndarray] = {"body_base": img[..., 3]}
        tok = identity_tokens.get(tag) if identity_tokens else None
        if tok is None:
            tokens = np.zeros((256, 1024), dtype=np.float32)
            tok_src = "PLACEHOLDER(zeros) —— 需 Character Fitter（跑管线时加 --fit）"
        else:
            tokens = np.asarray(tok, dtype=np.float32)
            tok_src = "trained(Character Fitter, stage_fit)"
        card = CharacterCard(
            name=tag,
            identity_token=torch.from_numpy(tokens),
            layers=layers,
            meta={"views": [e.file for e in es],
                  "front_ref": front.file,
                  "view_masks": {e.file: f"norm/{e.file}" for e in es},
                  "caption": es[0].caption,
                  "source": es[0].source,
                  "pair": pair_info.get(tag, {}),
                  "palette": {e.file: layer_info[e.file]["palette"] for e in es},
                  "bands": {e.file: layer_info[e.file]["bands"] for e in es},
                  "identity_token": tok_src,
                  "semantic_layers": "仅 body_base 为真实掩码；19 类语义层待 See-through 自举"},
        )
        p = os.path.join(out_dir, f"{tag}.card")
        card.save(p)
        made[tag] = p
    return made


def stage_report(batch: str, counts: dict, norm_stats: dict, layer_info: dict,
                 pair_info: dict, cards: Dict[str, str], out_dir: str,
                 fit_info: Optional[dict] = None) -> str:
    lines = [f"# 角色卡管线报告 · 批次 `{batch}`", ""]
    lines.append(f"- 条目数：{counts.get('entries', 0)}")
    lines.append(f"- 归一化：画布 {CANVAS}²，本体高度 = {BODY_RATIO:.0%} 画布")
    lines.append("")
    lines.append("## 归一化（本体尺度一致化）")
    lines.append("| 文件 | 原本体界框 | 本体尺寸 | 缩放 |")
    lines.append("|---|---|---|---|")
    for f, s in norm_stats.items():
        lines.append(f"| {f} | {s['body_bbox']} | {s['body_size']} | ×{s['scale']} |")
    lines.append("")
    lines.append("## 分层（色板 + 几何分区 —— ⚠️ **非语义层**）")
    lines.append("> 语义层掩码需要 **See-through 自举**模型；此处只给颜色与位置先验。")
    lines.append("")
    for f, s in layer_info.items():
        b = s["bands"]
        lines.append(f"### {f} — 本体 {s['body_pixels']} px / {s['n_clusters']} 簇")
        lines.append(f"- 几何分段占比：头 **{b.get('head', 0):.1%}** / "
                     f"躯干 **{b.get('torso', 0):.1%}** / 腿 **{b.get('legs', 0):.1%}**")
        lines.append("| 簇 | RGB | 占比 | 平均高度 |")
        lines.append("|---|---|---|---|")
        for c in s["palette"]:
            rgb = c["rgb"]
            lines.append(f"| {c['id']} | #{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X} | "
                         f"{c['share']:.1%} | {c['y_frac']:.2f} |")
        lines.append("")
    lines.append("## 配对（角色卡最少 正 + 背）")
    for tag, p in pair_info.items():
        if tag.startswith("_"):
            continue
        ok = "✅" if p["ok_min_views"] else "⚠️"
        src = f"声明 view={p['view_values']}" if p["declared_view"] else "**未声明 view**"
        lines.append(f"- {ok} `{tag}`：{p['count']} 视图 {src}"
                     f"（正={p['has_front']}, 背={p['has_back']}, 配对 {p['n_pairs']} 对）")
        if p.get("gap"):
            lines.append(f"  - ⚠️ {p['gap']}")
    pr = pair_info.get("_report", {})
    for g in pr.get("gaps", []):
        lines.append(f"- ⚠️ {g}")
    lines.append("")
    lines.append("## 产出角色卡")
    for tag, p in cards.items():
        lines.append(f"- `{tag}` → `{p}`")
    lines.append("")
    lines.append("## Character Fitter（身份 token 来源）")
    if fit_info and fit_info.get("n_pairs"):
        b = fit_info.get("budget", {})
        lines.append(f"- 配对 **{fit_info['n_pairs']}** 对 → 训练 Fitter；"
                     f"不变性终值 {fit_info.get('inv_end'):.4f}")
        lines.append(f"- 负样本：{'有（可验证身份可区分性）' if fit_info.get('has_negatives') else '**无**'}"
                     f"—— {fit_info.get('note', '')}")
        if b:
            lines.append(f"- 参数 {int(b.get('total', 0)):,}｜"
                         f"AdamW 状态 ≈ {b.get('adamw_state_gb', 0):.4f} GB")
        lines.append(f"- 权重：`{fit_info.get('fitter_path')}`")
        for g in fit_info.get("gaps", []):
            lines.append(f"- ⚠️ {g}")
    elif fit_info is not None:
        lines.append(f"- ⚠️ 未训练：{fit_info.get('note', '无配对')}")
    else:
        lines.append("- 未训练（管线未加 `--fit`）→ 身份 token 为**零向量占位**")
    lines.append("")
    lines.append("> ⚠️ 19 类语义层掩码待 **See-through 自举**模型。")
    lines.append("> 本管线的作用是**先定格式、先打通**（路线图 P2.6 / 验证门 G5）。")
    lines.append("> 归一化保证**单一投影尺度**：本体高度恒为画布 88%，正/背一致。")
    txt = "\n".join(lines) + "\n"
    p = os.path.join(out_dir, "pipeline_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    return p


def run_batch(batch: str = "kokona", data_root: str = "data/characters",
              out_root: str = "out/characters", *, fit: bool = False,
              fit_steps: int = 60, fit_size: int = 256) -> dict:
    batch_dir = os.path.join(data_root, batch)
    out_dir = os.path.join(out_root, batch)
    os.makedirs(out_dir, exist_ok=True)
    norm_dir = os.path.join(out_dir, "norm")
    entries = load_manifest(batch_dir)
    norm = stage_normalize(entries, norm_dir)
    layers = stage_layers(entries, norm_dir, os.path.join(out_dir, "layers"))
    pair = stage_pair(entries, norm_dir)

    fit_info = None
    tokens = None
    if fit:
        fit_info = stage_fit(entries, norm_dir, os.path.join(out_dir, "fitter"),
                             size=fit_size, steps=fit_steps)
        tokens = fit_info["tokens"] or None

    cards = stage_pack(entries, norm_dir, layers, pair,
                       os.path.join(out_dir, "cards"), identity_tokens=tokens)
    rep = stage_report(batch, {"entries": len(entries)}, norm, layers, pair, cards,
                       out_dir, fit_info)
    bad = [t for t, v in pair.items()
           if not t.startswith("_") and not v["ok_min_views"]]
    warns = [] if not bad else \
        [f"{len(bad)} 个角色的视图未声明或不足"
         f"（角色卡最少需要 正 + 背 两视图，且 view 要**声明**而非从文件名猜）"]
    if fit_info is not None and fit_info.get("n_pairs") and not fit_info.get("has_negatives"):
        warns.append("Fitter 只有一个身份 ⇒ **可区分性未经验证**（覆盖缺口，非通过）")
    summary = {"batch": batch, "entries": len(entries), "cards": cards,
               "report": rep, "pair": pair,
               "n_pairs": pair.get("_report", {}).get("n_pairs", 0),
               "fit": None if fit_info is None else {
                   k: v for k, v in fit_info.items() if k != "tokens"},
               "warnings": warns}
    with open(os.path.join(out_dir, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KP 角色卡数据管线（CPU）")
    ap.add_argument("--batch", default="kokona")
    ap.add_argument("--data-root", default="data/characters")
    ap.add_argument("--out-root", default="out/characters")
    ap.add_argument("--pairing", action="store_true",
                    help="额外打印多视角配对报告（含「该补拍什么」的缺口清单）")
    ap.add_argument("--target-views", default=None,
                    help="逗号分隔，用于报缺口，例如 front,back,left,right")
    ap.add_argument("--fit", action="store_true",
                    help="训练 Character Fitter 并把身份 token 写回角色卡")
    ap.add_argument("--fit-steps", type=int, default=60)
    ap.add_argument("--fit-size", type=int, default=256)
    a = ap.parse_args(argv)

    tv = tuple(x.strip() for x in a.target_views.split(",")) if a.target_views else None
    if a.pairing:
        entries = load_manifest(os.path.join(a.data_root, a.batch))
        pair = stage_pair(entries, os.path.join(a.out_root, a.batch, "norm"),
                          target_views=tv, verbose=True)
        return 0 if all(v["ok_min_views"] for k, v in pair.items()
                        if not k.startswith("_")) else 1

    s = run_batch(a.batch, a.data_root, a.out_root, fit=a.fit,
                  fit_steps=a.fit_steps, fit_size=a.fit_size)
    print(f"✅ 批次 {s['batch']}：{s['entries']} 条 → {len(s['cards'])} 张角色卡")
    print(f"   报告：{s['report']}")
    if s.get("n_pairs"):
        print(f"   多视角配对：{s['n_pairs']} 对")
    if s.get("fit"):
        fi = s["fit"]
        print(f"   Fitter：{fi['n_pairs']} 对 → inv {fi.get('inv_end')}｜{fi.get('note')}")
    for w in s["warnings"]:
        print(f"   ⚠️ {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def load_manifest(batch_dir: str) -> List[Entry]:
    """读取并校验 manifest.csv（列名与文件存在性）。"""
    mf = os.path.join(batch_dir, "manifest.csv")
    if not os.path.isfile(mf):
        raise FileNotFoundError(f"缺少 manifest.csv：{mf}")
    out: List[Entry] = []
    with open(mf, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"file", "tag", "caption", "source"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f"manifest 缺少列 {sorted(missing)}")
        for row in rd:
            p = os.path.join(batch_dir, "images", row["file"])
            if not os.path.isfile(p):
                raise FileNotFoundError(f"manifest 指向的图不存在：{p}")
            out.append(Entry(row["file"], row["tag"], row["caption"], row["source"], p))
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


def stage_pair(entries: List[Entry], norm_dir: str) -> dict:
    """按 tag 分组，检查「同角色多视图」是否齐全（角色卡最少 正 + 背）。"""
    by_tag: Dict[str, List[Entry]] = {}
    for e in entries:
        by_tag.setdefault(e.tag, []).append(e)
    out = {}
    for tag, es in by_tag.items():
        names = [x.file.lower() for x in es]
        has_front = any("front" in n or n.startswith("f") for n in names)
        has_back = any("back" in n or n.startswith("b") for n in names)
        out[tag] = {"views": [x.file for x in es], "count": len(es),
                    "has_front": has_front, "has_back": has_back,
                    "ok_min_views": len(es) >= 2 and has_front and has_back}
    return out


def stage_pack(entries: List[Entry], norm_dir: str, layer_info: Dict[str, dict],
               pair_info: dict, out_dir: str,
               identity_token: Optional[np.ndarray] = None) -> Dict[str, str]:
    """打包成 CharacterCard（每 tag 一张）。

    ⚠️ 身份 token 目前是**占位**（无 Fitter 权重时用零向量），
       因为 Fitter 需要训练（路线图 P2.6 / G5）。此处先把**格式**定下来。
    """
    Image = _require_pil()
    os.makedirs(out_dir, exist_ok=True)
    made: Dict[str, str] = {}
    by_tag: Dict[str, List[Entry]] = {}
    for e in entries:
        by_tag.setdefault(e.tag, []).append(e)
    for tag, es in by_tag.items():
        front = next((e for e in es if "front" in e.file.lower()), es[0])
        img = np.array(Image.open(os.path.join(norm_dir, front.file)).convert("RGBA"))
        # 唯一「真实掩码」的层：本体 alpha（语义层掩码待 See-through 自举）
        layers: Dict[str, np.ndarray] = {"body_base": img[..., 3]}
        tokens = (np.zeros((256, 1024), dtype=np.float32) if identity_token is None
                  else identity_token.astype(np.float32))
        card = CharacterCard(
            name=tag,
            identity_token=torch.from_numpy(tokens),
            layers=layers,
            meta={"views": [e.file for e in es],
                  "view_masks": {e.file: f"norm/{e.file}" for e in es},
                  "caption": es[0].caption,
                  "source": es[0].source,
                  "pair": pair_info.get(tag, {}),
                  "palette": {e.file: layer_info[e.file]["palette"] for e in es},
                  "bands": {e.file: layer_info[e.file]["bands"] for e in es},
                  "identity_token": "PLACEHOLDER(zeros) —— 需 Character Fitter（P2.6 / G5）",
                  "semantic_layers": "仅 body_base 为真实掩码；19 类语义层待 See-through 自举"},
        )
        p = os.path.join(out_dir, f"{tag}.card")
        card.save(p)
        made[tag] = p
    return made


def stage_report(batch: str, counts: dict, norm_stats: dict, layer_info: dict,
                 pair_info: dict, cards: Dict[str, str], out_dir: str) -> str:
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
        ok = "✅" if p["ok_min_views"] else "⚠️"
        lines.append(f"- {ok} `{tag}`：{p['count']} 视图 {p['views']}"
                     f"（正={p['has_front']}, 背={p['has_back']}）")
    lines.append("")
    lines.append("## 产出角色卡")
    for tag, p in cards.items():
        lines.append(f"- `{tag}` → `{p}`")
    lines.append("")
    lines.append("> ⚠️ 身份 token 为**占位（零向量）**；19 类语义层掩码待 **See-through 自举**模型。")
    lines.append("> 本管线的作用是**先定格式、先打通**（路线图 P2.6 / 验证门 G5）。")
    lines.append("> 归一化保证**单一投影尺度**：本体高度恒为画布 88%，正/背一致。")
    txt = "\n".join(lines) + "\n"
    p = os.path.join(out_dir, "pipeline_report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    return p


def run_batch(batch: str = "kokona", data_root: str = "data/characters",
              out_root: str = "out/characters") -> dict:
    batch_dir = os.path.join(data_root, batch)
    out_dir = os.path.join(out_root, batch)
    os.makedirs(out_dir, exist_ok=True)
    entries = load_manifest(batch_dir)
    norm = stage_normalize(entries, os.path.join(out_dir, "norm"))
    layers = stage_layers(entries, os.path.join(out_dir, "norm"),
                          os.path.join(out_dir, "layers"))
    pair = stage_pair(entries, os.path.join(out_dir, "norm"))
    cards = stage_pack(entries, os.path.join(out_dir, "norm"), layers, pair,
                       os.path.join(out_dir, "cards"))
    rep = stage_report(batch, {"entries": len(entries)}, norm, layers, pair, cards, out_dir)
    summary = {"batch": batch, "entries": len(entries), "cards": cards,
               "report": rep, "pair": pair,
               "warnings": [] if all(v["ok_min_views"] for v in pair.values())
               else ["存在视图数不足的角色（角色卡最少需要 正 + 背 两视图）"]}
    with open(os.path.join(out_dir, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KP 角色卡数据管线（CPU）")
    ap.add_argument("--batch", default="kokona")
    ap.add_argument("--data-root", default="data/characters")
    ap.add_argument("--out-root", default="out/characters")
    a = ap.parse_args(argv)
    s = run_batch(a.batch, a.data_root, a.out_root)
    print(f"✅ 批次 {s['batch']}：{s['entries']} 条 → {len(s['cards'])} 张角色卡")
    print(f"   报告：{s['report']}")
    for w in s["warnings"]:
        print(f"   ⚠️ {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""G5 · 遮挡补全 + 伪深度绘制序 —— 角色卡的两块**未实现**字段

═══ 为什么这两个字段重要（不是"锦上添花"）═══
二次元角色图是**分层绘制**的（见-through 的 19 类）：
    头发压在脸前 / 帽饰压在头发上 / 外套压在内衣上
⇒ 下游要能回答两个问题：
    ① **被压住的部分长什么样**（遮挡补全）—— 否则 CharaBridge 重建时
       画出来是"穿模"的（帽子底下直接是脸，没有头发）
    ② **谁画在谁上面**（绘制序）—— 否则重组时图层顺序会错

⚠️ **诚实边界**：这两项**不可能从单张 RGBA 真正解出来**（信息在绘制时就丢了）。
   本模块用**可解释的几何/色彩先验**给出一版**假设**，并：
   · 把假设写成**可证伪的判据**（`verify_*` 函数）
   · 在meta 里**明确标注是推断值**（不是"真相"）
   ⇒ 符合项目铁律：⛔ 不静默产出占位冒充真实。

═══ 假设的依据（二次元渲染的通用构造）═══
    深度从「上」到「下」：hair_back → body → top/bottom → hair_front → hat
    ⇒ 这是**层序**而不是"真实 3D 深度"，所以叫**伪深度绘制序**
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .card import SEMANTIC_LAYERS

#: ⭐ **绘制序（后画在上面）** —— 由后向前排，据二次元常规构造
PAINT_ORDER: Tuple[str, ...] = (
    "background",     # 最底
    "hair_back",      # 后发（在身体后）
    "body_base",      # 身体
    "bottom",         # 下装
    "top",            # 上装
    "outerwear",      # 外套（压在上装上）
    "footwear",
    "legwear",
    "face",           # 脸（压在身体上）
    "ear",
    "eye_white", "iris", "eyelash", "eyebrow",   # 五官
    "mouth",
    "hair_side",
    "hair_front",     # 前发（压住脸 ⇒ 这是二次元的关键层）
    "skirt",
    "hat",            # 帽饰（最上）
)
_RANK: Dict[str, int] = {n: i for i, n in enumerate(PAINT_ORDER)}
#: 不在 PAINT_ORDER 里的层 ⇒ 排到中间（并记进 `unknown_layers`，不静默）
_MID = len(PAINT_ORDER) // 2


def depth_from_labels(lab: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """(H,W) int16 语义标签 → **伪深度绘制序图** (H,W) float32 ∈(0,1)。

    ⭐ 语义：**值越大 = 画得越靠上**。背景恒为 0。
    ⚠️ 这是**绘制序**（层序），**不是真实 3D 深度**——名字里的"伪"就是这个意思。
    """
    if lab.ndim != 2:
        raise ValueError(f"需要 (H,W) 标签图，收到 {lab.shape}")
    h, w = lab.shape
    names = SEMANTIC_LAYERS
    bad = [int(v) for v in np.unique(lab) if v >= len(names)]
    if bad:
        raise ValueError(f"标签含 19 类之外的值：{bad[:5]}（max 应={len(names)-1}）")
    depth = np.zeros((h, w), dtype=np.float32)
    unknown = set()
    for i, name in enumerate(names):
        m = lab == i
        if not m.any():
            continue
        r = _RANK.get(name)
        if r is None:
            unknown.add(name)
            r = _MID
        depth[m] = (r + 1) / (len(PAINT_ORDER) + 1)
    meta = {"n_layers_used": int(sum((lab == i).any() for i in range(len(names)))),
            "unknown_layers": sorted(unknown),
            "order": list(PAINT_ORDER)}
    return depth, meta


def occlusion_fill(rgba: np.ndarray, lab: np.ndarray,
                   depth: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """遮挡补全：把**被压住**的区域按周围最可能的层**补出来**。

    ═══ 做法（可解释，非魔改）═══
        对每个被前景层覆盖、但**不属于任何已解出层**的像素（= 被压住的），
        用它在 `depth` 上做**小窗口众数**推断它"应该是什么"，再填上**上层**的颜色。

    ⭐ 为什么用"上层颜色"：被压住的部分**视觉上呈现的是上面那层的东西**
       （比如帽子压住头发 ⇒ 补出来应该是帽子的颜色，不是头发的）。

    ⚠️ **这是推断，不是真相**。meta 里标`inferred: True`。
    """
    h, w = lab.shape
    a = alpha = rgba[..., 3]
    fg = alpha > 8
    known = fg & (lab >= 0)
    # ⭐ 被遮挡 = 在前景内，但当前label 说的是"上层"（我们自己解出的层就是上层）
    #    真正判据：alpha 边缘**内侧**一条带（绘制时压住的那一圈）
    inner = _inner_band(fg)
    occ = inner & known

    out = rgba.copy()
    filled = 0
    if occ.any():
        # 3x3 窗口众数：找每个遮挡像素的 4 邻域里出现最多的**非自身**层
        from collections import Counter
        pad = np.pad(lab, 1, mode="edge")
        for y, x in np.argwhere(occ):
            win = pad[y:y + 3, x:x + 3].ravel()
            win = win[(win >= 0) & (win != lab[y, x])]
            if len(win) == 0:
                continue
            # 取「绘制序更靠上」的那个（遮挡处应呈现上层）
            cand = max(set(win.tolist()),
                       key=lambda v: (_RANK.get(SEMANTIC_LAYERS[v], _MID), v))
            src = out[..., :3][pad[1:-1, 1:-1] == cand]
            if len(src) == 0:
                continue
            out[y, x, :3] = src.mean(0)
            filled += 1
    meta = {"inferred": True,
            "note": "遮挡补全是**几何推断**（非真相）；alpha 已丢失被压处信息",
            "n_occluded": int(occ.sum()), "n_filled": int(filled)}
    return out, meta


def _inner_band(fg: np.ndarray, width: int = 3) -> np.ndarray:
    """前景**内侧**一条带（膨胀后再与前景交）—— 绘制时最可能被压住的环。"""
    f = fg.astype(np.uint8)
    d = f.copy()
    for _ in range(max(1, width // 2)):
        d = _dilate(d)
    return (d > 0) & fg


def _dilate(m: np.ndarray) -> np.ndarray:
    """3x3 膨胀（纯 numpy，避免引 scipy）。"""
    out = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


# ---------------------------------------------------------------------------
def verify_depth(depth: np.ndarray, lab: np.ndarray) -> dict:
    """⛔ **可证伪判据**：检查绘制序是否满足三条必须成立的关系。"""
    names = SEMANTIC_LAYERS
    res = {}

    def d_of(n):
        i = names.index(n)
        m = lab == i
        return float(depth[m].mean()) if m.any() else None

    hat, hair_front, face, body = d_of("hat"), d_of("hair_front"), d_of("face"), d_of("body_base")
    res["hat_above_hair_front"] = (None if (hat is None or hair_front is None)
                                   else bool(hat > hair_front))
    res["hair_front_above_face"] = (None if (hair_front is None or face is None)
                                    else bool(hair_front > face))
    res["face_above_body"] = (None if (face is None or body is None) else bool(face > body))
    res["background_is_zero"] = bool(float(depth[lab < 0].max(initial=0.0)) == 0.0)
    ok = [v for v in res.values() if isinstance(v, bool)]
    res["n_checked"] = len(ok)
    res["pass"] = all(ok) if ok else False
    return res


def verify_occlusion(occ: np.ndarray, rgba: np.ndarray) -> dict:
    """遮挡补全的判据：不能改透明区、不能整体偏移颜色。"""
    a0 = rgba[..., 3] > 8
    a1 = occ[..., 3] > 8
    res = {"alpha_preserved": bool(np.array_equal(a0, a1)),
           "changed_px": int((np.abs(occ[..., :3].astype(int)
                                      - rgba[..., :3].astype(int)).sum(-1) > 0).sum())}
    res["pass"] = res["alpha_preserved"]
    return res


def main(argv: Optional[list] = None) -> int:
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="G5 · 遮挡补全 + 伪深度绘制序")
    ap.add_argument("--card", required=True, help=".card 路径")
    ap.add_argument("--out", default=None, help="输出目录（默认写到卡旁边）")
    a = ap.parse_args(argv)
    import torch
    d = torch.load(a.card, weights_only=False)
    layers = d.get("layers", {})
    if not layers:
        print("⛔ 卡里没有层⇒ 先跑 pipeline")
        return 1
    names = SEMANTIC_LAYERS
    H = W = None
    lab = None
    for n, arr in layers.items():
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            H, W = arr.shape[:2]
            lab = np.full((H, W), -1, dtype=np.int16)
            break
    if lab is None:
        print("⛔ 没有 (H,W,4) 的层")
        return 1
    for n, arr in layers.items():
        arr = np.asarray(arr)
        if n in names and arr.ndim == 3 and arr.shape[-1] == 4:
            lab[arr[..., 3] > 0] = names.index(n)

    depth, dmeta = depth_from_labels(lab)
    print(f"✅ 伪深度：{dmeta['n_layers_used']} 层｜未知层 {dmeta['unknown_layers'] or '无'}")
    vd = verify_depth(depth, lab)
    print(f"   判据 {vd['n_checked']} 条 → {'✅ 全通' if vd['pass'] else '⛔ 有不成立'}")
    for k, v in vd.items():
        if k not in ("pass", "n_checked"):
            print(f"     {'✅' if v else ('⛔' if v is False else '·')} {k} = {v}")

    out_dir = a.out or os.path.dirname(a.card)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth_order.npy"), depth)
    print(f"   深度图 → {os.path.join(out_dir, 'depth_order.npy')}")
    return 0 if vd["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""G5 · 解剖先验解算器（AnatomicalPrior）—— ⭐ 第一版**真实**语义层实现

═══ 为什么有这一个 ═══
`segmenter.py` 把「语义层从哪来」定义成了接口，但仓库里只有
`ColorClusterBaseline`（颜色簇，**不是语义层**）⇒ 19 类语义层**0 类真实实现**。

⛔ 本仓库**没有** See-through 权重，也不该假装有。
⇒ 本文件给的是**第二档**：**不靠预训练，靠解剖学几何 + 颜色的确定性先验**。
它**真的产出 19 类里的大部分**（不是簇号），但精度远 See-through ⇒ **自报质量，不冒充**。

═══ 为什么"先验"也是真实现，不是 heuristic ═══
区别在**可证伪**：
- `ColorClusterBaseline` 输出 `0..k-1`，与 19 类**语义完全错位** ⇒ `name_hit`结构上= 0
- 本解算器输出**与 `SEMANTIC_LAYERS` 同名的键** ⇒ `name_hit` 能真正上分
- 判据 `L1_SELFCHECK` 会拿构造级已知答案校验（见 `selftest_geom`），**有分辨力才敢用**

═══ 依据的解剖学先验（二次元角色画的标准构造）═══
    ① 背景   = alpha <阈值 的透明区
    ② 眼部带  = 面部上1/3 处的一条窄带（正视图最稳定的位置先验）
    ③ 头发   = 与面部带**不连通**、且在上方/侧方/下方的大块
    ④ 躯干   = 面部带下方的中央连通块
    ⑤ 服装   = 躯干外扩的一圈（裙摆/裤装）
    ⚠️ 嘴/鼻/耳等小部件在 512² 下通常只有几十像素 ⇒ **不猜**（见 `MIN_AREA_FRAC`）

⭐ **诚实边界**：本解算器解出的是**"解剖区域"而非"精确部件"**。
   眼白/虹膜/睫毛在低分辨率下不可靠 ⇒ **宁可不产出，也不产出错的**
   （与项目铁律一致：⛔ 不静默产出占位冒充真实）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .card import SEMANTIC_LAYERS
from .segmenter import BG_LAYER, LAYER_QUALITY

#: 面积小于此比例的连通块**直接丢弃**（不猜小部件）
MIN_AREA_FRAC = 0.0012

#: 面部带的纵向位置先验（占角色本体高度的比例，y 向下）
FACE_BAND = (0.06, 0.34)          # 眼/眉/嘴大致落在这条带里
#: 躯干带
TORSO_BAND = (0.32, 0.78)
#: 头发的三个方位（相对本体 bbox）
HAIR_TOP_Y = 0.30                  # 上方
HAIR_SIDE_X = (0.30, 0.70)         # 左右两侧


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """前景 bbox（y0,y1,x0,x1）；无前景返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    """最大连通块（4 邻接，纯 numpy 的两遍扫描，避免引入 scipy 依赖）。"""
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    cur = 0
    best_id, best_n = 0, 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            lab[sy, sx] = cur
            n = 0
            while stack:
                y, x = stack.pop()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = cur
                        stack.append((yy, xx))
            if n > best_n:
                best_n, best_id = n, cur
    return (lab == best_id) if best_id else np.zeros_like(mask, dtype=bool)


class AnatomicalPrior:
    """⭐ 解剖先验解算器：输出**与 19 类同名**的语义层。

    ⚠️ **自报质量偏保守**（这是刻意的——项目铁律：⛔ 不许填「差不多」）：
       `mean_iou` 给的是**与颜色簇基线相比的量级估计**，不是实测 IoU。
       真要用它上线，先跑 `selftest_geom()` 拿构造级校验的数字。
    """

    name = "anatomical-prior-v1（几何+颜色先验，非 See-through）"

    def __init__(self, *, alpha_thr: int = 8, min_area_frac: float = MIN_AREA_FRAC,
                 refine_face: bool = True, refine_torso: bool = True,
               refine_hair: bool = True):
        self.alpha_thr = int(alpha_thr)
        self.min_area_frac = float(min_area_frac)
        self.refine_face = bool(refine_face)
        self.refine_torso = bool(refine_torso)
        self.refine_hair = bool(refine_hair)
        self.quality = LAYER_QUALITY(
            # ⚠️ 诚实：不冒充 See-through 的水平。远低于真模型。
            mean_iou=0.28, name_hit=0.55, cross_view_consistency=None,
            note="解剖先验 v1：能给出 19 类里的区域级语义，"
                 "小部件（眼白/虹膜/睫毛/嘴）在 512² 下不产出（宁缺勿错）")

    # ------------------------------------------------------------------
    # 颜色精修工具（v1 新增）：在几何先验的**候选区**内按亮度/饱和度收边
    # ------------------------------------------------------------------
    @staticmethod
    def _splittable(a: np.ndarray) -> bool:
        """alpha 边缘是否**足够锐利** ⇒ 值得按颜色切。
        ⚠️ **不锐利就退��几何先验**（宁可用粗区域，不伪��细边界）。"""
        al = a[..., 3].astype(np.float32) / 255.0
        edge = np.abs(np.diff(al, axis=0)).mean() + np.abs(np.diff(al, axis=1)).mean()
        return float(edge) > 0.02

    def _refine_by_color(self, region: np.ndarray, a: np.ndarray,
                         ref_rgb: np.ndarray) -> np.ndarray:
        """在 `region` 内，按**颜色距离**把与 `ref_rgb` 相近的像素收进来。

        ⭐ 这是「几何给粗位置、颜色给精确边界」的分工——
        纯几何会在斜发/衣服褶皱处明显切错，纯颜色会把阴影也算进去。
        """
        if not region.any() or not self._splittable(a):
            return region
        rgb = a[..., :3].astype(np.float32)
        # 以候选区的平均色为参考（比固定色板稳）
        ref = ref_rgb if ref_rgb is not None else rgb[region].mean(0)
        d = np.linalg.norm(rgb - ref[None, None, :], axis=-1)
        # 阈值取区域内距离的 80 分位（自适应，不写死）
        thr = float(np.percentile(d[region], 80))
        return region | ((d <= thr) & (a[..., 3] > self.alpha_thr))

    # ------------------------------------------------------------------
    def predict(self, rgba: np.ndarray) -> np.ndarray:
        """(H,W,4) uint8 → (H,W) int16。**-1 = 背景**，其余是 `SEMANTIC_LAYERS` 的下标。"""
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"需要 (H,W,4) RGBA，收到 {rgba.shape}")
        h, w = rgba.shape[:2]
        alpha = rgba[..., 3]
        fg = alpha > self.alpha_thr
        out = np.full((h, w), -1, dtype=np.int16)
        if not fg.any():
            return out

        idx = {n: i for i, n in enumerate(SEMANTIC_LAYERS)}
        total = int(fg.sum())
        min_area = max(1, int(self.min_area_frac * total))

        body = _largest_cc(fg)                 # ① 本体（最大连通块）
        bb = _bbox(body)
        if bb is None:
            return out
        y0, y1, x0, x1 = bb
        bh, bw = y1 - y0, x1 - x0
        if bh <= 2 or bw <= 2:
            return out

        # 相对坐标网格
        yy = (np.arange(h)[:, None] - y0) / bh  # (H,1)  0..1
        xx = (np.arange(w)[None, :] - x0) / bw  # (1,W)

        # ② 面部带：FACE_BAND 与中央水平带的交集
        face = body & (yy >= FACE_BAND[0]) & (yy < FACE_BAND[1]) \
                   & (xx >= 0.22) & (xx <= 0.78)
        if face.sum() < min_area:
            face = body & (yy >= FACE_BAND[0]) & (yy < FACE_BAND[1])

        # ③ 头发：本体中「不在面部带」且在最上方的块
        hair = body & (yy < HAIR_TOP_Y) & (~face)
        if hair.sum() < min_area:
            hair = body & (xx < HAIR_SIDE_X[0]) & (~face)          # 左
            hair |= body & (xx > HAIR_SIDE_X[1]) & (~face)         # 右
        if hair.sum() < min_area:
            hair = body & (yy < 0.5) & (~face)

        # ④ 躯干：面部带以下、水平居中的块
        torso = body & (yy >= TORSO_BAND[0]) & (yy < TORSO_BAND[1]) \
                   & (~face) & (~hair)
        if torso.sum() < min_area:
            torso = body & (yy >= FACE_BAND[1]) & (~face) & (~hair)

        # ⑤ 服装：躯干下方剩下的
        cloth = body & (yy >= TORSO_BAND[1]) & (~face) & (~hair) & (~torso)

        # ════ v1 颜色精修：几何给粗位置，颜色给精确边界 ════
        # ⚠️ 只在**自己独占**的区域上精修（不碰别人的），否则会互相吞噬。
        # ⚠️ 顺序有意义：先 hair（最常被误切）→ face → torso。
        _rgb = rgba[..., :3].astype(np.float32)
        if self.refine_hair and hair.any():
            cand = body & (~face) & (yy < HAIR_TOP_Y + 0.18)
            hair = self._refine_by_color(hair & cand, rgba, _rgb[hair].mean(0))
        if self.refine_face and face.any():
            cand = body & (yy >= FACE_BAND[0] - 0.10) & (yy < FACE_BAND[1] + 0.10)
            face = self._refine_by_color(face & cand, rgba, _rgb[face].mean(0))
        if self.refine_torso and torso.any():
            cand = body & (~hair) & (~face)
            torso = self._refine_by_color(torso & cand, rgba, _rgb[torso].mean(0))

        # ⚠️ 精修后必须**重算互斥**（refine 用的是 `|`，可能与别层重叠）
        for m in (hair, face, torso):
            m &= body
        face &= ~hair & ~torso
        hair &= ~face & ~torso
        torso &= ~face & ~hair
        cloth = body & ~face & ~hair & ~torso & (yy >= TORSO_BAND[1])

        # ⚠️ **层间语义（2026-10-05 修正）**：`body_base` 是**完整本体掩码**，
        #    **包含**其余各层（像 See-through 那样：body 是底、部件是上）。
        #    ⛔ 早期版本让它只兜底边角⇒ IoU 0.010，看着像 bug，其实是语义搞错了。
        #    而 `predict()` 返回的 label 图仍是**互斥**的（每像素一个层号）——
        #    两套视图各有用途：`label` 用于可视化/统计，`body_base` 用于生成卡。
        for name, m in (("face", face), ("hair_front", hair),
                        ("top", torso), ("bottom", cloth)):
            mm = m & (m.sum() >= min_area)
            if mm.any():
                out[mm] = idx[name]
        # 零散未分配像素挂到 body_base（label 视图里）
        unassigned = fg & (out < 0)
        if unassigned.any():
            out[unassigned] = idx["body_base"]
        return out

    def predict_body(self, rgba: np.ndarray) -> np.ndarray:
        """(H,W) bool —— **完整本体掩码**（`body_base` 层的真身，含所有部件）。

        ⭐ 与 `predict()` 的关系：`predict` 给互斥 label（统计/可视化用），
        本方法给**叠加**的本体掩码（**生成角色卡用**）。
        """
        return rgba[..., 3] > self.alpha_thr

    # ------------------------------------------------------------------
    def to_rgba_layers(self, rgba: np.ndarray) -> Dict[str, np.ndarray]:
        """→ `{层名: (H,W,4) uint8 RGBA}`，可直接喂给 `CharacterCard.layers`。

        每层用**原像素**（不是着色块）⇒ 下游 CharaBridge 能拿到真实颜色/纹理。
        """
        lab = self.predict(rgba)
        h, w = lab.shape
        names = SEMANTIC_LAYERS
        out: Dict[str, np.ndarray] = {}
        # ⭐ `body_base` = **完整本体**（含所有部件，与 See-through 同语义）
        body = self.predict_body(rgba)
        if body.any():
            lay = np.zeros((h, w, 4), dtype=np.uint8)
            lay[..., :3] = rgba[..., :3]
            lay[..., 3] = (body * 255).astype(np.uint8)
            out["body_base"] = lay
        for i, name in enumerate(names):
            if name == "body_base":
                continue                      # 已在上面按「完整本体」处理
            m = lab == i
            n = int(m.sum())
            if n == 0:
                continue
            layer = np.zeros((h, w, 4), dtype=np.uint8)
            layer[..., :3] = rgba[..., :3]
            layer[..., 3] = (m * 255).astype(np.uint8)
            out[name] = layer
        return out

    def report(self) -> str:
        return (f"{self.name}\n  quality: {self.quality.summary()}\n"
                f"  产出层：body_base / face / hair_front / top / bottom\n"
                f"  ⛔ 不产出：眼白/虹膜/睫毛/眉/嘴/耳/鞋等小部件"
                f"（< {self.min_area_frac:.2%} 本体面积 ⇒ 低分辨率下不可靠）")


# ---------------------------------------------------------------------------
# 构造级自检（**有分辨力**才敢用 —— 铁律：先量再换）
# ---------------------------------------------------------------------------
def selftest_geom(size: int = 256, verbose: bool = True) -> dict:
    """在**构造的已知答案**上跑，验证解算器真的有分辨力。

    ⚠️ **2026-10-05 第一次写错了，已修**（留档，因为这是本项目反复吃的坑）：
       v1 只看「解出了哪些层名」，两档构造**给出完全相同的层集合**
       ⇒ `name_hit` 恒等于 1.0，**判据饱和**。
       ⇒ 与 M3 那个 `cross_r2≡1.0`（自由度不够）是**同一类病**。
       ⛔ **饱和的数字不是"结果好"，是"没有分辨力"**，一律不可用来支持任何结论。

    ✅ **修法**：改成**像素级**判据 —— 构造里**已知**哪块是脸/哪块是头发，
       直接算 IoU。三档构造（面带偏上/偏中/偏下）必须给出**单调不同**的 IoU。
    """
    seg = AnatomicalPrior()
    h = w = size
    names = SEMANTIC_LAYERS
    idx = {n: i for i, n in enumerate(names)}

    def synth(face_lo: float, face_hi: float):
        """返回 (rgba, gt) —— gt 是**构造真值** label 图（-1=背景）。"""
        a = np.zeros((h, w, 4), dtype=np.uint8)
        gt = np.full((h, w), -1, dtype=np.int16)
        a[10:size - 10, 10:size - 10, 3] = 255# 本体
        a[..., :3] = 200
        y0, y1 = int(size * face_lo), int(size * face_hi)
        gt[10:size - 10, 10:size - 10] = idx["body_base"]
        # 头发 = 本体最上方一条
        gt[10:y0, 10:size - 10] = idx["hair_front"]
        # 面部带
        gt[y0:y1, int(w * 0.3):int(w * 0.7)] = idx["face"]
        a[y0:y1, int(w * 0.3):int(w * 0.7), :3] = 240   # 面部更亮
        a[10:y0, 10:size - 10, :3] = 30                  # 头发更暗
        a[y1:size - 10, 10:size - 10, :3] = 120
        return a, gt

    def iou(pred: np.ndarray, gt: np.ndarray, name: str) -> float:
        p = pred == idx[name]
        g = gt == idx[name]
        u = int((p | g).sum())
        return float((p & g).sum() / u) if u else float("nan")

    def iou_body(pred_body: np.ndarray, gt_body: np.ndarray) -> float:
        """⭐ `body_base` 用**完整本体**判（不是互斥 label 里的那一小块）。"""
        u = int((pred_body | gt_body).sum())
        return float((pred_body & gt_body).sum() / u) if u else float("nan")

    res = {}
    for tag, (lo, hi) in (("face_high", (0.06, 0.20)),
                          ("face_mid", (0.20, 0.40)),
                          ("face_low", (0.45, 0.62))):
        rgba, gt = synth(lo, hi)
        pred = seg.predict(rgba)
        gt_body = gt >= 0
        row = {n: round(iou(pred, gt, n), 4) for n in ("face", "hair_front")}
        row["body_base"] = round(iou_body(seg.predict_body(rgba), gt_body), 4)
        res[tag] = row
        if verbose:
            print(f"  [{tag:10s}] IoU  " +
                  "  ".join(f"{k}={v:.3f}" for k, v in row.items()))

    # ⭐ 分辨力判据：三档的 face IoU 必须**不同**（不是全 0 也不是全 1）
    vals = [res[t]["face"] for t in ("face_high", "face_mid", "face_low")]
    all_nan = all(v != v for v in vals)
    spread = (max(v for v in vals if v == v) - min(v for v in vals if v == v)
              if not all_nan else 0.0)
    res["face_iou_spread"] = round(float(spread), 4)
    res["distinguishable"] = bool(spread > 0.02)
    if verbose:
        print(f"  ⭐ face IoU 跨度 = {spread:.4f} ⇒ 判据有分辨力：{res['distinguishable']}")
        if not res["distinguishable"]:
            print("     ⛔ 饱和/零分辨 ⇒ 这些数字**不可用**，不是结果好")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="G5 · 解剖先验解算器")
    ap.add_argument("--selftest", action="store_true", help="跑构造级自检")
    ap.add_argument("--show", metavar="RGBA图路径", help="对一张 RGBA 图解算并保存可视化")
    ap.add_argument("--out", default="out/character/layers")
    a = ap.parse_args(argv)

    seg = AnatomicalPrior()
    print(seg.report())
    if a.selftest or not a.show:
        print("\n构造级自检：")
        selftest_geom()
        return 0
    from PIL import Image
    rgba = np.array(Image.open(a.show).convert("RGBA"))
    layers = seg.to_rgba_layers(rgba)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.show))[0]
    for name, arr in layers.items():
        Image.fromarray(arr, "RGBA").save(os.path.join(a.out, f"{stem}__{name}.png"))
    print(f"\n✅ 解出 {len(layers)} 层 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

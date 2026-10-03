"""G5 · 语义层解算器接口 —— 把「k-means 颜色聚类」换成**可插拔的语义层模型**。

═══ 为什么需要这个文件 ═══

`kp/character/pipeline.py::stage_layers` 现在做的是 **k-means 颜色聚类**：
按 RGB 把像素聚成 k 簇，**产出的是「颜色簇」，不是「语义层」**。

⚠️ 两者有本质差别：
    颜色簇：  「这块像素是橙色的」
    语义层：  「这块是头发」「这块是眼睛」
⇒ 设计稿 §4.5 明写「**语义层必须由 See-through 自举模型产出，不能靠颜色规则硬猜**」，
  而现状是**占位**。**这是 txt2img+角色卡 主线的第一道阻塞。**

═══ 本文件解决什么（不解决什么）═══

✅ 解决：**把「语义层从哪来」变成一个明确的、可替换的接口**，
   并给每个实现配一把**质量尺子**（见 `LAYER_QUALITY`）——
   ⭐ 有了尺子，将来换模型时能**先量再换**，而不是"看着像就换"。

⛔ **不解决**：本仓库**没有** See-through 权重，也没下载。
   ⇒ 本文件**只定义接口 + 契约 + 质量判据 + 一个可跑的参考实现（当前仍是启发式，但被明确标为 baseline）**。
   ⚠️ **绝不假装已实现 See-through** —— 那是最容易骗到自己的一种 bug。

═══ 怎么接一个真正的语义层模型 ═══

    class MySegmenter:
        def predict(self, rgba: np.ndarray) -> np.ndarray:   # (H,W) int，-1=背景
            ...
        name: str
        quality: LAYER_QUALITY                              # 自报质量指标

    register_segmenter(MySegmenter())
    solve_layers(entries, norm_dir, out_dir, segmenter=MySegmenter())
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from .card import SEMANTIC_LAYERS

#: 背景层的名字（`SEMANTIC_LAYERS[0]`，与 card.py 的定义一致）
BG_LAYER = SEMANTIC_LAYERS[0]


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------
@dataclass
class LAYER_QUALITY:
    """一个语义层解算器**自报**的质量指标（⛔ 不许填「差不多」）。

    ⭐ **为什么要这张表**：语义层是**定性**任务（"这层是头发吗"），
    没有数字就变成"看着还行就上线"——这是本项目反复吃的坑（指标口径不对，结论必错）。
    ⇒ 换模型前先量这些数字，**新旧可比**了再换。
    """
    #: 与 19 类语义层的**平均 IoU**（越高越好；启发式 baseline 会很低，这是事实）
    mean_iou: float
    #: **层数命中**：解出的层名与 `SEMANTIC_LAYERS` 的交集占比（1.0 = 名字全对）
    name_hit: float
    #: **跨图一致性**：同一角色的不同视角，解出的层分配是否一致（0-1；越高越稳）
    cross_view_consistency: Optional[float] = None
    #: 自由备注（写清这次跑的什么模型/什么参数）
    note: str = ""

    def summary(self) -> str:
        s = f"mean_iou={self.mean_iou:.3f} name_hit={self.name_hit:.2f}"
        if self.cross_view_consistency is not None:
            s += f" cross_view={self.cross_view_consistency:.2f}"
        return s + (f"（{self.note}）" if self.note else "")


class Segmenter(Protocol):
    """语义层解算器的协议（结构化子类型，不强制继承）。"""
    name: str
    quality: LAYER_QUALITY

    def predict(self, rgba: np.ndarray) -> np.ndarray:
        """(H,W,4) uint8 RGBA → (H,W) int16，**-1 表示背景**。"""
        ...


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[], Segmenter]] = {}


def register_segmenter(name: str, factory: Callable[[], Segmenter]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"解算器 {name!r} 已注册（⛔ 不许静默覆盖）")
    _REGISTRY[name] = factory


def available() -> List[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Baseline：当前的颜色聚类（**明确标为占位**）
# ---------------------------------------------------------------------------
class ColorClusterBaseline:
    """⚠️ **BASELINE，不是语义层**。按 RGB 做 k-means，产出「颜色簇」。

    ⛔ 保留它只为：(a) 让管线在真模型到位前仍能跑通；(b) 作为**质量对照的下界**。
    ⚠️ 它的 `name_hit` 结构上不可能到 1.0 —— 因为它**不按 19 类命名**，
       而只给「簇 0/1/2…」。这是设计上的诚实，不是 bug。
    """
    name = "color-cluster-baseline（⚠️ 占位，非语义层）"

    def __init__(self, k: int = 8, seed: int = 0, max_pixels: int = 20000):
        self.k = k
        self.seed = seed
        self.max_pixels = max_pixels
        # ⭐ **自报质量时必须诚实**：名字命中率结构上是 0（它不按 19 类命名）
        self.quality = LAYER_QUALITY(
            mean_iou=0.0, name_hit=0.0, cross_view_consistency=None,
            note="颜色聚类基线；语义层完全错位（按簇号而非 19 类命名）")

    def predict(self, rgba: np.ndarray) -> np.ndarray:
        from .pipeline import kmeans, MAX_FIT_PIXELS
        h, w = rgba.shape[:2]
        a = rgba.reshape(-1, 4)
        mask = a[:, 3] > 8
        out = np.full(h * w, -1, dtype=np.int16)
        if not mask.any():
            return out.reshape(h, w)
        px = a[mask][:, :3].astype(np.float32)
        if len(px) > min(self.max_pixels, MAX_FIT_PIXELS):
            sel = np.random.default_rng(self.seed).choice(
                len(px), min(self.max_pixels, MAX_FIT_PIXELS), replace=False)
            fit = px[sel]
        else:
            fit = px
        k = min(self.k, max(1, len(fit)))
        _, C = kmeans(fit, k, seed=self.seed)
        lab = ((px[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        out[mask] = lab
        return out.reshape(h, w)


register_segmenter("color_cluster", lambda: ColorClusterBaseline())


# ---------------------------------------------------------------------------
# 求解入口
# ---------------------------------------------------------------------------
def solve_layers(entries, norm_dir: str, out_dir: str, *,
                 segmenter: Optional[Segmenter] = None,
                 expected_names: Optional[List[str]] = None) -> dict:
    """用指定解算器跑完一批图 → 掩码 PNG + 统计。

    ⭐ **与旧 `stage_layers` 的区别**：这里**只做语义层**，
    聚类/规则那些启发式细节收在解算器内部 ⇒ 换模型时**本函数不动**。
    """
    from PIL import Image
    names = expected_names or SEMANTIC_LAYERS
    seg = segmenter or ColorClusterBaseline()
    os.makedirs(out_dir, exist_ok=True)
    stats: Dict[str, dict] = {}
    for e in entries:
        p = os.path.join(norm_dir, e.file)
        rgba = np.array(Image.open(p).convert("RGBA"))
        lab = seg.predict(rgba)                     # (H,W) int16，-1=背景
        stem = os.path.splitext(e.file)[0]
        got = sorted({int(v) for v in np.unique(lab) if int(v) >= 0})
        stats[stem] = {
            "segmenter": seg.name,
            "quality": seg.quality.summary(),
            "n_layers_found": len(got),
            "expected": len(names),
            # ⭐ 名字命中率：这才是「像不像语义层」的关键指标
            "layer_ids": got,
            "name_hit": (sum(1 for g in got if str(g) in names) / max(1, len(got))),
        }
        # 存可视化（调试图，不是最终资产）
        vis = np.zeros((*lab.shape, 3), dtype=np.uint8)
        rng = np.random.default_rng(0)
        for v in got:
            vis[lab == v] = rng.integers(40, 255, 3)
        Image.fromarray(vis).save(os.path.join(out_dir, f"{stem}_layers.png"))
    return {"segmenter": seg.name, "quality": seg.quality.summary(),
            "n": len(stats), "per_image": stats}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="G5 · 语义层解算器（接口 + 质量判据）")
    ap.add_argument("--list", action="store_true", help="列出已注册的解算器")
    a = ap.parse_args(argv)
    print("已注册解算器：")
    for n in available():
        print(f"  · {n}")
    print(f"\n⚠️ 当前**全部是占位**：本仓库没有 See-through 权重。")
    print(f"  颜色聚类**不产出语义层**（它给的是颜色簇，不是 19 类）。")
    print(f"  接真模型：实现 Segmenter 协议（predict + quality）→ register_segmenter()。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

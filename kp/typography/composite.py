"""composite.py —— 把 ROI 分支的低压缩结果**贴回全图 latent**（排版链路第三步）。

链路全景（`ONBOARDING.md` §9 待办 4）::

    ① Layout Planner   plan(spec)            → 纯 JSON 版面（boxes: x/y/w/h, 像素）
    ② ROI 分支         roi_boxes(layout)     → 像素 ROI；ROIBranch → 低压缩 token
    ③ 拼回（本文件）    composite_latent(...) → 全图 latent：框内换成 ROI 内容，框外逐位不变

⭐ **为什么第 ③ 步以前没人写**：前两步的产物形状对不上。
   - Layout Planner 活在**像素**坐标系；
   - 主干 latent 活在 **32× 格点**坐标系（`kp/config.py::LATENT.spatial`，1024² → 32×32）；
   - ROIBranch 的输出是 **cross-attention token**（`(B, n_roi*T, dim)`），压根不是 dense latent。
   三者之间缺一张**坐标映射 + 写入规则**的契约 —— 那就是本文件。

⚠️ **本文件的核心取舍（写在最前面，免得被当成 trivial 的 paste）**：

1. **框是任意矩形**（避头尾 / 竖排 ⇒ 不成网格），量化到 32× 格点后会**互相重叠**。
   ⭐ `layout.validate()` 只在**像素级**判重叠；两个相距 10px 的框在像素级完全合法，
   但在 latent 级会挤进同一个格 —— **这是像素级判据看不到的冲突**，必须在这里二次裁决。
2. **同一 latent 格被多个框覆盖 ⇒ 默认「先到先得」（painter's algorithm）**。
   理由：latent 不是像素，**两个不同汉字的 latent 做线性混合，得到的既不是甲也不是乙，
   是一团糊** —— 这不是「半透明」，是**信息论上的销毁**。所以宁可让后来的框让位，
   也要保证**每个被写入的格只属于一个字**。（`policy="last"` / `"blend"` 保留，默认 `"first"`。）
3. **框外保持原样**：本函数是**掩码覆盖**（不是加法、不是平均），
   ⇒ 未被任何框覆盖的格**逐位不变**（bit-exact），这是可证伪的第一判据。
4. **纯函数 / 可测**：入参不变则输出不变；判据 `verify_composite` **不复用**合成循环，
   而是用一条独立的累加路径**重新算出应有结果**再逐位比对 —— 否则是「自己判自己及格」。

⚠️ **诚实的边界**：在 32× 格点上「拼回」本质是**有损降采样**（4× 的字形 latent
被 area-average 到 1 格）。它保证的是**字落在正确的格子上**（位置正确、可被主干解出为
正确的笔画块），**不保证字形高频无损**。要 100% 字形正确，走 T0 像素域合成（见本文件末尾）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import LATENT
from .layout import LayoutSpec, TextBlock, plan, roi_boxes, validate
from .typography_pack import ROIBranch


# ---------------------------------------------------------------------------
# 异常（退化输入必须「响亮」，不能静默返回）
# ---------------------------------------------------------------------------
class TypographyChainError(Exception):
    """排版链路通用异常基类。"""


class DegenerateLayoutError(TypographyChainError):
    """⚠️ 版面退化（空 layout / 零个文字框 / 框完全落在画布外 / ROI 数与 patch 数不符）。

    为什么必须抛异常而不是返回一个「看起来正常」的 latent：
    静默退化会让人以为「文字已经写上去了」，而实际上一个格都没动 —— 这是**假阳性**
    最容易滋生的地方。宁可让流水线当场炸。
    """


class ROIConflictError(TypographyChainError):
    """`policy="error"` 下检测到 latent 级重叠框。"""


# ---------------------------------------------------------------------------
# 形状描述：像素矩形 → latent 格窗口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CellWindow:
    """一个文字框在 **latent 格点**上占据的窗口（半开区间 `[cx0, cx1) × [cy0, cy1)`）。"""

    index: int                 # 在 rois 列表里的序号（与 patches 一一对应）
    cx0: int
    cy0: int
    cx1: int
    cy1: int
    px0: float                 # 原像素矩形（画布内）
    py0: float
    px1: float
    py1: float
    coverage: float = 1.0      # 窗口内各格的平均覆盖率 ∈ (0, 1]
    edge_cells: int = 0        # 只被「擦到边」的格数（align="cover" 时为覆盖率 < 1 的格）

    @property
    def height(self) -> int:
        return self.cy1 - self.cy0

    @property
    def width(self) -> int:
        return self.cx1 - self.cx0

    @property
    def cells(self) -> int:
        return self.height * self.width

    def as_dict(self) -> Dict:
        return {"index": self.index, "cells": f"({self.cy0},{self.cy0 + self.height})"
                                             f"×({self.cx0},{self.cx0 + self.width})",
                "n_cells": self.cells, "px": [self.px0, self.py0, self.px1, self.py1],
                "coverage": round(self.coverage, 4), "edge_cells": self.edge_cells}


def _ceil_int(v: float) -> int:
    return int(math.ceil(v - 1e-9))


def pixel_rect_to_cells(px0: float, py0: float, px1: float, py1: float,
                        scale: int, shape: Tuple[int, int],
                        align: str = "center") -> Optional[Tuple[int, int, int, int, float, int]]:
    """像素矩形 → latent 格窗口。返回 `(cx0, cy0, cx1, cy1, coverage, edge_cells)`；
    矩形与画布无交集时返回 **None**（交由上层报缺口）。

    ⭐ `align` 决定「擦到边的格子算不算数」，这直接决定**背景会不会被吃掉**：

    - `"center"`（默认）**格心判定**：格中心落在矩形内才算。
      ⇒ 一个只被文字压住 3% 的边角格**保持背景原样**，不会被一整块字形 latent 抹掉。
      代价：字会「缩」进去半格（≈16px @1024），这是 32× 量化的固有粒度。
    - `"cover"` **触达判定**：只要有像素重叠就整格写入。
      ⇒ 字更贴近原位，但会把边角格（大部分是背景）整格换成字形 latent。
    """
    H, W = shape
    if align == "center":
        cx0, cx1 = _ceil_int(px0 / scale - 0.5), _ceil_int(px1 / scale - 0.5)
        cy0, cy1 = _ceil_int(py0 / scale - 0.5), _ceil_int(py1 / scale - 0.5)
    elif align == "cover":
        cx0, cx1 = int(math.floor(px0 / scale)), _ceil_int(px1 / scale)
        cy0, cy1 = int(math.floor(py0 / scale)), _ceil_int(py1 / scale)
    else:
        raise ValueError(f"align 只支持 'center' / 'cover'，收到 {align!r}")
    cx0, cx1 = max(0, min(W, cx0)), max(0, min(W, cx1))
    cy0, cy1 = max(0, min(H, cy0)), max(0, min(H, cy1))
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    # 覆盖率统计（逐格几何求交，纯 Python —— 窗口只有几十格，够快且零依赖）
    cov_sum, n, edge = 0.0, 0, 0
    for i in range(cy0, cy1):
        for j in range(cx0, cx1):
            ox = max(0.0, min(px1, (j + 1) * scale) - max(px0, j * scale))
            oy = max(0.0, min(py1, (i + 1) * scale) - max(py0, i * scale))
            cov = (ox * oy) / float(scale * scale)
            cov_sum += cov
            n += 1
            if cov < 1.0 - 1e-9:
                edge += 1
    return cx0, cy0, cx1, cy1, cov_sum / max(1, n), edge


def roi_windows(rois: Sequence[Mapping], *, scale: int = LATENT.spatial,
                shape: Optional[Tuple[int, int]] = None,
                align: str = "center") -> Tuple[List[CellWindow], List[str]]:
    """像素 ROI 列表（`layout.roi_boxes()` 的契约：`x0/y0/x1/y1`）→ 格窗口。

    返回 `(windows, gaps)`：`gaps` 是**显式报出的缺口**（退化/出界/字段非法），
    **绝不在这里悄悄丢框** —— 丢一个框 = 那一块文字凭空消失。
    """
    windows: List[CellWindow] = []
    gaps: List[str] = []
    # ⚠️ `shape` 是 **latent** 尺寸（格），而 x0/y0 是**像素** —— 出界判断必须先换算回像素画布，
    #    否则「64px 的框」会被拿去和「32 格的 latent」比，得出「框在画布外」的胡话。
    cw = shape[1] * scale if shape is not None else None
    ch = shape[0] * scale if shape is not None else None
    for i, r in enumerate(rois or []):
        try:
            x0, y0 = float(r["x0"]), float(r["y0"])
            x1, y1 = float(r["x1"]), float(r["y1"])
        except (KeyError, TypeError, ValueError):
            gaps.append(f"ROI #{i} 缺少/非法 x0,y0,x1,y1：{r!r}")
            continue
        if x1 <= x0 or y1 <= y0:
            why = ""
            if cw is not None and (x0 >= cw or y0 >= ch):
                why = "（框完全落在画布外：roi_boxes 已裁剪 ⇒ x0 ≥ x1）"
            gaps.append(f"ROI #{i} 退化，宽/高 ≤ 0：x0={x0} y0={y0} x1={x1} y1={y1}{why}")
            continue
        if shape is None:
            gaps.append(f"ROI #{i} 需要 latent 空间尺寸 shape 才能定格；请传 shape=(H_lat, W_lat)")
            continue
        got = pixel_rect_to_cells(x0, y0, x1, y1, scale, shape, align)
        if got is None:
            if x1 <= 0 or y1 <= 0 or x0 >= cw or y0 >= ch:
                gaps.append(f"ROI #{i} 与画布无交集（框完全落在图外）：px=({x0},{y0})-({x1},{y1})")
            else:
                gaps.append(
                    f"ROI #{i} 太小：align={align!r} 下没有任何 latent 格被覆盖"
                    f"（px 高/宽 = {y1 - y0:g}/{x1 - x0:g} < 一个格 {scale}px）"
                    f" ⇒ 这个框的文字在 latent 里**无处可放**，请调大字号或改 align='cover'")
            continue
        cx0, cy0, cx1, cy1, cov, edge = got
        windows.append(CellWindow(index=len(windows), cx0=cx0, cy0=cy0, cx1=cx1, cy1=cy1,
                                  px0=x0, py0=y0, px1=x1, py1=y1,
                                  coverage=cov, edge_cells=edge))
    return windows, gaps


def windows_from_layout(layout: Mapping, *, scale: int = LATENT.spatial,
                        pad: int = 0, shape: Optional[Tuple[int, int]] = None,
                        align: str = "center") -> Tuple[List[CellWindow], List[str]]:
    """直接吃 Layout Planner 的 JSON（走 `roi_boxes()` 契约）。"""
    if not isinstance(layout, Mapping) or "boxes" not in layout:
        raise DegenerateLayoutError(f"layout 必须含 'boxes' 键，收到 {type(layout).__name__}")
    if not layout.get("canvas"):
        raise DegenerateLayoutError(f"layout 缺 'canvas'：{sorted(layout.keys())}")
    rois = roi_boxes(dict(layout), pad)
    if not rois:
        raise DegenerateLayoutError(
            f"layout 里一个文字框都没有（canvas={layout.get('canvas')}）"
            " ⇒ 没有可写的 ROI，拼回是空操作。")
    return roi_windows(rois, scale=scale, shape=shape, align=align)


# ---------------------------------------------------------------------------
# 合成
# ---------------------------------------------------------------------------
@dataclass
class CompositeReport:
    """一次拼回的**账本**（只作人读；判据不依赖它，见 `verify_composite`）。"""

    policy: str = "first"
    align: str = "center"
    scale: int = LATENT.spatial
    windows: List[CellWindow] = field(default_factory=list)
    written_cells: int = 0          # 实际被写入的格数（各 batch 取同一布局 ⇒ 单份计数）
    contested_cells: int = 0        # 被多个框争夺、最终按 policy 裁决的格数
    resampled_patches: int = 0      # 发生过重采样（=有损）的 ROI 块数
    gaps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    total_cells: int = 0
    shape: Tuple[int, int] = (0, 0)
    channels: int = 0

    @property
    def roi_fraction(self) -> float:
        return (self.written_cells / self.total_cells) if self.total_cells else 0.0

    def as_dict(self) -> Dict:
        return {"policy": self.policy, "align": self.align, "scale": self.scale,
                "latent_shape": list(self.shape), "channels": self.channels,
                "n_windows": len(self.windows), "written_cells": self.written_cells,
                "total_cells": self.total_cells, "roi_fraction": round(self.roi_fraction, 6),
                "contested_cells": self.contested_cells,
                "resampled_patches": self.resampled_patches,
                "windows": [w.as_dict() for w in self.windows],
                "gaps": list(self.gaps), "notes": list(self.notes)}

    def lines(self) -> List[str]:
        L = [f"策略 policy={self.policy}（先到先得＝每个格只属于一个字）"
             if self.policy == "first" else f"策略 policy={self.policy}",
             f"对齐 align={self.align}",
             f"格点 {self.shape[0]}×{self.shape[1]}，每格 {self.scale}×{self.scale} 像素",
             f"窗口 {len(self.windows)} 个｜写入 {self.written_cells}/{self.total_cells} 格 "
             f"（占画面 {self.roi_fraction * 100:.2f}%）",
             f"重叠争夺格 {self.contested_cells}｜重采样（有损）ROI 块 {self.resampled_patches}"]
        L += [f"  缺口：{g}" for g in self.gaps]
        L += [f"  提示：{n}" for n in self.notes]
        return L


def fit_patch(patch: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, bool]:
    """把一块 ROI 结果整形成 `(C, height, width)`。返回 `(张量, 是否重采样)`。

    尺寸已相符时**原样返回（零拷贝视图、bit-exact）** —— 这是最常见的路径；
    否则做重采样：缩小用 `area`（= 区域平均），放大用 `bilinear`。
    ⚠️ 重采样**有损**：4× 的字形 latent 贴回 32× 主干时必然丢掉高频笔画细节，
    换来的是「字落在正确的格子上」。报告里的 `resampled_patches` 就是这笔账。
    """
    if patch.dim() == 4:
        if patch.shape[0] != 1:
            raise ValueError(f"ROI 块应为 (C,h,w) 或 (1,C,h,w)，收到 {tuple(patch.shape)}")
        patch = patch[0]
    if patch.dim() != 3:
        raise ValueError(f"ROI 块应为 (C,h,w)，收到 {tuple(patch.shape)}")
    if int(patch.shape[-2]) == height and int(patch.shape[-1]) == width:
        return patch, False
    down = height <= patch.shape[-2] and width <= patch.shape[-1]
    kw = {} if down else {"align_corners": False}
    out = F.interpolate(patch.unsqueeze(0).float(), size=(height, width),
                        mode="area" if down else "bilinear", **kw)[0]
    return out.to(patch.dtype), True


def _batch_patch(patches: Sequence[torch.Tensor], index: int, like: torch.Tensor,
                 batch: int) -> torch.Tensor:
    """取第 `batch` 个样本对应的 ROI 块 → `(C,h,w)`；`(1,C,h,w)` 视为广播。"""
    p = patches[index]
    if p.dim() == 3:
        return p
    if p.dim() == 4 and int(p.shape[0]) in (batch, 1):
        return p[int(p.shape[0]) - 1 if int(p.shape[0]) == 1 else batch]
    raise ValueError(f"ROI 块 #{index} 应为 (C,h,w) 或 (1,C,h,w)，与 batch={batch} 不匹配："
                     f"{tuple(p.shape)}")


def composite_latent(z: torch.Tensor, patches: Sequence[torch.Tensor], *,
                     rois: Optional[Sequence[Mapping]] = None,
                     layout: Optional[Mapping] = None,
                     scale: int = LATENT.spatial, pad: int = 0,
                     align: str = "center", policy: str = "first",
                     strict: bool = True) -> Tuple[torch.Tensor, CompositeReport]:
    """⭐ 本模块的主函数：把 ROI 分支结果**掩码覆盖**到全图 latent 的对应格窗口上。

    参数
    ----
    z : `(B, C, H_lat, W_lat)` 全图 latent（主干出的底图）
    patches : 每框一块 `(C, h, w)`（或 `(1,C,h,w)`）；尺寸不必等于窗口尺寸，会被 `fit_patch` 整形
    rois : 像素 ROI `[{"x0","y0","x1","y1"}, ...]`（`layout.roi_boxes()` 的契约）
    layout : 或直接给 Layout Planner 的 JSON（内部走 `roi_boxes(layout, pad)`）
    scale : 像素 / latent 格（默认 `LATENT.spatial = 32`）
    align : `"center"`（默认，格心判定，边角格保背景）/ `"cover"`（触达判定）
    policy : `"first"`（默认，先到先得）/ `"last"` / `"blend"` / `"error"`
    strict : True 时遇到缺口**抛 `DegenerateLayoutError`**，绝不静默返回

    返回
    ----
    `(z_out, report)`。`z_out` 与 `z` **同 dtype / 同 device**，
    且未被任何窗口覆盖的格**逐位不变**。

    ⭐ 可逆性：这是**覆盖**不是累加 ⇒ 背景可从 `z` 精确恢复
    （`z_out[..., ~mask] == z[..., ~mask]` bit-exact，`z_out[..., mask] == 贴入的 ROI`）。
    """
    if (rois is None) == (layout is None):
        raise ValueError("rois 与 layout 必须且只能给一个")
    if z.dim() != 4:
        raise ValueError(f"latent 必须是 (B,C,H,W)，收到 {tuple(z.shape)}")
    shape = (int(z.shape[-2]), int(z.shape[-1]))

    if layout is not None:
        ws, gaps = windows_from_layout(layout, scale=scale, pad=pad, shape=shape, align=align)
    else:
        rois = list(rois or [])
        if not rois:
            gaps = ["ROI 列表为空：没有任何文字框 ⇒ 拼回是空操作"]
            ws = []
        else:
            ws, gaps = roi_windows(rois, scale=scale, shape=shape, align=align)
    if gaps and strict:
        raise DegenerateLayoutError("；".join(gaps))
    if len(patches) != len(ws):
        msg = (f"ROI 块数与格窗口数不符：{len(patches)} vs {len(ws)}"
               f"（逐位一一对应，不允许漏配）")
        if strict:
            raise DegenerateLayoutError(msg)
        gaps = gaps + [msg]

    rep = CompositeReport(policy=policy, align=align, scale=scale, windows=ws,
                         gaps=gaps, total_cells=shape[0] * shape[1],
                         shape=shape, channels=int(z.shape[1]))

    owner = torch.full(shape, -1, dtype=torch.long)     # 每格归属哪个框（-1 = 无主）
    z_out = z.clone()
    batch = int(z.shape[0])

    for w in ws:
        grid_y, grid_x = torch.meshgrid(torch.arange(w.cy0, w.cy1),
                                        torch.arange(w.cx0, w.cx1), indexing="ij")
        taken = owner[w.cy0:w.cy1, w.cx0:w.cx1] >= 0
        rep.contested_cells += int(taken.sum())
        if policy == "error" and bool(taken.any()):
            raise ROIConflictError(
                f"latent 级重叠：框 #{w.index} 的窗口与前序框争夺 "
                f"{int(taken.sum())} 个格（像素级 validate 看不到这种冲突）")
        if policy == "first":
            sel = ~taken
            if not bool(sel.any()):
                rep.notes.append(f"框 #{w.index} 的格已被前序框全部占走 ⇒ 未写入任何内容")
                continue
        else:
            sel = torch.ones_like(taken)
        # ⚠️ py/px 一律取成一维（与 `grid_y[sel]` 的展平顺序一致），
        #    否则「全选」时会得到二维索引而形状对不上
        py = (grid_y - w.cy0)[sel]
        px = (grid_x - w.cx0)[sel]

        for bi in range(batch):
            p, resampled = fit_patch(_batch_patch(patches, w.index, z, bi), w.height, w.width)
            if bi == 0:
                rep.resampled_patches += int(resampled)   # 按**块**计数，不随 batch 放大
            tgt = (grid_y[sel], grid_x[sel])
            if policy == "blend":
                old = z_out[bi][:, tgt[0], tgt[1]]
                z_out[bi][:, tgt[0], tgt[1]] = 0.5 * (old + p[:, py, px])
            else:
                z_out[bi][:, tgt[0], tgt[1]] = p[:, py, px]
        owner[grid_y[sel], grid_x[sel]] = w.index
        rep.written_cells += int(sel.sum())

    if rep.contested_cells:
        rep.notes.append(
            f"有 {rep.contested_cells} 个格被多框覆盖，已按 policy={policy} 裁决"
            "（相邻文字框距离 < 1 个 latent 格就会发生 —— 像素级 validate 判不出来）")
    return z_out, rep


# ---------------------------------------------------------------------------
# ⭐ 判据（独立复算，绝不复用合成循环）
# ---------------------------------------------------------------------------
def expected_composite(z: torch.Tensor, windows: Sequence[CellWindow],
                       patches: Sequence[torch.Tensor], *,
                       policy: str = "first") -> torch.Tensor:
    """⚠️ 这是**判据专用的**复算路径：从 `(z, windows, patches, policy)` 重新累加出
    「应该长什么样」。它与 `composite_latent` 是**两份独立实现** —— 如果只有一份，
    合成器的 bug 会同时污染实现和判据，变成「自己判自己及格」。
    """
    out = z.clone()
    owner = torch.full((int(z.shape[-2]), int(z.shape[-1])), -1, dtype=torch.long)
    batch = int(z.shape[0])
    for w in windows:
        grid_y, grid_x = torch.meshgrid(torch.arange(w.cy0, w.cy1),
                                        torch.arange(w.cx0, w.cx1), indexing="ij")
        taken = owner[w.cy0:w.cy1, w.cx0:w.cx1] >= 0
        sel = ~taken if policy == "first" else torch.ones_like(taken)
        if not bool(sel.any()):
            continue
        py, px = (grid_y - w.cy0)[sel], (grid_x - w.cx0)[sel]
        for bi in range(batch):
            p, _ = fit_patch(patches[w.index] if patches[w.index].dim() == 3
                             else patches[w.index][bi], w.height, w.width)
            tgt = (grid_y[sel], grid_x[sel])
            if policy == "blend":
                old = out[bi][:, tgt[0], tgt[1]]
                out[bi][:, tgt[0], tgt[1]] = 0.5 * (old + p[:, py, px])
            else:
                out[bi][:, tgt[0], tgt[1]] = p[:, py, px]
        owner[grid_y[sel], grid_x[sel]] = w.index
    return out


def verify_composite(z_bg: torch.Tensor, z_out: torch.Tensor, windows: Sequence[CellWindow],
                     patches: Sequence[torch.Tensor], *, policy: str = "first") -> Dict:
    """⭐ 拼回结果的**可证伪判据**。五条同时成立才算通过：

    ① `full_bit_equal`    —— `z_out` 与独立复算出的应有结果**逐位相等**（含框内取值）
    ② `outside_changed == 0` —— 框外区域**逐位不变**（背景没被动过）
    ③ `inside_match_rate == 1.0` —— 框内每一格的值**确实来自给定的 ROI 块**
    ④ `changed_cells > 0` —— 真的改了东西（否则「什么都没写」会被误判为通过）
    ⑤ `roi_energy > 0`    —— **ROI 载荷本身**非全零（全零 ⇒ 框里没有任何文字信息）

    ⭐ ③④⑤ 合起来才让判据**可证伪**：只查 ①③ 的话，把 ROI 换成全零照样「通过」——
    那正是本模块要防的假阳性。
    ⚠️ ⑤ 取的是**输入载荷**的能量而不是输出里框内的能量：policy="blend" 下
    全零载荷仍会让输出非零（背景被折半），只有看载荷才抓得住这种「空写」。
    """
    reasons: List[str] = []
    H, W = int(z_bg.shape[-2]), int(z_bg.shape[-1])
    exp = expected_composite(z_bg, windows, patches, policy=policy)

    inside = torch.zeros((H, W), dtype=torch.bool)
    for w in windows:
        inside[w.cy0:w.cy1, w.cx0:w.cx1] = True
    outside = ~inside

    diff_exp = (z_out != exp)                 # 与「应有结果」的逐元素差异
    diff_bg = (z_out != z_bg)                 # 与「原底图」的逐元素差异
    in_exp, in_bg = diff_exp[:, :, inside], diff_bg[:, :, inside]
    out_bg = diff_bg[:, :, outside]

    full_bit_equal = bool(torch.equal(z_out, exp))
    outside_changed = int(out_bg.sum()) if out_bg.numel() else 0
    inside_changed = int(in_exp.sum()) if in_exp.numel() else 0
    n_in = int(in_exp.numel())
    inside_match_rate = 1.0 if n_in == 0 else float(n_in - inside_changed) / n_in
    changed_cells = int(in_bg.any(dim=1).sum()) if in_bg.numel() else 0
    written_cells = int(inside.sum())
    # 载荷能量只看**输入 ROI 块**：blend 策略下空载荷也会让输出非零，只有看输入才抓得住
    roi_energy = 0.0
    for w in windows:
        for bi in range(int(z_bg.shape[0])):
            p, _ = fit_patch(_batch_patch(patches, w.index, z_bg, bi), w.height, w.width)
            roi_energy += float(p.detach().abs().sum())

    if not full_bit_equal:
        reasons.append(f"① 逐位比对失败：{int((z_out != exp).sum())} 个元素与应有结果不同")
    if outside_changed:
        reasons.append(f"② 框外背景被改动：{outside_changed} 个元素不再与原 latent 相同")
    if inside_changed:
        reasons.append(f"③ 框内内容与 ROI 分支输出不一致：{inside_changed} 个元素对不上")
    if written_cells == 0:
        reasons.append("④ 一个格都没写进去（掩码为空）")
    elif changed_cells == 0:
        reasons.append("④ 写入的格与原值完全相同 ⇒ 无法证明 ROI 内容真的进去了")
    if roi_energy <= 0.0:
        reasons.append("⑤ ROI 载荷能量为 0（给定的 ROI 块全是 0 ⇒ 框内没有任何文字信息可写）")

    return {"ok": not reasons, "reasons": reasons,
            "full_bit_equal": full_bit_equal,
            "outside_changed": outside_changed, "inside_changed": inside_changed,
            "inside_match_rate": round(inside_match_rate, 6),
            "written_cells": written_cells, "changed_cells": changed_cells,
            "roi_energy": round(roi_energy, 6),
            "inside_cells": written_cells, "outside_cells": int(outside.sum())}


# ---------------------------------------------------------------------------
# 合成底图 / ROI 载荷提供者
# ---------------------------------------------------------------------------
def synthetic_latent(batch: int = 1, channels: int = LATENT.total_ch,
                     height: int = 32, width: int = 32, seed: int = 0,
                     noise: float = 0.02) -> torch.Tensor:
    """确定性合成底图 latent（`(B,C,H,W)`，低频正弦 + 固定种子噪声）。

    ⭐ 为什么要有它：链路自检必须在**没有任何真实图片/模型**时可复现地跑通。
    同 `seed` ⇒ 逐位相同 ⇒ 验收数字可比对。
    """
    g = torch.Generator().manual_seed(int(seed))
    yy = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    base = (0.50 * torch.sin(yy / max(1.0, height / 6.0))
            + 0.30 * torch.cos(xx / max(1.0, width / 5.0))
            + 0.20 * torch.sin((xx + yy) / max(1.0, height / 4.0)))
    x = base.expand(batch, channels, height, width).clone()
    if noise:
        x = x + noise * torch.randn(batch, channels, height, width, generator=g)
    return x


def synthetic_image_from_latent(z: torch.Tensor, image_size: int) -> torch.Tensor:
    """把 latent 前 3 通道上采样成一张 `(B,3,S,S)` 的**占位**像素图，喂给 ROI 分支。

    ⚠️ 这**不是** VAE 解码，只为让链路在没有真 VAE 时也能端到端跑起来；
    真实链路应喂「主干解码图」或 `x0` 预测。
    """
    b = int(z.shape[0])
    return F.interpolate(z[:, :3].float(), size=(image_size, image_size),
                         mode="bilinear", align_corners=False).to(z.dtype)


def roi_patches_from_tokens(tokens: torch.Tensor, n_rois: int, tokens_per_roi: int,
                            channels: int, seed: int = 0) -> List[torch.Tensor]:
    """ROIBranch 的 token → 每 ROI 一块 dense `(B,C,side,side)` 补丁。

    ⭐ 为什么需要这一步：ROIBranch 的真实输出是 **cross-attention token**
    `(B, n_roi*T, dim)`（`typography_pack.py`），**不是 dense latent**，
    两者的形状/语义都对不上。补齐这一步，排版链路的三件套才真正闭合。

    ⚠️ `dim → channels` 用的是**固定种子的高斯投影占位**（未训练）。
    它只保证「形状能对上、数值确定可复现」，**不表示任何学习到的解码能力** ——
    真链路里 dense ROI latent 应来自 ROI 分支自己的 decoder。
    """
    if tokens.dim() != 3:
        raise TypographyChainError(f"ROI token 应为 (B,n_roi*T,dim)，收到 {tuple(tokens.shape)}")
    b, ntok, dim = tokens.shape
    side = int(round(math.sqrt(tokens_per_roi)))
    if side * side != tokens_per_roi or ntok != n_rois * tokens_per_roi:
        raise TypographyChainError(
            f"token 形状与 ROI 数不符：{tuple(tokens.shape)}，n_rois={n_rois}, T={tokens_per_roi}")
    g = torch.Generator().manual_seed(int(seed))
    proj = torch.randn(dim, channels, generator=g) / math.sqrt(dim)   # (dim, C)
    t = tokens.reshape(b, n_rois, side, side, dim).permute(0, 1, 4, 2, 3).reshape(b * n_rois, dim, side, side)
    t = (t.permute(0, 2, 3, 1) @ proj).permute(0, 3, 1, 2)             # (N,dim,s,s) → (N,C,s,s)
    # N 的展平顺序是 (batch, roi) ⇒ 第 i 个 ROI 的块是 `t[i::n_rois]`
    # ⚠️ detach：这里只是形状适配器，链路自检不消费梯度；断开 autograd 保证输出确定、无副作用
    return [t[i::n_rois].detach() for i in range(n_rois)]


def token_roi_provider(*, compression: int = 4, tokens_per_roi: int = 16,
                        dim: int = 64, seed: int = 0) -> Callable:
    """构造 provider：**跑真·ROIBranch**（门开、随机初始化、CPU）→ token → dense 补丁。

    ⚠️ 权重是随机初始化的，所以输出**不含任何真实文字语义**；这个 provider 证明的是
    「② 的形状与接线正确」，不是「字写得对」。字形正确性由 T0 字形 provider 负责。
    """

    def provider(image: torch.Tensor, rois, windows, channels: int, **_):
        torch.manual_seed(int(seed))
        branch = ROIBranch(compression=compression, dim=dim, base=16,
                           tokens_per_roi=tokens_per_roi).set_gate(1.0)
        tokens = branch(image, [dict(r) for r in rois])
        if tokens is None:
            raise TypographyChainError("ROIBranch 返回 None（门关 / ROI 为空）—— 链路断了")
        return roi_patches_from_tokens(tokens, len(rois), tokens_per_roi, channels, seed=seed)

    return provider


def constant_roi_provider(value: float = 0.0, side: int = 4) -> Callable:
    """负对照 provider：ROI 载荷 = 常量（`0.0` 即**全零**，用来验证判据的假阳性防线）。"""

    def provider(image, rois, windows, channels: int, **_):
        return [torch.full((int(image.shape[0]), channels, side, side), float(value),
                           dtype=image.dtype) for _ in rois]

    return provider


def noise_roi_provider(seed: int = 1, side: int = 4) -> Callable:
    """负对照 provider：ROI 载荷 = 固定种子随机噪声（形状与真 provider 一致）。"""

    def provider(image, rois, windows, channels: int, **_):
        g = torch.Generator().manual_seed(int(seed))
        return [torch.randn(int(image.shape[0]), channels, side, side, generator=g).to(image.dtype)
                for _ in rois]

    return provider


# ---------------------------------------------------------------------------
# ⭐ 端到端闭环：plan → render → composite
# ---------------------------------------------------------------------------
def run_chain(*, layout: Optional[Mapping] = None, spec: Optional[LayoutSpec] = None,
              image_size: int = 1024, channels: int = LATENT.total_ch,
              scale: int = LATENT.spatial, seed: int = 0,
              provider: Optional[Callable] = None, z_bg: Optional[torch.Tensor] = None,
              policy: str = "first", align: str = "center", pad: int = 0,
              strict: bool = True, shift_cells: int = 0) -> Dict:
    """⭐ 三步闭环（纯 CPU）：`plan → ROI render → composite`，并**当场跑判据**。

    ⚠️ `shift_cells` 是**故意的 bug 注入开关**：把 ROI 矩形整体平移 N 个 latent 格
    再贴，模拟「坐标换算差了一格」这个最常见也最难发现的错误；
    判据始终按**真实**版面复核 ⇒ 框外必然被改 ⇒ 判据必须报错（负对照 C）。

    ⭐ 为什么不用「换一个 scale 来贴」当负对照：换了 scale 就等于换了 latent 网格，
    坐标会直接落到网格外变成「框出界」异常 —— 测到的变成另一条规则，不是这一条。
    """
    lay = dict(layout) if layout is not None else plan(spec or LayoutSpec())
    z = (synthetic_latent(1, channels, image_size // scale, image_size // scale, seed)
         if z_bg is None else z_bg)
    shape = (int(z.shape[-2]), int(z.shape[-1]))
    rois = roi_boxes(lay, pad)
    windows, gaps = roi_windows(rois, scale=scale, shape=shape, align=align)
    if gaps and strict:
        raise DegenerateLayoutError("；".join(gaps))

    img = synthetic_image_from_latent(z, image_size)
    prov = provider or token_roi_provider(seed=seed)
    patches = list(prov(img, rois, windows, channels))

    # ---- bug 注入：整体平移 shift_cells 个格（贴到画布右缘时改向左，避免退化）----
    paste_rois = rois
    if shift_cells:
        canvas_w = shape[1] * scale
        d = shift_cells * scale
        shift = d if all(r["x1"] + d <= canvas_w for r in rois) else -d
        paste_rois = [{**r, "x0": r["x0"] + shift, "x1": r["x1"] + shift} for r in rois]

    z_out, rep = composite_latent(z, patches, rois=paste_rois, scale=scale,
                                  align=align, policy=policy, strict=strict)
    verdict = verify_composite(z, z_out, windows, patches, policy=policy)
    return {"layout": lay, "rois": rois, "paste_rois": paste_rois,
            "windows": windows, "patches": patches, "shift_cells": shift_cells,
            "z_bg": z, "z_out": z_out, "image": img, "report": rep,
            "verify": verdict, "gaps": gaps}


# ---------------------------------------------------------------------------
# ⭐ 验收：正样本 / 负对照 / 退化输入（可证伪，缺一不可）
# ---------------------------------------------------------------------------
def default_spec(image_size: int = 1024) -> LayoutSpec:
    """验收用的默认版面。⚠️ 字号随画布**等比缩放**——写死 96px 在 512² 画布上会
    小到「框比一个 latent 格还薄」，那样失败的是版面而不是链路，验收就失去意义。"""
    k = image_size / 1024.0
    return LayoutSpec(canvas=(image_size, image_size), margin=max(8, int(64 * k)), blocks=[
        TextBlock("心夏北极星", size=max(8, int(96 * k)), align="center"),
        TextBlock("排版链路闭环：Layout Planner → ROI 分支 → 拼回 latent。",
                  size=max(6, int(40 * k))),
        TextBlock("竖排也走同一条路", size=max(8, int(56 * k)), direction="v", align="right"),
    ])


def run_acceptance(*, image_size: int = 1024, channels: int = LATENT.total_ch,
                   scale: int = LATENT.spatial, seed: int = 0,
                   policy: str = "first", align: str = "center",
                   spec: Optional[LayoutSpec] = None) -> Dict:
    """跑完整验收，返回可 JSON 序列化的结论字典。

    ⚠️ 负对照的存在意义：**判据必须能分辨「ROI 真来了」和「什么都没来」**。
    只报「跑通了」的验收等于没验收。

    ⚠️ 本函数**永不抛异常**：任何一环出错都记成一条 `ok=False` 的用例。
    验收装置自己崩掉 = 拿不到结论 = 等于没验收。
    """
    spec = spec or default_spec(image_size)
    cases: List[Dict] = []

    def _case(name: str, kind: str, expect: str, fn: Callable) -> Dict:
        try:
            return fn(name, kind, expect)
        except Exception as exc:                      # noqa: BLE001 — 装置不崩，缺陷记成用例
            return {"name": name, "kind": kind, "expect": expect, "ok": False,
                    "note": f"❌ 执行抛异常：{type(exc).__name__}: {exc}",
                    "reasons": [f"{type(exc).__name__}: {exc}"]}

    # ---------------- 正样本：真·ROI 分支 ----------------
    def _positive(name, kind, expect):
        good = run_chain(spec=spec, image_size=image_size, channels=channels,
                         scale=scale, seed=seed, policy=policy)
        return {"name": name, "kind": kind, "expect": expect,
                "ok": bool(good["verify"]["ok"]), "verify": good["verify"],
                "report": good["report"].as_dict(),
                "note": "✅ 通过" if good["verify"]["ok"] else "❌ 未通过",
                "reasons": good["verify"]["reasons"], "_truth": good["patches"]}

    cases.append(_case("正样本（ROIBranch 载荷）", "positive",
                       "框内逐位来自 ROI 块、框外逐位不变、载荷非零", _positive))
    if not cases[-1].get("verify"):                   # 正样本炸了 ⇒ 后面无从比对，直接收尾
        return {"ok": False, "n_cases": 1, "n_failed": 1, "cases": cases}
    truth = cases[-1].pop("_truth")

    # ---------------- 负对照 A：ROI 全零 ----------------
    def _neg_zero(name, kind, expect):
        zero = run_chain(spec=spec, image_size=image_size, channels=channels,
                         scale=scale, seed=seed, policy=policy,
                         provider=constant_roi_provider(0.0))
        # 判据拿「真 ROI」当期望 ⇒ 必须报出「框内对不上」
        v_true = verify_composite(zero["z_bg"], zero["z_out"], zero["windows"], truth,
                                  policy=policy)
        # 判据拿「全零」当期望 ⇒ 结构检查会过，但载荷能量为 0 ⇒ 仍必须判不合格
        v_zero = verify_composite(zero["z_bg"], zero["z_out"], zero["windows"],
                                  zero["patches"], policy=policy)
        ok = (not v_true["ok"]) and (not v_zero["ok"])
        return {"name": name, "kind": kind, "expect": expect, "ok": bool(ok),
                "verify_vs_true": v_true, "verify_vs_zero": v_zero,
                "note": "✅ 判据能分辨" if ok else "❌ 判据失效（假阳性）",
                "reasons": v_true["reasons"] + v_zero["reasons"]}

    cases.append(_case("负对照 A（ROI 全零）", "negative",
                       "判据必须判不合格（既要比不出真 ROI，也要比得出「空写」）", _neg_zero))

    # ---------------- 负对照 B：ROI 随机噪声 ----------------
    def _neg_noise(name, kind, expect):
        noisy = run_chain(spec=spec, image_size=image_size, channels=channels,
                          scale=scale, seed=seed, policy=policy,
                          provider=noise_roi_provider(seed=seed + 1))
        v = verify_composite(noisy["z_bg"], noisy["z_out"], noisy["windows"], truth,
                             policy=policy)
        return {"name": name, "kind": kind, "expect": expect, "ok": bool(not v["ok"]),
                "verify": v, "note": "✅ 判据能分辨" if not v["ok"] else "❌ 判据失效（假阳性）",
                "reasons": v["reasons"]}

    cases.append(_case("负对照 B（ROI 随机噪声）", "negative",
                       "判据必须判不合格（框内 ≠ 真 ROI）", _neg_noise))

    # ---------------- 负对照 C：坐标换算差一格（最常见也最难发现的 bug） ----------------
    def _neg_scale(name, kind, expect):
        bad = run_chain(spec=spec, image_size=image_size, channels=channels,
                        scale=scale, seed=seed, policy=policy, shift_cells=1)
        v = verify_composite(bad["z_bg"], bad["z_out"], bad["windows"], truth, policy=policy)
        ok = (not v["ok"]) and v["outside_changed"] > 0
        return {"name": name, "kind": kind, "expect": expect, "ok": bool(ok), "verify": v,
                "note": "✅ 判据能分辨" if ok else "❌ 判据失效（框外被改却放行）",
                "reasons": v["reasons"]}

    cases.append(_case("负对照 C（ROI 平移 1 格 ⇒ 坐标换算差一格）", "negative",
                       "框外出现改动 ⇒ 判据必须抓出来（证明「框外逐位不变」不是空话）",
                       _neg_scale))

    # ---------------- 退化输入：必须显式报缺口 ----------------
    deg = _degenerate_cases(image_size, scale, channels)
    for c in deg:
        cases.append(c)

    # ---------------- 重叠框：latent 级冲突裁决 ----------------
    cases += _overlap_cases(image_size, scale, channels)

    overall = all(c["ok"] for c in cases)
    return {"ok": overall, "n_cases": len(cases),
            "n_failed": sum(1 for c in cases if not c["ok"]),
            "cases": cases}


def _mk_box(text: str, x: float, y: float, w: float, h: float, font_px: int = 32) -> Dict:
    return {"text": text, "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "font_px": int(font_px), "align": "left", "direction": "h", "lines": [text]}


def _overlap_cases(image_size: int, scale: int, channels: int) -> List[Dict]:
    """重叠框两个必答项：

    (D) **像素级合法、latent 级冲突** —— 两个框相隔不到一个 latent 格，
        `layout.validate()` 判它们没问题，但贴回时必然抢同一格；
    (E) **像素级就相交** —— `validate` 已判 `valid=False`，链路仍必须给出确定结果
        （而不是崩掉或随机糊），并显式报出争夺了多少格。
    """
    out: List[Dict] = []
    s = float(scale)                                # ⚠️ 全部夹具用**格**为单位表达
    z = synthetic_latent(1, channels, image_size // scale, image_size // scale, 0)
    g = torch.Generator().manual_seed(7)
    patches = [torch.randn(channels, 4, 4, generator=g) for _ in range(2)]

    # (D) 两框像素上不相交（甲 2.05–2.45 格行、乙 2.50–2.90 格行，
    #     两者都严丝合缝落在第 2 个格行内 ⇒ 像素合法、latent 抢格）
    d = {"canvas": [image_size, image_size], "margin": 64, "warnings": [], "valid": True,
         "boxes": [_mk_box("甲", 1.0 * s, 2.05 * s, 6.0 * s, 0.40 * s),
                   _mk_box("乙", 1.0 * s, 2.50 * s, 6.0 * s, 0.40 * s)]}
    pixel_ok = bool(validate(d))
    rois = roi_boxes(d, 0)
    w_cover, gaps_cover = roi_windows(rois, scale=scale, shape=(z.shape[-2], z.shape[-1]),
                                      align="cover")
    z_out, rep = composite_latent(z, patches, rois=rois, scale=scale, align="cover")
    v = verify_composite(z, z_out, w_cover, patches)
    ok_d = pixel_ok and not gaps_cover and rep.contested_cells > 0 and v["ok"]
    out.append({"name": "重叠 D：像素级合法、latent 级抢格（align=cover）", "kind": "overlap",
                "expect": "layout.validate 判合法，但贴回时必须报出争夺格数并仍逐位可判",
                "ok": bool(ok_d), "pixel_valid": pixel_ok, "verify": v,
                "detail": {"contested_cells": rep.contested_cells,
                           "written_cells": rep.written_cells,
                           "notes": rep.notes, "gaps": rep.gaps},
                "note": ("✅ 像素级判据漏掉的冲突被这里抓住并裁决" if ok_d else "❌ 冲突未暴露"),
                "reasons": v["reasons"] + gaps_cover})

    # (E) 两框像素级就相交（上占第 2–4 行、下占第 4–6 行 ⇒ 共抢第 4 行）
    e = {"canvas": [image_size, image_size], "margin": 64, "warnings": [], "valid": False,
         "boxes": [_mk_box("上", 1.0 * s, 2.0 * s, 8.0 * s, 3.0 * s),
                   _mk_box("下", 1.0 * s, 4.0 * s, 8.0 * s, 3.0 * s)]}
    rois = roi_boxes(e, 0)
    ws, gaps = roi_windows(rois, scale=scale, shape=(z.shape[-2], z.shape[-1]))
    per_policy = {}
    all_ok = True
    for pol in ("first", "last", "blend"):
        zo, rp = composite_latent(z, patches, rois=rois, scale=scale, policy=pol, strict=False)
        vv = verify_composite(z, zo, ws, patches, policy=pol)
        per_policy[pol] = {"contested": rp.contested_cells, "verify_ok": vv["ok"],
                           "reasons": vv["reasons"]}
        all_ok = all_ok and vv["ok"] and rp.contested_cells > 0
    try:
        composite_latent(z, patches, rois=rois, scale=scale, policy="error")
        raised = None
        err_ok = False
    except ROIConflictError as exc:
        raised = str(exc)
        err_ok = True
    ok_e = all_ok and err_ok
    out.append({"name": "重叠 E：像素级相交（policy=first/last/blend/error）", "kind": "overlap",
                "expect": "三种 policy 都要给出逐位可复核的确定结果；policy=error 必须抛 ROIConflictError",
                "ok": bool(ok_e), "per_policy": per_policy, "error_raised": raised,
                "note": ("✅ 冲突有确定裁决" if ok_e else "❌ 冲突未确定性处理"),
                "reasons": [r for p in per_policy.values() for r in p["reasons"]]})
    return out


def _gap_case(name: str, expect: str, ok: bool, detail: Dict,
              note: str = "") -> Dict:
    return {"name": name, "kind": "degenerate", "expect": expect, "ok": bool(ok),
            "note": note or ("✅ 显式报缺口" if ok else "❌ 静默返回了"), "detail": detail}


def _degenerate_cases(image_size: int, scale: int, channels: int) -> List[Dict]:
    """退化输入：空 Layout / 零个文字框 / 框完全落在画布外 / 框比一个格还薄。

    ⚠️ 所有夹具的坐标都按 **scale 的倍数**表达。写死像素值会让这些用例
    在别的 `--scale` 下「因为错误的原因通过」（例如顺手触发了「块数不符」），
    那样验收就是自欺欺人。
    """
    out: List[Dict] = []
    s = float(scale)
    z = synthetic_latent(1, channels, image_size // scale, image_size // scale, 0)

    # (1) 空 Layout：连 boxes 键都没有
    try:
        composite_latent(z, [], layout={}, scale=scale, strict=True)
        out.append(_gap_case("退化 1：空 Layout（无 boxes 键）", "必须抛 DegenerateLayoutError",
                             False, {"raised": None}))
    except DegenerateLayoutError as e:
        out.append(_gap_case("退化 1：空 Layout（无 boxes 键）", "必须抛 DegenerateLayoutError",
                             True, {"raised": str(e)}))

    # (2) 零个文字框
    empty = {"canvas": [image_size, image_size], "margin": 64, "boxes": [], "warnings": []}
    try:
        composite_latent(z, [], layout=empty, scale=scale, strict=True)
        out.append(_gap_case("退化 2：零个文字框", "必须抛 DegenerateLayoutError",
                             False, {"raised": None}))
    except DegenerateLayoutError as e:
        out.append(_gap_case("退化 2：零个文字框", "必须抛 DegenerateLayoutError",
                             True, {"raised": str(e)}))

    # (3) 框完全落在画布外
    outside = {"canvas": [image_size, image_size], "margin": 64, "warnings": [], "valid": False,
               "boxes": [_mk_box("界外", image_size * 1.5, image_size * 1.5,
                                 4.0 * s, 1.5 * s)]}
    try:
        composite_latent(z, [torch.zeros(channels, 1, 1)], layout=outside, scale=scale, strict=True)
        out.append(_gap_case("退化 3：框完全落在画布外", "必须抛 DegenerateLayoutError",
                             False, {"raised": None}))
    except DegenerateLayoutError as e:
        out.append(_gap_case("退化 3：框完全落在画布外", "必须抛 DegenerateLayoutError",
                             True, {"raised": str(e)}))

    # (4) strict=False 时也必须把缺口写进报告（不能悄悄返回一张没变的图）
    z_out, rep = composite_latent(z, [], layout=outside, scale=scale, strict=False)
    ok4 = bool(rep.gaps) and bool(torch.equal(z_out, z))
    out.append(_gap_case("退化 4：strict=False 仍需在报告里列出缺口",
                         "报告.gaps 非空，且输出 == 原 latent（确认是空操作而非乱写）",
                         ok4, {"gaps": rep.gaps, "unchanged": bool(torch.equal(z_out, z))}))

    # (5) 框太薄：整框落在第 2 格行的前半段（2.05–2.40 格），**不含格心 2.5 格**
    #     ⇒ 格心判定下无处可放，但 cover 判定下有 1 行 ⇒ 缺口确实来自判定口径
    thin = {"canvas": [image_size, image_size], "margin": 64, "warnings": [], "valid": True,
            "boxes": [_mk_box("细", 1.0 * s, 2.05 * s, 6.0 * s, 0.35 * s),
                      _mk_box("正常", 1.0 * s, 4.0 * s, 6.0 * s, 2.0 * s)]}
    n_rois = len(roi_boxes(thin, 0))
    pads = [torch.zeros(channels, 1, 1) for _ in range(n_rois)]  # ⚠️ 块数要配平，否则测的是另一条规则
    try:
        composite_latent(z, pads, layout=thin, scale=scale, strict=True)
        out.append(_gap_case("退化 5：框比一个 latent 格还薄", "必须抛 DegenerateLayoutError",
                             False, {"raised": None, "n_patches": len(pads)}))
    except DegenerateLayoutError as e:
        out.append(_gap_case("退化 5：框比一个 latent 格还薄", "必须抛 DegenerateLayoutError",
                             True, {"raised": str(e), "n_patches": len(pads)}))
    # 同一份输入在 align='cover' 下能写进去 —— 证明缺口是「格心判定」造成的，不是几何错
    ws_c, gaps_c = windows_from_layout(thin, scale=scale, shape=(z.shape[-2], z.shape[-1]),
                                       align="cover")
    out.append(_gap_case("退化 5b：同一输入在 align='cover' 下可写（对照组）",
                         "必须无缺口且窗口非空",
                         (not gaps_c) and len(ws_c) == 2,
                         {"n_windows": len(ws_c), "gaps": gaps_c,
                          "cells": [w.cells for w in ws_c]},
                         note="✅ 对照通过：缺口来自「格心判定」，不是几何算错"))
    return out


__all__ = [
    "CellWindow", "CompositeReport", "TypographyChainError", "DegenerateLayoutError",
    "ROIConflictError", "pixel_rect_to_cells", "roi_windows", "windows_from_layout",
    "fit_patch", "composite_latent", "expected_composite", "verify_composite",
    "synthetic_latent", "synthetic_image_from_latent", "roi_patches_from_tokens",
    "token_roi_provider", "constant_roi_provider", "noise_roi_provider",
    "run_chain", "run_acceptance", "default_spec",
]
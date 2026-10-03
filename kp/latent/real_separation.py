"""G2 真图版 —— 把通道分离验收从**合成数据**推进到**真实图像 + 真实 VAE 编码器**。

════ 为什么需要这个模块 ════
`kp/latent/separation.py` 已经把装置建好了（尺子 oracle / leaky、判据 = 交叉扰动下的
分支依赖度、门线 ≤ 0.20），但它**喂的是合成 latent**。它证明的是
「**装置正确 + 显式监督在可控数据上有效**」，
**不等于「真图经过 32× 混合 VAE 编码后，两块通道还能被分开」**。

本模块把链路接完整：
    真实 PNG（RGBA） → 背景合成 + 确定性视图增广 → HybridVAE.encode
    → 40ch 混合 latent（8ch 语义 + 32ch 细节） → 交叉扰动 → 依赖度判据

════ 三条必须写在最前面的诚实声明 ════
① **仓库里没有训练好的 HybridVAE 权重。**
   `kp/models/vae.py` 的 docstring 自己写着「骨架只要求形状与梯度正确 …… 真正的 VAE
   要蒸馏 DC-AE，权重走单独的训练线，**不在这份参考实现的范围里**」。全仓扫描确认：
   无任何 HybridVAE 的 checkpoint（`models/` 下唯一的 VAE 权重是 Sana 的，通道数都不是
   40ch，且按任务边界不得触碰）。
   ⇒ 「真图版」真正能做的只有两件事，本模块都做了：
     · Arm A：把**随机初始化**的编码器冻结，把真图 latent 喂给既有 G2 装置
       （可复现，但**信息量为零**：随机投影当然可分）；
     · Arm B/C/D/E：让编码器**被真实像素 + 显式监督训起来**，
       问的是「**这套架构 + 这套监督，在真实图像上能不能把 8/32 两块逼开**」。

② **真图独立样本量少，必须显式记账。**
   `data/characters/kokona/images/` 只有 `front.png` / `back.png` 两张去底角色图 ——
   交叉扰动需要 batch ≥ 2，两张图**刚好卡在退化线上**（批内置换只有 2! 种），
   实测此时 latent 几乎是秩 2 的，**打乱另一块通道在 latent 上几乎不留痕迹**，
   依赖度会假低。所以默认语料 = 角色立绘 + `out/e5b/g1/bf16` 的真实位图。
   ⇒ 报告里 **三个数分开报**：`n_sources`（来源文件数）、`n_independent_stems`
     （去掉 `_sNNN` 采样步后缀后的**独立图像族数**，更保守）、`n_views`（增广后张数）。
     `n_independent_stems < MIN_SOURCES` 时写一条**显式缺口**。
     ⛔ 禁止用单张/两张图得出「真图能分离」的结论。

③ **合成对照必须同时跑。**
   真图上**没有已知答案**（没有「这两块本来该怎么分」的真值），
   所以装置的**分辨力**只能靠对照证明：
   · 尺子：`oracle`（构造上分离）必须 PASS、`leaky`（语义分支读细节块）必须 FAIL；
   · 负对照：同一批真图、同一损失，关掉交叉不变性（`w_inv=0`）应当变差；
   · 合成对照：完整跑一遍既有 `separation.run_g2()`，确认装置本体没坏。
   对照不成立 ⇒ 真图上的数字不可信，直接报缺口。

════ 本模块对「单点采样」与「退化输入」的额外加固 ════
`separation.evaluate()` 的三行（shuffle_detail / shuffle_semantic / reverse）在
**同一个 seed 下算出完全相同的数** —— `branch_dependency()` 内部固定跑那两种置换，
`mode` 只是标签。顺带一提：`cross_perturb(..., "reverse")` 对**两块用同一个 `flip(0)`**，
等于整张 latent 换了个批内编号，**配对没被打散**，对分支输出是恒等变换 ——
它其实不是一种扰动。（以上是既有文件的既有行为，本模块**只记录、不改动**。）

所以本模块自带**更严的判据聚合**：
  · `--n-perm` 个**独立置换**（seed 逐个递增）⇒ 报 mean / max / std，**不报单点**；
  · 三种扰动分别真算（shuffle_detail / shuffle_semantic / reverse）；
  · 判定取 **max（最坏置换）**，比单点更保守，**没有放松门线**；
  · 每个置换报 `perm_energy`（扰动实际动了多少）—— 置换若近似恒等，依赖度就是假 0；
  · 每个分支报 `liveness`（输出相对 0 的量级）—— 分母塌到 0 时依赖度会**假过**，
    这是必须显式拦掉的退化情形（退化输入不得静默放假通过）。

⚠️ 本模块**只读** `separation.py` / `hybrid.py` / `models/vae.py`，不改它们。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LATENT
from ..models.vae import HybridVAE
from ..paths import DATA, OUT
from . import separation as sep
from .hybrid import channel_mi_penalty, split_channels

__all__ = [
    "BG_CHOICES", "MIN_SOURCES", "kokona_image_dir", "scenes_image_dir",
    "kp_out_dir",
    "RealViewSet", "load_real_views", "LatentBundle", "encode_views",
    "latent_stats", "latent_gaps", "block_cross_r2",
    "perturbation_energy", "branch_liveness", "dependency_over_permutations",
    "RealReport", "format_report", "run_rulers",
    "train_joint", "train_semantic_routed", "latent_probe_r2", "run_real_g2",
]

# 独立来源少于这个数 ⇒ 必须报缺口（不能据此说「真图能分离」）
MIN_SOURCES = 4

BG_CHOICES = ("white", "black")

# 确定性视图增广计划：(缩放, dx, dy)。dx/dy 是裁切窗在边长上的相对偏移 ∈ [-0.5, 0.5]。
# ⚠️ 刻意**不含左右翻转**：角色图翻面会造出解剖上不成立的视图。
#    增广只做「轻微裁切 + 平移」，目的是让 batch ≥ 8 以便构造置换，
#    **不是为了增加独立信息量**（独立来源仍然只有 n_sources 张）。
_VIEW_PLAN: Tuple[Tuple[float, float, float], ...] = (
    (1.00, 0.00, 0.00),
    (0.94, 0.00, 0.00), (0.94, 0.14, 0.05), (0.94, -0.12, 0.11),
    (0.88, 0.00, 0.00), (0.88, 0.16, -0.13), (0.88, -0.15, 0.09),
    (0.82, 0.00, 0.00),
)


# ===========================================================================
# 0. 路径 / 产物
# ===========================================================================
def kokona_image_dir(batch: str = "kokona") -> Path:
    """仓库内真实角色图目录（默认 `data/characters/kokona/images`）。"""
    return DATA / "characters" / batch / "images"


def scenes_image_dir() -> Path:
    """仓库内第二套真实图像语料（E5B/G1 采样出的真实位图，非角色立绘）。

    ⛔ 不是 `models/` 下的 Sana 权重，只是它**采样出来的 PNG**（纯 CPU 可读，不碰模型）。
    存在的意义只有一个：**把 G2 真图版的独立样本量从 2 张提到几十张** ——
    只有 2 张时 latent 几乎是秩 2 的，交叉扰动根本没有作用面。
    """
    return kp_out_dir() / "e5b" / "g1" / "bf16"


def kp_out_dir() -> Path:
    """临时产物目录：环境变量 `KP_OUT` 优先，否则 `kp.paths.OUT`。
    ⛔ 绝不用 `tempfile.gettempdir()`（跨机不可迁移）。"""
    env = os.environ.get("KP_OUT")
    return Path(env).expanduser().resolve() if env else OUT


# ===========================================================================
# 1. 真实图像装载（alpha 处理在这里，且必须说清口径）
# ===========================================================================
@dataclass
class RealViewSet:
    """一批真实图像视图。⚠️ `n_views` ≠ 独立样本量，看 `n_sources`。"""
    images: torch.Tensor                     # (N,3,size,size) ∈ [-1,1]
    sources: List[str]
    variants: List[str]
    size: int
    background: str
    alpha_policy: str
    body_ratio: float
    dirs: List[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return int(self.images.shape[0])

    @property
    def n_sources(self) -> int:
        return len(set(self.sources))

    @property
    def n_stems(self) -> int:
        """**去掉采样步后缀**后的独立来源数（更保守的样本量口径）。

        `out/e5b/g1/bf16/01_en_scene_s100.png` 与 `..._s101.png` 是**同一个 seed 的
        相邻采样步**，图像高度相关 ⇒ 按文件数算样本量会高估。
        这里把 `_s\\d+$` 去掉再数，得到真正独立的图像族数。
        """
        import re
        stems = {re.sub(r"_s\d+$", "", s.rsplit("/", 1)[-1].rsplit(".", 1)[0])
                 for s in set(self.sources)}
        return len(stems)

    def gaps(self) -> List[str]:
        """显式缺口：样本量、来源数、分辨率。**不合格不得静默**。"""
        g: List[str] = []
        if self.n_stems < MIN_SOURCES:
            g.append(
                f"样本量缺口：独立图像族（去掉 `_sNNN` 采样步后缀）仅 {self.n_stems} 族"
                f"（< {MIN_SOURCES}）—— 文件数 {self.n_sources} 张里有相邻采样步，"
                f"高度相关；增广到 {self.n_views} 个视图只增加**测量稳定性**，"
                f"不增加独立信息量")
        if self.n_sources <= 2:
            g.append(
                f"批内置换空间仅 {self.n_sources}! 种 ⇒ 交叉扰动的「打乱」强度"
                f"天然受限，依赖度会被系统性**压低**（乐观偏置）")
        if self.size < 256:
            g.append(f"输入分辨率 {self.size}² 过低，32× 压缩后 latent 仅 "
                     f"{self.size // LATENT.spatial}² 格，细节通道几乎没有空间可分")
        return g

    def to_dict(self) -> dict:
        return {
            "n_views": self.n_views,
            "n_sources": self.n_sources,
            "n_independent_stems": self.n_stems,
            "dirs": list(self.dirs),
            "sources": sorted(set(self.sources)),
            "variants": sorted(set(self.variants)),
            "size": self.size,
            "background": self.background,
            "alpha_policy": self.alpha_policy,
            "body_ratio": self.body_ratio,
            "range": "[-1,1]（kp/character/dataset.py:load_image 的仓库口径）",
            "gaps": self.gaps(),
        }


def _composite_over(rgba: torch.Tensor, bg: float) -> torch.Tensor:
    """(4,S,S) 的 RGBA → (3,S,S)，背景按 `bg` 填。

    ⚠️ 必须**先做 alpha 合成再取值**：RGBA 里透明区的 RGB 是未定义值
    （PNG 解码出来通常是 0 或残留色），直接 `convert("RGB")` 会把黑边
    当成角色本体编码进 latent。这里用标准 over 合成：`rgb*a + bg*(1-a)`。
    """
    rgb, a = rgba[:3], rgba[3:4].clamp(0, 1)
    return rgb * a + (1.0 - a) * bg


def _crop_to_alpha(rgba: torch.Tensor, thresh: int = 8) -> torch.Tensor:
    """按 alpha 界框裁切（口径同 `kp/character/pipeline.py:stage_normalize`）。"""
    mask = rgba[3] > (thresh / 255.0)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        raise ValueError("图像全透明（去底失败？）")
    return rgba[:, int(ys.min()):int(ys.max()) + 1,
                int(xs.min()):int(xs.max()) + 1]


def _fit_to_canvas(rgba: torch.Tensor, size: int, body_ratio: float,
                   bg: float) -> torch.Tensor:
    """裁到本体 → 等比缩放到 `body_ratio·size` → 居中贴到方画布 → 合成到背景。"""
    s = max(1, int(round(size * body_ratio)))
    ch, cw = rgba.shape[1], rgba.shape[2]
    scale = s / max(ch, cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    body = F.interpolate(rgba.unsqueeze(0), size=(nh, nw),
                         mode="bilinear", align_corners=False)[0]
    canvas = torch.zeros((4, size, size), dtype=rgba.dtype)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[:, oy:oy + nh, ox:ox + nw] = body           # alpha 覆盖写入
    return _composite_over(canvas, bg)


def _augment(img: torch.Tensor, zoom: float, dx: float, dy: float) -> torch.Tensor:
    """中心窗裁切（窗大小 = side/zoom）→ 缩放回 side。纯张量操作，可复现。"""
    s = img.shape[-1]
    cw = max(8, min(s, int(round(s / zoom))))
    x0 = max(0, min(s - cw, int(round((s - cw) * (0.5 + dx)))))
    y0 = max(0, min(s - cw, int(round((s - cw) * (0.5 + dy)))))
    crop = img[:, y0:y0 + cw, x0:x0 + cw].unsqueeze(0)
    return F.interpolate(crop, size=(s, s), mode="bilinear",
                         align_corners=False)[0]


def _collect_files(dirs: Sequence[Path], max_views: int) -> List[Tuple[str, Path]]:
    """多目录收集 + **两轮配额**：① 每个目录先领 `max_views // n_dirs` 个额度；
    ② 目录内部按「图像族」轮转取（`_sNNN` 采样步后缀归为同一族）。

    两个配额各解决一个坑：
      · 目录配额：`data/characters/kokona/images` 只有 2 张，而
        `out/e5b/g1/bf16` 有 39 张 —— 不限流的话场景图会把角色立绘整个挤掉。
      · 族内轮转：`bf16` 里文件名按 `01_en_scene_s100..s115` 排序，
        直接取前 N 个会**全落在同一个 prompt 上**（实测：8 张来源只有 3 个独立族）。
        轮转取用能保证 3 个 prompt（en_scene / zh_text / anime）都进得来。
    """
    import re

    def stem_of(p: Path) -> str:
        return re.sub(r"_s\d+$", "", p.stem)

    per_dir: List[List[Path]] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"真实图像目录不存在：{d}")
        fs = sorted(p for p in d.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if not fs:
            raise FileNotFoundError(f"{d} 下没有可用图像")
        fam: Dict[str, List[Path]] = {}
        for p in fs:
            fam.setdefault(stem_of(p), []).append(p)
        interleaved: List[Path] = []
        k = 0
        while len(interleaved) < len(fs):
            for s in sorted(fam):
                if k < len(fam[s]):
                    interleaved.append(fam[s][k])
            k += 1
        per_dir.append(interleaved)
    quota = max(1, max_views // max(1, len(dirs)))
    picked = {i: min(quota, len(per_dir[i])) for i in range(len(dirs))}
    # 轮转取用：每个目录先领 `quota` 个额度，谁先领完谁让位，
    # 保证「角色立绘（只有 2 张）不会被场景图（39 张）整个挤掉」。
    out: List[Tuple[str, Path]] = []
    idx = {i: 0 for i in range(len(dirs))}
    while len(out) < max_views and any(idx[i] < picked[i] for i in range(len(dirs))):
        for i in range(len(dirs)):
            if len(out) >= max_views:
                break
            if idx[i] < picked[i]:
                out.append((dirs[i].name, per_dir[i][idx[i]]))
                idx[i] += 1
    return out


def load_real_views(image_dir: Optional[os.PathLike | str | Sequence] = None, *,
                    size: int = 384, background: str = "white",
                    body_ratio: float = 0.80, views_per_source: int = 2,
                    max_views: int = 24,
                    require_batch2: bool = True) -> RealViewSet:
    """读真实 PNG → 合成背景 → 确定性视图增广 → `(N,3,size,size) ∈ [-1,1]`。

    `image_dir` 可以是**一个或多个目录**（默认：`data/characters/kokona/images`
    + `out/e5b/g1/bf16`，后者不存在时自动跳过）。
    `size` 必须能被 `LATENT.spatial`(32) 整除（5 级 stride-2）。
    ⚠️ `require_batch2=True` 时 **batch < 2 直接抛错**，不返回「永远通过」的单样本。
    """
    if size % LATENT.spatial:
        raise ValueError(f"size={size} 不是 {LATENT.spatial} 的整数倍（32× 压缩）")
    if background not in BG_CHOICES:
        raise ValueError(f"background 只能是 {BG_CHOICES}")
    if image_dir is None:
        dirs = [kokona_image_dir()]
        if scenes_image_dir().is_dir():
            dirs.append(scenes_image_dir())
    elif isinstance(image_dir, (str, os.PathLike)):
        dirs = [Path(image_dir)]
    else:
        dirs = [Path(d) for d in image_dir]
    files = _collect_files(dirs, max(2, max_views // max(1, views_per_source)))

    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError("需要 Pillow + numpy") from e

    bg = 1.0 if background == "white" else 0.0
    imgs, srcs, names = [], [], []
    for tag, f in files:
        with Image.open(f) as im:
            arr = torch.from_numpy(
                np.asarray(im.convert("RGBA"), dtype="float32") / 255.0
            ).permute(2, 0, 1).contiguous()              # (4,H,W) ∈[0,1]
        base = _fit_to_canvas(_crop_to_alpha(arr), size, body_ratio, bg)
        for (z, dx, dy) in _VIEW_PLAN[:max(1, views_per_source)]:
            imgs.append(_augment(base, z, dx, dy))
            srcs.append(f"{tag}/{f.name}")
            names.append(f"z{z:.2f}_dx{dx:+.2f}_dy{dy:+.2f}")
    x = torch.stack(imgs)
    if require_batch2 and x.shape[0] < 2:
        raise ValueError(
            f"G2 真图验收需要 batch ≥ 2（退化输入必须显式报缺口），"
            f"当前只有 {x.shape[0]} 个视图")
    return RealViewSet(
        images=x, sources=srcs, variants=names, size=size,
        background=background,
        alpha_policy=(f"alpha>8 裁本体界框 → over 合成到 {background} 底"
                     f"（透明区不参与编码）"),
        body_ratio=body_ratio, dirs=[str(d) for d in dirs])


# ===========================================================================
# 2. 真实 VAE 编码
# ===========================================================================
@dataclass
class LatentBundle:
    """编码结果 + 活性诊断（**塌成常数的 latent 会让依赖度假过**，必须查）。"""
    z: torch.Tensor
    stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"shape": list(self.z.shape),
             "semantic_ch": LATENT.semantic_ch, "detail_ch": LATENT.detail_ch,
             "latent_side": int(self.z.shape[-1])}
        d.update(self.stats)
        return d


def latent_stats(z: torch.Tensor) -> Dict[str, float]:
    """latent 活性诊断 + 通道间可预测性。

    ⚠️ `collapsed` 为真 ⇒ 编码器塌成常数 ⇒ 依赖度分母趋 0 ⇒ **假 PASS**。
    ⚠️ `view_sensitivity`（跨视图 std ÷ 视图内 std）很小时 ⇒ latent 对输入几乎不敏感
       ⇒ 打乱另一块通道时「换了个样本」这件事根本没在 latent 上留下痕迹
       ⇒ 依赖度低是**数据退化的产物**，不是分离得好。这是最隐蔽的一种假过。
    `cross_r2_*` 独立于 mixer：直接问「一块能不能线性预测另一块」——
    这才是「两块通道真的解耦了吗」的通道级证据。
    """
    sem, det = split_channels(z)
    flat = z.flatten(1)
    inner = float(flat.std(dim=1).mean())
    inter = float(flat.std(dim=0).mean())
    st = {
        "abs_mean": float(z.abs().mean()),
        "per_view_std": inner,
        "inter_view_std": inter,
        "view_sensitivity": inter / (inner + 1e-12),
        "sem_abs_mean": float(sem.abs().mean()),
        "det_abs_mean": float(det.abs().mean()),
        "sem_over_det_energy": float(sem.abs().mean() / (det.abs().mean() + 1e-12)),
        "mi_penalty": float(channel_mi_penalty(sem, det, n_samples=4096, seed=0)),
    }
    st["collapsed"] = bool(inner < 1e-4)
    st["degenerate"] = bool(st["view_sensitivity"] < 0.05)
    st.update(block_cross_r2(z))
    return st


def latent_gaps(st: Dict[str, float], tag: str) -> List[str]:
    """把 latent 的退化诊断翻成**显式缺口**（不静默）。"""
    g: List[str] = []
    if st.get("collapsed"):
        g.append(f"[{tag}] latent 塌成常数（视图内 std={st['per_view_std']:.2e}）"
                 f" ⇒ 依赖度分母趋 0，**判为假过**")
    if st.get("degenerate"):
        g.append(f"[{tag}] latent 对输入几乎不敏感"
                 f"（跨视图/视图内 std = {st['view_sensitivity']:.4f} < 0.05）"
                 f" ⇒ 打乱另一块通道在 latent 上几乎不留痕迹，"
                 f"依赖度低是**数据退化**而非分离得好")
    if st.get("probe_r2", 1.0) < 0.05:
        g.append(f"[{tag}] latent 对图像的线性可读出 R²={st['probe_r2']:.3f}（<0.05）"
                 f" ⇒ latent 基本没带图像信息")
    return g


def _r2(feat: torch.Tensor, tgt: torch.Tensor) -> float:
    """**逐位置**线性可预测性 R²（feat/tgt 同为 `(N,C,h,w)`）。

    ⚠️ 为什么不是「按图展平」再回归：那样样本数只有 N（16），而特征数是
       `40·h·w`（几千），最小二乘会**过拟合到 R²≈1** —— 数字看着漂亮，
       其实一条信息都没预测到（本模块第一版就踩了这个坑：所有臂都报 1.0000）。
       把 `N·h·w` 个**空间位置**当样本才是良态问题（2304 样本 / 41 特征）。
    """
    f = feat.detach().permute(0, 2, 3, 1).reshape(-1, feat.shape[1]).double()
    t = tgt.detach().permute(0, 2, 3, 1).reshape(-1, tgt.shape[1]).double()
    f = torch.cat([f, torch.ones(f.shape[0], 1, dtype=torch.float64)], 1)
    sol = torch.linalg.lstsq(f, t).solution
    res = float(((f @ sol) - t).pow(2).sum())
    tot = float((t - t.mean(0, keepdim=True)).pow(2).sum())
    return 1.0 - res / max(tot, 1e-12)


def block_cross_r2(z: torch.Tensor) -> Dict[str, float]:
    """跨块线性可预测性：R²(细节|语义) 与 R²(语义|细节)。**越低越解耦**。

    这是**通道级**证据，完全不经过 mixer —— 与分支依赖度互为印证。
    """
    sem, det = split_channels(z)
    return {"cross_r2_detail_from_sem": _r2(sem, det),
            "cross_r2_sem_from_detail": _r2(det, sem)}


def encode_views(views: RealViewSet, *, base: int = 16, seed: int = 0,
                 batch: int = 0) -> Tuple[HybridVAE, LatentBundle]:
    """真实图像 → `HybridVAE.encode` → 40ch 混合 latent（fp32 / 纯 CPU / 不做后验采样）。

    ⚠️ `HybridVAE.encode` **内部没有任何归一化**（`nn.Sequential` 直上直下），
       输入口径由调用方负责。本仓库的图像口径是 `[-1,1]`
       （见 `kp/character/dataset.py:load_image`），本模块沿用。
    """
    torch.manual_seed(seed)
    vae = HybridVAE(base=base)
    vae.eval()
    z_parts = []
    with torch.no_grad():
        step = batch or views.n_views
        for i in range(0, views.n_views, max(1, step)):
            z_parts.append(vae.encode(views.images[i:i + step]))
    z = torch.cat(z_parts, 0)
    return vae, LatentBundle(z=z, stats=latent_stats(z))


@torch.no_grad()
def latent_probe_r2(vae: HybridVAE, z: torch.Tensor, imgs: torch.Tensor) -> float:
    """`z` 对图像内容的线性可读出 R²（**独立于训练**，只作事后体检）。

    R² ≈ 0 ⇒ latent 没带图像信息 ⇒ 依赖度再「漂亮」也没有意义。
    """
    _ = vae
    return _r2(z, F.adaptive_avg_pool2d(imgs, z.shape[-2:]))


# ===========================================================================
# 3. 判据聚合：多置换 + 多扰动 + 活性守卫
# ===========================================================================
def perturbation_energy(z: torch.Tensor, mode: str, seed: int) -> Dict[str, float]:
    """这个置换**实际把 latent 动了多少**（相对自身尺度）。

    ≈ 0 ⇒ 「扰动」近似恒等 ⇒ 后面测到的依赖度 0 是假的，必须报缺口。
    """
    zp = sep.cross_perturb(z, mode, seed=seed)
    out = {}
    for tag, sl in (("semantic", slice(0, LATENT.semantic_ch)),
                    ("detail", slice(LATENT.semantic_ch, None))):
        out[f"perm_energy_{tag}"] = float(
            (zp[:, sl] - z[:, sl]).abs().mean() / (float(z[:, sl].abs().mean()) + 1e-12))
    return out


def branch_liveness(branch, z: torch.Tensor) -> Dict[str, float]:
    """分支输出相对 0 的量级（用它自己那一块 latent 做尺度）。

    ⚠️ `branch_dependency` 的分母是 `|Δ| + |branch(z)|`：若分支是**常函数**
       （两侧都不看），`|Δ| = |branch(z)| = 0`，依赖度算出来是 `0/eps = 0`
       ⇒ **会假过**。所以必须显式查活性。
    """
    out = {}
    with torch.no_grad():
        for tag, fn, sl in (("semantic", branch.sem_path, slice(0, LATENT.semantic_ch)),
                            ("detail", branch.det_path,
                             slice(LATENT.semantic_ch, None))):
            out[f"liveness_{tag}"] = float(fn(z).abs().mean()) / (
                float(z[:, sl].abs().mean()) + 1e-12)
    return out


def _agg(vals: Sequence[float]) -> Dict[str, float]:
    t = torch.tensor(list(vals), dtype=torch.float64)
    return {"mean": float(t.mean()), "max": float(t.max()),
            "std": float(t.std(unbiased=False)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


@dataclass
class RealReport:
    """把 `n_perm` 个独立置换聚合成「一行一个扰动」的判据表。"""
    rows: List[dict] = field(default_factory=list)
    max_dep: float = 0.20
    n_perm: int = 1

    @property
    def overall(self) -> bool:
        return bool(self.rows) and all(r["pass"] for r in self.rows)

    def to_dict(self) -> dict:
        return {"max_dep": self.max_dep, "n_perm": self.n_perm,
                "overall_pass": self.overall, "rows": self.rows}


def dependency_over_permutations(branch, z: torch.Tensor, *,
                                 n_perm: int = 8, seed: int = 0,
                                 max_dep: float = 0.20,
                                 include_reverse: bool = True) -> RealReport:
    """多置换版交叉扰动判据（公式与 `separation.branch_dependency` 完全一致）。

    判定口径：**取 max（最坏置换）**，比 `separation.evaluate()` 的单点更保守。
    聚合：同一扰动下跑 `n_perm` 个**独立置换**（seed 逐个递增），
    报 mean / max / std —— ⛔ 不报单点（项目踩过「某个 batch 恰好测出 0.0」的坑）。
    """
    if z.shape[0] < 2:
        raise ValueError("dependency_over_permutations 需要 batch ≥ 2"
                         "（退化输入必须显式报缺口）")
    rep = RealReport(max_dep=max_dep, n_perm=n_perm)
    live = branch_liveness(branch, z)

    plans = [("shuffle_detail", "semantic"), ("shuffle_semantic", "detail")]
    if include_reverse:
        # ⚠️ `cross_perturb(..., "reverse")` 对两块用同一个 flip(0) ⇒ 配对未被打散，
        #    对分支输出近似恒等变换。仍然算出来并**如实报能量**，
        #    让「它几乎不动」变成数字而不是猜测。
        plans.append(("reverse", "both"))

    for mode, which in plans:
        d_sem, d_det, e_sem, e_det = [], [], [], []
        for k in range(n_perm):
            s = seed + k
            r = sep.branch_dependency(branch, z, mode, seed=s)
            d_sem.append(r["dep_semantic"])
            d_det.append(r["dep_detail"])
            e = perturbation_energy(z, mode, s)
            e_sem.append(e["perm_energy_semantic"])
            e_det.append(e["perm_energy_detail"])
        row = {"mode": mode, "which_side_perturbed": which,
               "dep_semantic": _agg(d_sem), "dep_detail": _agg(d_det),
               "perm_energy_semantic": _agg(e_sem),
               "perm_energy_detail": _agg(e_det), **live}
        worst_sem, worst_det = row["dep_semantic"]["max"], row["dep_detail"]["max"]
        row["worst_dep"] = max(worst_sem, worst_det)
        row["sem_pass"] = worst_sem <= max_dep
        row["det_pass"] = worst_det <= max_dep
        # 活性守卫：分支若对两侧都是常函数，依赖度没有意义
        row["liveness_ok"] = bool(live["liveness_semantic"] > 1e-3
                                  and live["liveness_detail"] > 1e-3)
        row["pass"] = bool(row["sem_pass"] and row["det_pass"] and row["liveness_ok"])
        rep.rows.append(row)
    return rep


def format_report(rep: RealReport, title: str) -> str:
    head = (f"{'扰动':<18}{'语义依赖 mean':>14}{'max':>9}{'细节依赖 mean':>14}"
            f"{'max':>9}{'判定':>8}")
    out = [f"=== {title} ===", f"    {rep.n_perm} 个独立置换，门线 ≤ {rep.max_dep:.2f}"
           "（取最坏置换）", head, "-" * 74]
    for r in rep.rows:
        out.append(
            f"{r['mode']:<18}{r['dep_semantic']['mean']:>14.4f}"
            f"{r['dep_semantic']['max']:>9.4f}{r['dep_detail']['mean']:>14.4f}"
            f"{r['dep_detail']['max']:>9.4f}{'PASS' if r['pass'] else 'FAIL':>9}")
    out.append("-" * 74)
    out.append(f"    扰动能量 max：语义块 {max(r['perm_energy_semantic']['max'] for r in rep.rows):.4f}"
               f" / 细节块 {max(r['perm_energy_detail']['max'] for r in rep.rows):.4f}"
               f"   ·   分支活性：语义 {rep.rows[0]['liveness_semantic']:.3f}"
               f" / 细节 {rep.rows[0]['liveness_detail']:.3f}")
    out.append(f"    总判定：{'✅ 通过' if rep.overall else '❌ 未通过'}")
    return "\n".join(out)


# ===========================================================================
# 4. 尺子对照（真图上也要有 oracle / leaky）
# ===========================================================================
def run_rulers(z: torch.Tensor, *, n_perm: int, seed: int,
               max_dep: float) -> Tuple[dict, RealReport, RealReport]:
    """在**真图 latent** 上跑 `run_g2` 用的同一对尺子。

    `separation._Oracle` / `_Leaky` 是既有文件里的私有尺子（`run_g2` 也用它们）；
    这里**只 import 不改动**，保证「合成上用的尺子」与「真图上用的尺子」是同一把。
    """
    rep_o = dependency_over_permutations(sep._Oracle(), z, n_perm=n_perm,
                                         seed=seed, max_dep=max_dep)
    rep_l = dependency_over_permutations(sep._Leaky(), z, n_perm=n_perm,
                                         seed=seed, max_dep=max_dep)
    return ({"oracle": rep_o.to_dict(), "leaky": rep_l.to_dict(),
             "oracle_pass": rep_o.overall, "leaky_fails": not rep_l.overall,
             "sanity_ok": bool(rep_o.overall and not rep_l.overall)},
            rep_o, rep_l)


# ===========================================================================
# 5. 训练
# ===========================================================================
class _Probe(nn.Module):
    """1×1 线性探针：`z → 下采样图像` / `z → 结构` 与 `z → 纹理残差`。

    ⛔ **不带非线性**：与 `separation._Mixer` 同理 —— 只有线性映射，
       「这一块通道能不能读出这个量」才是通道内容的问题，而不是映射算力的问题。
    """

    def __init__(self, cin: int = LATENT.total_ch, cout: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, z):
        return self.conv(z)


def semantic_texture_targets(imgs: torch.Tensor, side: int
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """图像 → (结构目标, 纹理残差目标)，都在 latent 分辨率上。

    · 结构：5×5 均值模糊后再下采样，再取亮度 ⇒ 低频「是什么/在哪」。
    · 纹理：`img − blur` 下采样 ⇒ 高频「长什么样」（色偏/纹理/描边）。
    ⚠️ 这是**代理目标**，不是 DINOv3 语义 —— 它只要求「结构 vs 纹理」这条粗分界，
       不是设计稿里 8ch 语义块的真身。报告里必须这样标注。
    """
    blur = F.avg_pool2d(F.pad(imgs, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    struct = F.adaptive_avg_pool2d(blur, (side, side)).mean(1, keepdim=True)
    tex = F.adaptive_avg_pool2d(imgs - blur, (side, side))
    return struct, tex


class _ProbeBranches:
    """把两个 1×1 探针包装成 G2 判据要的 `sem_path` / `det_path` 接口。"""

    def __init__(self, ps: nn.Module, pd: nn.Module):
        self.ps, self.pd = ps, pd

    def sem_path(self, z):
        return self.ps(z)

    def det_path(self, z):
        return self.pd(z)


def train_joint(imgs: torch.Tensor, *, steps: int = 200, lr: float = 5e-2,
                lr_vae: float = 1e-3, w_inv: float = 1.0, w_anchor: float = 1.0,
                base: int = 16, seed: int = 0,
                latent: Optional[torch.Tensor] = None
                ) -> Tuple[HybridVAE, nn.Module, dict]:
    """**joint 训练**：编码器 + 分支一起训；G2 主项逐字调用 `separation.recon_loss`。

    · G2 主项：`total, parts = sep.recon_loss(mixer, z, (None,None), w_inv=w_inv)`，
      其中 `z = vae.encode(img)` **每个 step 现算** ⇒ 梯度回传到编码器。
    · 锚项 `w_anchor * MSE(probe(z), pool(img))`：**不是** G2 判据的一部分，
      唯一作用是**阻止编码器把 latent 塌成常数**。塌了就**假过**
      （分母 `|Δ|+|branch(z)|` 同时趋 0 ⇒ 依赖度算成 `0/eps = 0`）。
      ⇒ 设 `w_anchor=0` 的 Arm D 是**塌缩体检**：它报出「塌/没塌」，
      不预设答案（实测在真图上没塌，但护栏必须留着）。

    ⚠️ 损失定义在 **latent 上**（扰动 = 打乱 latent 的通道块）。
      若误把图像本身喂给 `recon_loss`，`cross_perturb` 会去打乱**像素**，
      那是「整张图换人」，`inv` 会被常函数平凡满足 ⇒ 判据失效。
    ⚠️ 编码器用**更小的学习率**（`lr_vae`）：mixer 是从零学的小网络，
      编码器是预置结构，5e-2 会直接把 GroupNorm 打飞。
    """
    torch.manual_seed(seed)
    vae = HybridVAE(base=base)
    mixer = sep.SeparationModel()
    probe = _Probe()
    if latent is None:
        groups = [{"params": list(mixer.parameters()) + list(probe.parameters()),
                   "lr": lr},
                  {"params": list(vae.parameters()), "lr": lr_vae}]
    else:                                     # 编码器冻结（Arm A）
        for p in vae.parameters():
            p.requires_grad_(False)
        groups = [{"params": list(mixer.parameters()) + list(probe.parameters()),
                   "lr": lr}]
    opt = torch.optim.Adam(groups, lr=lr)

    last: dict = {}
    for _ in range(max(1, steps)):
        opt.zero_grad()
        z = latent if latent is not None else vae.encode(imgs)
        total, parts = sep.recon_loss(mixer, z, (None, None), w_inv=w_inv)
        tgt = F.adaptive_avg_pool2d(imgs, z.shape[-2:])
        a = F.mse_loss(probe(z), tgt)
        (total + w_anchor * a).backward()
        opt.step()
        last = dict(parts)
        last["anchor"] = float(a.detach())
    for p in vae.parameters():
        p.requires_grad_(True)
    return vae, mixer, last


def train_semantic_routed(imgs: torch.Tensor, *, steps: int = 200,
                          lr: float = 5e-2, lr_vae: float = 1e-3,
                          w_inv: float = 1.0, base: int = 16, seed: int = 0
                          ) -> Tuple[HybridVAE, _ProbeBranches, dict]:
    """**Arm E：结构/纹理路由监督**（本模块自带的增强实验，不是 G2 的原判据）。

    动机：G2 原损失的目标是「重建自己那一块 latent」，线性 mixer 只要学会
    **按通道选取**就能拿到近零损失 ⇒ 它对**编码器把什么放进哪一块**几乎没有压力。
    要问「真图能不能分离」，得给两块**不同含义的目标**：

        probe_sem(z) → 结构目标   （只准靠语义块）
        probe_det(z) → 纹理残差目标（只准靠细节块）
        + 与 G2 完全同款的交叉不变性项

    两个探针吃的是**全部 40ch**（否则「分离」就成了循环论证），
    由 `w_inv` 那一项把它们按回各读各的 ⇒ 判据仍是**同一套**。

    ⚠️ 结构/纹理目标是**代理**，不是 DINOv3 语义 ⇒ 结论口径是
       「结构 vs 纹理这条粗分界能不能被逼开」，不是「语义块真的懂角色」。
    """
    torch.manual_seed(seed)
    vae = HybridVAE(base=base)
    ps, pd = _Probe(cout=1), _Probe(cout=3)
    branches = _ProbeBranches(ps, pd)
    opt = torch.optim.Adam(
        [{"params": list(ps.parameters()) + list(pd.parameters()), "lr": lr},
         {"params": list(vae.parameters()), "lr": lr_vae}], lr=lr)

    last: dict = {}
    for _ in range(max(1, steps)):
        opt.zero_grad()
        z = vae.encode(imgs)
        side = z.shape[-1]
        struct, tex = semantic_texture_targets(imgs, side)
        s_pred, d_pred = branches.sem_path(z), branches.det_path(z)
        l_sem = F.mse_loss(s_pred, struct)
        l_det = F.mse_loss(d_pred, tex)
        inv = (F.mse_loss(branches.sem_path(
                    sep.cross_perturb(z, "shuffle_detail")), s_pred.detach())
               + F.mse_loss(branches.det_path(
                    sep.cross_perturb(z, "shuffle_semantic")), d_pred.detach()))
        total = l_sem + l_det + w_inv * inv
        total.backward()
        opt.step()
        last = {"sem": float(l_sem.detach()), "det": float(l_det.detach()),
                "inv": float(inv.detach()),
                "sem_r2": _r2(z, struct),
                "tex_r2": _r2(z, tex)}
    return vae, branches, last


# ===========================================================================
# 6. 端到端
# ===========================================================================
def run_real_g2(*, image_dir: Optional[str] = None, size: int = 384,
                background: str = "white", bg_sensitivity: bool = True,
                views_per_source: int = 2, max_views: int = 24,
                n_perm: int = 8, steps: int = 200,
                lr_vae: float = 1e-3, w_inv: float = 1.0, w_anchor: float = 1.0,
                base: int = 16, seed: int = 0, max_dep: float = 0.20,
                anchor_ablation: bool = True, frozen_arm: bool = True,
                semantic_arm: bool = True, synth_control: bool = True,
                synth_steps: int = 400, verbose: bool = False) -> dict:
    """真图版 G2 全流程：装载 → 编码 → 尺子 → 多臂对照 → 聚合判据。

    六个臂：
      · `rulers`            oracle（必过）/ leaky（必挂），在**真图 latent** 上；
      · `frozen_random_encoder` 随机初始化编码器**冻结**（等价于把
        `separation.train_separation` 的输入换成真图 latent）。几乎必然 PASS，
        但**信息量为零**：任何线性可分的随机投影都能被 1×1 mixer 分开；
      · `joint_supervised`  **主实验**：编码器与分支一起训，带信息锚；
      · `joint_negative`    同上但 `w_inv=0`（关掉交叉不变性）⇒ 监督的净收益；
      · `joint_no_anchor`   `w_anchor=0` 的**塌缩体检**：检验去掉信息锚后
        latent 是否塌成常数（塌了依赖度就会**假过**）。⚠️ 这是**检验**不是
        **断言** —— 实测在真图上并没有塌，报告里如实写实际结果；
      · `semantic_routed`   Arm E：结构/纹理代理目标 + 同一套不变性项。
    """
    gaps: List[str] = []

    views = load_real_views(image_dir, size=size, background=background,
                            views_per_source=views_per_source,
                            max_views=max_views)
    gaps += views.gaps()
    if verbose:
        print(f"  视图：{views.n_sources} 张来源 / {views.n_stems} 个独立图像族"
              f" → {views.n_views} 个视图（{size}²，{background} 底）")
        print(f"  alpha 口径：{views.alpha_policy}")
        print(f"  语料：{' + '.join(views.dirs) or '(默认)'}")
        for g in views.gaps():
            print(f"  ⚠️ 缺口：{g}")
        print()

    vae0, bundle = encode_views(views, base=base, seed=seed)
    if bundle.stats["collapsed"]:
        gaps.append("随机初始化编码器输出近似常数（未训编码器的预期行为）")

    def one(tag: str, rep: RealReport, extra: dict) -> dict:
        d = {"tag": tag, "report": rep.to_dict(), **extra}
        d["worst_dep"] = max(r["worst_dep"] for r in rep.rows)
        d["verdict"] = "PASS" if rep.overall else "FAIL"
        if verbose:
            print(format_report(rep, tag))
            print()
        return d

    rulers_dict, rep_o, rep_l = run_rulers(bundle.z, n_perm=n_perm, seed=seed,
                                           max_dep=max_dep)
    if verbose:
        print(format_report(rep_o, "尺子·oracle（构造上分离，必须 PASS）"))
        print()
        print(format_report(rep_l, "尺子·leaky（语义分支读细节块，必须 FAIL）"))
        print()
    if not rulers_dict["sanity_ok"]:
        gaps.append("尺子自检失败（oracle 未过 或 leaky 未挂）⇒ 真图上的数字不可信")

    arms: Dict[str, dict] = {}

    # ---- Arm A：冻结的随机编码器 ----
    if frozen_arm:
        _v, mixer_a, loss_a = train_joint(views.images, steps=steps, w_inv=w_inv,
                                          w_anchor=w_anchor, base=base, seed=seed,
                                          latent=bundle.z)
        rep_a = dependency_over_permutations(mixer_a, bundle.z, n_perm=n_perm,
                                             seed=seed, max_dep=max_dep)
        st_a = bundle.stats | {"probe_r2": latent_probe_r2(vae0, bundle.z, views.images)}
        gaps += latent_gaps(st_a, "Arm A")
        arms["frozen_random_encoder"] = one(
            "Arm A · 冻结随机编码器（无信息量，仅作对照）", rep_a,
            {"note": "编码器随机初始化且冻结 ⇒ latent 无图像语义；此臂只说明"
                     "「随机投影当然可分」，不能当作真图结论",
             "latent": st_a,
             "train_loss": loss_a})

    # ---- Arm B：joint 训练（主实验） ----
    vae_b, mixer_b, loss_b = train_joint(views.images, steps=steps, w_inv=w_inv,
                                         w_anchor=w_anchor, base=base, seed=seed,
                                         lr_vae=lr_vae)
    with torch.no_grad():
        z_b = vae_b.encode(views.images).detach()
    st_b = latent_stats(z_b) | {"probe_r2": latent_probe_r2(vae_b, z_b, views.images)}
    gaps += latent_gaps(st_b, "Arm B")
    rep_b = dependency_over_permutations(mixer_b, z_b, n_perm=n_perm,
                                         seed=seed, max_dep=max_dep)
    arms["joint_supervised"] = one(
        f"Arm B · joint 训练 w_inv={w_inv} w_anchor={w_anchor}（主实验）", rep_b,
        {"latent": st_b, "train_loss": loss_b})

    # ---- Arm C：负对照 w_inv=0 ----
    vae_c, mixer_c, loss_c = train_joint(views.images, steps=steps, w_inv=0.0,
                                         w_anchor=w_anchor, base=base, seed=seed,
                                         lr_vae=lr_vae)
    with torch.no_grad():
        z_c = vae_c.encode(views.images).detach()
    rep_c = dependency_over_permutations(mixer_c, z_c, n_perm=n_perm,
                                         seed=seed, max_dep=max_dep)
    arms["joint_negative"] = one(
        "Arm C · joint 负对照 w_inv=0（关掉交叉不变性）", rep_c,
        {"latent": latent_stats(z_c) | {"probe_r2": latent_probe_r2(vae_c, z_c, views.images)},
         "train_loss": loss_c})
    gaps += latent_gaps(arms["joint_negative"]["latent"], "Arm C")

    # ---- Arm D：塌缩体检 w_anchor=0 ----
    if anchor_ablation:
        vae_d, mixer_d, loss_d = train_joint(views.images, steps=steps, w_inv=w_inv,
                                             w_anchor=0.0, base=base, seed=seed,
                                             lr_vae=lr_vae)
        with torch.no_grad():
            z_d = vae_d.encode(views.images).detach()
        st_d = latent_stats(z_d) | {"probe_r2": latent_probe_r2(vae_d, z_d, views.images)}
        rep_d = dependency_over_permutations(mixer_d, z_d, n_perm=n_perm,
                                             seed=seed, max_dep=max_dep)
        gaps += latent_gaps(st_d, "Arm D")
        arms["joint_no_anchor"] = one(
            "Arm D · 塌缩体检 w_anchor=0（去掉信息锚，看会不会假过）", rep_d,
            {"latent": st_d, "train_loss": loss_d})

    # ---- Arm E：结构/纹理路由监督 ----
    if semantic_arm:
        vae_e, br_e, loss_e = train_semantic_routed(
            views.images, steps=steps, w_inv=w_inv, base=base, seed=seed,
            lr_vae=lr_vae)
        with torch.no_grad():
            z_e = vae_e.encode(views.images).detach()
        rep_e = dependency_over_permutations(br_e, z_e, n_perm=n_perm,
                                             seed=seed, max_dep=max_dep)
        st_e = latent_stats(z_e) | {"probe_r2": latent_probe_r2(vae_e, z_e, views.images)}
        gaps += latent_gaps(st_e, "Arm E")
        arms["semantic_routed"] = one(
            "Arm E · 结构/纹理代理监督（真图上「强制路由」的直接检验）", rep_e,
            {"note": "目标是结构/纹理代理，**不是** DINOv3 语义；"
                     "结论口径限于「结构 vs 纹理这条粗分界」",
             "latent": st_e,
             "train_loss": loss_e})
        # ⚠️ Arm E 的两个探针学的是**不同**目标，活性尺度不可比：
        #    纹理残差目标本身能量很小，探针输出小不代表它没用。
        #    真正要看的是它相对**自己的目标**的 R²（sem_r2 / tex_r2）。
        #    守卫用**相对**判据（弱侧 < 强侧的一半 ⇒ 弱侧的数字不可解释），
        #    不用拍脑袋的绝对阈值 —— 否则就是拿阈值去迁就结果。
        sr2 = float(loss_e.get("sem_r2", 0.0))
        tr2 = float(loss_e.get("tex_r2", 0.0))
        if sr2 > 0 and tr2 < 0.5 * sr2:
            gaps.append(f"[Arm E] 两个探针对**各自目标**的线性可读出 R² 相差过大"
                        f"（语义 {sr2:.3f} vs 纹理 {tr2:.3f}）"
                        f" ⇒ 弱侧（纹理）基本没学到东西，"
                        f"该侧的依赖度数字**不可解释**，不能算作「细节侧分不开」")
        elif sr2 < 0.05 or tr2 < 0.05:
            gaps.append(f"[Arm E] 探针 R² 过低（语义 {sr2:.3f} / 纹理 {tr2:.3f}）"
                        f" ⇒ 该臂的依赖度数字不可解释")

    # ---- 背景敏感性 ----
    bg_arm = None
    if bg_sensitivity and background in BG_CHOICES:
        other = "black" if background == "white" else "white"
        v2 = load_real_views(image_dir, size=size, background=other,
                             views_per_source=views_per_source, max_views=max_views)
        v2_vae, v2_mixer, _ = train_joint(v2.images, steps=steps, w_inv=w_inv,
                                          w_anchor=w_anchor, base=base, seed=seed,
                                          lr_vae=lr_vae)
        with torch.no_grad():
            z2 = v2_vae.encode(v2.images).detach()
        rep2 = dependency_over_permutations(v2_mixer, z2, n_perm=n_perm,
                                            seed=seed, max_dep=max_dep)
        bg_arm = {"background": other, "report": rep2.to_dict(),
                  "verdict": "PASS" if rep2.overall else "FAIL",
                  "latent": latent_stats(z2)}
        if verbose:
            print(format_report(rep2, f"背景敏感性 · {other} 底（同一主实验）"))
            print()

    # ---- 合成对照：完整跑一遍既有 run_g2，确认装置本体没坏 ----
    synth = None
    if synth_control:
        synth = sep.run_g2(n=32, side=32, steps=synth_steps, seed=seed,
                           w_inv=w_inv, max_dep=max_dep, mix=0.6, mix_baseline=0.0)
        if not synth["sanity_ok"]:
            gaps.append("合成对照的尺子自检失败 ⇒ 装置本体有问题")

    gap = {
        "worst_dep_supervised": arms["joint_supervised"]["worst_dep"],
        "worst_dep_negative": arms["joint_negative"]["worst_dep"],
        "improvement_x": (arms["joint_negative"]["worst_dep"]
                          / max(arms["joint_supervised"]["worst_dep"], 1e-12)),
    }

    return {
        "config": {"image_dir": (list(image_dir) if isinstance(image_dir, (list, tuple))
                                 else str(image_dir) if image_dir else
                                 "默认：data/characters/kokona/images + out/e5b/g1/bf16"),
                   "size": size, "background": background,
                   "bg_sensitivity": bg_sensitivity,
                   "views_per_source": views_per_source, "max_views": max_views,
                   "n_perm": n_perm, "steps": steps, "lr_vae": lr_vae,
                   "w_inv": w_inv, "w_anchor": w_anchor, "base": base,
                   "seed": seed, "max_dep": max_dep, "synth_steps": synth_steps,
                   "gate_policy": "沿用 G2 门线 0.20，**未做任何放松**；"
                                  "判定取 n_perm 个独立置换里的 max（最坏置换）"},
        "data": views.to_dict(),
        "random_encoder_latent": bundle.to_dict(),
        "rulers": rulers_dict,
        "arms": arms,
        "background_sensitivity": bg_arm,
        "synthetic_control": synth,
        "gap": gap,
        "gaps": gaps,
        "verdict": arms["joint_supervised"]["verdict"],
        "sanity_ok": rulers_dict["sanity_ok"],
        "conclusion_strength": ("INSUFFICIENT_SAMPLE"
                                if views.n_stems < MIN_SOURCES else "OK"),
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    r = run_real_g2(verbose=True)
    print(f"\n判定：{r['verdict']}   结论强度：{r['conclusion_strength']}")
    for g in r["gaps"]:
        print(f"  ⚠️ {g}")

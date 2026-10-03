"""真探针 —— 把 G3.5 的四测接到**真实主干 + 真实轴注入路径**上。

## 和 `kp/probe/axis.py` 的关系

`axis.py` 是**尺子**（装置），`synthetic.py` 是**已知答案的对照样本**。
本模块是**真探针**：responder 不再是合成函数，而是

    domain(16 维条件向量) → domain_embed → adaLN(mod) → DiT blocks → out_proj → 响应

即主文档 §4.8.7⑤ 说的「**16 维、逐维命名的已标定控制面板**」那条真实路径。

## ⚠️ 一个必须先讲清的前提（本次实测发现，不是推测）

`kp/models/dit.py` 是 **adaLN-Zero** 初始化的：`adaLN[-1].weight ≡ 0`，
只把 attn/mlp/txt 三个 gate 的 **bias** 置 1。于是初值时

    mod = adaLN(c) ≡ bias = 常数   ⇒   **domain 与 t 都进不去输出**

实测（`tests`/本模块 `domain_response_dev`）：`domain=0` / `domain=随机` /
`domain=None` 三者输出 **bit-exact 相同**，连 `t` 也不影响输出。

⇒ **在未训练的主干上直接跑 G3.5，测到的是「响应恒为零」，
这不是「轴不成立」，而是「门还没开」。** 本模块因此显式提供
`open_domain_door()`：把 `adaLN[-1].weight`（初值恒 0）替换为**固定种子的噪声**，
这是训练第一步本来就会做的事，且是**唯一**的改动（bias、结构、其余权重都不动）。

⚠️ **即便如此，结论也必须分层读**：本机没有训练好的 checkpoint，
随机权重下每个轴的响应方向是**任意**的，所以
  · 「四测数字」= **这条注入路径的几何/管道性质**（可复现、可 CPU 复跑）；
  · 「轴语义」（写实↔二次元 真的成立吗）= **训后复测才能判**。
报告里两者必须分开写，不能拿管道层的通过率冒充语义层的结论。

## 装置自身的一个盲点（本次实测发现，已在真探针里补守卫）

`axis.py` 的 ① 单调性用的是 Spearman ρ。当**响应恒为常数**（轴完全无作用）时，
`proj ≡ 0` ⇒ 秩相关退化为 **ρ = 1.00**；② 里零方向两两余弦 0 ⇒ 过；
③ 里 `dp+dm=0` 且分母为 0 被 `continue` 跳过 ⇒ 1.00 过。
⇒ **一条完全失效的轴会拿到「3/4 通过」，只挂 ④。**

真探针因此在子类里加**非空性守卫**（`inert`）：响应在整条扫描上**数值为零**时，
① 直接判 0 并标 `inert=True`（注意判据是「零」而不是「弱」——
「弱」是 ④ 低比特行程的职责，不能用 ① 重复抓，否则 ④ 的不可替代性就没了）。
`axis.py` 本身**未改动**（它是已通过自检的既有件），缺陷以报告形式提交。

## 四测在真主干上的口径

| # | 合成装置口径 | **真探针口径（本模块）** |
|---|---|---|
| ① 单调 | Spearman ρ（跨上下文取最小） | 同左 **+ 非空性守卫** |
| ② 正交 | 跨轴方向余弦 max \\|cos\\| | 同左（方向来自真主干响应） |
| ③ 可逆 | 奇对称度 `1 − max err` | 同左 |
| ④ ⭐ 低比特行程 | `quant_fp4(读出)` 后相邻档是否可分辨 | **把 `kp/quant/nvfp4.py` 的量化器接进主干本体**：所有 `GatedLinear` 挂 `DESIGN_SPEC`（W4A8），同一条扫重跑一遍，取「严格递增档数(量化) ÷ 严格递增档数(bf16)」—— 这是设计稿 §2.3② 的原话「**FP4 下可分辨的轴步数 ÷ bf16 下的步数**」 |

④ 两种口径都报：**门的判据用「量化主干两遍法」**（更贴设计原话、更严格），
`quant_fp4(读出)` 的老口径作为辅助对照。

纯 CPU、固定种子、可复现；不碰 GPU。
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..config import AXIS, LATENT, DiTCfg
from ..models import SingleStreamDiT
from ..quant.nvfp4 import DESIGN_SPEC, OFFICIAL_DEFAULT_SPEC, QuantSpec
from .axis import AxisProbe, AxisProbeReport, AxisResult

__all__ = [
    "DOMAIN_AXES", "DOMAIN_AXIS_NAMES", "DOMAIN_AXIS_DESC", "DOMAIN_AXIS_SRC",
    "readout_layout", "latent_readout", "latent_readout_blocks",
    "quantized_backbone", "BackboneDomainResponder", "make_responder",
    "open_domain_door", "corrupt_domain_embed",
    "RealAxisProbe", "RealProbeReport", "run_real_probe", "real_axis_report_text",
]


# ---------------------------------------------------------------------------
# ① 16 维条件轴：逐维命名（**每一条都有设计依据，不凭空发明**）
# ---------------------------------------------------------------------------
# 设计依据：
#   主§4.8.1/§4.8.4 「把写实度变成一根旋钮」+ shader 配对可覆盖的风格轴
#                    （写实 PBR / 赛璐璐+描边 / 平涂 / 水彩 / 厚涂 / 线稿）
#   补03§2          L1 解决什么：风格、写实度、**光照**、**色调**、笔触
#   补03§3          SAC-LDM 的**解耦风格编辑方向**：描边、局部对比、水彩化、几何图案
#   补08§2.3⑤      显式给出的轴名：写实度 / 描边强度 / 平涂度 / 冷暖 / 对比度 / 笔触粗度
#   补08§2.2 风险4  正交性必须守住的三个量：构图、**光照方向**、**色彩饱和**、**色相漂移**
#   主§4.8.5        通道分离：语义 8ch 管结构、细节 32ch 管画风 ⇒ 细节通道负载
DOMAIN_AXES: Tuple[Tuple[str, str, str], ...] = (
    ("realism",         "写实度（PBR ↔ 二次元）",  "主§4.8.1/§4.8.4"),
    ("cel_shading",     "赛璐璐着色度",            "主§4.8.4"),
    ("outline",         "描边强度",                "主§4.8.4·补03§3"),
    ("flat_fill",       "平涂度",                  "主§4.8.4"),
    ("watercolor",      "水彩化",                  "主§4.8.4·补03§3"),
    ("impasto",         "厚涂度",                  "主§4.8.4"),
    ("lineart",         "线稿度",                  "主§4.8.4"),
    ("brush_size",      "笔触粗度",                "补08§2.3⑤"),
    ("local_contrast",  "局部对比",                "补03§3(SAC-LDM)"),
    ("global_contrast", "全局对比度",              "补08§2.3⑤"),
    ("brightness",      "明度",                    "补03§2(L1 光照)"),
    ("color_temp",      "冷暖（色温）",            "补08§2.3⑤·风险4"),
    ("saturation",      "饱和度",                  "补08§2.2 风险4"),
    ("hue_shift",       "色相漂移",                "补08§2.2 风险4"),
    ("light_direction", "光照方向",                "补08§2.2 风险4"),
    ("detail_density",  "细节密度（细节 32ch 负载）", "主§4.8.5"),
)

DOMAIN_AXIS_NAMES: List[str] = [n for n, _, _ in DOMAIN_AXES]
DOMAIN_AXIS_DESC: Dict[str, str] = {n: d for n, d, _ in DOMAIN_AXES}
DOMAIN_AXIS_SRC: Dict[str, str] = {n: s for n, _, s in DOMAIN_AXES}


# ---------------------------------------------------------------------------
# ② 响应读出：主干输出的**确定性向量读出**（分块 → 可解释）
# ---------------------------------------------------------------------------
# 设计稿对 ② 正交性的判据是「构图 / 光照（平均亮度·色温）/ 色相漂移」。
# 未训练的主干不能出图，所以这里取**latent 场的结构性代理**，逐块与那几项对应：
#   amp          全局幅度          ← 亮度 / 曝光代理
#   sem_mean     语义 8ch 均值      ← 结构通道（§4.8.5）
#   det_mean     细节 32ch 均值     ← 画风通道（§4.8.5）
#   layout       2×2 象限均值       ← 构图 / bbox 布局差代理
#   grad         空间梯度能量       ← 结构密度 / 描边强度代理
#   hf           离散拉普拉斯能量    ← 高频细节 / 笔触密度代理
_BLOCK_SIZES = ("amp", "sem_mean", "det_mean", "layout", "grad", "hf")


def readout_layout(sem_ch: Optional[int] = None, det_ch: Optional[int] = None,
                   grid: int = 2) -> Dict[str, slice]:
    """各读出块在响应向量里的切片（顺序与 `latent_readout` 一致）。"""
    sem_ch = LATENT.semantic_ch if sem_ch is None else sem_ch
    det_ch = LATENT.detail_ch if det_ch is None else det_ch
    sizes = {"amp": 1, "sem_mean": sem_ch, "det_mean": det_ch,
             "layout": grid * grid, "grad": 2, "hf": 1}
    out, i = {}, 0
    for name in _BLOCK_SIZES:
        out[name] = slice(i, i + sizes[name])
        i += sizes[name]
    return out


def latent_readout_blocks(v: torch.Tensor, sem_ch: Optional[int] = None,
                          grid: int = 2) -> Dict[str, torch.Tensor]:
    """主干输出 `(B, C, H, W)` → 具名块字典（每块 `(B, n)`）。"""
    assert v.dim() == 4, f"期望 (B,C,H,W)，得到 {tuple(v.shape)}"
    B, C, H, W = v.shape
    sem_ch = LATENT.semantic_ch if sem_ch is None else sem_ch
    if H < 3 or W < 3:
        raise ValueError(f"读出需要 H,W ≥ 3（拉普拉斯），得到 {H}×{W}")
    b: Dict[str, torch.Tensor] = {}
    b["amp"] = v.abs().mean(dim=(1, 2, 3)).unsqueeze(1)
    b["sem_mean"] = v[:, :sem_ch].mean(dim=(2, 3))
    b["det_mean"] = v[:, sem_ch:].mean(dim=(2, 3))
    hs = [int(round(H * i / grid)) for i in range(grid + 1)]
    ws = [int(round(W * j / grid)) for j in range(grid + 1)]
    quads = [v[:, :, hs[i]:hs[i + 1], ws[j]:ws[j + 1]].mean(dim=(1, 2, 3))
             for i in range(grid) for j in range(grid)]
    b["layout"] = torch.stack(quads, dim=1)
    dh = (v[:, :, 1:, :] - v[:, :, :-1, :]).abs().mean(dim=(1, 2, 3)).unsqueeze(1)
    dw = (v[:, :, :, 1:] - v[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)).unsqueeze(1)
    b["grad"] = torch.cat([dh, dw], dim=1)
    lap = (4.0 * v[:, :, 1:-1, 1:-1] - v[:, :, :-2, 1:-1] - v[:, :, 2:, 1:-1]
           - v[:, :, 1:-1, :-2] - v[:, :, 1:-1, 2:])
    b["hf"] = lap.abs().mean(dim=(1, 2, 3)).unsqueeze(1)
    return b


def latent_readout(v: torch.Tensor, sem_ch: Optional[int] = None,
                   det_ch: Optional[int] = None, grid: int = 2) -> torch.Tensor:
    """把具名块按固定顺序拼成响应向量 `(B, D)`（D = 1+8+32+4+2+1 = 48）。"""
    b = latent_readout_blocks(v, sem_ch, grid)
    return torch.cat([b[k] for k in _BLOCK_SIZES], dim=1)


def readout_dim(sem_ch: Optional[int] = None, det_ch: Optional[int] = None,
                grid: int = 2) -> int:
    lay = readout_layout(sem_ch, det_ch, grid)
    return lay["hf"].stop


# ⭐ ② 正交性判据的**分辨力底线**：n 个随机方向在 D 维空间里的最大 |cos|
def orthogonality_null_baseline(n_axes: int, dim: int, trials: int = 200,
                                seed: int = 0) -> Dict[str, float]:
    """随机方向基线下 ② 会读到什么 —— 判据「能不能分辨」的前提。

    ⚠️ 这是本装置在真主干的**读出维选择**上的一个硬约束：
       16 个轴的方向都住在同一个读出空间里，随机方向的 |cos| 期望约 `1/√D`，
       取 16 个里 120 对的最大值，实测（见返回的 `max`）在 D=48 时已达 ~0.5。
       ⇒ **门线 |cos| ≤ 0.30 在「16 轴 + 48 维读出」下不可达**，
         除非网络真的把轴解耦（那是训练目标，不是随机初始化的性质）。
       所以报告里必须同时给出这条基线：测到的值 ≈ 基线 ⇒ ② 无分辨力。
    """
    g = torch.Generator().manual_seed(int(seed) + 31337)
    worst = []
    for _ in range(int(trials)):
        A = torch.randn(int(dim), int(n_axes), generator=g)
        A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
        C = (A.T @ A).abs()
        C.fill_diagonal_(0.0)
        worst.append(float(C.max()))
    t = torch.tensor(worst)
    return {"dim": int(dim), "n_axes": int(n_axes), "trials": int(trials),
            "mean_max_cos": float(t.mean()), "max_max_cos": float(t.max()),
            "min_max_cos": float(t.min())}


# ---------------------------------------------------------------------------
# ③ 量化回路（⭐ ④ 测的核心：把 kp/quant/nvfp4.py 真的接进主干）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def quantized_backbone(model: SingleStreamDiT, spec: Optional[QuantSpec]):
    """临时给主干**所有** `GatedLinear` 挂上量化规格，退出时精确还原。

    ⚠️ 只动 `GatedLinear.quant`（主干算子）：`y = F.linear(q(x), q(W)) + Σ pack(x)`。
       设计稿「注入点必须在量化器之外」说的是**能力包**；本模块没有挂包，
       探的是「轴信号穿过量化激活后还剩多少行程」，正是补充08 §2.2 风险 3(b)。
    """
    layers = list(model.gated_linears().values())
    old = [L.quant for L in layers]
    if spec is not None:
        for L in layers:
            L.set_quant(spec)
    try:
        yield model
    finally:
        for L, o in zip(layers, old):
            L.set_quant(o)


# ---------------------------------------------------------------------------
# ④ 真主干 responder
# ---------------------------------------------------------------------------
class BackboneDomainResponder:
    """`V (B, 16) → 主干输出读出 (B, D)`。

    除 `domain`（= 被测轴值）外的**一切输入都是固定的**（x / t / text_ctx 同种子生成
    且在所有求值中共享），所以响应差异只可能来自轴值本身 —— 这是「可复现」的前提。
    """

    def __init__(self, model: SingleStreamDiT, *, x: torch.Tensor, t: torch.Tensor,
                 text: Optional[torch.Tensor], quant: Optional[QuantSpec] = None,
                 sem_ch: Optional[int] = None, det_ch: Optional[int] = None,
                 grid: int = 2, readout: str = "structured"):
        self.model = model
        self.x, self.t, self.text = x, t, text
        self.quant = quant
        self.sem_ch = LATENT.semantic_ch if sem_ch is None else sem_ch
        self._det_ch = det_ch
        self.grid = grid
        if readout not in ("structured", "field"):
            raise ValueError(f"未知读出 {readout!r}（structured | field）")
        self.readout = readout
        self.calls = 0

    def with_quant(self, spec: Optional[QuantSpec]) -> "BackboneDomainResponder":
        """派生一个**共享同一批固定输入**、只换量化规格的孪生 responder（④ 用）。"""
        return BackboneDomainResponder(self.model, x=self.x, t=self.t, text=self.text,
                                       quant=spec, sem_ch=self.sem_ch,
                                       det_ch=self._det_ch, grid=self.grid,
                                       readout=self.readout)

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        B = int(V.shape[0])
        self.calls += 1
        with torch.no_grad():
            with quantized_backbone(self.model, self.quant):
                out = self.model(
                    self.x.expand(B, -1, -1, -1),
                    self.t.expand(B),
                    text_ctx=None if self.text is None else self.text.expand(B, -1, -1),
                    domain=V,
                )
        if self.readout == "field":                  # 高维读出（分辨力强）
            return out.reshape(out.shape[0], -1)
        return latent_readout(out, self.sem_ch, self.det_ch, self.grid)

    @property
    def det_ch(self) -> int:
        """细节通道数（显式给定，否则 = 总通道 − 语义通道，保证切片与真张量一致）。"""
        if self._det_ch is not None:
            return int(self._det_ch)
        return int(self.x.shape[1]) - int(self.sem_ch)

    def layout(self) -> Dict[str, slice]:
        return readout_layout(self.sem_ch, self.det_ch, self.grid)

    def describe(self) -> dict:
        C, H, W = self.x.shape[1], self.x.shape[2], self.x.shape[3]
        rd = (C * H * W if self.readout == "field"
              else readout_dim(self.sem_ch, self.det_ch, self.grid))
        return {"tokens": H * W, "latent": f"{C}x{H}x{W}",
                "text_len": 0 if self.text is None else int(self.text.shape[1]),
                "quant": None if self.quant is None else f"{self.quant.weight}/{self.quant.act}",
                "readout": self.readout, "readout_dim": rd}


def make_responder(model: SingleStreamDiT, *, tokens: int = 8, use_text: bool = True,
                   seed: int = 0, quant: Optional[QuantSpec] = None,
                   latent_ch: Optional[int] = None, text_len: int = 12,
                   readout: str = "structured") -> BackboneDomainResponder:
    """构造真 responder（固定输入由 `seed` 决定 ⇒ 同种子逐位可复现）。"""
    C = latent_ch or model.latent_ch
    g = torch.Generator().manual_seed(int(seed) + 20261003)
    x = torch.randn(1, C, tokens, tokens, generator=g)
    t = torch.full((1,), 500.0)
    text = (torch.randn(1, text_len, model.text_dim, generator=g)
            if use_text else None)
    return BackboneDomainResponder(model, x=x, t=t, text=text, quant=quant,
                                   det_ch=C - LATENT.semantic_ch, readout=readout)


# ---------------------------------------------------------------------------
# ⑤ 「开门」：adaLN-Zero ⇒ 域向量在初值时完全进不去
# ---------------------------------------------------------------------------
def open_domain_door(model: SingleStreamDiT, *, scale: float = 0.02,
                     seed: int = 0) -> Dict[str, object]:
    """显式开门：把每个 block `adaLN[-1].weight`（初值**恒为 0**）换成固定种子噪声。

    ⚠️ 这是**唯一**的改动：
      · 不动 `bias`（attn/mlp/txt 三个 gate 的 1.0 保留、身份门控第 9 段仍为 0）；
      · 不动结构、不动任何 `GatedLinear` 权重、不动 `domain_embed`。
    语义 = 「训练走完第一步之后，门就不再是死的」—— 让**注入路径**可被观测。

    返回审计信息（改了几个参数、门控取值范围），便于报告里如实披露。
    """
    g = torch.Generator().manual_seed(int(seed) + 777)
    n = 0
    for blk in model.blocks:
        last = blk.adaLN[-1]
        with torch.no_grad():
            last.weight.copy_(torch.randn(last.weight.shape, generator=g) * float(scale))
        n += int(last.weight.numel())
    return {"door_opened": True, "scale": float(scale), "seed": int(seed),
            "n_params_touched": n}


def gate_ranges(model: SingleStreamDiT, responder: BackboneDomainResponder,
                n_probe: int = 9) -> Dict[str, float]:
    """开门后 attn/mlp/txt 三个 gate 的取值范围（确认**没有翻符号**）。

    门控是 `mod[:, seg]`（seg = 2/5/6）。设计要求 attn/mlp 通路不被关死；
    若噪声把 gate 打到负数，那已经不是在「开门」而是在破坏主干，必须报出来。
    """
    d = model.cfg.dim
    segs = (2, 5, 6)
    g = torch.Generator().manual_seed(4242)
    V = (torch.rand(n_probe, len(DOMAIN_AXIS_NAMES), generator=g) * 2 - 1)
    with torch.no_grad():
        c = model.t_embed(responder.t.expand(n_probe)) + model.domain_embed(V)
        vals: List[torch.Tensor] = []
        for blk in model.blocks:
            mod = blk.adaLN(c)
            for s in segs:
                vals.append(mod[:, s * d:(s + 1) * d].reshape(-1))
    allv = torch.cat(vals)
    return {"gate_min": float(allv.min()), "gate_max": float(allv.max()),
            "gate_mean": float(allv.mean())}


def noise_floor(responder: BackboneDomainResponder, batch: int = 9,
                n_axes: Optional[int] = None) -> float:
    """**同批次行间数值噪声底线** —— 非空性判据的分母。

    ⚠️ 实测根因（不是猜的）：CPU 上的 oneDNN GEMM **不是逐行可复现的**。
       把同一批次的若干行喂成**完全相同**的输入，输出行之间仍有 ~1e-8 的差异
       （定位过程：`patch_embed`/`adaLN` 逐位相同 → `_attn_core` 相同 →
       `out` 那层 `F.linear` 在 M=612 时出现行间差异；M=17 时不出现）。
       ⇒ 「这个轴有没有作用」**不能**用「diff == 0」判，必须用
         「diff 是否显著高于同一批次、同一形状下测出来的噪声底线」判。
       否则门关（adaLN-Zero）时的 6e-8 假信号会被当成"轴有响应"。
    """
    n_axes = len(DOMAIN_AXIS_NAMES) if n_axes is None else n_axes
    V = torch.zeros(int(batch), n_axes)
    with torch.no_grad():
        F = responder(V)
    return float((F - F[:1]).abs().max())


def domain_response_dev(model: SingleStreamDiT,
                        responder: BackboneDomainResponder,
                        axes: Optional[Sequence[int]] = None
                        ) -> Tuple[Dict[int, float], float]:
    """逐轴「单独激励 vs 零值」的响应偏差 + **同一批次测出的噪声底线**。

    返回值 `(devs, floor)`：`dev == 0` 的严格判据在 CPU 上不成立（见
    `noise_floor` 的实测根因），所以调用方应当用 `dev > k·floor` 判「有响应」。

    ⚠️ 零值行与全部激励行放进**同一次调用**、并额外复制一份用于测底线，
       以消除 batch 大小带来的 GEMM 分块差异（最初版本踩过：跨 batch 相减
       会得到 ~1e-8 的假响应，把「门关 ⇒ 域向量失效」这条真结论盖掉）。
    """
    axes = list(axes) if axes is not None else list(range(len(DOMAIN_AXIS_NAMES)))
    n = len(DOMAIN_AXIS_NAMES)
    V = torch.zeros(len(axes) + 1, n)
    for k, a in enumerate(axes):
        V[k + 1, a] = 1.0
    with torch.no_grad():
        F = responder(torch.cat([V, V], dim=0))
    m = V.shape[0]
    F, Fd = F[:m], F[m:]
    floor = float((F - Fd).abs().max())
    F0 = F[0]
    devs = {a: float((F[k + 1] - F0).abs().max()) for k, a in enumerate(axes)}
    return devs, floor


def corrupt_domain_embed(model: SingleStreamDiT, *, kind: str, a: int = 2, b: int = 3,
                         factor: float = 0.02) -> Dict[str, object]:
    """**负对照**：故意在真注入路径上注入已知缺陷，看真探针抓不抓得住。

    · `kind="coupled"`    ：`W[:, b] ← W[:, a]` ⇒ 轴 a 与 b 完全共线 ⇒ ② 必挂。
    · `kind="suppressed"`：`W[:, a] *= factor` ⇒ 轴 a 的效果被压到量化台阶以下 ⇒ ④ 必挂。

    ⚠️ `factor` 不能取太小：实测 CPU 上同批次行间数值噪声底线约 1e-8，
       `factor=1e-3` 时响应偏差只剩 ~3e-7，会落到「既像无响应、又像弱响应」的
       灰区（非空性守卫与 ④ 会同时报警）⇒ 默认取 0.02（响应 ~6e-6，
       高于噪声底线两个数量级，**但仍远低于 W4A8 的量化台阶**）。

    ⚠️ 只动 `domain_embed`（`nn.Linear`，真参数），不动主干本体。
       返回「期望挂掉哪一测」，供自检与报告做**可证伪**对照。
    """
    W = model.domain_embed.weight
    with torch.no_grad():
        if kind == "coupled":
            W[:, b] = W[:, a]
            return {"kind": kind, "axes": [a, b], "expect_fail": ["正交"]}
        if kind == "suppressed":
            W[:, a] = W[:, a] * float(factor)
            return {"kind": kind, "axes": [a], "expect_fail": ["低比特行程"],
                    "factor": float(factor)}
    raise ValueError(f"未知负对照类型 {kind!r}")


# ---------------------------------------------------------------------------
# ⑥ 真探针
# ---------------------------------------------------------------------------
class RealAxisProbe(AxisProbe):
    """真探针：responder 是**真主干 + 真轴注入路径**，④ 走**真量化回路**。

    与 `AxisProbe` 的差异只有两处（都在子类里，`axis.py` 未改动）：
      1. ① 加**非空性守卫**（响应数值恒零 ⇒ 判 0，`aux["inert"]=True`）——
         否则装置会对一条完全失效的轴给出「ρ=1.00 单调通过」的假结论。
      2. ④ 换成**量化主干两遍法**（设计稿 §2.3② 的原话口径）。
    """

    def __init__(self, model: SingleStreamDiT, *, tokens: int = 8, use_text: bool = True,
                 seed: int = 0, quant_spec: Optional[QuantSpec] = DESIGN_SPEC,
                 names: Optional[Sequence[str]] = None,
                 n_axes: Optional[int] = None, sweep_steps: Optional[int] = None,
                 n_random: Optional[int] = None, latent_ch: Optional[int] = None,
                 readout: str = "structured"):
        n_axes = len(DOMAIN_AXIS_NAMES) if n_axes is None else n_axes
        names = DOMAIN_AXIS_NAMES[:n_axes] if names is None else names
        self.model = model
        self.quant_spec = quant_spec
        self.readout = readout
        self.base = make_responder(model, tokens=tokens, use_text=use_text, seed=seed,
                                   quant=None, latent_ch=latent_ch, readout=readout)
        self.qresp = self.base.with_quant(quant_spec)
        self.aux: Dict[int, Dict[str, object]] = {}
        self._inert: Dict[int, bool] = {}
        super().__init__(self.base, n_axes=n_axes, names=names,
                         sweep_steps=sweep_steps, n_random=n_random, seed=seed)

    # ---- 辅助读出（高维读出下线性读出不适定，如实标「不适用」）----
    def probe_identifiability(self, axis: int,
                              V: Optional[torch.Tensor] = None) -> float:
        g = torch.Generator().manual_seed(self.seed + 1)
        if V is None:
            V = torch.rand(self.n_random, self.n_axes, generator=g) * 2 - 1
        D = int(self._eval(V[:1]).shape[1])
        if D > max(16, V.shape[0] // 2):
            # 岭回归在 D ≫ n 时不适定（XᵀX 近奇异），此处**不报假数**，标 -1 = 不适用
            self._aux(axis)["ident_undefined"] = True
            return -1.0
        return super().probe_identifiability(axis, V)

    # ---- 工具 ----
    def _eval_r(self, f: Callable[[torch.Tensor], torch.Tensor],
                V: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = f(V.to(torch.float32))
        return out.reshape(out.shape[0], -1).to(torch.float32)

    def _aux(self, axis: int) -> Dict[str, object]:
        return self.aux.setdefault(axis, {})

    # ---- 非空性预检（**先于四测**，因为它决定 ① 是否成立）----
    def ensure_inert(self, axes: Optional[Sequence[int]] = None) -> Dict[int, bool]:
        """逐轴判「这个轴对输出到底有没有影响」，判据 = 响应显著高于**实测噪声底线**。

        必须在四测之前做：装置 ① 用 Spearman ρ，而响应恒定时 `proj ≡ 0`
        会让秩相关退化成 **ρ = 1.00**（复现过），②③ 也会**假通过**。
        """
        axs = [int(a) for a in (axes if axes is not None else range(self.n_axes))]
        if all(a in self._inert for a in axs):
            return {a: self._inert[a] for a in axs}
        # 噪声底线：同一批次形状下、**完全相同**的输入行之间的最大差（实测 ~1e-8）
        floor = noise_floor(self.f, batch=self.sweep_steps + 1, n_axes=self.n_axes)
        devs, _ = domain_response_dev(self.model, self.f, axs)
        thr = 16.0 * floor + 1e-12
        for a in axs:
            self._inert[a] = bool(devs[a] <= thr)
            self._aux(a).update(resp_dev=devs[a], noise_floor=floor,
                                inert_threshold=thr, inert=self._inert[a])
        return {a: self._inert[a] for a in axs}

    # ---- ① 单调性（+ 非空性守卫）----
    def probe_monotonicity(self, axis: int, contexts: Optional[List[torch.Tensor]] = None
                           ) -> Tuple[float, torch.Tensor]:
        mono, d = super().probe_monotonicity(axis, contexts)
        if self.ensure_inert([axis])[axis]:
            return 0.0, torch.zeros_like(d)
        return mono, d

    # ---- ④ 低比特行程（真量化回路：量化主干两遍法）----
    @staticmethod
    def _strict_steps(p: torch.Tensor) -> int:
        return int((p[1:] > p[:-1]).sum().item())

    def probe_low_bit_travel(self, axis: int, d: torch.Tensor) -> Tuple[float, float]:
        """`travel_keep = 量化主干下严格递增档数 ÷ bf16 下严格递增档数`。

        ⭐ 与装置原口径（`quant_fp4(读出)`）的区别：量化器这次真的挂在
        **主干的所有 `GatedLinear`** 上（W4A8 = 设计档），所以测的是
        「轴信号穿过量化激活之后还剩多少行程」—— 即补充08 §2.2 风险 3(b)。
        """
        V = self._sweep(axis)
        Fb = self._eval(V)
        Fq = self._eval_r(self.qresp, V)
        pb, pq = Fb @ d, Fq @ d
        nb, nq = self._strict_steps(pb), self._strict_steps(pq)
        total = max(1, int(pb.numel()) - 1)
        keep = (nq / nb) if nb > 0 else 0.0
        rng = float(pb.max() - pb.min())
        rng_q = float(pq.max() - pq.min())
        # 辅助：装置原口径（对读出做 quant_fp4），以及 W4A4 最坏档
        keep_rd, range_rd = super().probe_low_bit_travel(axis, d)
        keep_a4 = 0.0
        if self.quant_spec is not None and self.quant_spec.act != "fp4":
            saved = self.qresp
            try:
                self.qresp = self.base.with_quant(OFFICIAL_DEFAULT_SPEC)
                F4 = self._eval_r(self.qresp, V)
                p4 = F4 @ d
                n4 = self._strict_steps(p4)
                keep_a4 = (n4 / nb) if nb > 0 else 0.0
            finally:
                self.qresp = saved
        self._aux(axis).update({
            "travel_steps_total": total,
            "travel_steps_bf16": nb,
            "travel_steps_quant_w4a8": nq,
            "travel_keep_w4a4": min(keep_a4, 1.0),
            "travel_keep_readout_quant": keep_rd,
            "travel_range_keep_readout_quant": range_rd,
            "quant_relative_error": float((Fq - Fb).norm() / (Fb.norm() + 1e-12)),
        })
        return min(keep, 1.0), (rng_q / rng if rng > 1e-9 else 0.0)

    # ---- 辅助：响应落在哪个读出块（对应 构图/光照/通道分离）----
    def _attach_block_shares(self, axes: Sequence[int]) -> None:
        if self.readout != "structured":
            return
        lay = self.base.layout()
        with torch.no_grad():
            F0 = self._eval(torch.zeros(1, self.n_axes))
            for a in axes:
                V = torch.zeros(1, self.n_axes)
                V[0, a] = 1.0
                dF = (self._eval(V)[0] - F0[0])
                tot = float((dF ** 2).sum())
                share = {}
                for name, sl in lay.items():
                    e = float((dF[sl] ** 2).sum())
                    share[name] = round(e / tot, 4) if tot > 1e-30 else 0.0
                top = max(share, key=share.get) if tot > 1e-30 else "-"
                self._aux(a)["block_share"] = share
                self._aux(a)["block_top"] = top

    def run(self, axes: Optional[Sequence[int]] = None) -> AxisProbeReport:
        axs = list(axes) if axes is not None else list(range(self.n_axes))
        self.ensure_inert(axs)                 # ⭐ 非空性预检必须先于四测
        rep = super().run(axes)
        for r in rep.results:                  # 无响应 ⇒ ① 不成立（即使 ρ 退化成了 1.00）
            if self._inert.get(r.index):
                r.mono = 0.0
                r._pass["单调"] = False
        self._attach_block_shares(axs)
        return rep


# ---------------------------------------------------------------------------
# ⑦ 报告
# ---------------------------------------------------------------------------
@dataclass
class RealProbeReport:
    probe: AxisProbeReport
    state: str                       # "as-init（门关）" | "door-open（代理）"
    door_scale: float
    seed: int
    shape: Dict[str, object]
    quant_spec: str
    aux: Dict[int, Dict[str, object]]
    n_inert: int
    n_pass: int
    l1_axes: List[str]
    l3_axes: List[str]
    fail_reasons: Dict[str, str] = field(default_factory=dict)
    null_baseline: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "state": self.state, "door_scale": self.door_scale, "seed": self.seed,
            "shape": self.shape, "quant_spec": self.quant_spec,
            "n_axes": self.probe.n_axes, "n_pass": self.n_pass, "n_inert": self.n_inert,
            "l1_axes": self.l1_axes, "l3_axes": self.l3_axes,
            "fail_reasons": self.fail_reasons,
            "thresholds": self.probe.thresholds,
            "orthogonality_null_baseline": self.null_baseline,
            "results": [r.as_dict() for r in self.probe.results],
            "aux": {str(k): v for k, v in self.aux.items()},
        }


def real_axis_report_text(rep: RealProbeReport) -> str:
    L: List[str] = []
    th = rep.probe.thresholds
    L.append("=" * 96)
    L.append("  真探针 · G3.5「L1 条件轴真实性」四测 —— 接在**真实主干 + 真实轴注入路径**上")
    L.append("=" * 96)
    L.append(f"  状态：{rep.state}｜门尺度 {rep.door_scale}｜种子 {rep.seed}")
    sh = rep.shape
    L.append(f"  主干：dim {sh.get('dim')} × {sh.get('layers')} 层（heads {sh.get('heads')}）"
             f"｜latent {sh.get('latent')}｜token {sh.get('tokens')}"
             f"｜文本 len {sh.get('text_len')}"
             f"｜读出 {sh.get('readout')}({sh.get('readout_dim')})"
             f"｜量化档 {rep.quant_spec}")
    nb = rep.null_baseline or {}
    if nb:
        L.append(f"  ⭐ ② 正交性的**随机方向基线**（D={nb.get('dim')} 维里放 "
                 f"{nb.get('n_axes')} 个随机方向，{nb.get('trials')} 次）："
                 f"max|cos| 均值 {nb.get('mean_max_cos'):.3f}"
                 f"（min {nb.get('min_max_cos'):.3f} / max {nb.get('max_max_cos'):.3f}）"
                 f"　← 门线 {th['ortho']:.2f} 高于此基线才谈得上分辨力")
    L.append(f"  轴注入点：domain({rep.probe.n_axes}) → domain_embed → adaLN → blocks"
             f"｜门线：单调≥{th['mono']:.2f} 正交≤{th['ortho']:.2f} "
             f"可逆≥{th['rev']:.2f} 行程≥{th['travel_keep']:.2f}")
    L.append("")
    L.append("-" * 96)
    L.append(f"  {'#':<3}{'轴名':<17}{'①单调':>8}{'②串扰':>8}{'③可逆':>8}{'④行程':>8}"
             f"{'B/Q档':>9}{'R²':>7}  判定 / 失败项")
    L.append("-" * 96)
    for r in rep.probe.results:
        a = rep.aux.get(r.index, {})
        inert = bool(a.get("inert"))
        flag = "✅ L1" if r.passed else "❌ L3"
        tail = "" if r.passed else "  ← " + ",".join(r.failures)
        if inert:
            # ⚠️ inert 轴的 ②③④ 读数是**数值噪声**（响应恒定时方向/差异都由
            #    GEMM 行间噪声决定），报告里显示 `--` 而不是给出误导性的数字。
            L.append(f"  {r.index:<3}{r.name:<17}{'--':>8}{'--':>8}{'--':>8}{'--':>8}"
                     f"{'--':>9}{'--':>7}  {flag} [inert：无响应]")
            continue
        bq = f"{a.get('travel_steps_bf16', '-')}/{a.get('travel_steps_quant_w4a8', '-')}"
        r2 = "--" if r.ident_r2 < 0 else f"{r.ident_r2:.3f}"
        L.append(f"  {r.index:<3}{r.name:<17}{r.mono:>8.3f}{r.ortho:>8.3f}"
                 f"{r.rev:>8.3f}{r.travel_keep:>8.3f}{bq:>9}{r2:>7}  {flag}{tail}")
    L.append("-" * 96)
    n_live = rep.probe.n_axes - rep.n_inert
    L.append(f"  通过 {rep.n_pass}/{rep.probe.n_axes}"
             f"｜数值无响应（inert）{rep.n_inert} 条｜有效测量 {n_live} 条")
    L.append("")
    L.append("  ── 辅助诊断（不设门线）──")
    L.append(f"  {'轴名':<17}{'响应幅度':>10}{'噪声底线':>10}{'主读出块':>12}"
             f"{'构图(layout)':>13}{'语义8ch':>9}{'细节32ch':>10}{'W4A4行程':>10}")
    for r in rep.probe.results:
        a = rep.aux.get(r.index, {})
        bs = a.get("block_share") or {}
        L.append(f"  {r.name:<17}{a.get('resp_dev', 0.0):>10.3e}"
                 f"{a.get('noise_floor', 0.0):>10.2e}"
                 f"{str(a.get('block_top', '-')):>12}"
                 f"{bs.get('layout', 0.0):>13.3f}{bs.get('sem_mean', 0.0):>9.3f}"
                 f"{bs.get('det_mean', 0.0):>10.3f}"
                 f"{a.get('travel_keep_w4a4', 0.0):>10.3f}")
    L.append("     注：响应幅度必须**显著高于噪声底线**才有意义 —— CPU 的 oneDNN GEMM")
    L.append("         不是逐行可复现的，同批次相同输入的行间差约 1e-8（实测，非估计）。")
    if rep.n_inert:
        L.append(f"     ⚠️ 本报告有 {rep.n_inert} 条 **inert（响应=噪声）** 的轴：它们的")
        L.append("        ②③④ 读数是数值噪声，**不构成测量结论**；判归 L3 依据的是")
        L.append("        ①的非空性前置（响应恒定时 Spearman ρ 会退化成 1.00 假通过）。")
    L.append("")
    L.append("-" * 96)
    if rep.l3_axes:
        L.append(f"  ⚠️ 未过门的轴默认归 **L3**（{len(rep.l3_axes)}/{rep.probe.n_axes}）："
                 + ", ".join(rep.l3_axes))
    if rep.l1_axes:
        L.append(f"  ✅ 留在 **L1** 的轴（{len(rep.l1_axes)}/{rep.probe.n_axes}）："
                 + ", ".join(rep.l1_axes))
    if not rep.l1_axes and not rep.l3_axes:
        L.append("  （无轴）")
    L.append("     判定律：「能显式写入的，就不要隐式请求」（控制权原则，补08§2.3④）")
    L.append("=" * 96)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# ⑧ 高层入口
# ---------------------------------------------------------------------------
DEFAULT_SHAPE = dict(dim=192, layers=6, heads=6)


def build_test_backbone(*, dim: int = 192, layers: int = 6, heads: int = 6,
                        seed: int = 0, latent_ch: Optional[int] = None,
                        identity_anchor_layers: Sequence[int] = ()
                        ) -> SingleStreamDiT:
    """测试形状主干（**不是** KP-S 全尺寸：纯 CPU 可跑是硬要求）。

    固定种子 ⇒ 权重逐位可复现 ⇒ 四测数字可复算。
    """
    torch.manual_seed(int(seed))
    cfg = DiTCfg(dim=int(dim), layers=int(layers), heads=int(heads), mlp_ratio=2.0,
                 double_stream_blocks=min(2, int(layers)),
                 matryoshka_tokens=(16, 64))
    model = SingleStreamDiT(cfg, latent_ch=latent_ch or LATENT.total_ch,
                            identity_anchor_layers=list(identity_anchor_layers),
                            domain_dim=len(DOMAIN_AXIS_NAMES))
    model.eval()
    return model


def run_real_probe(*, door: bool = True, door_scale: float = 0.02, seed: int = 0,
                   dim: int = 192, layers: int = 6, heads: int = 6, tokens: int = 8,
                   use_text: bool = True, quant_spec: Optional[QuantSpec] = DESIGN_SPEC,
                   model: Optional[SingleStreamDiT] = None,
                   sweep_steps: Optional[int] = None,
                   n_random: Optional[int] = None,
                   latent_ch: Optional[int] = None,
                   readout: str = "structured") -> RealProbeReport:
    """建主干（或复用）→ 开门（可选）→ 四测 → 出报告。纯 CPU / 固定种子。"""
    if model is None:
        model = build_test_backbone(dim=dim, layers=layers, heads=heads, seed=seed,
                                    latent_ch=latent_ch)
    if door:
        open_domain_door(model, scale=door_scale, seed=seed)

    probe = RealAxisProbe(model, tokens=tokens, use_text=use_text, seed=seed,
                          quant_spec=quant_spec, sweep_steps=sweep_steps,
                          n_random=n_random, latent_ch=latent_ch, readout=readout)
    rep = probe.run()
    shape = {**probe.base.describe(), "dim": model.cfg.dim, "layers": model.cfg.layers,
             "heads": model.cfg.heads}

    results = rep.results
    l1 = [r.name for r in results if r.passed]
    l3 = [r.name for r in results if not r.passed]
    n_inert = sum(1 for r in results if probe.aux.get(r.index, {}).get("inert"))
    reasons = {r.name: ",".join(r.failures) for r in results if not r.passed}
    base = orthogonality_null_baseline(len(DOMAIN_AXIS_NAMES),
                                       int(shape["readout_dim"]), seed=seed)

    return RealProbeReport(
        probe=rep,
        state=("door-open（**未训练主干的开门代理**：只测注入路径）" if door
               else "as-init（门关：adaLN-Zero，域向量进不去输出）"),
        door_scale=float(door_scale) if door else 0.0,
        seed=int(seed),
        shape=shape,
        quant_spec=("none" if quant_spec is None else f"{quant_spec.weight}/{quant_spec.act}"),
        aux=probe.aux,
        n_inert=n_inert,
        n_pass=rep.n_pass,
        l1_axes=l1, l3_axes=l3, fail_reasons=reasons,
        null_baseline=base,
    )

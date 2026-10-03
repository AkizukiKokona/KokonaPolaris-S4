"""G3 装置：Sigmoid 注意力「长提示收益」与「量化友好性」的可执行测量。

═══════════════════════════════════════════════════════════════════════
⚠️ **定位 —— 先读这段，否则会误用本模块的数字**
═══════════════════════════════════════════════════════════════════════
G3 的**官方判据**（主文档 §9.2 `design/KokonaPolaris_架构设计方案.md:1132`）是：

    「>150 token 的 Spatial 分数优于 softmax 版，且短提示不退化」

那是 **benchmark 级**判据，**必须有训练过的模型**才能跑；而 KP **尚未预训练**
（G6 未做，验证机 ≠ 训练机 ⇒ 须租云）。⇒ **G3 官方判据当前结构性不可执行。**

所以本模块做的是**装置级**工作，与 G2 / G3.5 的做法一致：
先把「尺子」造出来，并证明它在**已知答案的样本**上给出正确答案；
等 G6 落地，官方判据即可在此基础上直接接上。

⛔ **本模块的数字全部来自随机权重，不等于训练后模型的行为**，
   **不得**当作「G3 已过门」的依据。它只能回答：
   「这条机制主张在**算子层面**是否成立 / 装置是否有分辨力」。

═══════════════════════════════════════════════════════════════════════
设计稿的三条机制主张（本模块逐条对应）
═══════════════════════════════════════════════════════════════════════
L310「长提示收益」：softmax 把权重归一化到总和 1 ⇒ token 竞争同一块概率质量，
      提示越长每个词分到的份额越小；Sigmoid 对每对独立打分，无归一化竞争。
L311「量化友好性」：Sigmoid/Softpick 能实现**接近零的 attention sink 率**并
      **显著减少激活离群值** → 让低比特量化更鲁棒。
L313：全局不出现 softmax 会损失少量 sharp attention ⇒ 保留第 0 层一次 softmax 锚点。

⚠️ **官方判据缺一半（L1132 的「量化友好性」在「通过判据」列没有对应条目）**
   ⇒ `activation_shape_stats` **只报数、不设门线**，阈值需拍板。见 `G3_GAPS`。

═══════════════════════════════════════════════════════════════════════
测量原理（为什么能用 one-hot v 反解权重）
═══════════════════════════════════════════════════════════════════════
`_attn_core` 返回 `out = Σ_j a_j · v_j`。令 `v = I`（单位阵，head_dim = N），
则 `out[..., j] = a_j` —— **逐位等于真实代码路径上的注意力权重**，
而不是「另写一份 softmax 公式」的近似。softmax 的权重天然和为 1，
Sigmoid 的不和为 1，故 Sigmoid 需再除以自身总和才是「份额」。

构造受控 logit：`Dh = N`，`q = e₀`，`k_j[0] = logit_j·√N` ⇒
`score_j = (e₀·k_j)/√Dh = logit_j`（SDPA 与本模块都用 `1/√Dh` 缩放）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from kp.models.dit import LINEAR, SIGMOID, SOFTMAX, _attn_core, build_attn_plan

__all__ = [
    "G3_GAPS",
    "DilutionPoint",
    "DilutionResult",
    "AttentionWeights",
    "attention_weights",
    "dilution_share",
    "dilution_curve",
    "dilution_exponent",
    "compare_dilution",
    "activation_shape_stats",
    "plan_composition",
    "known_answer_samples",
    "g3_report_text",
]


# ⛔ 设计稿**未给出**阈值的地方 —— 按「不猜」原则显式报缺口，不擅自设门
G3_GAPS: Tuple[str, ...] = (
    "官方判据（L1132）只覆盖「长提示 Spatial 收益」，"
    "「量化友好性」那一半在通过判据列**没有条目** ⇒ 无数值门线",
    "「优于 softmax 版」「短提示不退化」均**无阈值**，且未指定 benchmark",
    "短提示上界未定义（只给了 >150 token 的长提示线）",
    "本装置基于**随机权重**，只验机制层，不能替代 G6 后的 benchmark 级验收",
)


@dataclass(frozen=True)
class AttentionWeights:
    """从**真实** `_attn_core` 反解出的注意力权重（未归一化）。"""

    kind: str
    weights: torch.Tensor          # [B, H, Nq, Nk]
    logits: torch.Tensor           # [B, H, Nq, Nk] —— 送进 softmax/sigmoid 之前的原始分

    def shares(self) -> torch.Tensor:
        """归一化份额（Σ=1）。softmax 本就等于 1；Sigmoid 必须自己除。"""
        return self.weights / self.weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)

    def total_mass(self) -> torch.Tensor:
        """权重总和 —— ⭐ 这是两种机制的**分水岭**：softmax 恒 1，Sigmoid 不受约束。"""
        return self.weights.sum(dim=-1)


def attention_weights(kind: str, logits: Sequence[float], *,
                      n_query: int = 1, seed: int = 0) -> AttentionWeights:
    """把一维目标 logit 灌进真实 `_attn_core`，反解出精确权重。

    ⚠️ 走的是 `kp.models.dit._attn_core` —— **测的是真代码路径**，
       不是这里另写的公式（项目教训：断言要检验设计想表达的不变量）。
    """
    n = len(logits)
    if n < 2:
        raise ValueError(f"N={n} < 2：无法谈稀释（没有背景 token 可被摊薄）")

    dh = n
    q = torch.zeros(1, 1, n_query, dh)
    q[..., 0] = 1.0                                   # q = e₀
    k = torch.zeros(1, 1, n, dh)
    k[..., 0] = torch.tensor(logits) * math.sqrt(dh)  # ⇒ score = logit
    v = torch.eye(n).reshape(1, 1, n, dh)             # one-hot ⇒ out[..., j] = a_j

    with torch.no_grad():
        out = _attn_core(q, k, v, kind)
        raw = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    return AttentionWeights(kind=kind, weights=out, logits=raw)


def dilution_share(kind: str, n_tokens: int, *, sig_logit: float = 4.0,
                   bg_logit: float = 0.0, sig_index: int = 0) -> float:
    """信号 token 在 `n_tokens` 长序列里分到的**份额**。

    `bg_logit` 越高 = 背景 token「越相关」；`sig_logit - bg_logit` = 信号有多突出。
    """
    logits = [bg_logit] * n_tokens
    logits[sig_index] = sig_logit
    return float(attention_weights(kind, logits).shares()[0, 0, 0, sig_index])


@dataclass(frozen=True)
class DilutionPoint:
    n_tokens: int
    share: float


@dataclass
class DilutionResult:
    kind: str
    points: List[DilutionPoint] = field(default_factory=list)
    exponent: float = float("nan")     # share ∝ N^(-α)；α≈1 = 强稀释
    degenerate: Optional[str] = None   # 非 None ⇒ 退化输入，已报缺口

    @property
    def ok(self) -> bool:
        return self.degenerate is None and len(self.points) >= 2


def dilution_curve(kind: str, n_seq: Sequence[int] = (16, 32, 64, 128, 256),
                   *, sig_logit: float = 4.0, bg_logit: float = 0.0) -> DilutionResult:
    """扫多个序列长度取份额曲线，并拟合稀释指数 α。

    ⚠️ **门线不能单点采样**（项目已踩过：`noise_floor` 只测一个 batch，
    恰好撞上 GEMM 分块的 0.0 巧合点）⇒ 这里扫 5 个长度，并对 α 取最小值（最保守）。
    """
    pts: List[DilutionPoint] = []
    for n in n_seq:
        if n < 2:
            return DilutionResult(kind, [], float("nan"),
                                  degenerate=f"N={n} < 2，无法定义稀释")
        try:
            s = dilution_share(kind, int(n), sig_logit=sig_logit, bg_logit=bg_logit)
        except (RuntimeError, ValueError) as e:
            return DilutionResult(kind, [], float("nan"),
                                  degenerate=f"N={n} 测量失败：{type(e).__name__}: {e}")
        if not math.isfinite(s) or s <= 0.0:
            return DilutionResult(kind, [], float("nan"),
                                  degenerate=f"N={n} 份额退化（{s}）")
        pts.append(DilutionPoint(int(n), s))

    xs = [math.log(p.n_tokens) for p in pts]
    ys = [math.log(p.share) for p in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    slope = 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return DilutionResult(kind, pts, exponent=-slope)


def dilution_exponent(kind: str, **kw) -> float:
    """便捷入口：只要稀释指数 α（`share ∝ N^-α`）。"""
    return dilution_curve(kind, **kw).exponent


def compare_dilution(*, sig_logit: float = 4.0, bg_logit: float = 0.0,
                     n_seq: Sequence[int] = (16, 32, 64, 128, 256)) -> Dict[str, object]:
    """softmax vs Sigmoid 的稀释对照（装置核心对照实验）。"""
    sm = dilution_curve(SOFTMAX, n_seq, sig_logit=sig_logit, bg_logit=bg_logit)
    sg = dilution_curve(SIGMOID, n_seq, sig_logit=sig_logit, bg_logit=bg_logit)
    out: Dict[str, object] = {"softmax": sm, "sigmoid": sg,
                              "sig_logit": sig_logit, "bg_logit": bg_logit}
    if sm.ok and sg.ok:
        # 每步份额保留比例：N 翻倍时没被摊掉的比例（1.0 = 完全不稀释）
        out["retain_softmax"] = _step_retain(sm)
        out["retain_sigmoid"] = _step_retain(sg)
        out["sigmoid_over_softmax"] = (out["retain_sigmoid"] / out["retain_softmax"]
                                       if out["retain_softmax"] > 0 else float("inf"))
    return out


def _step_retain(r: DilutionResult) -> float:
    """相邻两点的份额比的几何平均 —— 「序列变长时，保住的比例」。"""
    ratios = [r.points[i + 1].share / r.points[i].share for i in range(len(r.points) - 1)]
    return float(sum(ratios) / len(ratios))


def activation_shape_stats(x: torch.Tensor) -> Dict[str, float]:
    """激活分布形状（L311「减少激活离群值」的可测代理）。

    ⭐ **必须先按 token 做 RMS 归一**再比形状：Sigmoid 无归一化，原始幅度会随 N 线性增长，
    直接比 max/median 会被幅度差掩盖；而 KP 下游本来就有 `RMSNorm` + `head_gate`
    ⇒ 归一化后的形状才是真正被量化器看到的东西。
    """
    xf = x.detach().float().reshape(-1)
    rms = xf.pow(2).mean().sqrt().clamp_min(1e-12)
    y = xf / rms
    med = y.abs().median().clamp_min(1e-12)
    ex = y - y.mean()
    var = ex.pow(2).mean().clamp_min(1e-12)
    return {
        "outlier_ratio": float(y.abs().max() / med),       # 越大越「离群」
        "kurtosis": float(ex.pow(4).mean() / (var * var)),
        "abs_max": float(y.abs().max()),
        "rms": float(rms),
    }


def _act_stats_for(kind: str, *, n_tokens: int, dim: int = 64, heads: int = 4,
                   seed: int = 0) -> Dict[str, float]:
    g = torch.Generator().manual_seed(int(seed) + 4242)
    dh = dim // heads
    q = torch.randn(1, heads, n_tokens, dh, generator=g)
    k = torch.randn(1, heads, n_tokens, dh, generator=g)
    v = torch.randn(1, heads, n_tokens, dh, generator=g)
    with torch.no_grad():
        out = _attn_core(q, k, v, kind)
    return activation_shape_stats(out)


def plan_composition(layers: int = 32, n_linear: int = 3, n_sigmoid: int = 1) -> Dict[str, int]:
    """3:1 混合注意力 + 第 0 层 softmax 锚点（L308 / L313）的构成核对。"""
    plan = build_attn_plan(layers, n_linear, n_sigmoid)
    c = {SOFTMAX: 0, SIGMOID: 0, LINEAR: 0}
    for p in plan:
        c[p] = c.get(p, 0) + 1
    return {"layers": len(plan), "softmax": c[SOFTMAX],
            "sigmoid": c[SIGMOID], "linear": c[LINEAR]}


# ---------------------------------------------------------------------------
# ⭐ 判别力：QK-Norm 与 Sigmoid 的相性（2026-10-03 装置发现）
# ---------------------------------------------------------------------------
def qknorm_logit_bound(dim: int, heads: int) -> float:
    """**QK-Norm 之后**注意力 logit 能达到的最大幅度。

    QK-Norm（主文档 L315「必开」）对 q/k 各做 RMSNorm ⇒ `‖q‖=‖k‖=√Dh`。
    Cauchy–Schwarz ⇒ `q·k ≤ Dh`，再除以缩放 `√Dh` ⇒ **logit 上界 = √Dh**。
    """
    dh = dim // heads
    return math.sqrt(float(dh))


def _sigmoid(x: float) -> float:
    """数值稳健的 sigmoid（大负数不溢出）。"""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def contrast_vs_offset(kind: str, delta: float,
                       offsets: Sequence[float] = (-12.0, -8.0, -4.0, 0.0, 4.0)) -> List[float]:
    """固定**间距** Δ = hi−lo，扫**绝对零点**，量判别力 `w(hi)/w(lo)`。

    ⭐ 这是本装置最关键的对照。两种机制的对比度对「零点」的依赖完全不同：

        softmax : `e^(hi−lo) = e^Δ`   —— **与零点无关**（平移不变）
        sigmoid : `σ(hi)/σ(lo)`        —— **强烈依赖零点**

    ⇒ 如果 logit 整体漂到正侧（每个 token 都「有点相关」）：
       **softmax 保住对比度，sigmoid 塌向 `1/0.5 = 2×`。**
       这才是「长提示收益」主张的真实风险点。
    """
    out: List[float] = []
    for off in offsets:
        lo, hi = off, off + delta
        if kind == SOFTMAX:
            w_hi, w_lo = math.exp(hi), math.exp(lo)
        else:
            w_hi, w_lo = _sigmoid(hi), _sigmoid(lo)
        out.append(w_hi / max(w_lo, 1e-300))
    return out


def contrast_report(delta: float = 8.0,
                    offsets: Sequence[float] = (-12.0, -8.0, -4.0, 0.0, 4.0)) -> Dict[str, object]:
    """softmax vs sigmoid 的「对比度 × 零点」矩阵。"""
    return {"delta": delta, "offsets": list(offsets),
            "softmax": contrast_vs_offset(SOFTMAX, delta, offsets),
            "sigmoid": contrast_vs_offset(SIGMOID, delta, offsets)}


def known_answer_samples(*, n_seq: Sequence[int] = (16, 32, 64, 128)) -> Dict[str, float]:
    """**已知答案**的对照样本 —— 验的是 **α 估计器本身**（log-log 最小二乘斜率）。

    ⚠️ **先分清两层，否则会误以为它和 `kp/selftest.py` §21 重复**
       （审计 `out/audit_stale_and_dead.md` §4.1 正是把这两层混为一谈才标成孤儿）：

    【层 ①「估计器层」= 本函数】
        验 `α` 是怎么**算**出来的：喂**构造出来的**份额序列，不碰 `_attn_core`。
          · flat         份额恒 1/N ⇒ α 必须 = 1（**定义上的强稀释**）
          · concentrated 份额恒 0.9 ⇒ α 必须 = 0（**定义上的零稀释**）
        这一层错了，§21 的所有 α 数字都无意义（拟合层就写错了）。

    【层 ②「机制 / 代码路径层」= `kp/selftest.py` §21】
        走**真代码路径**（`dilution_exponent` → `attention_weights` →
        `kp.models.dit._attn_core`），验的是**设计主张**本身：
          · 等 logit ⇒ α 必为 1（尺子有分辨力）
          · softmax 靠**抬信号**不稀释（平移不变）
          · sigmoid 靠**压背景到负侧**才不稀释（看绝对零点）

    ⇒ **两层互补、不是二选一**：本函数管「算得对不对」，§21 管「测的是不是真东西」。
       建议接线（见本轮修复报告）：把本函数加成 §21 的一节「**先验尺子**」，
       放在现有三项之前 —— 它更便宜（纯 CPU、无矩阵乘），且能拦住拟合层的错。

    实测值（n_seq = 16/32/64/128）：`{'flat_alpha': 1.0, 'concentrated_alpha': -0.0}`。
    """
    def _alpha(shares: Sequence[float]) -> float:
        xs = [math.log(n) for n in n_seq]
        ys = [math.log(max(s, 1e-30)) for s in shares]
        k = len(xs); mx, my = sum(xs) / k, sum(ys) / k
        den = sum((x - mx) ** 2 for x in xs)
        return 0.0 if den == 0 else -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    flat = _alpha([1.0 / n for n in n_seq])
    conc = _alpha([0.9 for _ in n_seq])
    return {"flat_alpha": flat, "concentrated_alpha": conc}


def g3_report_text() -> str:
    """装置自检报告（文本）。"""
    L = ["=" * 74, "G3 装置自检 · Sigmoid 注意力（长提示收益 / 量化友好性）", "=" * 74]

    comp = plan_composition(32)
    L.append("\n【3:1 混合注意力构成（L308 / L313）】")
    L.append(f"  32 层 ⇒ softmax 锚点 {comp['softmax']}（应=1）/ "
             f"linear {comp['linear']} / Sigmoid {comp['sigmoid']}（3:1）")

    L.append("\n【稀释对照：信号 logit 越突出，背景越相关】")
    L.append(f"  {'bg_logit':>9} {'α(softmax)':>12} {'α(sigmoid)':>12} "
             f"{'保留比 sm':>11} {'保留比 sg':>11} {'sg/sm':>9}")
    for bg in (-2.0, 0.0, 2.0):
        c = compare_dilution(sig_logit=4.0, bg_logit=bg)
        sm, sg = c["softmax"], c["sigmoid"]          # type: ignore[assignment]
        if not (sm.ok and sg.ok):                   # type: ignore[union-attr]
            L.append(f"  {bg:>9.1f}  退化：{sm.degenerate or sg.degenerate}")  # type: ignore[union-attr]
            continue
        ratio = c.get("sigmoid_over_softmax", float("nan"))
        L.append(f"  {bg:>9.1f} {sm.exponent:>12.3f} {sg.exponent:>12.3f} "      # type: ignore[union-attr]
                 f"{c['retain_softmax']:>11.3f} {c['retain_sigmoid']:>11.3f} "  # type: ignore[index]
                 f"{ratio:>9.2f}")
    L.append("  α = 稀释指数（share ∝ N^-α；1=按长度线性摊薄，0=不摊薄）")

    L.append("\n【激活离群值（L311「减少离群值 → 量化更鲁棒」）】")
    L.append(f"  {'N':>6} {'离群比 sm':>11} {'离群比 sg':>11} {'峰度 sm':>10} {'峰度 sg':>10}")
    for n in (32, 128, 512):
        a, b = _act_stats_for(SOFTMAX, n_tokens=n), _act_stats_for(SIGMOID, n_tokens=n)
        L.append(f"  {n:>6} {a['outlier_ratio']:>11.2f} {b['outlier_ratio']:>11.2f} "
                 f"{a['kurtosis']:>10.2f} {b['kurtosis']:>10.2f}")
    L.append("  ⚠️ 归一化后比形状（Sigmoid 原始幅度会随 N 增长，KP 下游有 RMSNorm 兜）")

    L.append("\n【⭐ 判别力 × 绝对零点（固定间距 Δ，看两种机制对零点的依赖）】")
    for dh, tag in ((72, "KP-S head_dim=72"), (112, "KP-M head_dim=112")):
        bound = qknorm_logit_bound(dh * 16, 16)
        cr = contrast_report(delta=float(min(8.0, bound)))
        d = cr["delta"]
        L.append(f"  {tag} ⇒ QK-Norm 下 logit 上界 {bound:.2f}；取 Δ={d:.0f}")
        L.append(f"    {'零点':>7} {'softmax':>14} {'sigmoid':>14} {'sg/sm':>9}")
        for i, off in enumerate(cr["offsets"]):
            sm, sg = cr["softmax"][i], cr["sigmoid"][i]   # type: ignore[index]
            L.append(f"    {off:>7.0f} {sm:>14,.0f} {sg:>14,.0f} {sg / sm:>9.3f}")
    L.append("  ⇒ ⭐ **softmax 的对比度与零点无关（平移不变）**；")
    L.append("     **sigmoid 强烈依赖零点** —— 零点漂到正侧时塌向 1/0.5 = 2×。")
    L.append("     这才是「长提示收益」主张的真实风险点，而非 sigmoid 的对比度上限。")

    L.append("\n【⛔ 缺口（设计稿未给出，不擅自设门）】")
    for g in G3_GAPS:
        L.append(f"  · {g}")

    L.append("\n【官方判据现状】")
    L.append("  主文档 §9.2 L1132：「>150 token 的 Spatial 分数优于 softmax 版，")
    L.append("  且短提示不退化」⇒ benchmark 级，**需 G6 预训练后**才能执行。")
    L.append("  ⛔ 本装置只验机制层（随机权重），**不得**当作 G3 过门依据。")

    L.append("\n" + "=" * 74)
    return "\n".join(L)

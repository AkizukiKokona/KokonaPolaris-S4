"""Axis Probe —— L1 条件轴真实性的测量装置（验证门 **G3.5**）。

## 为什么这件东西必须存在

三层控制栈里，**L1「主干内生条件轴」承担约 90% 的调用量**，但它是全案**唯一
无法自证的环节**：

| | L3 子空间约束 LoRA | **L1 内生条件轴** |
|---|---|---|
| 失败了能看出来吗 | 能（谱检查可跑） | 只能靠**出图间接看** |
| 能修吗 | 能（重训包，小时级） | **只能重训主干**（天级） |
| 控制权在谁手上 | 我们 | ⚠️ **悄悄转移到基座** |

⇒ **「我们能控制 L1 的旋钮」这件事本身，是一个没有先例的假设。**
本模块把它从形容词变成**四个可测的数字**。

## 四测（G3.5）

| # | 测什么 | 判据 | 不过意味着什么 |
|---|---|---|---|
| ① | **单调性** | 轴值单调 ⇒ 响应沿该轴方向单调（Spearman ρ ≥ 阈值） | 「旋钮」其实是开关或乱跳的 |
| ② | **正交性** | 跨轴方向余弦 `abs(cos)` ≤ 阈值 | 转一个轴会带歪另一个 ⇒ 面板不可标定 |
| ③ | **可逆性** | `+a` 再 `−a` 回到原位（奇对称，不对称误差 ≤ 阈值） | 轴有滞回 / 二次项 ⇒ 设了回不来 |
| ④ | ⭐ **低比特行程** | NVFP4 量化后仍可区分的行程比例 ≥ 阈值 | **旋钮行程变短、转到一半就饱和** |

### ④ 为什么是四条里最容易致命的一条

FP4 激活只有 **16 级 E2M1**。若某个条件轴是靠**激活值的精细结构**传递的，
量化会**系统性地压缩该轴的分辨率** —— 表现为「轴还在、也单调，但后半程没有效果」。
而这条恰恰**在 bf16 上调参时完全看不出来**，只有把量化器接进回路才暴露。

> ⚠️ 口径：本装置测的是**给定 responder 的轴性质**，不是「KP 主干一定如何」。
> 真实用法是把 responder 换成「主干 + 该轴注入路径」的探针函数。
> 自检里用**合成 responder** 验证装置本身能同时抓对/抓错。

## 判定律

**任何一测不过 ⇒ 该轴默认归 L3**（降级为子空间约束 LoRA，见 `kp.capability`）。
这是设计稿写死的：**能显式写入的（L3），不要隐式请求（L1）**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..config import AXIS
from ..quant.nvfp4 import quant_fp4

Responder = Callable[[torch.Tensor], torch.Tensor]
"""`(B, n_axes) -> (B, D)`：给一组轴值，返回可观测的响应（如特征/中间激活）。"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman 秩相关（无 scipy 依赖；轴值是等距采样，无并列秩）。"""
    n = a.numel()
    if n < 3:
        return 0.0

    def _rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x)
        r = torch.empty(n, dtype=torch.float32)
        r[order] = torch.arange(n, dtype=torch.float32)
        return r

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    if denom <= 1e-12:
        return 0.0
    return float((ra @ rb) / denom)


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = float(v.norm())
    return v / n if n > 1e-12 else v


def _ridge_r2(X: torch.Tensor, y: torch.Tensor, lam: float = 1e-3,
              test_frac: float = 0.3) -> float:
    """岭回归 X→y 的测试集 R²（可辨识性的判据）。"""
    n = X.shape[0]
    n_test = max(1, int(n * test_frac))
    perm = torch.randperm(n)
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    # 标准化（否则尺度差异会让 R² 失真）
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1)
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * torch.eye(d)
    w = torch.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class AxisResult:
    index: int
    name: str
    mono: float
    ortho: float          # 与其他轴的最大 |cos|（越小越好）
    rev: float            # ③ 可逆性：1 − 最大不对称误差（越大越好）
    travel_keep: float    # NVFP4 下仍可区分的行程比例
    # ---- 辅助诊断（不设门线）----
    ident_r2: float = 0.0
    travel_range_keep: float = 0.0
    _pass: Dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self._pass.values())

    @property
    def verdict(self) -> str:
        return "L1（内生轴，可直接当旋钮用）" if self.passed else \
               "L3（未过门，降级为子空间约束 LoRA）"

    @property
    def failures(self) -> List[str]:
        return [k for k, v in self._pass.items() if not v]

    def as_dict(self) -> dict:
        return {
            "index": self.index, "name": self.name,
            "mono": round(self.mono, 4), "ortho": round(self.ortho, 4),
            "rev": round(self.rev, 4),
            "travel_keep": round(self.travel_keep, 4),
            "aux_ident_r2": round(self.ident_r2, 4),
            "aux_travel_range_keep": round(self.travel_range_keep, 4),
            "passed": self.passed, "failures": self.failures,
            "verdict": self.verdict,
        }


@dataclass
class AxisProbeReport:
    n_axes: int
    n_sweep: int
    results: List[AxisResult]
    thresholds: Dict[str, float]

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def l3_axes(self) -> List[str]:
        return [r.name for r in self.results if not r.passed]

    def as_dict(self) -> dict:
        return {
            "n_axes": self.n_axes, "n_sweep": self.n_sweep,
            "thresholds": self.thresholds,
            "n_pass": self.n_pass,
            "l3_axes": self.l3_axes,
            "results": [r.as_dict() for r in self.results],
        }


def axis_report_text(rep: AxisProbeReport) -> str:
    L = []
    L.append("=" * 78)
    L.append("  Axis Probe · G3.5「L1 条件轴真实性」四测")
    L.append("=" * 78)
    th = rep.thresholds
    L.append(f"  轴数 {rep.n_axes}｜单轴扫描 {rep.n_sweep} 步｜"
             f"门线：单调≥{th['mono']:.2f} 正交≤{th['ortho']:.2f} "
             f"可逆≥{th['rev']:.2f} 行程≥{th['travel_keep']:.2f}")
    L.append("")
    L.append("-" * 78)
    L.append(f"  {'轴':<18}{'①单调':>8}{'②串扰':>8}{'③可逆':>8}{'④行程':>8}   判定")
    L.append("-" * 78)
    for r in rep.results:
        flag = "✅" if r.passed else "❌"
        L.append(f"  {r.name:<18}{r.mono:>8.3f}{r.ortho:>8.3f}"
                 f"{r.rev:>8.3f}{r.travel_keep:>8.3f}   {flag} "
                 + ("" if r.passed else "不过：" + ",".join(r.failures)))
    L.append("-" * 78)
    L.append(f"  通过 {rep.n_pass}/{rep.n_axes}")
    if rep.l3_axes:
        L.append(f"  ⚠️ 未过门的轴默认归 **L3**：{', '.join(rep.l3_axes)}")
        L.append("     ——「能显式写入的，就不要隐式请求」（控制权原则）")
    L.append("=" * 78)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 探针本体
# ---------------------------------------------------------------------------
class AxisProbe:
    """对一组条件轴做四测。`responder` 决定「响应」是什么（特征 / 中间激活 / 出图）。"""

    def __init__(self, responder: Responder, n_axes: Optional[int] = None,
                 names: Optional[Sequence[str]] = None, *,
                 sweep_steps: Optional[int] = None,
                 n_random: Optional[int] = None, seed: int = 0):
        self.f = responder
        self.n_axes = int(n_axes if n_axes is not None else AXIS.n_axes)
        self.names = list(names) if names else [f"axis{i:02d}" for i in range(self.n_axes)]
        self.sweep_steps = int(sweep_steps or AXIS.sweep_steps)
        self.n_random = int(n_random or AXIS.n_random)
        self.seed = seed
        if len(self.names) != self.n_axes:
            raise ValueError(f"names 长度 {len(self.names)} ≠ n_axes {self.n_axes}")

    # ---------------- 基本操作 ----------------
    def _eval(self, V: torch.Tensor) -> torch.Tensor:
        """V: (B, n_axes) → (B, D)，梯度关闭、fp32。"""
        with torch.no_grad():
            out = self.f(V.to(torch.float32))
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.reshape(out.shape[0], -1).to(torch.float32)

    def _sweep(self, axis: int, ctx: Optional[torch.Tensor] = None,
               lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
        """单轴等距扫描 → (steps, n_axes)。"""
        vals = torch.linspace(lo, hi, self.sweep_steps)
        V = torch.zeros(self.sweep_steps, self.n_axes)
        if ctx is not None:
            V += ctx.reshape(1, -1)
        V[:, axis] = vals
        return V

    def axis_values(self, axis: int) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.sweep_steps)

    def _default_contexts(self) -> List[Optional[torch.Tensor]]:
        """单调性与可逆性都要**跨上下文**成立，不能只在原点附近成立。"""
        g = torch.Generator().manual_seed(self.seed)
        return [None,
                (torch.rand(self.n_axes, generator=g) - 0.5) * 0.6,
                (torch.rand(self.n_axes, generator=g) - 0.5) * 0.6]

    # ---------------- ① 单调性 ----------------
    def probe_monotonicity(self, axis: int, contexts: Optional[List[torch.Tensor]] = None
                           ) -> Tuple[float, torch.Tensor]:
        """返回 (min Spearman ρ, 该轴方向 d)。

        方向 d 由**零上下文**的增量均值定出，再在多个上下文下检验单调性
        —— 单调必须是**跨上下文稳定**的，只在某一处单调不算数。
        """
        if contexts is None:
            contexts = self._default_contexts()

        # 方向 d：零上下文下的平均增量
        V0 = self._sweep(axis, contexts[0])
        F0 = self._eval(V0)
        delta = F0[1:] - F0[:-1]
        d = _unit(delta.mean(0))

        scores = []
        for ctx in contexts:
            V = self._sweep(axis, ctx)
            F = self._eval(V)
            proj = F @ d                      # 沿轴方向的响应
            rho = _spearman(self.axis_values(axis), proj)
            scores.append(rho)
        return min(scores), d

    # ---------------- ② 正交性 ----------------
    def probe_orthogonality(self, axes: Optional[Sequence[int]] = None
                            ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """返回 (串扰矩阵 C[i,j]=|cos(d_i,d_j)|, 方向字典)。只算 i≠j 的最大值。"""
        axes = list(axes) if axes is not None else list(range(self.n_axes))
        dirs: Dict[int, torch.Tensor] = {}
        for i in axes:
            V = self._sweep(i)
            F = self._eval(V)
            dirs[i] = _unit((F[1:] - F[:-1]).mean(0))
        C = torch.eye(len(axes))
        for a, i in enumerate(axes):
            for b, j in enumerate(axes):
                if a != b:
                    C[a, b] = abs(float(dirs[i] @ dirs[j]))
        return C, dirs

    # ---------------- ③ 可逆性 ----------------
    def probe_reversibility(self, axis: int,
                            contexts: Optional[List[torch.Tensor]] = None
                            ) -> float:
        """`+a` 再 `−a` 是否回到原位 —— 即响应沿该轴**奇对称**。

        判据（设计稿「`+a` 再 `−a` 回到原图，在容差内」的无状态等价形式）：
            令 d± = f(v=±a) − f(v=0)，
            不对称误差 err = ‖d₊ + d₋‖ / (‖d₊‖ + ‖d₋‖)，对所有 |a| 取最大。
            返回 **1 − max err**（1 = 完美可逆，0 = 完全不可逆）。

        为什么不是"再走一遍就回去了"：对**无状态**的前馈函数那恒成立（废话）。
        真正要防的是**二次项 / 滞回** —— 那才是"旋钮转过去回不来"的物理来源。
        """
        if contexts is None:
            contexts = self._default_contexts()
        half = self.sweep_steps // 2
        if half < 1:
            return 1.0

        worst = 0.0
        for ctx in contexts:
            V = self._sweep(axis, ctx)
            F = self._eval(V)
            # ⚠️ 参考点必须是**本次扫描的中心**（V[:,axis]=0、其余=ctx），
            #    不能沿用零上下文的 F(0) —— 否则上下文本身就被算成了不对称。
            F0 = F[half]
            for k in range(1, half + 1):
                dp = F[half + k] - F0
                dm = F[half - k] - F0
                denom = float(dp.norm() + dm.norm())
                if denom <= 1e-9:
                    continue
                err = float((dp + dm).norm()) / denom
                worst = max(worst, err)
        return 1.0 - worst

    # ---------------- 辅助：线性可辨识 ----------------
    def probe_identifiability(self, axis: int,
                              V: Optional[torch.Tensor] = None) -> float:
        """辅助诊断（**不设门线**）：从响应线性回归出轴值的测试集 R²。

        它回答的是"轴值还在不在响应里"，与 ③ 可逆性互补但不等价
        —— 单调且对称的线性轴，二者应当同向。
        """
        g = torch.Generator().manual_seed(self.seed + 1)
        if V is None:
            V = torch.rand(self.n_random, self.n_axes, generator=g) * 2 - 1
        F = self._eval(V)
        return _ridge_r2(F, V[:, axis])

    # ---------------- ④ 低比特行程 ----------------
    def probe_low_bit_travel(self, axis: int, d: torch.Tensor
                             ) -> Tuple[float, float]:
        """把响应按 NVFP4 量化后再沿 d 投影，测**还能分辨多少行程**。

        返回 (可区分行程比例, 幅度保留比例)。

        ⭐ 这是四条里**只有在量化回路里才看得见**的一条：轴可能依然单调、
        依然可辨识，但 FP4 的 16 级 E2M1 把相邻档位压进了同一格
        —— 「轴还在，后半程没效果」。
        """
        vals = self.axis_values(axis)
        V = self._sweep(axis)
        F = self._eval(V)
        Fq = quant_fp4(F, ste=False)

        # 相邻档是否仍可区分
        step_f = (F[1:] - F[:-1]).norm(dim=1)
        step_q = (Fq[1:] - Fq[:-1]).norm(dim=1)
        alive = (step_q > 0).float()
        keep = float(alive.mean()) if alive.numel() else 0.0

        # 幅度保留（辅助指标，不作门线）
        p = F @ d
        pq = Fq @ d
        rng = float(p.max() - p.min())
        rng_q = float(pq.max() - pq.min())
        range_keep = (rng_q / rng) if rng > 1e-9 else 0.0
        return keep, range_keep

    # ---------------- 全部 ----------------
    def run(self, axes: Optional[Sequence[int]] = None) -> AxisProbeReport:
        axes = list(axes) if axes is not None else list(range(self.n_axes))
        C, dirs = self.probe_orthogonality(axes)
        idx = {a: k for k, a in enumerate(axes)}

        results: List[AxisResult] = []
        for a, i in enumerate(axes):
            mono, d = self.probe_monotonicity(i)
            # 与其余轴的串扰：取该行除自身外的最大值
            row = C[a].clone()
            row[a] = 0.0
            ortho = float(row.max()) if len(axes) > 1 else 0.0
            rev = self.probe_reversibility(i)
            keep, range_keep = self.probe_low_bit_travel(i, d)
            r2 = self.probe_identifiability(i)          # 辅助，不设门线
            res = AxisResult(
                index=i, name=self.names[i], mono=mono, ortho=ortho,
                rev=rev, travel_keep=keep,
                ident_r2=r2, travel_range_keep=range_keep,
                _pass={
                    "单调": mono >= AXIS.mono_threshold,
                    "正交": ortho <= AXIS.ortho_threshold,
                    "可逆": rev >= AXIS.rev_threshold,
                    "低比特行程": keep >= AXIS.travel_threshold,
                },
            )
            results.append(res)

        return AxisProbeReport(
            n_axes=len(axes), n_sweep=self.sweep_steps, results=results,
            thresholds={"mono": AXIS.mono_threshold, "ortho": AXIS.ortho_threshold,
                        "rev": AXIS.rev_threshold,
                        "travel_keep": AXIS.travel_threshold},
        )


__all__ = ["AxisProbe", "AxisResult", "AxisProbeReport", "axis_report_text",
           "Responder"]

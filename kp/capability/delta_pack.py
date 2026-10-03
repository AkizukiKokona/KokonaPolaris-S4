"""Δ-Pack —— 子空间约束的低秩权重扰动（同域长尾能力）。

⭐ 硬条款：「禁止入侵维度」。
  · ΔW 必须建在 **W0 自身的 SVD 子空间**内（由 top-k 奇异方向张成）
  · 训练后做**谱检查**：若存在与 W0 全奇异向量 cos < 阈值 的**高排名**分量 → 判不合格
  · 门控默认 0 ⇒ 与裸模型 bit-exact（由基类 CapabilityPack 保证短路）

为什么必须这样（arXiv 2410.21228）：
  LoRA 会学到 **intruder dimensions** —— 近似正交于底模权重奇异向量的、
  全新的高排名分量。全量微调**不**产生这种分量；它们会让模型「成为预训练分布
  的更差模型」。所以判据不是「是否覆盖」，而是「是否侵入」。

规格要点：
  · **块对角分解**（RaLoRA 式）——有效秩 r 可提升为 n_l × r，参数量不变
  · **rsLoRA 缩放** α/√r（比朴素 α/r 在高秩下更稳）
  · 按 **GID** 分层、**W_o 优先**分配预算 —— 由上层调度器决定，本文件只提供算子
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CAP
from .bus import CapabilityPack


# ---------------------------------------------------------------------------
# 谱检查
# ---------------------------------------------------------------------------
@dataclass
class SpectralReport:
    """Δ-Pack 的「禁止入侵维度」检查结果。"""
    passed: bool
    min_cos_high_rank: float          # 高排名分量与 W0 左奇异子空间的最小 cos
    max_cos_high_rank: float
    energy_in_subspace: float         # 落在 W0 子空间内的能量占比
    energy_out_subspace: float        # 正交（侵入）分量能量占比
    rank_delta: int
    n_high_rank: int
    threshold: float
    target: str = ""

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "min_cos_high_rank": round(self.min_cos_high_rank, 5),
            "max_cos_high_rank": round(self.max_cos_high_rank, 5),
            "energy_in_subspace": round(self.energy_in_subspace, 5),
            "energy_out_subspace": round(self.energy_out_subspace, 5),
            "rank_delta": self.rank_delta,
            "n_high_rank": self.n_high_rank,
            "threshold": self.threshold,
            "target": self.target,
        }

    def __str__(self) -> str:  # pragma: no cover
        flag = "✅ 合格" if self.passed else "❌ 不合格（存在侵入维度）"
        return (f"谱检查[{self.target or '-'}] {flag}｜"
                f"高排名 cos_min={self.min_cos_high_rank:.4f}"
                f"（阈值 {self.threshold}）｜子空间内能量 "
                f"{self.energy_in_subspace:.1%}／正交能量 {self.energy_out_subspace:.1%}｜"
                f"ΔW rank={self.rank_delta}, 高排名分量 {self.n_high_rank} 个")


def spectral_check(delta_w: torch.Tensor, w0: torch.Tensor, *,
                   cos_threshold: Optional[float] = None,
                   rank_ratio: Optional[float] = None,
                   energy_floor: float = 0.1,
                   target: str = "") -> SpectralReport:
    """对 ΔW 做「与 W0 是否正交」的谱检查。

    判据（设计稿 v1.4「禁止入侵维度」条款）：
        取 ΔW 的奇异三元组 (uᵢ, sᵢ, vᵢ)，只看**高排名**（sᵢ ≥ energy_floor·s₀）分量；
        令 cosᵢ = ‖U₀ᵀuᵢ‖（uᵢ 与 W0 左奇异子空间的夹角余弦）。
        若存在 cosᵢ < cos_threshold 的高排名分量 → **判不合格**。

    参数
    ----
    delta_w        : (out, in) 权重扰动
    w0             : (out, in) 底模权重
    cos_threshold  : 默认取 CAP.spectral_cos_threshold（0.1）
    rank_ratio     : W0 子空间保留比例，默认 CAP.spectral_rank_ratio（0.5）
    energy_floor   : 判定「高排名」的奇异值下限（相对 s₀）
    """
    if cos_threshold is None:
        cos_threshold = CAP.spectral_cos_threshold
    if rank_ratio is None:
        rank_ratio = CAP.spectral_rank_ratio
    if delta_w.shape != w0.shape:
        raise ValueError(f"ΔW {tuple(delta_w.shape)} 与 W0 {tuple(w0.shape)} 形状不符")

    dw = delta_w.detach().to(torch.float32).cpu()
    w = w0.detach().to(torch.float32).cpu()
    m, n = w.shape

    # ⚠️ 零扰动必须**单独处理**：对 ΔW ≡ 0 求 SVD 得到的是「任意正交基 × 奇异值 0」，
    #    拿它的左奇异向量去和 W0 比 cos 纯属比较噪声（实测会给出 cos≈0.53 的假"不合格"）。
    #    语义上零扰动的「高排名分量」只有 0 个 ⇒ 不可能有侵入维度 ⇒ 判合格。
    if float((dw ** 2).sum()) <= 1e-24:
        return SpectralReport(
            passed=True, min_cos_high_rank=1.0, max_cos_high_rank=1.0,
            energy_in_subspace=1.0, energy_out_subspace=0.0,
            rank_delta=0, n_high_rank=0,
            threshold=float(cos_threshold if cos_threshold is not None
                            else CAP.spectral_cos_threshold),
            target=target,
        )

    k0 = max(1, int(rank_ratio * min(m, n)))
    U0, _, _ = torch.linalg.svd(w, full_matrices=False)   # (m, min(m,n))
    U0 = U0[:, :k0]                                        # 正交基 (m, k0)

    Ud, Sd, _ = torch.linalg.svd(dw, full_matrices=False)
    # 每个左奇异向量与 span(U0) 的 cos（U0 列正交 ⇒ 投影范数即 cos）
    cos = torch.linalg.norm(U0.transpose(0, 1) @ Ud, dim=0)     # (min(m,n),)

    if float(Sd[0]) <= 0.0:
        high = torch.zeros_like(Sd, dtype=torch.bool)
        high[0] = True
    else:
        high = Sd >= float(energy_floor) * float(Sd[0])
        if not bool(high.any()):
            high[0] = True
    cos_high = cos[high]

    # 子空间内外能量：P(dw) = U0 U0ᵀ · dw · V0 V0ᵀ
    V0 = torch.linalg.svd(w, full_matrices=False)[2].transpose(0, 1)[:, :k0]  # (n, k0)
    proj = U0 @ (U0.transpose(0, 1) @ dw @ V0) @ V0.transpose(0, 1)
    e_tot = float((dw ** 2).sum()) + 1e-12
    e_in = float((proj ** 2).sum())
    e_out = max(0.0, e_tot - e_in)

    min_cos = float(cos_high.min())
    return SpectralReport(
        passed=bool(min_cos >= cos_threshold),
        min_cos_high_rank=min_cos,
        max_cos_high_rank=float(cos_high.max()),
        energy_in_subspace=e_in / e_tot,
        energy_out_subspace=e_out / e_tot,
        rank_delta=int(torch.linalg.matrix_rank(dw).item()),
        n_high_rank=int(high.sum().item()),
        threshold=float(cos_threshold),
        target=target,
    )


# ---------------------------------------------------------------------------
# Δ-Pack
# ---------------------------------------------------------------------------
class DeltaPack(CapabilityPack):
    """子空间约束的低秩权重扰动包。

    数学形式（块对角 + rsLoRA）：
        ΔW = blockdiag( B₁A₁, …, B_blocks·A_blocks ) · (α / √r)

    参数
    ----
    name         : 包名（门控按名寻址）
    in_features  : 宿主线性层输入维
    out_features : 宿主线性层输出维
    rank         : **每块** 的有效秩 r（决定「低的有效秩」这条软肋能否被绕开）
    alpha        : rsLoRA 分子；默认 = rank ⇒ 缩放 ≈ √r
    blocks       : 块对角块数（RaLoRA 式，有效秩 → blocks×rank，参数量不变）
    seed         : 初始化随机种子
    """

    def __init__(self, name: str, in_features: int, out_features: int,
                 rank: int = 8, alpha: Optional[float] = None,
                 blocks: int = 1, gate: Optional[float] = None,
                 seed: int = 0, dtype: torch.dtype = torch.float32):
        super().__init__(name, gate)
        if in_features % blocks or out_features % blocks:
            raise ValueError(f"blocks={blocks} 无法整除 in={in_features} / out={out_features}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.blocks = int(blocks)
        self.alpha = float(rank if alpha is None else alpha)
        self.in_block = self.in_features // self.blocks
        self.out_block = self.out_features // self.blocks

        g = torch.Generator().manual_seed(int(seed))
        # A: (blocks, r, in_block)   B: (blocks, out_block, r)
        A = torch.randn(self.blocks, self.rank, self.in_block, generator=g, dtype=dtype)
        B = torch.zeros(self.blocks, self.out_block, self.rank, dtype=dtype)
        # 零初始化 B ⇒ 初始 ΔW = 0（比门控更早的一层保险）
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.meta = {"kind": "delta", "version": 1, "rank": self.rank,
                     "alpha": self.alpha, "blocks": self.blocks,
                     "subspace": None}

    # ---------------- 缩放 ----------------
    @property
    def scale(self) -> float:
        """rsLoRA 缩放 α/√r。"""
        return self.alpha / math.sqrt(max(1, self.rank))

    # ---------------- 算子 ----------------
    def delta_weight(self) -> torch.Tensor:
        """把块对角因子拼成显式 ΔW（(out, in)）。"""
        dw = torch.zeros(self.out_features, self.in_features,
                         dtype=self.A.dtype, device=self.A.device)
        for i in range(self.blocks):
            dw[i * self.out_block:(i + 1) * self.out_block,
               i * self.in_block:(i + 1) * self.in_block] = self.B[i] @ self.A[i]
        return dw * self.scale

    def delta_output(self, x: torch.Tensor) -> torch.Tensor:
        """对输出的贡献（未乘门控）：走两次低秩线性，比重建满矩阵更省。"""
        outs = []
        for i in range(self.blocks):
            xi = x[..., i * self.in_block:(i + 1) * self.in_block]
            outs.append(F.linear(F.linear(xi, self.A[i]), self.B[i]))
        return torch.cat(outs, dim=-1) * self.scale

    # ---------------- 子空间初始化 ----------------
    @torch.no_grad()
    def init_in_subspace(self, w0: torch.Tensor, *, span_mult: int = 1,
                         seed: Optional[int] = None) -> float:
        """把 ΔW 初始化到 **W0 每个对角块自身的 top-k 奇异子空间内**。

        ⚠️ 这里必须**逐对角块**做：块对角 ΔW 的「W0 子空间」是**分块**子空间，
        用整矩阵的全局奇异向量会错位（曾导致 54% 的假阳性正交能量）。

        返回：因子重建 ΔW 与子空间内随机目标的最大相对误差（诊断；
              span_mult=1 时为精确秩-r 分解，误差应 ≈ 0）。
        """
        if w0.shape != (self.out_features, self.in_features):
            raise ValueError(f"w0 {tuple(w0.shape)} 与包形状 "
                             f"({self.out_features}, {self.in_features}) 不符")
        w = w0.detach().to(self.A.dtype)
        g = torch.Generator().manual_seed(0 if seed is None else int(seed))
        errs = []

        for i in range(self.blocks):
            ob, ib = self.out_block, self.in_block
            wb = w[i * ob:(i + 1) * ob, i * ib:(i + 1) * ib]
            k = min(max(self.rank * span_mult, self.rank), min(wb.shape))
            U, _, Vh = torch.linalg.svd(wb, full_matrices=False)
            U, Vh = U[:, :k], Vh[:k]
            C = torch.randn(k, k, generator=g, dtype=self.A.dtype)
            target = U @ C @ Vh                       # ★ 完全落在该块的 top-k 子空间内
            U2, S2, Vh2 = torch.linalg.svd(target, full_matrices=False)
            r = self.rank
            Bk = U2[:, :r] * S2[:r].unsqueeze(0)      # (ob, r)
            Ak = Vh2[:r]                              # (r, ib)
            self.B[i].copy_(Bk.to(self.B.dtype))
            self.A[i].copy_(Ak.to(self.A.dtype))
            recon = Bk @ Ak
            errs.append(float((recon - target).norm() / (target.norm() + 1e-12)))
        self.meta["subspace"] = {"span_mult": span_mult, "seed": seed}
        return max(errs)

    # ---------------- 谱检查 ----------------
    def spectral_report(self, w0: torch.Tensor) -> SpectralReport:
        """**逐对角块**做谱检查并聚合（块对角 ΔW 的正确检查方式）。"""
        w = w0.detach().to(self.A.dtype)
        reps = []
        for i in range(self.blocks):
            ob, ib = self.out_block, self.in_block
            dw_i = (self.B[i] @ self.A[i]) * self.scale
            w_i = w[i * ob:(i + 1) * ob, i * ib:(i + 1) * ib]
            reps.append(spectral_check(dw_i, w_i, target=f"{self.name}#block{i}"))
        e_in = sum(r.energy_in_subspace for r in reps) / len(reps)
        return SpectralReport(
            passed=all(r.passed for r in reps),
            min_cos_high_rank=min(r.min_cos_high_rank for r in reps),
            max_cos_high_rank=max(r.max_cos_high_rank for r in reps),
            energy_in_subspace=e_in,
            energy_out_subspace=1.0 - e_in,
            rank_delta=sum(r.rank_delta for r in reps),
            n_high_rank=sum(r.n_high_rank for r in reps),
            threshold=reps[0].threshold,
            target=f"{self.name}[blocks={self.blocks},r={self.rank}]",
        )

    # ---------------- 序列化 ----------------
    def to_spec(self, target: str) -> dict:
        return {
            "name": self.name, "kind": "delta", "target": target,
            "state": {"A": self.A.detach().cpu(), "B": self.B.detach().cpu()},
            "meta": {**self.meta, "gate": float(self.gate.detach())},
        }

    @classmethod
    def from_spec(cls, w0: torch.Tensor, spec: Dict) -> "DeltaPack":
        """从加载载荷重建。支持两种载荷：
            ① 显式因子 {"state": {"A":…, "B":…}}
            ② 子空间初始化 {"state": {"init": {"span_mult":…, "seed":…}}}
        """
        meta = spec.get("meta", {})
        state = spec.get("state", {})
        out_features, in_features = w0.shape
        pack = cls(
            name=spec.get("name", "delta"),
            in_features=in_features, out_features=out_features,
            rank=int(meta.get("rank", 8)),
            alpha=meta.get("alpha", None),
            blocks=int(meta.get("blocks", 1)),
            gate=meta.get("gate", None),
            seed=int(meta.get("seed", 0)),
            dtype=w0.dtype,
        )
        if "A" in state and "B" in state:
            with torch.no_grad():
                pack.A.copy_(state["A"].to(pack.A.dtype))
                pack.B.copy_(state["B"].to(pack.B.dtype))
        elif "init" in state:
            cfg = state["init"] or {}
            pack.init_in_subspace(w0, span_mult=int(cfg.get("span_mult", 2)),
                                  seed=cfg.get("seed", None))
        else:
            raise ValueError("DeltaPack 载荷缺少 'A'/'B' 或 'init'")
        pack.meta.update({k: v for k, v in meta.items() if k != "gate"})
        return pack

    def describe(self) -> dict:
        d = super().describe()
        d.update({"rank": self.rank, "blocks": self.blocks, "alpha": self.alpha,
                  "params": int(self.A.numel() + self.B.numel())})
        return d


__all__ = ["DeltaPack", "spectral_check", "SpectralReport"]

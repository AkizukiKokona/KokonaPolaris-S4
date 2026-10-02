"""SVD 子空间约束能力包（LoRA-X 式）—— `ΔW = Ũ · ΔΣ · Ṽᵀ`。

⭐ 与 Δ-Pack 的**本质区别**：

| | Δ-Pack | **SVDPack** |
|---|---|---|
| ΔW 从哪来 | 训练出的低秩因子 `B·A`（自由） | **左右奇空间钉死在 W0 的 SVD 基上** |
| 可训参数 | `A`(r,in) + `B`(out,r) | 只有对角强度 `ΔΣ`(r) |
| 入侵维度 | **训练后**才做谱检查，可能不合格 | **按构造不可能产生**，谱检查必然通过 |
| 可迁移性 | 依赖 W0 的基（隐含） | Ũ/Ṽ **被存储**（不从 W0 重算）⇒ **跨自家版本直接可用** |

为什么能跨版本：设计稿要的不是「跨 SDXL/Flux」，而是
**跨我们自己演进中的主干版本**。只要把基显式存下来，换版本时基不变、ΔW 直接可用
—— 这就把「能力包可迁移」变成了**零成本的默认属性**，而不是需要额外工程去争取的东西。

代价：参数不是 r×(in+out) 而是 r + r×(in+out)（多存一个基）。
       **这笔开销正是"可迁移 + 无入侵维度"的保费**，我们认为值。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .bus import CapabilityPack
from .delta_pack import SpectralReport, spectral_check


class SVDPack(CapabilityPack):
    """子空间约束包：`ΔW = U · diag(σ) · Vᵀ`，U/V 固定为 W0 的 SVD 基。"""

    def __init__(self, name: str, in_features: int, out_features: int,
                 rank: int = 8, band: Optional[Tuple[int, int]] = None,
                 gate: Optional[float] = None, dtype: torch.dtype = torch.float32):
        super().__init__(name, gate)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.band = band
        # 基：**存储**（不是每次从 W0 重算）⇒ 跨主干版本可迁移
        self.register_buffer("U", torch.zeros(self.out_features, self.rank, dtype=dtype))
        self.register_buffer("V", torch.zeros(self.in_features, self.rank, dtype=dtype))
        # 唯一可训参数：对角强度 ΔΣ。零初始化 ⇒ 初始 ΔW ≡ 0
        self.sigma = nn.Parameter(torch.zeros(self.rank, dtype=dtype))
        self.meta = {"kind": "svd", "version": 1, "rank": self.rank,
                     "band": list(band) if band else None}

    # ---------------- 构造 ----------------
    @classmethod
    def init_from(cls, name: str, w0: torch.Tensor, rank: int = 8,
                  band: Optional[Tuple[int, int]] = None,
                  gate: Optional[float] = None) -> "SVDPack":
        """从 W0 的 SVD 取基。band=(lo,hi) 可指定奇异值带（默认取最强的前 rank）。"""
        p = cls(name, w0.shape[1], w0.shape[0], rank, band, gate, w0.dtype)
        p.set_basis(w0)
        return p

    @torch.no_grad()
    def set_basis(self, w0: torch.Tensor) -> torch.Tensor:
        """重设基（换版本时若想重新对齐 W0 可调用）。返回选中的奇异值索引。"""
        w = w0.detach().to(torch.float32)
        U, _, Vh = torch.linalg.svd(w, full_matrices=False)
        n = U.shape[1]
        lo, hi = (0, self.rank) if self.band is None else self.band
        hi = min(hi, n)
        if hi - lo < self.rank:
            lo, hi = 0, min(self.rank, n)
        idx = torch.arange(lo, hi)[: self.rank]
        if idx.numel() < self.rank:                      # 维数不足时右侧补零列
            idx = torch.cat([idx, torch.full((self.rank - idx.numel(),), idx[-1])])
        self.U.copy_(U[:, idx].to(self.U.dtype))
        self.V.copy_(Vh[idx].transpose(0, 1).to(self.V.dtype))
        return idx

    # ---------------- 算子 ----------------
    def delta_weight(self) -> torch.Tensor:
        return (self.U * self.sigma.unsqueeze(0)) @ self.V.transpose(0, 1)

    def delta_output(self, x: torch.Tensor) -> torch.Tensor:
        h = (x @ self.V) * self.sigma.to(x.dtype)        # (..., r)
        return h @ self.U.transpose(0, 1)                # (..., out)

    @property
    def trainable_params(self) -> int:
        return self.rank

    @property
    def stored_params(self) -> int:
        return self.U.numel() + self.V.numel() + self.sigma.numel()

    # ---------------- 谱检查（按构造必然通过） ----------------
    def spectral_report(self, w0: torch.Tensor) -> SpectralReport:
        return spectral_check(self.delta_weight(), w0,
                              target=f"{self.name}[svd,r={self.rank}]")

    # ---------------- 序列化 ----------------
    def to_spec(self, target: str) -> dict:
        return {"name": self.name, "kind": "svd", "target": target,
                "state": {"U": self.U.detach().cpu(), "V": self.V.detach().cpu(),
                          "sigma": self.sigma.detach().cpu()},
                "meta": {**self.meta, "gate": float(self.gate.detach())}}

    @classmethod
    def from_spec(cls, w0: torch.Tensor, spec: Dict) -> "SVDPack":
        """⚠️ **不从 w0 重算基** —— 基来自载荷本身，这正是跨版本可迁移的关键。"""
        meta = spec.get("meta", {})
        st = spec.get("state", {})
        out_features, in_features = w0.shape
        p = cls(spec.get("name", "svd"), in_features, out_features,
                rank=int(meta.get("rank", 8)),
                band=tuple(meta["band"]) if meta.get("band") else None,
                gate=meta.get("gate", None), dtype=w0.dtype)
        with torch.no_grad():
            p.U.copy_(st["U"].to(p.U.dtype))
            p.V.copy_(st["V"].to(p.V.dtype))
            p.sigma.copy_(st["sigma"].to(p.sigma.dtype))
        return p

    def describe(self) -> dict:
        d = super().describe()
        d.update({"rank": self.rank, "band": self.band,
                  "trainable_params": self.trainable_params,
                  "stored_params": self.stored_params,
                  "portable_across_versions": True})
        return d


__all__ = ["SVDPack"]

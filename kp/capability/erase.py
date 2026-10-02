"""擦除算子 E 与可逆账本 —— 「加能力 = 减能力取反」。

核心恒等式：
        Pack_recover ≡ E⁻¹

把合规抑制实现为**显式低秩算子** `E = Σ αᵢ uᵢ vᵢᵀ`，并**记账**
{基 / 有效秩 / 作用层 / 强度 / 保留集边界}，则可以闭式取反 → 秒级恢复，
无训练、无新数据。

为什么成立（设计稿 §4.7.10）：
  · 擦除多为「引导式回避」而非销毁（arXiv 2505.17013，可被向轨迹注入噪声捞回）
  · 安全微调本质是**近正交 ΔW + 零空间投影**（滤波器，不是重学）
  · 安全子空间**有效秩 k≈6**（LoX）⇒ 低秩假设站得住
  · 闭式擦除 22 秒（CURE, NeurIPS 2025）
  · ⭐ 因为数据是我们自己过滤的 → **我们知道擦了什么**（社区做不到）

作用层选择：放在**语义已分化**的深度之后（经验 ~2/3 处，见 MACE 的 SSB 观察）。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 擦除算子
# ---------------------------------------------------------------------------
class EraseOperator:
    """低秩擦除算子 `E = U · diag(α) · Vᵀ`（作用在一个权重矩阵上）。

        W_erased  = W − E          （apply）
        W_back    = W_erased + E   （recover ≡ E⁻¹）

    在精确算术下 `recover(apply(W)) == W`；浮点下误差在 eps 量级
    （由 `erasure_roundtrip` 的 max_abs_diff 度量）。
    """

    def __init__(self, name: str, layer: str, U: torch.Tensor, V: torch.Tensor,
                 alpha: torch.Tensor, retained: Optional[list] = None,
                 meta: Optional[dict] = None):
        if U.dim() != 2 or V.dim() != 2:
            raise ValueError("U / V 必须是 (dim, k) 的二维张量")
        if U.shape[1] != V.shape[1]:
            raise ValueError(f"U/V 的有效秩不一致：{U.shape[1]} vs {V.shape[1]}")
        self.name = name
        self.layer = layer
        self.U = U.detach().to(torch.float32).cpu()
        self.V = V.detach().to(torch.float32).cpu()
        self.alpha = alpha.detach().reshape(-1).to(torch.float32).cpu()
        if self.alpha.numel() != self.U.shape[1]:
            raise ValueError("alpha 长度必须等于有效秩 k")
        self.retained = retained or []          # 保留集边界（不该被擦到的方向）
        self.meta = meta or {}

    # ---------------- 基本属性 ----------------
    @property
    def rank(self) -> int:
        return int(self.U.shape[1])

    @property
    def shape(self) -> tuple:
        return (int(self.U.shape[0]), int(self.V.shape[0]))

    def delta_w(self) -> torch.Tensor:
        """显式 E（(out, in)）。"""
        return (self.U * self.alpha.unsqueeze(0)) @ self.V.transpose(0, 1)

    @property
    def strength(self) -> float:
        return float(self.delta_w().norm().item())

    # ---------------- 作用 ----------------
    def apply(self, w: torch.Tensor) -> torch.Tensor:
        """擦除：W ↦ W − E。"""
        return w - self.delta_w().to(w.dtype)

    def recover(self, w: torch.Tensor) -> torch.Tensor:
        """恢复 ≡ E⁻¹：W ↦ W + E。"""
        return w + self.delta_w().to(w.dtype)

    def apply_output(self, x: torch.Tensor) -> torch.Tensor:
        """激活空间擦除：y ↦ y − U diag(α) (Vᵀ x)。"""
        U = self.U.to(x.dtype)
        V = self.V.to(x.dtype)
        a = self.alpha.to(x.dtype)
        return x - (U * a.unsqueeze(0)) @ (V.transpose(0, 1) @ x)

    # ---------------- 构造 ----------------
    @classmethod
    def from_directions(cls, name: str, layer: str,
                        direction_out: torch.Tensor, direction_in: torch.Tensor,
                        strength: float = 1.0,
                        retained: Optional[list] = None) -> "EraseOperator":
        """由单个方向对 (u, v) 构造 rank-1 擦除。"""
        u = F.normalize(direction_out.reshape(-1, 1).float(), dim=0)
        v = F.normalize(direction_in.reshape(-1, 1).float(), dim=0)
        return cls(name, layer, u, v, torch.tensor([float(strength)]), retained)

    # ---------------- 序列化 ----------------
    def as_dict(self) -> dict:
        return {
            "name": self.name, "layer": self.layer,
            "rank": self.rank, "shape": list(self.shape), "strength": self.strength,
            "U": self.U, "V": self.V, "alpha": self.alpha,
            "retained": self.retained, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EraseOperator":
        return cls(d["name"], d["layer"], d["U"], d["V"], d["alpha"],
                   d.get("retained"), d.get("meta"))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EraseOperator({self.name!r}, layer={self.layer!r}, "
                f"rank={self.rank}, shape={self.shape}, strength={self.strength:.4g})")


# ---------------------------------------------------------------------------
# 账本
# ---------------------------------------------------------------------------
@dataclass
class EraseLedger:
    """擦除账本：所有已施加擦除的**唯一真源**（缺了它就无法「留门」）。"""
    entries: List[dict] = field(default_factory=list)
    version: int = 1

    def add(self, op: EraseOperator, *, stage: str = "", note: str = "") -> "EraseLedger":
        self.entries.append({**op.as_dict(), "stage": stage, "note": note})
        return self

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> List[dict]:
        return [{"name": e["name"], "layer": e["layer"], "rank": e["rank"],
                 "shape": e["shape"], "strength": round(float(e["strength"]), 5),
                 "stage": e.get("stage", ""), "retained": e.get("retained", [])}
                for e in self.entries]

    def operators(self) -> List[EraseOperator]:
        return [EraseOperator.from_dict(e) for e in self.entries]

    def total_rank(self) -> int:
        return sum(int(e["rank"]) for e in self.entries)

    # 保存用 torch（张量友好）；同时给一份可读 json 概览
    def save(self, path: str) -> None:
        torch.save({"ledger": self.entries, "version": self.version}, path)
        meta_path = os.path.splitext(path)[0] + ".summary.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"version": self.version, "entries": self.summary()},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "EraseLedger":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(entries=list(blob.get("ledger", [])),
                   version=int(blob.get("version", 1)))

    def __repr__(self) -> str:  # pragma: no cover
        return f"EraseLedger({len(self)} 条，总有效秩 {self.total_rank()})"


# ---------------------------------------------------------------------------
# KL 检验
# ---------------------------------------------------------------------------
def kl_test(p: torch.Tensor, q: torch.Tensor, *, logits: bool = True,
            eps: float = 1e-12) -> float:
    """KL(p ‖ q)。p,q 为同形状的 logits（默认）或概率。

    用于擦除可逆性判据：`E⁻¹∘E` 后输出分布应当回到基线（KL < 阈值）。
    """
    p = p.detach().to(torch.float32)
    q = q.detach().to(torch.float32)
    if logits:
        lp = F.log_softmax(p, dim=-1)
        lq = F.log_softmax(q, dim=-1)
    else:
        lp = torch.log(p.clamp_min(eps))
        lq = torch.log(q.clamp_min(eps))
    pp = lp.exp()
    return float((pp * (lp - lq)).sum(dim=-1).mean().item())


def erasure_roundtrip(op: EraseOperator, forward_fn: Callable,
                      w0: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
    """`E⁻¹∘E` 往返测试。

    forward_fn(w, x) → logits。返回 {kl, max_abs_diff, w_roundtrip_err}。
      · kl               : 擦除前 vs 往返后 的输出分布 KL（越小越可逆）
      · max_abs_diff     : 前两者的逐元素最大差
      · w_roundtrip_err  : `recover(apply(W)) − W` 的最大差（纯数值误差上界）
    """
    w_erased = op.apply(w0)
    w_back = op.recover(w_erased)
    y0 = forward_fn(w0, x)
    y1 = forward_fn(w_back, x)
    return {
        "kl": kl_test(y0, y1),
        "max_abs_diff": float((y0 - y1).abs().max().item()),
        "w_roundtrip_err": float((w_back - w0).abs().max().item()),
    }


__all__ = ["EraseOperator", "EraseLedger", "kl_test", "erasure_roundtrip"]

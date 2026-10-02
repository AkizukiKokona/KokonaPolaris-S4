"""∥-Pack —— 并联/旁路能力包（跨域能力：写实、文字渲染等）。

与 Δ-Pack 的**根本区别**：
  · 它不是「对 W0 的扰动」，而是一个**独立的小网络**，输出被加到主干输出上。
  · 因此它**物理上不可能覆盖/替换底模权重** → 天然「不打架」。
  · 代价：它承担跨域能力（体量大两个数量级），需**天级**重训。

设计律对应：「能力越大越域外，越不能住在 ΔW 里。」

安全出口（三层）：
  ① 上投影**零初始化** ⇒ 初始输出严格为 0；
  ② 门控默认 0 ⇒ 门控为 0 时整条旁路被短路（基类 CapabilityPack 保证 bit-exact）；
  ③ `delta_weight()` 返回 None —— 明确声明「本包不可表示为权重扰动」，
     挂载时不会被误当作 ΔW 处理。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bus import CapabilityPack


class ParallelPack(CapabilityPack):
    """并联旁路网络（bottleneck MLP）。

        y += g · Up( act( Down(x) ) )

    参数
    ----
    in_features / out_features : 宿主线性层的输入/输出维
    hidden                     : 瓶颈宽度（跨域能力通常远大于 Δ-Pack）
    nonlinear                  : 是否插入非线性（线性瓶颈等价于一个低秩 ΔW）
    act                        : 非线性类型，'gelu' | 'silu' | 'relu'
    """

    def __init__(self, name: str, in_features: int, out_features: int,
                 hidden: int = 512, nonlinear: bool = True, act: str = "gelu",
                 gate: Optional[float] = None, seed: int = 0,
                 dtype: torch.dtype = torch.float32):
        super().__init__(name, gate)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden = int(hidden)
        self.nonlinear = bool(nonlinear)
        self.act_name = act

        g = torch.Generator().manual_seed(int(seed))
        self.down = nn.Linear(in_features, hidden, bias=True, dtype=dtype)
        self.up = nn.Linear(hidden, out_features, bias=True, dtype=dtype)
        with torch.no_grad():
            # Kaiming 初始化 down；★ up 零初始化（+ 零 bias）⇒ 初始输出为 0
            nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5, generator=g)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        self.meta = {"kind": "parallel", "version": 1, "hidden": self.hidden,
                     "nonlinear": self.nonlinear, "act": self.act_name}

    # ---------------- 算子 ----------------
    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if not self.nonlinear:
            return x
        if self.act_name == "gelu":
            return F.gelu(x)
        if self.act_name == "silu":
            return F.silu(x)
        if self.act_name == "relu":
            return F.relu(x)
        raise ValueError(f"未知激活 {self.act_name!r}")

    def delta_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self._act(self.down(x)))

    def delta_weight(self) -> Optional[torch.Tensor]:
        """★ 明确返回 None：并联包**不可**表示为权重扰动（这正是它不打架的原因）。"""
        return None

    # ---------------- 序列化 ----------------
    def to_spec(self, target: str) -> dict:
        return {
            "name": self.name, "kind": "parallel", "target": target,
            "state": {"down.weight": self.down.weight.detach().cpu(),
                      "down.bias": self.down.bias.detach().cpu(),
                      "up.weight": self.up.weight.detach().cpu(),
                      "up.bias": self.up.bias.detach().cpu()},
            "meta": {**self.meta, "gate": float(self.gate.detach())},
        }

    @classmethod
    def from_spec(cls, in_features: int, out_features: int, spec: Dict) -> "ParallelPack":
        meta = spec.get("meta", {})
        state = spec.get("state", {})
        pack = cls(
            name=spec.get("name", "parallel"),
            in_features=in_features, out_features=out_features,
            hidden=int(meta.get("hidden", 512)),
            nonlinear=bool(meta.get("nonlinear", True)),
            act=meta.get("act", "gelu"),
            gate=meta.get("gate", None),
            seed=int(meta.get("seed", 0)),
        )
        with torch.no_grad():
            pack.down.weight.copy_(state["down.weight"].to(pack.down.weight.dtype))
            pack.down.bias.copy_(state["down.bias"].to(pack.down.bias.dtype))
            pack.up.weight.copy_(state["up.weight"].to(pack.up.weight.dtype))
            pack.up.bias.copy_(state["up.bias"].to(pack.up.bias.dtype))
        return pack

    def describe(self) -> dict:
        d = super().describe()
        d.update({"hidden": self.hidden, "nonlinear": self.nonlinear,
                  "params": int(sum(p.numel() for p in self.parameters()))})
        return d


__all__ = ["ParallelPack"]

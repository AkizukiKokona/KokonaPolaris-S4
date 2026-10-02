"""能力总线：冻结主干 + 可插拔能力包 + 门控。

★ 核心不变量（验收基线）：**全部门控 = 0 时，输出必须与裸模型 bit-exact。**

实现方式不是"乘 0 再相加"，而是**整条旁路被短路跳过** —— 因为
`y + 0*x` 在 IEEE 下虽等于 y，但仍会引入一次额外的舍入与一次算子调度；
把分支彻底跳掉才能保证**逐位相同**，也才能让"接口成本严格为零"成立。

其余两条硬约束：
  · 门控全程 fp32，不参与量化
  · **注入点必须绕过 NVFP4 量化器**（否则 ΔW 会先被量化再相加，
    既破坏可逆性、也让 bit-exact 失效）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CAP


# ---------------------------------------------------------------------------
# 能力包协议
# ---------------------------------------------------------------------------
class CapabilityPack(nn.Module, ABC):
    """一个能力包 = {ΔW, g, 元数据}。"""

    def __init__(self, name: str, gate: float = None):
        super().__init__()
        self.name = name
        self.gate = nn.Parameter(
            torch.tensor(float(CAP.gate_init if gate is None else gate),
                         dtype=torch.float32),
            requires_grad=False,
        )
        self.meta: dict = {"kind": "abstract", "version": 1}

    # --- 子类实现 ---
    @abstractmethod
    def delta_output(self, x: torch.Tensor) -> torch.Tensor:
        """包本身对输出空间的贡献（未乘门控）。"""

    @abstractmethod
    def delta_weight(self) -> Optional[torch.Tensor]:
        """等价的 ΔW（若不可显式表示则返回 None，例如纯旁路网络）。"""

    # --- 通用 ---
    @property
    def is_off(self) -> bool:
        return float(self.gate.detach()) == 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_off:                      # ★ 短路，不参与任何计算
            return torch.zeros_like(x)
        return self.gate.to(x.dtype) * self.delta_output(x)

    def describe(self) -> dict:
        return {**self.meta, "name": self.name, "gate": float(self.gate.detach())}


def all_gates_zero(packs: Iterable[CapabilityPack]) -> bool:
    return all(p.is_off for p in packs)


# ---------------------------------------------------------------------------
# 门控线性层
# ---------------------------------------------------------------------------
class GatedLinear(nn.Module):
    """冻结 W0 的线性层 + 任意数量能力包。

    门控全 0 时走**原始 F.linear 路径**（与裸模型逐位相同）。
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None,
                 quantized: bool = False):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias",
                             None if bias is None else bias.detach().clone())
        # 标志：本体是否被 NVFP4 量化（仅作档案，注入点永远在量化器之外）
        self.base_quantized = quantized
        self.quant = None            # QuantSpec | None（None = 不量化）
        self.packs = nn.ModuleList()
        self._bypass_off = True

    # --- 量化（⚠️ 注入点始终在量化器之外） ---
    def set_quant(self, spec) -> "GatedLinear":
        """设置量化规格（`None` = 不量化）。返回 self 便于链式调用。

        ⭐ 量化只作用于**主干算子**：`y = F.linear(q(x), q(W)) + Σ pack(x)`。
        能力包看到的是**未量化**的激活，其输出也**不被量化** ——
        否则 ΔW 会先被量化再相加，既破坏可逆性、也让 bit-exact 失效。
        """
        self.quant = spec
        return self

    # --- 装配 ---
    def add_pack(self, pack: CapabilityPack) -> "GatedLinear":
        if pack.delta_weight() is not None:
            dw = pack.delta_weight()
            if dw.shape != self.weight.shape:
                raise ValueError(
                    f"ΔW 形状 {tuple(dw.shape)} 与 W0 {tuple(self.weight.shape)} 不符")
        self.packs.append(pack)
        self.sync()
        return self

    def sync(self) -> None:
        """重新计算"短路开关"。任何 gate 变动后必须调用。"""
        self._bypass_off = all_gates_zero(self.packs)

    def set_gate(self, name: str, value: float) -> None:
        for p in self.packs:
            if p.name == name:
                p.gate.data.fill_(float(value))
        self.sync()

    def gates(self) -> dict:
        return {p.name: float(p.gate.detach()) for p in self.packs}

    # --- 前向 ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant is None:
            if self._bypass_off:                     # ★ bit-exact 路径
                return F.linear(x, self.weight, self.bias)
            y = F.linear(x, self.weight, self.bias)  # 主干
        else:                                        # 量化主干（模拟量化 + STE）
            y = F.linear(self.quant.quantize_act(x),
                         self.quant.quantize_weight(self.weight), self.bias)
        for p in self.packs:                         # 注入点：量化器之外
            if not p.is_off:
                y = y + p(x)                         # ★ 包看到的是未量化的 x
        return y

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"packs={[p.name for p in self.packs]}, bypass_off={self._bypass_off}, "
                f"quant={None if self.quant is None else self.quant.weight + '/' + self.quant.act}")


# ---------------------------------------------------------------------------
# 总线
# ---------------------------------------------------------------------------
class CapabilityBus:
    """管理一整套 GatedLinear，提供批量开关与统一入口。"""

    def __init__(self, layers: Sequence[GatedLinear] = ()):
        self.layers: List[GatedLinear] = list(layers)

    def register(self, layer: GatedLinear) -> GatedLinear:
        self.layers.append(layer)
        return layer

    def all_off(self) -> None:
        for L in self.layers:
            for p in L.packs:
                p.gate.data.fill_(0.0)
            L.sync()

    def all_on(self) -> None:
        for L in self.layers:
            for p in L.packs:
                if float(p.gate.detach()) == 0.0:
                    p.gate.data.fill_(1.0)
            L.sync()

    def set_gate(self, name: str, value: float) -> None:
        for L in self.layers:
            L.set_gate(name, value)

    def inventory(self) -> List[dict]:
        out = []
        for i, L in enumerate(self.layers):
            for p in L.packs:
                out.append({"layer": i, **p.describe()})
        return out

    def offload_cpu(self) -> None:
        """低显存模式：把全部门控为 0 的包移到 CPU（账本不占运行时开销）。"""
        for L in self.layers:
            for p in L.packs:
                if p.is_off:
                    p.to("cpu")


# ---------------------------------------------------------------------------
# 统一加载入口（对外四接口之一）
# ---------------------------------------------------------------------------
def load_adapter(model: nn.Module, path: str, *, gate: float = None,
                 strict: bool = True) -> List[CapabilityPack]:
    """统一 `load_adapter()`：从磁盘加载能力包并挂到匹配的层上。

    支持的载荷（`.pt`，torch.save 的 dict）：
        {"packs": [{"name":..., "kind": "delta"|"parallel", "target": "blocks.0.mlp.fc1",
                    "state": {...}, "meta": {...}}, ...]}

    返回被挂载的包列表。`gate=None` 表示沿用文件里记录的 gate（默认应为 0）。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "packs" not in blob:
        raise ValueError(f"{path} 里没有 'packs' 字段")

    named = dict(model.named_modules())
    mounted: List[CapabilityPack] = []
    for spec in blob["packs"]:
        target = spec["target"]
        if target not in named:
            if strict:
                raise KeyError(f"模型里找不到目标层 {target!r}")
            continue
        host = named[target]
        if not isinstance(host, GatedLinear):
            if strict:
                raise TypeError(f"{target} 不是 GatedLinear，而是 {type(host).__name__}")
            continue

        kind = spec.get("kind", "delta")
        if kind == "delta":
            from .delta_pack import DeltaPack
            pack = DeltaPack.from_spec(host.weight, spec)
        elif kind == "parallel":
            from .parallel_pack import ParallelPack
            pack = ParallelPack.from_spec(host.in_features, host.out_features, spec)
        else:
            raise ValueError(f"未知能力包类型 {kind!r}")

        if gate is not None:
            pack.gate.data.fill_(float(gate))
        host.add_pack(pack)
        mounted.append(pack)
    return mounted

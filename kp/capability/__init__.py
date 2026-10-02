"""kp.capability —— 可插拔能力总线（∥-Pack / Δ-Pack / 擦除算子 E）"""
from .bus import (  # noqa: F401
    CapabilityPack,
    GatedLinear,
    CapabilityBus,
    load_adapter,
    all_gates_zero,
)
from .delta_pack import DeltaPack, spectral_check, SpectralReport  # noqa: F401
from .parallel_pack import ParallelPack  # noqa: F401
from .erase import EraseOperator, EraseLedger, kl_test  # noqa: F401

__all__ = [
    "CapabilityPack", "GatedLinear", "CapabilityBus", "load_adapter",
    "all_gates_zero", "DeltaPack", "spectral_check", "SpectralReport",
    "ParallelPack", "EraseOperator", "EraseLedger", "kl_test",
]

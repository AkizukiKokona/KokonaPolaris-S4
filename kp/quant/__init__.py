"""kp.quant —— NVFP4 原生量化（模拟量化 + STE，可微）。"""
from .nvfp4 import (  # noqa: F401
    quant_fp4,
    quant_fp8,
    quant_fp4_ste,
    quant_fp8_ste,
    relative_error,
    QuantSpec,
    DESIGN_SPEC,
    OFFICIAL_DEFAULT_SPEC,
    precision_table,
    DEFAULT_BLOCK,
)

__all__ = [
    "quant_fp4", "quant_fp8", "quant_fp4_ste", "quant_fp8_ste", "relative_error",
    "QuantSpec", "DESIGN_SPEC", "OFFICIAL_DEFAULT_SPEC", "precision_table",
    "DEFAULT_BLOCK",
]

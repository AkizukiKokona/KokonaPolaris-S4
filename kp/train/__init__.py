"""kp.train —— 训练与 QAD（量化感知）路径。"""
from .qad import (  # noqa: F401
    set_quant,
    clear_quant,
    freeze_backbone,
    attach_delta,
    budget,
    budget_projection,
    run_qad,
    grad_health,
    iter_gated,
    QADResult,
    SKIP_DEFAULT,
)

__all__ = ["set_quant", "clear_quant", "freeze_backbone", "attach_delta", "budget",
           "budget_projection", "run_qad", "grad_health", "iter_gated",
           "QADResult", "SKIP_DEFAULT"]

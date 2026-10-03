"""kp.train —— 训练路径：QAD（量化感知）与 Character Fitter。"""
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
from .fitter import (  # noqa: F401
    fitter_loss,
    fitter_budget,
    train_fitter,
    train,
    run_closed_loop,
    shuffle_positives,
    holdout_invariance,
    set_dropout,
    FitResult,
    ClosedLoopResult,
)

__all__ = ["set_quant", "clear_quant", "freeze_backbone", "attach_delta", "budget",
           "budget_projection", "run_qad", "grad_health", "iter_gated",
           "QADResult", "SKIP_DEFAULT",
           "fitter_loss", "fitter_budget", "train_fitter", "train",
           "run_closed_loop", "shuffle_positives", "holdout_invariance", "set_dropout",
           "FitResult", "ClosedLoopResult"]

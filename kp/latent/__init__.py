"""kp.latent —— 混合 latent（32× 空间压缩，8ch 语义 + 32ch 细节）

含 **G2 通道分离监督件与验收装置**（`separation.py`）：
    设计稿硬要求「通道分离必须显式监督」，否则主干会把两个通道都塞满
    ⇒ 画风解耦失效 ⇒ L1（~90% 调用量）全盘作废。G2 是**结构性门**。
"""
from .hybrid import (  # noqa: F401
    pack_latent,
    unpack_latent,
    split_channels,
    join_channels,
    HybridLatentShape,
    check_shapes,
    channel_mi_penalty,
    swap_channels,
)
from .separation import (  # noqa: F401
    synthetic_batch,
    structural_metric,
    texture_metric,
    SeparationModel,
    SeparationReport,
    shuffle_within_batch,
    cross_perturb,
    recon_loss,
    train_separation,
    branch_dependency,
    evaluate,
    run_g2,
)

__all__ = [
    "pack_latent", "unpack_latent", "split_channels", "join_channels",
    "HybridLatentShape", "check_shapes", "channel_mi_penalty", "swap_channels",
    "synthetic_batch", "structural_metric", "texture_metric", "SeparationModel",
    "SeparationReport", "shuffle_within_batch", "cross_perturb", "recon_loss",
    "train_separation", "branch_dependency", "evaluate", "run_g2",
]

"""kp.latent —— 混合 latent（32× 空间压缩，8ch 语义 + 32ch 细节）"""
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

__all__ = [
    "pack_latent", "unpack_latent", "split_channels", "join_channels",
    "HybridLatentShape", "check_shapes", "channel_mi_penalty", "swap_channels",
]

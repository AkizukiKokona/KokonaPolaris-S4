"""kp.character —— 角色卡数据对象 + Character Fitter + 多视角配对 + 数据加载器。

    CharacterCard   身份载体（2.5D 分层 + 身份 token，跨骨干版本通用）
    CharacterFitter 多视角 → 身份 token（一次性训练；换角色 = 前向一次，秒级）
    pairing         多视角**声明式**配对（同角色 + 只动一个旋钮）
    dataset         配对 → Fitter 训练批（真实批次 / 合成可控信号）
"""
from .card import CharacterCard, SEMANTIC_LAYERS, N_LAYERS, CARD_FORMAT_VERSION  # noqa: F401
from .fitter import CharacterFitter, ViewEncoder  # noqa: F401
from .pairing import (  # noqa: F401
    AXES, Record, PairSpec, Pair, PairReport,
    build_pairs, format_pair_report,
)
from .dataset import (  # noqa: F401
    ViewSample, FitPair, PairViewLoader, format_dataset_report, load_image,
    DEFAULT_SIZE,
)

__all__ = ["CharacterCard", "SEMANTIC_LAYERS", "N_LAYERS", "CARD_FORMAT_VERSION",
           "CharacterFitter", "ViewEncoder",
           "AXES", "Record", "PairSpec", "Pair", "PairReport",
           "build_pairs", "format_pair_report",
           "ViewSample", "FitPair", "PairViewLoader", "format_dataset_report",
           "load_image", "DEFAULT_SIZE"]

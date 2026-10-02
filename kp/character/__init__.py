"""kp.character —— 角色卡数据对象 + Character Fitter + 多视角配对。

    CharacterCard   身份载体（2.5D 分层 + 身份 token，跨骨干版本通用）
    CharacterFitter 多视角 → 身份 token（一次性训练；换角色 = 前向一次，秒级）
    pairing         多视角**声明式**配对（同角色 + 只动一个旋钮）
"""
from .card import CharacterCard, SEMANTIC_LAYERS, N_LAYERS, CARD_FORMAT_VERSION  # noqa: F401
from .fitter import CharacterFitter, ViewEncoder  # noqa: F401
from .pairing import (  # noqa: F401
    AXES, Record, PairSpec, Pair, PairReport,
    build_pairs, format_pair_report,
)

__all__ = ["CharacterCard", "SEMANTIC_LAYERS", "N_LAYERS", "CARD_FORMAT_VERSION",
           "CharacterFitter", "ViewEncoder",
           "AXES", "Record", "PairSpec", "Pair", "PairReport",
           "build_pairs", "format_pair_report"]

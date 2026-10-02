"""kp.character —— 角色卡数据对象 + Character Fitter。

    CharacterCard   身份载体（2.5D 分层 + 身份 token，跨骨干版本通用）
    CharacterFitter 多视角 → 身份 token（一次性训练；换角色 = 前向一次，秒级）
"""
from .card import CharacterCard, SEMANTIC_LAYERS, N_LAYERS, CARD_FORMAT_VERSION  # noqa: F401
from .fitter import CharacterFitter, ViewEncoder  # noqa: F401

__all__ = ["CharacterCard", "SEMANTIC_LAYERS", "N_LAYERS", "CARD_FORMAT_VERSION",
           "CharacterFitter", "ViewEncoder"]

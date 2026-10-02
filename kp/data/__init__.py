"""kp.data —— 数据管线（路线 P1.8「中文自然语言支持」的语料侧基础设施）。

设计背景（为什么这层必须**早**建）：
  · 文本塔是 ~220M **LLM**（自 Qwen3-4B 蒸馏）⇒ **中文是原生能力**，不需要像
    SDXL 那样「先训一个中文文本塔」。
  · 但**中文够不够好**只取决于一件事：训练数据里中文 caption 的**占比与形态**。
  · 而该占比在**预训练前定稿后不可逆** —— 数据管线定稿是全程唯一硬死线。
  ⇒ 所以「语言配比」不是事后调参，是**架构级设计变量**；本模块是它的**可执行判据**。

本模块只做**审计与规划**，不做训练。纯 CPU / IO。
"""
from .captions import (  # noqa: F401
    CaptionStat,
    CaptionAudit,
    LangMix,
    detect_language,
    tag_soup_score,
    analyze_caption,
    audit_captions,
    suggest_language_plan,
    CJK_RANGES,
)

__all__ = [
    "CaptionStat", "CaptionAudit", "LangMix",
    "detect_language", "tag_soup_score", "analyze_caption",
    "audit_captions", "suggest_language_plan", "CJK_RANGES",
]

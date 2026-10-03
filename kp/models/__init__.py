"""kp.models —— 主干件：单流 DiT + HybridVAE + TextTower + CharaBridge。

⚠️ **2026-10-03 重写版**。原包因 `.gitignore` 第 43 行的 `models/`
   **没有前导斜杠**（git 规则：匹配任意层级同名目录）被连带忽略，
   ⇒ **从未入库、全历史都不存在**，换机后 `python -m kp.selftest`
   在第 6 节 `No module named 'kp.models'` 直接中断。
   `.gitignore` 已改为锚定的 `/models/`（见 MEMORY.md 版本控制节）。

   本版按 `design/` 主文档 + 补充01/04 + `kp/config.py` +
   `selftest.py` / `arch_report.py` / `train/qad.py` 的**实际调用面**重写，
   并保持与原版一致的参数量结构（**17·d²/block**，见 `dit.py` 顶部推导；
   ⚠️ 2026-10-03 由 18·d² 改为 17·d² —— 删掉 adaLN 里全代码库无人读取的
   段 7「g_geo」死参数，详见 `dit.py` 顶部「为什么是 8 段不是 9 段」）。

分层：
    kp.models.dit         单流 DiT 主干（3:1 混合注意力 / QK-Norm / Matryoshka）
    kp.models.vae         HybridVAE（32× 空间压缩，40ch = 8 语义 + 32 细节）
    kp.models.text_tower  ~220M 多语言文本塔（自 Qwen3-4B 蒸馏）
    kp.models.charabridge 多视角身份 + 几何双分支（可关断返回 None）
    kp.models.common      RMSNorm / 正弦位置编码 / 时间步嵌入
"""
from __future__ import annotations

from .charabridge import CharaBridge
from .common import RMSNorm, TimestepEmbedding, sincos_1d, sincos_2d
from .dit import (Attention, DiTBlock, MLP, SingleStreamDiT, TextLayoutRouter,
                  build_attn_plan)
from .text_tower import TextTower, TextTowerCfg
from .vae import HybridVAE

__all__ = [
    "SingleStreamDiT", "DiTBlock", "Attention", "MLP", "TextLayoutRouter",
    "build_attn_plan",
    "HybridVAE",
    "TextTower", "TextTowerCfg",
    "CharaBridge",
    "RMSNorm", "TimestepEmbedding", "sincos_1d", "sincos_2d",
]

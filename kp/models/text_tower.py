"""TextTower —— ~220M 多语言文本塔（自 **Qwen3.5-4B-Base** 蒸馏）。

⚠️ 本文件属于 2026-10-03 **重写版**（原包因 `.gitignore` 未锚定被连带忽略而丢失）。

## 🔴 教师换版（2026-10-05 用户决策）
原设计写 `Qwen3.5-4B-Base`（27 处引用）⇒ **已过时**。改为 **`Qwen/Qwen3.5-4B-Base`**：

| |旧 Qwen3.5-4B-Base | **新 Qwen3.5-4B-Base** |
|---|---|---|
| 发布 | 2025 | **2026-02-27** |
| 许可 | — | **Apache-2.0**（可商用）|
| 词表 | 151,669（实测）| **248,320**（卡片值，⛔ 未实测）|
| 语言覆盖 | — | **201 种** |
| 架构 | 纯文本 Transformer | ⚠️ **多模态早融合**（`Qwen3_5ForConditionalGeneration`）|

⭐ **换教师不增加蒸馏工作量**（蒸馏是训练过程，不是抄参数）⇒ 零成本升级。
⚠️ **但有两处真实成本，见下文「换教师的连带改动」**。

## ⭐「剥离视觉」的正确理解（2026-10-05 用户提问）
用户问「我们要剥离视觉」。**准确说法是「不取」，不是「剥离」**：
- 3.5 是多模态模型（含视觉塔 + 早融合）
- 蒸馏时只取**语言侧的 hidden states** ⇒ 视觉塔**根本不进计算图**
- ⇒ 不需要任何"剥离"操作，**也不该为此写代码**

⚠️ **真实成本在这里**：3.5 的层结构是 **Gated DeltaNet + Gated Attention 混合**
（`8 × (3 × (DeltaNet→FFN) + 1 × (Attention→FFN))`），**不是**标准 Transformer
⇒ `kp/text/distill.py` 的逐层 hook **必须适配**，否则取不到 hidden states。

## 换教师的连带改动（⛔ 三处，缺一即静默出错）
1. **词表**：`TextTowerCfg.vocab_size` 默认 151936 是 Qwen3.5-4B-Base 的口径
   ⇒ 换 3.5 后**必须重跑 `kp.text.tokenizer.probe()` 量出真实上界**
   （⛔ 不要照抄 248,320 —— 本项目已在 Qwen3.5-4B-Base 上实测过「卡片值 ≠ 真实索引上界」）
2. **层数/隐藏维**：3.5 是 32 层 × 2560（≠旧配置 14 层 × 768）
   ⇒ `layers/dim/heads` 是**学生**规模，可不动；但**教师侧**的层数要按 3.5 配
3. **蒸馏脚本**：见上文「真实成本」—— DeltaNet 层不能当普通 attention 层处理

## ⭐ 为什么这一层是「设计变量」而不是普通编码器
SDXL 的文本塔是 **CLIP** —— 中文 token 从未训过 ⇒ **必须全英文**。
KP 的文本塔是 **LLM**（自 Qwen3.5-4B-Base 蒸馏）⇒ **中文是原生能力**，
于是输入形态从「逗号标签串」变成「**自然语言句子**」（设计稿补充11 §4.8）。

⚠️ **但中文能力不是白送的**：它取决于训练数据里中文 caption 的占比与形态，
   而这在**预训练前定稿后不可逆** ⇒ 那组旋钮在 `kp/config.py::CaptionCfg`。

## 骨架规模
默认配置 ≈ 216M，对齐设计稿标称的「~220M」。真实蒸馏权重是**另一条训练线**，
不在这份参考实现里（这里只固定接口与参数量级）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..capability.bus import GatedLinear
from .common import RMSNorm

__all__ = ["TextTower", "TextTowerCfg"]


@dataclass(frozen=True)
class TextTowerCfg:
    vocab_size: int = 151936        # ⚠️ 见下方「词表口径」—— 换 3.5 后**这个数会变**
    dim: int = 768
    layers: int = 14
    heads: int = 12
    mlp_ratio: float = 4.0
    out_dim: int = 1024             # 输出给主干的 ctx 维度
    max_len: int = 512
    causal: bool = False            # 作为条件编码器用双向；蒸馏自因果 LLM，可切换
    tie_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def param_count(self) -> int:
        """参数量估算（用来核对「~220M」这个量级）。"""
        d = self.dim
        emb = self.vocab_size * d
        per_layer = 4 * d * d + 2 * d * int(d * self.mlp_ratio) + 4 * d
        return emb + self.layers * per_layer + d * self.out_dim + 2 * d

    # ════════════════════════════════════════════════════════════════════
    # 词表口径（⭐ 换教师时**必读**）
    # ════════════════════════════════════════════════════════════════════
    # ⚠️ **三种数不一样，别照抄模型卡**（2026-10-05 实测 Qwen3.5-4B-Base tokenizer）：
    #   ① tok.vocab_size      = 151643  ← **不含** added_tokens
    #   ② len(tok)             = 151669  ← 含 added_tokens（26 个）
    #   ③ max(token id) + 1= 151669  ← ⭐ **真正能安全索引的上界**
    #   卡片/config 写的 151936比 ③ 还大 267 ⇒ 按它建表**不会越界**，
    #   但**白占 267×768 = 0.2M 参数**。
    #
    # 🔴 **换 Qwen3.5-4B-Base 后这三个数都会变**（卡片称词表 248,320，
    #    但**同样不能照抄** —— 3.5 是 `Qwen3_5ForConditionalGeneration` 多模态架构，
    #    tokenizer 可能带更多 control token / added_tokens）。
    #⇒ **正确做法**：`kp.text.tokenizer.probe()` 量出 ③，再回来改这个默认值。
    #   在此之前**保持 151936**（宁可大、不越界），并让蒸馏脚本报差异而非静默采用。
    VOCAB_SOURCE = "Qwen3.5-4B-Base（默认）｜换 3.5 后必须重跑 kp.text.tokenizer.probe()"


def _gl(in_f: int, out_f: int, std: float = 0.02, bias: bool = False) -> GatedLinear:
    w = torch.empty(out_f, in_f)
    nn.init.normal_(w, std=std / math.sqrt(max(1, in_f) / 64.0))
    return GatedLinear(w, torch.zeros(out_f) if bias else None)


class _Block(nn.Module):
    def __init__(self, cfg: TextTowerCfg):
        super().__init__()
        d = cfg.dim
        self.norm1 = RMSNorm(d)
        self.qkv = _gl(d, 3 * d)
        self.out = _gl(d, d)
        self.norm2 = RMSNorm(d)
        hidden = int(d * cfg.mlp_ratio)
        self.fc1 = _gl(d, hidden)
        self.fc2 = _gl(hidden, d)
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.causal = cfg.causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        if self.causal:
            m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(m, float("-inf"))
            o = scores.softmax(dim=-1) @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(o.transpose(1, 2).reshape(B, L, -1))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class TextTower(nn.Module):
    """token ids → 上下文序列 [B, L, out_dim]。"""

    def __init__(self, cfg: Optional[TextTowerCfg] = None):
        super().__init__()
        cfg = cfg or TextTowerCfg()
        self.cfg = cfg
        d = cfg.dim
        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pos = nn.Parameter(torch.randn(cfg.max_len, d) * 0.02)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(d)
        self.out_proj = _gl(d, cfg.out_dim)
        self._freeze()
        self.register_buffer("_n_layers", torch.tensor(float(cfg.layers)), persistent=False)

    def _freeze(self) -> int:
        n = 0
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
        return n

    def gated_linears(self) -> dict:
        return {n: m for n, m in self.named_modules() if isinstance(m, GatedLinear)}

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.shape[1]
        if L > self.cfg.max_len:
            raise ValueError(f"序列长度 {L} 超过 max_len={self.cfg.max_len}")
        x = self.embed(ids) + self.pos[:L].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.norm(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

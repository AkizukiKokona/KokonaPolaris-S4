"""TextTower —— ~220M 多语言文本塔（自 Qwen3-4B 蒸馏）。

⚠️ 本文件属于 2026-10-03 **重写版**（原包因 `.gitignore` 未锚定被连带忽略而丢失）。

## ⭐ 为什么这一层是「设计变量」而不是普通编码器
SDXL 的文本塔是 **CLIP** —— 中文 token 从未训过 ⇒ **必须全英文**。
KP 的文本塔是 **LLM**（自 Qwen3-4B 蒸馏）⇒ **中文是原生能力**，
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
    vocab_size: int = 151936        # Qwen3 系词表量级
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

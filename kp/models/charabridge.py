"""CharaBridge —— 多视角身份 + 几何双分支（gated 注入，可关断）。

⚠️ 本文件属于 2026-10-03 **重写版**（原包因 `.gitignore` 未锚定被连带忽略而丢失）。

## 设计要点（补充04 §身份载体）
- **身份 token 走 cross-attention，不拼主序列**：预算 **256 token**；
  全图只要 1024 个图像 token，拼进主序列会让注意力成本 ×4。
  这里的 `n_tokens` 就是那个预算（骨架自检用 16 做小规模验证）。
- **可关断 = 返回 `None`，不是零向量**：零向量进 cross-attention 仍会算出非零输出
  （`softmax/sigmoid(QKᵀ)·V ≠ 0`），所以「乘 0 再相加」**不等于**「不计算」。
  只有整条分支返回 `None`、调用方据此跳过，才可能 bit-exact。
- **几何双分支**：参考图的 normal / depth / ray-pose 与 RGB 身份分开编码，
  在融合前相加。多解几何是 CharaBridge 比 IP-Adapter 强的地方。
- ⭐ 与「角色卡」的关系：角色卡（2.5D 分层）→ **Character Fitter** → 身份 token →
  本模块。**换角色 = Fitter 前向一次（秒级）**，不重训主干。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..capability.bus import GatedLinear
from .common import RMSNorm

__all__ = ["CharaBridge"]


def _gl(in_f: int, out_f: int, std: float = 0.02) -> GatedLinear:
    w = torch.empty(out_f, in_f)
    nn.init.normal_(w, std=std / math.sqrt(max(1, in_f) / 64.0))
    return GatedLinear(w)


class _ViewEncoder(nn.Module):
    """单个视图 → 特征向量（轻量 CNN，骨架级）。"""

    def __init__(self, view_dim: int, ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(ch * 2, view_dim, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)          # [N, view_dim]


class CharaBridge(nn.Module):
    """`refs: [B, V, 3, H, W]` → 身份 token `[B, n_tokens, dim]`（或 `None`）。"""

    def __init__(self, dim: int = 1024, n_tokens: int = 256, view_dim: int = 256,
                 heads: int = 16, gate: float = 0.0):
        super().__init__()
        heads = max(1, min(heads, dim))
        while dim % heads:
            heads -= 1
        self.dim, self.n_tokens, self.view_dim = dim, n_tokens, view_dim
        self.heads, self.head_dim = heads, dim // heads

        self.identity_enc = _ViewEncoder(view_dim)
        self.geometry_enc = _ViewEncoder(view_dim)       # ⭐ 几何独立分支
        self.kv = _gl(view_dim, 2 * dim)
        self.geo_kv = _gl(view_dim, 2 * dim)             # 几何只进 K/V
        self.q = _gl(dim, dim)
        self.out = _gl(dim, dim)
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(self.head_dim)

        # 门控：0 ⇒ 整条分支返回 None（**不是**返回零向量）
        self.gate = nn.Parameter(torch.tensor(float(gate)), requires_grad=False)
        self.register_buffer("_n_views", torch.tensor(0.0), persistent=False)

    # --- 门控（与 capability.bus 同一套语义：精确 == 0 才算关断）---
    @property
    def is_off(self) -> bool:
        return float(self.gate.detach()) == 0.0

    def set_gate(self, value: float) -> None:
        with torch.no_grad():
            self.gate.fill_(float(value))

    def gated_linears(self) -> dict:
        return {n: m for n, m in self.named_modules() if isinstance(m, GatedLinear)}

    # --- 前向 ---
    def forward(self, refs: torch.Tensor,
                geo: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.is_off:
            return None                                   # ★ 整条分支短路

        B, V = refs.shape[0], refs.shape[1]
        flat = refs.reshape(B * V, *refs.shape[2:])
        feat = self.identity_enc(flat).reshape(B, V, -1)   # [B, V, view_dim]
        kv = self.kv(feat)                                 # [B, V, 2d]

        if geo is not None:
            gflat = geo.reshape(B * geo.shape[1], *geo.shape[2:])
            gfeat = self.geometry_enc(gflat).reshape(B, geo.shape[1], -1)
            kv = kv + self.geo_kv(gfeat)                   # ⭐ 几何分支参与融合

        k, v = kv.chunk(2, dim=-1)
        q = self.q(self.norm_q(self.queries)).unsqueeze(0).expand(B, -1, -1)
        q = q.reshape(B, self.n_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, V, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, V, self.heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(k)
        # 身份 token 用 **非归一化** 打分：身份浓度不应被 softmax 的份额竞争稀释
        scores = torch.sigmoid(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim))
        o = (scores @ v).transpose(1, 2).reshape(B, self.n_tokens, -1)
        return self.out(o)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_tokens={self.n_tokens}, view_dim={self.view_dim}, "
                f"heads={self.heads}, gate={float(self.gate.detach())}")

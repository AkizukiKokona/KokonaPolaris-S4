"""Character Fitter —— 一次性训练，之后「换角色 = 前向一次」。

流程（设计稿 §4.6）：
    ① 一次性训练 Character Fitter（本模块）
    ② 换角色 = Fitter 前向一次（**秒级**，峰值 ~1.2GB）
    ③ 可选：在角色卡上做轻量精修

输入：同一角色的**多视角参考图**
     · 最少 = **正视图 + 背视图**（2 视图即可起跑）
     · 参考图分辨率口径：看「**角色本体占多大**」，不是「画布多大」
       （本体界框长边 ≥1024 / 短边 ≥512）
     · ⚠️ 单一投影尺度：不同尺度的图混进来要么缩回白给、要么放大造假

双分支（对应 CharaBridge 的几何解耦先验）：
    · 身份分支：RGB → 外观 token
    · 几何分支：normal / depth / ray-pose → 几何 token（预留）
最终由 cross-view attention 汇聚成 **身份 token（256 × dim）**，
交由主干的 cross-attention 使用（**不拼主序列**，避免序列翻倍、注意力成本 ×4）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CAP


class ViewEncoder(nn.Module):
    """共享权重的单视图编码器（RGB → token 序列）。"""

    def __init__(self, in_ch: int = 3, base: int = 32, dim: int = 256):
        super().__init__()
        c = base
        self.stem = nn.Conv2d(in_ch, c, 3, padding=1)
        blocks = []
        for _ in range(4):
            blocks += [nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(1, c), nn.SiLU()]
        self.body = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(c, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, dim)。"""
        h = self.proj(self.body(self.stem(x)))
        return h.flatten(2).transpose(1, 2)


class CharacterFitter(nn.Module):
    """多视角 → 身份 token。"""

    def __init__(self, dim: int = 1024, n_tokens: int = None,
                 view_dim: int = 256, heads: int = 8, use_geometry: bool = True):
        super().__init__()
        self.dim = dim
        self.n_tokens = int(n_tokens or CAP.identity_tokens)
        self.use_geometry = use_geometry
        self.rgb_enc = ViewEncoder(3, 32, view_dim)
        self.geo_enc = ViewEncoder(3, 32, view_dim) if use_geometry else None
        layer = nn.TransformerEncoderLayer(d_model=view_dim, nhead=heads,
                                           dim_feedforward=4 * view_dim,
                                           batch_first=True, norm_first=True)
        self.view_fuse = nn.TransformerEncoder(layer, num_layers=2)
        # 可学习的身份 query（聚合为固定 256 token）
        self.queries = nn.Parameter(torch.randn(self.n_tokens, view_dim) * 0.02)
        self.to_dim = nn.Linear(view_dim, dim)

    def forward(self, views: torch.Tensor,
                geo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """views: (B, V, 3, H, W) → identity tokens (B, T, dim)。"""
        B, V = views.shape[:2]
        x = views.reshape(B * V, *views.shape[2:])
        tok = self.rgb_enc(x)                                   # (B*V, N, view_dim)
        if self.geo_enc is not None and geo is not None:
            g = geo.reshape(B * V, *geo.shape[2:])
            tok = tok + self.geo_enc(g)
        N = tok.shape[1]
        tok = self.view_fuse(tok).reshape(B, V * N, -1)         # 跨视图汇聚
        q = self.queries.unsqueeze(0).expand(B, -1, -1)          # (B, T, view_dim)
        attn = torch.softmax(q @ tok.transpose(1, 2) / (tok.shape[-1] ** 0.5), dim=-1)
        pooled = attn @ tok                                      # (B, T, view_dim)
        return self.to_dim(pooled)

    @torch.no_grad()
    def make_identity(self, views: torch.Tensor, geo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """推理路径：换角色时调用一次（秒级），返回身份 token。"""
        self.eval()
        return self.forward(views, geo)


__all__ = ["CharacterFitter", "ViewEncoder"]

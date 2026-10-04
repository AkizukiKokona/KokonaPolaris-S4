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
    """多视角 → 身份 token。

    ⚠️ **2026-10-05 删除 `use_geometry` 参数**（项目教训：「预留接口不该用每层都付钱的方式存在」）
       审计发现：训练侧三处都显式传 `use_geometry=False`（`kp/train/fitter.py:174/295/309`）
       ⇒ `geo_enc` **在正式训练里永不执行**，但 `__init__` 默认 `True`
       ⇒ 有人写 `CharacterFitter()` 不传参就会**白建一个 `geo_enc`**。

       ⭐ 保留的只有 `forward(views, geo=...)` 的**形参**（纯数据通路，不建模块）——
       将来真要用几何分支时，只需在这里加回 `geo_enc = ViewEncoder(...)` 一行。
       这与 2026-10-03 删 adaLN 第 7 段 `g_geo` 的处理**同一个理由**。
    """

    def __init__(self, dim: int = 1024, n_tokens: int = None,
                 view_dim: int = 256, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_tokens = int(n_tokens or CAP.identity_tokens)
        self.rgb_enc = ViewEncoder(3, 32, view_dim)
        # ⛔ 不再无条件建 geo_enc（它没有读取者⇒ 死参数）
        self.geo_enc: Optional[nn.Module] = None
        layer = nn.TransformerEncoderLayer(d_model=view_dim, nhead=heads,
                                           dim_feedforward=4 * view_dim,
                                           batch_first=True, norm_first=True)
        self.view_fuse = nn.TransformerEncoder(layer, num_layers=2)
        # 可学习的身份 query（聚合为固定 256 token）
        self.queries = nn.Parameter(torch.randn(self.n_tokens, view_dim) * 0.02)
        self.to_dim = nn.Linear(view_dim, dim)

    def enable_geometry(self, view_dim: int = 256) -> "CharacterFitter":
        """☘️ **显式**打开几何分支（默认关闭 —— 不预留、不白建）。"""
        if self.geo_enc is None:
            self.geo_enc = ViewEncoder(3, 32, view_dim)
        return self

    def forward(self, views: torch.Tensor,
                geo: Optional[torch.Tensor] = None) -> torch.Tensor:
        """views: (B, V, 3, H, W) → identity tokens (B, T, dim)。

        ⚠️ 传了 `geo` 但没 `enable_geometry()` ⇒ **明确报错**，不静默忽略
        （静默忽略 = 几何信息悄悄消失，是最难查的一类 bug）。
        """
        B, V = views.shape[:2]
        x = views.reshape(B * V, *views.shape[2:])
        tok = self.rgb_enc(x)                                   # (B*V, N, view_dim)
        if geo is not None:
            if self.geo_enc is None:
                raise ValueError(
                    "传了 `geo` 但几何分支未启用 ⇒ 调 `model.enable_geometry()`。"
                    "⛔ 不静默忽略（否则几何信息悄悄消失）。")
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

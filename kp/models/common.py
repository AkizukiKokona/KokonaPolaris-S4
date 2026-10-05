"""kp.models 公共件（位置编码 / 归一化 / 时间步嵌入）。

⚠️ 本文件属于 2026-10-03 **重写版**：原 `kp/models/` 因 `.gitignore` 的
   `models/`（未锚定根目录）被连带忽略，**从未入库**，换机后丢失。
   本次按 design/ 主文档 + 补充01/04 + `kp/config.py` + `selftest`/`arch_report`
   的调用面重写，并保持与原版一致的**参数量结构**（见 dit.py 顶部推导）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "sincos_1d", "sincos_2d", "TimestepEmbedding"]


class RMSNorm(nn.Module):
    """无 bias 的 RMSNorm（DiT 标配；比 LayerNorm 少一半参数、更适合低比特）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def sincos_1d(pos: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """一维正弦位置编码。pos: [N] 或 [N,1] → [N, dim]。dim 会向下取偶。"""
    if dim % 2:
        dim -= 1
    pos = pos.reshape(-1, 1).float()
    half = dim // 2
    # ⚠️ 2026-10-05 修：`freqs` 默认建在 CPU ⇒ 与 GPU 上的 pos 设备不一致
    #    （原写法只 .float()，没跟设备）⇒ GPU 训练一跑就炸
    freqs = torch.exp(-math.log(base)
                      * torch.arange(half, dtype=torch.float32,
                                     device=pos.device) / half)
    ang = pos * freqs.reshape(1, -1)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def sincos_2d(h: int, w: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """二维网格位置编码 → [h*w, dim]。

    ⭐ **分辨率无关**：Matryoshka 早期步在 16 token 网格上跑、后期长到 1024，
       位置编码必须是「算出来的」而不是「学出来的固定长度表」，
       否则事后加不上（设计稿明确要求「一开始就留好位置编码」）。
    """
    dim_h = (dim // 2) // 2 * 2
    dim_w = dim - dim_h
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    ey = sincos_1d(ys, dim_h, base)              # [h, dim_h]
    ex = sincos_1d(xs, dim_w, base)              # [w, dim_w]
    ey = ey.reshape(h, 1, dim_h).expand(h, w, dim_h)
    ex = ex.reshape(1, w, dim_w).expand(h, w, dim_w)
    return torch.cat([ey, ex], dim=-1).reshape(h * w, dim)


class TimestepEmbedding(nn.Module):
    """正弦时间步嵌入 + 两层 MLP。

    ⚠️ 用 `nn.Linear`（真参数）而非 `GatedLinear`（buffer）—— 与 `train/qad.py`
       `freeze_backbone` 的说明一致：「骨架里 adaLN / t-embed / domain-embed
       仍是 nn.Linear（真参数）」，它们必须被显式冻结，否则可训占比会虚高 110×。
    """

    def __init__(self, dim: int, hidden: int = None, t_dim: int = 256):
        super().__init__()
        hidden = hidden or dim * 4
        self.t_dim = t_dim
        self.mlp = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sincos_1d(t, self.t_dim))

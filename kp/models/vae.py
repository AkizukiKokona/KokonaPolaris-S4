"""HybridVAE —— 32× 空间压缩的混合 latent 编解码器。

⚠️ 本文件属于 2026-10-03 **重写版**（原包因 `.gitignore` 未锚定被连带忽略而丢失）。

## 与设计稿的对应
- **32× 空间压缩**：5 级 stride-2 ⇒ 256² → 8²、1024² → 32²（`LATENT.spatial = 32`）。
- **40ch 混合 latent** = 8ch 语义（对齐 DINOv3）+ 32ch 细节（DC-AE 式）。
  这里把通道显式切开再拼回，**让「语义 / 细节」的边界在代码里是可见的** ——
  因为模块化前提是「通道分离必须被显式监督」（`kp/latent` 里有对应监督件）。
- ⚠️ 设计稿：**不量化 VAE**（`config.QUANT` 只作用于主干）。

## 为什么是「骨架级」规模
骨架只要求形状与梯度正确、纯 CPU 可跑。真正的 VAE 要蒸馏 DC-AE，
权重走单独的训练线，**不在这份参考实现的范围里**。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LATENT

__all__ = ["HybridVAE"]


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    g = groups
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class HybridVAE(nn.Module):
    """3 → 40ch 的 32× VAE（语义 8 + 细节 32）。"""

    LATENT_CH = LATENT.total_ch          # 40
    STRIDES = 5                          # 2**5 = 32

    def __init__(self, base: int = 16, semantic_ch: int = None,
                 detail_ch: int = None, img_ch: int = 3):
        super().__init__()
        self.base = base
        self.semantic_ch = LATENT.semantic_ch if semantic_ch is None else semantic_ch
        self.detail_ch = LATENT.detail_ch if detail_ch is None else detail_ch
        self.latent_ch = self.semantic_ch + self.detail_ch
        self.img_ch = img_ch
        top = base * (2 ** self.STRIDES)

        # ---- encoder ----
        enc: List[nn.Module] = [_gn(img_ch, 1), nn.SiLU(),
                                nn.Conv2d(img_ch, base, 3, padding=1)]
        ch = base
        for _ in range(self.STRIDES):
            enc += [_gn(ch), nn.SiLU(), nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
            ch *= 2
        enc += [_gn(ch), nn.SiLU(), nn.Conv2d(ch, self.latent_ch, 1)]
        self.encoder = nn.Sequential(*enc)

        # ---- decoder ----
        dec: List[nn.Module] = [nn.Conv2d(self.latent_ch, top, 1)]
        ch = top
        for _ in range(self.STRIDES):
            dec += [_gn(ch), nn.SiLU(),
                    nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1)]
            ch //= 2
        dec += [_gn(ch, 1), nn.SiLU(), nn.Conv2d(ch, img_ch, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

        self.register_buffer("_scale", torch.tensor(1.0), persistent=False)

    # --- 接口 ---
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)

    def encode_latent(self, img: torch.Tensor) -> torch.Tensor:
        """图像 → 混合 latent（**不做后验采样**：Rectified Flow 用的是确定性 latent）。"""
        return self.encode(img)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # --- 通道结构（显式可见，供通道分离监督件使用）---
    def split_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (语义 8ch, 细节 32ch)。"""
        return z[:, :self.semantic_ch], z[:, self.semantic_ch:]

    def merge_latent(self, semantic: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        return torch.cat([semantic, detail], dim=1)

    def compression(self) -> int:
        return 2 ** self.STRIDES

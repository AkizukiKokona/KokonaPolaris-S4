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


class ResBlock(nn.Module):
    """残差块（2026-10-05 新增）—— ⭐ 这是"图糊"的**真正架构修复**。

    ═══ 为什么必须加它 ═══
    实测排除了两个假设（都不是瓶颈）：
      · 容量：4.4M→17.6M 涨 +2.5dB，但 **17.6M→70M 不再涨**（16.42 vs 16.70）
      · 分辨率：256px→512px **也不涨**（16.38 vs 16.70）
    ⚠️ 而原架构**每个分辨率层级只有 1 个卷积**（无残差、无多卷积）
      ⇒ 参数全堆在"通道宽"上，**深度结构太浅**。
    ✅ DC-AE / SD-VAE 都是「每层多个残差块」⇒ 这里补上。
    """

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        # ⭐ 零初始化最后卷积 ⇒ 初始时恒等映射（残差块标准做法，训练更稳）
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class HybridVAE(nn.Module):
    """3 → 40ch 的 32× VAE（语义 8 + 细节 32）。

    ⚠️ 2026-10-05：新增 `res_blocks` 参数（每层残差块数，默认 0 = 旧行为）。
       ⭐ 传 >0 就启用残差结构 —— 这是治"糊"的**架构级**修复。
       ⛔ 注意：新旧权重**不通用**（结构不同）。
    """

    LATENT_CH = LATENT.total_ch          # 40
    STRIDES = 5                          # 2**5 = 32

    def __init__(self, base: int = 16, semantic_ch: int = None,
                 detail_ch: int = None, img_ch: int = 3,
                 res_blocks: int = 0, ch_cap: int = 512):
        super().__init__()
        self.base = base
        self.res_blocks = int(res_blocks)
        self.ch_cap = int(ch_cap)
        self.semantic_ch = LATENT.semantic_ch if semantic_ch is None else semantic_ch
        self.detail_ch = LATENT.detail_ch if detail_ch is None else detail_ch
        self.latent_ch = self.semantic_ch + self.detail_ch
        self.img_ch = img_ch
        self.strides = self.STRIDES
        # ⭐ 有残差时通道不再无脑翻倍（否则 2^5·base 把显存打爆）
        # ⛔ 重要：res_blocks==0 时**必须沿用旧的通道表**（base·2^i），
        #    否则旧权重全部 load 失败（2026-10-05 实测踩到）。
        if self.res_blocks == 0:
            wid = (lambda i: base * (2 ** min(i, self.strides)))
        else:
            wid = (lambda i: min(self.ch_cap, base * (2 ** min(i, 3))))
        top = wid(self.strides)

        # ---- encoder ----
        enc: List[nn.Module] = [_gn(img_ch, 1), nn.SiLU(),
                                nn.Conv2d(img_ch, base, 3, padding=1)]
        ch = base
        for i in range(self.strides):
            for _ in range(self.res_blocks):
                enc += [ResBlock(ch)]
            nxt = wid(i + 1)
            enc += [_gn(ch), nn.SiLU(), nn.Conv2d(ch, nxt, 3, stride=2, padding=1)]
            ch = nxt
        for _ in range(self.res_blocks):
            enc += [ResBlock(ch)]
        enc += [_gn(ch), nn.SiLU(), nn.Conv2d(ch, self.latent_ch, 1)]
        self.encoder = nn.Sequential(*enc)

        # ---- decoder ----
        dec: List[nn.Module] = [nn.Conv2d(self.latent_ch, top, 1)]
        ch = top
        for _ in range(self.res_blocks):
            dec += [ResBlock(ch)]
        for i in range(self.strides, 0, -1):
            nxt = wid(i - 1)
            dec += [_gn(ch), nn.SiLU(),
                    nn.ConvTranspose2d(ch, nxt, 4, stride=2, padding=1)]
            ch = nxt
            for _ in range(self.res_blocks):
                dec += [ResBlock(ch)]
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


# ══════════════════════════════════════════════════════════════════════════
# PatchGAN 判别器（2026-10-05 新增）
# ══════════════════════════════════════════════════════════════════════════
# ⭐ **为什么要它**（这是"图糊"的真根因）：
#   我们原来只用 L1 重建损失，而 **L1 的数学最优解 = 所有图的平均值 = 模糊**。
#   ⚠️ L1 甚至**惩罚"猜对细节"** —— 猜对了就偏离平均图，L1 反而变大。
#   ⇒ 加参数的实测极限只有 +2.5dB（且花 1GB 显存，**不值**）。
#   ✅ 业界所有 SD 系 VAE 都用「L1 + 对抗 + (LPIPS)」，靠判别器逼 decoder
#      **编造**像真的高频细节，而不是回归到平均。
#
# ⚠️ 为什么用 **PatchGAN** 而不是普通判别器：
#   - 普通判别器只输出 1 个数（整图真假）⇒ 对细节无梯度
#   - PatchGAN 输出 **特征图**（每个 patch 一个真假分）⇒ **局部细节**才有监督
#   ⇒ 这正是"糊"要治的地方。
class PatchDiscriminator(nn.Module):
    """轻量 PatchGAN：输出 (B,1,h,w) 的**局部真假分**。

    ⚠️ 刻意做小（base=64，3 层）—— 8GB 显存铁律下，判别器不能喧宾夺主。
    """

    def __init__(self, in_ch: int = 3, base: int = 64, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base
        for i in range(1, n_layers):
            nxt = min(ch * 2, 256)
            layers += [
                nn.Conv2d(ch, nxt, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, nxt), nxt),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = nxt
        # ⭐ 最后一层 stride=1（保留 patch 网格，不打成 1x1）
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge GAN 损失（判别器侧）—— 比原始 GAN 稳，比 WGAN-GP 省显存。"""
    return (torch.relu(1.0 - real_logits).mean()
            + torch.relu(1.0 + fake_logits).mean()) * 0.5


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge GAN 损失（生成器侧）。"""
    return -fake_logits.mean()


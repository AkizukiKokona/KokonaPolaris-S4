"""Rectified Flow 采样器 + 分辨率 Matryoshka 调度。

⭐ Matryoshka（PostDiff：FID 18.42→15.69，FLOPs −60%）：
   早期去噪步在低 token 网格上跑（如 16×16 = 256 token），后期长到 32×32 = 1024。
   ⚠️ 必须**一开始**就留好位置编码与噪声调度的位置，不能事后加。

⚠️ G1 永久规范：**逐像素 PSNR/SSIM 只用于「同轨迹复现性」，不可判画质** ——
   20 步采样的早期微扰会被放大成「另一个同样合理的样本」。
   所以采样器的正确性判据是「同 seed 同轨迹可复现」，不是像素对齐。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def make_matryoshka_schedule(steps: int, lo_tokens: int = 256, hi_tokens: int = 1024,
                             switch_ratio: float = 0.5) -> List[int]:
    """返回每一步应使用的 token 数（前 switch_ratio 段用 lo，之后用 hi）。"""
    k = int(round(steps * switch_ratio))
    return [lo_tokens] * k + [hi_tokens] * (steps - k)


def _resize_latent(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """把 (B,C,H,W) 的 latent 网格重采样到约 target_tokens 个格（保持方形）。"""
    B, C, H, W = x.shape
    side = max(1, int(round(target_tokens ** 0.5)))
    if side == H and side == W:
        return x
    return F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)


@torch.no_grad()
def sample(model, shape: Tuple[int, ...], *, steps: int = 20,
           text_ctx: Optional[torch.Tensor] = None,
           identity_ctx: Optional[torch.Tensor] = None,
           domain: Optional[torch.Tensor] = None,
           device: str = "cpu", dtype: torch.dtype = torch.float32,
           seed: int = 0, matryoshka: Optional[Sequence[int]] = (256, 1024)) -> torch.Tensor:
    """Euler 采样器。

    shape: (B, C, H, W) —— 目标 latent 网格（1024² 图 → (B,40,32,32)）。
    返回 (B, C, H, W) 的 latent；解码到图像请用 HybridVAE.decode。
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(*shape, generator=g).to(device=device, dtype=dtype)
    t_scale = 1000.0

    sched = make_matryoshka_schedule(steps, *matryoshka) if matryoshka else [shape[-1] * shape[-2]] * steps
    for i in range(steps):
        t = torch.full((shape[0],), (i / steps) * t_scale, device=device, dtype=dtype)
        xt = _resize_latent(x, sched[i]) if matryoshka else x
        v = model(xt, t, text_ctx=text_ctx, identity_ctx=identity_ctx, domain=domain)
        if matryoshka and v.shape != x.shape:
            v = _resize_latent(v, x.shape[-1] * x.shape[-2])
        x = x + v / steps
    return x


__all__ = ["sample", "make_matryoshka_schedule"]

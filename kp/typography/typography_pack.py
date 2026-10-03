"""TypographyPack 的模型侧骨架 —— **低压缩 ROI 分支**。

为什么必须独立分支（设计稿 §4.9）：
  · 主干是 **32× 压缩** —— 40px 汉字只占 **1.25 个 latent 格**，字在编码阶段就被丢掉。
    **已实证**：Sana 的 32× DC-AE 输出「心夏北极星」→ `EinntaPears,` 乱码。
  · 所以文字**不能指望主干**，必须另开一条**低压缩**的眼睛（例如 4× 或 2×），
    只盯 Layout Planner 给出来的 ROI 框。

本文件是**形状级骨架**：把 ROI 裁剪 → 低压缩编码 → 输出给主干的**附加 token**，
且同样遵守「能力总线」的两条铁律：
  ① 门控为 0 时整条分支跳过 ⇒ `forward` 返回 **None** ⇒ 主干逐位不变；
  ② 附加 token **走 cross-attention**，不拼主序列（避免序列翻倍、注意力成本 ×4）。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import CAP


class ROIBranch(nn.Module):
    """ROI → 低压缩 token（供主干 cross-attention 使用）。"""

    def __init__(self, compression: int = 4, dim: int = 1024, base: int = 32,
                 tokens_per_roi: int = 16, gate: float = None):
        super().__init__()
        self.compression = int(compression)
        self.dim = dim
        self.tokens_per_roi = int(tokens_per_roi)
        steps = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[self.compression]
        c = base
        layers = [nn.Conv2d(3, c, 3, padding=1), nn.SiLU()]
        for _ in range(steps):
            layers += [nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(1, c), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((int(tokens_per_roi ** 0.5), int(tokens_per_roi ** 0.5)))
        self.proj = nn.Linear(c, dim)
        self.gate = nn.Parameter(
            torch.tensor(float(CAP.gate_init if gate is None else gate)), requires_grad=False)
        self.meta = {"kind": "roi", "compression": self.compression,
                     "tokens_per_roi": self.tokens_per_roi}

    @property
    def is_off(self) -> bool:
        g = self.gate
        if g.is_meta or g.device.type == "meta":
            return False
        return float(g.detach()) == 0.0

    def set_gate(self, v: float) -> "ROIBranch":
        self.gate.data.fill_(float(v))
        return self

    def forward(self, image: torch.Tensor, rois: Optional[List[dict]] = None) -> Optional[torch.Tensor]:
        """image: (B,3,H,W)；rois: [{x0,y0,x1,y1}, …] →
        返回 (B, n_roi*tokens_per_roi, dim)；**关断或没有 ROI 时返回 None**。
        """
        if self.is_off or not rois:
            return None
        feats = []
        for r in rois:
            crop = image[..., int(r["y0"]):int(r["y1"]), int(r["x0"]):int(r["x1"])]
            if crop.numel() == 0:
                continue
            h = self.proj(self.pool(self.encoder(crop)).flatten(2).transpose(1, 2))
            feats.append(h)
        if not feats:
            return None
        return torch.cat(feats, dim=1)


__all__ = ["ROIBranch"]

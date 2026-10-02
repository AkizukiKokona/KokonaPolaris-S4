"""角色卡 —— 身份载体（一等公民）。

⭐ 为什么不是 LoRA / 不是 3D 网格：
  · LoRA 绑死旧 `W₀`，换骨干即失效；且会产生**入侵维度**。
  · 3D 网格对二次元「不实用且美学不兼容」（See-through, SIGGRAPH 2026）。
  ⇒ **二次元的正确表征是 2.5D 分层**：把角色拆成若干语义层 + 遮挡补全 + 伪深度绘制序。

构成（约 1–5 MB）：
  ① **身份 token**（256 × dim，走 cross-attention，禁止拼主序列）
  ② **语义层**：19 类（与 See-through / TypographyPack 共用同一套格式）
  ③ **遮挡补全**：被遮挡区域的推测填充（保证任意角度可用）
  ④ **伪深度绘制序**：决定图层的先后（免 3D）
  ⑤ **元数据**：来源 / 版本 / 分辨率口径

⭐ 关键性质：**不依赖底模坐标系 ⇒ 跨主干版本通用**。
   换角色 = Character Fitter 前向一次（秒级），不再「每个角色训一个 LoRA」。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

# 19 类语义层（与 See-through 的 19 类对齐；TypingPack 复用同一套）
SEMANTIC_LAYERS: List[str] = [
    "background", "body_base", "face", "eye_white", "iris", "eyelash", "eyebrow",
    "mouth", "ear", "hair_front", "hair_side", "hair_back", "hat",
    "top", "bottom", "skirt", "outerwear", "legwear", "footwear",
]
N_LAYERS = len(SEMANTIC_LAYERS)
CARD_FORMAT_VERSION = 1


@dataclass
class CharacterCard:
    """一个角色的完整身份载体（跨骨干版本可用）。"""

    name: str
    identity_token: torch.Tensor                     # (T, dim)，T = CAP.identity_tokens
    layers: Dict[str, np.ndarray] = field(default_factory=dict)   # 语义层：name → RGBA (H,W,4)
    occlusion: Optional[np.ndarray] = None           # 遮挡补全图 (H,W,4)
    depth_order: Optional[np.ndarray] = None         # 伪深度绘制序 (H,W) float
    meta: Dict = field(default_factory=dict)

    # ---------------- 校验 ----------------
    def validate(self) -> List[str]:
        warn = []
        if self.identity_token.dim() != 2:
            raise ValueError(f"identity_token 必须是 (T, dim)，收到 {tuple(self.identity_token.shape)}")
        unknown = [k for k in self.layers if k not in SEMANTIC_LAYERS]
        if unknown:
            warn.append(f"未知语义层 {unknown}（不在 19 类之内）")
        if not self.layers:
            warn.append("角色卡没有任何语义层")
        return warn

    @property
    def num_tokens(self) -> int:
        return int(self.identity_token.shape[0])

    def size_bytes(self) -> int:
        n = self.identity_token.numel() * self.identity_token.element_size()
        for a in self.layers.values():
            n += a.nbytes
        if self.occlusion is not None:
            n += self.occlusion.nbytes
        if self.depth_order is not None:
            n += self.depth_order.nbytes
        return n

    # ---------------- 序列化（自描述、跨版本） ----------------
    def save(self, path: str) -> None:
        blob = {
            "format_version": CARD_FORMAT_VERSION,
            "name": self.name,
            "identity_token": self.identity_token.detach().cpu(),
            "layers": {k: v for k, v in self.layers.items()},
            "occlusion": self.occlusion,
            "depth_order": self.depth_order,
            "meta": self.meta,
        }
        torch.save(blob, path)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"format_version": CARD_FORMAT_VERSION, "name": self.name,
                       "tokens": self.num_tokens, "layers": sorted(self.layers.keys()),
                       "meta": {k: v for k, v in self.meta.items()}},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharacterCard":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ver = int(blob.get("format_version", 0))
        if ver > CARD_FORMAT_VERSION:
            raise ValueError(f"角色卡格式版本 {ver} 高于本实现 {CARD_FORMAT_VERSION}")
        return cls(name=blob["name"], identity_token=blob["identity_token"],
                   layers=dict(blob.get("layers", {})),
                   occlusion=blob.get("occlusion"),
                   depth_order=blob.get("depth_order"),
                   meta=dict(blob.get("meta", {})))

    def describe(self) -> dict:
        return {"name": self.name, "tokens": self.num_tokens,
                "token_dim": int(self.identity_token.shape[1]),
                "layers": len(self.layers), "size_kb": round(self.size_bytes() / 1024, 1),
                "meta": self.meta}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"CharacterCard({self.name!r}, tokens={self.num_tokens}, "
                f"layers={len(self.layers)}, {self.size_bytes()/1024:.0f}KB)")


__all__ = ["CharacterCard", "SEMANTIC_LAYERS", "N_LAYERS", "CARD_FORMAT_VERSION"]

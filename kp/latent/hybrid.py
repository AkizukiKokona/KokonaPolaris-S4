"""混合 Latent：32× 空间压缩 + 句法/细节通道分离。

设计要点（主文档 §4 / 补充 02）：
  · 语义 8ch  —— 与冻结 DINOv3 特征对齐（管"是什么"）
  · 细节 32ch —— DC-AE 式（管"长什么样"）
  · 1024² 图 → 32×32 格 → 每格 40ch → **1024 个 token**（patch_size = 1）

⭐ 本模块提供的 `pack_latent` / `unpack_latent` 是**纯函数、无状态**，
   属于「对外四接口」之一（latent 打包解包），必须在任何序列化/传输路径上复用，
   保证不同实现之间二进制兼容。

⚠️ 通道分离**必须显式监督**，否则主干会把两个通道都塞满信息 → 画风解耦失效
   → 三层控制栈的 L1 全盘作废。监督手段见本文件的 `channel_mi_penalty` 与
   `swap_channels`（交叉扰动），以及 `kp/tests/test_latent.py` 的用法示例。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from ..config import LATENT


# ---------------------------------------------------------------------------
# 形状描述
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HybridLatentShape:
    """(B, C, H, W)，C = semantic_ch + detail_ch。"""
    batch: int
    height: int
    width: int
    semantic_ch: int = LATENT.semantic_ch
    detail_ch: int = LATENT.detail_ch

    @property
    def total_ch(self) -> int:
        return self.semantic_ch + self.detail_ch

    @property
    def num_tokens(self) -> int:
        return self.batch * self.height * self.width

    def __str__(self) -> str:  # pragma: no cover
        return (f"HybridLatent(B={self.batch}, C={self.total_ch}"
                f"[{self.semantic_ch}+{self.detail_ch}], {self.height}x{self.width}, "
                f"tokens={self.num_tokens})")


def shape_of(x: torch.Tensor, semantic_ch: int = LATENT.semantic_ch,
             detail_ch: int = LATENT.detail_ch) -> HybridLatentShape:
    if x.dim() != 4:
        raise ValueError(f"混合 latent 必须是 (B,C,H,W)，收到 {tuple(x.shape)}")
    return HybridLatentShape(x.shape[0], x.shape[2], x.shape[3], semantic_ch, detail_ch)


# ---------------------------------------------------------------------------
# 打包 / 解包（纯函数）
# ---------------------------------------------------------------------------
_MAGIC = 0x4B503031          # "KP01"
_VERSION = 1
_HEAD_FMT = "<IIHHHHHHII"    # magic ver B C H W sem detail dtype image_size
_HEAD_LEN = 28


def _tensor_to_bytes(x: torch.Tensor):
    """→ (bytes, dtype_code)。bf16 用 uint16 视图绕过 numpy 无 bf16 的问题。"""
    import numpy as np
    x = x.detach().contiguous().cpu()
    if x.dtype == torch.float32:
        return x.numpy().astype(np.float32).tobytes(), 0
    if x.dtype == torch.float16:
        return x.numpy().astype(np.float16).tobytes(), 1
    if x.dtype == torch.bfloat16:
        return x.view(torch.uint16).numpy().astype(np.uint16).tobytes(), 2
    raise ValueError(f"不支持的 dtype {x.dtype}（需 fp32/fp16/bf16）")


def pack_latent(x: torch.Tensor, *, image_size: int = 0) -> bytes:
    """把混合 latent 打成自描述字节流（头部含 magic / 版本 / 形状 / dtype）。

    ⚠️ 任何跨进程、跨版本、跨实现传递都必须走这个函数，禁止裸 torch.save 张量。
    """
    import struct
    if x.dim() != 4:
        raise ValueError("pack_latent 需要 (B,C,H,W)")
    shp = shape_of(x)
    if shp.total_ch != LATENT.total_ch:
        raise ValueError(f"通道数 {shp.total_ch} != 设计值 {LATENT.total_ch}")

    raw, dt = _tensor_to_bytes(x)
    head = struct.pack(
        _HEAD_FMT,
        _MAGIC, _VERSION,
        shp.batch, shp.total_ch, shp.height, shp.width,
        shp.semantic_ch, shp.detail_ch, dt, int(image_size),
    )
    assert len(head) == _HEAD_LEN, (len(head), _HEAD_LEN)
    return head + raw


def unpack_latent(blob: bytes):
    """pack_latent 的逆。返回 (tensor, meta)。"""
    import struct
    import numpy as np

    fields = struct.unpack(_HEAD_FMT, blob[:_HEAD_LEN])
    magic, ver, b, c, h, w, sc, dc, dt, image_size = fields
    if magic != _MAGIC:
        raise ValueError(f"magic 不匹配：{magic:#x} != {_MAGIC:#x}")
    if ver != _VERSION:
        raise ValueError(f"版本不匹配：{ver} != {_VERSION}")

    itemsize = {0: 4, 1: 2, 2: 2}[dt]
    nbytes = b * c * h * w * itemsize
    buf = blob[_HEAD_LEN:_HEAD_LEN + nbytes]
    if len(buf) != nbytes:
        raise ValueError(f"载荷长度不符：{len(buf)} != {nbytes}")

    if dt == 0:
        t = torch.from_numpy(np.frombuffer(buf, np.float32).copy())
        dtype = torch.float32
    elif dt == 1:
        t = torch.from_numpy(np.frombuffer(buf, np.float16).copy())
        dtype = torch.float16
    else:
        u = torch.from_numpy(np.frombuffer(buf, np.uint16).copy())
        t, dtype = u.view(torch.bfloat16), torch.bfloat16

    x = t.reshape(b, c, h, w).to(dtype)
    return x, {"batch": b, "channels": c, "height": h, "width": w,
               "semantic_ch": sc, "detail_ch": dc,
               "dtype": str(dtype), "image_size": image_size}


# ---------------------------------------------------------------------------
# 通道操作
# ---------------------------------------------------------------------------
def split_channels(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """→ (semantic, detail)。"""
    sc = LATENT.semantic_ch
    return x[:, :sc], x[:, sc:]


def join_channels(semantic: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """(semantic, detail) → 混合 latent。"""
    if semantic.shape[0] != detail.shape[0]:
        raise ValueError("batch 不一致")
    if semantic.shape[2:] != detail.shape[2:]:
        raise ValueError("空间尺寸不一致")
    return torch.cat([semantic, detail], dim=1)


# ---------------------------------------------------------------------------
# ⭐ 通道分离的显式监督（设计稿硬要求）
# ---------------------------------------------------------------------------
def swap_channels(x: torch.Tensor) -> torch.Tensor:
    """交叉扰动：把批内 A 的语义通道换成 B 的、细节换成 A 自己的。

    训练时用法：把 swap 后的 latent 喂进主干，**画风相关输出应当随之改变、
    内容相关输出应当基本不变**。若两者都变 → 通道没解耦。
    """
    if x.shape[0] < 2:
        return x
    sc = LATENT.semantic_ch
    y = x.clone()
    y[0, :sc] = x[1, :sc]      # 语义来自另一张
    y[0, sc:] = x[0, sc:]      # 细节保持自己
    return y


def channel_mi_penalty(semantic: torch.Tensor, detail: torch.Tensor,
                       n_samples: int = 4096, seed: int = 0) -> torch.Tensor:
    """通道互信息惩罚（可微）。

    做法：把两个分支在空间上展平成 (N, C)，随机采样若干位置，计算
    **归一化互协方差矩阵的非对角能量**。语义与细节应当尽量独立 → 惩罚其相关性。
    """
    b = semantic.shape[0]
    s = semantic.flatten(2).transpose(1, 2).reshape(-1, LATENT.semantic_ch)
    d = detail.flatten(2).transpose(1, 2).reshape(-1, LATENT.detail_ch)
    n = s.shape[0]
    if n == 0:
        return torch.zeros((), dtype=semantic.dtype, device=semantic.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:min(n_samples, n)].to(s.device)
    s, d = s[idx], d[idx]

    s = s - s.mean(0, keepdim=True)
    d = d - d.mean(0, keepdim=True)
    s = s / (s.std(0, keepdim=True) + 1e-6)
    d = d / (d.std(0, keepdim=True) + 1e-6)

    cross = (s.T @ d) / max(1, s.shape[0] - 1)      # (Cs, Cd)
    return (cross ** 2).mean()


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check_shapes(image_sizes=(256, 512, 1024)) -> list:
    """形状/ token 数自检，返回行列表。"""
    rows = []
    for s in image_sizes:
        side = s // LATENT.spatial
        tk = LATENT.tokens(s)
        rows.append({
            "image": f"{s}x{s}",
            "latent": f"{side}x{side}",
            "channels": LATENT.total_ch,
            "tokens": tk,
            "expected": side * side,
            "ok": tk == side * side,
        })
    return rows

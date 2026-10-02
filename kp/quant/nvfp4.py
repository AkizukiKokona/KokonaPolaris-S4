"""NVFP4 模拟量化（可微，STE 直通梯度）。

⭐ 为什么手写而不是用 ModelOpt（实测根因，见补充 09 §8）：
  · ModelOpt 0.46.0 的 NVFP4/FP8 fake-quant **不提供反向**；
  · 真根因是插件把 SDPA 换成了**只实现 forward 的 `FP8SDPA`**，与量化器开关无关；
  · 且全参 QAT 在本机不可行（AdamW 状态 ≈ 12.8GB）⇒ 必须 adapter 式 QAD。
  ⇒ 训练期只需要「模拟量化」，不需要 packed/scale 布局 → 手写 + STE 即可，完全可微。

数值规格（与官方 `NVFP4_DEFAULT_CFG` 对齐）：
  · 权重：**E2M1 4-bit**，per-**block=16** absmax → **E4M3** scale（effective 4.5 bit）
  · 激活：设计档 **W4A8**（E4M3，256 级）。⚠️ 官方默认是 W4A4；E5 实测误差几乎
    全来自激活量化（激活降 4bit 再 +21.98 个百分点，3.84×）→ 我们选 FP8 激活。
  · `proj_out` **不量化**（官方默认；实测最敏感）

⚠️ 两条永久规范（补充 09 §5）：
  ① 逐像素 PSNR/SSIM 只用于「同轨迹复现性」，**不可判画质** → 判据须分布级；
  ② PTQ 是模拟量化，其**峰值显存不能论证 NVFP4 的显存收益**。

✅ 本实现与 `tools/e5b_qad.py` 的 `quant_fp4_ste` **逐位一致**（自检第 9 节对拍）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

_E4M3 = torch.finfo(torch.float8_e4m3fn)
DEFAULT_BLOCK = 16
FP4_MAX = 6.0          # E2M1 可精确表示的最大值


# ---------------------------------------------------------------------------
# 基本算子
# ---------------------------------------------------------------------------
def quant_fp4(x: torch.Tensor, blk: int = DEFAULT_BLOCK, *, ste: bool = True) -> torch.Tensor:
    """NVFP4：per-block(absmax) → E4M3 scale → E2M1 取整(±6)。

    ste=True 时梯度直通（`x + (xq - x).detach()`），用于 QAD 训练；
    ste=False 时为纯前向（PTQ / 误差测量）。
    """
    shape = x.shape
    inn = shape[-1]
    if inn % blk != 0:
        blk = inn
    xb = x.reshape(-1, inn // blk, blk)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / FP4_MAX).to(torch.float8_e4m3fn).to(x.dtype).clamp(min=1e-12)
    q = (xb / scale).round().clamp(-FP4_MAX, FP4_MAX)
    xq = (q * scale).reshape(shape)
    return x + (xq - x).detach() if ste else xq


def quant_fp8(x: torch.Tensor, *, ste: bool = True) -> torch.Tensor:
    """FP8 E4M3 per-tensor 模拟量化。"""
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / _E4M3.max).clamp(min=1e-12)
    q = (x / scale).clamp(_E4M3.min, _E4M3.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x).detach() if ste else q


# 便捷别名（对齐参考脚本的命名习惯）
quant_fp4_ste = lambda x, blk=DEFAULT_BLOCK: quant_fp4(x, blk, ste=True)   # noqa: E731
quant_fp8_ste = lambda x: quant_fp8(x, ste=True)                          # noqa: E731


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """‖a−b‖ / ‖b‖（相对误差，补充 09 的主口径）。"""
    return float(((a - b).norm() / (b.norm() + 1e-12)).item())


# ---------------------------------------------------------------------------
# 量化规格（供宿主层使用）
# ---------------------------------------------------------------------------
@dataclass
class QuantSpec:
    """一个线性层的量化规格。

    ⚠️ 注入点（能力包）**永远在量化器之外** —— 见 `kp/capability/bus.py`：
       `y = F.linear(q(x), q(W)) + Σ pack(x)`，包看到的是**未量化**的激活。
    """
    weight: str = "fp4"          # fp4 | none
    act: str = "fp8"             # fp8 | fp4 | none
    block: int = DEFAULT_BLOCK
    ste: bool = True

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.weight == "none":
            return w
        if self.weight == "fp4":
            return quant_fp4(w, self.block, ste=self.ste)
        raise ValueError(f"未知权重精度 {self.weight!r}")

    def quantize_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "none":
            return x
        if self.act == "fp8":
            return quant_fp8(x, ste=self.ste)
        if self.act == "fp4":
            return quant_fp4(x, self.block, ste=self.ste)
        raise ValueError(f"未知激活精度 {self.act!r}")

    def describe(self) -> dict:
        return {"weight": self.weight, "act": self.act, "block": self.block,
                "ste": self.ste,
                "effective_weight_bits": 4.0 + 8.0 / self.block if self.weight == "fp4" else 16.0}


# 设计档（CONFIG 里的 QUANT）：W4A8
DESIGN_SPEC = QuantSpec(weight="fp4", act="fp8", block=DEFAULT_BLOCK)
# 官方默认档：W4A4
OFFICIAL_DEFAULT_SPEC = QuantSpec(weight="fp4", act="fp4", block=DEFAULT_BLOCK)


# ---------------------------------------------------------------------------
# 误差诊断
# ---------------------------------------------------------------------------
def precision_table(w: torch.Tensor) -> list:
    """对同一权重比较各精度的相对误差（用于复核 E5 的排序结论）。"""
    rows = []
    for name, fn in (("bf16（基线）", lambda t: t),
                     ("fp8 E4M3（per-tensor）", lambda t: quant_fp8(t, ste=False)),
                     ("nvfp4 E2M1（block16）", lambda t: quant_fp4(t, 16, ste=False)),
                     ("nvfp4 E2M1（block32）", lambda t: quant_fp4(t, 32, ste=False))):
        rows.append({"精度": name, "相对误差": relative_error(fn(w), w)})
    return rows


__all__ = ["quant_fp4", "quant_fp8", "quant_fp4_ste", "quant_fp8_ste",
           "relative_error", "QuantSpec", "DESIGN_SPEC", "OFFICIAL_DEFAULT_SPEC",
           "precision_table", "DEFAULT_BLOCK"]

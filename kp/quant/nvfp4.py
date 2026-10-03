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

import torch

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


def spec_from_config(cfg) -> QuantSpec:
    """⭐ **从 `config.QuantCfg` 构造规格 —— 让 `QUANT` 成为真正的单一真源。**

    🔴 **为什么用「注入」而不是 `from ..config import QUANT`**
    （这是本函数存在的理由，不要改成直接 import）：

        `nvfp4.py` 是**纯数值算子层**，它不知道 DiT、latent、能力总线这些概念。
        若它 `import config`，就把「数值规格」和「架构常量」绑在一起 ——
        后果是：想单独测一个量化算子（不搭整个模型）就必须先 import 整个架构配置。

        ⇒ 约定：**算子层只认 `QuantSpec`；由宿主层（`qad.py`）负责把 `QUANT` 翻译成 `QuantSpec`。**
        这样 `QUANT` 改了就**真的会变**（不再是死旋钮），但依赖方向仍是单向的（架构 → 算子）。

    字段映射（含**哪些字段刻意不接线**）：

        weight_bits=4 / weight_fmt="E2M1" → `weight="fp4"`
            ⚠️ **只支持 fp4**：E2M1 是 NVFP4 的**格式定义**，不是可调项。
               若 `weight_bits` 改成 8，这里**直接报错**而不是悄悄忽略 ——
               「静默忽略配置」正是死旋钮的原始病因。
        act_bits=8 / act_fmt="E4M3"       → `act="fp8"`（`E4M3` 是 fp8 唯一格式）
        block_size=16                     → `block`
        scale_fmt="E4M3"                  → ⛔ **不接线**：`quant_fp4` 里 scale 恒为
               `torch.float8_e4m3fn`。这是 **NVFP4 格式的硬性规定**（E4M3 scale），
               不是可选项；接线它等于允许生成一个**不是 NVFP4** 的格式。
        quantize_proj_out=False           → ⛔ **不在本函数接线**（属"哪些层被量化"，
               由 `qad.py` 的 `SKIP_DEFAULT` 决定，见该文件）。
        sm_arch=120                       → ⛔ **不接线**：纯环境标注，无执行点。
    """
    if int(cfg.weight_bits) != 4 or str(cfg.weight_fmt).upper() != "E2M1":
        raise ValueError(
            f"本实现只支持 NVFP4 的 4-bit/E2M1 权重（config 给的是 "
            f"{cfg.weight_bits}-bit/{cfg.weight_fmt}）。"
            f"⚠️ 故意**不静默忽略** —— 静默忽略就是死旋钮的病因。")
    if int(cfg.act_bits) == 8:
        act = "fp8"
    elif int(cfg.act_bits) == 4:
        act = "fp4"
    else:
        raise ValueError(f"未知激活位宽 {cfg.act_bits}（支持 8 / 4）")
    if str(cfg.scale_fmt).upper() != "E4M3":
        raise ValueError(
            f"NVFP4 的 scale 格式**固定**为 E4M3（格式规定，非可调项），"
            f"config 给的是 {cfg.scale_fmt}")
    return QuantSpec(weight="fp4", act=act, block=int(cfg.block_size))


# 官方默认档：W4A4（**不由 config 驱动** —— 它是"官方默认"的参照，不是我们的设计档）
OFFICIAL_DEFAULT_SPEC = QuantSpec(weight="fp4", act="fp4", block=DEFAULT_BLOCK)


# 🔴🔴 「`QUANT` 是死旋钮」的修复 —— 2026-10-03 全局审查
#
# 审查发现：`QuantCfg` 的全部 8 个字段**唯一读者是 `arch_report.py` 的 print**，
#   真正的执行路径全部是写死的常量（`DEFAULT_BLOCK=16` / `FP4_MAX=6.0` / `SKIP_DEFAULT`）
#   ⇒ 改 `config.QUANT` 只会改一行打印输出，**量化行为零变化**。
#
# ✅ 修法（**不是**简单 `from ..config import QUANT`，理由见 `spec_from_config` docstring）：
#   ① 加 `spec_from_config(cfg)` —— 把 `QuantCfg` 翻译成 `QuantSpec`（真接线，含严格校验）；
#   ② 本常量**由 `config.QUANT` 实际构造**（不再是独立写死的字面量）；
#   ③ 加自检 `nvfp4_design_spec_matches_config`（见 `kp/selftest.py` §9）
#      —— 断言本常量与 `config.QUANT` **逐字段一致**，任何一侧改动不同步都会立刻报错。
#
# ⚠️ 为什么保留成「模块级常量」而不是函数：`DESIGN_SPEC` 被 `probe/real.py` 与
#   `train/qad.py` 用作**函数默认参数值**，改成惰性求值会破坏它们的签名。
#   ⇒ 折中：常量在**导入时**从 config 构造（真接线），一致性由自检持续守卫。
try:                                    # 惰性导入：避免算子层硬依赖架构配置
    from ..config import QUANT as _QUANT
    DESIGN_SPEC = spec_from_config(_QUANT)
except Exception:                       # pragma: no cover - 纯数值工具可独立使用
    DESIGN_SPEC = QuantSpec(weight="fp4", act="fp8", block=DEFAULT_BLOCK)


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

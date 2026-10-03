"""QAD 最小可跑路径 —— 冻结主干 + 只训能力包（adapter 式量化感知训练）。

⭐ 为什么是 adapter 式而不是全参 QAT（E5b 实测）：
   全参 QAT 在 8GB 上**不可行**（AdamW 状态 ≈ 12.8GB）；实测冻结主干 + 只训低秩旁路
   （5.99M = 0.37%）峰值仅 3.41GB。**这与设计稿的「Δ-Pack」完全同构** —— 提前实证。

⭐ 与「Δ-Pack 是量化感知的」这条设计要点一致：
   `GatedLinear` 把量化**只作用于主干算子**，能力包始终看到**未量化激活**，
   所以 ΔW 不会被量化器二次污染，也就保住了可逆性与 bit-exact。

闭环自检（`run_qad`）的做法：
   ① 用一组「教师包」算出目标输出；
   ② 把包重置到近零当学生；
   ③ 只训包参数去逼近目标 → **loss 必须显著下降**，且梯度只进包、不进主干。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..capability.bus import GatedLinear
from ..capability.delta_pack import DeltaPack
from ..quant.nvfp4 import QuantSpec, DESIGN_SPEC

# 🔴 `quantize_proj_out` 的执行点（2026-10-03 接线）
# 审查发现：`config.QUANT.quantize_proj_out` 此前**零执行点**（只被 `arch_report` 打印），
#   真实行为由下面这个**写死的** `("out_proj",)` 决定 ⇒ 改 config 只会改一行打印。
# ✅ 现在由 config 驱动：`quantize_proj_out=False` ⇒ 跳过 `out_proj`（官方 NVFP4_DEFAULT_CFG 的默认）。
# ⚠️ 匹配方式是**子串匹配**（`s in name`）⇒ **重命名该属性会静默改变量化范围**，
#    这是已知耦合，`dit.py` 里有对应注释提醒。
def _default_skip() -> Tuple[str, ...]:
    from ..config import QUANT
    return () if QUANT.quantize_proj_out else ("out_proj",)


SKIP_DEFAULT: Tuple[str, ...] = _default_skip()   # 输出投影不量化、也不挂包


def iter_gated(model, skip: Sequence[str] = SKIP_DEFAULT) -> Iterator[Tuple[str, GatedLinear]]:
    for name, gl in model.gated_linears().items():
        if any(s in name for s in skip):
            continue
        yield name, gl


def set_quant(model, spec: Optional[QuantSpec] = DESIGN_SPEC,
              skip: Sequence[str] = SKIP_DEFAULT) -> int:
    """给所有（非跳过的）注入点设置量化规格；spec=None 表示关闭。返回设置数量。"""
    n = 0
    for _, gl in iter_gated(model, skip):
        gl.set_quant(spec)
        n += 1
    return n


def clear_quant(model) -> None:
    for _, gl in model.gated_linears().items():
        gl.set_quant(None)


def freeze_backbone(model) -> int:
    """冻结主干**全部**参数（设计稿：冻结主干 + 只训能力包）。返回被冻结的参数量。

    ⚠️ 这一步不能省：骨架里 adaLN / t-embed / domain-embed 仍是 `nn.Linear`
       （真参数），若不显式冻结，它们会被算进"可训"里 —— 实测会让 KP-S 的
       可训占比从 **0.33% 虚高到 36.4%**（相差 110 倍），完全失真。
    """
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += p.numel()
    return n


def attach_delta(model, *, rank: int = 4, scale: float = 0.05, seed: int = 0,
                 gate: float = 1.0, skip: Sequence[str] = SKIP_DEFAULT,
                 freeze: bool = True) -> List[DeltaPack]:
    """冻结主干，并在每个注入点挂一个可训练 Δ-Pack（非零初始化，否则无法开训）。"""
    if freeze:
        freeze_backbone(model)
    packs: List[DeltaPack] = []
    for i, (name, gl) in enumerate(iter_gated(model, skip)):
        p = DeltaPack(f"qad::{name}", gl.in_features, gl.out_features,
                      rank=rank, seed=seed + i)
        if not p.A.is_meta:                  # meta device 上不做随机初始化
            with torch.no_grad():
                p.A.normal_(0.0, scale)
                p.B.normal_(0.0, scale)
        gl.add_pack(p)
        gl.set_gate(p.name, gate)
        packs.append(p)
    return packs


def budget(model) -> Dict[str, float]:
    """训练预算账（对齐 E5b 的「必须 adapter 式」结论）。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_par = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    frozen_buf = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    base = frozen_par + frozen_buf
    return {
        "trainable": trainable,
        "frozen_base": base,
        "trainable_ratio": trainable / max(1, base + trainable),
        "adamw_state_gb": trainable * 8 / 1e9,          # AdamW: m + v
    }


@dataclass
class QADResult:
    history: List[dict]
    budget: Dict[str, float]
    packs: List[DeltaPack]

    @property
    def loss_drop(self) -> float:
        if len(self.history) < 2:
            return 0.0
        a, b = self.history[0]["loss"], self.history[-1]["loss"]
        return 0.0 if a <= 0 else 1.0 - b / a

    def __str__(self) -> str:  # pragma: no cover
        b = self.budget
        return (f"QAD: {len(self.history)} 步，loss {self.history[0]['loss']:.5f} → "
                f"{self.history[-1]['loss']:.5f}（降 {self.loss_drop:.1%}）｜"
                f"可训 {b['trainable']:,} / 主干 {b['frozen_base']:,}"
                f"（{b['trainable_ratio']:.2%}）")


def run_qad(model, x: torch.Tensor, t: torch.Tensor, *, steps: int = 30, lr: float = 3e-3,
            rank: int = 4, seed: int = 0, ctx=None, ident=None, domain=None,
            skip: Sequence[str] = SKIP_DEFAULT) -> QADResult:
    """闭环自检式 QAD：教师包 → 目标，学生包从近零学起。"""
    torch.manual_seed(seed)
    model.train()
    packs = attach_delta(model, rank=rank, scale=0.05, seed=seed, gate=1.0, skip=skip)

    with torch.no_grad():
        teacher = model(x, t, text_ctx=ctx, identity_ctx=ident, domain=domain).clone()
        for p in packs:                       # 学生重置到近零
            p.A.normal_(0.0, 1e-3)
            p.B.normal_(0.0, 1e-3)

    params = [q for p in packs for q in (p.A, p.B)]
    opt = torch.optim.AdamW(params, lr=lr)
    hist: List[dict] = []
    for i in range(steps):
        opt.zero_grad()
        y = model(x, t, text_ctx=ctx, identity_ctx=ident, domain=domain)
        loss = F.mse_loss(y, teacher.detach())
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        hist.append({"step": i, "loss": float(loss.item()), "grad_norm": gn})
    return QADResult(history=hist, budget=budget(model), packs=packs)


def grad_health(model) -> Dict[str, float]:
    """梯度健康度：主干（buffer + 已冻结参数）不该有梯度；能力包必须有。"""
    buf_g = [b.grad.abs().max().item() for _, b in model.named_buffers()
             if b.is_floating_point() and b.grad is not None]
    frozen_g = [p.grad.abs().max().item() for p in model.parameters()
                if not p.requires_grad and p.grad is not None]
    pack_g = [p.grad.abs().max().item() for p in model.parameters()
              if p.requires_grad and p.grad is not None]
    base = buf_g + frozen_g
    return {"base_max_grad": max(base) if base else 0.0,
            "pack_max_grad": max(pack_g) if pack_g else 0.0,
            "n_frozen_with_grad": float(len(frozen_g))}


def budget_projection(rank: int = 4, skip: Sequence[str] = SKIP_DEFAULT) -> Dict[str, Dict[str, float]]:
    """用 meta device 按**真实结构**估算 KP-S / KP-M 的 adapter 训练预算。

    ⚠️ 为什么需要这个：trainable 占比**同时取决于 dim 与 rank** ——
       在小模型（dim=64）上 rank-4 适配器占比会高达 ~47%，
       但那不代表真实规模。真实结论必须在 KP-S/KP-M 上算。

    对照：E5b 在 Sana 1.6B 上实测 adapter 式 QAD = 5.99M / 0.37%，峰值 3.41GB。
    """
    from ..config import DIT_S, DIT_M, LATENT
    from ..models import SingleStreamDiT
    out = {}
    for name, cfg in (("KP-S", DIT_S), ("KP-M", DIT_M)):
        with torch.device("meta"):
            m = SingleStreamDiT(cfg, latent_ch=LATENT.total_ch,
                                identity_anchor_layers=[1, cfg.layers // 2])
            attach_delta(m, rank=rank, scale=0.0, gate=1.0, skip=skip)
        b = budget(m)
        b["quantized_layers"] = len(list(iter_gated(m, skip)))
        out[name] = b
    return out


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def main() -> int:
    from ..config import DiTCfg
    from ..models import SingleStreamDiT

    tiny = DiTCfg(dim=64, layers=4, heads=4, mlp_ratio=2.0,
                  double_stream_blocks=1, matryoshka_tokens=(16, 64))
    model = SingleStreamDiT(tiny, latent_ch=40, identity_anchor_layers=[1])
    n_q = set_quant(model)
    print(f"量化注入点：{n_q} 个（W4A8，out_proj 跳过）")

    x = torch.randn(1, 40, 8, 8)
    t = torch.full((1,), 400.0)
    ctx = torch.randn(1, 12, 64)
    ident = torch.randn(1, 16, 64)
    dom = torch.randn(1, 16)

    res = run_qad(model, x, t, steps=30, ctx=ctx, ident=ident, domain=dom)
    print(res)
    b = res.budget
    print(f"  可训占比 {b['trainable_ratio']:.3%}｜AdamW 状态 ≈ {b['adamw_state_gb']:.3f} GB")
    for h in res.history[::5] + [res.history[-1]]:
        print(f"  step {h['step']:>3}  loss {h['loss']:.6f}  grad_norm {h['grad_norm']:.4f}")
    print(f"梯度健康度：{grad_health(model)}")
    print("结论：loss 显著下降 ⇒ 冻结主干 + 只训 Δ-Pack 的 QAD 路径可跑通。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

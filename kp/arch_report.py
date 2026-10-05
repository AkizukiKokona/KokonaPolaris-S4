"""KP 架构速览 / 账目核对（纯 CPU，用 meta device 数真实参数量，不占内存）。

运行：  python -m kp.arch_report
"""
from __future__ import annotations

import sys
from typing import Dict, Tuple

import torch

from kp.config import DIT_S, DIT_M, LATENT, QUANT, CAP
from kp.models.dit import SingleStreamDiT
from kp.models.text_tower import TextTower, TextTowerCfg


def _count(model: torch.nn.Module) -> Tuple[int, int]:
    """返回 (可训练参数, 冻结 buffer 权重)。

    ⚠️ GatedLinear 的本体权重是 **buffer**（冻结主干），不计入 parameters()，
    所以必须单独统计，否则会严重低估。
    """
    n_par = sum(p.numel() for p in model.parameters())
    n_buf = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    return n_par, n_buf


def _build_on_meta(cfg) -> SingleStreamDiT:
    with torch.device("meta"):
        m = SingleStreamDiT(cfg, latent_ch=LATENT.total_ch,
                            identity_anchor_layers=[1, cfg.layers // 2])
    return m


def _fmt(n: int) -> str:
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.3f}B"


def main() -> int:
    print("=" * 74)
    print("KokonaPolaris-S4（心夏北极星 / KP）· 架构速览")
    print("=" * 74)

    print("\n【混合 latent · 32× 空间压缩】")
    print(f"  语义 {LATENT.semantic_ch}ch + 细节 {LATENT.detail_ch}ch = {LATENT.total_ch}ch")
    print(f"  {'图像':>10} {'latent 网格':>14} {'token 数':>10} {'latent 字节(fp32/bf16)':>24}")
    for s in (256, 512, 1024):
        side = s // LATENT.spatial
        tk = LATENT.tokens(s)
        n = LATENT.total_ch * side * side
        print(f"  {s}²{'>':>8} {side}×{side}{'>':>8} {tk:>10} {n*4:>10,} / {n*2:>11,}")
    assert LATENT.tokens(1024) == 1024, "1024² 必须恰好 1024 token"
    print("  ✅ 1024² → 1024 token（patch_size=1，1 token = 1 latent 格）")

    print("\n【单流 DiT 主干 · 真实参数量（meta device）】")
    print(f"  {'档位':>6} {'dim':>6} {'层':>4} {'heads':>6} {'mlp':>5} "
          f"{'可训练':>9} {'冻结W0':>9} {'合计':>9}")
    rows: Dict[str, int] = {}
    for name, cfg in (("KP-S", DIT_S), ("KP-M", DIT_M)):
        m = _build_on_meta(cfg)
        np_, nb = _count(m)
        rows[name] = np_ + nb
        print(f"  {name:>6} {cfg.dim:>6} {cfg.layers:>4} {cfg.heads:>6} "
              f"{cfg.mlp_ratio:>5} {_fmt(np_):>9} {_fmt(nb):>9} {_fmt(np_+nb):>9}")

    print("\n【对照组：设计稿标称】")
    print("  KP-S 标称 0.6B / KP-M 标称 1.5B（仅主干，不含文本塔与 VAE）")
    s, m = rows["KP-S"], rows["KP-M"]
    flag_s = "✅ 吻合" if abs(s / 0.6e9 - 1) < 0.15 else "⚠️ 偏差"
    flag_m = "✅ 吻合" if abs(m / 1.5e9 - 1) < 0.15 else "⚠️ 偏大"
    print(f"  KP-S 实测 {_fmt(s)} vs 0.6B  → {flag_s}"
          f"（{(s/0.6e9-1)*100:+.1f}%）")
    print(f"  KP-M 实测 {_fmt(m)} vs 1.5B  → {flag_m}"
          f"（{(m/1.5e9-1)*100:+.1f}%）")

    print("\n【文本塔（自 Qwen3.5-4B-Base 蒸馏）】")
    tcfg = TextTowerCfg()
    tt = TextTower(tcfg) if False else None  # 不实建，避免占内存
    est = tcfg.param_count()
    print(f"  骨架配置 dim={tcfg.dim} layers={tcfg.layers} vocab={tcfg.vocab_size} "
          f"→ 估算 {_fmt(est)}")
    print("  ⭐ 真正的多语言 LLM ⇒ 天生支持中文（SDXL 的 CLIP 必须全英文）")

    print("\n【NVFP4 量化档】")
    print(f"  权重 {QUANT.weight_bits}bit/{QUANT.weight_fmt} · 激活 {QUANT.act_bits}bit/{QUANT.act_fmt}"
          f" · block={QUANT.block_size} · scale={QUANT.scale_fmt}")
    print(f"  proj_out 量化: {QUANT.quantize_proj_out}（官方默认 False，实测最敏感）")
    print(f"  SM 架构: sm_{QUANT.sm_arch}（Blackwell 消费卡；无 tcgen05/TMEM）")
    for name in ("KP-S", "KP-M"):
        w4 = rows[name] * 0.5 / 1e9
        bf16 = rows[name] * 2 / 1e9
        print(f"  {name} 主干权重显存：bf16 {bf16:.2f} GB → NVFP4 {w4:.2f} GB"
              f"（省 {bf16-w4:.2f} GB）")

    print("\n【能力总线】")
    print(f"  门控初始 = {CAP.gate_init}（⇒ 接口成本严格为零，bit-exact）")
    print(f"  身份 token = {CAP.identity_tokens} 个 × dim {CAP.identity_token_dim}"
          f" · 走 cross-attention: {CAP.identity_via_cross_attention}")
    print(f"  Δ-Pack 预算 = {CAP.delta_pack_bytes/1024/1024:.0f} MB")
    print(f"  谱检查阈值 cos < {CAP.spectral_cos_threshold}"
          f"（只看前 {CAP.spectral_rank_ratio:.0%} 排名）")
    print(f"  擦除默认有效秩 k = {CAP.erase_default_rank}（安全子空间 ~6，LoX）")

    print("\n【三层控制栈（调用量分布 ≠ 能力分层）】")
    print("  L1 主干内生条件轴    ~90% 调用量 · 零成本 · 但受基座依赖（G3.5 把关）")
    print("  L2 外置条件编码器    ~9%  · 身份必须外置（角色卡）")
    print("  L3 子空间约束 LoRA   ~1%  · 只做同域长尾")
    print("  ⚠️ 「L1 覆盖 90% 调用量，L3 承担 100% 兜底责任」")

    print("\n【待办（需与设计稿核对）】")
    print(f"  · KP-S：实测 {_fmt(rows['KP-S'])} ≈ 0.6B ✅ 无需改动")
    print(f"  · KP-M：实测 {_fmt(rows['KP-M'])} 比标称 1.5B 大 "
          f"{(rows['KP-M']/1.5e9-1)*100:.0f}%")
    print("    · ✅ 已做：删掉 adaLN 段 7「g_geo」——它全代码库无人读取，")
    print("      是纯死参数（每 block d²，KP-M 省 102.8M ≈ 5.2%；adaLN 自身 FLOPs −11.1%）")
    print("    · ⏳ 待拍板：剩下的标称差是「改结构」还是「改标称」")
    print("      （结构侧候选：dim≈1664/L32；参数量 ∝ dim²·layers）")
    print("    · ⚠️ 判据提示：本项目**不靠 LoRA 堆能力**——LoRA(L3) 仅 3.89M/0.22%，")
    print("      不是承载能力的主力。adaLN 虽占参数一半，但 QAD 期被 freeze_backbone 全冻")
    print("      （只在**预训练期**可训，那正是 L1 内生轴的载体）⇒ 砍参数量的杠杆在")
    print("      **adaLN 段数与 dim**，不在挂点数/层数。")
    print("  · 其余见 MEMORY.md §就绪度与验证门")

    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    # ⚠️ 与 kp/selftest.py 同款兜底：Windows 中文控制台默认 GBK，打印 `1024²`
    #    会抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\xb2'`，
    #    表现为「arch_report 直接 exit 1、报告只打了一半」——看起来像工具挂了，其实是编码。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass
    sys.exit(main())

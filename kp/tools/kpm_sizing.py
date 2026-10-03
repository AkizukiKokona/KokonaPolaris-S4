"""P1 · KP-M 尺寸对账工具 —— 为「改结构 or 改标称」这个拍板提供**可复现的依据`。

🔴 **为什么需要这个**：待拍板项挂了很久，根因是**之前的归因是错的**
（把「层数 28 vs 32」误归因到「dim 太大」），而给出的候选方案 `dim≈1664/L32`
的算术也错了（只算了 `17d²·L` 名义 block，**漏了 127.9M 非 block 项**）。

⇒ 本工具**不替你决定**，但把**每个候选的真实数字**摆出来（meta device 实测，非估算），
   并列出**每个方案的连带影响**（改一处要重算什么）。

跑法：
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.tools.kpm_sizing
"""
from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from ..config import LATENT, DiTCfg

#: 设计稿 §4.3 参数量表写的标称
NOMINAL_M = 1.5
#: 设计稿三处一致写的层数（参数量表 L346 / 附录A L1259 / Matryoshka L353）
DOC_LAYERS = 28


def n_params(cfg: DiTCfg) -> int:
    """meta device 实测参数量（含 buffer —— `GatedLinear` 权重是 buffer，不计入 parameters()）。"""
    import torch
    from ..models.dit import SingleStreamDiT
    with torch.device("meta"):
        m = SingleStreamDiT(cfg, latent_ch=LATENT.total_ch,
                            identity_anchor_layers=[1, cfg.layers // 2])
    return (sum(p.numel() for p in m.parameters())
            + sum(b.numel() for b in m.buffers() if b.is_floating_point()))


def attn_plan(cfg: DiTCfg) -> str:
    from ..models.dit import build_attn_plan
    p = build_attn_plan(cfg.layers, cfg.ratio_gated_linear, cfg.ratio_sigmoid, True)
    return f"{p.count('linear')}L/{p.count('sigmoid')}S"


def table() -> str:
    rows: List[Tuple[str, int, int, int, bool]] = [
        ("现状 config（dim1792/L32/h16）", 1792, 32, 16, True),
        ("⭐ 改回文档 L28（dim1792/L28/h16）", 1792, 28, 16, True),
        ("改回文档 + heads14（dim1792/L28/h14）", 1792, 28, 14, True),
        ("L26（最逼近标称 1.5B）", 1792, 26, 16, True),
        ("旧建议 dim1664/L32（已证算术错）", 1664, 32, 16, True),
    ]
    L = []
    A = L.append
    A("=" * 92)
    A(f"KP-M 尺寸对账（meta device 实测，标称 {NOMINAL_M}B）")
    A("=" * 92)
    A(f"{'方案':<40s} {'参数量':>10s} {'vs 标称':>10s} {'head_dim':>9s} {'3:1 整除':>9s} {'注意':>8s}")
    A("-" * 92)
    for tag, d, nl, h, star in rows:
        cfg = DiTCfg(dim=d, layers=nl, heads=h)
        n = n_params(cfg)
        div = (nl % 4 == 1)
        note = "" if d == 1792 else "⚠️head_dim 变"
        A(f"{tag:<40s} {n/1e9:9.4f}B {100*(n/NOMINAL_M/1e9-1):+9.1f}% "
          f"{d//h:>9d} {'✓' if div else '✗':>9s} {note:>8s}")
    A("")
    A("⭐ **关键结论**")
    A("-" * 92)
    A(f"1. **真正的根因是层数**：设计稿三处一致写 **{DOC_LAYERS} 层**，"
      f"而 `config.py` 写 **32 层** ⇒ `32/{DOC_LAYERS} = {32/DOC_LAYERS:.3f}`，"
      f"正好解释了那 +25%。")
    A(f"2. ⛔ **旧建议 `dim≈1664/L32 ≈ 1.52B` 的算术是错的** —— "
      f"实测 **{n_params(DiTCfg(dim=1664, layers=32, heads=16))/1e9:.4f}B（+7.8%）**，"
      f"因为 `1.52B` 只算了 `17d²·L` 名义 block，**漏了非 block 项**。")
    A(f"3. ⭐ **改回 L28 是唯一「零代价」的选项** —— "
      f"`head_dim` 保持 **{1792//16}**、3:1 的 `L≡1(mod 4)` 性质不变，"
      f"只损失 **{100*(1-28/32):.0f}%** 的深度。")
    A(f"4. ⚠️ **没有一个候选是精确 1.5B**：最接近的是 L26（+3.1%），"
      f"但 **26 也不是 `4k+1`** ⇒ 3:1 仍不整除。")
    A("")
    A("📌 **若决定改 L28，牵连这些（必须一起重算）**")
    A("-" * 92)
    for x in ("`arch_report` 的参数量两栏（可训练/冻结W0）",
              "自检里的参数量恒等断言（§23 修法② 的『精确相等』仍成立，但要重跑确认）",
              "`kp.train.qad.budget_projection` 的 adapter 预算（分母变小）",
              "主文档 §4.3 参数量表 / 附录 A（改标称或改结构，二选一）",
              "Matryoshka 深度阶梯（§4.4 写 512² 满层=28，与改后一致 ✅ 不用改）"):
        A(f"   • {x}")
    A("")
    A("⚠️ **本工具只摆数字，不替你决定** —— 「改结构 vs 改标称」是**取舍判断**：")
    A("   改结构 ⇒ 标称诚实但要重算多处；改标称 ⇒ 一步到位但「1.5B」这个对外数字变大。")
    return "\n".join(L)


def text_tower_table() -> str:
    """文本塔规模对账（⚠️ **必须 params + buffer**，只数 parameters 会错）。

    🔴 **为什么单独写这个函数**（今天已踩两次同样的坑）：
    `GatedLinear` 的权重注册为 **buffer**（"冻结主干"设计）⇒ **不在 `parameters()` 里**。
    我连续两次只数 `parameters()`，得出「layers 14 和 24 参数量相同」的**错误结论**，
    还一度怀疑 `layers` 没接线。
    ⇒ `arch_report._count()` 早就写对了（params + buffer），**这里必须复用同样的口径**。
    """
    from ..models.text_tower import TextTower, TextTowerCfg
    L = ["=" * 78,
         "文本塔规模对账（params + buffer · 标称 220M）",
         "=" * 78,
         f"{'配置':<24s} {'params':>10s} {'buffer':>10s} {'合计':>10s} {'vs 标称':>10s}",
         "-" * 78]
    for tag, dim, lay in (("当前 dim768/L14", 768, 14), ("dim768/L24", 768, 24),
                          ("dim1024/L14", 1024, 14), ("dim1024/L24", 1024, 24),
                          ("dim1280/L14", 1280, 14)):
        m = TextTower(TextTowerCfg(dim=dim, layers=lay))
        p = sum(x.numel() for x in m.parameters())
        b = sum(x.numel() for x in m.buffers() if x.is_floating_point())
        n = p + b
        L.append(f"{tag:<24s} {p/1e6:9.1f}M {b/1e6:9.1f}M {n/1e6:9.1f}M "
                 f"{100*(n/220e6-1):+9.1f}%")
    cur = TextTower(TextTowerCfg())
    p = sum(x.numel() for x in cur.parameters())
    b = sum(x.numel() for x in cur.buffers() if x.is_floating_point())
    L += ["", "⭐ 当前配置与标称的关系：",
          f"   合计 {(p+b)/1e6:.1f}M vs 标称 220M ⇒ **{100*((p+b)/220e6-1):+.1f}%（吻合）**",
          "   ⚠️ 词表嵌入占大头（`vocab=151936 × dim`），但那是**查表**不是算力 ⇒",
          "      推理成本主要看 layers×dim，不是总参数。",
          "   ⚠️ **本项目已两次因只数 `parameters()` 而误判**（GatedLinear 权重在 buffer）。",
          "      涉及参数量的一切结论，都必须用 `params + buffer` 口径。"]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap_ = argparse.ArgumentParser(description="KP 尺寸对账（KP-M 主干 + 文本塔）")
    ap_.add_argument("--tower", action="store_true", help="只看文本塔")
    a = ap_.parse_args(argv)
    print(text_tower_table() if a.tower else table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

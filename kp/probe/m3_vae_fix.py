"""🔬 M3 VAE 侧解法预研：**分块重建**（结构约束，不是损失权重）

═══ 要解决的问题（已实测）═══

    我们 40ch（语义8|细节32）：随机 encoder `cross_r2 = 0.2876`
                                →训练后 **0.9997**
    DC-AE 32ch（训练充分）：        **0.4375**

⇒ **训练把冗余做满了**。根因：**重建损失只看「图」，不管两块怎么分工** ⇒
   对编码器而言「两块都放全部信息」就是最优解（latent 各维之间**没有竞争**，不必分工）。

⚠️ **为什么修法①（跨块去相关 penalty）大概无效**：
   penalty 与重建损失**方向相反**，重建压力更大 ⇒ 会被按回去。

═══ 本文件试的结构解法（不靠"说服"损失函数）═══

    **分块重建**：decoder 重建时**只喂给它其中一块**。
    · 只喂语义 8ch ⇒ 细节块拿不到语义信息 ⇒ 想重建好语义**必须**放在语义块里
    · 只喂细节 32ch ⇒ 同理
    ⇒ **分工从「软偏好」变成「结构必需」** —— 冗余在结构上不再可能。

⚠️ **本文件是【预研】，不是成品**：
    ① 用**极小规模**（`base=8`、32×32 图、几十步）验证「机制上有没有效」，
       ⛔ **不追求重建质量**（那需要 P1 的数据量）
    ② 结论只回答：「**结构约束能否在训练中把冗余压下去**」
    ③ 真要训出可用 VAE，仍需解决数据量（见 CHECKPOINT 阻塞项 #1）

═══ 三组对照（同一份数据、同一套超参，只换约束）═══
    A 无约束        ⇒ 预期冗余高（复现"训练做满冗余"）
    B penalty       ⇒ 预期仍高（重建压力按回去）—— **验证修法① 到底行不行**
    C 分块重建     ⇒ 预期显著低 —— ⭐ 候选解法
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import LATENT
from ..latent.real_separation import _r2
from ..models.vae import HybridVAE


def cross_r2(vae: HybridVAE, x: torch.Tensor) -> float:
    vae.eval()
    with torch.no_grad():
        z = vae.encode_latent(x)
    return _r2(z[:, :LATENT.semantic_ch], z[:, LATENT.semantic_ch:])


def train_one(mode: str, *, n: int = 8, size: int = 32, steps: int = 60,
              base: int = 8, lr: float = 2e-3, w_penalty: float = 1.0,
              seed: int = 0) -> Dict:
    """训一个小 VAE，测训练后的块间冗余。`mode` ∈ {none, penalty, blockwise}。"""
    torch.manual_seed(seed)
    sc = LATENT.semantic_ch
    # ⚠️ 合成数据只用于「验证机制」，**不能**当训练集（见 docstring 的 ⛔）
    g = torch.Generator().manual_seed(seed)
    base_imgs = torch.randn(n, 3, size, size, generator=g)
    # 加一点结构，使重建任务非平凡
    base_imgs[:, :, size // 4:3 * size // 4, size // 4:3 * size // 4] += 1.0

    vae = HybridVAE(base=base)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    for step in range(steps):
        rec = torch.tanh(vae.decode(vae.encode_latent(base_imgs)))
        l1 = (rec - base_imgs).abs().mean()
        loss = l1
        if mode == "penalty":
            z = vae.encode_latent(base_imgs)
            a, b = z[:, :sc], z[:, sc:]
            # ⭐ 逐**位置**算跨块相关（(C_a,C_b) 矩阵），不是 (B,B) —— 维度要对齐
            # ⚠️ latent 是 4D (N,C,H,W) ⇒ 先摊平成 (N,C,P)，P=H*W
            N, Ca, Hh, Ww = a.shape
            an = a.reshape(N, Ca, Hh * Ww)
            bn = b.reshape(N, b.shape[1], Hh * Ww)
            an = an / (an.norm(dim=2, keepdim=True) + 1e-6)
            bn = bn / (bn.norm(dim=2, keepdim=True) + 1e-6)
            g = torch.einsum("ncp,ndp->ncd", an, bn)            # (N, C_a, C_b)
            loss = loss + w_penalty * g.pow(2).mean()
        elif mode == "blockwise":
            # ⭐ 分块重建：解码时**只喂一块** ⇒ 两块无法互相代偿
            z = vae.encode_latent(base_imgs)
            rec_s = torch.tanh(vae.decode(
                torch.cat([z[:, :sc], torch.zeros_like(z[:, sc:])], 1)))
            rec_d = torch.tanh(vae.decode(
                torch.cat([torch.zeros_like(z[:, :sc]), z[:, sc:]], 1)))
            # 两块各自承担 1/2 的重建误差
            loss = 0.5 * (rec_s - base_imgs).abs().mean() \
                 + 0.5 * (rec_d - base_imgs).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"mode": mode, "cross_r2": round(cross_r2(vae, base_imgs), 4)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M3 VAE 侧解法预研（分块重建）")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--out", default="out/m3_vae_fix.json")
    a = ap.parse_args(argv)

    print("=" * 70)
    print("M3 VAE 侧解法预研（🔴 合成数据，只验机制不追质量）")
    print("=" * 70)
    rows = []
    for mode, tag in (("none", "A 无约束（基线）"),
                      ("penalty", "B 跨块 penalty（修法①）"),
                      ("blockwise", "C 分块重建（结构解法）")):
        r = train_one(mode, n=a.n, size=a.size, steps=a.steps)
        rows.append(r)
        print(f"  {tag:26s} 训练后 cross_r2 = {r['cross_r2']:.4f}")

    base = rows[0]["cross_r2"]
    pen = rows[1]["cross_r2"]
    blk = rows[2]["cross_r2"]
    print("\n" + "=" * 70)
    print(f"A 基线 {base:.4f} ｜ B penalty {pen:.4f} ｜ C 分块重建 {blk:.4f}")
    print("=" * 70)
    verdict = {
        "A_无约束": base, "B_penalty": pen, "C_分块重建": blk,
    }
    if pen < base * 0.7:
        print("🟢 修法①（penalty）在本规模下**有效**")
    else:
        print("🔴 修法①（penalty）**压不住** ⇒ 与「重建压力会按回去」的预判一致")
    if blk < min(base, pen) * 0.7:
        print("🟢 **分块重建（结构解法）显著更优** ⇒ 建议作为 M3 的主解法")
    else:
        print("🟡 分块重建未见明显优势 ⇒ 需调结构或换解法")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "⚠️_限定": "**合成数据 + 极小规模**（base=8/32²/60步）⇒ 只验证「机制上有没有效」，"
                   "⛔ **不代表**训出的 VAE 可用（重建质量需真数据，见 CHECKPOINT 阻塞#1）",
        "steps": a.steps, "n": a.n, "size": a.size, "结果": verdict,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

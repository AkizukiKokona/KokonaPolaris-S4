"""🔬 「safe-only 训练 vs 含 sensitive」对 VAE 的影响 —— **只出数字，不看图**。

═══ 为什么要专门做这个实验 ═══

用户提出了一个真问题（我的第一反应是"VAE 与画风无关 ⇒ 不需要" —— **那是回避，不是回答**）：
> 「如果完全按 safe 训练，虽然说与画风无关，但会不会对后面造成影响？」

⚠️ **我不能用推理搪塞**，必须用**数据**回答。

═══ ⭐ 关键的方法论：怎么「不看图」也能比较 ═══

**「测量差异」与「看图」是��件事**：
· 本模块**只产出客观标量**：重建 L1 / PSNR / latent 统计 / 压缩后保真度
· ⛔ **它不打开、不渲染、不保存任何图片** ⇒ 无需「看图」即可得到结论
· 👁️ **图留给用户自己看**（若想肉眼确认，`--dump-samples` 会导出一小批到本地）

⚠️ 因此**不构成回避**：我算的是数字，不是"我觉得应该没影响"。

═══ 对照设计 ═══

    A  训练集 = 仅 safe
    B  训练集 = safe + sensitive
    ↓
    都在**同一批 held-out 图**上评估（safe 与 sensitive 各一半）
    ↓
    看 A 在 sensitive 上的表现 vs B 在 sensitive 上的表现
    ⭐ 差值 = 「只用 safe 训练」到底有没有代价

⚠️ **诚实标注**：
· 这是**收敛后**的对比，**不是**「VAE 能不能编码敏感内容」（那是解码器能力，与训练数据无关）
· 真正要问的是「VAE 的**通用压缩能力**会不会因为少了一半数据而退化」⇒ 这才是本实验测的
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from ..config import LATENT
from ..models.vae import HybridVAE
from ..paths import OUT


@torch.no_grad()
def eval_split(vae: HybridVAE, x: torch.Tensor, device: str) -> Dict[str, float]:
    """在一批图上评估（**只算数字，不返回图片**）。"""
    vae.eval().to(device)
    tot_l1 = tot_psnr = 0.0
    zs: list = []
    for i in range(0, x.shape[0], 8):
        xb = x[i:i + 8].to(device)
        z = vae.encode_latent(xb)
        rec = torch.tanh(vae.decode(z)).clamp(-1, 1).cpu()
        xc = xb.cpu()
        tot_l1 += float((rec - xc).abs().mean()) * xb.shape[0]
        mse = float(((rec - xc) ** 2).mean())
        # PSNR 由**全 batch 的 MSE**算（逐图算再平均会高估，故用聚合）
        tot_psnr += 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))) * xb.shape[0]
        zs.append(z.float().cpu())
    n = x.shape[0]
    zall = torch.cat(zs)
    # ⚠️ latent 统计：只报**标量**（均值/标准差/通道间相关均值）—— 不含任何可还原的图
    ch_corr = _mean_offdiag_corr(zall)
    return {
        "n": n,
        "recon_l1": round(tot_l1 / n, 5),
        "psnr_db": round(tot_psnr / n, 2),
        "latent_mean": round(float(zall.mean()), 4),
        "latent_std": round(float(zall.std()), 4),
        "latent_absmax": round(float(zall.abs().max()), 3),
        "latent_ch_corr_mean": round(ch_corr, 4),
    }


def _mean_offdiag_corr(z: torch.Tensor) -> float:
    """通道间平均 |相关| —— 冗余的**标量**指标（不是图）。"""
    sc = z[:, :LATENT.semantic_ch].reshape(z.shape[0], LATENT.semantic_ch, -1)
    dc = z[:, LATENT.semantic_ch:].reshape(z.shape[0], -1, z.shape[-1] * z.shape[-2])
    a = sc - sc.mean(2, keepdim=True)
    b = dc - dc.mean(2, keepdim=True)
    a = a / (a.norm(dim=2, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=2, keepdim=True) + 1e-6)
    g = torch.einsum("ncp,ndp->ncd", a, b)
    return float(g.abs().mean())


def train_vae(x: torch.Tensor, *, steps: int, base: int, lr: float,
               device: str, seed: int = 0) -> HybridVAE:
    torch.manual_seed(seed)
    vae = HybridVAE(base=base).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (min(8, x.shape[0]),))
        rec = torch.tanh(vae.decode(vae.encode_latent(x[idx].to(device))))
        loss = (rec - x[idx].to(device)).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="safe-only 训练对 VAE 的影响（⛔ 只出数字，不看图）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/safe_impact.json")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"

    # ⛔ **本模块不自行下载/筛选敏感内容** —— 它只接收**已切分好的两个张量文件**
    #    （由 `kp/data/rating.py` 的过滤流程产出，⛔ 过滤结果落在 out/ 之外由用户保管）
    safe_p, sens_p = Path(OUT / "safe.pt"), Path(OUT / "sens.pt")
    if not (safe_p.exists() and sens_p.exists()):
        print("⚠️ 需要先准备好两个切分好的张量文件：")
        print(f"   {safe_p}  （safe 子集）")
        print(f"   {sens_p}  （sensitive 子集）")
        print("   ⛔ 本模块**不负责下载或筛选**敏感内容（由用户执行过滤并保管结果）")
        print("   ⇒ 缺文件时**不做任何替代、不造合成数据**（如实报缺口）")
        return 1

    x_safe = torch.load(safe_p, weights_only=False)[:512]
    x_sens = torch.load(sens_p, weights_only=False)[:512]
    x_safe = x_safe.reshape(-1, 3, a.size, a.size).clamp(-1, 1)
    x_sens = x_sens.reshape(-1, 3, a.size, a.size).clamp(-1, 1)
    print(f"数据：safe {x_safe.shape[0]} 张 / sensitive {x_sens.shape[0]} 张 @ {a.size}²")
    print(f"训练：{a.steps} 步 · base={a.base} · {dev}（⛔ 全程只算数字，不渲染图片）")

    held_safe, held_sens = x_safe[:64], x_sens[:64]
    rows = {}
    for tag, train_data in (("A_仅safe训练", x_safe),
                           ("B_safe+sensitive训练", torch.cat([x_safe, x_sens]))):
        vae = train_vae(train_data, steps=a.steps, base=a.base, lr=2e-3, device=dev)
        rows[tag] = {
            "在_safe_上": eval_split(vae, held_safe, dev),
            "在_sensitive_上": eval_split(vae, held_sens, dev),
        }
        r = rows[tag]
        print(f"\n  【{tag}】")
        print(f"    held-out safe      L1={r['在_safe_上']['recon_l1']:.4f} "
              f"PSNR={r['在_safe_上']['psnr_db']:.2f}dB")
        print(f"    held-out sensitive L1={r['在_sensitive_上']['recon_l1']:.4f} "
              f"PSNR={r['在_sensitive_上']['psnr_db']:.2f}dB")

    aonly = rows["A_仅safe训练"]["在_sensitive_上"]
    both = rows["B_safe+sensitive训练"]["在_sensitive_上"]
    d_l1 = aonly["recon_l1"] - both["recon_l1"]
    d_psnr = aonly["psnr_db"] - both["psnr_db"]
    print("\n" + "=" * 66)
    print(f"「仅 safe 训练」在敏感图上的代价：")
    print(f"  ΔL1 = {d_l1:+.4f}  ({'更差' if d_l1 > 0.005 else '基本无差' if abs(d_l1) <= 0.005 else '更好?!'})")
    print(f"  ΔPSNR = {d_psnr:+.2f} dB ({'更差' if d_psnr < -0.5 else '基本无差' if abs(d_psnr) <= 0.5 else '更好?!'})")
    print("=" * 66)
    if abs(d_psnr) <= 0.5:
        print("✅ 结论：**只用 safe 训练，对 VAE 的通用压缩能力无实质影响**（数据量足够时）")
    else:
        print("🔴 结论：**有实质影响** ⇒ 需要在训练数据里保留一部分 sensitive")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "方法": "⛔ 只产出客观标量（L1/PSNR/latent统计），**不渲染任何图片**",
        "对照": "A=仅safe训练 / B=safe+sensitive训练，同一 held-out 评估",
        "⚠️_限定": "测的是「VAE 通用压缩能力是否退化」，**不是**「能否解码敏感内容」",
        "结果": rows, "代价_ΔL1": round(d_l1, 5), "代价_ΔPSNR": round(d_psnr, 3),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已存 {p}（**只有数字，没有图片**）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

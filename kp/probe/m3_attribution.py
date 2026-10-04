"""🔴🔴 M3 归因实验：我们的 40ch 冗余，是**架构必然**还是**我们训练的问题**？

═══ 这个实验要回答什么 ═══

我们自己的 HybridVAE（40ch = 语义8 + 细节32）实测：
    `cross_r2_sem_from_detail = 0.988`（语义块 98.8% 可由细节块线性预测）
    且**训练越强越糟**（随机初始化 0.47 → 训练后 0.99）

⚠️ 当时无法判断这是「**32× 压缩本身**就注定冗余」还是「**我们没训好/监督不够**」。

⭐ **DC-AE f32c32 提供了判据的对照组**：
    它是**同一个压缩率（32×）**、**Apache-2.0**、**在海量通用图上训好的**真 VAE。
    ⇒ 若**它也冗余** ⇒ **架构必然**（32× 压缩的固有代价，不是我们的锅）
    ⇒ 若**它不冗余** ⇒ **我们的问题**（监督/训练/架构选型）
    ⇒ 这一条直接决定「M3 该不该继续投入」。

═══ 严格的做法 ═══

⚠️ **DC-AE 只有 32ch，没有语义/细节之分** ⇒ 不能直接算「跨块 R²」。
   但可以算**它自己的等价指标**：把 32ch 劈成「前 8 / 后 24」两半，
   看**后半能否线性预测前半** ⇒ 与我们的「语义 vs 细节」**同构可比**。

⚠️ 更重要的对照：**同一个模型（DC-AE）、同一批图、不同通道切法**
   ⇒ 能区分「冗余来自 latent 本身」还是「来自切法」。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

#: DC-AE 权重（Sana 的 32× VAE，Apache-2.0）
DCAE_CKPT = "models/dc_ae_f32c32_sana_1.0.safetensors"
DCAE_CONFIG = "models/dc_ae_f32c32"


def load_dcae(device: str = "cuda"):
    """加载 DC-AE。

    ⚠️ **必须用 `from_single_file` + 本地 config**（实测踩过）：
       · `from_pretrained` ⇒ 键名重叠 0（checkpoint 是 Sana 官方 `stages/op_list` 命名，
         diffusers 的 `AutoencoderDC` 是 `down_blocks/up_blocks`）⇒ 全部 meta ⇒ 直接崩；
       · `from_single_file(..., local_files_only=True)` ⇒ **成功**（它自带键名映射）。
    ⭐ 另注：输出是 `EncoderOutput(latent=...)`（**没有** `latent_dist`）
       ⇒ DC-AE 的 latent 是**确定性**的（与我们的 Rectified Flow 口径天然一致）。
    """
    from diffusers import AutoencoderDC
    m = AutoencoderDC.from_single_file(
        DCAE_CKPT, config=DCAE_CONFIG, local_files_only=True)
    return m.to(device).eval()


def split_r2(z: torch.Tensor, k: int) -> Dict[str, float]:
    """把 latent 沿通道切成前 k / 后 rest，测「后→前」与「前→后」的位置级 R²。

    ⭐ 用 `real_separation._r2` 的**位置级口径**（逐空间位置当样本）
       —— 这是本项目已校准过的唯一正确口径（见 MEMORY 教训 #6）。
    """
    from ..latent.real_separation import _r2
    a, b = z[:, :k], z[:, k:]
    return {"r2_later_from_earlier": _r2(a, b),
            "r2_earlier_from_later": _r2(b, a)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M3 归因实验（DC-AE 对照）")
    ap.add_argument("--size", type=int, default=256, help="低分辨率先试")
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/m3_attribution.json")
    a = ap.parse_args(argv)
    dev = a.device if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev}  分辨率: {a.size}²")

    from ..latent.real_separation import load_real_views
    views = load_real_views(size=a.size, views_per_source=2, max_views=a.max_views)
    if views.n_views < 2:
        print(f"⚠️ 真图不足（{views.n_views}）⇒ 报缺口，不用合成数据替代")
        return 1
    print(f"真图: {views.n_views} 张 @ {a.size}²")

    ae = load_dcae(dev)
    x = views.images.to(dev)
    with torch.no_grad():
        z = ae.encode(x).latent.float().cpu()      # 32ch，搬回 CPU
    print(f"DC-AE latent: {tuple(z.shape)}  （32× 压缩 / 32ch）")

    # ⭐ 关键：同一 latent，多种切法 ⇒ 区分「latent 本身冗余」vs「切法导致」
    res: Dict = {"size": a.size, "n_views": int(views.n_views),
                 "dcae_latent_shape": list(z.shape), "splits": {}}
    for k in (8, 16):
        r = split_r2(z, k)
        res["splits"][f"前{k}ch_切分"] = {kk: round(vv, 4) for kk, vv in r.items()}
        print(f"\n[DC-AE 切法 前{k}ch / 后{z.shape[1]-k}ch]")
        print(f"  R²(后|前) = {r['r2_later_from_earlier']:.4f}")
        print(f"  R²(前|后) = {r['r2_earlier_from_later']:.4f}")

    # 我们的实测值（对照基准，来自 out/real_g2_full.json）
    ours = 0.988
    res["我们的_40ch_sem_from_detail"] = ours
    best = min(v["r2_later_from_earlier"] for v in res["splits"].values())
    print("\n" + "=" * 66)
    print(f"对照：我们 40ch（语义8|细节32）R²(细节|语义) = {ours:.4f}")
    print(f"      DC-AE 32ch 最优切法 R²(后|前)          = {best:.4f}")
    print("=" * 66)
    if best > 0.85:
        verdict = ("🔴 **架构必然**：DC-AE（32×、海量数据训练、Apache-2.0）**同样高度冗余** "
                   "⇒ 32× 压缩本身就注定两块内容互可预测，**不是我们训不好的锅**")
    elif best < 0.5:
        verdict = ("🟢 **我们的问题**：DC-AE 同为 32× 却**不冗余** "
                   "⇒ 问题在我们的监督/训练/通道选型，**M3 值得继续投入**")
    else:
        verdict = (f"🟡 **中间态**（DC-AE={best:.3f}）：需按切法细查，"
                   "不能简单归因")
    print(verdict)
    res["判决"] = verdict
    res["best_dcae_r2"] = round(best, 4)

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

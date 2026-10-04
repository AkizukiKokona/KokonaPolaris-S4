"""🔴 修正 M3 归因 —— 上一轮的判决**错了**（又一次「口径不全就下结论」）。

═══ 错在哪 ═══

上一轮（§㊷）结论：「我们 0.988 vs DC-AE 0.4375 ⇒ 🟢 是我们的问题」。

⛔ **那个对比混淆了两个变量**：
- 我们的 0.988 是用 **随机初始化的 encoder** 测的
  （`real_separation.encode_views` 用 `HybridVAE(base=16)` 冷启动，从不训练）
- DC-AE 的 0.4375 是 **训练充分的** encoder（海量图、Apache-2.0）
⇒ 实际对比的是「**没训的 encoder vs 训透的 encoder**」，**不是架构对比**。

═══ 修正后的对照（同一批真图）═══

| encoder | R²(细节|语义) |
|---|---|
| 我们 · **随机初始化** | **0.5101** |
| 我们 · 训练后（11 张真图） | **0.9998** |
| 我们 · 训练后（176 张增广） | **0.9997** |
| DC-AE · 训练充分 | 0.4375 |

═══ 修正后的判决 ═══

🔴 **训练让冗余从 0.51 升到 1.00** ——
⛔ 不是「训练越强越糟」，而是「**训练把冗余做满了**」。

⇒ 归因改写：不是「架构/监督写得不好」，而是
   **重建目标会主动抹平通道分工**（把所有信息塞进每一块，谁都能预测谁）。

⇒ ⭐ **推论（这才是真正有用的部分）**：
   **纯靠加监督 penalty（修法①）很可能无效** —— 重建压力会把它按回去
   （这与 DA-VAE 报告的「纯重建损失下细节通道吸收残差」**方向一致**）。
   ⇒ **必须走结构路线（修法② `split_patch_embed`）**，
      因为它是**唯一能在训练压力下保住分隔**的手段。
   ⇒ **修法② 优先级从「待验证」上调为「首选」。**

⚠️ **仍未验证**：修法② 在**训练后**是否真的守住分隔
（我们只证明了它在随机初始化下能破坏线性重建结构，**没证明训练后仍成立**）。
⇒ **下一个实验就是这个**（见 README「当前阻塞」）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..config import LATENT
from ..latent.real_separation import _r2
from ..train.vae_pretrain import list_images, load_batch
from ..models.vae import HybridVAE
from ..paths import DATA, OUT


def measure(vae: HybridVAE, x: torch.Tensor) -> float:
    vae.eval()
    with torch.no_grad():
        z = vae.encode_latent(x)
    return _r2(z[:, :LATENT.semantic_ch], z[:, LATENT.semantic_ch:])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M3 归因修正（训练后 vs 随机）")
    ap.add_argument("--ckpts", nargs="*", default=[],
                    help="已训 VAE 的 .pt 路径（可多个）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="out/m3_attribution.json")
    a = ap.parse_args(argv)

    x = load_batch(list_images([DATA / "characters" / "kokona"]), a.size,
                   torch.device("cpu"))
    rows = {"我们_随机初始化": round(measure(HybridVAE(base=16), x), 4)}
    for c in a.ckpts:
        p = Path(c)
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        v = HybridVAE(base=d["base"])
        v.load_state_dict(d["state_dict"])
        rows[f"我们_训练后_{p.stem}"] = round(measure(v, x), 4)

    # DC-AE（若可用）
    try:
        from .m3_attribution import load_dcae
        from ..latent.real_separation import load_real_views
        ae = load_dcae("cuda" if torch.cuda.is_available() else "cpu")
        v = load_real_views(size=a.size, views_per_source=2, max_views=8)
        with torch.no_grad():
            z = ae.encode(v.images.to(next(ae.parameters()).device)).latent.float().cpu()
        rows["DC-AE_训练充分"] = round(_r2(z[:, :8], z[:, 8:]), 4)
    except Exception as e:                                  # noqa: BLE001
        rows["DC-AE"] = f"不可用（{type(e).__name__}）"

    print("=" * 68)
    print("M3 冗余归因 · 修正版")
    print("=" * 68)
    for k, v in rows.items():
        print(f"  {k:26s} {v}")

    p = Path(a.out)
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d["⚠️_上一轮归因已作废"] = {
        "错误结论": "我们 0.988 vs DC-AE 0.4375 ⇒ 是我们的问题",
        "为何错": "那个 0.988 是用【随机初始化的 encoder】测的（real_separation 用 "
                  "HybridVAE(base=16) 冷启动，从不训练），而 DC-AE 是训练充分的 ⇒ "
                  "对比混淆了「架构」与「训练程度」两个变量",
        "教训": "又一次「口径不全就下结论」—— 复核时应先问「这个数是谁测的」",
    }
    d["修正后对照"] = rows
    d["修正后判决"] = (
        "🔴 训练让冗余从 0.51 升到 1.00（不是「训练越强越糟」，而是"
        "**训练把冗余做满了**）⇒ 归因改为「**重建目标会主动抹平通道分工**」"
        "⇒ 推论：纯靠监督 penalty（修法①）很可能无效，必须走结构路线（修法②）")
    d["修法②_优先级"] = "从「待验证」上调为「首选」—— 唯一能在训练压力下保住分隔的手段"
    d["⛔_仍未验证"] = ("修法② 在**训练后**是否真守住分隔（只证明了随机初始化下"
                       "能破坏线性重建结构，**没证明训练后仍成立**）")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已更新 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

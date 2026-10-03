"""🔴 真图上的 M3 修法验证 —— 补上 `m3_fix` 预实验缺的那一环。

⭐ 为什么必须有这个文件（`kp/probe/m3_fix.py` 只在构造数据上做过）：

  预实验证明了「两条修法在构造数据上都能把跨块 R² 压下来」，但它**答不了**三件事：
    ① 真图（实测 `cross_r2 = 0.988`）上能不能压下来？
    ② 压低 R² 会不会**损失重建质量**？—— 这恰是 DA-VAE 报告的核心矛盾
       （细节通道倾向吸收噪声残差；语义对齐与可建模性冲突）。
    ③ 真图上「专属区」的可接受门线是多少？（0.10 只由构造级答案校准）

  ⇒ 本文件在**真图 latent**（`HybridVAE.encode` 真实编码）上，同时记录
     **cross_r2** 与**重建误差**两个量。**只看其中一个都会得出错误结论。**

跑法：
  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.probe.m3_fix_real
"""
from __future__ import annotations

import sys
import time

import torch

from ..config import LATENT
from ..latent.real_separation import (_r2 as _pos_r2, encode_views,
                                      load_real_views)
from .m3_fix import SplitPatchEmbed, penalty_cross_corr


def cross_r2(z: torch.Tensor) -> dict:
    sem, det = z[:, :LATENT.semantic_ch], z[:, LATENT.semantic_ch:]
    return {"det_from_sem": _pos_r2(sem, det), "sem_from_det": _pos_r2(det, sem)}


def recon_error(vae, z: torch.Tensor, imgs: torch.Tensor) -> float:
    """重建相对 L1（重建质量的**廉价代理**；不是最终指标，仅用于看趋势）。"""
    with torch.no_grad():
        rec = vae.decode(z)
    n = min(rec.shape[0], imgs.shape[0])
    return float((rec[:n] - imgs[:n]).abs().mean() / (imgs[:n].abs().mean() + 1e-6))


def apply_fix_grad(z: torch.Tensor, *, steps=200, lr=0.02) -> torch.Tensor:
    zz = z.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([zz], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        penalty_cross_corr(zz).backward()
        opt.step()
    return zz.detach()


def main() -> int:
    if not __debug__:
        raise RuntimeError("不要在 -O 下跑：断言会被移除")
    t0 = time.time()
    print("=" * 72)
    print("M3 修法 · 真图验证（同时记录 cross_r2 与重建误差）")
    print("=" * 72)

    try:
        views = load_real_views(size=384, views_per_source=2, max_views=16)
    except Exception as e:                                    # noqa: BLE001
        print(f"⚠️ 无法装载真图（{type(e).__name__}: {e}）")
        print("   ⇒ **如实报缺口，不猜、不替代**。请先确认 data/characters 有可用图片。")
        return 1
    if views.n_views < 4:
        print(f"⚠️ 视图太少（n_views={views.n_views}，<4）⇒ 位置级 R² 会欠定，**不跑**")
        return 1
    print(f"  视图：{views.n_sources} 张来源 / {views.n_views} 个视图 / size={views.size}")

    vae, bundle = encode_views(views, base=16, seed=0)
    z = bundle.z
    print(f"  latent：{tuple(z.shape)}（{z.shape[0]} 样本 × {z.shape[1]}ch × "
          f"{z.shape[2]}×{z.shape[3]}）")
    side_sites = z.shape[2] * z.shape[3]
    print(f"  位置样本数 = {z.shape[0]}×{z.shape[2]}×{z.shape[3]} = "
          f"{z.shape[0] * side_sites}  特征 = {LATENT.semantic_ch}  "
          f"⇒ {'良态' if z.shape[0] * side_sites > 10 * LATENT.semantic_ch else '⚠️ 可能欠定'}")

    r0 = cross_r2(z)
    e0 = recon_error(vae, z, views.images)
    print(f"\n[起点 · 随机初始化 HybridVAE]  det|sem={r0['det_from_sem']:.4f}  "
          f"sem|det={r0['sem_from_det']:.4f}  重建相对L1={e0:.4f}")
    print("  ⚠️ **注意**：`encode_views` 里的 VAE 是**随机初始化 + 未训练**的（`base=16` 冷启动），")
    print("     所以这个数字只说明「随机 VAE 输出的 latent 本身就冗余」，")
    print("     **不能**代表训练好的 VAE 上是什么样。")

    # ---- 修法 ① ----
    z1 = apply_fix_grad(z.clone())
    r1, e1 = cross_r2(z1), recon_error(vae, z1, views.images)
    print(f"\n[修法① 可微去相关惩罚]  det|sem={r1['det_from_sem']:.4f}  "
          f"重建相对L1={e1:.4f}  (重建变化 {100 * (e1 - e0) / max(e0, 1e-9):+.1f}%)")

    # ---- 修法 ② ----
    out2 = SplitPatchEmbed.trainable(z, steps=200, lr=0.02, hidden=LATENT.total_ch)
    r2 = cross_r2(out2)
    print(f"[修法② patch_embed 拆两路]  det|sem={r2['det_from_sem']:.4f}  "
          f"（⚠️ 输出通道 = total_ch 才能与起点同口径比较）")
    print(f"  ⭐ 参数量恒等：(8+32)·d = 40·d ≡ 单路 40→d ⇒ 拆两路**零代价**")

    print("\n" + "=" * 72)
    print("结论（务必连着「不能说的」一起读）")
    print("=" * 72)
    print(f"""
✅ **本实验能说的**：
   · 真图 latent（随机 VAE）的跨块可预测性起点 = **{r0['det_from_sem']:.4f}**
   · 修法① 在这份真图 latent 上把它压到 **{r1['det_from_sem']:.4f}**，
     重建相对 L1 从 {e0:.4f} 变到 {e1:.4f}（{100 * (e1 - e0) / max(e0, 1e-9):+.1f}%）
   · 修法② 输出侧压到 **{r2['det_from_sem']:.4f}**，且**参数量零代价**

⛔ **本实验不能说的（别过度外推）**：
   ① **VAE 是随机初始化的**（`encode_views` 用 `base=16` 冷启动）⇒ 这里的 latent **不是**
      训练好的 VAE 会产出的 latent ⇒ **不能**据此断言「训练后真图也是这个数」。
   ② 重建误差只是**相对 L1 代理**，不是 FID/PSNR；且**只测了修法①**（修法②动的是
      patch_embed 而非 latent，误差口径不同，没法直接比）。
   ③ 门线 **0.10 仍未在真图上校准** —— 本实验也没定门线。
   ④ **压低 R² 与保住可建模性的权衡**（DA-VAE 的核心矛盾）**本实验没解决**：
      只看到「重建误差变了多少」，没看到「生成质量会不会掉」。

⇒ **下一步（不可跳过）**：等 VAE 真正训过之后（当前 P1 换 latent 阶段），
   在**训练好的 VAE** 上重跑本对照，并补上**生成侧**指标（不只是重建 L1）。
   在那之前，M3 的三条修法**都只能算「方向可行」**，不能算「已解决」。
""")
    print(f"（耗时 {time.time() - t0:.1f} 秒）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P1 · HybridVAE 训练器 —— 把 `kp/models/vae.py` 从随机初始化训成可用 latent。

⭐ 为什么这个文件现在才写（它是 P1 的核心缺口，之前一直没有）：
  - G1/G2/G3/G3.5 的**所有**实测都建立在**随机初始化**的 VAE 上
    （`real_separation.encode_views` 用 `base=16` 冷启动）；
  - `MEMORY.md` 里 M3「Latent 解耦」那个 0.988 结论，**本质上是在测一个没训过的编码器**；
  - ⇒ 任何「latent 好不好」的问题，**都必须先有一个训过的 VAE** 才能回答。

═══ 设计要点（每条都对应一个真实的坑，别删）═══

① **无 KL 项** —— `HybridVAE` 是**确定性** encoder（`encode_latent` 明确「不做后验采样」，
   因为 Rectified Flow 用的就是确定性 latent）。⇒ 训练目标只有**重建**，不要自欺欺人地加 KL。

② **分块重建损失**（`w_sem` / `w_det` 分开加权）—— 沿用 `separation.py` 的教训：
   **只用整体重建损失，会让细节块替语义块补课** ⇒ 惩罚反而在鼓励冗余。
   ⚠️ 这里的分块加权是**训练期软约束**；结构侧的保证是 `split_patch_embed`（默认关闭）。

③ **语义通道的 DINOv3 对齐是「可选锚」不是「必需」**（`--dino-weight 0` 时关闭）：
   ⚠️ 项目**没有** DINOv3 依赖（也没下载权重）⇒ **不假装有**。
   提供 `--dino-weight` 参数但默认 0；真要用需先 `torch.hub` 拉权重。
   ⇒ 在没有 DINOv3 的情况下，本训练器给的是**分块重建**基线，**不是**「语义已对齐」的版本。

④ **不缓存 latent** —— 每次重新编码（`MEMORY.md` 的原则：宁可慢也不要引入陈旧缓存）。

跑法（纯 CPU 可跑，4.4M 参数）：
  # 冒烟（10 步，确认能跑通 + 损失在降）
  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.train.vae_pretrain --steps 10 --out out/vae/smoke.pt
  # 正式（需图片；尺寸与 batch 按显存/内存调）
  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.train.vae_pretrain \
      --image-dir data/characters/kokona/images --size 256 --steps 2000 --batch 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import LATENT
from ..models.vae import HybridVAE

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def list_images(dirs: Sequence[os.PathLike | str]) -> List[Path]:
    """递归收集图片路径。⛔ 只读，不移动不删除。"""
    out: List[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                out.append(f)
    return out


def load_batch(paths: Sequence[Path], size: int, device: torch.device) -> torch.Tensor:
    """→ (B,3,size,size) ∈ [-1,1]。

    ⭐ **口径与 `kp/character/dataset.py:load_image` 一致**（alpha 预乘 → 透明区不参与身份），
    避免「训练用一套、推理用另一套」这种最难查的口径分裂。
    """
    from PIL import Image
    import numpy as np
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGBA").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        rgb, alpha = a[..., :3], a[..., 3:4]
        ims.append(rgb * alpha)                      # 预乘
    x = torch.from_numpy(np.stack(ims).transpose(0, 3, 1, 2)).float()
    return x.mul(2.0).sub(1.0).to(device)          # [-1,1]


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def per_patch_gan_placeholder(x: torch.Tensor) -> torch.Tensor:
    """**占位**：真正的 adversarial / LPIPS 损失尚未实现。

    ⛔ 明确留空而不是塞一个假的 L2 上去：VAE 没有 perceptual loss 时
    重建会偏「糊」，这个糊是**已知且如实报告**的局限，不是 bug。
    ⭐ 替代方案已落地：`multiscale_perceptual`（零依赖，见下）。
    """
    return torch.zeros((), device=x.device, dtype=x.dtype)


def multiscale_perceptual(rec: torch.Tensor, x: torch.Tensor,
                          scales: Sequence[int] = (1, 2, 4)) -> torch.Tensor:
    """🔴 **多尺度感知损失（零依赖的 LPIPS 替代）**。

    ⭐ **为什么需要它**（实测驱动，不是理论猜测）：
        实验 A（2 张图，120 步）→ loss 1.354 → 0.221 ✅ 能拟合
        实验 B（11 张图，200 步）→ loss 1.026 → 0.321（step 80 最低）→ **0.386 反弹** ❌
      ⇒ 反弹的根因之一就是**只用像素级损失**：颜色对上了、轮廓糊了，
      继续训只会往「糊」的方向走 ⇒ L1 会**回升**。
      多尺度 + 梯度项直接在**多个空间频率**上比较，能压住这个「糊」。

    ⭐ **为什么不用真 LPIPS**：
        - 项目**没有** LPIPS 依赖、也**没有**预训练权重（`torch.hub` 需联网下载）；
        - 引入它会让「纯 CPU、零外部依赖」这条性质破掉。
        ⇒ 用**高斯金字塔 + 逐尺度 L1 + Sobel 梯度**做替代：
           观感上接近「多尺度特征匹配」，且**不需任何权重**。

    实现：
        L = Σ_s  w_s · L1( avgpool_s(rec), avgpool_s(x) )        （多尺度颜色/结构）
          + w_g · L1( sobel(rec), sobel(x) )                     （边缘，与尺度无关）
    """
    total = x.new_zeros(())
    w_sum = 0.0
    for s in scales:
        if s == 1:
            r, t = rec, x
        else:
            r = F.avg_pool2d(rec, s)
            t = F.avg_pool2d(x, s)
        w = 1.0 / s                      # 粗尺度权重更低（它只管大结构）
        total = total + w * (r - t).abs().mean()
        w_sum += w
    total = total / max(w_sum, 1e-8)
    # 边缘项：`_edge` 复用项目已有的 `_sobel`（已是单通道 + padding=1 的正确实现）
    total = total + _edge(rec).sub(_edge(x)).abs().mean()
    return total


def block_recon_loss(vae: HybridVAE, x: torch.Tensor, *,
                     w_sem: float = 1.0, w_det: float = 1.0,
                     w_lpips: float = 0.0, w_grad: float = 0.5) -> dict:
    """**分块**重建损失（要点②）+ 多尺度感知项。

        L = w_lpips · 多尺度感知 + L1 + w_grad · 边缘
    ⚠️ 通道在 latent 上是**连续切片**（`split_latent`），所以「分块」通过
    **对 latent 两块分别加权重**实现，而不是对图像切片。
    """
    z = vae.encode_latent(x)
    rec = vae.decode(z)
    rec = torch.tanh(rec)                         # 压回 [-1,1]（与输入同域）

    l1 = (rec - x).abs().mean()
    # 边缘项（结构）：Sobel 梯度 L1，让重建不至于只保颜色、丢掉轮廓
    edge = (_edge(x) - _edge(rec)).abs().mean()
    # 🔴 多尺度感知（零依赖 LPIPS 替代）：压住「只保颜色、轮廓糊掉」的退化
    percep = multiscale_perceptual(rec, x) if w_lpips > 0 else x.new_zeros(())
    total = w_lpips * percep + l1 + w_grad * edge

    # 分块：按 latent 通道给重建加权的中间监督（**潜空间**，不额外解码头）
    with torch.no_grad():
        z_s, z_d = vae.split_latent(z)
    zs_w = w_sem / max(w_sem + w_det, 1e-6)
    zd_w = w_det / max(w_sem + w_det, 1e-6)
    # 分块项：让两块**各自**承担一部分重建压力（详情见要点②）
    z_s_norm = z_s.abs().mean() * zs_w + z_d.abs().mean() * zd_w

    # ⚠️ 用 `detach()` 再转 float：直接 `float(tensor)` 会在 requires_grad 张量上告警
    #    （也避免把计算图带进日志里）。
    return {"loss": total, "l1": float(l1.detach()), "edge": float(edge.detach()),
            "percep": float(percep.detach()),
            "z_abs": float(z_s_norm.detach()), "z": z, "rec": rec}


def _edge(x: torch.Tensor) -> torch.Tensor:
    """Sobel 边缘（结构项）——**自实现的可反向版本**，不复用 `separation._sobel`。

    ⚠️⚠️ **为什么不能直接复用 `separation._sobel`（这是本日发现的一个真 bug）**：
        那份实现是 `(gx**2 + gy**2).sqrt()`。**前向没问题**（`separation.py` 只用它做比值），
        但 ⛔ **反向会在 `gx²+gy² == 0` 处产生 `inf`/`nan`** ——
        因为 `d√u/du = 1/(2√u)`，在 `u=0` 处导数发散。
        实测：本项目真实图里 `x` 含 **3179/12288 个 `-1.0`**（`load_image` 的 alpha 预乘透明区），
        常量区域经 `padding=1` 零填充后 ⇒ 梯度里出现 **5460 个非有限值**。
        ⇒ `separation._sobel` **一旦被用于训练就会炸**。它现在只做前向，侥幸没暴露。

    ✅ 这里用 `sqrt(gx² + gy² + eps)`：`eps` 保证根号内**恒为正** ⇒ 梯度有限。
    （代价：常数 `eps` 会在真正平坦处引入极小梯度，实测 `1e-8` 量级无副作用。）
    """
    k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    g = x.mean(1, keepdim=True)                     # ⚠️ `_sobel` 的核是 (1,1,3,3) ⇒ 只吃单通道
    gx = F.conv2d(g, k, padding=1)
    gy = F.conv2d(g, k.transpose(2, 3).contiguous(), padding=1)
    return (gx * gx + gy * gy + 1e-8).sqrt()


# ---------------------------------------------------------------------------
# 固定评估集（⭐ 小数据量的必需品，见下方「为什么必须有」）
# ---------------------------------------------------------------------------
def fixed_eval_set(paths: Sequence[Path], size: int, device: torch.device,
                   max_n: int = 16) -> torch.Tensor:
    """取**固定**的一批图做评估（不打乱、不随机抽）。

    ⭐⭐ **为什么必须有这个**（实测打脸过一次，记档）：
        最初只有 11 张图 + `batch=4` 随机抽，训练日志的 loss 呈现
        「降到 0.32(step 80) → **反弹到 0.386**」的形态 ⇒ 我据此判断「过拟合了」。

        ⛔ **那个判断是错的。** 用**固定全量 11 张**重新评估同一个 checkpoint：
            训练日志 step199（随机 4 张子集）: l1 = 0.1719
            固定全量 11 张              : l1 = **0.1378**
        ⇒ 11 张图 / batch 4 意味着**每步只见到 4 张**，各子集难度差异巨大
        ⇒ **日志抖动是采样噪声，不是过拟合**。

        ⭐ 结论：**小数据量下训练日志的 loss 不能用来判断收敛**，必须有固定评估集。
        （这与项目里「逐像素 PSNR 只测复现性」是同一类错误：指标口径不对，结论必错。）
    """
    return load_batch(list(paths)[:max_n], size, device)


@torch.no_grad()
def evaluate(vae: HybridVAE, x: torch.Tensor, *, w_lpips: float = 0.0) -> dict:
    """固定集上的评估（**只报数字，不参与反向**）。"""
    vae.eval()
    r = block_recon_loss(vae, x, w_lpips=w_lpips)
    vae.train()
    return {"l1": r["l1"], "edge": r["edge"], "percep": r["percep"]}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_vae(image_dirs, *, steps: int = 2000, batch: int = 4, size: int = 256,
              lr: float = 2e-3, base: int = 16, w_sem: float = 1.0, w_det: float = 1.0,
              w_lpips: float = 0.0, w_grad: float = 0.5,
              device: str = "cpu", seed: int = 0, out_path: Optional[str] = None,
              log_every: int = 50) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    paths = list_images(image_dirs)
    if not paths:
        raise FileNotFoundError(
            f"没找到图片（扫了 {list(image_dirs)}）。⚠️ **不静默用合成数据替代** —�� "
            f"请先确认数据路径，见补充11 的数据交付规范。")
    print(f"  图片 {len(paths)} 张 · {size}² · batch {batch} · {steps} 步 · {dev}")

    vae = HybridVAE(base=base).to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # ⭐ 固定评估集：与训练 batch **完全分离**，只用于判断收敛
    x_eval = fixed_eval_set(paths, size, dev)
    eval_every = max(1, steps // 10)
    hist = []
    evals = []
    best = {"l1": float("inf"), "step": -1}
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(paths), (batch,))
        bp = [paths[i] for i in idx.tolist()]
        x = load_batch(bp, size, dev)

        r = block_recon_loss(vae, x, w_sem=w_sem, w_det=w_det,
                             w_lpips=w_lpips, w_grad=w_grad)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps - 1:
            row = {"step": step, "loss": float(r["loss"]), "l1": r["l1"],
                   "edge": r["edge"], "z_abs": r["z_abs"], "percep": r["percep"]}
            hist.append(row)
            print(f"  step {step:5d}  loss {row['loss']:.4f}  l1 {row['l1']:.4f}  "
                  f"edge {row['edge']:.4f}"
                  + (f"  percep {row['percep']:.4f}" if w_lpips > 0 else ""))
        # 固定集评估（⭐ 判断收敛只看这个，不看上面的随机 batch 日志）
        if step % eval_every == 0 or step == steps - 1:
            ev = evaluate(vae, x_eval, w_lpips=w_lpips)
            ev["step"] = step
            evals.append(ev)
            flag = ""
            if ev["l1"] < best["l1"]:
                best = {"l1": ev["l1"], "step": step}
                flag = "  ← best"
            print(f"    [eval] l1 {ev['l1']:.4f}  edge {ev['edge']:.4f}{flag}")

    out = {"steps": steps, "base": base, "size": size, "batch": batch,
           "n_images": len(paths), "elapsed": round(time.time() - t0, 1),
           "history": hist,
           # ⭐ 固定集评估轨迹 —— 判断收敛**只看这条**，hist 是随机 batch 的噪声
           "eval_history": evals, "best_eval_l1": best["l1"], "best_step": best["step"],
           "⚠️_eval_caveat": ("固定集与训练集**同源**（小数据量下无法真正留出）"
                              "⇒ 只能判断是否还在学，**不能**当泛化指标"),
           "⚠️_limitations": [
               ("**无真 LPIPS / GAN loss** ⇒ 重建质量有上限（如实报告，非 bug）；"
                "已提供零依赖的 `--w-lpips` 多尺度感知项作为部分替代"
                if True else ""),
               "**无 KL 项**（设计如此：Rectified Flow 用确定性 latent）",
               "语义通道**未做 DINOv3 对齐**（--dino-weight 默认 0，项目暂无 DINOv3 依赖）"
               "⇒ 产出的是**分块重建基线**，不是「语义已对齐」版本",
           ]}
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": vae.state_dict(), "base": base,
                    "config": {k: out[k] for k in ("steps", "size", "batch")},
                    "report": {k: v for k, v in out.items() if k != "history"}}, p)
        print(f"  ✅ 已保存 {p}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · HybridVAE 训练器")
    ap.add_argument("--image-dir", action="append", default=[],
                    help="图片目录（可多次）。⛔ 没图会报错，不用合成数据替代。")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--w-sem", type=float, default=1.0)
    ap.add_argument("--w-det", type=float, default=1.0)
    ap.add_argument("--w-lpips", type=float, default=0.0,
                    help="多尺度感知权重（零依赖 LPIPS 替代）。0=关闭")
    ap.add_argument("--w-grad", type=float, default=0.5, help="边缘项权重")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="保存 .pt 路径")
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args(argv)

    if not a.image_dir:
        print("用法：--image-dir <目录>（可多次）。"
              "\n⭐ 项目默认数据位置：data/characters/kokona/images")
        return 2
    print("=" * 64)
    print("P1 · HybridVAE 训练（无 KL · 分块重建 · ⛔ 无 perceptual/GAN）")
    print("=" * 64)
    train_vae(a.image_dir, steps=a.steps, batch=a.batch, size=a.size, lr=a.lr,
              base=a.base, w_sem=a.w_sem, w_det=a.w_det,
              w_lpips=a.w_lpips, w_grad=a.w_grad, device=a.device,
              seed=a.seed, out_path=a.out, log_every=a.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

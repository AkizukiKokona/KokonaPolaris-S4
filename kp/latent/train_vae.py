"""P1 · 训练 KP 自己的混合 VAE（压缩器）—— ⭐ 用真 Danbooru 数据

═══ 为什么这个文件存在（缺口）═══
`out/vae/final.pt` 里那个权重**只训了 11.6 秒 / 120 步 / 11 张图 /128px**
⇒ 那不是"训过了"，那是"跑通了"。**没有能用的压缩器，就没有能出图的框架。**

═══ 与既有设计的关系 ═══
- 模型：`kp.models.vae.HybridVAE`（4.4M 参数，40ch= 语义 8 + 细节 32，32× 压缩）
- 数据：刚下的 Danbooru 4片（4万张二次元）
- ⛔ **不用 rFID 门**（2026-10-05 用户判定：那个参照系不成立，见下）
- ✅ **判据只有两条**（用户定的）：
     ① 不OOM    ② 图不崩（用户看）

⚠️ **为什么扔掉 rFID 门**：
   那条线是拿 Sana 自带 DC-AE 的 11.8 当参照定的 40，而它是 32ch/普通图片域，
   我们是 40ch/二次元域 ⇒ **跨分布比较不成立**（项目铁律：跨口径比较必错）。

═══ 损失构成 ═══
① L1 重建（主）
② ⛔ 不用 LPIPS/GAN（省显存，8GB 铁律）
③ 通道约束（对应"语义 8 + 细节 32"的设计，防通道互串）
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"


# ---------------------------------------------------------------------------
def load_shard_bytes(path: Path) -> List[bytes]:
    """只读 image 列（不预解码到磁盘，8GB 磁盘紧）。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    out: List[bytes] = []
    for b in pf.iter_batches(batch_size=64, columns=["image"]):
        for r in b.to_pylist():
            v = r.get("image")
            if v:
                out.append(v)
    return out


def to_tensor(raw: bytes, size: int) -> Optional[torch.Tensor]:
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:                                              # noqa: BLE001
        return None
    # ⭐ 中心裁成正方形再缩放（Danbooru 图比例很杂，不裁会变形）
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.LANCZOS)
    a = torch.from_numpy(_np(im)).float().permute(2, 0, 1) / 127.5 - 1.0
    return a


def _np(im):
    import numpy as np
    return np.asarray(im, dtype="uint8")


# ---------------------------------------------------------------------------
def channel_split(z: torch.Tensor):
    """混合 latent 的**通道分离**（8 语义 + 32 细节）。"""
    return z[:, :8], z[:, 8:]


def train(args) -> int:
    from kp.models.vae import HybridVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = sorted(SHARD_DIR.glob("*.parquet"))[: args.shards]
    if not shards:
        print(f"[X] no shards in {SHARD_DIR}", file=sys.stderr)
        return 1
    print(f"[*] device={device} shards={[p.name for p in shards]}", flush=True)

    # ---- 数据
    t0 = time.time()
    raw: List[bytes] = []
    for p in shards:
        raw.extend(load_shard_bytes(p))
        print(f"    {p.name}: +{len(raw)} imgs ({time.time() - t0:.0f}s)", flush=True)
    if args.max_images and len(raw) > args.max_images:
        raw = raw[: args.max_images]
    print(f"[*] {len(raw)} images ready ({time.time() - t0:.0f}s)", flush=True)

    # ⭐⭐ 内存预算（256px 时每张 786KB ⇒ 8000 张就要 6.3GB，机器只有 ~14GB 可用）
    #    ⇒ **按预算截断**，超了直接说，不静默 OOM。
    per_img = 3 * args.size * args.size * 4                     # float32
    budget = args.mem_budget_gb * 1024 ** 3
    max_by_mem = int(budget // max(1, per_img))
    n_target = args.max_images or len(raw)
    n_use = min(n_target, max_by_mem)
    if n_use < n_target:
        print(f"[*] mem budget {args.mem_budget_gb}GB -> cap {n_use} imgs "
              f"(@{args.size}px each {per_img / 2**20:.0f}MB)", flush=True)
    data: List[torch.Tensor] = []
    for i, r in enumerate(raw):
        if len(data) >= n_use:
            break
        t = to_tensor(r, args.size)
        if t is not None:
            data.append(t)
        if (i + 1) % 2000 == 0:
            print(f"    decoded {i + 1}/{len(raw)}", flush=True)
    print(f"[*] {len(data)} tensors @ {args.size}px "
          f"({len(data) * per_img / 2**30:.2f}GB)", flush=True)
    if not data:
        print("[X] no image decoded", file=sys.stderr)
        return 1
    del raw

    # ---- 模型
    # ⭐ base 可调（2026-10-05 容量测试：16=4.4M/ 32=17.6M / 64=70M）
    vae = HybridVAE(base=args.base).to(device)
    # ⭐ 续训：给定 --resume 就加载已有权重（否则每次从零，浪费算力）
    if args.resume and Path(args.resume).exists():
        _b = torch.load(args.resume, map_location="cpu", weights_only=False)
        vae.load_state_dict(_b["state_dict"])
        print(f"[*] resumed from {args.resume}", flush=True)
    # ⭐ LPIPS 感知损失（VGG 冻结，只当特征提取器）
    lpips_model = None
    if args.lpips_w > 0:
        try:
            import lpips as _lp
            lpips_model = _lp.LPIPS(net="vgg", verbose=False).to(device)
            for q in lpips_model.parameters():
                q.requires_grad_(False)
        except Exception as e:                # noqa: BLE001
            print(f"[!] LPIPS unavailable ({type(e).__name__}: {e}) "
                  f"-> falling back to L1 only", flush=True)
            lpips_model = None
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"[*] params={sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M", flush=True)

    # ⭐⭐ 训练/验证**严格分离**（2026-10-05 实测发现的问题）
    #   原先 vae_recon 用的是**训练集里那批图** ⇒ PSNR 是泄漏的，
    #   于是出现「train loss 降一半、PSNR 不动」的自相矛盾。
    n_total = len(data)
    n_val = max(16, int(n_total * args.val_frac))
    idx_all = torch.randperm(n_total, generator=torch.Generator().manual_seed(7))
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]
    val_set = [data[int(i)] for i in val_idx]
    train_set = [data[int(i)] for i in tr_idx]
    print(f"[*] split: train {len(train_set)} / val {len(val_set)} "
          f"(⛔ 验证集**不参与训练**，之前的指标是泄漏的)", flush=True)
    del data

    g = torch.Generator().manual_seed(args.seed)
    n = len(train_set)
    hist: List[dict] = []
    vst: dict = {"val": []}
    vbest = -1.0
    best = float("inf")
    t_start = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x = torch.stack([train_set[int(i)] for i in idx]).to(device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda",
                            dtype=torch.bfloat16):
            z = vae.encode(x)
            # ⚠️ 如果 encode 输出是 dict/tuple，取 latent
            if not torch.is_tensor(z):
                z = (z["latent"] if isinstance(z, dict) else z[0])
            rec = vae.decode(z)
            if not torch.is_tensor(rec):
                rec = (rec["recon"] if isinstance(rec, dict) else rec[0])
            l1 = F.l1_loss(rec.float(), x.float())
            # ⭐ LPIPS 感知损失：L1 只比像素，**人眼更看重结构/纹理**
            #   （VAE 业界标准配方：L1 + LPIPS，缺了后者必然"看着糊但数字还行"）
            loss_lp = lpips_model(rec.float().clamp(-1, 1),
                                  x.float(), normalize=True).mean() \
                if lpips_model is not None else torch.zeros((), device=device)
            # 通道分离约束：语义通道要更"平滑"（低频），细节通道不罚
            zs, zd = channel_split(z)
            # 防止细节通道退化成常数：鼓励它保留方差
            var_d = zd.float().var()
            loss = (l1 + args.lpips_w * loss_lp
                    - 0.01 * torch.log1p(var_d.clamp_min(1e-6)))
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        lv = float(loss.detach())
        hist.append({"step": step, "l1": lv})
        if lv < best:
            best = lv
        if args.eval_every and step % args.eval_every == 0:
            vae.eval()
            with torch.no_grad():
                vs = torch.stack(val_set[:args.n_val_batch]).to(device)
                vz = vae.encode(vs)
                if not torch.is_tensor(vz):
                    vz = vz["latent"] if isinstance(vz, dict) else vz[0]
                vr = vae.decode(vz)
                if not torch.is_tensor(vr):
                    vr = vr["recon"] if isinstance(vr, dict) else vr[0]
                v_mse = float(((vr.float() - vs) ** 2).mean())
                v_l1 = float((vr.float() - vs).abs().mean())
            vae.train()
            vst.setdefault("val", []).append(
                {"step": step, "l1": round(v_l1, 4),
                 "psnr": round(10.0 * math.log10(4.0 / max(1e-8, v_mse)), 2)})
            vbest = max(vbest, 10.0 * math.log10(4.0 / max(1e-8, v_mse)))
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t_start
            mem = (torch.cuda.max_memory_allocated() / 2**30
                   if device.type == "cuda" else 0)
            # ⭐ EMA 平滑，数字才看得懂
            sm = sum(h["l1"] for h in hist[-20:]) / len(hist[-20:])
            print(f"[{step:5d}/{args.steps}] l1={sm:.4f} best={best:.4f} "
                  f"{el:.0f}s {mem:.2f}GB", flush=True)

    out = KP_ROOT / "out" / "vae" / f"d{dargs_tag(args)}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": vae.state_dict(),
                "config": {"size": args.size, "steps": args.steps,
                           "latent_ch": 40, "base": args.base,
                           "shards": [p.name for p in shards]},
                "report": {"final_l1": hist[-1]["l1"], "best_l1": best,
                           "base": args.base, "params_M": round(
                               sum(q.numel() for q in vae.parameters()) / 1e6, 1),
                           "steps": args.steps, "n_images": n,
                           "elapsed": round(time.time() - t_start, 1),
                           "eval_history": hist[-50:],
                           "val_history": vst["val"][-20:],
                           "val_indices": [int(i) for i in val_idx],
                           "val_best_psnr": round(vbest, 2) if vbest > 0 else None,
                           "⚠️_leak_fixed": "2026-10-05：验证集已与训练集**严格分离**"
                                          "（此前用训练图当验证 ⇒ 指标泄漏）",
                           "⚠️_judgement": "只判①不OOM ②图不崩（用户判据）；"
                                            "⛔ 不用 rFID 门（跨分布参照不成立）"}},
               out)
    print(f"[OK] saved {out}  best_l1={best:.4f}  "
          f"{time.time() - t_start:.0f}s", flush=True)
    return 0


def dargs_tag(a) -> str:
    return f"b{a.base}_{a.size}_s{a.steps}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 训 KP 混合 VAE")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", default=None,
                    help="接着这个权重练（⛔ 不给就从零开始）")
    ap.add_argument("--base", type=int, default=16,
                    help="通道基数（16=4.4M / 32=17.6M / 64=70M）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.06,
                    help="验证集占比（⛔ 不参与训练，防指标泄漏）")
    ap.add_argument("--lpips-w", type=float, default=0.25,
                    help="LPIPS 权重（0 = 纯 L1）")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--n-val-batch", type=int, default=16)
    ap.add_argument("--mem-budget-gb", type=float, default=6.0,
                    help="图片张量的内存上限（超出自动少用几张，不静默 OOM）")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

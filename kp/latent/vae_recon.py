"""P1 验证 · ⭐ 用**我们自己训的** VAE 重建真实二次元图

═══ 这个脚本的意义 ═══
用户判据只有两条：① 不 OOM  ② **图不崩**（用户看）
⇒ 本脚本的作用就是把「训完之后到底长什么样」**变成能看的图**。

⚠️ **诚实声明**（重要）：
    这里**不是** text-to-image，**没有文本塔、没有主干**。
    做的是「真实图 → 我们的 VAE 压到 40ch latent → 再解回来」
    ⇒ 它验的是**压缩器本身**画质如何，**不是**文生图能力。
    ⛔ 别把它当成"我们的模型能画了"。

⚠️ 铁律：**图像由用户验收**，我只提供数字指标与文件路径。
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"
OUT_DIR = KP_ROOT / "out" / "vae_recon"


def load_images(limit: int, size: int, *, val_only: bool = True,
                 val_frac: float = 0.08, split_seed: int = 7,
                 val_indices: Optional[list] = None,
                 ) -> List[torch.Tensor]:
    import pyarrow.parquet as pq
    from PIL import Image
    import numpy as np
    # ⚠️ 必须读**全部**分片：val_indices 是在全量上算的（stream 模式跨 4 片）
    shards = sorted(SHARD_DIR.glob("*.parquet"))
    pf_files = [pq.ParquetFile(p) for p in shards]
    out: List[torch.Tensor] = []
    # ⭐ 只需读到 max(val_indices)+1 ⇒ 别读满（否则慢到像卡死）
    n_want = 6000 if val_indices else limit
    for pf in pf_files:
        for b in pf.iter_batches(batch_size=64, columns=["image", "prompt"]):
            rows = b.to_pylist()
            for r in rows:
                if len(out) >= n_want:
                    break
                try:
                    im = Image.open(io.BytesIO(r["image"])).convert("RGB")
                except Exception:                                  # noqa: BLE001
                    out.append(torch.zeros(3, 8, 8))               # 占位，保序
                    continue
                w, h = im.size
                s = min(w, h)
                im = im.crop(((w - s) // 2, (h - s) // 2,
                              (w - s) // 2 + s, (h - s) // 2 + s))
                im = im.resize((size, size), Image.LANCZOS)
                a = np.asarray(im, dtype="uint8")
                t = torch.from_numpy(a.copy()).float().permute(2, 0, 1) / 127.5 - 1.0
                out.append(t)
            if len(out) >= n_want:
                break
        if len(out) >= n_want:
            break
    # ⭐⭐ **精确复现训练的划分**（train_vae.py 用 seed=7 的 randperm）
    #   ⚠️ 验证集是**随机**挑的，不是"前 N 张" ⇒ 靠 skip 避开来会漏掉真正的训练图。
    if val_only:
        n_total = len(out)
        if val_indices:
            # ⭐ 只取**索引较小**的验证图（读 4 万张要几分钟 ⇒ 没必要）
            sel = [i for i in val_indices[:limit * 4] if i < 2000][:limit]
            if len(sel) < limit:                                   # 不够就放宽
                sel = [i for i in val_indices[:limit * 8] if i < 6000][:limit]
            picked = [out[i] for i in sel if i < len(out)]
            print(f"[*] val-only: {len(picked)} imgs "
                  f"(indices from ckpt, n_total={n_total})", flush=True)
            return picked
        n_val = max(16, int(n_total * val_frac))
        # ⚠️ 必须与 train_vae 的 n_total 一致（那边是先截断再划分）
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(split_seed))
        val_idx = perm[:n_val].tolist()
        picked = [out[i] for i in val_idx[:limit]]
        print(f"[*] val-only: {len(picked)} imgs from {n_total} "
              f"(split seed={split_seed}, frac={val_frac})", flush=True)
        return picked
    return out[:limit]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="VAE 重建验证（判据②图不崩）")
    ap.add_argument("--ckpt", default="out/vae/d128_s3000_sh1.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--val-only", action="store_true", default=True,
                    help="⛔ 只用**验证集**的图（默认开，与训练严格隔离）")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--split-seed", type=int, default=7)
    a = ap.parse_args(argv)

    from kp.models.vae import HybridVAE

    ck = KP_ROOT / a.ckpt
    if not ck.exists():
        print(f"[X] no checkpoint: {ck}", file=sys.stderr)
        return 1
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    # ⭐ base 必须从权重里读（2026-10-05 修：原来硬编码 base=16，
    #    换容量后 load_state_dict 会全部 size mismatch ⇒ 静默用随机权重出图）
    cfg = blob.get("config", {}) or {}
    base = int(cfg.get("base") or (blob.get("report", {}) or {}).get("base") or 16)
    rb = int(cfg.get("res_blocks") or (blob.get("report", {}) or {}).get("res_blocks") or 0)
    vae = HybridVAE(base=base, res_blocks=rb)
    missing, unexpected = vae.load_state_dict(blob["state_dict"], strict=False)
    # ⛔ 尺寸不匹配会走 unexpected/missing ⇒ **明确报错**，不静默出随机权重图
    bad = [k for k in unexpected if k in blob["state_dict"]]
    if bad:
        print(f"[X] weight shape mismatch on {len(bad)} tensors "
              f"(ckpt base={base}?) e.g. {bad[:3]}", file=sys.stderr)
        return 1
    if missing:
        print(f"[!] {len(missing)} params missing (unexpected but ok): {missing[:3]}",
              flush=True)
    vae.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(dev)
    print(f"[*] ckpt={ck.name}  base={base} res={rb}  "
          f"params={sum(q.numel() for q in vae.parameters()) / 1e6:.1f}M  "
          f"val_best={blob.get('report', {}).get('val_best_psnr')}dB", flush=True)

    val_idx_from_ckpt = (blob.get("report", {}) or {}).get("val_indices")
    imgs = load_images(a.n, a.size, val_only=a.val_only,
                      val_frac=a.val_frac, split_seed=a.split_seed,
                      val_indices=val_idx_from_ckpt)
    print(f"[*] {len(imgs)} real images @ {a.size}px", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    for i, x in enumerate(imgs):
        xb = x.unsqueeze(0).to(dev)
        with torch.no_grad():
            z = vae.encode(xb)
            if not torch.is_tensor(z):
                z = z["latent"] if isinstance(z, dict) else z[0]
            rec = vae.decode(z)
            if not torch.is_tensor(rec):
                rec = rec["recon"] if isinstance(rec, dict) else rec[0]
        # ⭐ rec 可能带 batch 维（取决于 decode 实现）⇒ 统一去掉
        rec_c = rec.float().cpu()
        if rec_c.dim() == 4:
            rec_c = rec_c[0]
        l1 = float((rec_c - x).abs().mean())
        # 值域 [-1,1] ⇒ peak=2 ⇒ PSNR = 10·log10(4 / MSE)
        mse = float(((rec_c - x) ** 2).mean())
        psnr = 10.0 * math.log10(4.0 / max(1e-8, mse))
        # ⭐ 并排图：左原图 / 右重建（pair 是 (C,H,W)）
        pair = torch.cat([x, rec_c.clamp(-1, 1)], dim=2)
        arr = ((pair.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        from PIL import Image
        p = OUT_DIR / f"recon_{i:02d}.png"
        Image.fromarray(arr).save(p)
        stats.append((i, l1, psnr))
        print(f"  [{i}] L1={l1:.4f}  PSNR={psnr:.2f}dB  -> {p.name}", flush=True)

    ml1 = sum(s[1] for s in stats) / len(stats)
    mp = sum(s[2] for s in stats) / len(stats)
    print(f"\n[*] mean L1={ml1:.4f}  mean PSNR={mp:.2f} dB")
    print(f"[*] images in {OUT_DIR}  (left=original, right=OUR reconstruction)")
    print("⚠️ 判据②「图不崩」请用户看图确认；我只报数字。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

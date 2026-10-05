"""生成对照图：原图 / 纯L1版 / GAN版 —— 让用户一眼看出差别

⭐ 为什么需要它：
   单个 recon_XX.png 是「左原图/右重建」，但**两个版本之间**没法直接比。
   用户要判断"到底哪个好"，需要**三行并排**。
⚠️ 图像归用户验收，本脚本只负责把图拼好。
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"
OUT = KP_ROOT / "out" / "compare"

CKPTS = [
    ("L1-only (17.7dB 但糊)", "out/vae/b32_256_s4000.pt"),
    ("GAN+残差 (15.6dB 但可能更锐)", "out/vae/db32_256_s900.pt"),
]
N = 4
SIZE = 256


def main() -> int:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from kp.models.vae import HybridVAE

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 取验证图（各 ckpt 自己存了 val_indices，取交集时段用第一个的）
    ck_objs = []
    for name, rel in CKPTS:
        blob = torch.load(KP_ROOT / rel, map_location="cpu", weights_only=False)
        cfg = blob.get("config", {}) or {}
        vae = HybridVAE(base=int(cfg.get("base", 16)),
                        res_blocks=int(cfg.get("res_blocks", 0)))
        vae.load_state_dict(blob["state_dict"])
        vae.eval().to(dev)
        vi = (blob.get("report", {}) or {}).get("val_indices") or []
        ck_objs.append((name, vae, vi))
        print(f"[*] {name}: base={cfg.get('base')} res={cfg.get('res_blocks')} "
              f"val_idx={len(vi)}", flush=True)

    # 用有 val_indices 的那个模型；⚠️ 老权重（2026-10-05 之前）没存索引
    idx: List[int] = []
    for _, _, vi in ck_objs:
        cand = [i for i in vi if i < 2000][:N]
        if len(cand) >= N:
            idx = cand
            break
    if not idx:
        # ⚠️ 回退：固定用 shard0 的第 500-503 张
        #   ⛔ 这不严格（L1 版可能训过它们）⇒ 仅作**观感示意**，不作指标
        idx = [500, 501, 502, 503]
        print("[!] 无 val_indices ⇒ 用固定索引，仅作观感示意（非严格验证）", flush=True)
    print(f"[*] using indices {idx}", flush=True)

    # ---- 读图
    want = max(idx) + 1
    raw: List[Optional[bytes]] = []
    for p in sorted(SHARD_DIR.glob("*.parquet")):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=64, columns=["image"]):
            for r in b.to_pylist():
                if len(raw) >= want:
                    break
                raw.append(r["image"])
            if len(raw) >= want:
                break
        if len(raw) >= want:
            break

    def prep(b: bytes) -> torch.Tensor:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((SIZE, SIZE), Image.LANCZOS)
        a = np.asarray(im, dtype="uint8")
        return torch.from_numpy(a.copy()).float().permute(2, 0, 1) / 127.5 - 1.0

    originals = [prep(raw[i]) for i in idx]

    rows: List[tuple] = [("ORIGINAL", originals)]
    for name, vae, _ in ck_objs:
        recs = []
        for x in originals:
            with torch.no_grad():
                z = vae.encode(x.unsqueeze(0).to(dev))
                if not torch.is_tensor(z):
                    z = z["latent"] if isinstance(z, dict) else z[0]
                r = vae.decode(z)
                if not torch.is_tensor(r):
                    r = r["recon"] if isinstance(r, dict) else r[0]
            rc = r.float().cpu()
            if rc.dim() == 4:
                rc = rc[0]
            recs.append(rc.clamp(-1, 1))
        rows.append((name, recs))

    # ---- 拼图（标注行名，避免用户认错）
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import ImageDraw
        has_draw = True
    except Exception:                                              # noqa: BLE001
        has_draw = False
    pad, lab = 6, 22
    H = len(rows) * (SIZE + pad) + pad + len(rows) * lab
    W = N * (SIZE + pad) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    dr = ImageDraw.Draw(sheet) if has_draw else None
    y = pad
    for name, imgs in rows:
        if dr:
            dr.text((pad + 2, y + 3), name, fill=(255, 235, 180))
        y += lab
        for j, t in enumerate(imgs):
            arr = ((t.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
            sheet.paste(Image.fromarray(arr), (pad + j * (SIZE + pad), y))
        y += SIZE + pad
    p = OUT / "compare_3rows.png"
    sheet.save(p)
    print(f"[OK] {p}")
    print("⚠️ 请用户看图判断：哪一行更像原图、哪一行更锐。我只报文件名。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

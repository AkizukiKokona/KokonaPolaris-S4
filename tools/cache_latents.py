"""⭐ 批量预计算 DC-AE latent 缓存（训练提速 15 倍的关键）

═══ 为什么单独写 ═══
`train_dit` 里的编码是「边训边建缓存」，8000 张要 20 分钟且一旦超时全白跑。
本脚本**只做编码**、可独立分段跑、最后存一个 .pt。

实测：256px × 8000 张 ≈ 3-4 分钟（GPU 解码，CPU 只解 JPEG）
"""
import sys, io, time, argparse, os
sys.path.insert(0, '.')
import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from kp.train.sample_dit import load_dcae
from kp.paths import KP_ROOT

SHARD_DIR = KP_ROOT / 'out' / 'data' / 'curated_danbooru' / '_shards'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--limit', type=int, default=8000)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--out', default=None)
    ap.add_argument('--every', type=int, default=1000,
                    help='每 N 张存一次（断点续跑）')
    ap.add_argument('--resume', action='store_true',
                    help='接着已有缓存存')
    a = ap.parse_args()

    dev = torch.device('cuda')
    vae, sc = load_dcae(dev)
    print('[*] DC-AE loaded, scaling_factor=%.5f' % sc, flush=True)

    out = a.out or str(KP_ROOT / 'out' / 'dit' /
                        ('_latents_%d_n%d.pt' % (a.size, a.limit)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # ⭐ 断点续跑：已有就接着存（分段跑必需，IDE 里前台任务常被超时打断）
    zs, n, t0 = [], 0, time.time()
    if a.resume and os.path.exists(out):
        b = torch.load(out, map_location='cpu', weights_only=False)
        zs = [b['z']]
        n = int(b['z'].shape[0])
        print('[*] resumed from %s  n=%d' % (out, n), flush=True)

    def _flush():
        z = torch.cat(zs, 0)[:a.limit]
        torch.save({'z': z, 'scale': sc, 'size': a.size}, out)
        print('[flush] %d/%d  %.0fs' % (z.shape[0], a.limit, time.time() - t0),
              flush=True)
        return z

    files = sorted(SHARD_DIR.glob('*.parquet'))[:a.shards]
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=256, columns=['image']):
            rows = b.to_pylist()
            for i in range(0, len(rows), a.batch):
                chunk = rows[i:i + a.batch]
                ims = []
                for r in chunk:
                    try:
                        im = Image.open(io.BytesIO(r['image'])).convert('RGB')
                    except Exception:
                        continue
                    w, h = im.size
                    s = min(w, h)
                    im = im.crop(((w - s) // 2, (h - s) // 2,
                                  (w - s) // 2 + s, (h - s) // 2 + s))
                    im = im.resize((a.size, a.size), Image.LANCZOS)
                    ims.append(torch.from_numpy(
                        np.asarray(im, dtype='uint8').copy()).float()
                        .permute(2, 0, 1) / 127.5 - 1.0)
                if not ims:
                    continue
                x = torch.stack(ims).to(dev)
                with torch.no_grad():
                    z = vae.encode(x).float() * sc
                zs.append(z.cpu())
                n += z.shape[0]
                if a.every and n % a.every < a.batch:
                    _flush()
            if n >= a.limit:
                break
        print('  %s -> %d  (%.0fs)' % (f.name, n, time.time() - t0), flush=True)
        if n >= a.limit:
            break

    z = _flush()
    print('[OK] %s  shape=%s  %.0fMB  %.0fs'
          % (out, tuple(z.shape), os.path.getsize(out) / 2**20, time.time() - t0))


if __name__ == '__main__':
    main()

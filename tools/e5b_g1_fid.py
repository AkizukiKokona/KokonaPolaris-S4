"""E5b · 任务3：G1 判决 —— 四臂 FID(各臂 vs bf16 臂) + PSNR/SSIM 附注

⚠️ 权重通道说明：pytorch-fid 官方的 TF-Inception 权重托管在 GitHub releases，
   本机 GitHub 不可达；本脚本改用 pytorch-fid 库自带的
   `InceptionV3(use_fid_inception=False)`（即 torchvision InceptionV3 ImageNet 权重，
   已离线缓存到 TORCH_HOME/hub/checkpoints）。
   四臂用**同一特征提取器**，因此 FID 的**相对比较**（G1 判据）有效；
   绝对数值不与文献 TF-FID 直接可比。

⚠️ 逐像素 PSNR/SSIM 只作附注，不作门（本项目永久规范：它测的是采样轨迹稳定性）。

用法：source /d/model/env.sh && "$KP_PY" tools/e5b_g1_fid.py
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
from PIL import Image
from scipy import linalg

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations, calculate_frechet_distance

OUT = "D:/model/out/e5b"
IMG_ROOT = os.path.join(OUT, "g1_images")
DIMS = 2048


def feats(files, model, device, bs):
    return get_activations(files, model, batch_size=bs, dims=DIMS,
                           device=device, num_workers=0)


def psnr_ssim(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    p = 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    try:
        from skimage.metrics import structural_similarity as s
        ss = float(s(a, b, channel_axis=2, data_range=255))
    except Exception:
        ss = None
    return p, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bf16,PTQ-W4A8,PTQ-W4A4,TRAIN-W4A8,TRAIN-W4A4")
    ap.add_argument("--ref", default="bf16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]

    dev = torch.device(a.device)
    model = InceptionV3([3], use_fid_inception=False).to(dev)
    model.eval()
    print(f"[fid] 特征提取器 = torchvision InceptionV3 (2048-d pool3), device={dev}", flush=True)

    stats = {}
    for arm in arms:
        d = os.path.join(IMG_ROOT, arm)
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if not files:
            print(f"[fid] {arm}: 无图，跳过", flush=True)
            continue
        act = feats(files, model, dev, a.batch)
        mu, sig = np.mean(act, axis=0), np.cov(act, rowvar=False)
        stats[arm] = {"mu": mu, "sig": sig, "n": len(files), "files": files}
        print(f"[fid] {arm}: n={len(files)} feats={act.shape}", flush=True)

    if a.ref not in stats:
        print(f"[fid] 参考臂 {a.ref} 无图 ⇒ 无法判决", flush=True)
        return

    ref = stats[a.ref]
    res = {}
    for arm, st in stats.items():
        fid = 0.0 if arm == a.ref else float(calculate_frechet_distance(
            st["mu"], st["sig"], ref["mu"], ref["sig"]))
        # PSNR/SSIM 附注（按文件名配对）
        ps, ss = [], []
        if arm != a.ref:
            refmap = {os.path.basename(f): f for f in ref["files"]}
            for f in st["files"]:
                b = os.path.basename(f)
                if b in refmap:
                    ia = np.asarray(Image.open(f).convert("RGB"))
                    ib = np.asarray(Image.open(refmap[b]).convert("RGB"))
                    if ia.shape == ib.shape:
                        p, s = psnr_ssim(ia, ib)
                        ps.append(p)
                        if s is not None:
                            ss.append(s)
        res[arm] = {"n": st["n"], "fid_vs_ref": round(fid, 4),
                    "psnr_mean": round(float(np.mean(ps)), 2) if ps else None,
                    "ssim_mean": round(float(np.mean(ss)), 4) if ss else None}

    reffid = None
    # ---- 采样噪声地板：把 bf16 臂按随机种子奇偶拆成两半，算 FID(半A, 半B)。
    #      这是「同一分布、不同噪声」下 FID 的本底，用来判断某臂是否与 bf16 不可区分。
    floor = None
    import re as _re

    def seed_of(fn):
        m = _re.search(r"__s(\d+)\.png$", fn)
        return int(m.group(1)) if m else -1

    a_half = [f for f in ref["files"] if seed_of(os.path.basename(f)) % 2 == 0]
    b_half = [f for f in ref["files"] if seed_of(os.path.basename(f)) % 2 == 1]
    if len(a_half) >= 8 and len(b_half) >= 8:
        fa = feats(a_half, model, dev, a.batch)
        fb = feats(b_half, model, dev, a.batch)
        floor = {
            "n_a": len(a_half), "n_b": len(b_half),
            "fid": round(float(calculate_frechet_distance(
                np.mean(fa, axis=0), np.cov(fa, rowvar=False),
                np.mean(fb, axis=0), np.cov(fb, rowvar=False))), 4),
        }
        print(f"[fid] 采样噪声地板 FID(bf16 半A vs 半B, n={len(a_half)}/{len(b_half)}) = "
              f"{floor['fid']}", flush=True)
    else:
        print("[fid] 图太少，跳过地板估计", flush=True)
    print("\n" + "=" * 74)
    print(f"{'arm':<14}{'N':>5}{'FID vs bf16':>14}{'PSNR(dB)':>11}{'SSIM':>9}")
    print("-" * 74)
    for arm, r in res.items():
        print(f"{arm:<14}{r['n']:>5}{r['fid_vs_ref']:>14.4f}"
              f"{(r['psnr_mean'] if r['psnr_mean'] is not None else float('nan')):>11.2f}"
              f"{(r['ssim_mean'] if r['ssim_mean'] is not None else float('nan')):>9.4f}")

    fp = os.path.join(OUT, "g1_fid.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"ref": a.ref, "extractor": "torchvision_inception_v3_pool3",
                   "noise_floor": floor, "results": res}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 写入 {fp}", flush=True)


if __name__ == "__main__":
    main()

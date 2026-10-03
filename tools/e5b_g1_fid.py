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

# ⚠️ 入口自举（照 tools/onboard.py:18 / tools/e5b_common.py 的做法）：
#    `python tools/e5b_g1_fid.py` 时 sys.path[0] 是 tools/，不是仓库根
#    ⇒ 必须在 import kp 之前把仓库根插进去，否则 ModuleNotFoundError。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kp.paths import OUT
import numpy as np
import torch
from PIL import Image

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Fréchet 距离（FID），**自己实现**而不是调 `pytorch_fid` 的同名函数。

    ⚠️ 为什么不用库里的：`pytorch_fid.fid_score.calculate_frechet_distance`
       内部调用 `scipy.linalg.sqrtm(..., disp=False)`，而 **SciPy ≥ 1.18 已移除
       `disp` 参数** ⇒ 直接抛 `TypeError: sqrtm() got an unexpected keyword
       argument 'disp'`。本项目装的是新版 scipy ⇒ 那条路**必然崩**。
       （又是一次「报错文本 ≠ 根因」：表面像 scipy 坏了，实为上游库未适配。）

    秩亏时 `sqrtm` 的虚部会显著增大 ⇒ 这里显式检查并**如实抛错**，
    而不是像库那样 `.real` 一下把虚部悄悄丢掉（那样会得到一个看似正常的假数字）。
    """
    import numpy as np
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        raise ValueError(f"FID 计算出现非有限值（μ 差 {np.abs(diff).max():.3g}）")
    if np.iscomplexobj(covmean):
        # ⚠️ 必须检查**整个矩阵**的虚部，不能只看对角线 —— FID 用 trace（对角线之和），
        #    但虚部若在对角线之外显著，下游 float() 会触发 ComplexWarning
        #    「discards the imaginary part」⇒ 静默丢信息、给出看似正常的假数字。
        imax = float(np.abs(covmean.imag).max())
        if imax > 1e-3:
            raise ValueError(
                f"FID 的 sqrtm 虚部显著非零（max {imax:.3g}）"
                " ⇒ 协方差矩阵**秩亏**（样本数 n <= 特征维度 dims）。"
                "此时 FID 主要反映估计偏差而非分布差异，**不可作过门依据**。")
        covmean = covmean.real
    covmean = np.asarray(covmean, dtype=np.float64)
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


OUT = OUT / "e5b"
IMG_ROOT = os.path.join(OUT, "g1_images")
DIMS = 2048


def feats(files, model, device, bs):
    return get_activations(files, model, batch_size=bs, dims=DIMS,
                           device=device, num_workers=0)


def _seed_of(fn):
    import re as _re
    m = _re.search(r"__s(\d+)\.png$", os.path.basename(fn))
    return int(m.group(1)) if m else None


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
    ap.add_argument("--seeds", default="",
                    help="⭐ 只取这些 seed（逗号分隔，如 1000,1001）——"
                         "**保证各臂同 N**。FID 对样本数极其敏感，N 不等则不可比。")
    ap.add_argument("--dims", type=int, default=DIMS,
                    help="特征维度。⚠️ n <= dims 时协方差秩亏、FID 数值不稳（见运行时的警告）")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    seed_filter = {int(s) for s in a.seeds.split(",") if s.strip()} if a.seeds else None

    dev = torch.device(a.device)
    model = InceptionV3([3], use_fid_inception=False).to(dev)
    model.eval()
    print(f"[fid] 特征提取器 = torchvision InceptionV3 (2048-d pool3), device={dev}", flush=True)

    stats = {}
    ns = {}
    for arm in arms:
        d = os.path.join(IMG_ROOT, arm)
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if seed_filter is not None:          # ⭐ 强制各臂同 N
            files = [f for f in files if _seed_of(f) in seed_filter]
        if not files:
            print(f"[fid] {arm}: 无图，跳过", flush=True)
            continue
        act = feats(files, model, dev, a.batch)
        mu, sig = np.mean(act, axis=0), np.cov(act, rowvar=False)
        stats[arm] = {"mu": mu, "sig": sig, "n": len(files), "files": files}
        ns[arm] = len(files)
        print(f"[fid] {arm}: n={len(files)} feats={act.shape}", flush=True)

    # ⚠️ 样本数一致性守卫：N 不等 ⇒ FID 绝对不可比（FID 对 N 极敏感）
    if len(set(ns.values())) > 1:
        print(f"\n🚨 各臂样本数不一致 {ns} ⇒ **FID 不可比**，请用 --seeds 对齐", flush=True)
    # ⚠️ 秩亏守卫：n <= dims 时协方差奇异，FID 主要反映估计偏差而非分布差异
    for arm, n in ns.items():
        if n <= DIMS:
            print(f"⚠️ {arm}: n={n} <= dims={DIMS} ⇒ 协方差秩亏，FID 数值不稳；"
                  f"本结果只能作**方向性**参考，不能当正式过门依据", flush=True)

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
    # ---- 判决：各臂 FID 是否**显著高于**「同分布、不同噪声」的地板
    verdict = {}
    print("\n【判决 · 对采样噪声地板】")
    if floor is None:
        print("  ⏳ 无地板估计 ⇒ 无法判断信号是否高于噪声，如实报缺口")
    else:
        print(f"  地板 = FID(bf16 半A vs 半B, n={floor['n_a']}/{floor['n_b']}) = {floor['fid']}")
        for arm, r in res.items():
            if arm == a.ref:
                continue
            ratio = r["fid_vs_ref"] / floor["fid"] if floor["fid"] > 0 else float("inf")
            ok = ratio > 1.5
            verdict[arm] = {"fid_vs_ref": r["fid_vs_ref"], "floor": floor["fid"],
                            "ratio": round(ratio, 3),
                            "distinguishable": bool(ok)}
            print(f"  {arm:<14} FID {r['fid_vs_ref']:>9.4f}  ÷ 地板 = {ratio:>7.2f}×  "
                  f"⇒ {'高于地板，可分辨' if ok else '≈地板，与 bf16 不可分辨'}")
        print("  ⚠️ 门线取「比值 > 1.5×」是**方向性**判据；官方门（FID 差距 < 2%）")
        print("     是在大样本（数千张）上标定的，小 N 下**不适用**。")

    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"ref": a.ref, "extractor": "torchvision_inception_v3_pool3",
                   "dims": DIMS, "seeds": sorted(seed_filter) if seed_filter else None,
                   "n_per_arm": ns, "noise_floor": floor,
                   "verdict": verdict, "results": res}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 写入 {fp}", flush=True)


if __name__ == "__main__":
    main()

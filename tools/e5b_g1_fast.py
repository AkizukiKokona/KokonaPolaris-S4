"""E5b · G1 快速判据（零生成，~1 分钟）

思路（为什么能替代 FID）：
  臂之间**同 prompt 同 seed** ⇒ 天生配对，可直接比"量化把这张图推开了多远"。
  但"多远"要有**尺度参照**：不同 seed 本来就长得不一样。
  ⇒ 用 bf16 的 39 张图（3 prompt × 13 seed）算「同 prompt 不同 seed 的自然离散度 d_ref」，
     再算「同 prompt 同 seed 下 量化臂 vs bf16 的距离 d_quant」。
     判据：**d_quant / d_ref ≪ 1 ⇒ 量化造成的偏移远小于"换一个 seed"，
           即量化没有改变分布，只是同分布内的抖动。**

数据（全部现成，零生成）：
  A. out/e5/images/{bf16,W4A8,W4A16,W4A4}__{01_en_scene,02_zh_text,03_anime}.png  ← 同 seed=42，配对
  B. out/e5b/g1/bf16/*.png  ← 3 prompt × 13 seed，用作 d_ref 基准

特征：InceptionV3 2048-d（torchvision 权重已在本机缓存）
运行：source /d/model/env.sh && "$KP_PY" tools/e5b_g1_fast.py
"""

import sys
from pathlib import Path

# ⚠️ 入口自举：直接 `python tools/e5b_g1_fast.py` 时 `sys.path[0]` 是 **tools/** 而非仓库根
#    ⇒ `import kp.paths` 报 ModuleNotFoundError；照 tools/onboard.py:18，⛔ 不写死绝对路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.paths import OUT  # noqa: E402
import os, re, json, itertools
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import inception_v3, Inception_V3_Weights

E5 = OUT / "e5/images"
G1B = OUT / "e5b/g1/bf16"
OUT = OUT / "e5b/g1_fast.json"
DEV = "cuda"
PROMPTS = ["01_en_scene", "02_zh_text", "03_anime"]
ARMS = ["W4A8", "W4A16", "W4A4"]


class Feat(nn.Module):
    def __init__(self):
        super().__init__()
        m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        m.fc = nn.Identity()
        m.eval()
        self.m = m.to(DEV)

    @torch.no_grad()
    def __call__(self, paths):
        out = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((299, 299), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
            x = (x - 0.5) / 0.5
            out.append(x)
        x = torch.stack(out).to(DEV)
        return self.m(x)


def l2(a, b):
    return float((a - b).norm(dim=-1).mean())


def psnr(p, q):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
    b = np.asarray(Image.open(q).convert("RGB")).astype(np.float64)
    if a.shape != b.shape:
        return None
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


if __name__ == "__main__":
    F = Feat()
    rep = {"method": "paired feature-space distance vs cross-seed natural spread"}

    # ---- d_ref：bf16 同 prompt 不同 seed 的自然离散度 ----
    groups = {p: [] for p in PROMPTS}
    for f in sorted(os.listdir(G1B)):
        m = re.match(r"(\w+?)_s(\d+)\.png", f)
        if m and m.group(1) in groups:
            groups[m.group(1)].append(os.path.join(G1B, f))
    d_ref, n_ref = [], 0
    per_prompt = {}
    for p, fs in groups.items():
        if len(fs) < 3:
            continue
        X = F(fs)
        ds = [float((X[i] - X[j]).norm()) for i, j in itertools.combinations(range(len(fs)), 2)]
        per_prompt[p] = {"n_img": len(fs), "d_ref": round(float(np.mean(ds)), 3)}
        d_ref.extend(ds); n_ref += len(ds)
    D_REF = float(np.mean(d_ref))
    print(f"[d_ref] bf16 同 prompt 不同 seed：{n_ref} 对，平均特征距离 = {D_REF:.3f}")
    for p, v in per_prompt.items():
        print(f"        {p:<14} n={v['n_img']:>2}  d_ref={v['d_ref']:.3f}")
    rep["d_ref_mean"] = round(D_REF, 3)
    rep["d_ref_detail"] = per_prompt

    # ---- d_quant：同 prompt 同 seed 下 量化臂 vs bf16 ----
    ref_p = {p: os.path.join(E5, f"bf16__{p}.png") for p in PROMPTS}
    Xr = F([ref_p[p] for p in PROMPTS])
    rep["arms"] = {}
    print(f"\n{'arm':<8}{'d_quant':>10}{'d_quant/d_ref':>16}{'PSNR(dB)':>10}")
    print("-" * 46)
    for arm in ARMS:
        ps = [os.path.join(E5, f"{arm}__{p}.png") for p in PROMPTS]
        Xa = F(ps)
        ds = [float((Xr[i] - Xa[i]).norm()) for i in range(len(PROMPTS))]
        dq = float(np.mean(ds))
        pns = [psnr(ref_p[p], ps[i]) for i, p in enumerate(PROMPTS)]
        pns = [v for v in pns if v is not None]
        rep["arms"][arm] = {
            "d_quant": round(dq, 3),
            "ratio_vs_d_ref": round(dq / D_REF, 4),
            "psnr_db_mean": round(float(np.mean(pns)), 2) if pns else None,
            "psnr_db_detail": [round(v, 2) for v in pns],
        }
        print(f"{arm:<8}{dq:>10.3f}{dq/D_REF:>16.4f}"
              f"{(round(float(np.mean(pns)),2) if pns else float('nan')):>10.2f}")

    rep["note"] = ("d_quant/d_ref ≪ 1 ⇒ 量化偏移远小于「换一个 seed」，"
                   "即未改变分布；≈1 或更大 ⇒ 量化已把样本推到分布外（真掉点）。"
                   "PSNR 仅作附注（本项目规范：逐像素指标不作质量判决）。")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 写入 {OUT}")

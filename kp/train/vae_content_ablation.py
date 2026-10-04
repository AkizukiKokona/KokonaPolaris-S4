"""P1 · VAE「内容覆盖」消融 —— 用数字回答「VAE 训练数据里加不加敏感内容」。

═══ 要回答的问题（用户原话）═══
> 「vae 数据里加不加入敏感内容，对生成有没有影响？」

**理论预期**：VAE 是**压缩器不是生成器** ⇒ 它只影响**重建保真度**，
不影响**生成倾向**（后者由主干数据决定）。判据是「**VAE 覆盖 ⊇ 主干目标域**」。
⇒ 本实验测的是**前半句**：内容覆盖是否真的影响重建。

═══ ⭐ 实验设计（控制变量的关键）═══

⚠️ 天真做法是「A=仅 safe vs B=全部」，但那样 **A 和 B 的图片数不同** ⇒
分不清差异来自**内容**还是**数据量**。⇒ 必须**等量**：

    A = 训练池里的 safe 图，全部               （约 7,000 张）
    B = 从训练池里**随机抽 |A| 张**（保持各档比例） （同 7,000 张，含非 safe）

两者**张数相同、步数相同、种子相同、超参相同** ⇒ 唯一差异 = **内容构成**。

    eval = **分层留出集**（每档等量抽样），**A / B 都看不到**
    ⇒ 且评估**按档分解**：只有分解才能看出「哪一类变好/变差」

⚠️ `kp/train/vae_pretrain.py` 的固定评估集是 `paths[:16]`（同源，它自己标了
「不能当泛化指标」）⇒ 本实验**另建真正的留出集**，不复用那个。

═══ 判读规则（先说死，避免事后找理由）═══
    · B 在**非 safe 档**显著优于 A（Δl1 < 0）且 **safe 档基本持平** ⇒ **覆盖缺口成立**
      ⇒ 结论：「VAE 数据里加敏感内容**确实提升那类图的重建**」（但仍**不**影响生成倾向）
    · B ≈ A 在所有档位 ⇒ **覆盖缺口不成立** ⇒ 结论：「VAE 对内容不敏感」，strict 够用
    · B 在 safe 档也变差 ⇒ 说明是**数据量/难度混杂**，判据不干净，需复核

═══ 用法 ═══
    python -m kp.train.vae_content_ablation --cache out/cache/danbooru20k_256 \
        --steps 3000 --batch 16 --size 128 --n-eval-per-class 200 --out out/vae/ablation.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..models.vae import HybridVAE
from .vae_pretrain import block_recon_loss

CLASSES = ("safe", "sensitive", "questionable", "explicit")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_cache(base: str | Path) -> tuple:
    """读 `<base>.npy` + `<base>.json`。→ (memmap uint8, meta dict)。"""
    b = Path(base).with_suffix("")
    npy, js = b.with_suffix(".npy"), b.with_suffix(".json")
    if not npy.exists() or not js.exists():
        raise FileNotFoundError(f"缓存不存在：{npy} / {js}（先跑 kp.data.predecode）")
    arr = np.load(npy, mmap_mode="r")
    meta = json.loads(js.read_text(encoding="utf-8"))
    return arr, meta


def batch_to_tensor(arr, idx: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """uint8 (N,256,256,3) → (B,3,size,size) ∈ [-1,1]。

    ⚠️ 缓存是 **RGB**（无 alpha）⇒ 与 `vae_pretrain.load_batch` 的
    「RGBA → alpha 预乘」在无 alpha 时**等价**（预乘 = 乘 1），口径一致。
    """
    u8 = np.ascontiguousarray(arr[np.asarray(idx)])
    t = torch.from_numpy(u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
    return t.mul_(2.0).sub_(1.0)


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------
def build_splits(items: Sequence[dict], *, n_eval_per_class: int, seed: int) -> dict:
    """分层留出 eval + 等量 A/B 训练集。"""
    rng = np.random.default_rng(seed)
    by = {c: np.array([i for i, it in enumerate(items) if it["rating"] == c])
          for c in CLASSES}
    eval_idx: List[int] = []
    for c in CLASSES:
        pool = by[c]
        n = min(n_eval_per_class, max(1, len(pool) // 5))
        eval_idx.extend(rng.choice(pool, size=n, replace=False).tolist())
    eval_idx = np.array(sorted(eval_idx))
    is_eval = np.zeros(len(items), dtype=bool)
    is_eval[eval_idx] = True
    train_pool = np.array([i for i in range(len(items)) if not is_eval[i]])

    pool_ratings = np.array([items[i]["rating"] for i in train_pool])
    A = train_pool[pool_ratings == "safe"]                       # 仅 safe
    n = len(A)
    B = rng.choice(train_pool, size=min(n, len(train_pool)), replace=False)
    return {"eval_idx": eval_idx, "train_pool": train_pool, "A": A, "B": B,
            "by_class_eval": {c: np.array([i for i in eval_idx
                                           if items[i]["rating"] == c]) for c in CLASSES}}


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one(arr, idx: np.ndarray, *, steps: int, batch: int, size: int,
              device: torch.device, seed: int, lr: float, base: int,
              w_lpips: float, w_grad: float, log_every: int,
              tag: str) -> HybridVAE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    vae = HybridVAE(base=base).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        bi = rng.integers(0, len(idx), size=batch)
        x = batch_to_tensor(arr, idx[bi], size, device)
        r = block_recon_loss(vae, x, w_lpips=w_lpips, w_grad=w_grad)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"    [{tag}] step {step:5d}  loss {float(r['loss']):.4f}  "
                  f"l1 {r['l1']:.4f}  ({time.time()-t0:.0f}s)")
    vae.eval()
    print(f"    [{tag}] 训练完成 {steps} 步 / {time.time()-t0:.0f}s")
    return vae


@torch.no_grad()
def per_image_metrics(vae: HybridVAE, arr, idx: np.ndarray, *, size: int,
                      device: torch.device, batch: int = 32) -> dict:
    """逐图 L1 / PSNR（**逐图**算，才能按档分解）。"""
    l1s, psnrs = [], []
    for s in range(0, len(idx), batch):
        x = batch_to_tensor(arr, idx[s:s + batch], size, device)
        z = vae.encode_latent(x)
        rec = torch.tanh(vae.decode(z))
        e = (rec - x).abs().mean(dim=(1, 2, 3))
        mse = ((rec - x) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        # 值域 [-1,1] ⇒ 峰值 2 ⇒ 峰值² = 4
        psnr = 10.0 * torch.log10(4.0 / mse)
        l1s.extend(e.cpu().tolist())
        psnrs.extend(psnr.cpu().tolist())
    return {"l1": np.array(l1s), "psnr": np.array(psnrs)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _run_single(cache: str, *, steps: int, batch: int, size: int, base: int,
                lr: float, seed: int, device: str, n_eval_per_class: int,
                w_lpips: float, w_grad: float, log_every: int,
                verbose: bool) -> dict:
    """跑**一个种子**：A / B 各训一次，在同一留出集上分档评估。"""
    dev = torch.device(device)
    arr, meta = load_cache(cache)
    items = meta["items"]
    sp = build_splits(items, n_eval_per_class=n_eval_per_class, seed=seed)
    A, B, ev = sp["A"], sp["B"], sp["eval_idx"]
    cnt_B = {c: int(sum(items[i]["rating"] == c for i in B)) for c in CLASSES}
    if verbose:
        print(f"  缓存 {meta['n']:,} 张 · {meta['size']}² · "
              f"分片 {len(meta['source_shards'])} 个")
        print(f"  档位分布: {meta['rating_counts']}")
        print()
        print("=" * 74)
        print("实验设计（⭐ 等量控制变量）")
        print("=" * 74)
        print(f"  留出评估集 : {len(ev):,} 张（每档 {n_eval_per_class}，分层）")
        print(f"  训练池     : {len(sp['train_pool']):,} 张")
        print(f"  A 仅 safe  : {len(A):,} 张")
        print(f"  B 等量混合 : {len(B):,} 张   构成 {cnt_B}")
        print("  ⇒ A/B **张数相同**、步数/种子/超参相同 ⇒ 唯一差异 = 内容构成")
        print()
        print("=" * 74)
        print("训练")
        print("=" * 74)

    mA = train_one(arr, A, steps=steps, batch=batch, size=size, device=dev,
                   seed=seed, lr=lr, base=base, w_lpips=w_lpips, w_grad=w_grad,
                   log_every=log_every, tag=f"A(safe,s{seed})")
    mB = train_one(arr, B, steps=steps, batch=batch, size=size, device=dev,
                   seed=seed, lr=lr, base=base, w_lpips=w_lpips, w_grad=w_grad,
                   log_every=log_every, tag=f"B(mix ,s{seed})")
    resA = per_image_metrics(mA, arr, ev, size=size, device=dev)
    resB = per_image_metrics(mB, arr, ev, size=size, device=dev)
    del mA, mB
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pos = {i: k for k, i in enumerate(ev)}
    table = []
    for c in CLASSES:
        ii = sp["by_class_eval"][c]
        if len(ii) == 0:
            continue
        k = np.array([pos[i] for i in ii])
        a_l1, b_l1 = resA["l1"][k].mean(), resB["l1"][k].mean()
        a_p, b_p = resA["psnr"][k].mean(), resB["psnr"][k].mean()
        table.append({"cls": c, "n": int(len(ii)),
                      "A_l1": float(a_l1), "B_l1": float(b_l1),
                      "d_l1": float(b_l1 - a_l1),
                      "A_psnr": float(a_p), "B_psnr": float(b_p),
                      "d_psnr": float(b_p - a_p)})
    allrow = {"A_l1": float(resA["l1"].mean()), "B_l1": float(resB["l1"].mean()),
              "d_l1": float(resB["l1"].mean() - resA["l1"].mean()),
              "A_psnr": float(resA["psnr"].mean()),
              "B_psnr": float(resB["psnr"].mean()),
              "d_psnr": float(resB["psnr"].mean() - resA["psnr"].mean())}
    if verbose:
        print()
        print("=" * 74)
        print("结果（⛔ 同一留出集，按档分解 —— 绝对值不判画质，只看 A/B 相对差）")
        print("=" * 74)
        print(f"  {'档位':<13}{'n':>5}{'A_l1':>9}{'B_l1':>9}{'Δl1':>9}"
              f"{'A_psnr':>9}{'B_psnr':>9}{'Δpsnr':>9}")
        for r in table:
            print(f"  {r['cls']:<13}{r['n']:>5}{r['A_l1']:>9.4f}{r['B_l1']:>9.4f}"
                  f"{r['d_l1']:>+9.4f}{r['A_psnr']:>9.2f}{r['B_psnr']:>9.2f}"
                  f"{r['d_psnr']:>+9.2f}")
        print(f"  {'全部':<13}{len(ev):>5}{allrow['A_l1']:>9.4f}{allrow['B_l1']:>9.4f}"
              f"{allrow['d_l1']:>+9.4f}{allrow['A_psnr']:>9.2f}{allrow['B_psnr']:>9.2f}"
              f"{allrow['d_psnr']:>+9.2f}")
    return {"seed": seed, "per_class": table, "all": allrow,
            "n_A": int(len(A)), "n_B": int(len(B)), "n_eval": int(len(ev)),
            "B_class_counts": cnt_B, "rating_counts": meta["rating_counts"]}


def run(cache: str, *, steps: int, batch: int, size: int, base: int, lr: float,
        seeds: Sequence[int], device: str, n_eval_per_class: int,
        w_lpips: float, w_grad: float, log_every: int,
        out: Optional[str]) -> dict:
    """跑多个种子并汇总 —— ⚠️ 单种子分不清「真效应」与「噪声」。"""
    per_seed = []
    for i, s in enumerate(seeds):
        print(f"\n{'#' * 74}\n# 种子 {s}（{i+1}/{len(seeds)}）\n{'#' * 74}")
        per_seed.append(_run_single(cache, steps=steps, batch=batch, size=size,
                                    base=base, lr=lr, seed=s, device=device,
                                    n_eval_per_class=n_eval_per_class,
                                    w_lpips=w_lpips, w_grad=w_grad,
                                    log_every=log_every, verbose=(i == 0)))

    # ---- 汇总 ----
    agg = []
    for c in CLASSES:
        dl1 = [r["d_l1"] for rep in per_seed for r in rep["per_class"] if r["cls"] == c]
        a_l1 = [r["A_l1"] for rep in per_seed for r in rep["per_class"] if r["cls"] == c]
        b_l1 = [r["B_l1"] for rep in per_seed for r in rep["per_class"] if r["cls"] == c]
        if not dl1:
            continue
        agg.append({"cls": c, "n_seeds": len(dl1),
                    "A_l1": float(np.mean(a_l1)), "B_l1": float(np.mean(b_l1)),
                    "d_l1_mean": float(np.mean(dl1)), "d_l1_std": float(np.std(dl1)),
                    "d_l1_all": [float(x) for x in dl1]})
    d_all = [rep["all"]["d_l1"] for rep in per_seed]

    print()
    print("=" * 74)
    print(f"汇总（{len(seeds)} 个种子：{list(seeds)}）")
    print("=" * 74)
    print(f"  {'档位':<13}{'A_l1':>9}{'B_l1':>9}{'Δl1 均值':>11}{'Δl1 标准差':>12}")
    for r in agg:
        print(f"  {r['cls']:<13}{r['A_l1']:>9.4f}{r['B_l1']:>9.4f}"
              f"{r['d_l1_mean']:>+11.4f}{r['d_l1_std']:>12.4f}")
    print(f"  {'全部':<13}{'':>9}{'':>9}{np.mean(d_all):>+11.4f}{np.std(d_all):>12.4f}")

    non_safe = [r for r in agg if r["cls"] != "safe"]
    safe_row = next((r for r in agg if r["cls"] == "safe"), None)
    d_non = float(np.mean([r["d_l1_mean"] for r in non_safe])) if non_safe else 0.0
    sd_non = float(np.mean([r["d_l1_std"] for r in non_safe])) if non_safe else 0.0
    # ⭐ A/B 在**同一种子**下用**同一留出集**评估 ⇒ 是**配对差** ⇒
    #    判显著性该用**标准误** SE = SD/√n，而不是 SD 本身。
    #    ⚠️ 两者都打印出来，不下藏 —— n 小的时候差别很大。
    se_non = sd_non / max(np.sqrt(len(seeds)), 1e-9)
    sd_safe = safe_row["d_l1_std"] if safe_row else 0.01
    d_safe = safe_row["d_l1_mean"] if safe_row else 0.0
    significant = abs(d_non) > max(2.0 * se_non, 0.003)
    if significant and d_non < 0 and abs(d_safe) < max(2.0 * (sd_safe / np.sqrt(len(seeds))), 0.005):
        verdict = "覆盖缺口成立：B 在非 safe 档显著更好，safe 档持平"
        detail = ("⇒ **VAE 数据里加敏感内容确实提升那类图的重建**；"
                  "⚠️ 但这**不**等于「主干会画它」—— 生成倾向由主干数据决定")
    elif not significant:
        verdict = "覆盖缺口不成立：差异未超过种子间噪声 ⇒ VAE 对内容不敏感"
        detail = "⇒ 支持 `strict`（只留 safe）在本任务上足够"
    else:
        verdict = "判据不干净（safe 档也明显变化，或方向不一致）⇒ 需复核"
        detail = "⇒ 检查 A/B 张数、种子、步数是否真的一致"
    print()
    print("=" * 74)
    print(f"判读：{verdict}")
    print(f"  Δl1(非safe 平均) = {d_non:+.4f}")
    print(f"    种子间标准差 SD = {sd_non:.4f}   ⇒ 标准误 SE = SD/√{len(seeds)} = {se_non:.4f}")
    print(f"    判定阈 |Δ| > max(2·SE, 0.003) = {max(2.0*se_non, 0.003):.4f}  ⇒ 显著={significant}")
    print(f"  Δl1(safe) = {d_safe:+.4f}（SD {sd_safe:.4f}）")
    print(f"  {detail}")
    print("=" * 74)

    rep = {"cache": str(cache), "steps": steps, "batch": batch, "size": size,
           "base": base, "lr": lr, "seeds": list(seeds),
           "w_lpips": w_lpips, "w_grad": w_grad,
           "aggregate": agg, "per_seed": per_seed,
           "d_all_mean": float(np.mean(d_all)), "d_all_std": float(np.std(d_all)),
           "d_non_mean": d_non, "d_non_std": sd_non, "d_non_se": se_non,
           "d_safe_mean": d_safe,
           "significant": bool(significant), "verdict": verdict, "detail": detail,
           "⚠️_caveat": ("留出集与训练池同源（同一数据集的另一部分）⇒ 可当**相对**比较，"
                         "绝对水平不代表画质（项目规范：逐像素指标不判画质）")}
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  ✅ 结果已存 {p}")
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · VAE 内容覆盖消融")
    ap.add_argument("--cache", required=True, help="kp.data.predecode 的 <base>")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="多个种子（⚠️ 单种子分不清真效应与噪声）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval-per-class", type=int, default=200)
    ap.add_argument("--w-lpips", type=float, default=0.0)
    ap.add_argument("--w-grad", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print("=" * 74)
    print("P1 · VAE「内容覆盖」消融（A=仅safe vs B=等量混合）")
    print("=" * 74)
    run(a.cache, steps=a.steps, batch=a.batch, size=a.size, base=a.base, lr=a.lr,
        seeds=a.seeds, device=a.device, n_eval_per_class=a.n_eval_per_class,
        w_lpips=a.w_lpips, w_grad=a.w_grad, log_every=a.log_every, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

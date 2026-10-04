"""P1 · VAE 评估装置 —— 给「训出可用的 VAE」一个**可量化、且判得了画质**的门。

═══ 为什么必须有它（三条实测教训，都是本项目踩过的）═══

① 🔴 **逐像素指标测「像不像」，测不出「糊不糊」。**
   项目规范写死「逐像素 PSNR/SSIM 不可判画质」。
   实测：感知损失在 L1 上「更差」（0.1329 vs 0.1272）却 `edge` 几乎相同
   ⇒ **不是它没用，是像素指标看不见它**。

② 🔴 **⛔ 不能用训练损失本身当评估指标**（自证）。
   `multiscale_perceptual` 同时是 `--w-lpips` 那一臂的**训练损失**
   ⇒ 拿它当裁判等于让被告当法官 ⇒ 必须用**独立**的预训练感知度量（LPIPS）。

③ 🔴 **评估集必须固定且共享。**
   本项目已踩两次：旧行为 `paths[:16]` 是**同源**（那 16 张也在训练池里）；
   不同运行的随机留出集不同 ⇒ **L1 不可比**。本工具把评估集**固化到磁盘**，
   所有模型都评在同一批图上。

═══ 指标（分三层，⛔ 别混用）═══
| 层 | 指标 | 能判什么 | 不能判什么 |
|---|---|---|---|
| 像素 | **L1 / PSNR / SSIM** | 保真度（像不像） | **画质（糊不糊）** |
| 结构 | Sobel-edge L1 | 轮廓是否还在 | 高频细节是否糊 |
| **感知** | **LPIPS(alex)** | **感知距离（糊不糊）** | 分布级偏差 |
| **分布** | **rFID** | 重建分布 vs 原图分布 | 单图 |

⭐ 全部**独立于本项目的训练损失**（LPIPS/Inception 都是外部预训练权重）。

═══ 用法 ═══
    # ① 固化共享评估集（一次；所有模型都评在它上面）
    python -m kp.train.vae_eval --make-eval-set \\
        --shards out/data/curated_danbooru/_shards \\
        --n 2000 --size 512 --out out/eval/fixed2k

    # ② 评估一个或多个 checkpoint（可加 --include-untrained 当「下限参照」）
    python -m kp.train.vae_eval --eval-set out/eval/fixed2k --size 256 \\
        --ckpt out/vae/long256_lp0.pt out/vae/full338k_256_lp0.pt \\
        --include-untrained --out out/eval/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..models.vae import HybridVAE
from .vae_pretrain import _edge

#: TORCH_HOME 指到仓库内（⛔ 不许落 C 盘）
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / ".torchcache"))


# ---------------------------------------------------------------------------
# 评估集：固化到磁盘（npy + json）
# ---------------------------------------------------------------------------
def make_eval_set(shards: Sequence[str], n: int, size: int, out_base: str,
                  *, seed: int = 0, policy: str = "explicit_ok") -> dict:
    """从分片里固化一批**确定**的评估图（与训练的留出集**同一套索引**）。"""
    from .vae_pretrain import _ShardStreamSource

    src = _ShardStreamSource(shards, size, torch.device("cpu"), policy=policy)
    eval_idx, held = src.holdout_index(n, seed)
    if not held:
        raise ValueError(f"数据不足，无法留出（n={n}）。请加大数据或减小 --n。")
    print(f"  留出 {len(eval_idx)} 张（分片内前若干行，确定性 seed={seed}）")
    t = src.get(eval_idx, size)                       # (N,3,size,size) ∈ [-1,1]
    u8 = ((t + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = u8.permute(0, 2, 3, 1).cpu().numpy()        # → (N,size,size,3)

    b = Path(out_base).with_suffix("")
    npy, js = b.with_suffix(".npy"), b.with_suffix(".json")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, arr)
    meta = {"n": int(arr.shape[0]), "size": int(size),
            "source_shards": [f.name for f in src.files],
            "eval_idx": [int(i) for i in eval_idx], "seed": seed, "policy": policy,
            "⚠️_note": ("固化评估集：所有模型必须评在**同一份**上，否则 L1 不可比。"
                        "训练若想与它对齐，需 `--n-eval` ≥ 本 n 且 `--seed` 相同"
                        "（两者共用 holdout_index）。")}
    js.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ {npy.name}  {arr.shape}  {os.path.getsize(npy)/1e6:.0f} MB")
    print(f"  ✅ {js.name}")
    return meta


def load_eval_set(base: str | Path, size: int, device: torch.device) -> torch.Tensor:
    """读固化评估集 → (N,size,size,3) uint8 张量（放到 device，只缩放一次）。"""
    b = Path(base).with_suffix("")
    npy, js = b.with_suffix(".npy"), b.with_suffix(".json")
    if not npy.exists():
        raise FileNotFoundError(f"评估集不存在：{npy}（先 --make-eval-set）")
    arr = np.load(npy)
    t = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
    return t                                            # [0,1]，(N,3,size,size)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _ssim_batch(x01: np.ndarray, y01: np.ndarray) -> List[float]:
    """逐图 SSIM（skimage）。输入 (N,H,W,3) ∈ [0,1]。⚠️ CPU、逐图，慢但准。"""
    from skimage.metrics import structural_similarity as ssim
    return [float(ssim(x01[i], y01[i], data_range=1.0, channel_axis=2))
            for i in range(len(x01))]


@torch.no_grad()
def _inception_feats(batches, device: str, dim: int = 2048) -> np.ndarray:
    """用 pytorch-fid 的 Inception 取特征（输入 [0,1] NCHW，内部自行 resize/归一）。"""
    from pytorch_fid.inception import InceptionV3
    block = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
    inc = InceptionV3([block], resize_input=True, normalize_input=True,
                      use_fid_inception=True).to(device).eval()
    out: List[np.ndarray] = []
    for b in batches:
        f = inc(b)[0]
        out.append(f.squeeze(-1).squeeze(-1).cpu().numpy())
    del inc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def _fid_from_feats(f1: np.ndarray, f2: np.ndarray) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    mu2, s2 = f2.mean(0), np.cov(f2, rowvar=False)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


@torch.no_grad()
def evaluate_model(vae: HybridVAE, x01: torch.Tensor, *, device: str,
                   lpips_fn=None, batch: int = 32,
                   want_rfid: bool = True) -> Dict[str, float]:
    """在固化评估集上算全部指标。`x01` 是 [0,1] 的 (N,3,H,W)。"""
    vae.eval()
    n = x01.shape[0]
    l1s, psnrs, edges, lp, ss = [], [], [], [], []
    recs: List[np.ndarray] = []
    for s in range(0, n, batch):
        xb = x01[s:s + batch]
        xin = xb.mul(2.0).sub_(1.0)                     # [-1,1]
        rec = torch.tanh(vae.decode(vae.encode_latent(xin)))
        e = (rec - xin).abs().mean(dim=(1, 2, 3))
        mse = ((rec - xin) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        l1s.extend(e.cpu().tolist())
        psnrs.extend((10.0 * torch.log10(4.0 / mse)).cpu().tolist())
        edges.extend((_edge(xin) - _edge(rec)).abs().mean(dim=(1, 2, 3)).cpu().tolist())
        if lpips_fn is not None:
            lp.extend(lpips_fn(xin, rec).flatten().cpu().tolist())
        # ⚠️ SSIM 在 [0,1] 上算 ⇒ 先转回来；同时把 rec 存成 [0,1] 供 rFID 用
        rec01 = ((rec + 1.0) * 0.5).clamp(0, 1)
        ss.extend(_ssim_batch(xb.cpu().numpy().transpose(0, 2, 3, 1),
                              rec01.cpu().numpy().transpose(0, 2, 3, 1)))
        recs.append(rec01.cpu())
    vae.train()
    out = {"l1": float(np.mean(l1s)), "psnr": float(np.mean(psnrs)),
           "ssim": float(np.mean(ss)), "edge": float(np.mean(edges)), "n": int(n)}
    if lp:
        out["lpips"] = float(np.mean(lp))
    if want_rfid:
        rec_all = torch.cat(recs, 0)
        f_orig = _inception_feats((x01[s:s + batch] for s in range(0, n, batch)), device)
        f_rec = _inception_feats((rec_all[s:s + batch] for s in range(0, n, batch)), device)
        out["rfid"] = _fid_from_feats(f_orig, f_rec)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · VAE 评估装置（含感知/分布级指标）")
    ap.add_argument("--make-eval-set", action="store_true", help="固化共享评估集")
    ap.add_argument("--shards", nargs="*", default=None, help="（make 模式）parquet 分片或目录")
    ap.add_argument("--n", type=int, default=2000, help="（make 模式）评估集张数")
    ap.add_argument("--eval-size", type=int, default=512, help="（make 模式）评估集存盘分辨率")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="explicit_ok")
    ap.add_argument("--eval-set", default=None, help="固化评估集基名/<npy>")
    ap.add_argument("--ckpt", nargs="*", default=[], help="要评的 .pt（可多个）")
    ap.add_argument("--include-untrained", action="store_true",
                    help="额外评一个随机初始化模型当**下限参照**")
    ap.add_argument("--size", type=int, default=256, help="评估分辨率（所有模型一致）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-rfid", action="store_true", help="跳过 rFID（省时间）")
    ap.add_argument("--no-lpips", action="store_true", help="跳过 LPIPS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.make_eval_set:
        if not a.shards:
            print("⛔ --make-eval-set 需要 --shards"); return 2
        print("=" * 74); print("P1 · 固化共享评估集"); print("=" * 74)
        make_eval_set(a.shards, a.n, a.eval_size, a.out or "out/eval/fixed",
                      seed=a.seed, policy=a.policy)
        return 0

    if not a.eval_set or not (a.ckpt or a.include_untrained):
        print("用法：① --make-eval-set --shards ... --n ... --eval-size ... --out <base>"
              "\n      ② --eval-set <base> --ckpt a.pt b.pt [--include-untrained]")
        return 2

    dev = a.device
    x01 = load_eval_set(a.eval_set, a.size, torch.device(dev))
    print("=" * 74)
    print(f"P1 · VAE 评估   评估集 {tuple(x01.shape)}  评估分辨率 {a.size}²")
    print("=" * 74)
    lpips_fn = None
    if not a.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        print("  LPIPS(alex) 已加载（权重随包自带 ⇒ 独立于本项目训练损失）")

    rows = []
    todo = [(Path(p).stem, p) for p in a.ckpt]
    if a.include_untrained:
        todo.append(("untrained随机初始化", None))
    for tag, path in todo:
        if path is None:
            vae = HybridVAE(base=16).to(dev)
        else:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            base = int(blob.get("base", 16))
            vae = HybridVAE(base=base).to(dev)
            vae.load_state_dict(blob["state_dict"])
            tag = f"{tag}(base{base})"
        t0 = time.time()
        m = evaluate_model(vae, x01, device=dev, lpips_fn=lpips_fn,
                           batch=a.batch, want_rfid=not a.no_rfid)
        m["ckpt"] = tag
        m["sec"] = round(time.time() - t0, 1)
        rows.append(m)
        print(f"  ✅ {tag:<34} L1 {m['l1']:.4f}  PSNR {m['psnr']:5.2f}  "
              f"SSIM —  edge {m['edge']:.4f}"
              + (f"  LPIPS {m['lpips']:.4f}" if "lpips" in m else "")
              + (f"  rFID {m['rfid']:.1f}" if "rfid" in m else "")
              + f"   ({m['sec']}s)")
        del vae

    print()
    hdr = f"  {'模型':<36}{'L1':>9}{'PSNR':>8}{'SSIM':>9}{'edge':>9}"
    if any("lpips" in m for m in rows):
        hdr += f"{'LPIPS':>9}"
    if any("rfid" in m for m in rows):
        hdr += f"{'rFID':>9}"
    print(hdr)
    for m in rows:
        line = (f"  {m['ckpt']:<36}{m['l1']:>9.4f}{m['psnr']:>8.2f}"
                f"{m.get('ssim', float('nan')):>9.4f}{m['edge']:>9.4f}")
        if "lpips" in m:
            line += f"{m['lpips']:>9.4f}"
        if "rfid" in m:
            line += f"{m['rfid']:>9.1f}"
        print(line)
    print("\n  ⚠️ 方向：L1/edge/LPIPS/rFID **越低越好**；PSNR/SSIM **越高越好**。")
    print("  ⚠️ 「随机初始化」那行是**下限参照** —— 它到「完美」之间的距离，"
          "才是模型真正的进步空间。")

    rep = {"eval_set": str(a.eval_set), "eval_size": a.size, "rows": rows,
           "⚠️_caveat": ("LPIPS/Inception 都是**外部预训练** ⇒ **独立于本项目训练损失**"
                         "（不会自证）；rFID 在 ~2k 图上偏噪，仅用于**同一评估集**的相对比较。")}
    if a.out:
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  ✅ 报告已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

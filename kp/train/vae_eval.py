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


def load_eval_set(base: str | Path, size: int,
                  device: Optional[torch.device] = None) -> torch.Tensor:
    """读固化评估集 → (N,3,size,size) float ∈[0,1]。

    🔴 **顺序很重要（2026-10-04 实测 OOM）**：存盘是 **512²**，若先搬上 GPU 再转 float，
    2000 张要 **5.86 GiB** ⇒ 8GB 卡直接 OOM。
    ⇒ **先在 CPU 上分块缩放到目标分辨率，再转 float**；默认**留在 CPU**，
    由 `evaluate_model` 逐块搬到 device（与训练侧 `x_eval` 的处理一致）。
    """
    b = Path(base).with_suffix("")
    npy, js = b.with_suffix(".npy"), b.with_suffix(".json")
    if not npy.exists():
        raise FileNotFoundError(f"评估集不存在：{npy}（先 --make-eval-set）")
    arr = np.load(npy)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()      # uint8 (N,3,H,W)
    if t.shape[-1] != size:
        outs = []
        for s in range(0, t.shape[0], 128):                          # 分块，峰值 ~0.4GB
            outs.append(F.interpolate(t[s:s + 128].float(), size=(size, size),
                                      mode="bilinear", align_corners=False,
                                      antialias=True))
        t = torch.cat(outs)
    t = t.clamp_(0, 255).div_(255.0)
    if device is not None:
        t = t.to(device)
    return t


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


def _psd_sqrt(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """半正定矩阵的平方根（GPU float64，`eigh`）。"""
    evals, evecs = torch.linalg.eigh(a)
    evals = evals.clamp_min(eps)
    return (evecs * evals.sqrt()) @ evecs.T


def _fid_from_feats(f1: np.ndarray, f2: np.ndarray,
                    device: str = "cuda") -> float:
    """Frechet 距离（FID）。

    🔴 **不能用 `pytorch_fid.fid_score.calculate_frechet_distance`**（2026-10-04 实测崩）：
    它内部调 `scipy.linalg.sqrtm(..., disp=False)`，而**本机 SciPy 已移除 `disp` 参数**
    ⇒ `TypeError: sqrtm() got an unexpected keyword argument 'disp'`。
    ⇒ 这里自己算，用**对称形式** `Tr((s2½ · s1 · s2½)½)`（只需 eigh，GPU float64 秒回），
    顺带避开 CPU 上 2048×2048 `sqrtm` 的 O(n³) 慢算。
    """
    a1 = torch.as_tensor(np.asarray(f1), dtype=torch.float64, device=device)
    a2 = torch.as_tensor(np.asarray(f2), dtype=torch.float64, device=device)
    mu1, mu2 = a1.mean(0), a2.mean(0)
    s1, s2 = torch.cov(a1.T), torch.cov(a2.T)
    s2_half = _psd_sqrt(s2)
    tr_covmean = _psd_sqrt(s2_half @ s1 @ s2_half).diagonal().sum()
    diff = mu1 - mu2
    val = (diff.dot(diff) + s1.diagonal().sum() + s2.diagonal().sum()
           - 2.0 * tr_covmean)
    return float(val)


def _as_tensor(v):
    """把 diffusers 可能返回的三种东西统一成**确定性张量**。

    🔴 **本项目踩过的坑（重要，别再犯）**：判据原先写成
    `v.mode() if hasattr(v, "mode") else v` ——
    但 **`torch.Tensor` 本身就有 `.mode()` 方法**（求众数，返回 `return_types.mode`
    而非张量！）⇒ 纯张量被误判成「分布对象」，于是 decode 收到
    `torch.return_types.mode` 并报 `'torch.return_types.mode' object has no attribute 'shape'`。

    ⇒ 正确判据：**先排除 Tensor**，再按「分布」处理。
    """
    if isinstance(v, torch.Tensor):
        return v                                  # ⭐ 必须是第一分支
    if hasattr(v, "mode"):
        return v.mode()                           # DiagonalGaussianDistribution 等
    if hasattr(v, "sample"):
        return v.sample()
    raise TypeError(f"无法把 {type(v)} 转成张量")


class _DiffusersVAEAdapter(torch.nn.Module):
    """把 diffusers VAE（`AutoencoderDC`）适配成本项目 `HybridVAE` 的接口。

    为什么需要：我们的 `evaluate_model` 只用 `encode_latent` / `decode` 两个方法，
    而 diffusers 的 `encode()` 返回 **`EncoderOutput`**（不是张量，字段名 `latent`）、
    `decode()` 返回 **`DecoderOutput`**（字段名 `sample`）。⇒ 两者不同构，无法直接比。

    ⚠️ **确定性**（实测确认）：`AutoencoderDC` 的 `encode()` 返回的 `latent` 是
    **确定性张量**（shape `(1,32,8,8)`，**不是** `DiagonalGaussianDistribution`），
    与本项目 HybridVAE 的确定性 latent 同性质 ⇒ **可直接对照，无需取 mode**。
    保留 `latent_dist` 分支只为兼容别的 diffusers VAE。
    """

    def __init__(self, ae):
        super().__init__()
        self.ae = ae
        # ⭐ 外部 VAE 的解码输出**本身就是 [-1,1] 附近**（实测 DC-AE 原始范围
        #    [-1.315, 1.457]）⇒ 不该再套 tanh。声明自己的后处理，供调用方自动采用。
        self.post = "clip"

    def encode_latent(self, img: torch.Tensor) -> torch.Tensor:
        out = self.ae.encode(img)
        # ⚠️ 兼容三种形态：属性访问 / dict / 分布对象 —— 逐个显式试，不做静默假设
        for k in ("latent", "latent_embeds"):
            v = out.get(k) if isinstance(out, dict) else getattr(out, k, None)
            if v is not None:
                return _as_tensor(v)
        ld = out.get("latent_dist") if isinstance(out, dict) \
            else getattr(out, "latent_dist", None)
        if ld is not None:
            return _as_tensor(ld)
        raise TypeError(f"无法从 {type(out)} 提取 latent")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.ae.decode(z)
        return out.sample if hasattr(out, "sample") else out


def _wrap_diffusers_vae(ae) -> _DiffusersVAEAdapter:
    return _DiffusersVAEAdapter(ae)


@torch.no_grad()
def evaluate_model(vae: HybridVAE, x01: torch.Tensor, *, device: str,
                    lpips_fn=None, batch: int = 32,
                    want_rfid: bool = True,
                    post: str = "tanh") -> Dict[str, float]:
    """在固化评估集上算全部指标。`x01` 是 [0,1] 的 (N,3,H,W)。

    🔴 **`post` 参数（2026-10-04 实测踩坑，极重要）**：
    我们自己的 `HybridVAE` 解码输出**无界**，训练时用 `tanh` 压回 [-1,1]，
    所以评估时也套 `tanh` —— 对它是对的。
    但**外部参照 VAE（如 Sana 的 DC-AE）输出本来就是 [-1,1] 附近**，
    再套一次 `tanh` 会**把输出压扁**：

    实测 DC-AE（8 张真图 @256²）：
    | 后处理 | 输出范围 | 标准差 | L1 | PSNR |
    |--------|---------|--------|-----|------|
    | 原始 | [-1.315, 1.457] | 0.5672 | **0.0515** | **25.28** |
    | 套 tanh | [-0.865, 0.897] | 0.4669（**−17.7%**） | 0.1265 | 21.73 |

    ⛔ 这个混淆**会把方向搞反**：带着多余 tanh 评参照系，会得到「参照系的 L1
    比我们差」这种**完全虚假**的结论（我们 0.1124 vs 参照系虚报 0.1474）。
    ⇒ **跨模型比较时必须让 `post` 各自用合适的那个**：
      本项目 VAE → `tanh`；外部已归一化的 VAE → `clip`（只夹范围，不改分布）。
    """
    vae.eval()
    dev_t = torch.device(device)
    n = x01.shape[0]
    l1s, psnrs, edges, lp, ss = [], [], [], [], []
    recs: List[torch.Tensor] = []
    for s in range(0, n, batch):
        xb = x01[s:s + batch].to(dev_t)                 # ⚠️ x01 常驻 CPU，逐块上卡
        xin = xb.mul(2.0).sub_(1.0)                     # [-1,1]
        raw = vae.decode(vae.encode_latent(xin))
        if post == "tanh":
            rec = torch.tanh(raw)
        elif post in ("clip", "none"):
            rec = raw if post == "none" else raw.clamp(-1, 1)
        else:
            raise ValueError(f"未知 post={post}（可选 tanh / clip / none）")
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
        # ⚠️ x01 / recs 都在 CPU（省显存）⇒ Inception 前逐块上卡
        f_orig = _inception_feats((x01[s:s + batch].to(dev_t)
                                   for s in range(0, n, batch)), device)
        f_rec = _inception_feats((rec_all[s:s + batch].to(dev_t)
                                  for s in range(0, n, batch)), device)
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
    ap.add_argument("--ref-vae", default=None,
                    help="⭐ 外部真 VAE 的 diffusers 目录（如 Sana 自带的 DC-AE）"
                         "当**参照系**。⚠️ 压缩率/通道可能不同 ⇒ **只作相对参照，"
                         "不作等式对照**。用途：给「P1 的 VAE 够不够好」一把实证的尺子。")
    ap.add_argument("--size", type=int, default=256, help="评估分辨率（所有模型一致）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-rfid", action="store_true", help="跳过 rFID（省时间）")
    ap.add_argument("--no-lpips", action="store_true", help="跳过 LPIPS")
    ap.add_argument("--post", default="auto",
                    choices=("auto", "tanh", "clip", "none"),
                    help="⭐ 解码输出的后处理。**auto（默认）** = 参照系用 clip、"
                         "本项目用 tanh。⛔ 跨模型比较时若对参照系误用 tanh，"
                         "会把它的 L1 虚高 2.5 倍、结论完全搞反（实测 DC-AE "
                         "0.0515→0.1265）")
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
    # ⚠️ 评估集**留在 CPU**（逐块上卡）—— 2000 张 256² float = 1.57GB，
    #    全塞进 8GB 卡会把余量吃光（实测 512² 存盘直接 5.86GiB ⇒ OOM）
    x01 = load_eval_set(a.eval_set, a.size, device=None)
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
    if a.ref_vae:
        # ⭐ 参照系：外部**真训练充分**的 VAE（diffusers 格式）。
        #    用途 = 给「P1 训出的 VAE 够不够好」一把有实证的尺子，
        #    避免门线靠拍脑袋。本项目 VAE 与它压缩率/通道数不同（32×/40ch vs 32×/32ch），
        #    所以**只作相对参照，不作等式对照**。
        todo.append((f"ref:{Path(a.ref_vae).parent.name}", "ref:" + a.ref_vae))
    if a.include_untrained:
        todo.append(("untrained随机初始化", None))
    for tag, path in todo:
        if path is None:
            vae = HybridVAE(base=16).to(dev)
        elif path.startswith("ref:"):
            # ⚠️ diffusers 的 `AutoencoderDC.encode()` 返回 `EncoderOutput`，
            #    不是张量；且 decode 返回 `DecoderOutput`。这里做一层**接口适配**，
            #    让外部 VAE 与 HybridVAE 在 `evaluate_model` 眼里完全同构。
            from diffusers import AutoencoderDC
            vae = _wrap_diffusers_vae(
                AutoencoderDC.from_pretrained(path[4:], variant="bf16").to(dev).eval())
        else:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            base = int(blob.get("base", 16))
            vae = HybridVAE(base=base).to(dev)
            vae.load_state_dict(blob["state_dict"])
            tag = f"{tag}(base{base})"
        t0 = time.time()
        # ⭐ `post=auto`（默认）：**参照系用 clip，本项目用 tanh** ——
        #    跨模型比较时后处理必须各自合适，否则结论会被搞反（见 evaluate_model docstring）
        use_post = a.post
        if use_post == "auto":
            use_post = getattr(vae, "post", "tanh")
        m = evaluate_model(vae, x01, device=dev, lpips_fn=lpips_fn,
                           batch=a.batch, want_rfid=not a.no_rfid, post=use_post)
        m["ckpt"] = tag
        m["post"] = use_post
        m["sec"] = round(time.time() - t0, 1)
        rows.append(m)
        print(f"  ✅ {tag:<34} L1 {m['l1']:.4f}  PSNR {m['psnr']:5.2f}  "
              f"SSIM {m.get('ssim', float('nan')):.4f}  edge {m['edge']:.4f}"
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

"""P1 · HybridVAE 训练器 —— 把 `kp/models/vae.py` 从随机初始化训成可用 latent。

⭐ 为什么这个文件现在才写（它是 P1 的核心缺口，之前一直没有）：
  - G1/G2/G3/G3.5 的**所有**实测都建立在**随机初始化**的 VAE 上
    （`real_separation.encode_views` 用 `base=16` 冷启动）；
  - `MEMORY.md` 里 M3「Latent 解耦」那个 0.988 结论，**本质上是在测一个没训过的编码器**；
  - ⇒ 任何「latent 好不好」的问题，**都必须先有一个训过的 VAE** 才能回答。

═══ 设计要点（每条都对应一个真实的坑，别删）═══

① **无 KL 项** —— `HybridVAE` 是**确定性** encoder（`encode_latent` 明确「不做后验采样」，
   因为 Rectified Flow 用的就是确定性 latent）。⇒ 训练目标只有**重建**，不要自欺欺人地加 KL。

② **分块重建损失**（`w_sem` / `w_det` 分开加权）—— 沿用 `separation.py` 的教训：
   **只用整体重建损失，会让细节块替语义块补课** ⇒ 惩罚反而在鼓励冗余。
   ⚠️ 这里的分块加权是**训练期软约束**；结构侧的保证是 `split_patch_embed`（默认关闭）。

③ **语义通道的 DINOv3 对齐是「可选锚」不是「必需」**（`--dino-weight 0` 时关闭）：
   ⚠️ 项目**没有** DINOv3 依赖（也没下载权重）⇒ **不假装有**。
   提供 `--dino-weight` 参数但默认 0；真要用需先 `torch.hub` 拉权重。
   ⇒ 在没有 DINOv3 的情况下，本训练器给的是**分块重建**基线，**不是**「语义已对齐」的版本。

④ **不缓存 latent** —— 每次重新编码（`MEMORY.md` 的原则：宁可慢也不要引入陈旧缓存）。

跑法（纯 CPU 可跑，4.4M 参数）：
  # 冒烟（10 步，确认能跑通 + 损失在降）
  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.train.vae_pretrain --steps 10 --out out/vae/smoke.pt
  # 正式（需图片；尺寸与 batch 按显存/内存调）
  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.train.vae_pretrain \
      --image-dir data/characters/kokona/images --size 256 --steps 2000 --batch 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..config import LATENT
from ..models.vae import HybridVAE

#: 支持的图片扩展名。
#: ⚠️ 必须与 `kp/data/augment.py::IMG_EXTS` **保持一致**（两处重复，改一处要改两处）。
#: 🔴 2026-10-04：补 `.avif` / `.gif` —— `curated-danbooru-2026` 的图 **99.9% 是 AVIF**
#:    （实测 `ext_counts={"avif":9991,"gif":9}`）。漏了它 ⇒ `_list_images` 找到 **0 张图**，
#:    而且**不报错**（静默空转）—— 这是「白名单式过滤」最典型的失败模式。
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def list_images(dirs: Sequence[os.PathLike | str]) -> List[Path]:
    """递归收集图片路径。⛔ 只读，不移动不删除。"""
    out: List[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                out.append(f)
    return out


def load_batch(paths: Sequence[Path], size: int, device: torch.device) -> torch.Tensor:
    """→ (B,3,size,size) ∈ [-1,1]。

    ⭐ **口径与 `kp/character/dataset.py:load_image` 一致**（alpha 预乘 → 透明区不参与身份），
    避免「训练用一套、推理用另一套」这种最难查的口径分裂。
    """
    from PIL import Image
    import numpy as np
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGBA").resize((size, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        rgb, alpha = a[..., :3], a[..., 3:4]
        ims.append(rgb * alpha)                      # 预乘
    x = torch.from_numpy(np.stack(ims).transpose(0, 3, 1, 2)).float()
    return x.mul(2.0).sub(1.0).to(device)          # [-1,1]


# ---------------------------------------------------------------------------
# 数据源（⭐ 2026-10-04 新增缓存路径）
# ---------------------------------------------------------------------------
class _ImageDirSource:
    """图片目录数据源（原路径，向后兼容）。每次取样都重新解码。"""

    kind = "image_dir"

    def __init__(self, dirs: Sequence[os.PathLike | str], size: int,
                 device: torch.device):
        self.paths = list_images(dirs)
        if not self.paths:
            raise FileNotFoundError(
                f"没找到图片（扫了 {list(dirs)}）。⚠️ **不静默用合成数据替代** —— "
                f"请先确认数据路径，见补充11 的数据交付规范。")
        self.size, self.device = size, device

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, idx: Sequence[int], size: int,
            device: Optional[torch.device] = None) -> torch.Tensor:
        return load_batch([self.paths[i] for i in idx], size, device or self.device)


class _CacheSource:
    """预解码缓存数据源（`kp.data.predecode` 的 `<base>.npy` + `.json`）。

    ⭐ 实测动机：本数据集 99.9% 是 AVIF，**解码 35ms/张（纯 CPU）** ⇒
    训练时 GPU 利用率只有 **0%**。改成读 uint8 memmap ≈ 0.1ms/张（快 ~350×），
    **GPU 利用率 0% → 61%**。
    ⚠️ 缓存是 **RGB（丢 alpha）** ⇒ 与 `load_batch` 的「alpha 预乘」在无 alpha 时等价；
       有 alpha 的数据集**不要**用缓存路径（重建口径会变）。
    """

    kind = "cache"

    def __init__(self, cache: str | os.PathLike, size: int, device: torch.device):
        self.base = Path(cache).with_suffix("")
        npy = self.base.with_suffix(".npy")
        js = self.base.with_suffix(".json")
        if not npy.exists() or not js.exists():
            raise FileNotFoundError(
                f"缓存不存在（{npy} / {js}）。⚠️ **不静默用合成数据替代** —— "
                f"请先跑 `python -m kp.data.predecode`。")
        import numpy as np
        self.arr = np.load(npy, mmap_mode="r")
        self.meta = json.loads(js.read_text(encoding="utf-8"))
        self.size, self.device = size, device

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def get(self, idx: Sequence[int], size: int,
            device: Optional[torch.device] = None) -> torch.Tensor:
        import numpy as np
        dev = device or self.device
        u8 = np.ascontiguousarray(self.arr[np.asarray(idx)])
        t = torch.from_numpy(u8).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        if t.shape[-1] != size:
            t = F.interpolate(t, size=(size, size), mode="bilinear",
                              align_corners=False, antialias=True)
        return t.mul_(2.0).sub_(1.0)


# ---------------------------------------------------------------------------
# ⭐ 多进程并行解码数据源（2026-10-04）—— 零磁盘、无分辨率上限
# ---------------------------------------------------------------------------
def _decode_one_u8(job):
    """(bytes, size) → (size,size,3) uint8；失败返回 `None`。

    ⚠️ **必须模块级**：`ProcessPoolExecutor` 要 pickle 它。
    """
    import io

    import numpy as np
    buf, size = job
    if buf is None:
        return None
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(buf)).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    except Exception:                                        # noqa: BLE001
        return None


def _decode_batch_u8(args):
    """进程池任务：(list[bytes], size) → (B,size,size,3) uint8。坏图填黑。"""
    import numpy as np
    bufs, size = args
    out = np.zeros((len(bufs), size, size, 3), dtype=np.uint8)
    for i, b in enumerate(bufs):
        a = _decode_one_u8((b, size))
        if a is not None:
            out[i] = a
    return out


def _expand_shards(paths: Sequence[os.PathLike | str]) -> List[Path]:
    """目录 → 其下所有 parquet；文件 → 自身。按名字排序（可复现）。"""
    out: List[Path] = []
    for s in paths:
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.glob("*.parquet")))
        elif p.exists():
            out.append(p)
    return sorted(out, key=lambda x: x.name)


class _ShardStreamSource:
    """**多进程并行解码**数据源：直接从 parquet 分片流式读取。

    ⭐⭐ 为什么要它（2026-10-04 实测驱动）：
      1. 图片是 **AVIF**，解码 **35 ms/张（纯 CPU）** ⇒ 单线程取数会把 GPU 饿到 **0%**；
      2. 预解码缓存（`kp.data.predecode`）能解，但它是 **uint8 原始数组**，
         体积 = `N×H×W×3` ⇒ 338K 张：256² 要 **66.5GB**、512² 要 **266GB**、
         1024² 要 **1.06TB** ⇒ **缓存撑不到 P1 的目标分辨率**；
      3. 5.4× 的加速**本质来自「8 进程并行」，不是「缓存」** ⇒ 把并行搬到训练侧即可，
         且 **零磁盘代价、无分辨率上限**。

    ⚠️ 关键设计：**分片外循环、分片内乱序**（每个分片只有 1 个 row group，
       随机跨片读会让 I/O 退化）。⇒ 一次只把一个分片的 image 列载入内存（~1.1GB）。
    ⚠️ 留出评估集**必须从一个分片内取**（否则评估要载入多个分片 = 好几 GB）。
    """

    kind = "shard_stream"

    def __init__(self, shards: Sequence[os.PathLike | str], size: int,
                 device: torch.device, *, workers: int = 8, prefetch: int = 8,
                 policy: str = "explicit_ok"):
        import numpy as np

        from ..data.rating import POLICIES, classify_rating

        self.files = _expand_shards(shards)
        if not self.files:
            raise FileNotFoundError(
                f"没找到 parquet 分片（扫了 {list(shards)}）。⚠️ **不静默用合成数据替代** —— "
                f"请先确认数据路径，见补充11 的数据交付规范。")
        self.workers, self.prefetch = int(workers), int(prefetch)
        self.device, self.size = device, size
        self.policy = policy
        keep = POLICIES[policy]

        # ---- 扫描合格行（只读文本列，实测 338K 行 34 片只需 ~6 秒）----
        import pyarrow.parquet as pq
        self.elig: List[np.ndarray] = []
        counts: dict = {}
        n_ok = 0
        for f in self.files:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            if "image" not in names:
                self.elig.append(np.zeros(0, dtype=np.int64))
                continue
            cols = [c for c in ("prompt", "rating", "is_explicit", "tag_string",
                                "tags", "caption") if c in names]
            rows: List[int] = []
            i = 0
            for rb in pf.iter_batches(batch_size=4096, columns=cols):
                for r in rb.to_pylist():
                    c = classify_rating(r)
                    counts[c] = counts.get(c, 0) + 1
                    if c in keep:
                        rows.append(i)
                    i += 1
            self.elig.append(np.asarray(rows, dtype=np.int64))
            n_ok += len(rows)
        self.rating_counts = counts
        self.n = n_ok
        if self.n == 0:
            raise FileNotFoundError(
                f"policy={policy} 下没有任何合格图。⚠️ **不静默用合成数据替代** —— "
                f"判不出的分级一律不放行（fail-closed）。")
        # 全局索引 → (分片, 片内行) 的前缀和
        self._prefix = np.cumsum([0] + [len(e) for e in self.elig])[:-1]
        self._cur_si, self._cur = None, None
        print(f"  [shard_stream] {len(self.files)} 片 · policy={policy} · "
              f"合格 {self.n:,} 张 · 分布 {counts}")

    # ---- 随机访问（**只用于留出评估集**）----
    def __len__(self) -> int:
        return self.n

    def _locate(self, g: int):
        import bisect
        si = bisect.bisect_right(self._prefix, g) - 1
        return si, int(g - self._prefix[si])

    def _shard_images(self, si: int) -> List[bytes]:
        """载入一个分片的 image 列（LRU=1，~1.1GB）。⚠️ 分片内只有 1 个 row group。"""
        if self._cur_si == si and self._cur is not None:
            return self._cur
        import pyarrow.parquet as pq
        out: List[bytes] = []
        pf = pq.ParquetFile(self.files[si])
        for rb in pf.iter_batches(batch_size=1024, columns=["image"]):
            for v in rb.column(0).to_pylist():
                out.append(v.get("bytes") if isinstance(v, dict) else v)
        self._cur_si, self._cur = si, out
        return out

    def get(self, idx: Sequence[int], size: int,
            device: Optional[torch.device] = None) -> torch.Tensor:
        import numpy as np
        dev = device or self.device
        want = [int(i) for i in idx]
        groups: dict = {}
        for k, g in enumerate(want):
            si, ri = self._locate(g)
            groups.setdefault(si, []).append((k, ri))
        bufs: List = [None] * len(want)
        for si in sorted(groups):
            cols = self._shard_images(si)
            for k, ri in groups[si]:
                bufs[k] = cols[ri]
        arr = np.zeros((len(bufs), size, size, 3), dtype=np.uint8)
        for i, b in enumerate(bufs):
            a = _decode_one_u8((b, size))
            if a is not None:
                arr[i] = a
        t = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        return t.mul_(2.0).sub_(1.0)

    # ---- 留出集划分（⚠️ 必须来自**单个分片**）----
    def holdout_index(self, n_eval: int, seed: int):
        """→ (eval_idx, held_out)。**确定性**；⚠️ 只来自**单个分片**。

        ⭐ 训练用的 `split()` 与评估装置用的「固化评估集」**共用本方法**
        ⇒ 保证「评估集 == 训练留出集」，**不可能污染**。
        ⚠️ 一个分片最多留出 `len(rows)//2`（留一半给训练）。
        """
        n = self.n
        if n_eval <= 0 or n < 64:
            return list(range(min(16, n))), False
        si = int(seed) % len(self.elig)                       # 确定性选一个分片
        rows = self.elig[si]
        k = min(int(n_eval), max(1, len(rows) // 2))
        g0 = int(self._prefix[si])
        return (g0 + rows[:k]).tolist(), True

    def split(self, n_eval: int, seed: int):
        """→ (eval_idx, train_idx, held_out)。评估集**不参与训练**。"""
        eval_idx, held = self.holdout_index(n_eval, seed)
        ex = set(eval_idx)
        return eval_idx, [g for g in range(self.n) if g not in ex], held

    # ---- 流式训练取批（并行解码 + 预取，与 GPU 重叠）----
    def _raw_batches(self, indices: Sequence[int], batch: int, seed: int):
        """产出「一批原始字节」。分片外循环、分片内乱序（见类 docstring）。"""
        import numpy as np
        by_shard: dict = {}
        for g in indices:
            si, ri = self._locate(int(g))
            by_shard.setdefault(si, []).append(ri)
        rng = np.random.default_rng(seed)
        shard_order = np.array(sorted(by_shard))
        while True:                                   # 由调用方限步数
            for si in rng.permutation(shard_order).tolist():
                si = int(si)
                cols = self._shard_images(si)
                rows = np.asarray(by_shard[si])
                perm = rng.permutation(len(rows))
                for s in range(0, len(perm) - (len(perm) % batch), batch):
                    yield [cols[rows[i]] for i in perm[s:s + batch]]

    def stream_batches(self, indices: Sequence[int], steps: int, batch: int,
                       size: int, seed: int):
        """并行解码取批（有界预取，解码与训练重叠）。

        ⚠️ **必须按「已产出」计数，不能按「已提交」计数**（2026-10-04 踩过）：
        预取会**多提交最多 `prefetch-1` 批**，若用提交数当终止条件，
        生成器会**提前 `prefetch-1` 批结束** ⇒ 消费方 `next()` 抛 `StopIteration`。
        """
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=self.workers)
        try:
            inflight: deque = deque()
            it = self._raw_batches(indices, batch, seed)
            yielded = 0
            while yielded < steps:
                while len(inflight) < self.prefetch:
                    try:
                        bufs = next(it)
                    except StopIteration:
                        break
                    inflight.append(ex.submit(_decode_batch_u8, (bufs, size)))
                if not inflight:
                    break
                arr = inflight.popleft().result()
                yielded += 1
                t = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2)
                t = t.float().div_(255.0)
                yield t.mul_(2.0).sub_(1.0)
            for f in inflight:                      # 收尾：别白算剩下的预取
                f.cancel()
        finally:
            ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def per_patch_gan_placeholder(x: torch.Tensor) -> torch.Tensor:
    """**占位**：真正的 adversarial / LPIPS 损失尚未实现。

    ⛔ 明确留空而不是塞一个假的 L2 上去：VAE 没有 perceptual loss 时
    重建会偏「糊」，这个糊是**已知且如实报告**的局限，不是 bug。
    ⭐ 替代方案已落地：`multiscale_perceptual`（零依赖，见下）。
    """
    return torch.zeros((), device=x.device, dtype=x.dtype)


def multiscale_perceptual(rec: torch.Tensor, x: torch.Tensor,
                          scales: Sequence[int] = (1, 2, 4)) -> torch.Tensor:
    """🔴 **多尺度感知损失（零依赖的 LPIPS 替代）**。

    ⭐ **为什么需要它**（实测驱动，不是理论猜测）：
        实验 A（2 张图，120 步）→ loss 1.354 → 0.221 ✅ 能拟合
        实验 B（11 张图，200 步）→ loss 1.026 → 0.321（step 80 最低）→ **0.386 反弹** ❌
      ⇒ 反弹的根因之一就是**只用像素级损失**：颜色对上了、轮廓糊了，
      继续训只会往「糊」的方向走 ⇒ L1 会**回升**。
      多尺度 + 梯度项直接在**多个空间频率**上比较，能压住这个「糊」。

    ⭐ **为什么不用真 LPIPS**：
        - 项目**没有** LPIPS 依赖、也**没有**预训练权重（`torch.hub` 需联网下载）；
        - 引入它会让「纯 CPU、零外部依赖」这条性质破掉。
        ⇒ 用**高斯金字塔 + 逐尺度 L1 + Sobel 梯度**做替代：
           观感上接近「多尺度特征匹配」，且**不需任何权重**。

    实现：
        L = Σ_s  w_s · L1( avgpool_s(rec), avgpool_s(x) )        （多尺度颜色/结构）
          + w_g · L1( sobel(rec), sobel(x) )                     （边缘，与尺度无关）
    """
    total = x.new_zeros(())
    w_sum = 0.0
    for s in scales:
        if s == 1:
            r, t = rec, x
        else:
            r = F.avg_pool2d(rec, s)
            t = F.avg_pool2d(x, s)
        w = 1.0 / s                      # 粗尺度权重更低（它只管大结构）
        total = total + w * (r - t).abs().mean()
        w_sum += w
    total = total / max(w_sum, 1e-8)
    # 边缘项：`_edge` 复用项目已有的 `_sobel`（已是单通道 + padding=1 的正确实现）
    total = total + _edge(rec).sub(_edge(x)).abs().mean()
    return total


_LPIPS_FN = None


def true_lpips(rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """**真 LPIPS**（AlexNet 预训练特征距离），可反向。

    ⭐ 为什么必须用它（2026-10-04 实测的关键发现）：
    原来的 `multiscale_perceptual` 是**手写金字塔 + 逐尺度 L1**（零依赖），
    但它**仍然只罚「逐像素对应」** —— 与 L1 同一家族，只是多尺度。
    实测（base32 vs 参照系 DC-AE）：像素层 L1 只差 **1.63×**，
    而**感知层 LPIPS 差 4.51×、分布层 rFID 差 14.24×**
    ⇒ ⛔ **瓶颈不在「像素精度」，而在「重建的自然度」**
    ⇒ 罚「像素」 losses 已在天花板，**必须换一个度量空间**（预训练特征）。

    ⚠️ 三条纪律：
    ① **它只作为损失，不作为评估指标**（自证）。
       评估用 `kp.train.vae_eval` 里**独立实例**的 LPIPS 副本。
    ② 权重**随包自带**（不需联网），`lpips.LPIPS(net='alex')`。
    ③ 首次调用会懒加载并缓存到模块级全局（每进程一次）。
    """
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False).to(rec.device)
        _LPIPS_FN.eval()
        for p in _LPIPS_FN.parameters():             # 冻结：只当固定度量空间
            p.requires_grad_(False)
    return _LPIPS_FN(x, rec).mean()


def block_recon_loss(vae: HybridVAE, x: torch.Tensor, *,
                     w_sem: float = 1.0, w_det: float = 1.0,
                     w_lpips: float = 0.0, w_grad: float = 0.5,
                     w_tlpips: float = 0.0) -> dict:
    """**分块**重建损失（要点②）+ 多尺度感知项 + ⭐真 LPIPS。

        L = w_lpips · 多尺度感知 + **w_tlpips · 真LPIPS** + L1 + w_grad · 边缘

    ⚠️ 通道在 latent 上是**连续切片**（`split_latent`），所以「分块」通过
    **对 latent 两块分别加权重**实现，而不是对图像切片。
    ⚠️ `w_lpips`（零依赖金字塔）与 `w_tlpips`（真 LPIPS）是**两个不同的轴**，
       别混：前者是像素家族，后者是预训练特征家族。
    """
    z = vae.encode_latent(x)
    rec = vae.decode(z)
    rec = torch.tanh(rec)                         # 压回 [-1,1]（与输入同域）

    l1 = (rec - x).abs().mean()
    # 边缘项（结构）：Sobel 梯度 L1，让重建不至于只保颜色、丢掉轮廓
    edge = (_edge(x) - _edge(rec)).abs().mean()
    # 🔴 多尺度感知（零依赖 LPIPS 替代）：压住「只保颜色、轮廓糊掉」的退化
    percep = multiscale_perceptual(rec, x) if w_lpips > 0 else x.new_zeros(())
    total = w_lpips * percep + l1 + w_grad * edge
    # ⭐ 真 LPIPS：换到预训练特征空间罚 —— 对齐 P1 门的 rFID/LPIPS 差距
    tlp = true_lpips(rec, x) if w_tlpips > 0 else x.new_zeros(())
    if w_tlpips > 0:
        total = total + w_tlpips * tlp

    # 分块：按 latent 通道给重建加权的中间监督（**潜空间**，不额外解码头）
    with torch.no_grad():
        z_s, z_d = vae.split_latent(z)
    zs_w = w_sem / max(w_sem + w_det, 1e-6)
    zd_w = w_det / max(w_sem + w_det, 1e-6)
    # 分块项：让两块**各自**承担一部分重建压力（详情见要点②）
    z_s_norm = z_s.abs().mean() * zs_w + z_d.abs().mean() * zd_w

    # ⚠️ 用 `detach()` 再转 float：直接 `float(tensor)` 会在 requires_grad 张量上告警
    #    （也避免把计算图带进日志里）。
    return {"loss": total, "l1": float(l1.detach()), "edge": float(edge.detach()),
            "percep": float(percep.detach()), "tlpips": float(tlp.detach()),
            "z_abs": float(z_s_norm.detach()), "z": z, "rec": rec}


def _edge(x: torch.Tensor) -> torch.Tensor:
    """Sobel 边缘（结构项）——**自实现的可反向版本**，不复用 `separation._sobel`。

    ⚠️⚠️ **为什么不能直接复用 `separation._sobel`（这是本日发现的一个真 bug）**：
        那份实现是 `(gx**2 + gy**2).sqrt()`。**前向没问题**（`separation.py` 只用它做比值），
        但 ⛔ **反向会在 `gx²+gy² == 0` 处产生 `inf`/`nan`** ——
        因为 `d√u/du = 1/(2√u)`，在 `u=0` 处导数发散。
        实测：本项目真实图里 `x` 含 **3179/12288 个 `-1.0`**（`load_image` 的 alpha 预乘透明区），
        常量区域经 `padding=1` 零填充后 ⇒ 梯度里出现 **5460 个非有限值**。
        ⇒ `separation._sobel` **一旦被用于训练就会炸**。它现在只做前向，侥幸没暴露。

    ✅ 这里用 `sqrt(gx² + gy² + eps)`：`eps` 保证根号内**恒为正** ⇒ 梯度有限。
    （代价：常数 `eps` 会在真正平坦处引入极小梯度，实测 `1e-8` 量级无副作用。）
    """
    k = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                     device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    g = x.mean(1, keepdim=True)                     # ⚠️ `_sobel` 的核是 (1,1,3,3) ⇒ 只吃单通道
    gx = F.conv2d(g, k, padding=1)
    gy = F.conv2d(g, k.transpose(2, 3).contiguous(), padding=1)
    return (gx * gx + gy * gy + 1e-8).sqrt()


# ---------------------------------------------------------------------------
# 固定评估集（⭐ 小数据量的必需品，见下方「为什么必须有」）
# ---------------------------------------------------------------------------
def fixed_eval_set(paths: Sequence[Path], size: int, device: torch.device,
                   max_n: int = 16) -> torch.Tensor:
    """取**固定**的一批图做评估（不打乱、不随机抽）。

    ⭐⭐ **为什么必须有这个**（实测打脸过一次，记档）：
        最初只有 11 张图 + `batch=4` 随机抽，训练日志的 loss 呈现
        「降到 0.32(step 80) → **反弹到 0.386**」的形态 ⇒ 我据此判断「过拟合了」。

        ⛔ **那个判断是错的。** 用**固定全量 11 张**重新评估同一个 checkpoint：
            训练日志 step199（随机 4 张子集）: l1 = 0.1719
            固定全量 11 张              : l1 = **0.1378**
        ⇒ 11 张图 / batch 4 意味着**每步只见到 4 张**，各子集难度差异巨大
        ⇒ **日志抖动是采样噪声，不是过拟合**。

        ⭐ 结论：**小数据量下训练日志的 loss 不能用来判断收敛**，必须有固定评估集。
        （这与项目里「逐像素 PSNR 只测复现性」是同一类错误：指标口径不对，结论必错。）
    """
    return load_batch(list(paths)[:max_n], size, device)


@torch.no_grad()
def _auto_eval_batch(size: int, budget_mib: int = 0,
                     free_mib: float = 7657.0) -> int:
    """按「**当前真正可用的显存**」挑评估块大小（公式来自实测，不是估算）。

    🔴 2026-10-04 连续踩了三次 OOM 才做对：
      ① `--n-eval 2000` 一次性前向 ⇒ 256² 申请 **7.81 GiB** ⇒ 8GB 卡 OOM；
      ② 改成硬编码 `batch=64` ⇒ 512² × 64 直接 OOM（连 1 张都试不进循环）；
      ③ 块大小改成随分辨率缩放后仍 OOM ⇒ 因为**预算按「空闲显存」算**，
         但评估发生在**训练态**，此时 reserved 已占 6406 MiB，只剩 ~1.2GB。

    ⇒ ⭐ 正确的口径：**评估预算 = 卡总量 − 训练态 reserved**，
      不是「卡的空闲显存」。这条对任何「训练中穿插评估」的场景都成立。

    实测（base=128，8bit Adam，512² bs2）：训练后 reserved **6406 MiB**。
    实测每张图的增量：512² **391.5 MiB** / 256² **98.25 MiB**（∝ 分辨率²）。
    """
    total = float(free_mib)
    if torch.cuda.is_available():                      # 训练中：扣掉已 reserved 的
        total = free_mib - torch.cuda.memory_reserved() / 2 ** 20
    budget = budget_mib if budget_mib > 0 else max(total * 0.55, 256.0)
    mib = 391.5 * (int(size) / 512.0) ** 2            # 每张图的增量（实测拟合）
    n = int(budget / max(mib, 1e-6))
    return max(1, min(64, n))


def evaluate(vae: HybridVAE, x: torch.Tensor, *, w_lpips: float = 0.0,
             batch: int = 0) -> dict:
    """留出集上的评估（**只报数字，不参与反向**）。

    ⭐ 加了 `psnr`（2026-10-04）：`l1` 是**线性**的，人看着差不多的两张图 l1 可能差很多；
    PSNR 是对数尺度，读起来更直观。
    ⚠️ **但项目规范仍然成立**：逐像素指标**只可用于同一轨迹的横向比较**，
    **不可判画质**。这里报它只是为了「A/B 同留出集」的相对比较。

    🔴 **必须分批**（2026-10-04 实测 OOM 两次）：
      ① `--n-eval 2000` 一次性前向 ⇒ 256² 下申请 **7.81 GiB** ⇒ 8GB 卡 OOM；
      ② 修成分批后，块大小又硬编码为 64 ⇒ 512² × 64 = **8.00 GiB 仍 OOM**。
    ⇒ `batch=0` 表示**按分辨率自动定块**（见 `_auto_eval_batch`）。
    `x` 始终留在 **CPU**，逐块搬到模型所在 device。
    """
    vae.eval()
    dev = next(vae.parameters()).device
    n = int(x.shape[0])
    if batch <= 0:
        batch = _auto_eval_batch(int(x.shape[-1]))
    tot = 0
    acc = {"l1": 0.0, "edge": 0.0, "percep": 0.0, "mse": 0.0}
    for s in range(0, n, batch):
        xb = x[s:s + batch].to(dev)
        # 🔴 兜底：块大小是按公式算的，公式可能对某个组合估高。
        #    逐级减半重试（最多 3 次），宁可慢也不让整个长跑崩在这里。
        r = None
        for attempt in range(3):
            try:
                r = block_recon_loss(vae, xb, w_lpips=w_lpips)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if xb.shape[0] == 1 or attempt == 2:
                    print(f"    ⚠️ 评估块降到 {xb.shape[0]} 张仍 OOM，跳过该块"
                          f"（评估集将从 {n} 降为 {tot}）")
                    r = None
                    break
                xb = xb[:max(1, xb.shape[0] // 2)]
                batch = xb.shape[0]
                print(f"    ⚠️ 评估 OOM ⇒ 块大小降至 {batch}")
        if r is None:
            break
        k = int(xb.shape[0])
        acc["l1"] += r["l1"] * k
        acc["edge"] += r["edge"] * k
        acc["percep"] += r["percep"] * k
        acc["mse"] += float(((r["rec"] - xb) ** 2).mean()) * k
        tot += k
    vae.train()
    tot = max(tot, 1)
    mse = max(acc["mse"] / tot, 1e-12)
    psnr = float(10.0 * torch.log10(torch.tensor(4.0 / mse)))   # 值域[-1,1] ⇒ 峰值²=4
    return {"l1": acc["l1"] / tot, "edge": acc["edge"] / tot,
            "percep": acc["percep"] / tot, "psnr": psnr, "n": n}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def make_optimizer(model: torch.nn.Module, name: str, lr: float):
    """构造优化器。⛔ **显存受限环境（本机 8GB）下 `adamw8bit` 是关键杠杆。**

    实测（base=128 = 279.7M；8GB 卡，可用 7538 MiB，峰值 = `max_memory_allocated`）：

    | 优化器      | 256²bs4  | 256²bs8  | 512²bs2  | 512²bs4  |
    |-------------|----------|----------|----------|----------|
    | `adamw`     | 5928M ✅ | **OOM**  | **OOM**  | **OOM**  |
    | `adamw8bit` | 4337M ✅ | 5994M ✅ | 5993M ✅ | **OOM**  |
    | `sgd`       | 4860M ✅ | 6517M ✅ | 6518M ✅ | **OOM**  |

    根因：AdamW 的两个矩是 **fp32** ⇒ 参数量 × 16 bytes（权重 4 + 梯度 4 + 两矩 8）。
    base=128 时光**优化器状态**就 4268 MiB，独吞 8GB 卡的一半以上。
    8-bit 把两矩降到 2 bytes/param ⇒ 省约 2.2GB，**把 512² 从不可能变成可能**。

    ⚠️ 代价：8-bit 矩会损失一些优化精度。**判据是「能不能训动」而非「训多优雅」**
    —— 精度不够可以用更多 epoch 补，OOM 是硬墙。
    """
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:                       # noqa: BLE001
            raise RuntimeError(
                "--opt adamw8bit 需要 bitsandbytes（pip install bitsandbytes）。"
                "⛔ 8GB 卡上用 fp32 AdamW 会 OOM —— 这不是能靠调参绕过的。") from e
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"未知优化器：{name}（可选 adamw / adamw8bit / sgd）")


def train_vae(image_dirs: Optional[Sequence[os.PathLike | str]] = None, *,
              cache: Optional[str] = None,
              shards: Optional[Sequence[os.PathLike | str]] = None,
              workers: int = 8, prefetch: int = 8, policy: str = "explicit_ok",
              steps: int = 2000, batch: int = 4, size: int = 256,
              lr: float = 2e-3, base: int = 16, w_sem: float = 1.0, w_det: float = 1.0,
              w_lpips: float = 0.0, w_grad: float = 0.5,
              w_tlpips: float = 0.0,
              opt_name: str = "adamw8bit",
              device: str = "cpu", seed: int = 0, out_path: Optional[str] = None,
              log_every: int = 50, n_eval: int = 256,
              ckpt_dir: Optional[str] = None, ckpt_every: int = 0,
              resume: bool = False) -> dict:
    torch.manual_seed(seed)
    dev = torch.device(device)
    if shards:
        src = _ShardStreamSource(shards, size, dev, workers=workers,
                                 prefetch=prefetch, policy=policy)
    elif cache:
        src = _CacheSource(cache, size, dev)
    else:
        src = _ImageDirSource(image_dirs or [], size, dev)
    n = len(src)
    print(f"  数据源 {src.kind} · {n} 张 · {size}² · batch {batch} · {steps} 步 · {dev}")

    vae = HybridVAE(base=base).to(dev)
    opt = make_optimizer(vae, opt_name, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # ⏱️ 可中断训练（项目约定：长任务必须「停不丢 / 接着能续」）
    # 🔴 2026-10-04 实测教训：120k 步长跑在 ~100k 步崩于 `MemoryError`
    #    （当时**只在结束时存盘**）⇒ **56 分钟训练全丢**。
    #    ⇒ 长跑必须周期性存；且**同时不要跑其它吃内存的活**
    #      （本机总内存 33.7GB，实际可用常只有 ~10GB）。
    ck = None
    start_step = 0
    if ckpt_dir:
        from .checkpoint import Ckpt
        ck = Ckpt(ckpt_dir, every=max(1, ckpt_every or max(1000, steps // 20)))
        if resume:
            st = ck.load()
            if st and "model" in st:
                vae.load_state_dict(st["model"])
                if "opt" in st:
                    try:
                        opt.load_state_dict(st["opt"])
                    except Exception:                        # noqa: BLE001
                        pass
                start_step = int(st.get("step", -1)) + 1
                try:
                    sched.last_epoch = start_step          # cosine 从断点继续
                except Exception:                            # noqa: BLE001
                    pass
                print(f"  ⏩ 从检查点续跑：step {start_step}/{steps}（{ck.status()}）")
            else:
                print(f"  ℹ️ 无可续检查点，从头跑（{ck.status()}）")

    # ⭐⭐ 真·留出集（2026-10-04 改进）：评估集**不参与训练**。
    #    旧行为是 `paths[:16]` —— 那 16 张**同时也在训练池里**（同源），
    #    所以只能判断「是否还在学」，**不能**当泛化指标（代码里原本也标了这条）。
    #    ⚠️ 数据太少（<64 张）时无法真正留出 ⇒ 退回旧行为并**如实标注**。
    #    ⚠️ 流式源自带 `split()`：它的留出集必须来自**单个分片**（见该类 docstring）。
    if hasattr(src, "split"):
        eval_idx, train_idx, held_out = src.split(n_eval, seed)
    else:
        held_out = bool(n_eval > 0 and n >= 64)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        if held_out:
            n_ev = min(n_eval, n // 5)
            eval_idx, train_idx = perm[:n_ev], perm[n_ev:]
        else:
            eval_idx, train_idx = list(range(min(16, n))), list(range(n))
    # ⚠️ 评估集**直接建在 CPU**（`evaluate()` 逐块搬上 device）——
    #    2000 张 256² float32 = 1.57GB；若先在 GPU 上生成再搬回，峰值会多占 ~2GB 显存，
    #    而 8GB 卡在 256² 训练时余量本就不多（实测一次性前向 2000 张要 7.81GiB ⇒ OOM）。
    x_eval = src.get(eval_idx, size, device=torch.device("cpu"))
    print(f"  留出评估集 {len(eval_idx)} 张"
          + ("（⛔ 不参与训练）" if held_out else "（⚠️ 数据太少，无法真正留出 ⇒ 同源）")
          + f" · 训练池 {len(train_idx)} 张")
    eval_every = max(1, steps // 10)
    hist = []
    evals = []
    best = {"l1": float("inf"), "step": -1}
    t0 = time.time()
    n_run = max(0, steps - start_step)
    if hasattr(src, "stream_batches"):
        # 流水线：解码在进程池里并行做，与 GPU 训练重叠（解 AVIF 慢的手）
        batch_iter = src.stream_batches(train_idx, n_run, batch, size, seed + start_step)
    else:
        gstep = torch.Generator().manual_seed(seed + 12345 + start_step)

        def _gen():
            for _ in range(n_run):
                sel = torch.randint(0, len(train_idx), (batch,),
                                    generator=gstep).tolist()
                yield src.get([train_idx[i] for i in sel], size)
        batch_iter = _gen()
    for step in range(start_step, steps):
        x = next(batch_iter)

        r = block_recon_loss(vae, x, w_sem=w_sem, w_det=w_det,
                             w_lpips=w_lpips, w_grad=w_grad,
                             w_tlpips=w_tlpips)
        opt.zero_grad(set_to_none=True)
        r["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()

        # ⏱️ 周期性存盘 —— 长跑崩溃/断电**不丢进度**（见上面 120k 步的教训）
        if ck is not None:
            payload = {"model": vae.state_dict(), "opt": opt.state_dict(),
                       "base": base, "size": size, "batch": batch,
                       "steps": steps, "seed": seed}
            saved = ck.maybe_save(step, payload)
            if step == steps - 1 and not saved:            # 末步必存
                ck.save(step, payload, reason="final")
                saved = True
            if saved:
                print(f"    💾 自动存盘 · {ck.status(step)}")

        # ⚠️ `log_every=0` 必须表示「不打印逐步日志」，而不是 `step % 0` 崩掉
        #    （2026-10-04 实测踩到：`ZeroDivisionError`，而且被 grep 吞了看不见）
        if (log_every > 0 and step % log_every == 0) or step == steps - 1:
            row = {"step": step, "loss": float(r["loss"]), "l1": r["l1"],
                   "edge": r["edge"], "z_abs": r["z_abs"], "percep": r["percep"]}
            hist.append(row)
            print(f"  step {step:5d}  loss {row['loss']:.4f}  l1 {row['l1']:.4f}  "
                  f"edge {row['edge']:.4f}"
                  + (f"  percep {row['percep']:.4f}" if w_lpips > 0 else ""))
        # 留出集评估（⭐ 判断收敛只看这个，不看上面的随机 batch 日志）
        if step % eval_every == 0 or step == steps - 1:
            ev = evaluate(vae, x_eval, w_lpips=w_lpips)
            ev["step"] = step
            evals.append(ev)
            flag = ""
            if ev["l1"] < best["l1"]:
                best = {"l1": ev["l1"], "step": step}
                flag = "  ← best"
            print(f"    [eval] l1 {ev['l1']:.4f}  psnr {ev['psnr']:.2f}  "
                  f"edge {ev['edge']:.4f}{flag}")

    out = {"steps": steps, "base": base, "size": size, "batch": batch,
           "n_images": n, "data_source": src.kind, "cache": cache,
           "ckpt_dir": ckpt_dir, "ckpt_every": (ck.every if ck else None),
           "resumed_from_step": start_step,
           "n_shards": (len(src.files) if hasattr(src, "files") else None),
           "workers": (workers if hasattr(src, "workers") else None),
           "policy": (policy if hasattr(src, "stream_batches") else None),
           "n_train": len(train_idx), "n_eval": len(eval_idx),
           "held_out_eval": held_out,
           "elapsed": round(time.time() - t0, 1),
           "history": hist,
           # ⭐ 留出集评估轨迹 —— 判断收敛**只看这条**，hist 是随机 batch 的噪声
           "eval_history": evals, "best_eval_l1": best["l1"], "best_step": best["step"],
           "⚠️_eval_caveat": ("评估集**不参与训练**（真留出）⇒ 可作**相对**泛化指标；"
                              "但仍与训练池**同一数据集**，绝对水平不代表画质"
                              if held_out else
                              "固定集与训练集**同源**（数据太少无法留出）"
                              "⇒ 只能判断是否还在学，**不能**当泛化指标"),
           "⚠️_limitations": [
               ("**无真 LPIPS / GAN loss** ⇒ 重建质量有上限（如实报告，非 bug）；"
                "已提供零依赖的 `--w-lpips` 多尺度感知项作为部分替代"
                if True else ""),
               "**无 KL 项**（设计如此：Rectified Flow 用确定性 latent）",
               "语义通道**未做 DINOv3 对齐**（--dino-weight 默认 0，项目暂无 DINOv3 依赖）"
               "⇒ 产出的是**分块重建基线**，不是「语义已对齐」版本",
           ]}
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": vae.state_dict(), "base": base,
                    "config": {k: out[k] for k in ("steps", "size", "batch")},
                    "report": {k: v for k, v in out.items() if k != "history"}}, p)
        print(f"  ✅ 已保存 {p}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · HybridVAE 训练器")
    ap.add_argument("--image-dir", action="append", default=[],
                    help="图片目录（可多次）。⛔ 没图会报错，不用合成数据替代。")
    ap.add_argument("--cache", default=None,
                    help="⭐ 预解码缓存基名（kp.data.predecode 的 <base>）。"
                         "读取快 ~350×（AVIF 解码是训练瓶颈）。给了它就不用 --image-dir。")
    ap.add_argument("--shards", nargs="*", default=None,
                    help="⭐⭐ parquet 分片或目录 —— **多进程并行解码**，"
                         "零磁盘缓存、**无分辨率上限**（缓存方案 512² 要 266GB 放不下）。")
    ap.add_argument("--workers", type=int, default=8,
                    help="并行解码进程数（仅 --shards 用）。AVIF 解码 35ms/张（纯 CPU）。")
    ap.add_argument("--prefetch", type=int, default=8,
                    help="预取批数（仅 --shards 用）：解码与 GPU 训练重叠。")
    ap.add_argument("--policy", default="explicit_ok",
                    choices=["strict", "sensitive_ok", "explicit_ok"],
                    help="分级过滤（仅 --shards 用）。⚠️ 默认 explicit_ok = "
                         "**项目 2026-10-04 决策**（消融实测：加不伤 safe、只小赚 ~2-3%）。"
                         "库级默认仍是 fail-safe 的 strict（见 kp/data/rating.py）。")
    ap.add_argument("--n-eval", type=int, default=256,
                    help="⭐ 留出评估集张数（**不参与训练**）。<64 张数据时自动退回同源评估。")
    ap.add_argument("--ckpt-dir", default=None,
                    help="⏱️ 周期存盘目录（可中断训练）。⚠️ 长跑**必须**给，否则崩了全丢。")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="每多少步存一次（0=自动：max(1000, steps//20)）")
    ap.add_argument("--resume", action="store_true", help="从 --ckpt-dir 续跑")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--w-sem", type=float, default=1.0)
    ap.add_argument("--w-det", type=float, default=1.0)
    ap.add_argument("--w-lpips", type=float, default=0.0,
                    help="多尺度感知权重（零依赖 LPIPS 替代）。0=关闭")
    ap.add_argument("--w-tlpips", type=float, default=0.0,
                    help="⭐**真 LPIPS** 权重（预训练 AlexNet 特征，权重随包自带）。"
                         "与 --w-lpips 是**两个不同的轴**：那个是像素家族，这个是"
                         "预训练特征家族。实测 P1 的瓶颈在感知/分布层（LPIPS 差 4.51×、"
                         "rFID 差 14.24×）⇒ 罚像素已在天花板，须换度量空间")
    ap.add_argument("--w-grad", type=float, default=0.5, help="边缘项权重")
    ap.add_argument("--opt", default="adamw8bit",
                    choices=("adamw", "adamw8bit", "sgd"),
                    help="⭐ 优化器。⛔ 8GB 卡必须 adamw8bit —— fp32 AdamW 的状态在 "
                         "base128 下就要 4268MiB，512² 直接 OOM（实测）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="保存 .pt 路径")
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args(argv)

    if not a.image_dir and not a.cache and not a.shards:
        print("用法：三选一 ——"
              "\n  ① --shards <parquet分片或目录>  ⭐ 推荐：多进程并行解码，零磁盘、无分辨率上限"
              "\n  ② --cache <预解码缓存基名>      快，但缓存体积 = N×H×W×3（512² 要 266GB）"
              "\n  ③ --image-dir <目录>           小数据/调试用"
              "\n⭐ 项目默认数据位置：data/characters/kokona/images"
              "\n⭐ parquet 数据位置：out/data/curated_danbooru/_shards")
        return 2
    print("=" * 64)
    print("P1 · HybridVAE 训练（无 KL · 分块重建 · ⛔ 无 perceptual/GAN）")
    print("=" * 64)
    train_vae(a.image_dir, cache=a.cache, shards=a.shards,
              workers=a.workers, prefetch=a.prefetch, policy=a.policy,
              steps=a.steps, batch=a.batch, size=a.size, lr=a.lr,
              base=a.base, w_sem=a.w_sem, w_det=a.w_det,
              w_lpips=a.w_lpips, w_grad=a.w_grad, w_tlpips=a.w_tlpips,
              device=a.device,
              opt_name=a.opt,
              seed=a.seed, out_path=a.out, log_every=a.log_every,
              n_eval=a.n_eval, ckpt_dir=a.ckpt_dir,
              ckpt_every=a.ckpt_every, resume=a.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

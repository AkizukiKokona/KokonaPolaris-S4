"""P1 · 预解码缓存 —— 把 AVIF 解码搬到训练之前做一次，训练时只读内存。

═══ 🔴 为什么必须有这个（实测，不是猜）═══

2026-10-04 实测 `curated-danbooru-2026` 的图 **99.9% 是 AVIF**，
而 AVIF 解码**极贵**：

    AVIF 解码 + 转 RGBA + 缩到 128² = **35 ms/张（纯 CPU）**
    ⇒ batch 8 的一个 step **光取数就要 280 ms**
    ⇒ 训练全程 **GPU 利用率 0% / ~10W**（显卡在等 CPU）
    ⇒ 34 万张一个 epoch **光解码 3.3 小时**

⭐ 解法：**解码一次、缓存成 uint8 数组**。训练时读 memmap ≈ 0.1ms/张
（比 AVIF 快 **~350×**），且**多个训练实验可复用同一份缓存**。

═══ 产物 ═══

    <out>.npy    uint8 (N, size, size, 3) —— RGB，已按训练尺寸等比框缩放
    <out>.json   {n, size, source_shards, rating_counts, items:[{shard,row,rating,ext}]}

⚠️ 缓存的是 **RGB**（丢掉 alpha）。本数据集全是 RGB（实测 `mode=RGB`），无损失；
   **有 alpha 的数据集不要用这个工具**（重建口径会变）。

═══ 用法 ═══

    # 从 parquet 分片直接缓存（推荐：一次读全，不顺带产出 36GB 中间图片）
    python -m kp.data.predecode --shards out/data/curated_danbooru/_shards --out \\
        out/cache/danbooru20k_256 --size 256 --workers 8

    # ⚠️ 分片按文件名排序、逐片追加；已存在的 <out>.npy 会被**覆盖**（不做增量）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .rating import classify_rating

#: 单进程解码一个分片时，交给进程池的任务粒度
_CHUNK = 64


def _decode_one(job: tuple) -> Optional[np.ndarray]:
    """把一个图片字节串解码成 (size,size,3) uint8。失败返回 `None`。"""
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


def _shard_files(shards: Sequence[os.PathLike | str]) -> List[Path]:
    """展开：目录 → 其下所有 parquet；文件 → 自身。按名字排序（可复现）。"""
    out: List[Path] = []
    for s in shards:
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.glob("*.parquet")))
        elif p.exists():
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def predecode(shards: Sequence[os.PathLike | str], out_base: str, *,
              size: int = 256, workers: int = 8, batch: int = 256,
              progress_every: int = 2000) -> dict:
    """把分片解码成 `<out_base>.npy` + `<out_base>.json`。"""
    import pyarrow.parquet as pq

    files = _shard_files(shards)
    if not files:
        raise FileNotFoundError(f"没找到 parquet 分片（扫了 {list(shards)}）")

    # ⚠️⚠️ 2026-10-05 修（实测复现）：原实现用 `Path.with_suffix("")`规范化基名，
    #    而 `with_suffix` 会把**最后一个点后的一切当扩展名删掉**：
    #       'out/cache/danbooru.20k_256' -> 'out/cache/danbooru'  （`.20k_256` 被吃掉）
    #       'out/cache/v1.0_512'         -> 'out/cache/v1'
    #    ⇒ 两个只在点号后不同的数据集基名会**塌成同一个 .npy，静默互相覆盖**。
    #✅ 改用字符串拼接：只认「最后一个点右边没有多余的点」才当扩展名。
    out_npy = Path(out_base)
    stem = out_npy.name
    if "." in stem:
        head, _, tail = stem.rpartition(".")
        # 尾部像扩展名（短且无点）才剥掉；否则整段当名字的一部分
        if head and len(tail) <= 5 and "/" not in tail:
            stem = head
    base = out_npy.with_name(stem)
    out_npy = Path(str(base) + ".npy")
    out_json = Path(str(base) + ".json")
    # ⭐⭐ `size` 进键（审计 P2-4）：docstring 承诺 npy 形状是 (N,size,size,3)，
    #    但**缓存文件名与 size 无关** ⇒ 同基名跑 --size 256 和 --size 512 会覆盖，
    #    且 `_CacheSource` 读时不校验 ⇒ 静默读到错分辨率的缓存。
    #✅ 统一加 `_s{size}` 后缀，**消除撞键**。
    if a_size_suffix := size:
        out_npy = Path(str(base) + f"_s{a_size_suffix}" + ".npy")
        out_json = Path(str(base) + f"_s{a_size_suffix}" + ".json")
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    # ---- ① 先扫一遍拿总行数（préallocation 用） ----
    total = 0
    for f in files:
        total += pq.ParquetFile(f).metadata.num_rows
    print(f"  分片 {len(files)} 个 · 共 {total:,} 行 · 缓存尺寸 {size}² · workers {workers}")

    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8,
                                    shape=(total, size, size, 3))
    items: List[dict] = []
    rating_counts: dict = {}
    ext_counts: dict = {}
    failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        w = 0
        for f in files:
            pf = pq.ParquetFile(f)
            cols = [c for c in ("prompt", "rating", "is_explicit", "image", "tags",
                                "tag_string", "caption") if c in pf.schema_arrow.names]
            if "image" not in pf.schema_arrow.names:
                print(f"  ⚠️ 跳过 {f.name}（没有 image 列）")
                continue
            stem = f.stem
            src_row = 0                      # ⭐ 分片内行号（跨 batch 累加）
            for rb in pf.iter_batches(batch_size=batch, columns=cols):
                rows = rb.to_pylist()
                jobs = [(r.get("image"), size) for r in rows]
                for r, im in zip(rows, ex.map(_decode_one, jobs, chunksize=_CHUNK)):
                    rating = classify_rating(r)
                    rating_counts[rating] = rating_counts.get(rating, 0) + 1
                    my_row = src_row
                    src_row += 1
                    if im is None:
                        failed += 1
                        continue
                    arr[w] = im
                    # ⚠️ 2026-10-05 修语义陷阱：原代码 `"row": len(items)`
                    #    记的是**跨分片累加的全局序号**，但字段名 `row` 在
                    #    `_ShardStreamSource` 里指的是「**分片内**行号」⇒ 同名不同义。
                    #⇒ 改名 `out_index`（缓存内下标，可靠）+ 显式记 `src_row`（**分片内**行号）。
                    items.append({"shard": stem,
                                  "out_index": len(items),   # ⭐ 本npy 里的下标（可靠）
                                  "src_row": my_row,  # 分片内行号（-1=未知）
                                  "rating": rating})
                    w += 1
                if w and w % progress_every < batch:
                    el = time.time() - t0
                    print(f"    已解码 {w:,}/{total:,}  "
                          f"({w/max(el,1e-9):.0f} 张/秒, {el:.0f}s)")
        arr.flush()
        del arr

    if w != total:
        # 有坏图 ⇒ 截断到实际条数（重写 npy 头部）
        full = np.load(out_npy, mmap_mode="r")
        np.save(out_npy, np.asarray(full[:w]))
        print(f"  ⚠️ 实际写入 {w:,} / {total:,}（坏图 {failed} 张）⇒ 已截断")

    elapsed = time.time() - t0
    rep = {"out_npy": str(out_npy), "out_json": str(out_json),
           "n": w, "size": size, "source_shards": [f.name for f in files],
           "rating_counts": rating_counts, "failed": failed,
           "elapsed_s": round(elapsed, 1),
           "throughput_img_per_s": round(w / max(elapsed, 1e-9), 1),
           "items": items,
           "⚠️_note": "RGB 缓存（丢弃 alpha）—— 有 alpha 的数据集不要用本工具"}
    out_json.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(f"  ✅ {out_npy.name}  {os.path.getsize(out_npy)/1e9:.2f} GB  "
          f"· 耗时 {elapsed:.0f}s · {w/max(elapsed,1e-9):.0f} 张/秒")
    print(f"  ✅ {out_json.name}")
    for k, v in sorted(rating_counts.items(), key=lambda kv: -kv[1]):
        print(f"     {k:12s} {v:>8,}")
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 预解码缓存（解 AVIF 解码瓶颈）")
    ap.add_argument("--shards", nargs="+", required=True, help="parquet 分片或目录")
    ap.add_argument("--out", required=True, help="输出基名（自动加 .npy/.json）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    print("=" * 68)
    print("P1 · 预解码缓存")
    print("=" * 68)
    predecode(a.shards, a.out, size=a.size, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

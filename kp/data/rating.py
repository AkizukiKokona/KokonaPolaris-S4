"""P1 · 数据分级过滤 —— 回答「敏感内容怎么办，你是否要我额外准备」。

═══ 结论（先说，省得你操心）═══

**⛔ 不需要额外准备语料。** Danbooru 系数据集**自带分级字段**
（`rating` / `is_explicit` / `tag_string` 里的 `rating:*` 元标签）——
这是该数据集的**标准列**，不是我们要外挂的东西。

⇒ 过滤发生在**读取时**，不是**下载时**：
   · 好图（`safe`）照样下、照样用；
   · 敏感图要么**不读**，要么**读了也不用**（权重学不到）。

⭐ **这意味着**：**下数据这件事本身没有任何合规负担**，
真正需要判断的是「**训练时用哪些**」，而那是**代码里的一个过滤条件**。

═══ 三档策略（按需选，默认最严）═══

| 档 | 保留 | 适用 |
|---|---|---|
| **`strict`（默认）** | 仅 `rating == "safe"` | ⭐ 推荐：P1 训 VAE 根本不需要敏感内容 |
| **`sensitive_ok`** | `safe` + `sensitive` | 若后续要训 LoRA 且目标含 NSFW |
| **`explicit_ok`** | 全留 | ⚠️ **不用于任何可分发权重** |

⚠️ **默认 `strict` 的理由**：P1 是训 **VAE**（学的是「图像怎么压成 latent」），
**这个任务与内容无关** —— 用 safe 语料训出来的 VAE，在**推理时照样能处理任意内容**
（VAE 是通用编码器）。⇒ **没必要为 VAE 引入敏感数据**。

⚠️ 但 **LoRA / 主干训练**要另议：模型「会什么」由**训练数据**决定，
不是 VAE 决定。⇒ 设计稿 §「部署三层责任」那条在这里适用。

═══ 怎么用 ═══

    # 查看一个分片里各档有多少（不下整片，只读元数据 + 首行）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.rating \
        --inspect out/data/danbooru_probe/_shards/*.parquet

    # 过滤出一个「只含 safe」的子集（流式，不整片加载）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.rating \
        --shards out/data/danbooru_probe/_shards --policy strict --out out/data/safe_only

    # 训练时直接指定
    .venv\\Scripts\\python.exe -m kp.train.vae_pretrain --image-dir out/data/safe_only ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: 分级判定顺序：先找显式字段，找不到再看 tag_string 里的 `rating:*` 元标签
EXPLICIT_TAGS = ("rating:explicit", "rating:sensitive")
SAFE_TAGS = ("rating:safe",)


def classify_rating(row: dict) -> str:
    """把一条记录判成 `safe` / `sensitive` / `explicit` / `unknown`。

    ⭐ **三级来源（按可靠性排序）**：
        ① 显式列 `rating`（Danbooru 官方导出必有）
        ② 显式列 `is_explicit`（bool，退化为 safe/explicit 两档）
        ③ `tag_string` 里的 `rating:*` 元标签（兜底）
    ⚠️ 判不出来时返回 `"unknown"` —— **不猜、不默认放行**。
    """
    r = row.get("rating")
    if isinstance(r, str) and r.strip():
        s = r.strip().lower()
        if s in ("safe", "sensitive", "explicit", "questionable"):
            return "questionable" if s == "questionable" else s
    ex = row.get("is_explicit")
    if isinstance(ex, bool):
        return "explicit" if ex else "safe"
    tags = row.get("tag_string") or row.get("tags") or ""
    if isinstance(tags, str) and tags:
        low = tags.lower()
        if "rating:explicit" in low:
            return "explicit"
        if "rating:sensitive" in low:
            return "sensitive"
        if "rating:safe" in low:
            return "safe"
    return "unknown"


POLICIES = {
    "strict": {"safe"},
    "sensitive_ok": {"safe", "sensitive"},
    "explicit_ok": {"safe", "sensitive", "explicit"},
}


def iter_rows(shard: Path, columns: Optional[Sequence[str]] = None,
              batch: int = 512):
    """流式读 parquet（⛔ 不整片加载进内存）。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    for rb in pf.iter_batches(batch_size=batch, columns=list(columns) if columns else None):
        for row in rb.to_pylist():
            yield row


def inspect_shard(shard: Path) -> dict:
    """统计一个分片的分级分布 + 字段名。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    cols = [f.name for f in pf.schema_arrow]
    have = [c for c in ("rating", "is_explicit", "tag_string", "tags", "image") if c in cols]
    counts: Dict[str, int] = {}
    n = 0
    img_col = "image" if "image" in cols else None
    for row in iter_rows(shard, have):
        n += 1
        r = classify_rating(row)
        counts[r] = counts.get(r, 0) + 1
    return {"shard": str(shard), "rows": n, "columns": cols,
            "rating_columns_found": have, "image_column": img_col,
            "rating_counts": counts}


def filter_shard(shard: Path, policy: str, out_dir: Path,
                 batch: int = 512) -> dict:
    """按策略过滤并把图片写成 PNG。⛔ 原分片**只读**，绝不修改。"""
    import pyarrow.parquet as pq
    from PIL import Image
    import io
    import numpy as np

    keep = POLICIES[policy]
    d = out_dir
    d.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(shard)
    cols = [f.name for f in pf.schema_arrow]
    need = [c for c in ("rating", "is_explicit", "tag_string", "tags", "image") if c in cols]
    if "image" not in cols:
        return {"error": "该分片没有 image 列（可能只含元数据）", "columns": cols}
    written, skipped, failed = 0, 0, 0
    stem = shard.stem
    for i, row in enumerate(iter_rows(shard, need)):
        r = classify_rating(row)
        if r not in keep:
            skipped += 1
            continue
        v = row.get("image")
        if v is None:
            failed += 1
            continue
        buf = v["bytes"] if isinstance(v, dict) else v
        try:
            im = Image.open(io.BytesIO(buf)).convert("RGB")
            im.save(d / f"{stem}_{i:06d}.png")
            written += 1
        except Exception:                                    # noqa: BLE001
            failed += 1
    return {"shard": str(shard), "policy": policy, "keep": sorted(keep),
            "written": written, "skipped": skipped, "failed": failed,
            "out": str(d)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 数据分级过滤（⛔ 不需额外准备语料）")
    ap.add_argument("--shards", nargs="*", default=[], help="parquet 分片（可多个）")
    ap.add_argument("--inspect", default=None, help="只统计分级分布")
    ap.add_argument("--policy", default="strict", choices=list(POLICIES),
                    help="保留哪一档（默认 strict = 只留 safe）")
    ap.add_argument("--out", default=None, help="过滤后图片的输出目录")
    a = ap.parse_args(argv)

    if a.inspect:
        p = Path(a.inspect)
        rep = inspect_shard(p)
        print("=" * 68)
        print(f"分级检查：{p.name}")
        print("=" * 68)
        print(f"  行数: {rep['rows']:,}")
        print(f"  图片列: {rep['image_column']}")
        print(f"  分级列: {rep['rating_columns_found']}")
        print(f"  分布:")
        tot = max(rep["rows"], 1)
        for k, v in sorted(rep["rating_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:10s} {v:>8,}  ({100*v/tot:5.2f}%)")
        print(f"\n  ⚠️ unknown = 判不出分级 ⇒ **不默认放行**")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0

    if a.shards and a.out:
        out = Path(a.out)
        reps = [filter_shard(Path(s), a.policy, out) for s in a.shards]
        print("=" * 68)
        print(f"过滤（policy={a.policy}，保留 {sorted(POLICIES[a.policy])}）")
        print("=" * 68)
        for r in reps:
            print(json.dumps(r, ensure_ascii=False))
        tot = sum(r.get("written", 0) for r in reps)
        print(f"\n✅ 共写出 {tot:,} 张到 {out}")
        print(f"   训练：.venv\\Scripts\\python.exe -m kp.train.vae_pretrain "
              f"--image-dir {out} --size 128 --steps 2000")
        return 0

    print(__doc__.split("═══ 怎么用")[0])
    print("用法：--inspect <shard.parquet> | --shards a.parquet b.parquet --out DIR")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

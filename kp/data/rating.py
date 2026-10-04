"""P1 · 数据分级过滤 —— 回答「敏感内容怎么办，你是否要我额外准备」。

═══ 结论（先说，省得你操心）═══

**⛔ 不需要额外准备语料。** Danbooru 系数据集**自带分级信息**（是本数据集的**标准列/token**，
不是我们要外挂的东西）⇒ 过滤发生在**读取时**，不是**下载时**：
   · 想留的照样下、照样用；
   · 不想留的要么**不读**，要么**读了也不用**（权重学不到）。

⭐ **这意味着**：**下数据这件事本身没有任何合规负担**，
真正需要判断的是「**训练时用哪些**」，而那是**代码里的一个过滤条件**。

═══ 🔴 2026-10-04 实测修正：分级「在哪一列」不是固定的 ═══

原版假设分级在 `rating` / `is_explicit` / `tag_string` 列里。
**实测 `aipracticecafe/curated-danbooru-2026` 一个这样的列都没有**
（该集是**预筛选过的** curated 集），它的实际列是：

    booru_id, image, prompt, bucket_idx, target_width/height,
    original_width/height, aspect_ratio, tier, aesthetic_tier, tag_weight

分级**作为裸 token 混在 `prompt` 字符串里**（`rating:` 前缀被去掉）：

    1girl, minori \\(senran kagura\\), senran kagura, sensitive, yaegashi nan, ...
    1girl, 2boys, 2b \\(nier:automata\\), ..., explicit, caisan, bad score, ...

⇒ 原版在这个集上 **100% 返回 `unknown`**（= 一条都判不出，管线实际空转）。
⇒ 本模块现在**同时认三种载体**：显式列 → 元标签 → **逗号 token 扫描**。

⚠️ token 扫描**必须整 token 匹配** —— `general` / `sensitive` 是常见英文词，
子串匹配必然误判（例如把 `sensitive_hair` 之类判成敏感）。
⚠️ `general` 是 Danbooru 对 **safe** 的正式叫法 ⇒ 归一化成 `safe`。
⚠️ 同一串里出现多个分级 token 时**取最严的那档**（fail-closed）。

═══ 三档策略（按需选，默认最严）═══

| 档 | 保留 | 适用 |
|---|---|---|
| **`strict`（默认）** | 仅 `safe`（Danbooru 写作 `general`） | ⭐ 推荐：P1 训 VAE 根本不需要敏感内容 |
| **`sensitive_ok`** | `safe` + `sensitive` | 若后续要训 LoRA 且目标含 NSFW |
| **`explicit_ok`** | **全留**（`safe`+`sensitive`+`questionable`+`explicit`） | ⚠️ **不用于任何可分发权重** |

⚠️ **`explicit_ok` 补 `questionable`（2026-10-04 修的一个静默 bug）**：原版
`{"safe","sensitive","explicit"}` **漏了 `questionable`** ⇒ 宣称「全留」却会**静默丢掉
~11% 的图**。这类「说全留其实没全留」的 bug 不报错，最难发现。

⚠️ **默认 `strict` 的理由**：P1 是训 **VAE**（学的是「图像怎么压成 latent」），
**这个任务与内容无关** —— 用 safe 语料训出来的 VAE，在**推理时照样能处理任意内容**
（VAE 是通用编码器）。⇒ **没必要为 VAE 引入敏感数据**。

⚠️ 但 **LoRA / 主干训练**要另议：模型「会什么」由**训练数据**决定，
不是 VAE 决定。⇒ 设计稿 §「部署三层责任」那条在这里适用。

🔴 **一条 rating 字段覆盖不到的硬约束（不可作为「覆盖率权衡」）**：
Danbooru 的 rating 是**性内容分级，不是年龄分级** ⇒ **`safe` ≠「角色是成年人」**。
要排除未成年人相关内容，**必须另加独立的一层过滤**（角色标签 + 人工审核），
本模块**不提供、不替代**那一层。

═══ 判不出怎么办 ═══
`unknown` **一律不保留**（fail-closed）。宁可少要，不可误放。

═══ 怎么用 ═══

    # 查看一个分片里各档有多少（不下整片）
    python -m kp.data.rating --inspect out/data/curated_danbooru/_shards/data_shard_00000.parquet

    # 过滤出一个子集（流式，不整片加载）
    python -m kp.data.rating --shards out/data/curated_danbooru/_shards --policy strict \
        --out out/data/safe_only

    # 训练时直接指定
    python -m kp.train.vae_pretrain --image-dir out/data/safe_only ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: 归一化词表：Danbooru 的评级词 → 本模块的规范档位。
#: ⚠️ `general` 是 Danbooru 对 **safe** 的正式叫法（不是本文档发明的别名）。
RATING_WORDS: Dict[str, str] = {
    "general": "safe",
    "safe": "safe",
    "sensitive": "sensitive",
    "questionable": "questionable",
    "explicit": "explicit",
}

#: 严重度排序（多个 token 命中时取最大 ⇒ fail-closed）
_SEVERITY = {"safe": 0, "sensitive": 1, "questionable": 2, "explicit": 3}

#: 分级可能藏在哪些文本列里（按可靠性排序）
TEXT_COLUMNS = ("tag_string", "tags", "prompt", "caption")

#: 兼容旧预期的元标签写法
EXPLICIT_TAGS = ("rating:explicit", "rating:sensitive")
SAFE_TAGS = ("rating:safe",)

#: token 分隔符：Danbooru 用逗号；兼容换行/分号
_SPLIT = re.compile(r"[,\n;]")


def _normalize_word(word: str) -> Optional[str]:
    """把一个词归一成规范档位；认不出返回 `None`。"""
    w = word.strip().lower()
    if w.startswith("rating:"):
        w = w.split(":", 1)[1].strip()
    return RATING_WORDS.get(w)


def scan_rating_tokens(text: object) -> Optional[str]:
    """从逗号分隔的标签串里找分级 token。⛔ **整 token 匹配**；多个命中取**最严**。"""
    if not isinstance(text, str) or not text:
        return None
    best: Optional[str] = None
    for raw in _SPLIT.split(text):
        got = _normalize_word(raw)
        if got is not None and (best is None or _SEVERITY[got] > _SEVERITY[best]):
            best = got
    return best


def classify_rating(row: dict) -> str:
    """把一条记录判成 `safe` / `sensitive` / `questionable` / `explicit` / `unknown`。

    ⭐ **四级来源（按可靠性排序）**：
        ① 显式列 `rating`（Danbooru 官方导出有；curated 集没有）
        ② 显式列 `is_explicit`（bool，退化为 safe/explicit 两档）
        ③ 文本列里的 `rating:*` 元标签（`tag_string` / `tags` / `prompt` / `caption`）
        ④ 文本列里的**裸 token**（`general` / `sensitive` / `questionable` / `explicit`）
    ⚠️ 判不出来时返回 `"unknown"` —— **不猜、不默认放行**。
    """
    # ① 显式列 rating
    r = row.get("rating")
    if isinstance(r, str) and r.strip():
        got = _normalize_word(r)
        if got is not None:
            return got
    # ② 显式列 is_explicit
    ex = row.get("is_explicit")
    if isinstance(ex, bool):
        return "explicit" if ex else "safe"
    # ③④ 文本列（元标签 + 裸 token 其实是同一套 token 扫描）
    for col in TEXT_COLUMNS:
        got = scan_rating_tokens(row.get(col))
        if got is not None:
            return got
    return "unknown"


POLICIES = {
    "strict": {"safe"},
    "sensitive_ok": {"safe", "sensitive"},
    # ⚠️ 必须含 questionable，否则「全留」是假的
    "explicit_ok": {"safe", "sensitive", "questionable", "explicit"},
}

#: 判断分级时**需要读**的列（含图像列，供 filter 用）
_SRC_COLUMNS = ("rating", "is_explicit", "image") + TEXT_COLUMNS


def iter_rows(shard: Path, columns: Optional[Sequence[str]] = None,
              batch: int = 512):
    """流式读 parquet（⛔ 不整片加载进内存）。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    for rb in pf.iter_batches(batch_size=batch, columns=list(columns) if columns else None):
        for row in rb.to_pylist():
            yield row


def inspect_shard(shard: Path) -> dict:
    """统计一个分片的分级分布 + 字段名 + **分级来自哪一列**。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    cols = [f.name for f in pf.schema_arrow]
    # ⚠️ 只统计分级，**不要读 image 列**（实测：读图 1084MB vs 不读 2.9MB，省 374×）
    have = [c for c in _SRC_COLUMNS if c in cols and c != "image"]
    counts: Dict[str, int] = {}
    source_hits: Dict[str, int] = {}
    n = 0
    img_col = "image" if "image" in cols else None
    for row in iter_rows(shard, have):
        n += 1
        r = classify_rating(row)
        counts[r] = counts.get(r, 0) + 1
        # 记录是哪一列判出来的（诊断用）
        if r != "unknown":
            src = "rating" if isinstance(row.get("rating"), str) and row.get("rating", "").strip() else (
                "is_explicit" if isinstance(row.get("is_explicit"), bool) else "text")
            source_hits[src] = source_hits.get(src, 0) + 1
    return {"shard": str(shard), "rows": n, "columns": cols,
            "rating_columns_found": have, "image_column": img_col,
            "rating_counts": counts, "rating_source": source_hits}


def filter_shard(shard: Path, policy: str, out_dir: Path,
                 batch: int = 512) -> dict:
    """按策略过滤并落盘图片。⛔ 原分片**只读**，绝不修改。

    ⭐ **保留原字节、不重编码**（2026-10-04）：源图平均 ~108KB，若统一转 PNG 会
    膨胀 ~10×（340K 张全量下 = 30GB vs 300GB，且白付一次编解码）。
    训练侧 `kp.train.vae_pretrain` 本来就会 `convert("RGBA")` 归一化格式，
    所以这里**没有**转码的必要。扩展名从文件头识别，取 `IMG_EXTS` 内的值。
    """
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    keep = POLICIES[policy]
    d = out_dir
    d.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(shard)
    cols = [f.name for f in pf.schema_arrow]
    need = [c for c in _SRC_COLUMNS if c in cols]
    if "image" not in cols:
        return {"error": "该分片没有 image 列（可能只含元数据）", "columns": cols}
    written, skipped, failed = 0, 0, 0
    counts: Dict[str, int] = {}
    exts: Dict[str, int] = {}
    stem = shard.stem
    for i, row in enumerate(iter_rows(shard, need)):
        r = classify_rating(row)
        counts[r] = counts.get(r, 0) + 1
        if r not in keep:
            skipped += 1
            continue
        v = row.get("image")
        if v is None:
            failed += 1
            continue
        buf = v["bytes"] if isinstance(v, dict) else v
        try:
            with Image.open(io.BytesIO(buf)) as im:
                fmt = (im.format or "png").lower()
                _ = im.size                     # 触发行头解析，坏图在此抛出
            ext = {"jpeg": "jpg", "mpo": "jpg"}.get(fmt, fmt)
            exts[ext] = exts.get(ext, 0) + 1
            (d / f"{stem}_{i:06d}.{ext}").write_bytes(buf)
            written += 1
        except Exception:                                    # noqa: BLE001
            failed += 1
    return {"shard": str(shard), "policy": policy, "keep": sorted(keep),
            "written": written, "skipped": skipped, "failed": failed,
            "rating_counts": counts, "ext_counts": exts, "out": str(d)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 数据分级过滤（⛔ 不需额外准备语料）")
    ap.add_argument("--shards", nargs="*", default=[], help="parquet 分片（可多个）")
    ap.add_argument("--inspect", default=None, help="只统计分级分布")
    ap.add_argument("--policy", default="strict", choices=list(POLICIES),
                    help="保留哪一档（默认 strict = 只留 safe/general）")
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
        print(f"  可用列: {rep['rating_columns_found']}")
        print(f"  判定来源: {rep['rating_source'] or '(全部 unknown)'}")
        print(f"  分布:")
        tot = max(rep["rows"], 1)
        for k, v in sorted(rep["rating_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v:>8,}  ({100*v/tot:5.2f}%)")
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
        print(f"   训练：python -m kp.train.vae_pretrain "
              f"--image-dir {out} --size 128 --steps 2000")
        return 0

    print(__doc__.split("═══ 怎么用")[0])
    print("用法：--inspect <shard.parquet> | --shards a.parquet b.parquet --out DIR")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

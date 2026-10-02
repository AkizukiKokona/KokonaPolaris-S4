"""KokonaPolaris · 数据交付校验器（补充 11 §4 的可执行实现）

用途：用户把一批数据交过来后，先跑这个脚本，确认格式合规、无重复、分辨率达标。

用法：
    python validate_manifest.py --dir D:\\model\\data\\<批次名>
    python validate_manifest.py --dir ... --min-side 1024 --json report.json
    python validate_manifest.py --dir ... --strict        # 警告也算失败

检查项：
    1. 目录结构：images/ 与 manifest.csv
    2. manifest 表头：file,tag,caption,source
    3. 文件存在性 + 未登记文件
    4. 图片可读 + 短边分辨率（默认 ≥ 1024）
    5. caption / tag / source 缺失
    6. 重复：完全重复（md5）+ 近似重复（dHash 汉明距离）
    7. tag 分布统计

退出码：0 = 无错误（有警告仍为 0，除非 --strict）；1 = 有错误。

⚠️ 纯 CPU / IO，可在夜间运行（不碰 GPU）。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

EXPECTED_HEADER = ["file", "tag", "caption", "source"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def fail(msg):
    print(f"  [错误] {msg}")


def warn(msg):
    print(f"  [警告] {msg}")


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash(path, size=8):
    """感知哈希：缩到 9x8 灰度，比较水平相邻像素 → 64 bit。"""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            px = list(im.tobytes())
        bits = 0
        for r in range(size):
            for c in range(size):
                left = px[r * (size + 1) + c]
                right = px[r * (size + 1) + c + 1]
                bits = (bits << 1) | (1 if left > right else 0)
        return bits
    except Exception:
        return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="批次目录（内含 images/ 与 manifest.csv）")
    ap.add_argument("--min-side", type=int, default=1024, help="图片短边最小像素（默认 1024）")
    ap.add_argument("--dup-threshold", type=int, default=6, help="dHash 汉明距离 ≤ 此值判为近似重复（默认 6）")
    ap.add_argument("--json", default=None, help="把报告写到 JSON")
    ap.add_argument("--strict", action="store_true", help="警告也视为失败")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    img_dir = os.path.join(root, "images")
    man = os.path.join(root, "manifest.csv")

    print("=" * 74)
    print("  KokonaPolaris · 数据交付校验（补充 11 §4）")
    print("=" * 74)
    print(f"  批次目录 : {root}")
    print(f"  短边下限 : {args.min_side} px")
    print(f"  近似重复 : dHash 汉明 ≤ {args.dup_threshold}")
    print()

    errors, warnings = [], []

    # --- 1. 目录结构 ---
    if not os.path.isdir(img_dir):
        errors.append(f"缺少 images/ 目录：{img_dir}")
    if not os.path.isfile(man):
        errors.append(f"缺少 manifest.csv：{man}")
    if errors:
        for e in errors:
            fail(e)
        print("\n  目录结构不完整，终止。")
        return 1

    # --- 2. manifest 表头 ---
    rows = []
    with open(man, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        norm = [h.strip().lower() for h in header]
        if norm[:4] != EXPECTED_HEADER:
            errors.append(f"表头应为 {EXPECTED_HEADER}，实际为 {header}")
        for r in reader:
            rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in r.items()})

    print(f"  manifest 记录数 : {len(rows)}")
    print()

    if not rows:
        errors.append("manifest.csv 没有任何数据行")

    # --- 3/4/5. 逐行检查 ---
    tag_counter = Counter()
    listed = set()
    size_buckets = Counter()
    md5_map = defaultdict(list)
    hash_map = defaultdict(list)

    for i, r in enumerate(rows, start=2):
        fn = r.get("file", "")
        if not fn:
            errors.append(f"第 {i} 行：file 为空")
            continue
        listed.add(os.path.basename(fn))
        p = os.path.join(img_dir, fn)
        if not os.path.isfile(p):
            errors.append(f"第 {i} 行：文件不存在 ({fn})")
            continue
        if os.path.splitext(fn)[1].lower() not in IMG_EXT:
            warnings.append(f"第 {i} 行：扩展名不在推荐集合内 ({fn})")

        if not r.get("tag"):
            errors.append(f"第 {i} 行：tag 缺失（无法定位擦除账本）")
        else:
            tag_counter[r["tag"]] += 1
        if not r.get("caption"):
            errors.append(f"第 {i} 行：caption 缺失")
        if not r.get("source"):
            warnings.append(f"第 {i} 行：source 为空（建议填写，便于合规追溯）")

        sz = image_size(p)
        if sz is None:
            errors.append(f"第 {i} 行：图片无法读取 ({fn})")
            continue
        w, h = sz
        short = min(w, h)
        size_buckets[f"{w}x{h}"] += 1
        if short < args.min_side:
            warnings.append(f"第 {i} 行：短边 {short} < {args.min_side} ({fn} {w}x{h})")

        md5_map[md5_of(p)].append(fn)
        dh = dhash(p)
        if dh is not None:
            hash_map[dh].append(fn)

    # --- 6. 重复检查 ---
    exact_dups = {k: v for k, v in md5_map.items() if len(v) > 1}
    for k, v in exact_dups.items():
        errors.append(f"完全重复（md5）：{', '.join(v)}")

    near_dups = []
    keys = list(hash_map.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = hamming(keys[a], keys[b])
            if d <= args.dup_threshold:
                for fa in hash_map[keys[a]]:
                    for fb in hash_map[keys[b]]:
                        near_dups.append((fa, fb, d))
    for fa, fb, d in near_dups[:50]:
        warnings.append(f"近似重复（dHash 距离 {d}）：{fa} ≈ {fb}")

    # --- 未登记文件 ---
    on_disk = {f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))}
    unlisted = sorted(on_disk - listed)
    for f in unlisted[:50]:
        warnings.append(f"images/ 中有未登记文件：{f}")
    if len(unlisted) > 50:
        warnings.append(f"... 另有 {len(unlisted) - 50} 个未登记文件")

    # --- 报告 ---
    print("-" * 74)
    print("  tag 分布")
    print("-" * 74)
    if tag_counter:
        for t, c in tag_counter.most_common():
            print(f"    {t:<24s} {c:>6d}")
    else:
        print("    （无）")
    print()

    print("-" * 74)
    print("  分辨率分布（前 10）")
    print("-" * 74)
    for s, c in size_buckets.most_common(10):
        print(f"    {s:<16s} {c:>6d}")
    print()

    print("=" * 74)
    print("  结论")
    print("=" * 74)
    print(f"    错误 {len(errors)}   ·   警告 {len(warnings)}")
    if errors:
        print("\n  —— 错误 ——")
        for e in errors[:80]:
            fail(e)
        if len(errors) > 80:
            print(f"  ... 另有 {len(errors) - 80} 条")
    if warnings:
        print("\n  —— 警告 ——")
        for w in warnings[:80]:
            warn(w)
        if len(warnings) > 80:
            print(f"  ... 另有 {len(warnings) - 80} 条")

    ok = (len(errors) == 0) and (not args.strict or len(warnings) == 0)
    print()
    print("  " + ("✅ 通过：可以交付" if ok else "❌ 未通过：请按上面修正后重跑"))
    print("=" * 74)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "dir": root, "rows": len(rows), "min_side": args.min_side,
                "errors": errors, "warnings": warnings,
                "tag_distribution": dict(tag_counter),
                "size_distribution": dict(size_buckets),
                "exact_duplicates": {k: v for k, v in exact_dups.items()},
                "near_duplicates": [{"a": a, "b": b, "distance": d} for a, b, d in near_dups],
                "unlisted": unlisted,
                "passed": ok,
            }, f, ensure_ascii=False, indent=2)
        print(f"  报告已写入：{args.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

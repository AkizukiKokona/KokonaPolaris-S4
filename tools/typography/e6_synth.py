"""e6_synth.py —— E6 交付物④：合成数据生成器（设计文档所说的「白拿」训练数据）。

设计文档原话：**字体光栅化是确定性的、成本≈0 → 零成本无限扩增，
这是全案唯一一种「白拿」训练数据**——用于日后训练 Layout Planner
（100M–1B 的独立小模型，输出 {canvas, panels, items[]} JSON）。

产出：成对的 (渲染图 PNG, 版面 JSON)。JSON 即 Ground Truth（规划器要学的目标）。
确定性：同 seed → 同 (图, JSON)。

用法：
  source /d/model/env.sh
  "$KP_PY" tools/typography/e6_synth.py --n 200 --res 512
"""
from __future__ import annotations

from kp.paths import MODELS_SANA, OUT

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kp_engine as E
import kp_schema as S
import kp_compose as C

OUT = OUT / "e6/synth"

# 额外字表：扩增稀见字/混合内容（真实训练应换成 GB2312 6763 常用字）
EXTRA = "雨夜灯影雪风雷鸣海浪云山门桥车站医院学校图书馆咖啡店天台走廊"


def rand_text(rng: random.Random, kind: str) -> str:
    if kind == "dialogue":
        base = rng.choice(S.DIALOGUE)
        if rng.random() < 0.35:
            k = rng.randint(1, 4)
            base += "".join(rng.choice(EXTRA) for _ in range(k)) + rng.choice("。！？")
        return base
    if kind == "narration":
        return rng.choice(S.NARRATION)
    if kind == "sfx":
        return rng.choice(S.SFX)
    if kind == "caption":
        return rng.choice(S.CAPTION)
    if kind == "mix":
        cn = "".join(rng.choice(EXTRA) for _ in range(rng.randint(3, 8)))
        en = rng.choice(["Latent", "Diffusion", "Typography", "KokonaPolaris-S4", "32x"])
        return f"{cn} {en} {rng.choice('。！？')}"
    return rng.choice(S.DIALOGUE)


def _pick_text(rng: random.Random, kind: str, cap: int, used: set) -> str:
    """按框容量 cap（字符数）挑一条能放下的文本；优先较长的候选。"""
    cap = max(1, min(cap, 60))
    if kind == "dialogue":
        pool = [t for t in S.DIALOGUE if len(t) <= cap]
        if not pool:
            pool = [(t[:cap] + "。") if len(t) > cap else t for t in S.DIALOGUE]
    elif kind == "narration":
        pool = [t for t in S.NARRATION if len(t) <= cap]
        if not pool:
            pool = ["三月后。", "那年夏天。", "风停了。", "夜很深。"]
    elif kind == "sfx":
        pool = [t for t in S.SFX if len(t) <= max(2, cap)] or S.SFX
    elif kind == "caption":
        pool = [t for t in S.CAPTION if len(t) <= max(4, cap + 4)] or S.CAPTION
    else:
        pool = S.DIALOGUE
    # 排除同页已用；若都用了就随机取
    fresh = [t for t in pool if t not in used]
    return rng.choice(fresh or pool)


def gen_one(idx: int, res: int, db: E.FontDB) -> dict:
    """生成单个样本，返回 manifest 条目。"""
    rng = random.Random(12345 + idx)
    layout = S.plan_page(res, res, seed=idx * 7919 + 13, margin=max(16, res // 22),
                         gutter=max(8, res // 48))
    # 以 plan_page 的随机性再替换文本：按框容量挑选可容纳的文本（同页去重）
    used: set[str] = set()
    for it in layout["items"]:
        ip = C._inner_poly(it)
        xs = [p[0] for p in ip]; ys = [p[1] for p in ip]
        bw = max(1.0, max(xs) - min(xs)); bh = max(1.0, max(ys) - min(ys))
        pad = it.get("padding", 6)
        tw, th = max(6.0, bw - 2 * pad), max(6.0, bh - 2 * pad)
        mode = it.get("writing_mode", "horizontal")
        lh = float(it.get("line_height", 1.3))

        # 目标字号：给 3–4 行/列余地，且不低于 15px（可读下限）
        target = int(min(tw, th) * (0.34 if mode == "horizontal" else 0.42))
        target = max(15, min(target, 56))
        per = max(1, int(tw // (target * (1.05 if mode == "horizontal" else 1.15))))
        rows = max(1, int(th // (target * lh)))
        cap = per * rows if mode == "horizontal" else max(1, int(th // target)) * \
            max(1, int(tw // (target * lh)))

        kind = {"bubble": "dialogue", "narration": "narration",
                "sfx": "sfx", "caption": "caption"}.get(it["type"], "dialogue")
        it["text"] = _pick_text(rng, kind, cap, used)
        used.add(it["text"])

        # 修正着重号 range 使其落在新文本内
        if it.get("emphasis"):
            fixed = []
            for e in it["emphasis"]:
                if "range" in e:
                    s, en = e["range"]
                    en = min(en, len(it["text"]))
                    s = min(s, max(0, en - 1))
                    if s < en:
                        fixed.append({"range": [s, en], "style": e.get("style", "dot")})
            it["emphasis"] = fixed or None
        # 真实引擎二分拟合字号（保证不溢出）
        it["size"] = E.fit_size(db, it["text"], it["font"], tw, th, target, mode, lh,
                                float(it.get("letter_spacing", 0.0)), floor=9)
        if it["type"] != "caption":
            it["size"] = max(9, int(it["size"] * rng.uniform(0.8, 1.0)))
    errs = S.validate(layout)
    img = C.render_layout(layout, db, show_panels=True)
    png = f"{OUT}/{idx:05d}.png"
    img.save(png, optimize=True)
    js = f"{OUT}/{idx:05d}.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, separators=(",", ":"))
    return {"idx": idx, "png": os.path.basename(png), "json": os.path.basename(js),
            "png_bytes": os.path.getsize(png), "json_bytes": os.path.getsize(js),
            "n_items": len(layout["items"]), "n_panels": len(layout["panels"]),
            "errors": errs}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    OUT = args.out or OUT
    os.makedirs(OUT, exist_ok=True)

    db = E.fontdb()
    print(f"[synth] 目标 {args.n} 样本 @ {args.res}px → {OUT}")

    manifest = []
    t0 = time.time()
    first10 = []
    for i in range(args.n):
        ts = time.time()
        e = gen_one(i, args.res, db)
        dt = time.time() - ts
        e["sec"] = round(dt, 4)
        manifest.append(e)
        if i < 10:
            first10.append(dt)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{args.n}  累计 {el:.1f}s  均 {el/(i+1)*1000:.0f} ms/样本")
    total = time.time() - t0

    png_total = sum(m["png_bytes"] for m in manifest)
    json_total = sum(m["json_bytes"] for m in manifest)
    n_err = sum(1 for m in manifest if m["errors"])
    per = total / args.n
    res_mb = (png_total + json_total) / 1e6

    stats = {
        "n": args.n, "res": args.res,
        "total_sec": round(total, 2),
        "per_sample_ms": round(per * 1000, 1),
        "warm_avg_ms": round(sum(first10[2:]) / max(1, len(first10) - 2) * 1000, 1),
        "png_total_mb": round(png_total / 1e6, 2),
        "json_total_mb": round(json_total / 1e6, 2),
        "total_mb": round(res_mb, 2),
        "per_sample_kb": round(res_mb * 1000 / args.n, 1),
        "schema_error_samples": n_err,
        # 外推
        "extrapolate_100k": {
            "sec": round(per * 100_000, 1),
            "hours": round(per * 100_000 / 3600, 2),
            "disk_gb": round(res_mb * 100_000 / args.n / 1024, 2),
        },
    }
    with open(f"{OUT}/_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "samples": manifest}, f, ensure_ascii=False)
    # 统计速查（人类可读）
    with open(f"{OUT}/_stats.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print("\n[synth] 完成")
    print(f"  样本数        : {args.n}")
    print(f"  总耗时        : {total:.2f}s  ({per*1000:.0f} ms/样本，冷启后 {stats['warm_avg_ms']:.0f} ms)")
    print(f"  磁盘占用      : {stats['total_mb']} MB  (PNG {stats['png_total_mb']} + JSON {stats['json_total_mb']})")
    print(f"  schema 失败   : {n_err} 条")
    ex = stats["extrapolate_100k"]
    print(f"  外推 10 万样本: {ex['hours']} 小时 / {ex['disk_gb']} GB")


if __name__ == "__main__":
    main()

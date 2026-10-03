"""P1 · 数据源获取（经代理 + **hf-mirror 镜像**）—— 解决「只有 11 张图」的数据缺口。

🔴 **为什么需要这个文件**（本轮实测）：
    P1（训 HybridVAE）卡在**数据量**上：全仓只有 **11 张**真图。
    增广（`kp.data.augment`）能扩到 176 张，但**那不是新内容**，只扩「不变性」。

═══ ⭐ 网络硬事实（本轮实测，别再踩）═══

| 端点 | 直连 | 经代理 `127.0.0.1:7897` |
|---|---|---|
| `github.com` | ✅ 200 | ✅ 200 |
| `huggingface.co` | ⛔ 超时 | ⚠️ **时通时断**（实测一次 200、一次 502 隧道失败） |
| **`hf-mirror.com`** | ⛔ | ✅ **稳定 200** |

⇒ 🔴 **本项目一律用 `HF_ENDPOINT=https://hf-mirror.com`**，不要用 `huggingface.co`。
⇒ ⚠️ 代理本身**不稳定**（隧道会 502）⇒ 下载**必须支持断点续跑 + 失败重试**。

═══ 已验证的候选数据集（2026-10-03 经 hf-mirror 实测）═══

| dataset_id | 许可 | 规模 | 备注 |
|---|---|---|---|
| **`aipracticecafe/curated-danbooru-2026`** | **apache-2.0** ✅ | 37 GB / 34 parquet / 100K–1M | ⭐ **首选**：许可干净、二次元、量够 |
| `deepghs/danbooru2023_index` | mit | 1.03 GB / 2011 json | 索引文件（小、快），适合作元数据 |
| `deepghs/anime_pictures-webp-4Mpixel` | other ⚠️ | 170 GB / 2000 tar | 量大但**许可不明** ⇒ 不用 |

⛔ **已验证不存在**（agent 实测 401，不要再试）：
`faztasia/baby-anime`、`huggan/anime-faces`

⚠️ **许可提醒**：`curated-danbooru-2026` 是 **Apache-2.0**（可商用可分发），
但**训练出的权重能否商用取决于所用的 tag 数据来源**（Danbooru 本身有争议）⇒
**这是法务问题不是技术问题**，本模块只做技术层，不替你判断。

═══ 用法 ═══
    # ① 探测（不下数据）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.fetch --probe

    # ② 下载前 N 张样本试水（推荐先做这步）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.fetch \
        --dataset aipracticecafe/curated-danbooru-2026 --sample 50 \
        --out out/data/curated_danbooru_probe

    # ③ 正式取（按 parquet 分片，天然支持断点续跑）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.fetch \
        --dataset aipracticecafe/curated-danbooru-2026 --max-shards 2 \
        --out out/data/curated_danbooru
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

# 🔴 必须在 import huggingface_hub **之前**设好（它读这两个环境变量）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

DEFAULT_PROXY = "http://127.0.0.1:7897"
MIRROR = "https://hf-mirror.com"

#: 已验证的候选（2026-10-03）。⚠️ `license=None` 或 `other` 的**不要**用来训可分发权重。
KNOWN_DATASETS = {
    "aipracticecafe/curated-danbooru-2026": {
        "license": "apache-2.0", "note": "⭐ 首选：许可干净 / 二次元 / 37GB",
    },
    "deepghs/danbooru2023_index": {
        "license": "mit", "note": "索引文件，小而快",
    },
    "deepghs/anime_pictures-webp-4Mpixel": {
        "license": "other", "note": "⚠️ 许可不明，不用于可分发权重",
    },
}

#: 已实测 401 不存在的（别再试）
DEAD_IDS = ("faztasia/baby-anime", "huggan/anime-faces")


def _session():
    import requests
    s = requests.Session()
    s.proxies.update({"http": DEFAULT_PROXY, "https": DEFAULT_PROXY})
    return s


def probe() -> dict:
    """探测各候选的**真实**可达性（走 hf-mirror）。"""
    out = {"endpoint": MIRROR, "proxy": DEFAULT_PROXY, "results": {}, "dead": {}}
    s = _session()
    for ds, meta in KNOWN_DATASETS.items():
        try:
            r = s.get(f"{MIRROR}/api/datasets/{ds}", timeout=25)
            if r.status_code != 200:
                out["results"][ds] = {"ok": False, "status": r.status_code}
                continue
            j = r.json()
            files = [x["rfilename"] for x in j.get("siblings", [])]
            data = [f for f in files if f.endswith((".parquet", ".tar", ".zip", ".json"))]
            out["results"][ds] = {
                "ok": True, "license": (j.get("cardData") or {}).get("license"),
                "n_data_files": len(data), "total_files": len(files),
                **meta,
            }
        except Exception as e:                              # noqa: BLE001
            out["results"][ds] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    for ds in DEAD_IDS:
        try:
            r = s.get(f"{MIRROR}/api/datasets/{ds}", timeout=20)
            out["dead"][ds] = r.status_code
        except Exception as e:                              # noqa: BLE001
            out["dead"][ds] = f"{type(e).__name__}"
    return out


def fetch_sample(dataset: str, n: int, out_dir: str,
                 retries: int = 3) -> dict:
    """下载 **n 张样本**（试水用）。⛔ 绝不整库下载。"""
    from huggingface_hub import hf_hub_download
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rep = {"dataset": dataset, "requested": n, "endpoint": MIRROR,
           "downloaded": 0, "files": [], "⚠️_note": "**样本试水**，不是完整数据集"}
    s = _session()
    try:
        r = s.get(f"{MIRROR}/api/datasets/{dataset}", timeout=25)
        if r.status_code != 200:
            rep["error"] = f"元数据不可达（HTTP {r.status_code}）"
            return rep
        files = [x["rfilename"] for x in r.json().get("siblings", [])
                 if x["rfilename"].endswith(".parquet")]
        if not files:
            rep["error"] = "该数据集没有 parquet 分片"
            return rep
        rep["n_shards"] = len(files)
        # 只取够 n 张的分片
        need = max(1, (n + 499) // 500)
        for f in files[:min(need, len(files))]:
            for attempt in range(retries):
                try:
                    p = hf_hub_download(repo_id=dataset, filename=f, repo_type="dataset",
                                        endpoint=MIRROR,
                                        local_dir=str(d / "_shards"))
                    rep["files"].append({"shard": f, "path": str(p),
                                         "bytes": os.path.getsize(p)})
                    break
                except Exception as e:                      # noqa: BLE001
                    if attempt == retries - 1:
                        rep.setdefault("errors", []).append(
                            f"{f}: {type(e).__name__}: {str(e)[:80]}")
                    else:
                        time.sleep(2 * (attempt + 1))       # 代理隧道不稳 ⇒ 退避重试
    except Exception as e:                                  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


def fetch_shards(dataset: str, max_shards: int, out_dir: str,
                 retries: int = 3) -> dict:
    """按 parquet 分片下载（**天然支持断点续跑**：已存在的文件会跳过）。"""
    from huggingface_hub import hf_hub_download
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    s = _session()
    rep = {"dataset": dataset, "endpoint": MIRROR, "shards": [], "skipped": 0}
    try:
        r = s.get(f"{MIRROR}/api/datasets/{dataset}", timeout=25)
        if r.status_code != 200:
            rep["error"] = f"元数据不可达（HTTP {r.status_code}）"
            return rep
        files = [x["rfilename"] for x in r.json().get("siblings", [])
                 if x["rfilename"].endswith(".parquet")]
        rep["n_shards_available"] = len(files)
        for f in files[:max_shards]:
            target = d / "_shards" / Path(f).name
            if target.exists() and target.stat().st_size > 0:
                rep["skipped"] += 1             # ← 断点续跑
                continue
            for attempt in range(retries):
                try:
                    p = hf_hub_download(repo_id=dataset, filename=f, repo_type="dataset",
                                        endpoint=MIRROR, local_dir=str(d / "_shards"))
                    rep["shards"].append({"shard": f, "bytes": os.path.getsize(p)})
                    break
                except Exception as e:                  # noqa: BLE001
                    if attempt == retries - 1:
                        rep.setdefault("errors", []).append(
                            f"{f}: {type(e).__name__}: {str(e)[:80]}")
                    else:
                        time.sleep(2 * (attempt + 1))
    except Exception as e:                                  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 数据源获取（hf-mirror + 代理）")
    ap.add_argument("--probe", action="store_true", help="只探测候选，不下载")
    ap.add_argument("--dataset", default="aipracticecafe/curated-danbooru-2026")
    ap.add_argument("--sample", type=int, default=0, help="下载 n 张样本")
    ap.add_argument("--max-shards", type=int, default=0, help="下载 n 个 parquet 分片")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    print("=" * 68)
    print(f"P1 · 数据源获取  endpoint={MIRROR}  proxy={DEFAULT_PROXY}")
    print("=" * 68)
    if a.probe:
        r = probe()
        for ds, v in r["results"].items():
            mark = "✅" if v.get("ok") else "⛔"
            print(f"  {mark} {ds}")
            if v.get("ok"):
                print(f"      license={v.get('license')}  分片={v.get('n_data_files')}  {v.get('note','')}")
            else:
                print(f"      {v.get('status') or v.get('error','')}")
        if r["dead"]:
            print(f"  ⛔ 已确认不存在: {r['dead']}")
        json.dump(r, open(Path(a.out or "out") / "data_probe" / "probe.json", "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
        return 0
    if a.sample:
        if not a.out:
            print("⛔ --sample 需要 --out")
            return 2
        r = fetch_sample(a.dataset, a.sample, a.out)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"\n⭐ 解包成 PNG：python -m kp.data.fetch --dataset {a.dataset} "
              f"--sample {a.sample} --out {a.out}")
        return 0
    if a.max_shards:
        if not a.out:
            print("⛔ --max-shards 需要 --out")
            return 2
        r = fetch_shards(a.dataset, a.max_shards, a.out)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0
    print("用法：--probe | --sample N --out DIR | --max-shards N --out DIR")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

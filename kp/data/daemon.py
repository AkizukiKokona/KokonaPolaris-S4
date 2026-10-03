"""P1 · 数据下载守护（断点续传 + 指数退避重试）—— 应对**时好时坏**的网络。

⭐ **为什么需要它**（实测环境的真实情况）：
    · `huggingface.co` 经代理**时通时断**（一次 200、一次 **502 隧道失败**）
    · `hf-mirror.com` 稳定，但吞吐只有 **~1.06 MB/s**
    · 单个 parquet 分片 **1084 MB** ⇒ 一次下完要 ~17 分钟，遇断必重来

⇒ **本模块的策略**：
    ① **分片级断点续跑**（`hf_hub_download` 自带 `.incomplete` 续传）
    ② **指数退避**重试（网络抖动能自愈）
    ③ **目标张数驱动**而非"下一个 N 个"—— 下到够用就停
    ④ **可长期挂着**（`nohup` / 后台），不占交互

跑法（**长期挂机**）：
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.daemon \
        --target-images 20000 --max-shards 40 --out out/data/curated_danbooru
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

MIRROR = "https://hf-mirror.com"
DATASET = "aipracticecafe/curated-danbooru-2026"

STATE = Path("out/data/_fetch_daemon_state.json")


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            pass
    return {"done_shards": [], "failed": {}, "started": None, "last_error": None}


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_shards() -> List[str]:
    import requests
    s = requests.Session()
    s.proxies.update({"http": os.environ["HTTP_PROXY"], "https": os.environ["HTTPS_PROXY"]})
    r = s.get(f"{MIRROR}/api/datasets/{DATASET}", timeout=30)
    r.raise_for_status()
    return [x["rfilename"] for x in r.json().get("siblings", [])
            if x["rfilename"].endswith(".parquet")]


def fetch_one(shard: str, out_dir: Path, retries: int = 6) -> Dict:
    """下一个分片，**指数退避**重试。⛔ 已有完整文件则秒退（断点续跑）。"""
    from huggingface_hub import hf_hub_download
    target = out_dir / "_shards" / Path(shard).name
    if target.exists() and target.stat().st_size > 0:
        return {"shard": shard, "status": "already", "bytes": target.stat().st_size}
    last = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            p = hf_hub_download(repo_id=DATASET, filename=shard, repo_type="dataset",
                                endpoint=MIRROR, local_dir=str(out_dir / "_shards"))
            dt = time.time() - t0
            sz = os.path.getsize(p)
            _log(f"✅ {Path(shard).name} 完成 {sz/1e6:.0f}MB / {dt:.0f}s "
                 f"({sz/dt/1e6:.2f} MB/s)")
            return {"shard": shard, "status": "ok", "bytes": sz, "sec": round(dt)}
        except Exception as e:                              # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:120]}"
            back = min(2 ** attempt, 60)                      # 2,4,8,16,32,60
            _log(f"⚠️  {Path(shard).name} 第 {attempt}/{retries} 次失败：{last}")
            _log(f"    退避 {back}s 后重试（网络抖动自愈）")
            time.sleep(back)
    return {"shard": shard, "status": "failed", "error": last}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 数据下载守护（断点续传 + 退避重试）")
    ap.add_argument("--out", default="out/data/curated_danbooru")
    ap.add_argument("--max-shards", type=int, default=40, help="最多下几个分片")
    ap.add_argument("--target-images", type=int, default=0,
                    help="下到约多少张就停（0 = 不限，只受 --max-shards 约束）")
    ap.add_argument("--loop", action="store_true", help="下完 max-shards 后再等一轮（常驻）")
    ap.add_argument("--status", action="store_true", help="只打印状态")
    a = ap.parse_args(argv)

    st = _load_state()
    if st.get("started") is None:
        st["started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if a.status:
        print(json.dumps(st, ensure_ascii=False, indent=1))
        return 0

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"守护启动 → {out_dir}（镜像 {MIRROR}）")
    _log(f"已完成的分片: {len(st['done_shards'])} 个")

    try:
        shards = list_shards()
    except Exception as e:                                  # noqa: BLE001
        _log(f"⛔ 列不出分片清单：{type(e).__name__}: {e}")
        _log("   （网络不通 ⇒ 稍后重跑本命令即可续）")
        _save_state(st)
        return 1
    _log(f"该数据集共 {len(shards)} 个分片；本次最多下 {a.max_shards} 个")

    todo = [s for s in shards if s not in st["done_shards"]][:a.max_shards]
    if not todo:
        _log("✅ 没有待下的分片（全部已完成或达到上限）")
        _save_state(st)
        return 0

    n_img = 0
    for i, sh in enumerate(todo, 1):
        _log(f"── [{i}/{len(todo)}] {Path(sh).name}")
        r = fetch_one(sh, out_dir)
        if r["status"] == "ok":
            st["done_shards"].append(sh)
            # 粗估：1 分片 ≈ 500 张（Danbooru 分片经验值，⚠️ 仅用于估算，不做断言）
            n_img += 500
            st["est_images"] = n_img
        elif r["status"] == "failed":
            st["failed"][sh] = r.get("error")
            st["last_error"] = r.get("error")
            _log(f"⛔ 该分片彻底失败：{r.get('error')}")
            _log("   继续下一个（网络可能整体抖动，稍后可重跑续）")
        st["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_state(st)
        if a.target_images and n_img >= a.target_images:
            _log(f"🎯 已达目标 ~{n_img} 张，停。")
            break

    _save_state(st)
    total = sum(p.stat().st_size for p in (out_dir / "_shards").glob("*")
                if p.is_file() and not p.name.startswith("."))
    _log(f"本轮结束：完成 {len(st['done_shards'])} 分片，"
         f"目录内 {_m(total)}MB（断点文件也算）")
    _log("提示：再跑一次本命令即可从断点继续。")
    return 0


def _m(x: int) -> str:
    return f"{x/1e6:.0f}MB"


if __name__ == "__main__":
    raise SystemExit(main())

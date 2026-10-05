"""通用分片下载器 —— 带**可分离的 TUI 监视器**的框架

═══ 为什么做成「框架」而不是一次性脚本 ═══
项目要下的数据集不止一个（Danbooru 37.4GB、还有可能的角色集/字体集），
每次都要「速度 / 进度 / ETA」三样信息⇒ 做成**可复用的**。

═══ 两个进程，一个管道 ═══
    下载进程（本文件，--serve） ──stdout──▶ 监视进程（--tui）
每个进度行是一个 JSON：`{"type":"progress", "file":..., "got":..., "total":..., "speed":...}`
⇒ 好处：**关掉监视器不影响下载**，重开监视器也能接上（读状态文件）。

⚠️ **不使用任何 GUI 库**（tkinter/rich 都不依赖）⇒ 纯 ANSI 转义刷新。

═══ 用法 ═══
    # 终端 1：下载（前台，会持续刷进度）
    python -m kp.data.fetch_tui --serve --max-shards 4

    # 终端 2：监视（另开一个窗口）
    python -m kp.data.fetch_tui --tui
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
STATE_DIR = KP_ROOT / "out" / "data"
STATUS_FILE = STATE_DIR / "_dl_status.json"

MIRROR = "https://hf-mirror.com"
DATASET = "aipracticecafe/curated-danbooru-2026"
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""


# ---------------------------------------------------------------------------
# 状态文件（监视器靠它接上）
# ---------------------------------------------------------------------------
def write_status(**kw) -> None:
    kw["ts"] = time.time()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(kw, ensure_ascii=False), encoding="utf-8")


def read_status() -> dict:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# 人类可读
# ---------------------------------------------------------------------------
def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}PB"


def human_time(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def bar(frac: float, width: int = 28) -> str:
    frac = min(1.0, max(0.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


# ---------------------------------------------------------------------------
# 下载（serve 模式）
# ---------------------------------------------------------------------------
def serve(max_shards: int, out_dir: Path) -> int:
    import requests

    session = requests.Session()
    if PROXY:
        session.proxies.update({"http": PROXY, "https": PROXY})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_shards").mkdir(parents=True, exist_ok=True)

    def emit(**kw) -> None:
        print(json.dumps(kw, ensure_ascii=False), flush=True)
        write_status(**kw)

    # 清单
    try:
        r = session.get(f"{MIRROR}/api/datasets/{DATASET}", timeout=30)
        r.raise_for_status()
        names = [x["rfilename"] for x in r.json().get("siblings", [])
                 if x["rfilename"].endswith(".parquet")]
    except Exception as e:                                      # noqa: BLE001
        print(f"[X] cannot list shards: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    total_files = len(names)
    todo = names[:max_shards]
    emit(type="start", total_shards=total_files, queued=len(todo))
    print(f"[*] {total_files} shards available, downloading {len(todo)}", flush=True)

    grand_bytes = 0
    grand_done = 0
    t_start = time.time()

    for idx, name in enumerate(todo, 1):
        target = out_dir / "_shards" / Path(name).name
        if target.exists() and target.stat().st_size > 0:
            grand_done += target.stat().st_size
            emit(type="skip", file=name, size=target.stat().st_size, index=idx)
            print(f"[skip] [{idx}/{len(todo)}] {Path(name).name} exists", flush=True)
            continue
        url = f"{MIRROR}/datasets/{DATASET}/resolve/main/{name}"
        for attempt in range(1, 7):
            try:
                t0 = time.time()
                resp = session.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                got = 0
                tprev = t0
                spd_prev = 0.0
                tmp = target.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        now = time.time()
                        # ⭐ 每0.3s 报一次（太快会刷爆管道）
                        if now - tprev >= 0.3:
                            spd = (got - spd_prev) / (now - tprev)
                            spd_prev = got
                            tprev = now
                            eta = (total - got) / spd if spd > 1e-6 else -1
                            emit(type="progress", file=Path(name).name,
                                 got=got, total=total, speed=spd, eta=eta,
                                 index=idx, of=len(todo))
                        if total and got >= total:
                            break
                os.replace(tmp, target)# ⭐ 原子替换
                sz = target.stat().st_size
                grand_bytes += sz
                grand_done += sz
                dt = time.time() - t0
                emit(type="done", file=Path(name).name, size=sz,
                     seconds=round(dt, 1), index=idx, of=len(todo))
                print(f"✅ [{idx}/{len(todo)}] {Path(name).name} "
                      f"{human_bytes(sz)} / {dt:.0f}s", flush=True)
                break
            except Exception as e:                              # noqa: BLE001
                back = min(2 ** attempt, 60)
                emit(type="error", file=Path(name).name,
                     error=f"{type(e).__name__}: {str(e)[:120]}", retry=attempt)
                print(f"[!] [{idx}/{len(todo)}] attempt {attempt} failed: "
                      f"{type(e).__name__}, retry in {back}s", flush=True)
                time.sleep(back)
        else:
            emit(type="failed", file=name)
            print(f"[X] [{idx}/{len(todo)}] {name} FAILED", flush=True)

    emit(type="all_done", seconds=round(time.time() - t_start, 1),
         grand_done=grand_done)
    print(f"\n[OK] all done, {human_bytes(grand_done)} this run", flush=True)
    return 0


# ---------------------------------------------------------------------------
# TUI 监视
# ---------------------------------------------------------------------------
def tui(interval: float = 0.4) -> int:
    """纯 ANSI 的监视窗口。可随时Ctrl-C 退出（**不影响下载**）。"""
    try:
        import colorama
        colorama.init()
    except Exception:                                           # noqa: BLE001
        pass
    sys.stdout.write("\x1b[?25l")                            # 隐藏光标
    try:
        while True:
            st = read_status()
            sys.stdout.write("\x1b[H\x1b[2J")                # 清屏 + 回家
            title = "KP Download Monitor"
            print("+- " + title + " " + "-" * max(0, 54 - len(title)))
            if not st:
                print("| waiting for download process...")
                print("| hint: run  python -m kp.data.fetch_tui --serve --max-shards 4"
                      "  in another window")
            else:
                t = st.get("type")
                if t == "start":
                    print(f"| queued {st['queued']} / {st['total_shards']} shards")
                elif t == "progress":
                    got, total = st["got"], st["total"]
                    frac = got / total if total else 0
                    spd = st.get("speed", 0)
                    eta = st.get("eta", -1)
                    print(f"| file: {st['file']}  [{st['index']}/{st['of']}]")
                    print(f"| {bar(frac)} {frac * 100:5.1f}%")
                    print(f"| got {human_bytes(got)} / {human_bytes(total)}")
                    print(f"| speed {human_bytes(spd)}/s"
                          f"   eta {_human(eta) if eta >= 0 else '--'}")
                elif t == "done":
                    print(f"| [OK] {st['file']} "
                          f"{human_bytes(st['size'])} / {st['seconds']}s")
                    print(f"| progress {st['index']}/{st['of']}, next...")
                elif t == "skip":
                    print(f"| [skip] {st['file']} exists ({st['index']}/{st['of']})")
                elif t == "error":
                    print(f"| [!] {st['file']} attempt {st['retry']} failed")
                    print(f"|{st['error']}")
                elif t == "failed":
                    print(f"| [X] {st['file']} FAILED")
                elif t == "all_done":
                    print(f"| [OK] all done! {human_bytes(st.get('grand_done', 0))}"
                          f" / {human_time(st.get('seconds', 0))}")
            age = time.time() - st.get("ts", 0) if st else -1
            if st and age > 3 and t != "all_done":
                print(f"| [!] last update {age:.0f}s ago (downloader may be stopped)")
            print("|" + "-" * 58)
            print(" Ctrl-C to exit monitor (download continues)")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h")                        # 归还光标
        sys.stdout.flush()
    return 0


def _human(sec: float) -> str:
    return human_time(sec)


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="通用分片下载器（带TUI 监视）")
    ap.add_argument("--serve", action="store_true", help="跑下载")
    ap.add_argument("--tui", action="store_true", help="跑监视窗口")
    ap.add_argument("--max-shards", type=int, default=4)
    ap.add_argument("--out", default="out/data/curated_danbooru")
    a = ap.parse_args(argv)
    if a.tui:
        return tui()
    if a.serve:
        return serve(a.max_shards, KP_ROOT / a.out)
    ap.print_help()
    print("\nusage: --serve (download)  or  --tui (monitor)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

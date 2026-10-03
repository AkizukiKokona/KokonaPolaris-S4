"""KokonaPolaris E4 · 下载 G1 靶子模型（Sana 1.6B）

选型理由：Sana 1.6B 与 KP 设计同源 —— DC-AE 32× 压缩（KP 是 32× 混合 latent）、
线性注意力（KP 是 3:1 gated linear）、1.6B（KP-M 档）、Apache 2.0。
仓库 22GB 中大半是重复格式（fp32 分片 / bf16 分片 / int4 各一份），此处按需精选。

运行：source ./env.sh && "$KP_PY" tools/fetch_sana.py --wave both
特性：**逐文件下载 + 重试 + 断点续传**，两波可分开跑，重跑即续传。

═══ 本机网络三条实测（2026-10-03，换机后重测，勿沿用旧结论）═══
① **直连 huggingface.co 不可用**：`curl` 报
   `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`，
   PowerShell `Invoke-WebRequest` 报 SSL 失败 ⇒ 只有 **Python/certifi 栈**能通。
② **旧记忆「代理直连比 hf-mirror 快 26×（86MB/s）」是迁出机的数字，本机不成立**。
   本机裸 HTTP 测速：hf-mirror 不走代理 2.11 / huggingface.co 走代理 2.25 /
   **hf-mirror 走代理 3.07 MB/s**。⇒ 一定要**自己重测**，别信记忆里的倍数。
③ ⚠️ **但「镜像更快」只对裸 HTTP 成立**：`snapshot_download` 走 hf-mirror 时
   `huggingface_hub` 报 `FileMetadataError: Distant resource does not seem to be on
   huggingface.co`（镜像未回 hub 需要的元数据头）。
   ⇒ 本脚本仍用**默认 endpoint（huggingface.co）+ 代理**。
④ **真正的故障是长连接被中断**（`IncompleteRead: 497MB read, 2.7GB more expected`），
   不是通道选错 ⇒ 故改成**逐文件 + 重试**，每次重试都从 `.incomplete` 分片续传。
⑤ ⚠️ **探测通道时别信 `os.environ["HF_ENDPOINT"]` 的运行时赋值**：
   `huggingface_hub.constants.ENDPOINT` 是 **import 时读进常量的**，
   运行时改环境变量**不生效** ⇒ 会得出「所有通道都通」的假结论（本机踩过）。
"""

import sys
from pathlib import Path

# ⚠️ 入口自举：直接 `python tools/fetch_sana.py` 时 `sys.path[0]` 是 **tools/** 而非仓库根
#    ⇒ `import kp.paths` 报 ModuleNotFoundError；照 tools/onboard.py:18，⛔ 不写死绝对路径
#    （注意：下面 _download_worker 里那行 sys.path.insert 是给子进程用的，与这里无关）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.paths import MODELS_SANA  # noqa: E402
import argparse
import os
import sys
import time
from huggingface_hub import HfApi, hf_hub_url

REPO = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
DEST = str(MODELS_SANA)

# 只要真正需要的：bf16 主干 + bf16 VAE + 官方 int4 参照 + 全部配置/分词器
# 不下：transformer fp32 分片(6.1GB)、text_encoder fp32/bf16(5.0GB)、vae fp32(1.2GB)
PATTERNS = [
    "transformer/diffusion_pytorch_model.bf16.safetensors",   # 3060 MB 主靶子
    "transformer/diffusion_pytorch_model.int4.safetensors",   # 1220 MB 官方 4bit 参照
    "transformer/config.json",
    "vae/diffusion_pytorch_model.bf16.safetensors",           # 1191 MB DC-AE(32×)
    "vae/config.json",
    "tokenizer/*",
    "scheduler/*",
    "model_index.json",
    "LICENSE",
    "README.md",
]

# 第二波（需要真实生成时再下）：text_encoder gemma-2-2b bf16 = 4986 MB
PATTERNS_WAVE2 = [
    "text_encoder/model.bf16-00001-of-00002.safetensors",
    "text_encoder/model.bf16-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.bf16.json",
    "text_encoder/config.json",
]

# ---- 超时参数（在函数定义之前，因为用作默认值）----
STALL_TIMEOUT = 60.0    # .part 多少秒无增长 => 判定卡死并 kill 子进程
READ_TIMEOUT = 20.0     # 子进程单次 socket 读超时
RETRIES = 60             # 允许很多次续传式尝试
RETRY_WAIT = 3.0         # 重试基础等待（线性递增）

# 文件名 → 期望字节数（拿到 repo 元数据后填充；用于「已完成」判定）
size_map: dict = {}


def _download_worker(url: str, part: str, offset: int) -> int:
    """子进程入口：**只做一件事** —— 从 `offset` 续传，写进 `part`，然后退出。

    退出码：0=正常读完；1=异常。父进程靠 `.part` 大小判断进度与死锁。

    ⚠️ 必须放在**独立子进程**里做，原因是本机踩到的三件事：
      1. `hf_hub_download` / `snapshot_download` 会**静默卡死**（TCP Established、
         零字节、无读超时）⇒ 永远不返回，外层 try/except **根本没机会触发**。
      2. 自写「停滞看门狗」也治不了 —— 线程 `close()` 抢不到控制权。
      3. ⭐ `signal.SIGALRM` 在 **Windows 的 Python 里根本不存在** ⇒
         `AttributeError: module 'signal' has no attribute 'SIGALRM'`，
         每次尝试**瞬间失败**（实测重试 40 次全是同一个错，`.part` 原地不动）。
         我当时只验证了"能编译"、没验证"能跑"，还把**自己代码的报错**
         误判成了**网络劣化**。
    ⇒ 唯一跨平台可靠解：**父进程按 `.part` 增长判断活性，卡住就 `kill` 子进程**
      —— `kill` 是操作系统级的，不需要目标进程配合。
    """
    import urllib.request
    req = urllib.request.Request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
        with open(part, "ab" if offset else "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
                f.flush()
    return 0


def download_one(rel_path: str, dest_root: str, *, retries: int = RETRIES,
                 retry_wait: float = RETRY_WAIT, stall_timeout: float = STALL_TIMEOUT,
                 chunk: int = 1 << 20) -> bool:
    """单文件下载：**子进程拉取 + 父进程按 `.part` 增长监管 + 卡死即 kill + Range 续传**。"""
    import subprocess
    import urllib.request

    final = os.path.join(dest_root, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(final), exist_ok=True)
    part = final + ".part"
    url = hf_hub_url(REPO, rel_path)
    total = size_map.get(rel_path)

    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have >= total:
            os.replace(part, final)
            return True
        code = (
            "import sys;sys.path.insert(0,%r);"
            "from tools.fetch_sana import _download_worker as w;"
            "sys.exit(w(%r,%r,%d))"
            % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               url, part, have)
        )
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:  # noqa: BLE001
            print(f"        {rel_path} 无法启动子进程：{e}", flush=True)
            time.sleep(retry_wait)
            continue

        # ---- 监管：按 .part 增长判活性，卡住就 kill ----
        last_size, last_t, killed = have, time.time(), False
        while proc.poll() is None:
            time.sleep(2.0)
            cur = os.path.getsize(part) if os.path.exists(part) else 0
            if cur > last_size:
                last_size, last_t = cur, time.time()
                print(f"        … {rel_path.split('/')[-1]} {cur/2**20:.0f} MB", flush=True)
            elif time.time() - last_t > stall_timeout:
                print(f"        ⛔ {rel_path} 停滞 {stall_timeout:.0f}s（.part 无增长）"
                      f" ⇒ kill 子进程重连", flush=True)
                proc.kill()
                killed = True
                break
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass

        size = os.path.getsize(part) if os.path.exists(part) else 0
        gained = size - have
        if total and size >= total:
            os.replace(part, final)
            print(f"        ✅ {rel_path} 完成（{size/2**20:.0f} MB）", flush=True)
            return True
        print(f"        {rel_path} 第 {attempt} 次结束：本次 +{gained/2**20:.1f} MB"
              f"（累计 {size/2**20:.0f}{'' if not total else f'/{total/2**20:.0f}'} MB）"
              f"{'，被 kill' if killed else ''}", flush=True)
        if attempt < retries:
            time.sleep(retry_wait)
    return False


ap = argparse.ArgumentParser(description="下载 G1 靶子 Sana 1.6B")
ap.add_argument("--wave", choices=["1", "2", "both"], default="both",
                help="1=主干+VAE+tokenizer（5.4GB）；2=text_encoder gemma-2-2b（4.9GB，出图必需）；both=依次")
ap.add_argument("--retries", type=int, default=6, help="单文件最大重试次数")
ap.add_argument("--retry-wait", type=float, default=15.0, help="重试基础等待秒数（线性递增）")
ap.add_argument("--stall-timeout", type=float, default=STALL_TIMEOUT, help=".part 多少秒无增长即 kill 子进程")
a = ap.parse_args()
RETRIES, RETRY_WAIT = a.retries, a.retry_wait
STALL_TIMEOUT = a.stall_timeout
WAVES = {"1": PATTERNS, "2": PATTERNS_WAVE2}
ORDER = ["1", "2"] if a.wave == "both" else [a.wave]

api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
info = api.model_info(REPO, files_metadata=True)
# 填充 size_map（download_one 用它判断「是否已下完」）
size_map.update({f.rfilename: (f.size or 0) for f in info.siblings})

def size_of(pats):
    tot = 0
    for f in info.siblings:
        for p in pats:
            if f.rfilename == p or (p.endswith("*") and f.rfilename.startswith(p[:-1])):
                tot += (f.size or 0)
                break
    return tot

print(f"仓库: {REPO}")
print(f"落地: {DEST}")
print(f"第一波 {size_of(PATTERNS)/2**30:.2f} GB  /  第二波(text_encoder) {size_of(PATTERNS_WAVE2)/2**30:.2f} GB")
print(f"本次下载: wave={a.wave} → {ORDER}")
print("-" * 68)

for w in ORDER:
    print(f"\n>>> 开始第 {w} 波（逐文件重试 + 断点续传）...", flush=True)
    files = [f.rfilename for f in info.siblings
             if any(f.rfilename == p or (p.endswith("*") and f.rfilename.startswith(p[:-1]))
                    for p in WAVES[w])]
    # ⭐ 按**文件大小降序**下：大文件先下（长连接失败率随持续时间上升，
    #    把大件放在连接最"新鲜"的时候；小文件最后，失败也容易重来）
    size_of_file = {f.rfilename: (f.size or 0) for f in info.siblings}
    files = sorted(files, key=lambda f: -size_of_file.get(f, 0))
    print(f"    本波 {len(files)} 个文件（按大小降序）", flush=True)
    failed = []
    for i, fn in enumerate(files, 1):
        size_mb = size_of_file.get(fn, 0)
        final_p = os.path.join(DEST, fn.replace("/", os.sep))
        # ⭐ 已完成则跳过 —— 否则 `--wave both` 重跑会把已下好的大件再下一遍
        #    （旧版就踩过：5.4GB 第一波下完后重跑，白等了 5 分钟）
        if os.path.exists(final_p) and os.path.getsize(final_p) == size_mb:
            print(f"    [{i}/{len(files)}] ⏭  {fn}  ({size_mb/2**20:.0f} MB) 已完整，跳过",
                  flush=True)
            continue
        t0 = time.time()
        ok = download_one(fn, DEST, retries=RETRIES, retry_wait=RETRY_WAIT,
                          stall_timeout=STALL_TIMEOUT)
        dt = time.time() - t0
        if ok:
            spd = (size_mb / 2 ** 20) / dt if dt > 0 else 0
            print(f"    [{i}/{len(files)}] ✅ {fn}  ({size_mb/2**20:.0f} MB, "
                  f"{dt:.0f}s, {spd:.2f} MB/s)", flush=True)
        else:
            failed.append(fn)
            print(f"    [{i}/{len(files)}] ❌ {fn} —— {RETRIES} 次均失败，跳过", flush=True)
    if failed:
        print(f"\n⚠️ 第 {w} 波有 {len(failed)} 个文件失败（**重跑本脚本即可从 .part 续传**）：")
        for f in failed:
            print(f"      {f}")
        print("   提示：失败多为长连接被中断或静默死连接，重跑会从 .part 续传。")
    else:
        print(f"✅ 第 {w} 波全部完成", flush=True)

print("\n=== 已下载文件核对 ===")
tot = 0
for root, _, files in os.walk(DEST):
    for fn in files:
        if fn.endswith(".cache") or "/.cache/" in root.replace("\\", "/"):
            continue
        fp = os.path.join(root, fn)
        sz = os.path.getsize(fp)
        tot += sz
        if sz > 1024 * 1024:
            rel = os.path.relpath(fp, DEST).replace("\\", "/")
            print(f"  {sz/2**20:>9.1f} MB  {rel}")
print(f"  ---- 合计 {tot/2**30:.2f} GB")

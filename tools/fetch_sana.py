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
from kp.paths import MODELS_SANA
import argparse
import os
import time
from huggingface_hub import HfApi, hf_hub_download

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

ap = argparse.ArgumentParser(description="下载 G1 靶子 Sana 1.6B")
ap.add_argument("--wave", choices=["1", "2", "both"], default="both",
                help="1=主干+VAE+tokenizer（5.4GB）；2=text_encoder gemma-2-2b（4.9GB，出图必需）；both=依次")
ap.add_argument("--retries", type=int, default=4, help="单文件最大重试次数")
ap.add_argument("--retry-wait", type=float, default=20.0, help="重试基础等待秒数（线性递增）")
a = ap.parse_args()
RETRIES, RETRY_WAIT = a.retries, a.retry_wait
WAVES = {"1": PATTERNS, "2": PATTERNS_WAVE2}
ORDER = ["1", "2"] if a.wave == "both" else [a.wave]

api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
info = api.model_info(REPO, files_metadata=True)

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
    print(f"    本波 {len(files)} 个文件", flush=True)
    failed = []
    for i, fn in enumerate(sorted(files), 1):
        size_mb = next((f.size for f in info.siblings if f.rfilename == fn), 0) or 0
        ok = False
        for attempt in range(1, RETRIES + 1):
            try:
                hf_hub_download(repo_id=REPO, filename=fn, local_dir=DEST)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                wait = RETRY_WAIT * attempt
                print(f"    [{i}/{len(files)}] {fn} 第 {attempt} 次失败："
                      f"{type(e).__name__}: {str(e)[:110]}", flush=True)
                if attempt < RETRIES:
                    print(f"        {wait}s 后重试（已下载分片会续传）...", flush=True)
                    time.sleep(wait)
        if ok:
            print(f"    [{i}/{len(files)}] ✅ {fn}  ({size_mb/2**20:.0f} MB)", flush=True)
        else:
            failed.append(fn)
            print(f"    [{i}/{len(files)}] ❌ {fn} —— {RETRIES} 次均失败，跳过", flush=True)
    if failed:
        print(f"\n⚠️ 第 {w} 波有 {len(failed)} 个文件失败（**重跑本脚本即可续传**）：")
        for f in failed:
            print(f"      {f}")
        print("   提示：失败多为长连接被中断（IncompleteRead），重跑会从分片续传。")
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

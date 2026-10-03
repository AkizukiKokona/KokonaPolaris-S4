"""KokonaPolaris E4 · 下载 G1 靶子模型（Sana 1.6B）

选型理由：Sana 1.6B 与 KP 设计同源 —— DC-AE 32× 压缩（KP 是 32× 混合 latent）、
线性注意力（KP 是 3:1 gated linear）、1.6B（KP-M 档）、Apache 2.0。
仓库 22GB 中大半是重复格式（fp32 分片 / bf16 分片 / int4 各一份），此处按需精选。

运行：source /d/model/env.sh && "$KP_PY" tools/fetch_sana.py
特性：可重复运行（增量补齐，断点续传）。
"""
from kp.paths import MODELS_SANA, OUT
import os
from huggingface_hub import snapshot_download, HfApi

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
print("-" * 68)
print("开始下载第一波（可中断后续传）...")

path = snapshot_download(
    repo_id=REPO,
    local_dir=DEST,
    allow_patterns=PATTERNS,
    max_workers=4,
)
print(f"\n✅ 完成，落地于: {path}")

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

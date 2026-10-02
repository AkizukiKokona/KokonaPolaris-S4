"""KokonaPolaris E5 探针 · 摸清 Sana 接口与显存账，为 PTQ 对照做准备

要回答的问题：
  1. SanaPipeline.encode_prompt 的输出形状（缓存 text embedding 用）
  2. SanaTransformer2DModel.forward 的真实签名（构造 forward_loop 用）
  3. transformer 单独驻卡吃掉多少显存
  4. 单次 forward 的耗时（估算校准时间）

只读探测，不写权重。
运行：source /d/model/env.sh && "$KP_PY" tools/e5_probe.py
"""
import os, time, inspect, torch
from diffusers import SanaPipeline, SanaTransformer2DModel

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"

print("=" * 74)
print("[1] SanaPipeline.encode_prompt 签名")
from diffusers.pipelines.sana.pipeline_sana import SanaPipeline as SP
print("   ", inspect.signature(SP.encode_prompt))

print("\n[2] SanaTransformer2DModel.forward 签名")
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel as ST
sig = inspect.signature(ST.forward)
for n, p in sig.parameters.items():
    if n == "self":
        continue
    print(f"    {n:<32} {p.annotation if p.annotation != inspect.Parameter.empty else ''}")

print("\n[3] 加载 pipeline（CPU，仅取 text_encoder）并缓存一条 embedding")
pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")
print(f"    pipeline 组件: {list(pipe.components.keys())}")

# 只看 encode_prompt 的接口
with torch.no_grad():
    out = pipe.encode_prompt(
        prompt="a cat sitting on a wooden table",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        clean_caption=False,
    )
print(f"    encode_prompt 返回 {len(out)} 项")
for i, o in enumerate(out):
    if torch.is_tensor(o):
        print(f"      [{i}] tensor {tuple(o.shape)} {o.dtype}  "
              f"min={o.min():.3f} max={o.max():.3f}")
    else:
        print(f"      [{i}] {type(o).__name__} = {o}")

# 释放 text encoder，测 transformer 单独驻卡
del pipe
import gc; gc.collect()

print("\n[4] transformer 单独加载到 GPU 的显存占用")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
free0, total = torch.cuda.mem_get_info()
print(f"    加载前可用 {free0/2**30:.2f} / {total/2**30:.2f} GB")
t = time.time()
tf = SanaTransformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
).to("cuda").eval()
torch.cuda.synchronize()
print(f"    ✅ 加载 {time.time()-t:.1f}s  权重占用 {torch.cuda.memory_allocated()/2**30:.2f} GB")
free1, _ = torch.cuda.mem_get_info()
print(f"    加载后可用 {free1/2**30:.2f} GB")

print("\n[5] 单次 forward 冒烟 + 耗时")
c = tf.config
L = 64  # 文本序列长度占位
hidden = torch.randn(1, c.in_channels, c.sample_size, c.sample_size,
                     dtype=torch.bfloat16, device="cuda")
emb = torch.randn(1, L, c.caption_channels, dtype=torch.bfloat16, device="cuda")
mask = torch.ones(1, L, dtype=torch.bfloat16, device="cuda")
ts = torch.tensor([500.0], device="cuda")

with torch.no_grad():
    t = time.time()
    y = tf(hidden_states=hidden, encoder_hidden_states=emb,
           encoder_attention_mask=mask, timestep=ts, return_dict=False)[0]
    torch.cuda.synchronize()
print(f"    ✅ 输出 {tuple(y.shape)} {y.dtype}  单次 {time.time()-t:.3f}s")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    y = tf(hidden_states=hidden, encoder_hidden_states=emb,
           encoder_attention_mask=mask, timestep=ts, return_dict=False)[0]
    torch.cuda.synchronize()
print(f"    forward 峰值显存 {torch.cuda.max_memory_allocated()/2**30:.2f} GB "
      f"(含权重 {(torch.cuda.max_memory_allocated())/2**30:.2f})")
print("=" * 74)

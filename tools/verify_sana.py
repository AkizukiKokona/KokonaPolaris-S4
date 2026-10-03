"""KokonaPolaris E4 验收：Sana 靶子模型可加载性 + 结构核对
只加载 transformer 与 vae（CPU, bf16），不占显存。
运行：source /d/model/env.sh && "$KP_PY" tools/verify_sana.py
"""
from kp.paths import MODELS_SANA, OUT
import os, json, torch
from diffusers import SanaTransformer2DModel, AutoencoderDC

MODEL = str(MODELS_SANA)

print("=" * 70)
print("model_index.json")
mi = json.load(open(os.path.join(MODEL, "model_index.json")))
for k, v in mi.items():
    print(f"  {k:<18} {v}")

print("\n[1] SanaTransformer2DModel 加载")
tf = SanaTransformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16)
n_tf = sum(p.numel() for p in tf.parameters())
c = tf.config
hidden = c.num_attention_heads * c.attention_head_dim
print(f"    ✅ 参数量 {n_tf/1e9:.4f} B")
print(f"    num_layers={c.num_layers}   hidden={c.num_attention_heads}heads × {c.attention_head_dim} = {hidden}")
print(f"    patch_size={c.patch_size}   in_ch={c.in_channels}   sample_size={c.sample_size}")
print(f"    caption_channels={c.caption_channels}   cross_attention_dim={c.cross_attention_dim}")
print(f"    mlp_ratio={c.mlp_ratio}   norm_eps={c.norm_eps}")
print(f"    参数 dtype: {next(tf.parameters()).dtype}")

print("\n[2] AutoencoderDC 加载（33× 空间压缩的核心）")
vae = AutoencoderDC.from_pretrained(
    MODEL, subfolder="vae", variant="bf16", torch_dtype=torch.bfloat16)
n_vae = sum(p.numel() for p in vae.parameters())
vc = vae.config
print(f"    ✅ 参数量 {n_vae/1e6:.2f} M")
print(f"    latent_channels={getattr(vc,'latent_channels','n/a')}")
print(f"    encoder_block_out_channels={getattr(vc,'encoder_block_out_channels','n/a')}")
print(f"    scaling_factor={getattr(vc,'scaling_factor','n/a')}")

print("\n[3] 实测压缩率（1024² 图像 → latent 尺寸）")
with torch.no_grad():
    x = torch.randn(1, 3, 1024, 1024, dtype=torch.bfloat16)
    z = vae.encode(x).latent
print(f"    输入 {tuple(x.shape)}  →  latent {tuple(z.shape)}")
h, w = z.shape[-2:]
print(f"    ✅ 空间压缩 = {1024/h:.1f}×  →  1024² 图 = {h*w} 个 latent 位置")

print("\n[4] 对照：KP 设计目标")
print(f"    设计: 1024² → 1024 token（32× 压缩）")
print(f"    实测 Sana: 1024² → {h*w} 位置 × {z.shape[1]} ch")
print(f"    → GPT 式算 token 数（patch={c.patch_size}）: {h*w//(c.patch_size**2)}")
print("=" * 70)

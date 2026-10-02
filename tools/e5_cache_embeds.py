"""KokonaPolaris E5 · 步骤1：离线缓存文本 embedding

动机（显存第一性原理）：
  text_encoder(Gemma2-2B, bf16) 5.2GB + transformer 3.06GB = 8.3GB > 可用 6.3GB
  → 整条 pipeline 无法同时驻卡。
  解法：把文本侧**离线算好、落盘**。之后所有量化实验只需 transformer + VAE。

产出：D:/model/out/e5/embeds.pt
      {name: {"pos": (1,300,2304), "pos_mask": (1,300),
              "neg": (1,300,2304), "neg_mask": (1,300)}}
运行：source /d/model/env.sh && "$KP_PY" tools/e5_cache_embeds.py
"""
import os, torch, json
from diffusers import SanaPipeline

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"
OUT = "D:/model/out/e5"
os.makedirs(OUT, exist_ok=True)

# 与 E4b 基线完全相同的 prompt 集（保证可比）
PROMPTS = [
    ("01_en_scene",
     "a cat sitting on a wooden table beside a window, soft morning light, "
     "detailed fur, shallow depth of field, photorealistic"),
    ("02_zh_text",
     "一张简洁的海报，深蓝色背景，画面正中央用白色粗体写着「心夏北极星」四个字，"
     "下方一行小字写着 KokonaPolaris-S4"),
    ("03_anime",
     "1girl, silver long hair, blue eyes, sailor school uniform, cherry blossoms, "
     "spring, anime illustration, clean lineart, vibrant colors"),
]
NEG = ""

print("=" * 74)
print("[1] 加载 pipeline（CPU）——只为拿 tokenizer + text_encoder")
pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")

store = {}
for name, prompt in PROMPTS:
    print(f"\n[2] encode: {name}")
    with torch.no_grad():
        pe, pm, ne, nm = pipe.encode_prompt(
            prompt=prompt, negative_prompt=NEG,
            do_classifier_free_guidance=True, num_images_per_prompt=1,
            device=torch.device("cpu"), clean_caption=False,
        )
    tcpu = lambda t: t.detach().to("cpu").clone()
    store[name] = {"pos": tcpu(pe), "pos_mask": tcpu(pm),
                   "neg": tcpu(ne), "neg_mask": tcpu(nm),
                   "prompt": prompt}
    # 有效 token 数 = mask 为 1 的位置
    n_tok = int(pm.sum().item())
    print(f"    pos {tuple(pe.shape)}  有效 token {n_tok} / {pe.shape[1]}"
          f"  ({n_tok/pe.shape[1]*100:.1f}% 非填充)")
    print(f"    文本 300 token  vs  图像 1024 token"
          f"  → 文本占序列 {300/(300+1024)*100:.1f}%")

fp = os.path.join(OUT, "embeds.pt")
torch.save(store, fp)
print(f"\n[3] ✅ 已缓存 → {fp}  ({os.path.getsize(fp)/2**20:.1f} MB)")
print("=" * 74)

"""KokonaPolaris E4b · Sana 1.6B bf16 基线出图（G1 的 BF16 参照臂）

显存现实：transformer 3.2G + text_encoder(Gemma2-2B) 5.2G + vae 0.62G ≈ 9.0G > 可用 6.84G
→ 采用 enable_model_cpu_offload()，任一时刻只有单个组件驻卡。

产出：D:/model/out/e4b_bf16/*.png + baseline_log.json（供 G1 复现与对照）
运行：source /d/model/env.sh && "$KP_PY" tools/sana_bf16_baseline.py [图片数]
"""
from kp.paths import MODELS_SANA, OUT
import os, sys, json, time, torch
from diffusers import SanaPipeline

MODEL = str(MODELS_SANA)
OUT = OUT / "e4b_bf16"
SEED = 42
STEPS = 20
GUIDANCE = 4.5

# 覆盖三类用途，为 G1 之后的重建质量对照留锚点
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

os.makedirs(OUT, exist_ok=True)

def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30

print("=" * 72)
print("[1] 加载 pipeline（bf16, CPU offload）")
t0 = time.time()
try:
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")
    print("    加载方式: variant='bf16'")
except Exception as e:
    print(f"    ⚠️ variant 方式失败({type(e).__name__})，回退默认加载")
    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
print(f"    ✅ 加载完成 {time.time()-t0:.1f}s")

pipe.enable_model_cpu_offload()
print("    ✅ 已启用 model cpu offload")

print("\n[2] 生成")
free, total = torch.cuda.mem_get_info()
print(f"    生成前可用显存 {free/2**30:.2f} / {total/2**30:.2f} GB")

n = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
log = {"model": MODEL, "seed": SEED, "steps": STEPS, "guidance": GUIDANCE,
       "resolution": 1024, "variant": "bf16(cpu-offload)", "runs": []}

for i, (name, prompt) in enumerate(PROMPTS[:n]):
    print(f"\n    --- [{i+1}/{min(n,len(PROMPTS))}] {name} ---")
    print(f"        prompt: {prompt[:70]}...")
    g = torch.Generator("cpu").manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        img = pipe(prompt, height=1024, width=1024,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   generator=g).images[0]
    except torch.cuda.OutOfMemoryError as e:
        print(f"        ❌ 显存不足: {e}")
        torch.cuda.empty_cache()
        continue
    dt = time.time() - t
    fp = os.path.join(OUT, f"{name}.png")
    img.save(fp)
    pk = peak_gb()
    print(f"        ✅ {dt:.1f}s  峰值显存 {pk:.2f} GB  → {fp}")
    log["runs"].append({"name": name, "prompt": prompt, "seconds": round(dt, 1),
                        "peak_gb": round(pk, 2), "file": fp})

with open(os.path.join(OUT, "baseline_log.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
print(f"\n[3] 日志已写入 {OUT}/baseline_log.json")
print("=" * 72)

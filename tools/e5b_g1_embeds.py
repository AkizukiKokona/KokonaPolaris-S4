"""E5b · G1 判决用固定 prompt 集（≥10 条）的文本 embedding 缓存。

产出 D:/model/out/e5b/g1_embeds.pt
  {name: {"pos":(1,300,2304), "pos_mask":(1,300), "neg":..., "neg_mask":..., "prompt":str}}

运行：source /d/model/env.sh && "$KP_PY" tools/e5b_g1_embeds.py
"""
from kp.paths import MODELS_SANA, OUT
import os, json, torch
from diffusers import SanaPipeline

MODEL = str(MODELS_SANA)
OUT = OUT / "e5b"
os.makedirs(OUT, exist_ok=True)

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
    ("04_en_portrait",
     "close-up portrait of an elderly fisherman, weathered skin, grey beard, "
     "stormy ocean behind him, dramatic side light, 85mm photo"),
    ("05_zh_landscape",
     "中国江南水乡，小桥流水，白墙黛瓦的民居，清晨薄雾，河面倒影，水墨画风格"),
    ("06_en_food",
     "a bowl of steaming ramen with soft boiled egg and chashu, top down view, "
     "food photography, warm light, shallow depth of field"),
    ("07_en_architecture",
     "modern glass skyscraper at dusk, curved facade, reflections of orange sky, "
     "low angle, architectural photography, ultra sharp"),
    ("08_zh_animal",
     "一只橘色虎斑猫躺在阳光斜照的木地板上，半闭着眼睛，毛发细节清晰，浅景深"),
    ("09_en_abstract",
     "abstract fluid art, swirling teal and gold ink, marbled texture, "
     "high detail, digital painting"),
    ("10_en_cyberpunk",
     "neon lit city street at night in the rain, cyberpunk, wet asphalt reflections, "
     "purple and cyan signs, cinematic"),
    ("11_zh_text2",
     "一张极简菜单，纯白背景，黑色手写体写着「今日特供」，下方写着「咖啡 15元」，排版干净"),
    ("12_en_macro",
     "macro photograph of a dragonfly resting on a green leaf, dew drops, "
     "iridescent wings, extreme detail, bokeh background"),
]
NEG = ""

print("=" * 74)
print(f"[1] 加载 pipeline（CPU）以拿 text_encoder；prompt 数 = {len(PROMPTS)}")
pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")

store = {}
for name, prompt in PROMPTS:
    with torch.no_grad():
        pe, pm, ne, nm = pipe.encode_prompt(
            prompt=prompt, negative_prompt=NEG,
            do_classifier_free_guidance=True, num_images_per_prompt=1,
            device=torch.device("cpu"), clean_caption=False,
        )
    t = lambda x: x.detach().to("cpu").clone()  # noqa: E731
    store[name] = {"pos": t(pe), "pos_mask": t(pm), "neg": t(ne), "neg_mask": t(nm),
                   "prompt": prompt}
    print(f"    {name}: pos {tuple(pe.shape)} 有效 token {int(pm.sum().item())}", flush=True)

fp = os.path.join(OUT, "g1_embeds.pt")
torch.save(store, fp)
with open(os.path.join(OUT, "g1_prompts.json"), "w", encoding="utf-8") as f:
    json.dump({k: v["prompt"] for k, v in store.items()}, f, ensure_ascii=False, indent=2)
print(f"[2] ✅ 已缓存 {len(store)} 条 → {fp} ({os.path.getsize(fp)/2**20:.1f} MB)")

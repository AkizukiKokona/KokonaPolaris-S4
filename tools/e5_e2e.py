"""KokonaPolaris E5 · 步骤3（端到端判据）：量化后真出图，与 bf16 逐像素对照

为什么还需要这一步：
  步骤2 量的是「单步 noise_pred 偏了多少」——是信号级证据。
  但 G1 的真判据是**图像级**：人眼看得见吗？本脚本给出可看的图 + PSNR/MAE。

显存方案（关键）：
  text_encoder 5.2G 不驻卡 —— 用步骤1 缓存好的 prompt_embeds 绕过 encode_prompt。
  卡上只需 transformer(3.06G) + VAE(0.62G)，实测可容纳。

运行：source /d/model/env.sh && "$KP_PY" tools/e5_e2e.py [arms]
  arms 例: bf16,W4A4,W4A8   默认全部
"""
from kp.paths import MODELS_SANA, OUT
import os, sys, gc, copy, json, time, torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
import modelopt.torch.quantization as mtq
from diffusers import SanaPipeline

MODEL = str(MODELS_SANA)
OUT = OUT / "e5"
IMG = os.path.join(OUT, "images")
os.makedirs(IMG, exist_ok=True)
SEED, STEPS, GUIDANCE = 42, 20, 4.5
PROMPT_KEYS = ["01_en_scene", "02_zh_text", "03_anime"]

store = torch.load(os.path.join(OUT, "embeds.pt"), map_location="cpu")

def set_weight_block(cfg, blk):
    cfg = copy.deepcopy(cfg)
    for entry in cfg["quant_cfg"]:
        c = entry.get("cfg")
        if not (isinstance(c, dict) and isinstance(c.get("block_sizes"), dict)):
            continue
        c["block_sizes"] = {
            (-1 if k in (-1, "-1") else k): (blk if k in (-1, "-1") else v)
            for k, v in c["block_sizes"].items()
        }
    return cfg

ALL_ARMS = {"bf16": None, "W4A4": mtq.NVFP4_DEFAULT_CFG, "W4A16": mtq.W4A16_NVFP4_CFG,
            "W4A8": mtq.W4A8_NVFP4_FP8_CFG,
            "W4A8_b16": set_weight_block(mtq.W4A8_NVFP4_FP8_CFG, 16)}
sel = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALL_ARMS)
ARMS = [(a, ALL_ARMS[a]) for a in sel if a in ALL_ARMS]

print("=" * 78)
print(f"[1] 加载 pipeline（bf16），arms = {[a for a,_ in ARMS]}")
def fresh_pipe():
    p = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")
    # text_encoder 不需要驻卡（用缓存 embeds 绕过）
    p.text_encoder = None
    return p

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))

def gen(pipe, key):
    e = store[key]
    g = torch.Generator("cpu").manual_seed(SEED)
    t0 = time.time()
    img = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=e["pos"].cuda().to(torch.bfloat16),
        prompt_attention_mask=e["pos_mask"].cuda(),
        negative_prompt_embeds=e["neg"].cuda().to(torch.bfloat16),
        negative_prompt_attention_mask=e["neg_mask"].cuda(),
        height=1024, width=1024, num_inference_steps=STEPS,
        guidance_scale=GUIDANCE, generator=g,
    ).images[0]
    return img, time.time() - t0

results = {}
golden_img = {}
calib_cases = None

for arm, cfg in ARMS:
    print(f"\n[2] === {arm} ===")
    pipe = fresh_pipe()
    # ⚠️ 必须先上卡再量化：ModelOpt 的校准循环喂 cuda 张量，
    #    若权重还在 CPU 会报 "Input type CUDABFloat16Type / weight type CPUBFloat16Type"
    pipe.transformer.to("cuda").eval()
    pipe.vae.to("cuda").eval()
    if cfg is not None:
        # 校准：用真实 embeds + 随机 latent 走若干步
        def _calib(model, _store=store):
            with torch.no_grad():
                for k in PROMPT_KEYS:
                    e = _store[k]
                    for t in [999.0, 500.0, 100.0]:
                        x = torch.randn(1, 32, 32, 32, generator=torch.Generator().manual_seed(0))
                        model(hidden_states=x.to(torch.bfloat16).cuda(),
                              encoder_hidden_states=e["pos"].cuda().to(torch.bfloat16),
                              timestep=torch.tensor([t], device="cuda"),
                              return_dict=False)
        t0 = time.time()
        mtq.quantize(pipe.transformer, cfg, forward_loop=_calib)
        nq = sum(1 for m in pipe.transformer.modules() if "QuantLinear" in type(m).__name__)
        print(f"    量化 {time.time()-t0:.1f}s  QuantLinear={nq}")

    results[arm] = []
    torch.cuda.reset_peak_memory_stats()
    for k in PROMPT_KEYS:
        try:
            img, dt = gen(pipe, k)
        except torch.cuda.OutOfMemoryError as e:
            print(f"    ❌ OOM on {k}: {e}")
            torch.cuda.empty_cache()
            results[arm].append({"key": k, "oom": True})
            continue
        arr = np.asarray(img.convert("RGB"))
        fp = os.path.join(IMG, f"{arm}__{k}.png")
        img.save(fp)
        rec = {"key": k, "seconds": round(dt, 1), "file": fp}
        if k in golden_img:
            rec["psnr"] = psnr(arr, golden_img[k])
            rec["ssim"] = float(ssim_fn(arr, golden_img[k], channel_axis=2, data_range=255))
            rec["mae"] = float(np.mean(np.abs(arr.astype(np.int16) - golden_img[k].astype(np.int16))))
            rec["maxdiff"] = int(np.abs(arr.astype(np.int16) - golden_img[k].astype(np.int16)).max())
        else:
            golden_img[k] = arr
            rec["psnr"] = rec["mae"] = rec["ssim"] = None
            rec["role"] = "golden"
        print(f"    {k}: {dt:.1f}s  " + (f"PSNR {rec['psnr']:.2f} dB  SSIM {rec['ssim']:.4f}  MAE {rec['mae']:.2f}"
              if rec.get("psnr") else "(golden)"))
        results[arm].append(rec)
    pk = torch.cuda.max_memory_allocated() / 2**30
    print(f"    峰值显存 {pk:.2f} GB")
    results[arm + "__peak_gb"] = round(pk, 2)
    del pipe; gc.collect(); torch.cuda.empty_cache()

print("\n" + "=" * 78)
print("汇总（对 bf16 的图像级偏差）")
print(f"{'arm':<10}{'key':<14}{'PSNR(dB)':>10}{'SSIM':>9}{'MAE':>8}{'maxdiff':>9}{'s':>7}")
print("-" * 88)
for arm, recs in results.items():
    if not isinstance(recs, list):
        continue
    for r in recs:
        if r.get("psnr") is None:
            print(f"{arm:<10}{r['key']:<14}{'(golden)':>10}")
        else:
            print(f"{arm:<10}{r['key']:<14}{r['psnr']:>10.2f}{r['ssim']:>9.4f}{r['mae']:>8.2f}"
                  f"{r['maxdiff']:>9}{r['seconds']:>7}")
fp = os.path.join(OUT, "e5_e2e.json")
with open(fp, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}\n✅ 图在 {IMG}/")
print("=" * 78)

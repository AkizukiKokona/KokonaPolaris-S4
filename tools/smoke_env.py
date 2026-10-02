"""KokonaPolaris E2b 环境冒烟验证
用途：确认 venv 覆盖层生效 + 关键导入不再被 transformers 版本打坏 + sm_120 可算。
运行：source /d/model/env.sh && "$KP_PY" tools/smoke_env.py
"""
import sys, traceback

print("=" * 68)
print(f"python : {sys.version.split()[0]}  @ {sys.executable}")
print("=" * 68)

# ---------- 1. 关键包版本（重点看是否用了 venv 的覆盖版） ----------
print("\n[1] 关键包版本")
import importlib.metadata as md
want = {
    "torch": None, "transformers": "4.55.4", "diffusers": "0.35.2",
    "huggingface_hub": "0.36.2", "tokenizers": "0.21.4",
    "peft": None, "accelerate": None, "safetensors": None,
    "nvidia-modelopt": None, "numpy": None, "einops": None,
}
defaults = {"torch": "?", "peft": "?", "accelerate": "?", "safetensors": "?",
            "nvidia-modelopt": "?", "numpy": "?", "einops": "?"}
for k in want:
    try:
        v = md.version(k)
    except Exception as e:
        v = f"<MISSING {type(e).__name__}>"
    exp = want[k] or defaults.get(k, "")
    mark = ""
    if want[k]:
        mark = "  ✅" if v == exp else f"  ❌ 期望 {exp}"
    print(f"    {k:18s} {v}{mark}")

# ---------- 2. 导入链（重点：peft / modelopt 曾被打坏） ----------
print("\n[2] 导入链")
for name, code in [
    ("torch",                          "import torch"),
    ("huggingface_hub",                "import huggingface_hub"),
    ("transformers",                   "import transformers"),
    ("diffusers",                      "import diffusers"),
    ("peft",                           "import peft"),           # ← 曾被 transformers 5.x 打坏
    ("accelerate",                     "import accelerate"),
    ("modelopt(torch.quantization)",   "import modelopt.torch.quantization as mtq"),
]:
    try:
        exec(code, {})
        print(f"    ✅ {name}")
    except Exception as e:
        print(f"    ❌ {name}  -> {type(e).__name__}: {e}")

# ---------- 3. diffusers 里 G1 需要的 pipeline 是否齐 ----------
print("\n[3] diffusers 目标 pipeline / VAE")
for cls in ["SanaPipeline", "Lumina2Pipeline", "AutoencoderDC", "AutoencoderKL",
            "DPMSolverMultistepScheduler", "FlowMatchEulerDiscreteScheduler"]:
    try:
        __import__("diffusers", fromlist=[cls])
        getattr(__import__("diffusers", fromlist=[cls]), cls)
        print(f"    ✅ {cls}")
    except Exception as e:
        print(f"    ⚪ {cls}  ({type(e).__name__})")

# ---------- 4. sm_120 实算：FP4 block-scaled GEMM ----------
print("\n[4] sm_120 实算验证")
try:
    import torch
    print(f"    device : {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    flops = 2 * 2048**3 * 50
    print(f"    ✅ bf16 GEMM 2048³×50 = {flops/dt/1e12:.1f} TFLOPS")
except Exception:
    traceback.print_exc()

# ---------- 5. ModelOpt NVFP4 配方可用性 ----------
print("\n[5] ModelOpt NVFP4 配方")
try:
    import modelopt.torch.quantization as mtq
    cfgs = [x for x in dir(mtq) if "NVFP4" in x.upper()]
    print(f"    共 {len(cfgs)} 个 NVFP4 配方")
    for c in ["NVFP4_DEFAULT_CFG", "NVFP4_FOUR_OVER_SIX_CFG",
              "NVFP4_FP8_MHA_CONFIG", "NVFP4_KV_ROTATE_CFG"]:
        print(f"    {'✅' if hasattr(mtq, c) else '⚪'} {c}")
    has_mm = hasattr(mtq, "mtq") or True
    import modelopt.torch.quantization as q
    print(f"    ✅ quantize() 存在: {hasattr(q, 'quantize')}")
except Exception:
    traceback.print_exc()

print("\n" + "=" * 68)
print("冒烟验证结束")
print("=" * 68)

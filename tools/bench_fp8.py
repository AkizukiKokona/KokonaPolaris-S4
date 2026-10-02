"""KokonaPolaris · FP8 吞吐补齐（本机此前只测过 bf16 / FP4）

⛔⛔ 不要在夜间运行本脚本 ⛔⛔
   夜间整机切安静模式，GPU 被强制压到 ~40 W，soak 拿不到有效数字、且违背用户意愿。
   **只在用户明确解锁功耗（115 W 档）的白天/非夜间窗口运行。**（2026-10-03 用户规则）

目的：回答「W4A4 vs W4A8 差多远」里的**算力**一维。
  - W4A4 = 权重 FP4 × 激活 FP4  → 走 FP4 tensor 通路
  - W4A8 = 权重 FP4 × 激活 FP8  → 权重反量化到 FP8，走 FP8 tensor 通路
所以两者的算力上限之比 = FP4 tensor 峰值 : FP8 tensor 峰值。

⚠️ 必须标注工况：本脚本先 soak 20s，再在同一工况下依次测 bf16 / FP8 / FP4，
   只看**比值**（本机功耗档由用户设定，比值才是最可信的架构性数字）。

用法：python bench_fp8.py
"""
import torch, time, subprocess, sys

DEV = "cuda"


def pwr(tag=""):
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        p, c, t, u = [x.strip() for x in o.split(",")]
        return f"{tag}[{p:>6} W | {c:>5} MHz | {t:>2}C | util {u:>3}%]"
    except Exception as e:
        return f"{tag}[采样失败 {e}]"


def bench(fn, n, iters, label):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    tf = 2 * n ** 3 / (ms / 1000) / 1e12
    print(f"  {label:28s} {n}³ ×{iters:<4d} {ms:8.3f} ms  →  {tf:7.1f} TFLOPS", flush=True)
    return tf


print("=" * 74)
print("  KokonaPolaris-S4 · FP8 吞吐补齐（bf16 / FP8 / FP4 同工况对照）")
print("=" * 74)
print(f"  torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
print(f"  {pwr('起始 ')}")

# ---- soak：把功耗拉到稳定工况 ----
print("-" * 74)
print("  [soak] 20s 满载（保证三条测量在同一工况下）")
x = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
y = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
t0 = time.time()
while time.time() - t0 < 20:
    z = x @ y
torch.cuda.synchronize()
print(f"  {pwr('soak 后 ')}")
del x, y, z
torch.cuda.empty_cache()

n = 4096
res = {}

# ---- bf16 ----
try:
    xb = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    yb = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
    res["bf16"] = bench(lambda: xb @ yb, n, 80, "bf16 (参考基准)")
    print(f"  {pwr('   ') }")
except Exception as ex:
    print(f"  bf16 失败：{ex}")

# ---- FP8 ----
try:
    xf = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn)
    yf = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn).t()
    s1 = torch.tensor([1.0], device=DEV)
    res["fp8"] = bench(lambda: torch._scaled_mm(xf, yf, s1, s1, out_dtype=torch.bfloat16),
                       n, 80, "FP8 e4m3 (_scaled_mm)")
    print(f"  {pwr('   ') }")
except Exception as ex:
    print(f"  FP8 失败：{type(ex).__name__}: {str(ex)[:200]}")

# ---- FP4 ----
try:
    af = torch.randint(0, 256, (n, n // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    bf = torch.randint(0, 256, (n, n // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    as_ = torch.ones(n, n // 16, device=DEV, dtype=torch.float8_e4m3fn)
    bs_ = torch.ones(n, n // 16, device=DEV, dtype=torch.float8_e4m3fn)
    res["fp4"] = bench(lambda: torch._scaled_mm(af, bf.t(), as_, bs_, out_dtype=torch.bfloat16),
                       n, 80, "FP4 block16 (_scaled_mm)")
    print(f"  {pwr('   ') }")
except Exception as ex:
    print(f"  FP4 失败：{type(ex).__name__}: {str(ex)[:200]}")

print()
print("=" * 74)
print("  结果（同一工况下的相对比值）")
print("=" * 74)
for k, v in res.items():
    print(f"    {k:6s} {v:7.1f} TFLOPS")
if {"bf16", "fp8", "fp4"} <= set(res):
    print(f"\n    FP8 / bf16 = {res['fp8']/res['bf16']:.2f}×")
    print(f"    FP4 / bf16 = {res['fp4']/res['bf16']:.2f}×")
    print(f"    FP4 / FP8  = {res['fp4']/res['fp8']:.2f}×   ← W4A4 相对 W4A8 的算力优势")
print(f"  {pwr('结束 ')}")

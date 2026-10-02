"""KokonaPolaris · G0 环境探针
目的：验证 SM120 上 bf16 / fp8 / fp4 的可用性与实际吞吐，判断 G0 门是否已通过。
"""
import torch, sys, time

SYS = "C:/Users/Akizuki/AppData/Local/Programs/Python/Python312/python.exe"
print("=" * 66)
print("torch", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用，停止")

p = torch.cuda.get_device_properties(0)
print("device         :", p.name)
print("capability     :", f"sm_{p.major}{p.minor}")
print("显存总量       :", f"{p.total_memory/1024**3:.2f} GiB")
print("SM 数量        :", p.multi_processor_count)
print("arch_list      :", torch.cuda.get_arch_list())
print("=" * 66)

# --- 1. bf16 矩阵乘（基础可用性） ---
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
bf16_t = (time.time() - t0) / 20
tflops = 2 * 4096**3 / bf16_t / 1e12
print(f"[bf16] 4096³ matmul  {bf16_t*1000:.2f} ms  →  {tflops:.1f} TFLOPS  (mean={c.float().abs().mean():.3f})")

# --- 2. 低比特 dtype 是否存在 ---
print("-" * 66)
for name in ["float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2", "float8_e8m0fnu"]:
    print(f"  torch.{name:22s}", "✓" if hasattr(torch, name) else "✗")

# --- 3. torch._scaled_mm （block-scaled matmul 入口） ---
print("-" * 66)
print("  torch._scaled_mm        ", "✓" if hasattr(torch, "_scaled_mm") else "✗")
try:
    from torch.nn.functional import scaled_mm as _sm  # noqa
    print("  F.scaled_mm             ✓")
except Exception as e:
    print("  F.scaled_mm             ✗", e)

# --- 4. fp8 scaled_mm 实跑（tensor core 是否真的走低比特路径） ---
try:
    from torch.nn.functional import scaled_mm
    M = N = K = 4096
    xq = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    wq = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
    xs = torch.tensor(1.0, device="cuda")
    ws = torch.tensor(1.0, device="cuda")
    out = scaled_mm(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        out = scaled_mm(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    fp8_t = (time.time() - t0) / 20
    print(f"[fp8 ] scaled_mm 4096³  {fp8_t*1000:.2f} ms  →  {2*M*N*K/fp8_t/1e12:.1f} TFLOPS  ✓ 可用")
except Exception as e:
    print("[fp8 ] scaled_mm 失败 :", type(e).__name__, str(e)[:180])

# --- 5. 关键库 ---
print("-" * 66)
for m in ["nunchaku", "modelopt", "triton", "flashinfer", "transformers", "diffusers", "peft", "bitsandbytes"]:
    try:
        mod = __import__(m)
        print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception:
        print(f"  {m:16s} --  未安装")

print("=" * 66)
print("当前显存占用 :", f"{torch.cuda.memory_allocated()/1024**2:.0f} MiB (本进程)")

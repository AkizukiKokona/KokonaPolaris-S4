"""KokonaPolaris · E7 基准复测（115W 满血工况）+ G0 收尾（ModelOpt 端到端 layout）

目的：
  1. 刺激功耗爬升（soak），拿到满血工况下的真实吞吐
  2. bf16 / FP4 对照，替换补充07 §3.3 的「静音下限值」
  3. G0 收尾：用 ModelOpt 量化真实 nn.Linear，确认 block-scale layout 端到端可用

用法：python bench_115w.py
"""
import torch, time, subprocess, sys, os

DEV = "cuda"
PY = sys.executable

def pwr(tag=""):
    """采样当前功耗/时钟/温度，返回可读字符串"""
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        p, clk, t, u, st = [x.strip() for x in o.split(",")]
        return f"{tag}[{p:>6} W | {clk:>5} MHz | {t:>2}°C | util {u:>3}% | {st}]"
    except Exception as e:
        return f"{tag}[采样失败 {e}]"

def tflops(dtype, n, iters, label):
    """用 CUDA events 精确测吞吐"""
    x = torch.randn(n, n, device=DEV, dtype=dtype)
    y = torch.randn(n, n, device=DEV, dtype=dtype)
    for _ in range(3):
        z = x @ y
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        z = x @ y
    e.record()
    torch.cuda.synchronize()

    ms = s.elapsed_time(e) / iters
    tf = 2 * n**3 / (ms / 1000) / 1e12
    print(f"  {label:24s} {n}³ ×{iters:<4d} {ms:7.3f} ms/iter  →  {tf:7.1f} TFLOPS", flush=True)
    del x, y, z
    torch.cuda.empty_cache()
    return tf

def fp4_mod(M, N, K, block=16):
    """NVFP4: E2M1 packed + block16 E4M3 scale（对齐 probe2 的调用方式）"""
    a = torch.randint(0, 256, (M, K // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    b = torch.randint(0, 256, (N, K // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    a_s = torch.ones(M, K // block, device=DEV, dtype=torch.float8_e4m3fn)
    b_s = torch.ones(N, K // block, device=DEV, dtype=torch.float8_e4m3fn)
    return torch._scaled_mm(a, b.t(), a_s, b_s, out_dtype=torch.bfloat16)

def bench_fp4(n, iters):
    for _ in range(5):
        fp4_mod(n, n, n)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fp4_mod(n, n, n)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    tf = 2 * n**3 / (ms / 1000) / 1e12
    print(f"  {'FP4 (block-scaled)':24s} {n}³ ×{iters:<4d} {ms:7.3f} ms/iter  →  {tf:7.1f} TFLOPS", flush=True)
    return tf

print("=" * 74)
print("  KokonaPolaris-S4 · E7 满血工况基准复测 + G0 收尾")
print("=" * 74)
print(f"  torch {torch.__version__}  |  device {torch.cuda.get_device_name(0)}")
print(f"  {pwr('起始状态 ')}")
print()

# ---------- Phase A：功耗刺激 ----------
print("-" * 74)
print("  [A] 功耗刺激：连续满载 30s，把 TGP 从静音档拉上去")
print("-" * 74)
x = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
y = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
t0, i = time.time(), 0
while time.time() - t0 < 30:
    z = x @ y
    i += 1
    el = time.time() - t0
    if i % 15 == 0:
        print(f"    t={el:5.1f}s  {pwr()}", flush=True)
torch.cuda.synchronize()
print(f"    soak 完成：{i} 轮 8192³，{pwr('结束状态 ')}")
del x, y, z
torch.cuda.empty_cache()
print()

# ---------- Phase B：bf16 吞吐 ----------
print("-" * 74)
print("  [B] bf16 吞吐（满血工况）")
print("-" * 74)
r = {}
r["bf16_8192"] = tflops(torch.bfloat16, 8192, 30, "bf16")
print(f"        测量时 {pwr()}")
r["bf16_12288"] = tflops(torch.bfloat16, 12288, 12, "bf16")
print(f"        测量时 {pwr()}")
print()

# ---------- Phase C：FP4 吞吐 ----------
print("-" * 74)
print("  [C] FP4 block-scaled 吞吐（满血工况）")
print("-" * 74)
try:
    r["fp4_2048"] = bench_fp4(2048, 200)
    r["fp4_4096"] = bench_fp4(4096, 40)
    r["fp4_8192"] = bench_fp4(8192, 8)
    print(f"        测量时 {pwr()}")
except Exception as ex:
    print(f"  FP4 失败：{type(ex).__name__}: {str(ex)[:200]}")
print()

# ---------- Phase D：G0 收尾 ----------
print("-" * 74)
print("  [D] G0 收尾：ModelOpt 量化真实 nn.Linear（端到端 layout 验证）")
print("-" * 74)
try:
    import modelopt
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
    print(f"  modelopt {modelopt.__version__}")

    torch.manual_seed(0)
    ref = torch.nn.Linear(1024, 1024).cuda().eval()
    xin = torch.randn(8, 1024, device=DEV)
    y_ref = ref(xin.clone())

    q = torch.nn.Sequential(torch.nn.Linear(1024, 1024).cuda().eval())
    q[0].load_state_dict(ref.state_dict())

    def calib(m):
        for _ in range(4):
            m(torch.randn(16, 1024, device=DEV))

    mtq.quantize(q, NVFP4_DEFAULT_CFG, forward_loop=calib)
    print("  量化完成 ✓")

    # 看量化后的层类型
    ln = q[0]
    print(f"  量化后层类型: {type(ln).__name__}")
    for nm, mod in ln.named_modules():
        if nm:
            print(f"    ├─ {nm:16s} {type(mod).__name__}")

    with torch.no_grad():
        y_q = q[0](xin.clone())
    diff = (y_q - y_ref).abs()
    rel = (diff / y_ref.abs().clamp(min=1e-3)).mean().item()
    print(f"  NVFP4 输出 vs BF16 参考：mean|Δ| = {diff.mean().item():.5f}  "
          f"max|Δ| = {diff.max().item():.5f}  mean rel = {rel*100:.2f}%")
    print(f"  参考输出量级 mean|y| = {y_ref.abs().mean().item():.4f}")
    ok = diff.mean().item() < 0.1 * y_ref.abs().mean().item()
    print(f"  layout 判定：{'✅ 端到端可用（误差在合理范围）' if ok else '⚠️ 误差偏大，需检查'}")
except Exception as ex:
    import traceback
    print(f"  ✗ {type(ex).__name__}: {str(ex)[:300]}")
    traceback.print_exc()

print()
print("=" * 74)
print("  汇总")
print("=" * 74)
for k, v in r.items():
    print(f"    {k:14s} {v:7.1f} TFLOPS")
print(f"  {pwr('结束状态 ')}")
free_, total = torch.cuda.mem_get_info()
print(f"  显存 {free_/1024**3:.2f} GiB 空闲 / {total/1024**3:.2f} GiB 总计")

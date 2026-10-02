"""KokonaPolaris · G0 决定性探针：NVFP4 block-scaled matmul 在 SM120 上能否实跑
外加：naming/模型量化工具链自检
"""
import torch, traceback, time, inspect

print("=" * 70)
print("### 1. torch._scaled_mm 的 FP4 支持")
print("=" * 70)
try:
    doc = (torch._scaled_mm.__doc__ or "")[:1600]
    print(doc)
except Exception as e:
    print("无法读取 doc:", e)

print("\n--- 尝试 FP4 block-scaled matmul (NVFP4: E2M1 + block16) ---")

def try_fp4(M, N, K, block=16):
    """NVFP4 布局：A [M,K] packed uint8 (K/2), 每 16 个元素一个 e4m3 scale"""
    dev = "cuda"
    # packed fp4: 每个 uint8 装 2 个 4bit 值
    a_packed = torch.randint(0, 256, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    b_packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    # block scale: [M, K/block] 与 [N, K/block]，E4M3
    a_s = torch.ones(M, K // block, device=dev, dtype=torch.float8_e4m3fn)
    b_s = torch.ones(N, K // block, device=dev, dtype=torch.float8_e4m3fn)
    out = torch._scaled_mm(a_packed, b_packed.t(), a_s, b_s, out_dtype=torch.bfloat16)
    return out

for (M, N, K) in [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048)]:
    try:
        out = try_fp4(M, N, K)
        torch.cuda.synchronize()
        print(f"  [{M}x{N}x{K}] FP4 block-scaled matmul  ✓ 输出 {tuple(out.shape)} {out.dtype}")
        if M >= 1024:
            t0 = time.time()
            for _ in range(20):
                try_fp4(M, N, K)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 20
            print(f"          吞吐 {2*M*N*K/dt/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"  [{M}x{N}x{K}] ✗ {type(e).__name__}: {str(e)[:220]}")

print("\n" + "=" * 70)
print("### 2. 稳定吞吐基准（更长循环，避开 P8 降频）")
print("=" * 70)
for dt_name, dtype in [("bf16", torch.bfloat16)]:
    a = torch.randn(8192, 8192, device="cuda", dtype=dtype)
    b = torch.randn(8192, 8192, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    n = 50
    t0 = time.time()
    for _ in range(n):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  [fp32/bf16] 8192³ x{n}: {dt*1000:.1f} ms → {2*8192**3/dt/1e12:.1f} TFLOPS")

print("\n" + "=" * 70)
print("### 3. NVIDIA ModelOpt 的 NVFP4 配方（G1 要用的工具）")
print("=" * 70)
try:
    import modelopt
    print("  modelopt", modelopt.__version__)
    from modelopt.torch.quantization import config as qc
    names = [n for n in dir(qc) if "NVFP4" in n.upper() or "FP4" in n.upper()]
    print("  fp4 相关配置:", ", ".join(names) if names else "（未在 config 顶层找到）")
    try:
        from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG
        print("  NVFP4_DEFAULT_CFG ✓ 存在，quant_cfg 条目数:", len(NVFP4_DEFAULT_CFG.get("quant_cfg", {})))
        for k, v in list(NVFP4_DEFAULT_CFG.get("quant_cfg", {}).items())[:10]:
            print(f"     {k:34s} {v}")
    except Exception as e:
        print("  NVFP4_DEFAULT_CFG:", type(e).__name__, str(e)[:150])
except Exception as e:
    print("  modelopt 不可用:", type(e).__name__, str(e)[:200])

print("\n" + "=" * 70)
print("### 4. nunchaku / flashinfer 导入诊断")
print("=" * 70)
for m in ["nunchaku", "flashinfer"]:
    try:
        mod = __import__(m)
        print(f"  {m}: ✓ {getattr(mod, '__version__', '?')}  @ {getattr(mod, '__file__', '?')}")
    except Exception as e:
        print(f"  {m}: ✗ {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 70)
print("### 5. 显存现状")
print("=" * 70)
free_, total = torch.cuda.mem_get_info()
print(f"  空闲 {free_/1024**3:.2f} GiB / 总计 {total/1024**3:.2f} GiB   （已占用 {(total-free_)/1024**3:.2f} GiB）")

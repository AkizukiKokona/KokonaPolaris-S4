"""KokonaPolaris E5 · 前置校验：FP8 模拟量化到底生效了没有

背景：跑 W4A8 时出现
    "CUDA extension for FP8 quantization could not be built and loaded,
     FP8 simulated quantization will not be available."
如果 FP8 真的没生效，那么 W4A8 的误差就全部来自「权重 block 32 比 16 更粗」，
而不是来自 FP8 激活 —— 结论会完全反过来。必须先证伪或证实。

判据（单层干净对照）：
  W4A16(权重b16, 激活bf16)   vs   W4A8(权重b32, 激活FP8)
  若两者误差**完全相同** → FP8 未生效（W4A8 退化成 W4A16+b32）
  若不同 → FP8 生效，误差差值是 FP8 与 block32 的混合效应

运行：source /d/model/env.sh && "$KP_PY" tools/e5_fp8_check.py
"""
import torch, copy, inspect
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import tensor_quant as tq

print("=" * 74)
print("[1] ModelOpt 的 FP8 实现路径")
try:
    src = inspect.getsource(tq.get_cuda_ext_fp8)
    print(src[:600])
except Exception as e:
    print("  取值失败:", e)
try:
    src = inspect.getsource(tq._fp8_eager)
    print("\n--- _fp8_eager ---")
    print(src[:900])
except Exception as e:
    print("  取值失败:", e)

print("\n[2] 单层干净对照（256→256 Linear，固定种子）")
torch.manual_seed(0)
lin = torch.nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
x = torch.randn(1, 16, 256, device="cuda", dtype=torch.bfloat16) * 3.0

with torch.no_grad():
    y_ref = lin(x)

def quant_copy(cfg):
    m = copy.deepcopy(lin)
    mtq.quantize(m, cfg)
    return m

arms = {
    "W4A16 (w:b16, a:bf16)": mtq.W4A16_NVFP4_CFG,
    "W4A8  (w:b32, a:FP8) ": mtq.W4A8_NVFP4_FP8_CFG,
    "W4A4  (w:b16, a:b16) ": mtq.NVFP4_DEFAULT_CFG,
}

outs = {}
for nm, cfg in arms.items():
    m = quant_copy(cfg)
    with torch.no_grad():
        y = m(x)
    rel = ((y - y_ref).norm() / y_ref.norm()).item()
    outs[nm] = y.clone()
    print(f"  {nm}  相对误差 {rel*100:>7.3f}%  输出哈希 {hash(tuple(y.flatten()[:8].tolist()))%10**6}")

print("\n[3] 判据")
d_16_8 = (outs["W4A16 (w:b16, a:bf16)"] - outs["W4A8  (w:b32, a:FP8) "]).norm().item()
denom = outs["W4A16 (w:b16, a:bf16)"].norm().item()
print(f"  ||W4A16 - W4A8|| / ||W4A16|| = {d_16_8/denom*100:.4f}%")
if d_16_8 / denom < 1e-6:
    print("  ❌ 两者完全一致 → FP8 模拟**未生效**，W4A8 退化为 W4A16(block32)")
else:
    print("  ✅ 两者有差异 → FP8 模拟**已生效**，W4A8 是真实的 FP8 激活")

# 进一步：直接测 FP8 quantizer 本身
print("\n[4] 直接验证 FP8 quantizer 的数值行为")
try:
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
    q = TensorQuantizer(num_bits=8, block_sizes=None, axis=None, disable_quantizer=False)
    q.cuda()
    xx = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16) * 5
    with torch.no_grad():
        yq = q(xx)
    print(f"    输入 dtype {xx.dtype} → 输出 dtype {yq.dtype}")
    print(f"    FP8 量化相对误差 {(yq.float()-xx.float()).norm().item()/xx.float().norm().item()*100:.4f}%")
    print(f"    quantizer 是否 enabled: {q.is_enabled}")
except Exception as e:
    print("    直接测失败:", type(e).__name__, e)
print("=" * 74)

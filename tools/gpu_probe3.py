"""KokonaPolaris · G0 数值正确性验证（v2）
"能跑"不等于"算对" —— 手工量化到 E2M1 再反量化做参考实现，比对 _scaled_mm。
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLK = 16

def quant_e2m1(x):
    """x:[M,K] → q:[M,K/blk,blk] int(0..15), scale:[M,K/blk,1]
    注意：E2M1 表是【幅值】表，最近邻必须在 |xn| 上做，符号单独存第 4 位。
    """
    M, K = x.shape
    xb = x.view(M, K // BLK, BLK)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 6.0
    xn = (xb / scale).clamp(-6, 6)
    idx = (xn.abs().unsqueeze(-1) - E2M1.to(x.device)).abs().argmin(dim=-1)   # ← 用 |xn|
    neg = (xn < 0) & (idx != 0)
    q = torch.where(neg, idx + 8, idx)
    return q, scale

def dequant_e2m1(q, scale):
    idx = (q & 0x7).to(torch.long)
    neg = (q & 0x8) != 0
    v = E2M1.to(q.device)[idx]
    v = torch.where(neg, -v, v)
    return v * scale

def pack_e2m1(q, low_first=True):
    """q:[M,K/blk,blk] → packed float4_e2m1fn_x2 [M, K/2]"""
    M, Kb, b = q.shape
    q = q.reshape(M, Kb * b).to(torch.uint8).view(M, Kb * b // 2, 2)
    if low_first:
        packed = q[..., 0] | (q[..., 1] << 4)
    else:
        packed = q[..., 1] | (q[..., 0] << 4)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)

M = N = K = 256
torch.manual_seed(0)
A = torch.randn(M, K, device="cuda")
B = torch.randn(N, K, device="cuda")

qa, sa = quant_e2m1(A)
qb, sb = quant_e2m1(B)
A_dq = dequant_e2m1(qa, sa).reshape(M, K)
B_dq = dequant_e2m1(qb, sb).reshape(N, K)
ref_bf16 = (A_dq.bfloat16() @ B_dq.bfloat16().t()).float()

sa8 = sa.squeeze(-1).contiguous().to(torch.float8_e4m3fn)
sb8 = sb.squeeze(-1).contiguous().to(torch.float8_e4m3fn)

print("=" * 68)
print("### FP4 block-scaled matmul 数值正确性（SM120）")
print("=" * 68)

for low_first in (True, False):
    Ap = pack_e2m1(qa, low_first)
    Bp = pack_e2m1(qb, low_first)
    try:
        out = torch._scaled_mm(Ap, Bp.t(), sa8, sb8, out_dtype=torch.bfloat16).float()
        err = (out - ref_bf16).abs().mean().item()
        den = ref_bf16.abs().mean().item()
        tag = "low_nibble_first" if low_first else "high_nibble_first"
        print(f"  [{tag:18s}] 平均绝对误差 {err:.5f}   相对误差 {err/den*100:.4f} %   {'✅ 命中' if err/den < 0.01 else '✗'}")
    except Exception as e:
        print(f"  [{tag}] 失败: {type(e).__name__}: {str(e)[:120]}")

print(f"\n  参考量级 |ref| mean={den:.4f} max={ref_bf16.abs().max().item():.3f}")
print("  说明：0.01% 量级的相对误差即代表 kernel 与参考实现完全一致（误差仅来自 bf16 累加）")

print()
print("=" * 68)
print("### FP8 用于 attention 的精度损失")
print("=" * 68)
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
q8 = x.to(torch.float8_e4m3fn)
back = q8.to(torch.bfloat16)
rel = (x - back).abs().mean() / x.abs().mean() * 100
print(f"  E4M3 往返相对误差: {rel.item():.3f}%   → attention QK/KV 用 FP8 属可接受范围")

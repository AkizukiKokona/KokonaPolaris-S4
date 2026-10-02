"""KokonaPolaris · G0 最终验证：正确的 NVFP4 block-scaled matmul 布局
上一轮已证明：E2M1 解码正确、scale 生效。
剩余偏差来源嫌疑：B 的转置是【非连续视图】，packed-fp4 下转置必须显式重打包。
本探针：显式构造 [K,N] 布局的 B，测 scale 两种朝向。
"""
import torch

VAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLK = 16

def nearest_idx(xn):
    t = torch.tensor(VAL, device=xn.device)
    return (xn.abs().unsqueeze(-1) - t).abs().argmin(dim=-1)

def pack_along_last(q):
    """q:[...,L] int → packed uint8 [...,L/2]"""
    *lead, L = q.shape
    q = q.to(torch.uint8).reshape(*lead, L // 2, 2)
    return (q[..., 0] | (q[..., 1] << 4)).contiguous()

def quant_A(x):
    """x [M,K] → packed fp4 [M,K/2], scale e4m3 [M,K/16], dequant [M,K]"""
    M, K = x.shape
    xb = x.view(M, K // BLK, BLK)
    s = (xb.abs().amax(-1, keepdim=True).clamp(min=1e-8)) / 6.0
    xn = (xb / s).clamp(-6, 6)
    idx = nearest_idx(xn)
    q = torch.where((xn < 0) & (idx != 0), idx + 8, idx)
    dq = torch.where((q & 0x8) != 0, -torch.tensor(VAL, device=x.device)[q & 0x7],
                     torch.tensor(VAL, device=x.device)[q & 0x7]) * s
    packed = pack_along_last(q.reshape(M, K)).view(torch.float4_e2m1fn_x2)
    return packed, s.squeeze(-1).contiguous().to(torch.float8_e4m3fn), dq.reshape(M, K)

def quant_B(bt):
    """bt [K,N]（B 的逻辑转置）→ packed fp4 [K/2,N], scale [K/16,N] 与 [N,K/16]"""
    K, N = bt.shape
    xb = bt.reshape(K // BLK, BLK, N)
    s = (xb.abs().amax(1, keepdim=True).clamp(min=1e-8)) / 6.0     # [K/16,1,N]
    xn = (xb / s).clamp(-6, 6)
    idx = nearest_idx(xn.transpose(0, 2)).transpose(0, 2)          # 在 N 维做最近邻
    q = torch.where((xn < 0) & (idx != 0), idx + 8, idx)
    t = torch.tensor(VAL, device=bt.device)
    dq = torch.where((q & 0x8) != 0, -t[q & 0x7], t[q & 0x7]) * s
    # pack 沿 K（dim0）：每两行 K 合成一个 byte
    qk = q.reshape(K, N).to(torch.uint8).reshape(K // 2, 2, N)
    packed = (qk[:, 0, :] | (qk[:, 1, :] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    s_kx = s.squeeze(1).contiguous().to(torch.float8_e4m3fn)        # [K/16, N]
    return packed, s_kx, s_kx.t().contiguous(), dq.reshape(K, N)

torch.manual_seed(0)
M = N = K = 512
A = torch.randn(M, K, device="cuda")
B = torch.randn(N, K, device="cuda")

Ap, sa, A_dq = quant_A(A)
Bp, sb_KN, sb_NK, B_dq = quant_B(B.t().contiguous())

ref = A_dq.float() @ B_dq.float()

print("=" * 72)
print(f"### NVFP4 block-scaled matmul（M=N=K={M}, block=16）")
print("=" * 72)
for label, sb in [("scale_b=[N,K/16]", sb_NK), ("scale_b=[K/16,N]", sb_KN)]:
    try:
        out = torch._scaled_mm(Ap, Bp, sa, sb, out_dtype=torch.bfloat16).float()
        err = (out - ref).abs().mean().item()
        den = ref.abs().mean().item()
        rel = err / den * 100
        flag = "✅ 与参考完全一致" if rel < 0.05 else "✗ 不符"
        print(f"  {label:18s} 相对误差 {rel:8.4f} %   {flag}")
    except Exception as e:
        print(f"  {label:18s} 异常 {type(e).__name__}: {str(e)[:110]}")

print()
print("  参考量级 |ref| mean = %.4f" % ref.abs().mean().item())
print("  （<0.05% 即代表 kernel 与手写反量化参考一致，残留误差仅来自 bf16 累加）")

print()
print("=" * 72)
print("### 吞吐：FP4 vs BF16（8192³，同机对比）")
print("=" * 72)
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(5):
    a @ b
torch.cuda.synchronize()
t0 = __import__("time").time()
for _ in range(30):
    a @ b
torch.cuda.synchronize()
dt = (__import__("time").time() - t0) / 30
print(f"  bf16 8192³ : {dt*1000:7.1f} ms  →  {2*8192**3/dt/1e12:6.1f} TFLOPS")

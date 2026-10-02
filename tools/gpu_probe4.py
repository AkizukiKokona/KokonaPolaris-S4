"""KokonaPolaris · G0 决定性测试：_scaled_mm 到底有没有按 E2M1 解码我给的字节？
方法：把 A 的所有 fp4 元素填成同一个已知值，B 也是 → 正确结果是一个可以手算出来的常数。
E2M1 幅值表（index 0..7）= [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
"""
import torch

VAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
M = N = K = 128

def fill(code):
    """code: 0..7，两个 nibble 都填同一个幅值码 → 每个 fp4 元素 = code/2 表值"""
    b = code | (code << 4)                     # 0x11, 0x22, 0x55 …
    raw = torch.full((M, K // 2), b, dtype=torch.uint8, device="cuda")
    return raw.view(torch.float4_e2m1fn_x2)

s1 = torch.ones(M, K // 16, dtype=torch.float8_e4m3fn, device="cuda")
s2 = torch.ones(N, K // 16, dtype=torch.float8_e4m3fn, device="cuda")

print("=" * 70)
print(f"### 常量填充测试  (M=N=K={K}，scale 全 1.0)")
print("=" * 70)
print(f"{'A码':>4} {'B码':>4} | {'手算应为':>10} | {'_scaled_mm 实测':>16} | 判定")
print("-" * 70)

for ca in (2, 5, 7):          # A = 1.0 / 3.0 / 6.0
    for cb in (1, 5):         # B = 0.5 / 3.0
        va, vb = VAL[ca], VAL[cb]
        expect = va * vb * K
        try:
            A = fill(ca)
            B = fill(cb)
            out = torch._scaled_mm(A, B.t(), s1, s2, out_dtype=torch.bfloat16)
            got = out.float().mean().item()
            ok = abs(got - expect) / max(abs(expect), 1e-6) < 0.01
            print(f"{ca:>4} {cb:>4} | {expect:>10.1f} | {got:>16.1f} | {'✅ 解码正确' if ok else '✗ 不符'}")
        except Exception as e:
            print(f"{ca:>4} {cb:>4} | {expect:>10.1f} | {'异常':>16} | {type(e).__name__}: {str(e)[:60]}")

print()
print("=" * 70)
print("### scale 是否被正确应用（A=1.0, B=3.0, scale_a=scale_b=2.0 → 应为 4×768=3072）")
print("=" * 70)
try:
    A = fill(2)
    B = fill(5)
    s2x = torch.full((M, K // 16), 2.0, dtype=torch.float8_e4m3fn, device="cuda")
    s2y = torch.full((N, K // 16), 2.0, dtype=torch.float8_e4m3fn, device="cuda")
    out = torch._scaled_mm(A, B.t(), s2x, s2y, out_dtype=torch.bfloat16)
    got = out.float().mean().item()
    print(f"  实测 {got:.1f}   期望 {1.0*3.0*K*2*2:.1f}   {'✅ scale 生效' if abs(got-3072)/3072<0.02 else '✗ scale 未按预期生效'}")
except Exception as e:
    print("  异常:", type(e).__name__, str(e)[:200])

print()
print("=" * 70)
print("### 对照：bf16 同尺寸结果（确认参照系没问题）")
print("=" * 70)
Abf = torch.ones(M, K, device="cuda", dtype=torch.bfloat16) * 1.0
Bbf = torch.ones(K, N, device="cuda", dtype=torch.bfloat16) * 3.0
print(f"  bf16 (1.0 × 3.0 × {K}) = {(Abf @ Bbf).float().mean().item():.1f}  （应 = {3.0*K:.1f}）")

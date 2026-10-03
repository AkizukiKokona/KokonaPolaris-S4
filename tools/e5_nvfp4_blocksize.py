"""KokonaPolaris E5 · 归因验证：NVFP4 权重的 block size（16 vs 32）到底有没有影响

背景：整模型实测里 W4A8(block32) 与 W4A8_b16(block16) 误差**逐位完全相同**（9.98%）。
这很反直觉（block 32 的 scale 粒度更粗，理应有更多噪声），必须独立验证。

方法：绕开 ModelOpt，手写标准 NVFP4 量化（E2M1 + block scale E4M3），
      在**真实 Sana 权重**上比较不同 block size 的重建误差。

运行：source /d/model/env.sh && "$KP_PY" tools/e5_nvfp4_blocksize.py
"""
from kp.paths import MODELS_SANA, OUT
import torch
from diffusers import SanaTransformer2DModel

MODEL = str(MODELS_SANA)

def nvfp4_roundtrip(w, blk):
    """标准 NVFP4：per-block(absmax) → E4M3 scale → E2M1 权重（±6），返回反量化结果。"""
    shape = w.shape
    inn = shape[-1]
    assert inn % blk == 0, f"{inn} % {blk} != 0"
    wb = w.reshape(-1, inn // blk, blk).float()
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 6.0)
    scale_q = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp(min=1e-12)
    q = (wb / scale_q).round().clamp(-6.0, 6.0)
    return (q * scale_q).reshape(shape)

def rel(a, b):
    return ((a - b).norm() / b.norm()).item()

print("=" * 76)
print("[1] 合成权重（含离群值 —— 量化的真正难点）")
torch.manual_seed(0)
for tag, w in [
    ("纯高斯",        torch.randn(512, 512)),
    ("高斯+1%离群",   torch.cat([torch.randn(512, 507) * 1.0,
                                 torch.randn(512, 5) * 30.0], dim=1)),
]:
    print(f"\n  --- {tag} ---")
    for blk in [16, 32, 64]:
        r = rel(nvfp4_roundtrip(w, blk), w) * 100
        print(f"    block={blk:>3}  重建相对误差 {r:>7.3f}%")

print("\n[2] 真实 Sana 权重（transformer 的若干关键层）")
tf = SanaTransformer2DModel.from_pretrained(
    MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16)
lin = [(n, m.weight) for n, m in tf.named_modules() if isinstance(m, torch.nn.Linear)]
print(f"  共 {len(lin)} 个 Linear，抽查 6 个（覆盖浅/中/深）")
idx = [0, len(lin)//5, 2*len(lin)//5, 3*len(lin)//5, 4*len(lin)//5, len(lin)-1]
targets = [lin[i] for i in idx]
print(f"\n  {'层':<26}{'shape':>16}{'b16':>10}{'b32':>10}{'b64':>10}{'32/16':>9}")
print("  " + "-" * 74)
for name, w in targets:
    w = w.detach().float()
    inn = w.shape[-1]
    row = []
    for blk in [16, 32, 64]:
        if inn % blk != 0:
            row.append(None)
        else:
            row.append(rel(nvfp4_roundtrip(w, blk), w) * 100)
    r16, r32, r64 = row
    ratio = (r32 / r16) if (r16 and r32) else float("nan")
    fmt = lambda v: f"{v:>9.3f}%" if v is not None else f"{'n/a':>10}"
    print(f"  {name:<26}{str(tuple(w.shape)):>16}{fmt(r16)}{fmt(r32)}{fmt(r64)}{ratio:>9.2f}")

print("\n[3] 各层权重「平坦度」检查（为什么 block 影响小）")
print("  说明：若权重分布接近均匀/单峰且无局部离群，block 16 与 32 的 absmax 几乎相同")
print("        → scale 一样 → 量化结果一样，此时 block size 自然无影响。")
for name, w in targets[:2]:
    w = w.detach().float()
    for blk in [16, 32]:
        wb = w.reshape(-1, w.shape[-1] // blk, blk)
        amax = wb.abs().amax(dim=-1)
        print(f"  {name:<24} block={blk:>3}  absmax 标准差/均值 = "
              f"{(amax.std()/amax.mean()).item():.4f}")
print("=" * 76)

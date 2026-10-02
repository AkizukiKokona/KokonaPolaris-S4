"""G0 收尾：NVFP4 的 W4A4 / W4A16 误差对照 + 校准算法影响 —— G1 的前哨证据"""
import warnings, copy
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import NVFP4_DEFAULT_CFG

torch.manual_seed(42)
D = 1024
ref = torch.nn.Linear(D, D).cuda().eval()
xin = torch.randn(64, D, device="cuda")
with torch.no_grad():
    y_ref = ref(xin)

def cfg_w4a4(alg="max"):
    c = copy.deepcopy(NVFP4_DEFAULT_CFG)
    c["algorithm"] = alg
    return c

def cfg_w4a16(alg="max"):
    c = copy.deepcopy(NVFP4_DEFAULT_CFG)
    c["algorithm"] = alg
    c["quant_cfg"] = c["quant_cfg"] + [{"quantizer_name": "*input_quantizer", "enable": False}]
    return c

def run(name, cfg, calib_n=512):
    q = torch.nn.Sequential(torch.nn.Linear(D, D).cuda().eval())
    q[0].load_state_dict(ref.state_dict())
    def calib(m):
        with torch.no_grad():
            for _ in range(calib_n // 64):
                m(torch.randn(64, D, device="cuda"))
    try:
        mtq.quantize(q, cfg, forward_loop=calib)
        with torch.no_grad():
            y = q[0](xin)
        d = (y - y_ref).abs()
        rel = d.mean().item() / y_ref.abs().mean().item()
        cos = F.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
        print(f"  {name:28s} rel={rel*100:6.2f}%   cos={cos:.6f}   mean|Δ|={d.mean().item():.5f}")
    except Exception as e:
        print(f"  {name:28s} ✗ {type(e).__name__}: {str(e)[:110]}")

print("=" * 74)
print("  NVFP4 精度组合对照（单层 1024×1024，512 样本校准）")
print("=" * 74)
print(f"  参考：mean|y| = {y_ref.abs().mean().item():.4f}")
print()
print("  ── A. 官方默认是 W4A4 + 朴素 max 校准 ──")
run("W4A4 / max（默认 cfg）", cfg_w4a4("max"))
print()
print("  ── B. 换校准算法（治 outlier）──")
for alg in ["awq", "smoothquant"]:
    run(f"W4A4 / {alg}", cfg_w4a4(alg))
print()
print("  ── C. 只量化权重（激活留 BF16）──")
run("W4A16 / max", cfg_w4a16("max"))
run("W4A16 / awq", cfg_w4a16("awq"))

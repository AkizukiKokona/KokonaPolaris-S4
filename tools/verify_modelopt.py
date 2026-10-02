"""KokonaPolaris E2b 第二步：ModelOpt 真实量化复验（G0 在新 venv 下复现）
背景：modelopt 0.46.0 会警告「transformers 4.55.4 不在其测试矩阵内」——
      但本次要验证的是 torch 量化通路，与 transformers 无关。本脚本把这点钉死。
判据：nn.Linear 被替换为 QuantLinear 且带 3 个 quantizer；前向无 layout 报错；FP4 出数。
运行：source /d/model/env.sh && "$KP_PY" tools/verify_modelopt.py
"""
import torch, torch.nn as nn
import modelopt.torch.quantization as mtq

print("=" * 68)
print("[A] NVFP4_DEFAULT_CFG 结构（真身）")
cfg = mtq.NVFP4_DEFAULT_CFG
print(f"    类型: {type(cfg).__name__}, 顶层键: {list(cfg.keys())}")
qc = cfg["quant_cfg"]
print(f"    quant_cfg: 共 {len(qc)} 条规则")
_dis = 0
for r in qc:
    if r.get("enable") is False:
        _dis += 1
        continue
    qn = r.get("quantizer_name", r.get("parent_class", "?"))
    inner = r.get("cfg", r)
    print(f"    [启用] {qn:<24} -> num_bits={inner.get('num_bits')} "
          f"block={inner.get('block_sizes', {}).get(-1)} bits")
print(f"    [禁用] {_dis} 条规则（含输出/归一化/嵌入类）")
print(f"    algorithm: {cfg['algorithm']!r}")

print("\n[B] 对真实 nn.Linear 做 NVFP4 量化")
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.Linear(1024, 1024, bias=False)).cuda().eval()
x_ref = torch.randn(16, 1024, device="cuda")
with torch.no_grad():
    y_ref = model(x_ref)

calib = [torch.randn(8, 1024, device="cuda") for _ in range(8)]
with torch.no_grad():
    for d in calib:
        model(d)

try:
    qmodel = mtq.quantize(model, cfg, forward_loop=lambda m: [m(d) for d in calib])
    print("    ✅ mtq.quantize 跑通，无 layout 报错")
except Exception as e:
    print(f"    ❌ 量化失败: {type(e).__name__}: {e}")
    raise SystemExit(1)

layer = qmodel[0]
print(f"    层类型: {type(layer).__name__}")
quants = [n for n, _ in layer.named_modules() if "quantizer" in n or "Quantizer" in type(_).__name__]
print(f"    quantizer 数: {len(quants)} -> {quants[:6]}")

print("\n[C] 量化后前向 + 精度对照")
try:
    with torch.no_grad():
        y_q = qmodel(x_ref)
    rel = (y_q - y_ref).abs().mean() / y_ref.abs().mean()
    cos = torch.nn.functional.cosine_similarity(
        y_q.flatten().float(), y_ref.flatten().float(), dim=0)
    print(f"    ✅ 前向跑通")
    print(f"    rel_err = {rel.item()*100:.2f} %   cos = {cos.item():.5f}")
    print(f"    （与系统环境 G0 记录量级一致即视为复现成功）")
except Exception as e:
    print(f"    ❌ 前向失败: {type(e).__name__}: {e}")

print("\n[D] 设备与显存")
print(f"    {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"    显存: 空闲 {free/2**30:.2f} GB / 共 {total/2**30:.2f} GB")
print("=" * 68)

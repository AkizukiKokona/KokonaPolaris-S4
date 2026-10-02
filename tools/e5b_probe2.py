"""E5b · 探针2：到底哪些量化配方可以反向传播（QAT 可行性）

探针1 发现 W4A8（FP8 激活）fake-quant 的 autograd.Function 没有 backward
（因为 FP8 CUDA 扩展未编译 → 走 eager 回退，回退函数缺 backward）。
本探针逐个配方验证 backward，定位「可训练」的子集。

运行：source /d/model/env.sh && "$KP_PY" tools/e5b_probe2.py
"""
import os, gc, json, time, torch
import torch.nn.functional as F
import modelopt.torch.quantization as mtq
from diffusers import SanaTransformer2DModel

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"
E5 = "D:/model/out/e5"
OUT = "D:/model/out/e5b"
DEV = "cuda"
store = torch.load(os.path.join(E5, "embeds.pt"), map_location="cpu")
_pos = store["01_en_scene"]["pos"].to(DEV).to(torch.bfloat16)
_mask = store["01_en_scene"]["pos_mask"].to(DEV)


def calib(model):
    with torch.no_grad():
        for t in [999.0, 500.0, 100.0]:
            x = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16, device=DEV)
            model(hidden_states=x, encoder_hidden_states=_pos, encoder_attention_mask=_mask,
                  timestep=torch.tensor([t], device=DEV), return_dict=False)


def step_ok(name, cfg):
    gc.collect(); torch.cuda.empty_cache()
    rec = {"name": name}
    try:
        m = SanaTransformer2DModel.from_pretrained(
            MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
        ).to(DEV)
        m.enable_gradient_checkpointing()
        mtq.quantize(m, cfg, forward_loop=calib)
        for p in m.parameters():
            p.requires_grad_(True)
        # 只看 fake-quant 可微性：取一个量化后 Linear 的输入输出
        n_lin = sum(1 for x in m.modules() if "QuantLinear" in type(x).__name__)
        x0 = torch.randn(1, 32, 32, 32, dtype=torch.bfloat16, device=DEV)
        eps = torch.randn_like(x0)
        sig = torch.tensor([0.5], device=DEV, dtype=torch.bfloat16)
        xt = (1 - sig) * x0 + sig * eps
        out = m(hidden_states=xt, encoder_hidden_states=_pos, encoder_attention_mask=_mask,
                timestep=torch.tensor([500.0], device=DEV), return_dict=False)[0]
        loss = F.mse_loss(out.float(), (eps - x0).float())
        loss.backward()
        tot = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
                  if p.grad is not None)
        n_with_grad = sum(1 for p in m.parameters() if p.grad is not None)
        # 量化器 amax 是否也有梯度？
        amax_grad = 0
        for nm, mod in m.named_modules():
            a = getattr(mod, "_amax", None)
            if isinstance(a, torch.Tensor) and a.requires_grad and a.grad is not None:
                amax_grad += 1
        rec.update(ok=True, n_quant_linear=n_lin, loss=round(loss.item(), 5),
                   grad_sq_sum=round(tot, 4), n_params_with_grad=n_with_grad,
                   n_amax_with_grad=amax_grad,
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(f"  ✅ {name:30} loss={loss.item():.4f} gn²={tot:.3f} "
              f"grad_params={n_with_grad} amax_grad={amax_grad}")
    except NotImplementedError as e:
        rec.update(ok=False, err="NO_BACKWARD: " + str(e)[:120])
        print(f"  ❌ {name:30} 无 backward: {str(e)[:80]}")
    except Exception as e:
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:150]}")
        print(f"  ⚠️ {name:30} {type(e).__name__}: {str(e)[:100]}")
    try:
        del m
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()
    return rec


print("=" * 78)
print("[C] 各配方 backward 可行性（1024² latent）")
CFGS = [
    ("bf16(no-quant)", None),
    ("NVFP4_DEFAULT_CFG (W4A4)", mtq.NVFP4_DEFAULT_CFG),
    ("NVFP4_FOUR_OVER_SIX_CFG (W4A4/4-6)", mtq.NVFP4_FOUR_OVER_SIX_CFG),
    ("W4A16_NVFP4_CFG", mtq.W4A16_NVFP4_CFG),
    ("W4A8_NVFP4_FP8_CFG (设计档)", mtq.W4A8_NVFP4_FP8_CFG),
    ("FP8_DEFAULT_CFG", mtq.FP8_DEFAULT_CFG),
]
rep = {}
for n, c in CFGS:
    torch.cuda.reset_peak_memory_stats()
    rep[n] = step_ok(n, c)

fp = os.path.join(OUT, "phase0_probe2_backward.json")
with open(fp, "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}")
print("=" * 78)

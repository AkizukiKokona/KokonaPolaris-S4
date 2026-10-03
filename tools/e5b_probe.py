"""KokonaPolaris E5b · 探针：QAT 可行性前置测量（显存 / 单步耗时 / 梯度）

回答三个问题（在正式阶段1之前先摸底）：
  1. NVFP4 fake-quant 之后 forward+backward 需要多少显存？8GB 装得下吗？
  2. 单步耗时多少？（决定阶段2的步数预算）
  3. 量化配置（FOUR_OVER_SIX / W4A8 / W4A4）的 quantize 耗时差多少？

要测量的变量：
  - res: latent 分辨率（1024² → (1,32,32,32)；512² → (1,32,16,16)）
  - grad_ckpt: 梯度检查点开关
  - full_grad: 是否让全部参数 requires_grad（否则只有极少参数训练）

运行：source /d/model/env.sh && "$KP_PY" tools/e5b_probe.py
"""
from kp.paths import MODELS_SANA, OUT
import os, gc, json, time, copy, torch
import torch.nn.functional as F
import modelopt.torch.quantization as mtq
from diffusers import SanaTransformer2DModel

MODEL = str(MODELS_SANA)
E5 = OUT / "e5"
OUT = OUT / "e5b"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

store = torch.load(os.path.join(E5, "embeds.pt"), map_location="cpu")
_pos = store["01_en_scene"]["pos"].to(DEV).to(torch.bfloat16)      # (1,300,2304)
_mask = store["01_en_scene"]["pos_mask"].to(DEV)

def set_weight_block(cfg, blk):
    cfg = copy.deepcopy(cfg)
    for entry in cfg["quant_cfg"]:
        c = entry.get("cfg")
        if not (isinstance(c, dict) and isinstance(c.get("block_sizes"), dict)):
            continue
        c["block_sizes"] = {(-1 if k in (-1, "-1") else k): (blk if k in (-1, "-1") else v)
                            for k, v in c["block_sizes"].items()}
    return cfg

def load_transformer(gc_on=False):
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    if gc_on:
        m.enable_gradient_checkpointing()
    return m

def calib_loop(model, n=3, res=1024):
    tok = 32 if res == 1024 else 16
    with torch.no_grad():
        for t in [999.0, 500.0, 100.0][:n]:
            x = torch.randn(1, 32, tok, tok, dtype=torch.bfloat16, device=DEV)
            model(hidden_states=x, encoder_hidden_states=_pos, encoder_attention_mask=_mask,
                  timestep=torch.tensor([t], device=DEV), return_dict=False)

def fm_step(model, res=1024, grad_ckpt=False):
    """一个 rectified-flow 训练步：返回 loss。"""
    tok = 32 if res == 1024 else 16
    x0 = torch.randn(1, 32, tok, tok, dtype=torch.bfloat16, device=DEV)
    eps = torch.randn_like(x0)
    sigma = torch.rand(1, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    xt = (1 - sigma) * x0 + sigma * eps
    target = eps - x0
    t = (sigma.float() * 1000).to(torch.bfloat16)
    out = model(hidden_states=xt, encoder_hidden_states=_pos, encoder_attention_mask=_mask,
                timestep=t, return_dict=False)[0]
    return F.mse_loss(out.float(), target.float())

def mem():
    return torch.cuda.max_memory_allocated() / 2**30

report = {}

print("=" * 78)
print("[A] 各量化配方的 quantize 耗时（1024² 校准 3 步）")
CFGS = [
    ("W4A8_NVFP4_FP8_CFG", mtq.W4A8_NVFP4_FP8_CFG),
    ("NVFP4_DEFAULT_CFG", mtq.NVFP4_DEFAULT_CFG),
    ("NVFP4_FOUR_OVER_SIX_CFG", mtq.NVFP4_FOUR_OVER_SIX_CFG),
]
report["quantize_time"] = {}
for name, cfg in CFGS:
    gc.collect(); torch.cuda.empty_cache()
    m = load_transformer()
    t0 = time.time()
    mtq.quantize(m, cfg, forward_loop=lambda mm: calib_loop(mm, 3, 1024))
    dt = time.time() - t0
    nq = sum(1 for x in m.modules() if "QuantLinear" in type(x).__name__)
    print(f"    {name:28} {dt:7.1f}s   QuantLinear={nq}")
    report["quantize_time"][name] = round(dt, 1)
    del m; gc.collect(); torch.cuda.empty_cache()

print("\n[B] forward+backward 显存/耗时矩阵")
report["train"] = {}


def trial(tag, quant_cfg, res, grad_ckpt, full_grad, trainable_substr=None, steps=3):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag, "res": res, "grad_ckpt": grad_ckpt, "full_grad": full_grad}
    try:
        m = load_transformer(gc_on=grad_ckpt)
        if quant_cfg is not None:
            mtq.quantize(m, quant_cfg, forward_loop=lambda mm: calib_loop(mm, 3, res))
        # 冻结策略
        for p in m.parameters():
            p.requires_grad_(False)
        if full_grad:
            for p in m.parameters():
                p.requires_grad_(True)
        elif trainable_substr:
            np = 0
            for n, p in m.named_parameters():
                if any(s in n for s in trainable_substr):
                    p.requires_grad_(True); np += p.numel()
            rec["trainable_params"] = np
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rec["n_trainable"] = ntr
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-6)
        m.train()
        base_mem = torch.cuda.memory_allocated() / 2**30
        # 第一步（建立优化器状态/图）
        t0 = time.time()
        loss = fm_step(m, res, grad_ckpt)
        loss.backward()
        gn = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
                 if p.grad is not None) ** 0.5
        opt.step(); opt.zero_grad(set_to_none=True)
        t_first = time.time() - t0
        peak1 = mem()
        # 若干稳态步
        t1 = time.time()
        for _ in range(steps):
            loss = fm_step(m, res, grad_ckpt)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        t_avg = (time.time() - t1) / steps
        peak = mem()
        rec.update(ok=True, base_mem_gb=round(base_mem, 2), first_step_s=round(t_first, 2),
                   step_s=round(t_avg, 2), peak_gb=round(peak, 2),
                   loss=round(loss.item(), 5), grad_norm=round(gn, 4))
        print(f"    ✅ {tag:34} peak={peak:5.2f}GB  step={t_avg:5.2f}s  loss={loss.item():.5f}  gn={gn:.3f}")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(ok=False, err="OOM:" + str(e).split("\n")[0][:120])
        print(f"    ❌ {tag:34} OOM")
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}")
        print(f"    ⚠️ {tag:34} {type(e).__name__}: {str(e)[:120]}")
    report["train"][tag] = rec
    try:
        del m, opt
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()


# 先测最激进的（全参、无检查点、1024²），再逐步降级
trial("bf16_full_1024", None, 1024, False, True)
trial("bf16_full_1024_gc", None, 1024, True, True)
trial("W4A8_full_1024_gc", mtq.W4A8_NVFP4_FP8_CFG, 1024, True, True)
trial("W4A8_full_512_gc", mtq.W4A8_NVFP4_FP8_CFG, 512, True, True)
trial("W4A8_full_512", mtq.W4A8_NVFP4_FP8_CFG, 512, False, True)

fp = os.path.join(OUT, "phase0_probe.json")
with open(fp, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}")
print("=" * 78)

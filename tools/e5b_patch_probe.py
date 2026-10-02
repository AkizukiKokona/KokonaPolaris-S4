"""E5b · 任务1验证 + 任务2可行性：修掉 ModelOpt 的 FP8SDPA 缺 backward 的 bug。

真根因（实测，见 out/e5b/msvc_retest.json）：
  ModelOpt 的 diffusers 插件把 `F.scaled_dot_product_attention` 换成了
  `_quantized_sdpa`，它内部调用 `FP8SDPA.apply(...)`。
  而 `FP8SDPA`（modelopt/torch/quantization/plugins/diffusion/diffusers.py:214）
  是一个只实现了 forward/symbolic 的 autograd.Function，**没有 backward**。
  ⇒ 只要模型里有 diffusers 的 Attention（Sana 就有），反向必断；
     与量化器是否 enable 无关、与 CUDA 扩展是否编译无关（解释了 B 路也断）。

修复（前向严格等价，因为 FP8SDPA.forward 本来就是直接调 original SDPA）：
  把模块级 `FP8SDPA` 换成一个直接调用 original_scaled_dot_product_attention
  的普通函数 ⇒ 图可微，STE 语义由原 SDPA 的真实梯度承担。

本脚本：
  mode=confirm : 验证 patch 后 W4A8 全参 backward 通不通（512²，gc）
  mode=lora    : 在 ModelOpt 量化模型上挂 LoRA 旁路，跑 5 步（测显存/耗时/参数量）
"""
import os, sys, gc, json, time, math, argparse, traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "D:/model/out/e5b"
os.makedirs(OUT, exist_ok=True)
REP = {"mode": None}


def _dump(name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(REP, f, ensure_ascii=False, indent=2)


# ---------------- 修复 patch ----------------
def patch_fp8sdpa():
    import modelopt.torch.quantization.plugins.diffusion.diffusers as md
    orig = md.original_scaled_dot_product_attention

    def _sdpa_ok(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                 scale=None, *a, **k):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale)

    class _SDPAPatch:
        apply = staticmethod(_sdpa_ok)

    md.FP8SDPA = _SDPAPatch
    return md.FP8SDPA


def disable_modelopt_cuda_ext():
    """本机 ModelOpt CUDA 扩展编译不了（nvcc 13.2 vs torch cu128 不匹配），
    且该扩展只是加速，数学等价 ⇒ 直接强制走纯 Python 回退，省掉每次 ~30s 编译尝试。"""
    none = lambda *a, **k: None  # noqa: E731
    import modelopt.torch.quantization.extensions as mqext
    import modelopt.torch.quantization.tensor_quant as mqtq
    for m in (mqext, mqtq):
        m.get_cuda_ext = none
        m.get_cuda_ext_fp8 = none
        m.get_cuda_ext_mx = none


# ---------------- 模型 / 数据 ----------------
import modelopt.torch.quantization as mtq  # noqa: E402
from diffusers import SanaTransformer2DModel  # noqa: E402

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"
DEV = "cuda"
store = torch.load("D:/model/out/e5/embeds.pt", map_location="cpu")
POS = store["01_en_scene"]["pos"].to(DEV).to(torch.bfloat16)
MASK = store["01_en_scene"]["pos_mask"].to(DEV)


def load_quant(cfg, res=512, gc_on=True):
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    if gc_on:
        m.enable_gradient_checkpointing()
    T = res // 32

    def calib(mm):
        with torch.no_grad():
            for t in (999.0, 500.0, 100.0):
                x = torch.randn(1, 32, T, T, dtype=torch.bfloat16, device=DEV)
                mm(hidden_states=x, encoder_hidden_states=POS, encoder_attention_mask=MASK,
                   timestep=torch.tensor([t], device=DEV), return_dict=False)

    t0 = time.time()
    mtq.quantize(m, cfg, forward_loop=calib)
    return m, round(time.time() - t0, 1)


def fm_batch(res, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    T = res // 32
    x0 = torch.randn(1, 32, T, T, dtype=torch.bfloat16, device=DEV)
    eps = torch.randn_like(x0)
    sig = torch.rand(1, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    xt = (1 - sig) * x0 + sig * eps
    t = (sig.float() * 1000).to(torch.bfloat16)
    return xt, t, (eps - x0)


def fwd(m, xt, t):
    return m(hidden_states=xt, encoder_hidden_states=POS, encoder_attention_mask=MASK,
             timestep=t, return_dict=False)[0]


# ---------------- LoRA 旁路（挂在 ModelOpt 的 QuantLinear 上） ----------------
class LoRALinearWrap(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0):
        super().__init__()
        self.base = base
        w = base.weight
        d_out, d_in = w.shape
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=torch.float32, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=torch.float32, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank

    def forward(self, x, *a, **kw):
        y = self.base(x, *a, **kw)
        xf = x.to(torch.float32)
        l = F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling
        if isinstance(y, tuple):
            return (y[0] + l.to(y[0].dtype),) + tuple(y[1:])
        return y + l.to(y.dtype)


def wrap_lora(m, rank=8, alpha=16.0):
    n = 0
    for name, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if type(child).__name__ == "QuantLinear":
                setattr(mod, cname, LoRALinearWrap(child, rank=rank, alpha=alpha))
                n += 1
    return n


def _grad_probe_params(m):
    """只让第 0 个 transformer block + proj_out 要梯度：
    反向仍必须穿过全部 block（含 attention），但只存极少梯度 → 8GB 不会 OOM。
    这是「整图可微性」的判定，而非全参训练可行性。"""
    for p in m.parameters():
        p.requires_grad_(False)
    chosen = []
    blocks = getattr(m, "transformer_blocks", None)
    if blocks is not None and len(blocks) > 0:
        chosen += list(blocks[0].parameters())
    if hasattr(m, "proj_out"):
        chosen += list(m.proj_out.parameters())
    n = 0
    for p in chosen:
        p.requires_grad_(True)
        n += p.numel()
    return n


def confirm(res=512):
    REP["mode"] = "confirm"
    patch_fp8sdpa()
    disable_modelopt_cuda_ext()
    REP["patched"] = True
    # bf16 对照
    mb = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    mb.enable_gradient_checkpointing()
    n = _grad_probe_params(mb)
    REP["probe_params"] = n
    torch.cuda.reset_peak_memory_stats()
    xt, t, tgt = fm_batch(res)
    try:
        loss = F.mse_loss(fwd(mb, xt, t).float(), tgt.float())
        loss.backward()
        REP["bf16_backward_ok"] = True
        REP["bf16_loss"] = round(loss.item(), 5)
    except Exception as e:
        REP["bf16_backward_ok"] = False
        REP["bf16_err"] = f"{type(e).__name__}: {str(e)[:200]}"
    REP["bf16_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"  bf16: backward_ok={REP.get('bf16_backward_ok')} {REP.get('bf16_err','')} "
          f"peak={REP['bf16_peak_gb']}GB", flush=True)
    del mb; gc.collect(); torch.cuda.empty_cache()

    for name, cfg in (("W4A8", mtq.W4A8_NVFP4_FP8_CFG), ("W4A4", mtq.NVFP4_DEFAULT_CFG)):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {}
        t0 = time.time()
        try:
            m, qt = load_quant(cfg, res=res)
            rec["quantize_s"] = qt
            _grad_probe_params(m)
            xt, t, tgt = fm_batch(res)
            loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
            loss.backward()
            gn = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
                     if p.grad is not None) ** 0.5
            rec.update(ok=True, loss=round(loss.item(), 5), grad_norm=round(gn, 4))
            del m
        except Exception as e:
            rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:200]}")
            rec["traceback"] = traceback.format_exc()[-1500:]
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        REP[f"{name}_fullbackward"] = rec
        print(f"  {name}: ok={rec.get('ok')} {rec.get('err','')} peak={rec['peak_gb']}GB", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    _dump("fp8sdpa_patch_confirm.json")


def lora_smoke(res=512, steps=5, rank=8):
    REP["mode"] = "lora"
    patch_fp8sdpa()
    disable_modelopt_cuda_ext()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m, qt = load_quant(mtq.W4A8_NVFP4_FP8_CFG, res=res)
    nw = wrap_lora(m, rank=rank)
    REP["wrapped"] = nw
    REP["quantize_s"] = qt
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, LoRALinearWrap):
            mod.lora_A.requires_grad_(True)
            mod.lora_B.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    REP["n_trainable"] = ntr
    REP["trainable_pct"] = round(ntr / 1.6045e9 * 100, 4)
    opt = torch.optim.AdamW(params, lr=1e-4)
    m.train()
    hist = []
    t0 = time.time()
    try:
        for i in range(steps):
            xt, t, tgt = fm_batch(res, seed=2000 + i)
            loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            hist.append(round(loss.item(), 5))
            print(f"    step {i+1} loss={loss.item():.5f} "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB "
                  f"{(time.time()-t0)/(i+1):.2f}s/step", flush=True)
        REP["ok"] = True
    except Exception as e:
        REP["ok"] = False
        REP["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        REP["traceback"] = traceback.format_exc()[-2000:]
    REP["hist"] = hist
    REP["step_s"] = round((time.time() - t0) / max(1, len(hist)), 3)
    REP["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"  lora: ok={REP.get('ok')} ntr={ntr} step_s={REP['step_s']} peak={REP['peak_gb']}GB", flush=True)
    _dump("fp8sdpa_patch_lora.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["confirm", "lora"])
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    a = ap.parse_args()
    print(f"[patch-probe] mode={a.mode} res={a.res}", flush=True)
    if a.mode == "confirm":
        confirm(res=a.res)
    else:
        lora_smoke(res=a.res, steps=a.steps, rank=a.rank)
    print(json.dumps(REP, ensure_ascii=False)[:1500], flush=True)

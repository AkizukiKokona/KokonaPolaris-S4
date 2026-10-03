"""E5b · 公共件：FP8SDPA 修复 / 量化加载 / LoRA 旁路 / 数据。

⚠️ 核心结论（见 out/e5b/msvc_retest.json）：
  ModelOpt 反向失败与 MSVC、与 CUDA 扩展**无关**。
  真凶是 `modelopt/torch/quantization/plugins/diffusion/diffusers.py:214`
  的 `FP8SDPA(autograd.Function)`：只写了 forward/symbolic，**没有 backward**。
  它被 `_quantized_sdpa`（替换 F.scaled_dot_product_attention）调用，
  只要模型里有 diffusers Attention（Sana 有），反向必断；
  与量化器 enable/disable 无关（所以"量化器全关"也断）。
  ⇒ 只需把模块级 FP8SDPA 换成一个直接调 original SDPA 的普通函数（前向严格等价）。
"""
import os, sys

# ⚠️ 入口自举：`python tools/e5b_g1_gen.py` 时 `sys.path[0]` 是 **tools/**，
#    不是仓库根 ⇒ `import kp.paths` 会 `ModuleNotFoundError: No module named 'kp'`。
#    `env.sh` 并没有设 PYTHONPATH，所以这一行是必需的（照 `tools/onboard.py:18` 的做法）。
#    ⛔ 用相对本文件的父目录，**不写死绝对路径**（`tools/portable_paths.py --verify` 须恒为 0）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kp.paths import MODELS_SANA, OUT
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = str(MODELS_SANA)
OUT = OUT / "e5b"
CKPT = os.path.join(OUT, "qat_ckpt")
DEV = "cuda"
os.makedirs(CKPT, exist_ok=True)


# ---------------------------------------------------------------- 修复
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


def disable_modelopt_cuda_ext():
    """本机 nvcc=12.4/13.2/13.3，torch=cu128，扩展编译不匹配；扩展只是加速、数学等价 ⇒ 直接走回退。"""
    none = lambda *a, **k: None  # noqa: E731
    import modelopt.torch.quantization.extensions as mqext
    import modelopt.torch.quantization.tensor_quant as mqtq
    for m in (mqext, mqtq):
        m.get_cuda_ext = none
        m.get_cuda_ext_fp8 = none
        m.get_cuda_ext_mx = none


def prepare():
    patch_fp8sdpa()
    disable_modelopt_cuda_ext()


# ---------------------------------------------------------------- 数据
_EMBEDS = os.path.join(OUT, "g1_embeds.pt")
if not os.path.exists(_EMBEDS):
    _EMBEDS = OUT / "e5/embeds.pt"
_STORE = torch.load(_EMBEDS, map_location="cpu")
_KEYS = list(_STORE.keys())


def pick_text(i):
    e = _STORE[_KEYS[i % len(_KEYS)]]
    return (e["pos"].to(DEV).to(torch.bfloat16), e["pos_mask"].to(DEV))


def tok(res):
    return res // 32


def fm_batch(res, seed=None, b=1):
    if seed is not None:
        torch.manual_seed(seed)
    T = tok(res)
    x0 = torch.randn(b, 32, T, T, dtype=torch.bfloat16, device=DEV)
    eps = torch.randn_like(x0)
    sig = torch.rand(1, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    xt = (1 - sig) * x0 + sig * eps
    t = (sig.float() * 1000).to(torch.bfloat16)
    return xt, t, (eps - x0)


def fwd(m, xt, t, pos, mask):
    return m(hidden_states=xt, encoder_hidden_states=pos, encoder_attention_mask=mask,
             timestep=t, return_dict=False)[0]


# ---------------------------------------------------------------- LoRA 旁路
class LoRALinearWrap(nn.Module):
    """包住 ModelOpt 的 QuantLinear：输出 = 量化主干(x) + B·A·x (fp32, 不量化)。"""

    def __init__(self, base, rank=16, alpha=32.0):
        super().__init__()
        self.base = base
        w = base.weight
        d_out, d_in = w.shape
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=torch.float32, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=torch.float32, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.lora_on = True

    def forward(self, x, *a, **kw):
        y = self.base(x, *a, **kw)
        if not self.lora_on:
            return y
        xf = x.to(torch.float32)
        l = F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling
        if isinstance(y, tuple):
            return (y[0] + l.to(y[0].dtype),) + tuple(y[1:])
        return y + l.to(y.dtype)


def wrap_lora(m, rank=16, alpha=32.0):
    n = 0
    for name, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if type(child).__name__ == "QuantLinear":
                setattr(mod, cname, LoRALinearWrap(child, rank=rank, alpha=alpha))
                n += 1
    return n


def collect_quant_toggles(m):
    wraps = [x for x in m.modules() if isinstance(x, LoRALinearWrap)]
    qs = [x for x in m.modules()
          if hasattr(x, "disable") and hasattr(x, "enable") and hasattr(x, "is_enabled")]
    return wraps, qs


# ---------------------------------------------------------------- 建模
def build_bf16(gc_on=True):
    from diffusers import SanaTransformer2DModel
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    if gc_on:
        m.enable_gradient_checkpointing()
    return m


def load_quant(cfg, res=512, gc_on=True):
    import modelopt.torch.quantization as mtq
    prepare()
    m = build_bf16(gc_on=gc_on)
    T = tok(res)

    def calib(mm):
        with torch.no_grad():
            for i, t in enumerate((999.0, 500.0, 100.0)):
                pos, mask = pick_text(i)
                x = torch.randn(1, 32, T, T, dtype=torch.bfloat16, device=DEV)
                mm(hidden_states=x, encoder_hidden_states=pos, encoder_attention_mask=mask,
                   timestep=torch.tensor([t], device=DEV), return_dict=False)

    t0 = time.time()
    mtq.quantize(m, cfg, forward_loop=calib)
    return m, round(time.time() - t0, 1)


def build_arm(arm, res=512, rank=16, alpha=32.0, gc_on=True, adapter=None):
    """arm ∈ {bf16, PTQ-W4A8, PTQ-W4A4, TRAIN-W4A8, TRAIN-W4A4}
    adapter: 训练臂的 ckpt 路径（含 sd）"""
    import modelopt.torch.quantization as mtq
    prepare()
    if arm == "bf16":
        return build_bf16(gc_on=gc_on)
    cfg = mtq.W4A8_NVFP4_FP8_CFG if "W4A8" in arm else mtq.NVFP4_DEFAULT_CFG
    m, _ = load_quant(cfg, res=res, gc_on=gc_on)
    if arm.startswith("TRAIN"):
        n = wrap_lora(m, rank=rank, alpha=alpha)
        if adapter:
            ck = torch.load(adapter, map_location="cpu")
            sd = ck["sd"] if "sd" in ck else ck
            tgt = dict(m.named_parameters())
            miss = 0
            for k, v in sd.items():
                if k in tgt:
                    with torch.no_grad():
                        tgt[k].copy_(v.to(tgt[k].device).to(tgt[k].dtype))
                else:
                    miss += 1
            print(f"[build_arm] 载入 adapter {adapter} 缺失 {miss} 项", flush=True)
    return m

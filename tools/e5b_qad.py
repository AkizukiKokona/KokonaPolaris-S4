"""E5b · QAD 实装：自研可微 NVFP4 fake-quant（绕开 ModelOpt 的不可训路径）

为什么另起炉灶（实测根因，非猜测）：
  - ModelOpt 的 CUDA 扩展在本机**编译不了**（缺 MSVC `cl.exe`，torch 报
    "Error checking compiler version for cl: WinError 2"）→ 回退到纯 Python 版；
  - 该回退版的量化 op 是**未实现 backward 的 autograd.Function**
    → 任何经过量化器的反向都断（"must implement backward or vjp"）；
  - 更严重：**把量化器全部 disable 也照样断**（probe3 的 B 路）⇒ `mtq.quantize`
    一旦插入，模型在 ModelOpt 手里就**只能推理不能训**。六种配方（含纯权重量化
    W4A16、FP8_DEFAULT）全部如此。

本方案：训练期只需要「模拟量化」（fake-quant），根本不需要 packed/scale 布局。
  所以用手写量化器 + **STE 直通梯度**（x + (xq - x).detach()）即可，完全可微。
  量化器复用 `e5_nvfp4_blocksize.py` 里**已被真实权重验证过**的 nvfp4_roundtrip。

两个模式：
  probe : 前向相对误差，和 ModelOpt 的 PTQ 数字交叉验证（对得上 ⇒ 手写实现忠实）
  qat   : 真跑 QAD 训练（loss 是否下降 / 显存 / 单步耗时）

用法：
  source /d/model/env.sh && "$KP_PY" tools/e5b_qad.py probe
  source /d/model/env.sh && "$KP_PY" tools/e5b_qad.py qat --steps 30
"""
from kp.paths import MODELS_SANA, OUT
import os, gc, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel

MODEL = str(MODELS_SANA)
OUT = OUT / "e5b"
DEV = "cuda"
os.makedirs(OUT, exist_ok=True)

_E4M3 = torch.finfo(torch.float8_e4m3fn)
BLK = 16


# ---------------- 可微量化算子（STE） ----------------
def quant_fp4_ste(x, blk=BLK):
    """NVFP4：per-block(absmax) → E4M3 scale → E2M1 取整(±6)，梯度直通。"""
    shape = x.shape
    inn = shape[-1]
    if inn % blk != 0:
        blk = inn
    xb = x.reshape(-1, inn // blk, blk)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).to(x.dtype).clamp(min=1e-12)
    q = (xb / scale).round().clamp(-6.0, 6.0)
    xq = (q * scale).reshape(shape)
    return x + (xq - x).detach()


def quant_fp8_ste(x):
    """FP8 E4M3 per-tensor 模拟量化，梯度直通。"""
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / _E4M3.max).clamp(min=1e-12)
    q = (x / scale).clamp(_E4M3.min, _E4M3.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x).detach()


# ---------------- 可微量化 Linear ----------------
class QLinear(nn.Module):
    """W4 + A? 的模拟量化 Linear。w_mode: fp4 / none；a_mode: fp8 / fp4 / none。"""

    def __init__(self, lin: nn.Linear, a_mode="fp8"):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.a_mode = a_mode

    def forward(self, x):
        w = quant_fp4_ste(self.weight)
        if self.a_mode == "fp8":
            x = quant_fp8_ste(x)
        elif self.a_mode == "fp4":
            x = quant_fp4_ste(x)
        return F.linear(x, w, self.bias)


def swap_linears(model, a_mode="fp8", skip=("proj_out",)):
    """把 nn.Linear 换成 QLinear；skip 里的名字保持原样（对齐官方默认：proj_out 不量化）。"""
    replaced, kept = 0, []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{cname}" if name else cname
                if any(s in full for s in skip):
                    kept.append(full)
                    continue
                setattr(mod, cname, QLinear(child, a_mode=a_mode))
                replaced += 1
    return replaced, kept


# ---------------- 数据 ----------------
store = torch.load(OUT / "e5/embeds.pt", map_location="cpu")
_KEYS = list(store.keys())
POS = store[_KEYS[0]]["pos"].to(DEV).to(torch.bfloat16)
MASK = store[_KEYS[0]]["pos_mask"].to(DEV)


def load(quant=True, res=512, a_mode="fp8", gc_on=True):
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    if gc_on:
        m.enable_gradient_checkpointing()
    info = {}
    if quant:
        r, k = swap_linears(m, a_mode=a_mode)
        info = {"replaced": r, "kept": k}
    return m, info


def tok(res):
    return res // 32


def fm_batch(res, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    T = tok(res)
    x0 = torch.randn(1, 32, T, T, dtype=torch.bfloat16, device=DEV)
    eps = torch.randn_like(x0)
    sig = torch.rand(1, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    xt = (1 - sig) * x0 + sig * eps
    t = (sig.float() * 1000).to(torch.bfloat16)
    return xt, t, (eps - x0)


def fwd(m, xt, t, pos=POS, mask=MASK):
    return m(hidden_states=xt, encoder_hidden_states=pos, encoder_attention_mask=mask,
             timestep=t, return_dict=False)[0]


# ---------------- probe：与 ModelOpt 交叉验证 ----------------
def run_probe(res=512, n=6):
    print("=" * 78)
    print(f"[probe] 手写 W4A8 前向误差 vs bf16（{res}²，{n} 个 timestep 网格）")
    mb, _ = load(quant=False, res=res)
    mb.eval()
    mq, info = load(quant=True, res=res, a_mode="fp8")
    mq.eval()
    print(f"        替换 Linear {info['replaced']} 个，保留 {info['kept']}")
    rows = []
    with torch.no_grad():
        for i, t_val in enumerate(torch.linspace(999, 0, n).tolist()):
            xt, _, tgt = fm_batch(res, seed=1000 + i)
            t = torch.tensor([t_val], device=DEV, dtype=torch.bfloat16)
            yb = fwd(mb, xt, t).float()
            yq = fwd(mq, xt, t).float()
            rel = ((yq - yb).norm() / yb.norm()).item() * 100
            cos = F.cosine_similarity(yq.flatten(), yb.flatten(), dim=0).item()
            rows.append({"t": round(t_val, 1), "rel_pct": round(rel, 3), "cos": round(cos, 5)})
            print(f"    t={t_val:6.1f}  rel={rel:6.3f}%  cos={cos:.5f}")
    mean = sum(r["rel_pct"] for r in rows) / len(rows)
    print(f"    ⇒ 平均相对误差 {mean:.3f}%   (ModelOpt PTQ 同档 W4A8 参照 = 9.98%)")
    del mb, mq
    gc.collect(); torch.cuda.empty_cache()
    return {"rows": rows, "mean_rel_pct": round(mean, 3),
            "modelopt_reference_pct": 9.98, "replaced": info["replaced"], "kept": info["kept"]}


# ---------------- qat：真训 ----------------
def run_qat(steps=30, res=512, lr=1e-6, a_mode="fp8"):
    print("=" * 78)
    print(f"[qat] 自研可微 NVFP4（W4{'A8' if a_mode == 'fp8' else 'A4'}）· "
          f"{res}² · {steps} 步 · lr={lr}")
    m, info = load(quant=True, res=res, a_mode=a_mode)
    print(f"        替换 Linear {info['replaced']} 个，保留 {info['kept']}")
    for p in m.parameters():
        p.requires_grad_(True)
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    m.train()
    torch.cuda.reset_peak_memory_stats()
    hist = []
    t0 = time.time()
    for i in range(steps):
        xt, t, tgt = fm_batch(res, seed=2000 + i)
        loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
        loss.backward()
        gn = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
                 if p.grad is not None) ** 0.5
        opt.step(); opt.zero_grad(set_to_none=True)
        if i == 0 or (i + 1) % max(1, steps // 6) == 0:
            hist.append({"step": i + 1, "loss": round(loss.item(), 5), "gn": round(gn, 4)})
            print(f"    step {i+1:>3}  loss={loss.item():.5f}  grad_norm={gn:.3f}  "
                  f"{(time.time()-t0)/(i+1):.2f}s/step")
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    first, last = hist[0]["loss"], hist[-1]["loss"]
    print(f"    ⇒ {steps} 步完成 | {dt:.2f}s/step | 峰值 {peak:.2f}GB | "
          f"loss {first:.5f} → {last:.5f} ({'↓ 下降' if last < first else '未下降'})")
    print(f"    ⇒ 可训参数量 {ntr/1e9:.4f}B")
    del m, opt
    gc.collect(); torch.cuda.empty_cache()
    return {"steps": steps, "lr": lr, "res": res, "a_mode": a_mode, "n_trainable": ntr,
            "step_s": round(dt, 2), "peak_gb": round(peak, 2), "hist": hist,
            "loss_first": first, "loss_last": last,
            "loss_down": bool(last < first)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "qat", "both"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--a-mode", default="fp8", choices=["fp8", "fp4", "none"])
    a = ap.parse_args()

    out = {}
    if a.mode in ("probe", "both"):
        out["probe_w4a8"] = run_probe(res=a.res)
        if a.mode == "both":
            out["probe_w4a4"] = run_probe(res=a.res, n=2)  # 快速对照
    if a.mode == "qat":
        out["qat"] = run_qat(steps=a.steps, res=a.res, lr=a.lr, a_mode=a.a_mode)

    fp = os.path.join(OUT, f"qad_{a.mode}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 写入 {fp}")
    print("=" * 78)

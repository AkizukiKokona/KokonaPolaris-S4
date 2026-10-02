"""E5b · 步1 修复：可训 NVFP4 QAD（adapter 路线为主）

背景（已实测确认，见 out/e5b/qat_repro.*）：
  原 `e5b_qad.py qat` 用 AdamW 训 1.604B 全参 → 优化器状态 2×1.604e9×4B ≈ 12.8GB，
  8GB 卡装不下 → 峰值 allocated 15.31GB（靠驱动 sysmem fallback 硬撑），
  每步耗时 6.39 → 11.80 → 13.73s 递增（换页），根本无法规模训练。

两条修复路线：
  (a) sgd_full : 全参 + SGD(无动量, momentum=0, weight_decay=0) → 优化器零状态，可小步数对照
  (b) adapter  : ⭐ 冻结主干权重，只训低秩旁路 lora_A/lora_B（LoRA 式，B 初始化为 0
                 ⇒ 起点等价 PTQ，训练期把量化损失补回来）。
                 优化器状态 ~2×(可训参数量)×4B，量级 MB —— 与设计稿 Δ-Pack/能力包同构。

QAD 目标：flow-matching MSE + λ·蒸馏(self-distill)
  teacher = 同一份权重、量化旁路关闭（等价 bf16 前向），no_grad；
  student = 量化开启（STE 模拟 NVFP4）。
  loss = MSE(student, fm_target) + λ·MSE(student, teacher.detach())
  —— teacher 不引入第二个模型，只是同权重走一遍不量化的前向，几乎不额外占显存。

用法：
  source /d/model/env.sh && "$KP_PY" tools/e5b_qad2.py adapter --steps 300 --res 512
  source /d/model/env.sh && "$KP_PY" tools/e5b_qad2.py sgd_full --steps 30 --res 512
  source /d/model/env.sh && "$KP_PY" tools/e5b_qad2.py adapter --steps 5 --res 512 --tag smoke
"""
import os, gc, json, time, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"
OUT = "D:/model/out/e5b"
CKPT = os.path.join(OUT, "qat_ckpt")
DEV = "cuda"
os.makedirs(CKPT, exist_ok=True)

_E4M3 = torch.finfo(torch.float8_e4m3fn)
BLK = 16


# ---------------- 可微量化算子（STE，与 e5b_qad.py 一致） ----------------
def quant_fp4_ste(x, blk=BLK):
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
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / _E4M3.max).clamp(min=1e-12)
    q = (x / scale).clamp(_E4M3.min, _E4M3.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x).detach()


class QADLinear(nn.Module):
    """W4(冻结) + A?(STE) + 可训低秩旁路。

    quant_on=False 时完全旁路量化（teacher / PTQ-off 对照）。
    lora_B 初始化为 0 ⇒ 初始 forward == 纯 PTQ。
    """

    def __init__(self, lin: nn.Linear, a_mode="fp8", rank=8, alpha=16.0):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.a_mode = a_mode
        self.rank = rank
        self.scaling = alpha / rank
        self.quant_on = True
        self.lora_on = True
        if rank > 0:
            d_out, d_in = lin.weight.shape
            dev = lin.weight.device
            self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=torch.float32, device=dev))
            self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x):
        w = quant_fp4_ste(self.weight) if self.quant_on else self.weight
        if self.quant_on:
            if self.a_mode == "fp8":
                x = quant_fp8_ste(x)
            elif self.a_mode == "fp4":
                x = quant_fp4_ste(x)
        y = F.linear(x, w, self.bias)
        if self.lora_A is not None and self.lora_on:
            xf = x.to(torch.float32)
            lora = F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling
            y = y + lora.to(y.dtype)
        return y


def swap_linears(model, a_mode="fp8", rank=8, alpha=16.0, skip=("proj_out",)):
    replaced, kept = 0, []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{cname}" if name else cname
                if any(s in full for s in skip):
                    kept.append(full)
                    continue
                setattr(mod, cname, QADLinear(child, a_mode=a_mode, rank=rank, alpha=alpha))
                replaced += 1
    return replaced, kept


def set_quant(m, on):
    n = 0
    for mod in m.modules():
        if isinstance(mod, QADLinear):
            mod.quant_on = on
            n += 1
    return n


def set_teacher(m):
    """teacher = 纯 bf16 参考：量化旁路 + LoRA 旁路全关。"""
    for mod in m.modules():
        if isinstance(mod, QADLinear):
            mod.quant_on = False
            mod.lora_on = False


def set_student(m, quant_on=True):
    """student = 部署路径：量化开 + LoRA 开。"""
    for mod in m.modules():
        if isinstance(mod, QADLinear):
            mod.quant_on = quant_on
            mod.lora_on = True


# ---------------- 数据 ----------------
store = torch.load("D:/model/out/e5/embeds.pt", map_location="cpu")
_KEYS = list(store.keys())


def pick_text(idx=0):
    k = _KEYS[idx % len(_KEYS)]
    e = store[k]
    pos = e["pos"].to(DEV).to(torch.bfloat16)
    mask = e["pos_mask"].to(DEV)
    return pos, mask


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


def fwd(m, xt, t, pos, mask):
    return m(hidden_states=xt, encoder_hidden_states=pos, encoder_attention_mask=mask,
             timestep=t, return_dict=False)[0]


def build(a_mode="fp8", rank=8, alpha=16.0, gc_on=True):
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    if gc_on:
        m.enable_gradient_checkpointing()
    r, k = swap_linears(m, a_mode=a_mode, rank=rank, alpha=alpha)
    return m, r, k


# ---------------- 训练 ----------------
def run(route, steps=300, res=512, lr=1e-4, a_mode="fp8", rank=8, alpha=16.0,
        lam=0.5, tag="run", warmup=5, log_every=None):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    m, replaced, kept = build(a_mode=a_mode, rank=rank, alpha=alpha)
    for p in m.parameters():
        p.requires_grad_(False)
    if route == "adapter":
        for mod in m.modules():
            if isinstance(mod, QADLinear) and mod.lora_A is not None:
                mod.lora_A.requires_grad_(True)
                mod.lora_B.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        opt_kind = "AdamW"
    else:  # sgd_full
        for p in m.parameters():
            p.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0, nesterov=False)
        opt_kind = "SGD(no-momentum)"

    ntr = sum(p.numel() for p in params)
    opt_bytes = 2 * ntr * 4  # AdamW 两个状态（fp32）；SGD 实际为 0
    m.train()
    hist = []
    t0 = time.time()
    step_time = None
    for i in range(steps):
        pos, mask = pick_text(i % len(_KEYS))
        xt, t, tgt = fm_batch(res, seed=2000 + i)
        with torch.no_grad():
            set_teacher(m)
            teacher = fwd(m, xt, t, pos, mask).float()
        set_student(m)
        out = fwd(m, xt, t, pos, mask)
        loss_fm = F.mse_loss(out.float(), tgt.float())
        loss_kd = F.mse_loss(out.float(), teacher.detach()) if lam > 0 else torch.zeros((), device=DEV)
        loss = loss_fm + lam * loss_kd
        loss.backward()
        gn = sum((p.grad.detach().float() ** 2).sum().item() for p in params
                 if p.grad is not None) ** 0.5
        opt.step(); opt.zero_grad(set_to_none=True)
        step_time = time.time() - t0
        if i >= warmup:
            pass
        if i == 0 or (i + 1) % (log_every or max(1, steps // 10)) == 0:
            hist.append({"step": i + 1, "loss": round(loss.item(), 5),
                         "fm": round(loss_fm.item(), 5), "kd": round(loss_kd.item(), 5),
                         "gn": round(gn, 4), "s": round(step_time / (i + 1), 3)})
    total_s = time.time() - t0
    # 稳态 s/step：完全复刻训练一步（teacher+student+backward+step），再测 3 步取平均
    if route == "adapter":
        tp = []
        for j in range(3):
            pos, mask = pick_text(j)
            xt, t, tgt = fm_batch(res, seed=9000 + j)
            tt = time.time()
            with torch.no_grad():
                set_teacher(m)
                teacher = fwd(m, xt, t, pos, mask).float()
            set_student(m)
            out = fwd(m, xt, t, pos, mask)
            loss = F.mse_loss(out.float(), tgt.float()) + lam * F.mse_loss(out.float(), teacher.detach())
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            tp.append(time.time() - tt)
        steady = sum(tp) / len(tp)
    else:
        steady = total_s / steps

    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    first = hist[0]["loss"] if hist else None
    last = hist[-1]["loss"] if hist else None

    ckpt_path = None
    if route == "adapter":
        sd = {k: v.detach().to("cpu") for k, v in m.state_dict().items()
              if k.endswith("lora_A") or k.endswith("lora_B")}
        cfg = {"rank": rank, "alpha": alpha, "a_mode": a_mode, "skip": kept,
               "quant_blk": BLK, "route": route, "steps": steps, "res": res}
        ckpt_path = os.path.join(CKPT, f"adapter_{tag}.pt")
        torch.save({"sd": sd, "cfg": cfg}, ckpt_path)

    rec = {
        "route": route, "tag": tag, "steps": steps, "res": res, "lr": lr,
        "a_mode": a_mode, "rank": rank, "alpha": alpha, "lam": lam,
        "optimizer": opt_kind,
        "replaced": replaced, "kept": kept,
        "n_trainable": ntr, "trainable_pct": round(ntr / 1.6045e9 * 100, 4),
        "opt_state_MB_est": round(opt_bytes / 2**20, 1),
        "step_s_steady": round(steady, 3),
        "step_s_avg_all": round(total_s / steps, 3),
        "peak_gb": round(peak, 2), "reserved_gb": round(reserved, 2),
        "wall_s": round(time.time() - t_start, 1),
        "loss_first": first, "loss_last": last,
        "loss_down": bool(last is not None and first is not None and last < first),
        "hist": hist, "ckpt": ckpt_path,
    }
    del m, opt
    gc.collect(); torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("route", choices=["adapter", "sgd_full"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--a-mode", default="fp8", choices=["fp8", "fp4", "none"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()
    lr = a.lr if a.route == "adapter" else 1e-6
    rec = run(a.route, steps=a.steps, res=a.res, lr=lr, a_mode=a.a_mode,
              rank=a.rank, alpha=a.alpha, lam=a.lam, tag=a.tag,
              log_every=(a.log_every or None))
    fp = os.path.join(OUT, f"qad2_{a.route}_{a.tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"OK -> {fp}  | {rec['optimizer']} | s/step={rec['step_s_steady']} | "
          f"peak={rec['peak_gb']}GB | loss {rec['loss_first']} -> {rec['loss_last']}")

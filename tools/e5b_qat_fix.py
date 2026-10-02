"""E5b · 定案探针：NVFP4 到底能不能训？

背景：probe2 实测 ModelOpt 六种配方（含 W4A16 纯权重量化）**全部** backward 失败：
  NotImplementedError: must implement backward or vjp for your custom autograd.Function
但源码里存在 `pass_through_bwd`（default=True，语义=梯度直通 STE）。
⇒ 必须一次分辨「开关问题」还是「结构性不可训」。

三条路一次判死（全部 512² + 梯度检查点，快）：
  A. 强制 pass_through_bwd=True（config 级 + 模块级双保险）→ 再 backward
        若通 ⇒ 一行配置解决，G1 直接解锁
  B. 关闭全部量化器（等价 bf16）→ backward
        可行性对照（同时也修掉 probe2 里 cfg=None 的脚本 bug）
  C. QAD 自蒸馏：同一模型双前向
        teacher = 量化开启 + no_grad | student = 量化关闭 + 有梯度
        loss = fm_loss(student) + MSE(student, teacher.detach())
        完全不需要对量化器求导
        若通 ⇒ G1 有第二条路（且这正是文档里 "NVFP4 QAD" 的原意）

运行：source /d/model/env.sh && "$KP_PY" tools/e5b_qat_fix.py
"""
import os, gc, copy, json, time, traceback
import torch
import torch.nn.functional as F
import modelopt.torch.quantization as mtq
from diffusers import SanaTransformer2DModel

MODEL = "D:/model/models/Sana_1600M_1024px_BF16_diffusers"
OUT = "D:/model/out/e5b"
DEV = "cuda"
RES = 512
TOK = 16  # 512² → (1,32,16,16)

store = torch.load("D:/model/out/e5/embeds.pt", map_location="cpu")
POS = store["01_en_scene"]["pos"].to(DEV).to(torch.bfloat16)
MASK = store["01_en_scene"]["pos_mask"].to(DEV)


def load(quant_cfg=None, force_ptb=False):
    m = SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to(DEV)
    m.enable_gradient_checkpointing()
    if quant_cfg is not None:
        cfg = copy.deepcopy(quant_cfg)
        if force_ptb:
            # ① config 级：把 pass_through_bwd 塞进每条规则的 cfg
            for e in cfg.get("quant_cfg", []):
                if isinstance(e.get("cfg"), dict):
                    e["cfg"]["pass_through_bwd"] = True

        def calib(mm):
            with torch.no_grad():
                for t in (999.0, 500.0, 100.0):
                    x = torch.randn(1, 32, TOK, TOK, dtype=torch.bfloat16, device=DEV)
                    mm(hidden_states=x, encoder_hidden_states=POS, encoder_attention_mask=MASK,
                       timestep=torch.tensor([t], device=DEV), return_dict=False)

        mtq.quantize(m, cfg, forward_loop=calib)
        if force_ptb:
            # ② 模块级：直接改属性
            n = 0
            for mod in m.modules():
                if hasattr(mod, "_pass_through_bwd"):
                    mod._pass_through_bwd = True
                    n += 1
            print(f"      [force_ptb] 模块级已设 {n} 个")
    return m


def quantizers(m):
    return [x for x in m.modules()
            if hasattr(x, "disable") and hasattr(x, "enable") and hasattr(x, "is_enabled")]


def set_quant(m, on):
    qs = quantizers(m)
    for q in qs:
        q.enable() if on else q.disable()
    return len(qs)


def fm_batch():
    x0 = torch.randn(1, 32, TOK, TOK, dtype=torch.bfloat16, device=DEV)
    eps = torch.randn_like(x0)
    sig = torch.rand(1, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    xt = (1 - sig) * x0 + sig * eps
    t = (sig.float() * 1000).to(torch.bfloat16)
    return xt, t, (eps - x0)


def fwd(m, xt, t):
    return m(hidden_states=xt, encoder_hidden_states=POS, encoder_attention_mask=MASK,
             timestep=t, return_dict=False)[0]


def report(tag, fn):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag}
    t0 = time.time()
    try:
        info = fn()
        rec.update(ok=True, **(info or {}))
        print(f"  ✅ {tag:34} {rec}")
    except Exception as e:
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}")
        print(f"  ❌ {tag:34} {type(e).__name__}: {str(e)[:110]}")
        if os.environ.get("KP_TB"):
            traceback.print_exc()
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    gc.collect(); torch.cuda.empty_cache()
    return rec


# ---------------- A: 强制 pass_through_bwd ----------------
def arm_A():
    m = load(mtq.W4A8_NVFP4_FP8_CFG, force_ptb=True)
    neg = sum(1 for p in m.parameters() if not p.is_floating_point())
    for p in m.parameters():
        p.requires_grad_(True)
    xt, t, tgt = fm_batch()
    loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
    loss.backward()
    gn = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
             if p.grad is not None) ** 0.5
    npg = sum(1 for p in m.parameters() if p.grad is not None)
    del m
    return dict(loss=round(loss.item(), 5), grad_norm=round(gn, 4), n_grad_params=npg)


# ---------------- B: 量化器全关（可行性对照） ----------------
def arm_B():
    m = load(mtq.W4A8_NVFP4_FP8_CFG, force_ptb=False)
    n = set_quant(m, False)
    for p in m.parameters():
        p.requires_grad_(True)
    xt, t, tgt = fm_batch()
    loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
    loss.backward()
    gn = sum((p.grad.detach().float() ** 2).sum().item() for p in m.parameters()
             if p.grad is not None) ** 0.5
    del m
    return dict(n_quantizers=n, loss=round(loss.item(), 5), grad_norm=round(gn, 4))


# ---------------- C: QAD 自蒸馏 ----------------
def arm_C(steps=5):
    m = load(mtq.W4A8_NVFP4_FP8_CFG, force_ptb=False)
    nq = set_quant(m, True)
    for p in m.parameters():
        p.requires_grad_(True)
    # 让量化器的 amax 等 buffer 不参与优化（它们本就不 require grad）
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-6)
    m.train()
    losses = []
    for i in range(steps):
        xt, t, tgt = fm_batch()
        # teacher：量化开启，无梯度
        set_quant(m, True)
        with torch.no_grad():
            out_q = fwd(m, xt, t)
        # student：量化关闭，有梯度
        set_quant(m, False)
        out_s = fwd(m, xt, t)
        loss = F.mse_loss(out_s.float(), tgt.float()) + 1.0 * F.mse_loss(out_s.float(), out_q.float())
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 5))
    set_quant(m, True)
    del m, opt
    return dict(n_quantizers=nq, losses=losses, trend="down" if losses[-1] < losses[0] else "flat/up")


print("=" * 78)
print(f"[E5b-定案] 分辨率 {RES}² (latent {TOK}×{TOK}) | W4A8_NVFP4_FP8_CFG = 设计档")
rep = {}
rep["A_force_pass_through_bwd"] = report("A 强制 pass_through_bwd → backward", arm_A)
rep["B_quantizers_disabled"] = report("B 量化器全关 → backward", arm_B)
rep["C_qad_self_teacher"] = report("C QAD 自蒸馏（不向量化器求导）", lambda: arm_C(5))

fp = os.path.join(OUT, "phase0_probe3_fix.json")
with open(fp, "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}")
print("=" * 78)

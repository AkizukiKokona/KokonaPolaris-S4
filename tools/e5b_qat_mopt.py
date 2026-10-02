"""E5b · 任务2：ModelOpt 真量化基座 + LoRA 适配器的 QAD 训练（8GB 可行）

关键前提（已实测）：ModelOpt 反向失败是 `FP8SDPA` 缺 backward 的 bug，
用 e5b_common.patch_fp8sdpa() 修掉后 W4A8/W4A4 均可反向。

为什么用适配器而非全参：
  全参 AdamW 状态 2×1.604e9×4B ≈ 12.8GB + 权重 3GB + 激活 ⇒ 8GB 不可能。
  适配器（冻结量化主干 + 低秩旁路）可训参数 ~6M，优化器状态 ~48MB。
  ⭐ 这正对应设计稿里的「Δ-Pack / 能力包」结构（低秩增量 + 量化基座）。

QAD 目标（自蒸馏）：
  teacher = 同权重、全部 TensorQuantizer 关闭（等价 bf16），no_grad；
  student = 量化开启 + LoRA 开启；
  loss = MSE(student, fm_target) + lam · MSE(student, teacher.detach())
  AdamW 只更新 lora_A/lora_B。

用法：
  source /d/model/env.sh && "$KP_PY" tools/e5b_qat_mopt.py --recipe W4A8 \
      --steps 1500 --res 512 --adapt-res 1024 --adapt-steps 120 --rank 16
"""
import os, gc, json, time, argparse
import torch
import torch.nn.functional as F

import e5b_common as C  # noqa: E402

OUT = C.OUT


def build_and_train(recipe, steps, res, adapt_res=0, adapt_steps=0, lr=1e-4,
                    rank=16, alpha=32.0, lam=1.0, tag="", log_every=0, batch=1,
                    calib_res=1024):
    import modelopt.torch.quantization as mtq
    C.prepare()
    cfg = mtq.W4A8_NVFP4_FP8_CFG if recipe == "W4A8" else mtq.NVFP4_DEFAULT_CFG

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # ⚠️ 所有臂的基座量化必须一致 ⇒ 校准分辨率与生成分辨率对齐（1024），
    #    与训练分辨率（512，省时）解耦。
    m, qt = C.load_quant(cfg, res=calib_res, gc_on=True)
    nw = C.wrap_lora(m, rank=rank, alpha=alpha)
    wraps, qs = C.collect_quant_toggles(m)

    for p in m.parameters():
        p.requires_grad_(False)
    for w in wraps:
        w.lora_A.requires_grad_(True)
        w.lora_B.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    def set_teacher():
        for w in wraps:
            w.lora_on = False
        for q in qs:
            q.disable()

    def set_student():
        for w in wraps:
            w.lora_on = True
        for q in qs:
            q.enable()

    m.train()
    hist = []

    def phase(n_steps, r, phase_tag, start_i):
        t0 = time.time()
        for i in range(n_steps):
            pos, mask = C.pick_text(i)
            xt, t, tgt = C.fm_batch(r, seed=3000 + start_i + i, b=batch)
            with torch.no_grad():
                set_teacher()
                tea = C.fwd(m, xt, t, pos, mask).float()
            set_student()
            out = C.fwd(m, xt, t, pos, mask)
            lfm = F.mse_loss(out.float(), tgt.float())
            lkd = F.mse_loss(out.float(), tea.detach())
            loss = lfm + lam * lkd
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            if i == 0 or (i + 1) % (log_every or max(1, n_steps // 8)) == 0:
                rec = {"phase": phase_tag, "step": i + 1, "loss": round(loss.item(), 5),
                       "fm": round(lfm.item(), 5), "kd": round(lkd.item(), 5),
                       "s": round((time.time() - t0) / (i + 1), 3),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                hist.append(rec)
                print(f"    [{phase_tag}] {i+1:>4}/{n_steps}  loss={rec['loss']:.5f} "
                      f"fm={rec['fm']:.5f} kd={rec['kd']:.5f}  {rec['s']}s/step  "
                      f"peak={rec['peak_gb']}GB", flush=True)
        return (time.time() - t0) / max(1, n_steps)

    s1 = phase(steps, res, f"train@{res}", 0)
    s2 = None
    if adapt_res and adapt_steps:
        s2 = phase(adapt_steps, adapt_res, f"adapt@{adapt_res}", 100000)

    peak = torch.cuda.max_memory_allocated() / 2**30
    sd = {k: v.detach().to("cpu") for k, v in m.named_parameters()
          if k.endswith("lora_A") or k.endswith("lora_B")}
    ck = os.path.join(C.CKPT, f"adapter_mopt_{recipe}{tag}.pt")
    torch.save({"sd": sd, "cfg": {"recipe": recipe, "rank": rank, "alpha": alpha,
                                  "lam": lam, "steps": steps, "res": res,
                                  "calib_res": calib_res,
                                  "adapt_res": adapt_res, "adapt_steps": adapt_steps,
                                  "lr": lr}}, ck)
    first = hist[0]["loss"] if hist else None
    last = hist[-1]["loss"] if hist else None
    rec = {"recipe": recipe, "wrapped": nw, "n_quantizers": len(qs),
           "calib_res": calib_res, "train_res": res,
           "n_trainable": ntr, "trainable_pct": round(ntr / 1.6045e9 * 100, 4),
           "opt_state_MB_est": round(2 * ntr * 4 / 2**20, 1),
           "quantize_s": qt, "step_s_train": round(s1, 3),
           "step_s_adapt": round(s2, 3) if s2 else None,
           "peak_gb": round(peak, 2), "wall_s": round(time.time() - t_start, 1),
           "loss_first": first, "loss_last": last,
           "loss_down": bool(first is not None and last is not None and last < first),
           "hist": hist, "ckpt": ck}
    del m, opt
    gc.collect(); torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="W4A8", choices=["W4A8", "W4A4"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--calib-res", type=int, default=1024)
    ap.add_argument("--adapt-res", type=int, default=0)
    ap.add_argument("--adapt-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rec = build_and_train(a.recipe, a.steps, a.res, a.adapt_res, a.adapt_steps,
                          lr=a.lr, rank=a.rank, alpha=a.alpha, lam=a.lam,
                          tag=a.tag, batch=a.batch, calib_res=a.calib_res)
    fp = os.path.join(OUT, f"qat_mopt_{a.recipe}{a.tag}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"OK -> {fp} | ntr={rec['n_trainable']} | peak={rec['peak_gb']}GB | "
          f"loss {rec['loss_first']} -> {rec['loss_last']} | down={rec['loss_down']}", flush=True)

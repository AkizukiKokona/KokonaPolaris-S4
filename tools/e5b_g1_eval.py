"""E5b · G1 主判据（最快路径）：三臂出图 + 分布级 FID

为什么这么设计（用户推动裁剪，正确）：
  G1 的主判据 = 「NVFP4 相对 BF16 掉不掉点」，**不需要任何训练**，出图就能算。
  "QAD 能否补回来" 是第二问，不塞进验证阶段。
  量化用自研 STE fake-quant（无需 ModelOpt 校准、无需打补丁），零开销。

三臂：bf16 / W4A8(设计档) / W4A4(官方默认档)
  3 条缓存 prompt × 17 seed = 51 张/臂（满足 ≥50）
  512² 出图（三臂同分辨率 ⇒ FID 可比；机制与分辨率无关，为速度取 512²）

另加 **FID 噪声底噪**：把 bf16 臂 51 张对半切成 25/26 求 FID
  ⇒ 这个数就是「纯采样噪声能造出多大的 FID 差」，用来判断臂间差值是否显著。

运行：source /d/model/env.sh && "$KP_PY" tools/e5b_g1_eval.py
"""
from kp.paths import MODELS_SANA, OUT
import os, gc, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaPipeline

MODEL = str(MODELS_SANA)
OUT = OUT / "e5b"
G1 = os.path.join(OUT, "g1")
DEV = "cuda"
RES = 1024
STEPS, GUID = 20, 4.5
PROMPTS = ["01_en_scene", "02_zh_text", "03_anime"]
SEEDS = list(range(100, 117))          # 17 seeds
_E4M3 = torch.finfo(torch.float8_e4m3fn)
BLK = 16
os.makedirs(G1, exist_ok=True)

store = torch.load(OUT / "e5/embeds.pt", map_location="cpu")


# ---------- 可微/可用的 NVFP4 模拟量化（STE；推理时 detach 无影响） ----------
def q_fp4(x, blk=BLK):
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


def q_fp8(x):
    amax = x.abs().amax().clamp(min=1e-12)
    scale = (amax / _E4M3.max).clamp(min=1e-12)
    q = (x / scale).clamp(_E4M3.min, _E4M3.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (q - x).detach()


class QLin(nn.Module):
    def __init__(self, lin, a_mode="fp8"):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.a_mode = a_mode

    def forward(self, x):
        w = q_fp4(self.weight)
        if self.a_mode == "fp8":
            x = q_fp8(x)
        elif self.a_mode == "fp4":
            x = q_fp4(x)
        return F.linear(x, w, self.bias)


def swap(m, a_mode, skip=("proj_out",)):
    n = 0
    for name, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{cname}" if name else cname
                if any(s in full for s in skip):
                    continue
                setattr(mod, cname, QLin(child, a_mode=a_mode))
                n += 1
    return n


ARMS = [("bf16", None), ("W4A8", "fp8"), ("W4A4", "fp4")]


def gen_arm(arm, a_mode, max_new=0):
    d = os.path.join(G1, arm)
    os.makedirs(d, exist_ok=True)
    done = len([f for f in os.listdir(d) if f.endswith(".png")])
    if done >= len(PROMPTS) * len(SEEDS):
        print(f"  [{arm}] 已完成 {done} 张，跳过")
        return {"arm": arm, "n": done, "skipped": True}

    pipe = SanaPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, variant="bf16")
    pipe.text_encoder = None                      # 用缓存 embeds，省 5.2GB
    pipe.transformer.to(DEV).eval()
    pipe.vae.to(DEV).eval()
    nq = swap(pipe.transformer, a_mode) if a_mode else 0
    pipe.set_progress_bar_config(disable=True)

    t0 = time.time()
    n = 0
    created = 0
    stop = False
    with torch.no_grad():
        for pk in PROMPTS:
            if stop:
                break
            e = store[pk]
            pe = e["pos"].to(DEV).to(torch.bfloat16)
            pm = e["pos_mask"].to(DEV)
            ne = e["neg"].to(DEV).to(torch.bfloat16)
            nm = e["neg_mask"].to(DEV)
            for sd in SEEDS:
                fp = os.path.join(d, f"{pk}_s{sd}.png")
                if os.path.exists(fp):
                    n += 1
                    continue
                g = torch.Generator("cpu").manual_seed(sd)
                img = pipe(prompt=None, negative_prompt=None,
                           prompt_embeds=pe, prompt_attention_mask=pm,
                           negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
                           height=RES, width=RES, num_inference_steps=STEPS,
                           guidance_scale=GUID, generator=g).images[0]
                img.save(fp)
                n += 1
                created += 1
                print(f"  [{arm}] {n}/{len(PROMPTS)*len(SEEDS)}  {(time.time()-t0)/n:.2f}s/img", flush=True)
                if max_new and created >= max_new:
                    stop = True
                    break
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  [{arm}] 完成 {n} 张 | {dt:.1f}s | {dt/max(n,1):.2f}s/img | 峰值 {peak:.2f}GB | 量化层 {nq}")
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {"arm": arm, "n": n, "s": round(dt, 1), "s_per_img": round(dt / max(n, 1), 2),
            "peak_gb": round(peak, 2), "n_quant_linear": nq}


# ---------- FID ----------
def fid(dirs):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    return calculate_fid_given_paths(dirs, batch_size=16, device=DEV, dims=2048, num_workers=0)


def split_half(src, dst_a, dst_b):
    import shutil
    for p in (dst_a, dst_b):
        os.makedirs(p, exist_ok=True)
    fs = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    for i, f in enumerate(fs):
        tgt = dst_a if i % 2 == 0 else dst_b
        d = os.path.join(tgt, f)
        if not os.path.exists(d):
            try:
                os.link(os.path.join(src, f), d)
            except OSError:
                shutil.copy2(os.path.join(src, f), d)
    return len(fs) // 2, len(fs) - len(fs) // 2


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bf16,W4A8,W4A4")
    ap.add_argument("--max-new", type=int, default=0, help="本次最多新生成几张（0=不限）；脚本可断点续跑")
    ap.add_argument("--no-fid", action="store_true")
    a = ap.parse_args()
    sel = set(x.strip() for x in a.arms.split(",") if x.strip())

    print("=" * 78)
    print(f"[G1] 三臂出图 {RES}² | {len(PROMPTS)} prompt × {len(SEEDS)} seed = {len(PROMPTS)*len(SEEDS)} 张/臂"
          f" | 本次上限 {a.max_new or '不限'}", flush=True)
    rep = {"res": RES, "steps": STEPS, "guidance": GUID,
           "prompts": PROMPTS, "n_seeds": len(SEEDS), "arms": {}}
    for arm, am in ARMS:
        if arm not in sel:
            continue
        torch.cuda.reset_peak_memory_stats()
        rep["arms"][arm] = gen_arm(arm, am, max_new=a.max_new)

    if a.no_fid:
        print("\n[FID] 已跳过（--no-fid）")
    else:
        print("\n[FID] 计算中（首次需下载 Inception 权重）...", flush=True)
        ref = os.path.join(G1, "bf16")
        try:
            rep["fid"] = {}
            for arm, _ in ARMS:
                if arm == "bf16":
                    continue
                rep["fid"][f"{arm}_vs_bf16"] = round(float(fid([os.path.join(G1, arm), ref])), 3)
                print(f"  FID({arm} vs bf16) = {rep['fid'][f'{arm}_vs_bf16']}")

            na, nb = split_half(ref, os.path.join(G1, "_nz_a"), os.path.join(G1, "_nz_b"))
            rep["fid_noise_floor"] = round(float(fid([os.path.join(G1, "_nz_a"), os.path.join(G1, "_nz_b")])), 3)
            print(f"  FID 噪声底噪 (bf16 对半切 {na}/{nb}) = {rep['fid_noise_floor']}")
        except Exception as e:
            rep["fid_error"] = f"{type(e).__name__}: {e}"
            print(f"  ⚠️ FID 失败：{rep['fid_error']}")

    fp = os.path.join(OUT, "g1_result.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 写入 {fp}\n✅ 图在 {G1}/", flush=True)
    print("=" * 78)

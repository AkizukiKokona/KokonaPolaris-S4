"""E5b · 任务3：G1 四臂出图（每臂 ≥50 张 1024²，固定 prompt + 固定种子）

臂：bf16 / PTQ-W4A8 / PTQ-W4A4 / TRAIN-W4A8(或 TRAIN-W4A4)
图存 out/e5b/g1_images/<arm>/<key>__s<seed>.png

用法：
  source /d/model/env.sh && "$KP_PY" tools/e5b_g1_gen.py --arm bf16 --n-seeds 5
  ... --arm TRAIN-W4A8 --adapter out/e5b/qat_ckpt/adapter_mopt_W4A8.pt
"""
import os, sys, json, time, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5b_common as C  # noqa: E402

from diffusers import SanaPipeline  # noqa: E402

OUT = C.OUT
IMG_ROOT = os.path.join(OUT, "g1_images")
os.makedirs(IMG_ROOT, exist_ok=True)

STEPS_DEF = 20
GUID_DEF = 4.5
RES_DEF = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=STEPS_DEF)
    ap.add_argument("--guidance", type=float, default=GUID_DEF)
    ap.add_argument("--res", type=int, default=RES_DEF)
    ap.add_argument("--calib-res", type=int, default=1024)
    ap.add_argument("--limit-keys", type=int, default=0, help="只做前 N 个 prompt（冒烟）")
    a = ap.parse_args()

    store = torch.load(os.path.join(OUT, "g1_embeds.pt"), map_location="cpu")
    keys = sorted(store.keys())
    if a.limit_keys:
        keys = keys[:a.limit_keys]
    outdir = os.path.join(IMG_ROOT, a.arm)
    os.makedirs(outdir, exist_ok=True)

    print(f"[gen] arm={a.arm} prompts={len(keys)} seeds={a.n_seeds} "
          f"res={a.res} steps={a.steps} guid={a.guidance}", flush=True)

    C.prepare()
    pipe = SanaPipeline.from_pretrained(C.MODEL, torch_dtype=torch.bfloat16, variant="bf16")
    pipe.text_encoder = None

    # 构建本臂 transformer（校准分辨率与所有臂一致）
    t0 = time.time()
    if a.arm == "bf16":
        tr = C.build_bf16(gc_on=False)
    else:
        tr = C.build_arm(a.arm, res=a.calib_res, rank=a.rank, alpha=a.alpha,
                         gc_on=False, adapter=a.adapter)
    tr.to("cuda").eval()
    print(f"[gen] 建模完成 {time.time()-t0:.1f}s", flush=True)
    pipe.transformer = tr
    pipe.vae.to("cuda").eval()

    torch.cuda.reset_peak_memory_stats()
    recs = []
    n_ok = n_skip = 0
    for ki, k in enumerate(keys):
        e = store[k]
        for si in range(a.n_seeds):
            seed = a.seed0 + si
            fp = os.path.join(outdir, f"{k}__s{seed}.png")
            if os.path.exists(fp):
                n_skip += 1
                continue
            g = torch.Generator("cpu").manual_seed(seed)
            t1 = time.time()
            try:
                img = pipe(
                    prompt=None, negative_prompt=None,
                    prompt_embeds=e["pos"].cuda().to(torch.bfloat16),
                    prompt_attention_mask=e["pos_mask"].cuda(),
                    negative_prompt_embeds=e["neg"].cuda().to(torch.bfloat16),
                    negative_prompt_attention_mask=e["neg_mask"].cuda(),
                    height=a.res, width=a.res, num_inference_steps=a.steps,
                    guidance_scale=a.guidance, generator=g,
                ).images[0]
                img.save(fp)
                n_ok += 1
                recs.append({"key": k, "seed": seed, "s": round(time.time() - t1, 1)})
            except torch.cuda.OutOfMemoryError as ex:
                print(f"  OOM {k} s{seed}: {str(ex)[:80]}", flush=True)
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  ERR {k} s{seed}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        print(f"  [{ki+1}/{len(keys)}] {k}  ok={n_ok} skip={n_skip}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    tot = sum(r["s"] for r in recs)
    rep = {"arm": a.arm, "adapter": a.adapter, "n_prompts": len(keys),
           "n_seeds": a.n_seeds, "res": a.res, "steps": a.steps,
           "guidance": a.guidance, "seed0": a.seed0, "calib_res": a.calib_res,
           "n_new": n_ok, "n_skip": n_skip, "peak_gb": round(peak, 2),
           "mean_s_per_img": round(tot / max(1, n_ok), 1),
           "outdir": outdir, "recs": recs}
    fp = os.path.join(OUT, f"g1_gen_{a.arm}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"OK -> {fp} | new={n_ok} skip={n_skip} peak={peak:.2f}GB "
          f"mean={rep['mean_s_per_img']}s/img", flush=True)


if __name__ == "__main__":
    main()

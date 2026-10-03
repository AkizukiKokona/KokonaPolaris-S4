"""KokonaPolaris E5 · 逐层误差谱（回答「哪一层最不能量化」）

动机：
  整体误差只告诉「偏了多少」，不告诉「偏在哪里」。
  设计里「分层精度白名单」（FP4 权重 + 部分层升精度）需要知道**误差沿深度的分布**，
  以及是否存在**少数层放大倍数异常**——那才是白名单该保护的对象。

指标：
  rel[i]      = ||y_q[i] - y_bf16[i]|| / ||y_bf16[i]||       每层输出的绝对偏差
  gain[i]     = rel[i] / rel[i-1]                             层间放大倍数（>1 表示该层放大误差）

运行：source /d/model/env.sh && "$KP_PY" tools/e5_layer_probe.py [arm]
  arm ∈ {W4A4, W4A16, W4A8}，默认 W4A8
"""
from kp.paths import MODELS_SANA, OUT
import os, sys, gc, json, torch, copy
import modelopt.torch.quantization as mtq
from diffusers import SanaTransformer2DModel

MODEL = str(MODELS_SANA)
OUT = OUT / "e5"
KEY, T = "01_en_scene", 500.0
ARM = sys.argv[1] if len(sys.argv) > 1 else "W4A8"

store = torch.load(os.path.join(OUT, "embeds.pt"), map_location="cpu")
e = store[KEY]
x = torch.randn(1, 32, 32, 32, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16).cuda()
emb = e["pos"].cuda().to(torch.bfloat16)
ts = torch.tensor([T], device="cuda")

def fresh():
    return SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to("cuda").eval()

def capture(model):
    """跑一次并记录每个 transformer_block 的输出 + 各模块类型的内部输入统计"""
    rec = {}
    hooks = []

    def mk(i, kind):
        def fn(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                rec.setdefault(i, {})[kind] = o.detach().float().cpu()
        return fn

    for i, blk in enumerate(model.transformer_blocks):
        hooks.append(blk.register_forward_hook(mk(i, "block")))
    with torch.no_grad():
        y = model(hidden_states=x, encoder_hidden_states=emb, timestep=ts, return_dict=False)[0]
    for h in hooks:
        h.remove()
    return rec, y.detach().float().cpu()

print("=" * 76)
print(f"[1] bf16 参照（case = {KEY}@t{int(T)}, arm = {ARM}）")
tf = fresh()
n_blocks = len(tf.transformer_blocks)
print(f"    transformer_blocks = {n_blocks}")
ref_layers, ref_out = capture(tf)
del tf; gc.collect(); torch.cuda.empty_cache()

print("\n[2] 量化 + 采集")
tf = fresh()
cfg = {"W4A4": mtq.NVFP4_DEFAULT_CFG,
       "W4A16": mtq.W4A16_NVFP4_CFG,
       "W4A8": mtq.W4A8_NVFP4_FP8_CFG}[ARM]
def calib(m):
    with torch.no_grad():
        m(hidden_states=x, encoder_hidden_states=emb, timestep=ts, return_dict=False)
mtq.quantize(tf, cfg, forward_loop=calib)
q_layers, q_out = capture(tf)

final_rel = (q_out - ref_out).norm().item() / ref_out.norm().item()
print(f"    最终输出相对误差 {final_rel*100:.2f}%")

print("\n[3] 逐层误差谱")
print(f"{'层':>4}{'绝对误差':>12}{'层间放大':>12}   柱状")
print("-" * 76)
rows, prev = [], None
for i in range(n_blocks):
    r, q = ref_layers[i]["block"], q_layers[i]["block"]
    rel = (q - r).norm().item() / (r.norm().item() + 1e-12)
    gain = (rel / prev) if prev and prev > 1e-9 else float("nan")
    rows.append({"layer": i, "rel": rel, "gain": gain})
    bar = "█" * min(int(rel * 400), 60)
    g = f"{gain:>11.2f}" if gain == gain else f"{'—':>11}"
    print(f"{i:>4}{rel*100:>11.2f}%{g}   {bar}")
    prev = rel

worst = max(rows, key=lambda z: z["rel"])
print(f"\n    最大绝对误差层: {worst['layer']} ({worst['rel']*100:.2f}%)")
valid = [z for z in rows if z["gain"] == z["gain"]]
hot = sorted(valid, key=lambda z: -z["gain"])[:3]
print("    放大倍数最高的 3 层（白名单候选）:")
for z in hot:
    print(f"      layer {z['layer']:>2}  gain {z['gain']:.2f}×  绝对 {z['rel']*100:.2f}%")

res = {"arm": ARM, "case": f"{KEY}@t{int(T)}", "n_blocks": n_blocks,
       "final_rel": final_rel, "layers": rows}
fp = os.path.join(OUT, f"e5_layers_{ARM}.json")
json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}")
print("=" * 76)

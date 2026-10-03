"""KokonaPolaris E5 · 步骤2（前哨核心）：NVFP4 量化在真实 DiT 上的信号级误差对照

设计动机
--------
E5 是「前哨」，目标是**用真模型的真权重**回答一个问题：
  「NVFP4 到底把 Sana 的预测打偏多少？我们的设计档落在哪？」

方法（不需要 VAE、不需要跑完整去噪 → 快、可重复、可归因）
  1. 用真实 prompt embedding + 真实 size 的 latent，在 5 个 timestep 上各跑一次 forward
  2. 记录 bf16 的输出作为 golden
  3. 对每个量化配置重新加载干净模型 → 量化 → 同输入 forward → 与 golden 比
  4. 指标：相对误差 ||Δy||/||y|| 与余弦相似度 cos(y_q, y_bf16)

对照臂（5 路，可做变量归因）
  bf16        —— 参照
  W4A4        —— NVFP4_DEFAULT_CFG（权重 b16 + 激活 b16）
  W4A16       —— W4A16_NVFP4_CFG（权重 b16 + bf16 激活）
  W4A8        —— W4A8_NVFP4_FP8_CFG（权重 **b32** + FP8 per-tensor）← 官方「设计档」
  W4A8_b16    —— 自定义（权重 **b16** + FP8）← 隔离出 block size 这个变量

运行：source /d/model/env.sh && "$KP_PY" tools/e5_forward_probe.py
"""

import sys
from pathlib import Path

# ⚠️ 入口自举：直接 `python tools/e5_forward_probe.py` 时 `sys.path[0]` 是 **tools/** 而非仓库根
#    ⇒ `import kp.paths` 报 ModuleNotFoundError；照 tools/onboard.py:18，⛔ 不写死绝对路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.paths import MODELS_SANA, OUT  # noqa: E402
import os, gc, copy, json, time, torch
import modelopt.torch.quantization as mtq
from diffusers import SanaTransformer2DModel

MODEL = str(MODELS_SANA)
OUT = OUT / "e5"
os.makedirs(OUT, exist_ok=True)
TIMESTEPS = [999.0, 750.0, 500.0, 250.0, 50.0]
PROMPT_KEYS = ["01_en_scene", "02_zh_text", "03_anime"]

print("=" * 78)
print("[0] 载入缓存的文本 embedding")
store = torch.load(os.path.join(OUT, "embeds.pt"), map_location="cpu")

# ---------- 构造测试集 ----------
def make_latent(seed):
    g = torch.Generator("cpu").manual_seed(seed)
    x = torch.randn(1, 32, 32, 32, generator=g, dtype=torch.float32)
    return x.to(torch.bfloat16).cuda()

CASES = []
for k in PROMPT_KEYS:
    e = store[k]
    pos = e["pos"].cuda().to(torch.bfloat16)
    for t in TIMESTEPS:
        CASES.append({
            "tag": f"{k}@t{int(t)}",
            "x": make_latent(abs(hash((k, t))) % (2**31)),
            "emb": pos,
            "t": torch.tensor([t], device="cuda"),
        })
print(f"    测试集: {len(CASES)} 例 = {len(PROMPT_KEYS)} prompt × {len(TIMESTEPS)} timestep")
print(f"    注: 只喂正向 embedding（CAG 的两路拆分对本对照无影响，误差度量在单路上更干净）")

# ---------- 量化配置 ----------
def set_weight_block(cfg, blk):
    """把配置里 weight_quantizer 的 block 粗粒度改掉（键可能是 int -1 也可能是 str '-1'，
    必须按类型处理，否则会造出重复键 → pydantic 报 'Dynamic block quantization only
    supports quantization last axis'）。"""
    cfg = copy.deepcopy(cfg)
    for entry in cfg["quant_cfg"]:
        c = entry.get("cfg")
        if not (isinstance(c, dict) and isinstance(c.get("block_sizes"), dict)):
            continue
        c["block_sizes"] = {
            (-1 if k in (-1, "-1") else k): (blk if k in (-1, "-1") else v)
            for k, v in c["block_sizes"].items()
        }
    return cfg

ARMS = [
    ("bf16",       None),
    ("W4A4",       mtq.NVFP4_DEFAULT_CFG),        # 权重 b16 + 激活 b16
    ("W4A16",      mtq.W4A16_NVFP4_CFG),          # 权重 b16 + 激活 bf16
    ("W4A8",       mtq.W4A8_NVFP4_FP8_CFG),       # 权重 b32 + 激活 FP8（官方「设计档」）
    ("W4A8_b16",   set_weight_block(mtq.W4A8_NVFP4_FP8_CFG, 16)),  # 隔离 block size 变量
    ("W4A4_46",    mtq.NVFP4_FOUR_OVER_SIX_CFG),  # 设计文档引用的 4/6 配比
]

# ---------- 载入干净模型 ----------
def fresh_model():
    return SanaTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", variant="bf16", torch_dtype=torch.bfloat16
    ).to("cuda").eval()

def run_forward(model, cases):
    outs = []
    with torch.no_grad():
        for c in cases:
            y = model(hidden_states=c["x"], encoder_hidden_states=c["emb"],
                      timestep=c["t"], return_dict=False)[0]
            outs.append(y.float().cpu())
    return outs

results = {}
print("\n[1] bf16 参照臂")
tf = fresh_model()
n_lin = sum(1 for m in tf.modules() if isinstance(m, torch.nn.Linear))
print(f"    nn.Linear 层数 = {n_lin}   权重显存 {torch.cuda.memory_allocated()/2**30:.2f} GB")
t0 = time.time()
golden = run_forward(tf, CASES)
print(f"    ✅ {len(golden)} 例，耗时 {time.time()-t0:.1f}s")
results["bf16"] = {"n_linear": n_lin, "weight_gb": round(torch.cuda.memory_allocated()/2**30, 2)}
del tf; gc.collect(); torch.cuda.empty_cache()

# 校准循环（对 max 算法有效）
def make_calib(cases):
    def _calib(model):
        with torch.no_grad():
            for c in cases[:10]:
                model(hidden_states=c["x"], encoder_hidden_states=c["emb"],
                      timestep=c["t"], return_dict=False)
    return _calib

for arm, cfg in ARMS[1:]:
    print(f"\n[2] {arm}")
    tf = fresh_model()
    t0 = time.time()
    mtq.quantize(tf, cfg, forward_loop=make_calib(CASES))
    nq = sum(1 for m in tf.modules() if "QuantLinear" in type(m).__name__)
    print(f"    量化完成 {time.time()-t0:.1f}s  QuantLinear = {nq} / {n_lin}")
    outs = run_forward(tf, CASES)

    per_t, per_p, rels, coss = {}, {}, [], []
    cases_out = []
    for c, g, y in zip(CASES, golden, outs):
        d = (y - g)
        rel = d.norm().item() / (g.norm().item() + 1e-12)
        cos = torch.nn.functional.cosine_similarity(
            y.flatten().unsqueeze(0), g.flatten().unsqueeze(0)).item()
        rels.append(rel); coss.append(cos)
        cases_out.append({"tag": c["tag"], "rel_err": rel, "cos": cos,
                          "max_abs": d.abs().max().item()})
        per_t.setdefault(c["tag"].split("@t")[1], []).append(rel)
        per_p.setdefault(c["tag"].split("@")[0], []).append(rel)

    mean_rel = sum(rels) / len(rels)
    mean_cos = sum(coss) / len(coss)
    print(f"    平均相对误差 {mean_rel*100:.2f}%   平均余弦 {mean_cos:.5f}"
          f"   最差相对误差 {max(rels)*100:.2f}%")
    for tt in [str(int(x)) for x in TIMESTEPS]:
        v = per_t[tt]
        print(f"      t={tt:>4}: 误差 {sum(v)/len(v)*100:>6.2f}%")
    results[arm] = {"n_quant_linear": nq, "mean_rel_err": mean_rel,
                    "mean_cos": mean_cos, "max_rel_err": max(rels),
                    "per_timestep": {k: sum(v)/len(v) for k, v in per_t.items()},
                    "per_prompt": {k: sum(v)/len(v) for k, v in per_p.items()},
                    "cases": cases_out}
    del tf; gc.collect(); torch.cuda.empty_cache()

# ---------- 汇总 ----------
print("\n" + "=" * 78)
print("汇总（相对 bf16）")
print(f"{'配置':<12}{'平均相对误差':>14}{'平均余弦':>12}{'最差':>10}{'QuantLinear':>14}")
print("-" * 78)
for arm, r in results.items():
    if arm == "bf16":
        continue
    print(f"{arm:<12}{r['mean_rel_err']*100:>13.2f}%{r['mean_cos']:>12.5f}"
          f"{r['max_rel_err']*100:>9.2f}%{r['n_quant_linear']:>14}")

fp = os.path.join(OUT, "e5_forward_probe.json")
with open(fp, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n✅ 写入 {fp}")

# 排列成"设计档落在哪"的判据
try:
    a4, a16, a8 = (results["W4A4"]["mean_rel_err"],
                   results["W4A16"]["mean_rel_err"],
                   results["W4A8"]["mean_rel_err"])
    print(f"\n[判据] W4A4 {a4*100:.2f}%  |  W4A8 {a8*100:.2f}%  |  W4A16 {a16*100:.2f}%")
    if a4 > a8 > a16:
        print("  ✅ 设计档（W4A8）确实落在 W4A4 与 W4A16 之间")
    else:
        print("  ⚠️ 顺序不符预期，需检查")
except KeyError:
    pass
print("=" * 78)

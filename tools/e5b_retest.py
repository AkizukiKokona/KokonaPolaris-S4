"""E5b · 任务1：暴露 cl.exe 后重测 ModelOpt 的 backward，并抓「缺 backward 的真身」。

产出 out/e5b/msvc_retest.json:
  {compiled_ext_ok, backward_ok, offending_function, offending_file, notes, ...}

运行（必须经 bat 包装以注入 MSVC 环境）：
  cd /d/model && cmd //c "tools\\e5b_msvc_env.bat .venv\\Scripts\\python.exe tools\\e5b_retest.py"
"""
import sys
from pathlib import Path

# ⚠️ 入口自举：直接 `python tools/e5b_retest.py` 时 `sys.path[0]` 是 **tools/** 而非仓库根
#    ⇒ `import kp.paths` 报 ModuleNotFoundError；照 tools/onboard.py:18，⛔ 不写死绝对路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.paths import MODELS_SANA, OUT  # noqa: E402
import os, sys, gc, json, time, traceback, inspect, copy, warnings, io
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

OUT = OUT / "e5b"
os.makedirs(OUT, exist_ok=True)

REP = {
    "argv0": sys.executable,
    "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR"),
    "DISTUTILS_USE_SDK": os.environ.get("DISTUTILS_USE_SDK"),
    "cl_on_path": None,
    "compiled_ext_ok": False,
    "backward_ok": False,
    "offending_function": None,
    "offending_file": None,
    "notes": [],
}


def dump():
    with open(os.path.join(OUT, "msvc_retest.json"), "w", encoding="utf-8") as f:
        json.dump(REP, f, ensure_ascii=False, indent=2)


def which(name):
    from shutil import which as _w
    return _w(name)


# ---------------------------------------------------------------- step 0
REP["cl_on_path"] = which("cl")
REP["link_on_path"] = which("link")
REP["ninja_on_path"] = which("ninja")
print(f"[0] cl={REP['cl_on_path']}", flush=True)

if REP["cl_on_path"] is None:
    REP["notes"].append("cl.exe 仍不在 PATH ⇒ bat 注入失败")
    dump()
    sys.exit(2)

# ---------------------------------------------------------------- step 1: 编译一个极小 CUDA 扩展
import torch  # noqa: E402
print(f"[1] torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0)}", flush=True)

try:
    from torch.utils.cpp_extension import load_inline
    buf = io.StringIO()
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        mod = load_inline(
            name="e5b_tiny_cuda",
            cpp_sources="int e5b_ping(){return 42;}",
            functions=["e5b_ping"],
            verbose=False,
        )
    REP["tiny_cuda_ext_ok"] = bool(mod.e5b_ping() == 42)
    REP["tiny_cuda_ext_s"] = round(time.time() - t0, 1)
    REP["tiny_cuda_ext_warns"] = [str(x.message)[:400] for x in wlist]
    REP["compiled_ext_ok"] = REP["tiny_cuda_ext_ok"]
except Exception as e:
    REP["tiny_cuda_ext_ok"] = False
    REP["tiny_cuda_ext_err"] = f"{type(e).__name__}: {str(e)[:1500]}"
    REP["notes"].append("极小 CUDA 扩展编译失败")
print(f"[1] tiny_cuda_ext_ok={REP.get('tiny_cuda_ext_ok')}", flush=True)
dump()

# ---------------------------------------------------------------- step 2: ModelOpt 两个真扩展能否加载
from modelopt.torch.quantization.extensions import get_cuda_ext, get_cuda_ext_fp8  # noqa: E402

try:
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        t0 = time.time()
        ext = get_cuda_ext(raise_if_failed=False)
        ext_fp8 = get_cuda_ext_fp8(raise_if_failed=False)
        REP["modelopt_ext_load_s"] = round(time.time() - t0, 1)
    REP["modelopt_cuda_ext_ok"] = ext is not None
    REP["modelopt_cuda_ext_fp8_ok"] = ext_fp8 is not None
    REP["modelopt_ext_warns"] = [str(x.message)[:600] for x in wlist]
    REP["modelopt_ext_fallback"] = any("falling back" in str(x.message) for x in wlist)
except Exception as e:
    REP["modelopt_cuda_ext_ok"] = False
    REP["modelopt_cuda_ext_fp8_ok"] = False
    REP["modelopt_ext_err"] = f"{type(e).__name__}: {str(e)[:1500]}"
print(f"[2] modelopt_ext ok={REP.get('modelopt_cuda_ext_ok')} fp8={REP.get('modelopt_cuda_ext_fp8_ok')} fallback={REP.get('modelopt_ext_fallback')}", flush=True)
dump()

# ---------------------------------------------------------------- step 3: 运行时抓「无 backward 的 Function 真身」
SUSPECTS = []
_orig_apply = torch.autograd.Function.apply.__func__


def _patched_apply(cls, *args, **kwargs):
    d = getattr(cls, "__dict__", {})
    if not any(k in d for k in ("backward", "vjp", "jvp")):
        rec = {"qualname": getattr(cls, "__qualname__", repr(cls)),
               "module": getattr(cls, "__module__", None)}
        try:
            rec["file"] = inspect.getsourcefile(cls)
        except Exception:
            rec["file"] = None
        if rec not in SUSPECTS:
            SUSPECTS.append(rec)
            print(f"      [SUSPECT] {rec['qualname']} @ {rec['file']}", flush=True)
    return _orig_apply(cls, *args, **kwargs)


torch.autograd.Function.apply = classmethod(_patched_apply)

# ---------------------------------------------------------------- step 4: 重建 A/B 两路
import torch.nn.functional as F  # noqa: E402
import modelopt.torch.quantization as mtq  # noqa: E402
from diffusers import SanaTransformer2DModel  # noqa: E402

MODEL = str(MODELS_SANA)
DEV = "cuda"
RES = 512
TOK = 16

store = torch.load(OUT / "e5/embeds.pt", map_location="cpu")
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
            n = 0
            for mod in m.modules():
                if hasattr(mod, "_pass_through_bwd"):
                    mod._pass_through_bwd = True
                    n += 1
            REP["force_ptb_modules"] = n
    return m


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


def run_arm(tag, quant_cfg, force_ptb, disable_quant=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag}
    nS = len(SUSPECTS)
    t0 = time.time()
    try:
        m = load(quant_cfg=quant_cfg, force_ptb=force_ptb)
        if disable_quant:
            nq = 0
            for q in m.modules():
                if hasattr(q, "disable") and hasattr(q, "enable") and hasattr(q, "is_enabled"):
                    q.disable(); nq += 1
            rec["quantizers_disabled"] = nq
        for p in m.parameters():
            p.requires_grad_(True)
        xt, t, tgt = fm_batch()
        loss = F.mse_loss(fwd(m, xt, t).float(), tgt.float())
        rec["loss"] = round(loss.item(), 5)
        loss.backward()
        rec["ok"] = True
        del m
    except Exception as e:
        rec["ok"] = False
        rec["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        tb = traceback.format_exc()
        rec["traceback"] = tb
        # 抽取 traceback 里所有 python 帧
        frames = []
        for fr in traceback.extract_tb(sys.exc_info()[2]):
            frames.append({"file": fr.filename, "line": fr.lineno, "func": fr.name})
        rec["tb_frames"] = frames
        # 非 torch 内部帧
        rec["tb_user_frames"] = [f for f in frames if "site-packages\\torch\\" not in f["file"]]
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    rec["suspects_delta"] = SUSPECTS[nS:]
    gc.collect(); torch.cuda.empty_cache()
    print(f"  {'OK ' if rec.get('ok') else 'FAIL'} {tag} | {rec.get('err','')}", flush=True)
    return rec


REP["arm_A_force_ptb"] = run_arm("A 强制 pass_through_bwd", mtq.W4A8_NVFP4_FP8_CFG, True)
dump()
REP["arm_B_quant_off"] = run_arm("B 量化器全关", mtq.W4A8_NVFP4_FP8_CFG, False, disable_quant=True)
dump()

# 罪魁汇总
all_suspects = list(SUSPECTS)
for arm in ("arm_A_force_ptb", "arm_B_quant_off"):
    for s in REP.get(arm, {}).get("suspects_delta", []):
        if s not in all_suspects:
            all_suspects.append(s)
REP["all_suspects"] = all_suspects

# 判决
a_ok = REP["arm_A_force_ptb"].get("ok")
b_ok = REP["arm_B_quant_off"].get("ok")
REP["backward_ok"] = bool(a_ok or b_ok)
if all_suspects:
    REP["offending_function"] = all_suspects[0]["qualname"]
    REP["offending_file"] = all_suspects[0]["file"]
elif not REP["backward_ok"]:
    # 没有 python 侧 suspect ⇒ 元凶在扩展 / 非 Function 路径
    REP["notes"].append("无 python 侧 autograd.Function suspect；元凶在扩展或其他路径，见 traceback")

REP["notes"].append(
    f"A(forced ptb) ok={a_ok} | B(quant off) ok={b_ok} | "
    f"modelopt_ext={REP.get('modelopt_cuda_ext_ok')} fp8={REP.get('modelopt_cuda_ext_fp8_ok')} "
    f"fallback={REP.get('modelopt_ext_fallback')}"
)
dump()
print("=" * 70, flush=True)
print(json.dumps({k: v for k, v in REP.items() if k not in ("arm_A_force_ptb", "arm_B_quant_off")},
                 ensure_ascii=False, indent=2), flush=True)
print("✅ -> D:/model/out/e5b/msvc_retest.json", flush=True)

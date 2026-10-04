"""本地出图器（GUI 版）—— ⭐ 隐私设计：**prompt 只在本地文件与你的屏幕之间流动**

═══ 为什么有这个文件 ═══
项目铁律：这是试验原型，**主线只有一个 = 出第一张图**。
且用户明确要求：**「敏感内容我自己填，不经过你的云端」**。

═══ 隐私边界（刻意设计，请勿改动）═══
本脚本从 `out/_local_prompt.txt` 读取 prompt。
⛔ **Agent 不读该文件**，只负责：起 GUI → 你在界面里填 → 你点「生成」 → 看图。
prompt 永远不经过 Agent 的上下文。这是本设计的核心。

═══ 用法 ═══
  双击 tools/gen_gui.ps1
  （或在 bash 里： powershell -ExecutionPolicy Bypass -File tools/gen_gui.ps1）

⚠️ 首次运行会加载 1.6B 模型（约 2 秒），之后出图每张约 8–24 秒。
⚠️ **不要传 `device_map="cuda"`** —— 实测会让显存占 8.48GB（>8GB 总量）
   ⇒ page swapping ⇒ 512px 都要跑 >280s。本文件默认走 `enable_model_cpu_offload`
   （实测峰值 4.96GB / 1024px 20 步 13.2s）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---- 路径一律走 KP_ROOT（禁止写死盘符，跨机可用）----
KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[1])
MODEL_DIR = KP_ROOT / "models" / "Sana_1600M_1024px_BF16_diffusers"
OUT_DIR = KP_ROOT / "out" / "local_gen"
PROMPT_FILE = KP_ROOT / "out" / "_local_prompt.txt"      # ⛔ Agent 不读
NEG_FILE = KP_ROOT / "out" / "_local_negative.txt"      # ⛔ Agent 不读
LOG_FILE = KP_ROOT / "out" / "_local_gen.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:                                       # noqa: BLE001
        pass


def _read_prompt_file(path: Path, default: str = "") -> str:
    """读 prompt。⛔ 本函数只在本机执行，输出不经过 Agent。"""
    if path.exists():
        try:
            txt = path.read_text(encoding="utf-8").strip()
            if txt:
                return txt
        except Exception:                                   # noqa: BLE001
            pass
    return default


def generate(prompt: str, *, negative: str = "", size: int = 1024,
             steps: int = 20, seed: int = -1, out_dir: Path = OUT_DIR,
             dtype: str = "bfloat16", load_4bit: bool = False,
             offload: bool = True) -> Path:
    """跑一次 Sana 出图。返回图片路径。

    ═══ ⭐ 2026-10-05 查明并解决「跑极慢」的根因（kokona实测）═══
    症状：`device_map="cuda"` 时 bf16 变体**占 8.48GB > 8GB 总量**
         ⇒ 疯狂 page swapping ⇒ 连 512px 都跑不完（>280s）。

    根因：**text_encoder（Gemma2）就占 4.87GB**，而它**只用来编码一次 prompt**。

    ✅ 解法：`pipe.enable_model_cpu_offload()` —— 逐模块搬运，编码完立刻挪回 CPU。
    实测（RTX 5050 Laptop / 20 SM / 8GB）：

| 配置 | 耗时 | 峰值显存 |
|---|---|---|
| `device_map="cuda"`（旧） | >280s（跑不完） | 8.48GB ⇒爆 |
| **`enable_model_cpu_offload`** | **512px/20步 19.5s** | **4.96 GB** |
| 同上 | 768px/20步 24.0s | — |
| 同上 | **1024px/20步 13.2s** | — |
| 同上 | 1024px/8步 **8.0s** | — |

    ⚠️ **`int4` 变体在本机不可用**：官方仓库只给了transformer 的 int4 权重，
       `vae` / `text_encoder` 只有 bf16 ⇒ `variant="int4"` 会报
       `no file named pytorch_model.bin`。**不是配置问题，是权重不全。**
    """
    import torch
    from diffusers import SanaPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    if seed < 0:
        seed = int(time.time() * 1000) % (2 ** 31)

    t0 = time.time()
    # ⭐ 不传 device_map（那会一次性全塞进 GPU ⇒ 爆），先建 CPU 管线再开 offload
    pipe = SanaPipeline.from_pretrained(
        str(MODEL_DIR), variant="bf16",
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
    )
    pipe.set_progress_bar_config(disable=True)
    if offload:
        pipe.enable_model_cpu_offload(device="cuda")   # ⭐ 峰值 4.96GB（实测）
    else:
        pipe = pipe.to("cuda")
    _log(f"模型就绪 {time.time() - t0:.0f}s"
         + ("（CPU offload）" if offload else "（全GPU⚠️ 8GB 卡可能 OOM）"))

    # ⚠️ 撞墙记录（2026-10-05 首次实跑）：Sana 的 `negative_prompt=None` 会走到
    #    `_text_preprocessing` 里 `text.lower()` ⇒ `AttributeError: 'NoneType'...`
    #    ⇒ **必须传字符串，空负向提示用空串而不是 None**。
    kw = dict(
        prompt=prompt,
        negative_prompt=negative if negative else "",   # ← 不是 None
        height=size,
        width=size,
        num_inference_steps=steps,
    )
    if seed >= 0:
        kw["generator"] = torch.Generator(device="cpu").manual_seed(seed)

    t1 = time.time()
    with torch.inference_mode():
        img = pipe(**kw).images[0]
    dt = time.time() - t1

    name = f"gen_{time.strftime('%Y%m%d_%H%M%S')}_s{seed}.png"
    p = out_dir / name
    img.save(p)
    _log(f"✅ 完成 {dt:.1f}s → {p}")
    _log(f"   prompt 字符数={len(prompt)}（内容不记录，隐私设计）")

    # 侧车元数据：**只存尺寸/步数/种子，不存 prompt**
    meta = {"file": name, "size": size, "steps": steps, "seed": seed,
            "seconds": round(dt, 1), "variant": "bf16", "offload": bool(offload),
            "prompt_len": len(prompt), "note": "prompt 不落盘"}
    (out_dir / (name + ".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="本地出图器（Sana 1.6B，全离线）")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--4bit", action="store_true", help="用 int4 量化权重（更省显存）")
    ap.add_argument("--prompt", default="", help="⛔ 仅本地调试用；GUI 走文件")
    a = ap.parse_args()

    prompt = a.prompt or _read_prompt_file(PROMPT_FILE)
    if not prompt.strip():
        print("⛔ 没有 prompt。请双击 tools/gen_gui.ps1 在界面里填写。", flush=True)
        return 1
    neg = _read_prompt_file(NEG_FILE)

    print(f"📁 输出目录：{OUT_DIR}", flush=True)
    p = generate(prompt, negative=neg, size=a.size, steps=a.steps,
                 seed=a.seed, load_4bit=a.__dict__.get("_4bit", a.__dict__.get("4bit", False)))
    print(f"\n🖼  {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

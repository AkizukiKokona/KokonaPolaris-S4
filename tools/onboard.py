"""新环境一键体检 —— 「拿到云端仓库后第一条命令」。

回答四个问题（迁移最常见的四个坑）：
    ① 路径解析对不对？          ② 依赖装齐了吗？
    ③ 骨架自检过不过？          ④ 缺什么、怎么补？

用法：
    python tools/onboard.py            # 体检（只读，不装任何东西）
    python tools/onboard.py --install # 顺带按 requirements.lock.txt 装依赖
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kp.paths import KP_ROOT, MODELS_SANA, portable_report  # noqa: E402

OK, WARN, BAD = "✅", "⚠️", "❌"

# (import 名, pip 名, 必需?, 缺了会怎样)
DEPS = [
    ("torch", "torch", True, "一切的前提"),
    ("numpy", "numpy", True, "latent 打包 / 采样"),
    ("PIL", "pillow", True, "角色卡管线 / 出图"),
    ("transformers", "transformers", False, "文本塔（真实权重才需要）"),
    ("diffusers", "diffusers", False, "G1 靶子 Sana"),
    ("huggingface_hub", "huggingface_hub", False, "下载模型"),
    ("modelopt", "nvidia-modelopt", False, "G1 量化实测（⚠️ 别装非官方名）"),
    ("rembg", "rembg", False, "角色卡去底"),
    ("scipy", "scipy", False, "FID / 统计"),
]


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def check_deps() -> tuple[list[str], list[str]]:
    miss_req, miss_opt = [], []
    for mod, pip, req, why in DEPS:
        if _have(mod):
            continue
        (miss_req if req else miss_opt).append(f"{pip}（{why}）")
    return miss_req, miss_opt


def main() -> int:
    ap = argparse.ArgumentParser(description="新环境体检")
    ap.add_argument("--install", action="store_true", help="安装缺失依赖")
    a = ap.parse_args()

    print(portable_report())
    print()
    print("② 依赖检查")
    print("  " + "-" * 54)
    miss_req, miss_opt = check_deps()
    for mod, pip, req, why in DEPS:
        ok = _have(mod)
        mark = OK if ok else (BAD if req else WARN)
        note = "已装" if ok else ("缺失·必需" if req else "缺失·可选")
        print(f"  {mark} {mod:<18}{note:<12}{'' if ok else why}")
    if miss_req:
        print(f"\n  必需缺失：{', '.join(miss_req)}")
    if miss_opt:
        print(f"  可选缺失：{', '.join(miss_opt)}")

    print()
    print("③ 骨架自检（72 项 · 纯 CPU）")
    r = subprocess.run([sys.executable, "-m", "kp.selftest"],
                       cwd=str(KP_ROOT), capture_output=True, text=True)
    tail = [ln for ln in (r.stdout or "").splitlines() if "通过" in ln or "❌" in ln]
    for ln in tail[-3:]:
        print("  " + ln)
    if r.returncode != 0:
        print("  ❌ 自检未过（迁移前请先修好）")

    print()
    print("④ 缺口清单")
    print("  " + "-" * 54)
    items = []
    if not MODELS_SANA.is_dir():
        items.append("Sana 靶子未下载 → source env.sh && \"$KP_PY\" tools/fetch_sana.py"
                     "（约 4.6GB，**仅 G1/G5 验证需要**，日常写代码不需要）")
    if miss_req:
        items.append("必需依赖缺失 → pip install -r requirements.lock.txt")
    if not (KP_ROOT / "out").is_dir():
        items.append("out/ 不存在 → 自检会自动创建")
    for it in items:
        print(f"  · {it}")
    if not items:
        print("  无缺口 ✅")

    if a.install and (miss_req or miss_opt):
        lock = KP_ROOT / "requirements.lock.txt"
        if lock.is_file():
            print(f"\n安装中（{lock.name}）…")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(lock)], cwd=str(KP_ROOT))
        else:
            print(f"\n⚠️ 找不到 {lock}")

    print("\n提示：⚠️ items 里标「可选」的都不是必需 —— 骨架自检 72 项全靠纯 CPU，")
    print("      装好 torch/numpy/pillow 就能跑设计验证，不需要先下 4.6GB 靶子。")
    return 0 if not miss_req and r.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
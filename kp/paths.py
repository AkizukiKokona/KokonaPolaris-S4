"""kp.paths —— **路径唯一真源**（迁移安全层）。

## 为什么有这个模块
项目早期脚本里到处硬编码 `D:/model/...`，换机后会让脚本
**静默写到错误位置**或直接崩。本模块把「项目根」变成**运行时探测**：

    KP_ROOT 环境变量 > 从本文件位置向上两级 > 当前工作目录

于是同一份代码在 D:\\model、E:\\kp、/home/x/kp、GitHub Codespaces 上都能跑，
**不需要改一行路径**。

## 不变量
- 纯 CPU、零依赖，可被任何模块 import（含 tools/ 下的裸脚本）。
- 所有路径统一 `pathlib.Path`，比较用 `Path` 而非字符串（Windows 大小写）。
- **不创建目录**：只解析。写入方自己 mkdir，避免 import 副作用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "KP_ROOT", "MODELS", "DATA", "OUT", "CACHE", "REPOS", "KP_PY",
    "MODELS_SANA", "repo_root", "ensure", "rel", "portable_report",
]


def _detect_root() -> Path:
    """三级回退探测项目根。"""
    env = os.environ.get("KP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # kp/paths.py → 上一级是 kp/ → 再上一级是项目根
    here = Path(__file__).resolve()
    cand = here.parent.parent
    if (cand / "kp").is_dir() and (cand / "design").is_dir():
        return cand
    # 兜底：从 CWD 向上找带 kp/ 与 design/ 的祖先
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "kp").is_dir() and (p / "design").is_dir():
            return p
    return cand


KP_ROOT: Path = _detect_root()

MODELS: Path = KP_ROOT / "models"
DATA: Path = KP_ROOT / "data"
OUT: Path = KP_ROOT / "out"
CACHE: Path = KP_ROOT / ".cache"
REPOS: Path = KP_ROOT / "repos"

# G1 靶子（Sana 1.6B，Apache 2.0）。⚠️ 不入库，靠 tools/fetch_sana.py 重下。
MODELS_SANA: Path = MODELS / "Sana_1600M_1024px_BF16_diffusers"


def ensure(*p: Path) -> Path:
    """确保目录存在并返回它（唯一允许创建目录的入口）。"""
    d = p[0] if len(p) == 1 else KP_ROOT.joinpath(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_root() -> Path:
    """git 仓库根（KP_ROOT 的 git 顶层；不在仓库内则回退 KP_ROOT）。"""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(KP_ROOT), capture_output=True,
                           text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass
    return KP_ROOT


def rel(p) -> str:
    """相对 KP_ROOT 的可读路径（迁移报告用，避免满屏绝对路径）。"""
    try:
        return str(Path(p).resolve().relative_to(KP_ROOT)).replace("\\", "/")
    except Exception:
        return str(p)


def portable_report() -> str:
    """新环境体检：路径 + 关键目录 + 解释器状态。"""
    L = ["KP 路径解析（迁移体检）", "=" * 56]
    L.append(f"  探测方式   : {'KP_ROOT 环境变量' if os.environ.get('KP_ROOT') else '按 __file__ 向上定位'}")
    L.append(f"  KP_ROOT    : {KP_ROOT}")
    L.append(f"  git 仓库   : {repo_root()}")
    L.append("")
    L.append(f"  {'目录':<10}{'路径（相对项目根）':<34}{'状态'}")
    L.append("  " + "-" * 54)
    for name, d in [("models", MODELS), ("data", DATA), ("out", OUT),
                    ("cache", CACHE), ("repos", REPOS)]:
        mark = "✅ 存在" if d.is_dir() else ("📦 需创建" if name in ("out", "cache")
                                            else "⬜ 不存在（按需）")
        L.append(f"  {name:<10}{rel(d):<34}{mark}")
    # Sana 靶子：只报存在性，不扫描内容
    s = "✅ 已就位" if MODELS_SANA.is_dir() else "⬜ 未下载（跑 tools/fetch_sana.py）"
    L.append(f"  {'Sana靶子':<8}{rel(MODELS_SANA):<36}{s}")
    L.append("")
    L.append(f"  当前解释器 : {sys.executable}")
    L.append(f"  python     : {sys.version.split()[0]}  ({'64bit' if sys.maxsize > 2**32 else '32bit'})")
    L.append("")
    L.append("  提示：C:\\ 盘只读访问字体是允许的；所有写入应落在 KP_ROOT 内。")
    return "\n".join(L)


if __name__ == "__main__":
    print(portable_report())
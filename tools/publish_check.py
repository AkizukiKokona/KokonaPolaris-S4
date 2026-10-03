"""发布到云端前的一键体检 —— **只读，不改任何东西**。

回答四个问题：
    ① 会推上去什么？多大？      ② 有没有不该进去的东西（权重/密钥/大文件）？
    ③ 换机后能不能跑起来？      ④ 5070/5060 硬件差异是否需要改配置？

用法：
    python tools/publish_check.py            # 体检 + 摘要
    python tools/publish_check.py --json     # 机器可读
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kp.paths import KP_ROOT, MODELS_SANA  # noqa: E402

OK, WARN, BAD = "✅", "⚠️", "❌"

# ---- 该被忽略的东西（若出现在受控文件里 = 事故）----
FORBIDDEN_EXT = {".safetensors", ".ckpt", ".onnx", ".pt", ".pth", ".bin",
                 ".gguf", ".ckpt", ".h5", ".msgpack", ".pkl", ".npy", ".npz"}
FORBIDDEN_DIR = {"models", ".venv", ".cache", ".pipcache", "repos", "__pycache__",
                 "node_modules", ".mypy_cache", ".pytest_cache"}
SECRET_PAT = re.compile(r"(api[_-]?key|secret[_-]?key|access[_-]?token|"
                        r"password|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY)", re.I)

# ---- 硬件档案：新机换卡后对照 ----
HW_PROFILE = {
    "本机（本轮迁出）": "RTX 5050 Laptop · 8GB · 20 SM · TGP 45–100W · sm_120",
    "新机（迁入）": "RTX 5060 Laptop · 8GB · 26 SM · TGP 45–115W · sm_120",
}


def tracked() -> list[Path]:
    r = subprocess.run(["git", "ls-files", "-z"], cwd=str(KP_ROOT),
                       capture_output=True, text=True)
    return [KP_ROOT / p for p in r.stdout.split("\0") if p]


def repo_bytes() -> tuple[int, int]:
    """返回 (.git 体积, 受控文件总体积)，字节。"""
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0
    return du(KP_ROOT / ".git"), sum(f.stat().st_size for f in tracked() if f.is_file())


def git_remote() -> list[str]:
    r = subprocess.run(["git", "remote", "-v"], cwd=str(KP_ROOT),
                       capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _mb(b: int) -> str:
    return f"{b/1048576:.1f} MB"


def audit(scan_secrets: bool = True) -> dict:
    files = tracked()
    problems, secrets = [], []
    for f in files:
        rel = f.relative_to(KP_ROOT)
        if rel.parts and rel.parts[0] in FORBIDDEN_DIR:
            problems.append(f"目录不该入库：{rel}")
        if f.suffix.lower() in FORBIDDEN_EXT:
            problems.append(f"权重/二进制不该入库：{rel}（{_mb(f.stat().st_size)}）")
        if f.is_file() and f.stat().st_size > 5 * 1024 * 1024:
            problems.append(f"超过 5MB：{rel}（{_mb(f.stat().st_size)}）")
        if scan_secrets and f.suffix.lower() in {".py", ".sh", ".md", ".json", ".txt", ".env"}:
            try:
                for i, ln in enumerate(f.read_text(encoding="utf-8",
                                                   errors="ignore").splitlines(), 1):
                    if SECRET_PAT.search(ln):
                        secrets.append(f"{rel}:{i}  {ln.strip()[:70]}")
            except Exception:
                pass
    gitb, filesb = repo_bytes()
    return {
        "n_files": len(files),
        "git_mb": round(gitb / 1048576, 2),
        "worktree_mb": round(filesb / 1048576, 2),
        "problems": problems,
        "possible_secrets": secrets,
        "remotes": git_remote(),
        "models_present": MODELS_SANA.is_dir(),
        "hw": HW_PROFILE,
        "head": subprocess.run(["git", "log", "-1", "--format=%h %s"],
                               cwd=str(KP_ROOT), capture_output=True,
                               text=True).stdout.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = audit()

    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if not r["problems"] else 1

    print("① 会推上去什么")
    print(f"   受控文件 {r['n_files']} 个 ｜ 工作树 {_mb(int(r['worktree_mb']*1048576))}"
          f" ｜ .git {_mb(int(r['git_mb']*1048576))}")
    print(f"   HEAD：{r['head']}")
    print(f"   远端：{r['remotes'] or '⬜ 只有本地裸镜像（还没接云端）'}")
    print(f"   Sana 靶子：{'已在本机' if r['models_present'] else '未下载'} → **不入库**，新环境按需 fetch_sana.py")

    print("\n② 不该入库的东西")
    if r["problems"]:
        for p in r["problems"]:
            print(f"   {BAD} {p}")
    else:
        print(f"   {OK} 无权重、无大文件、无虚拟环境")

    print("\n③ 疑似密钥（需你确认）")
    if r["possible_secrets"]:
        for s in r["possible_secrets"]:
            print(f"   {WARN} {s}")
    else:
        print(f"   {OK} 未发现密钥形态字符串")

    print("\n④ 换机硬件对照")
    for k, v in r["hw"].items():
        print(f"   {k:<18}{v}")
    print()
    print("   ⭐ 5060 vs 5050：**显存同为 8GB（硬约束不变）、sm_120（架构不变）、")
    print("      SM 20→26（算力 +30%）、TGP 上限 100→115W**。")
    print("      ⇒ **kp/config.py 的架构常量一个都不用改**；只有算力预算类结论会变宽松。")

    return 0 if not r["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
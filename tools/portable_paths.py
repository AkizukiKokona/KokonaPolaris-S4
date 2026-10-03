"""迁移工具 —— 把源码里的**硬编码绝对路径**改写成 `kp.paths` 调用。

## 为什么需要
项目早期脚本到处写死 `D:/model/...`。换机（新环境 / 云端仓库 / Linux）后，
这些路径会**静默指向不存在的位置**或写到别处 —— 比直接报错更难查。

## 策略（保守、可审计）
只改**字符串字面量**里的 `D:/model...`，且只替换成 `str(路径常量)` 形式：
    "D:/model/out/e5b"        →  str(OUT / "e5b")
    "D:/model/models/Sana..." →  str(MODELS_SANA)
- 注释 / docstring / 散文里的 `D:/model` **不动**（那是给人看的说明文字）。
- 每处替换都记录行号与原值，改完打印 diff 摘要，**可用 --check 只审计不改**。

用法：
    python tools/portable_paths.py --check      # 只审计，列出所有待改处
    python tools/portable_paths.py --apply      # 实际改写
    python tools/portable_paths.py --verify     # 改完后确认无残留
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kp.paths import KP_ROOT  # noqa: E402

# 只匹配出现在引号内的绝对路径（避免误伤注释与散文）
_LITERAL = re.compile(r"""(?P<q>['"])(?P<val>D:/model[^'"]*)(?P=q)""")

# 长路径优先匹配，避免前缀截断
# (硬编码前缀, 表达式, 承载该路径的 kp.paths 常量名)
_MAP = [
    ("D:/model/models/Sana_1600M_1024px_BF16_diffusers", "str(MODELS_SANA)", "MODELS_SANA"),
    ("D:/model/models", "str(MODELS)", "MODELS"),
    ("D:/model/data", "str(DATA)", "DATA"),
    ("D:/model/out", "str(OUT)", "OUT"),
    ("D:/model/.cache", "str(CACHE)", "CACHE"),
    ("D:/model/repos", "str(REPOS)", "REPOS"),
]

_IMPORT_BLOCK = "from kp.paths import {names}\n"

# 只改这些目录（设计稿 .md/.html 里的路径是给人看的，不动）
_SCAN_DIRS = ("tools", "kp")


def _tail(val: str) -> tuple[str, str] | None:
    """返回 (新表达式, 需要的常量名)；未覆盖则 None。"""
    for prefix, expr, name in _MAP:
        if val == prefix or val.startswith(prefix + "/") or val.startswith(prefix + "\\"):
            rest = val[len(prefix):].lstrip("/\\").replace("\\", "/")
            return (expr if not rest else f'{name} / "{rest}"'), name
    return None


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    """返回 [(文件, 行号, 原值, 新表达式)]，仅含字面量命中。"""
    hits = []
    for d in _SCAN_DIRS:
        for f in sorted((root / d).rglob("*.py")):
            if f.name == Path(__file__).name:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if "D:/model" not in line:
                    continue
                for m in _LITERAL.finditer(line):
                    got = _tail(m.group("val"))
                    if got:
                        hits.append((f, i, m.group("val"), got[0]))
    return hits


def _needed_names(hits) -> list[str]:
    used: set[str] = set()
    for _, _, _, expr in hits:
        for n in ("MODELS_SANA", "MODELS", "DATA", "OUT", "CACHE", "REPOS"):
            if re.search(rf"\b{n}\b", expr):
                used.add(n)
    return sorted(used)


def _insert_import(src: str, names: list[str]) -> str:
    if "from kp.paths import" in src:
        return src
    lines = src.splitlines(keepends=True)
    # 插在最后一个 __future__ 之后；没有就插在首个 import 前
    idx = 0
    for i, ln in enumerate(lines):
        if ln.startswith("from __future__"):
            idx = i + 1
    block = _IMPORT_BLOCK.format(names=", ".join(names))
    if idx:
        lines.insert(idx, "\n" + block.rstrip("\n") + "\n")
    else:
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ")):
                idx = i
                break
        lines.insert(idx, block)
    return "".join(lines)


def apply(root: Path, hits) -> int:
    by_file: dict[Path, list] = {}
    for f, ln, old, new in hits:
        by_file.setdefault(f, []).append((ln, old, new))
    n = 0
    for f, items in by_file.items():
        src = f.read_text(encoding="utf-8")
        for _, old, new in items:
            src = src.replace(f'"{old}"', new).replace(f"'{old}'", new)
            n += 1
        src = _insert_import(src, _needed_names(hits))
        f.write_text(src, encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="绝对路径可移植化")
    ap.add_argument("--check", action="store_true", help="只审计")
    ap.add_argument("--apply", action="store_true", help="实际改写")
    ap.add_argument("--verify", action="store_true", help="确认无残留")
    a = ap.parse_args()
    hits = scan(KP_ROOT)

    if a.verify:
        left = scan(KP_ROOT)
        print(f"剩余字面量硬编码：{len(left)}")
        for f, ln, old, _ in left:
            print(f"  {f}:{ln}  {old}")
        return 1 if left else 0

    if a.apply:
        n = apply(KP_ROOT, hits)
        print(f"✅ 改写 {n} 处，覆盖 {len({h[0] for h in hits})} 个文件")
        print(f"   涉及常量：{', '.join(_needed_names(hits))}")
    else:
        print(f"待改写 {len(hits)} 处，覆盖 {len({h[0] for h in hits})} 个文件：\n")
        cur = None
        for f, ln, old, new in hits:
            if f != cur:
                cur = f
                print(f"  {f.relative_to(KP_ROOT)}")
            print(f"    {ln:>4}: {old}  →  {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
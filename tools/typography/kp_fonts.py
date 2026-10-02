"""kp_fonts.py —— 系统字体发现与注册（E6 / T0 确定性排版）。

只读访问 C:/Windows/Fonts（允许），不写 C 盘。
.ttc 为字体集合，用 fontTools 枚举 face 并取 index。
不 import torch，不做任何 GPU 操作。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

FONT_DIRS = [
    "C:/Windows/Fonts",
    os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
]

# 优先探测的候选字体文件（中文/日文/等宽）
CANDIDATES = [
    "msyh.ttc", "msyhbd.ttc", "msyhl.ttc",      # 微软雅黑
    "simhei.ttf",                                # 黑体
    "simsun.ttc", "simsunb.ttf",                 # 宋体 / 新宋体
    "simkai.ttf", "simfang.ttf",                 # 楷体 / 仿宋
    "Deng.ttf", "Dengb.ttf", "Dengl.ttf",        # 等线
    "msjh.ttc", "msjhbd.ttc", "msjhl.ttc",       # 微软正黑（繁）
    "YuGothM.ttc", "YuGothB.ttc", "YuGothR.ttc", # 游ゴシック
    "YuMincho.ttc", "yumin.ttf",                 # 遊明朝
    "meiryo.ttc", "meiryob.ttc",                 # メイリオ
    "malgun.ttf", "malgunbd.ttf",                # 韩文
    "STXIHEI.TTF", "STZHONGS.TTF", "STSONG.TTF", "STKAITI.TTF", "STFANGSO.TTF",
    "FZSTK.TTF", "FZYTK.TTF",
    "simsun.ttc",
]

# 用作「已知含中文字形」的判定字符串（心夏北极星为本项目验收样例）
_CJK_PROBE = "心夏北极星"


@dataclass
class FontFace:
    path: str
    index: int          # .ttc 内的 face index；非集合为 0
    family: str
    subfamily: str
    is_cjk: bool = False
    has_probe: bool = False
    kind: str = "unknown"   # cjk-simplified / cjk-traditional / jp / kr / latin

    @property
    def key(self) -> str:
        return f"{os.path.basename(self.path)}#{self.index}:{self.subfamily}"

    def __str__(self) -> str:
        flags = []
        if self.is_cjk:
            flags.append("CJK")
        if self.has_probe:
            flags.append("HAS「心夏北极星」")
        return f"{self.family} [{self.subfamily}] {os.path.basename(self.path)}#{self.index} ({self.kind}) {' '.join(flags)}"


def _name_records(path: str, index: int):
    from fontTools.ttLib import TTFont, TTCollection
    family = subfamily = "?"
    try:
        if path.lower().endswith(".ttc"):
            coll = TTCollection(path, lazy=True)
            if index < len(coll.fonts):
                f = coll.fonts[index]
            else:
                return "?", "?", []
        else:
            f = TTFont(path, fontNumber=index, lazy=True)

        names = {}
        for rec in f["name"].names:
            try:
                s = rec.toUnicode()
            except Exception:
                continue
            names.setdefault(rec.nameID, []).append(s)
        family = (names.get(1) or ["?"])[0]
        subfamily = (names.get(2) or ["Regular"])[0]
        # 收集 cmap 覆盖用于 CJK 判定
        cov = set()
        try:
            cmap = f.getBestCmap()
            if cmap:
                cov = set(cmap.keys())
        except Exception:
            pass
        f.close()
        return family, subfamily, cov
    except Exception:
        return "?", "?", []


def _num_faces(path: str) -> int:
    try:
        if path.lower().endswith(".ttc"):
            from fontTools.ttLib import TTCollection
            return len(TTCollection(path, lazy=True).fonts)
        return 1
    except Exception:
        return 0


def _classify(family: str, path: str) -> str:
    fn = os.path.basename(path).lower()
    fam = family.lower()
    if any(k in fn for k in ("msjh", "yu", "meiryo", "malgun")) or "jhenghei" in fam:
        return "cjk-traditional"
    if any(k in fam for k in ("雅黑", "黑体", "宋体", "等线", "楷体", "仿宋")) or \
       any(k in fn for k in ("msyh", "simhei", "simsun", "simkai", "simfang", "deng")):
        return "cjk-simplified"
    if any(k in fam for k in ("yahei", "simhei", "simsun", "kai", "fang")):
        return "cjk-simplified"
    return "unknown"


def discover(verbose: bool = False) -> list[FontFace]:
    """扫描候选字体，返回所有可用 face。"""
    faces: list[FontFace] = []
    seen = set()
    for fname in CANDIDATES:
        for d in FONT_DIRS:
            path = os.path.join(d, fname)
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            n = _num_faces(path)
            for idx in range(n):
                family, subfamily, cov = _name_records(path, idx)
                is_cjk = bool(cov & set(map(ord, _CJK_PROBE))) or \
                    bool(cov & set(range(0x4E00, 0x4E00 + 256)))
                has_probe = all(ord(c) in cov for c in _CJK_PROBE) if cov else False
                faces.append(FontFace(
                    path=path, index=idx, family=family, subfamily=subfamily,
                    is_cjk=is_cjk, has_probe=has_probe,
                    kind=_classify(family, path),
                ))
    return faces


def best_cjk_faces() -> list[FontFace]:
    return [f for f in discover() if f.has_probe]


if __name__ == "__main__":
    fs = discover()
    print(f"共发现 {len(fs)} 个 face：\n")
    for f in fs:
        print(" ", f)

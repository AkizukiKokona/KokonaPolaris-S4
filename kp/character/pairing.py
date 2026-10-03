"""多视角配对数据扩增 —— **声明式**配对（不是从文件名猜）。

## 为什么必须声明而不是猜

原 `stage_pair` 靠文件名里有没有 `front` / 以 `f` 开头来判"这是正视图"。
这是**猜**，和之前被删掉的「用颜色规则猜语义层」是同一类错误：
一旦数据线换命名（渲染器导出、LoRA 批量生成、别处拿来的图），
猜测会**静默给出错误的配对**，而错误的配对直接污染 Fitter 的训练目标
—— 比报错难查得多。

⇒ 正确做法：**轴取值是机器产出的元数据**（渲染器知道视角；生成脚本知道提示词），
  用户**仍然只交 3 项**（图 + caption + tag），不额外手工标注。

## 什么是「一条合法的配对」

设计稿的**四旋钮**（字形光栅化 / 视角 / 着色器 / 分层可见性 / 姿态）里，
四者**正交** ⇒ 配对样本可以**组合式扩增**。角色卡线只用其中一个子集：

    同一角色 + **只动 vary 列出的轴** + **match 列出的轴完全相同**

这样得到的是「**单旋钮变化**」的训练对 —— 正是身份不变、几何/外观可控的监督信号。

⚠️ 只要有一列缺失，就**不给**这条记录配对（记为 `undeclared`），
绝不"用默认值凑一个"。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# 四旋钮中属于角色线的三个（字形光栅化属排版线，不在角色卡里）
AXES: Tuple[str, ...] = ("view", "pose", "shader")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class Record:
    """一条待配对样本。`attrs` 里**已声明**的轴取值（缺的键就是没声明）。"""
    key: str
    identity: str
    attrs: Dict[str, str] = field(default_factory=dict)

    def get(self, axis: str) -> Optional[str]:
        v = self.attrs.get(axis)
        if v is None:
            return None
        v = str(v).strip()
        return v or None


@dataclass(frozen=True)
class PairSpec:
    """配对规格：`vary` 列出的轴必须**不同**，`match` 列出的轴必须**相同**。"""
    vary: Tuple[str, ...] = ("view",)
    match: Tuple[str, ...] = ()
    identity: str = "character"

    def required_axes(self) -> Tuple[str, ...]:
        return tuple(self.vary) + tuple(self.match)


@dataclass
class Pair:
    identity: str
    anchor: str
    positive: str
    varied: Dict[str, Tuple[str, str]]     # 轴 → (anchor 取值, positive 取值)

    def as_dict(self) -> dict:
        return {"identity": self.identity, "anchor": self.anchor,
                "positive": self.positive,
                "varied": {k: list(v) for k, v in self.varied.items()}}


@dataclass
class PairReport:
    spec: PairSpec
    n_records: int
    pairs: List[Pair]
    coverage: Dict[str, Dict[str, Counter]]     # identity → axis → {取值: 条数}
    undeclared: List[Tuple[str, List[str]]]     # (record key, 缺哪些轴)
    isolated: List[str]                         # 声明齐全但凑不出对的记录
    gaps: List[str]                             # 人话版缺口说明
    balanced: bool = True

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def paired_keys(self) -> set:
        s = set()
        for p in self.pairs:
            s.add(p.anchor)
            s.add(p.positive)
        return s

    def as_dict(self) -> dict:
        return {
            "spec": {"vary": list(self.spec.vary), "match": list(self.spec.match),
                     "identity": self.spec.identity},
            "n_records": self.n_records, "n_pairs": self.n_pairs,
            "pairs": [p.as_dict() for p in self.pairs],
            "coverage": {i: {a: dict(c) for a, c in ax.items()}
                         for i, ax in self.coverage.items()},
            "undeclared": [{"key": k, "missing": m} for k, m in self.undeclared],
            "isolated": self.isolated,
            "gaps": self.gaps,
            "balanced": self.balanced,
        }


# ---------------------------------------------------------------------------
# 配对
# ---------------------------------------------------------------------------
def build_pairs(records: Sequence[Record], spec: PairSpec = PairSpec(),
                *, target_values: Optional[Dict[str, Sequence[str]]] = None,
                name_of: Optional[Dict[str, str]] = None) -> PairReport:
    """构建「同身份 + 单/多旋钮变化」的配对集，并给出缺口报告。

    `target_values` : 轴 → 期望取值的全集（给了就能报「还缺哪些视角」）。
    `name_of`       : key → 人类可读名（用于报告）。
    """
    need = spec.required_axes()
    name_of = name_of or {}
    undeclared: List[Tuple[str, List[str]]] = []
    usable: List[Record] = []

    for r in records:
        missing = [a for a in need if r.get(a) is None]
        if missing:
            undeclared.append((r.key, missing))
        else:
            usable.append(r)

    by_id: Dict[str, List[Record]] = defaultdict(list)
    for r in usable:
        by_id[r.identity].append(r)

    coverage: Dict[str, Dict[str, Counter]] = {}
    pairs: List[Pair] = []
    isolated: List[str] = []
    gaps: List[str] = []

    for ident, rs in by_id.items():
        cov: Dict[str, Counter] = {a: Counter() for a in need}
        for r in rs:
            for a in need:
                cov[a][r.get(a)] += 1
        coverage[ident] = cov

        paired = set()
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if any(a.get(m) != b.get(m) for m in spec.match):
                    continue
                varied = {}
                for v in spec.vary:
                    va, vb = a.get(v), b.get(v)
                    if va == vb:
                        varied = {}
                        break
                    varied[v] = (va, vb)
                if not varied:
                    continue
                pairs.append(Pair(ident, a.key, b.key, varied))
                paired.add(a.key)
                paired.add(b.key)
        for r in rs:
            if r.key not in paired:
                isolated.append(r.key)

        # ---- 缺口 ----
        if target_values:
            for a in spec.vary:
                want = [str(x) for x in target_values.get(a, ())]
                have = set(cov[a])
                miss = [x for x in want if x not in have]
                if miss:
                    gaps.append(
                        f"{ident}：轴「{a}」缺取值 {'/'.join(miss)}"
                        f"（已有 {'/'.join(sorted(have))}）")
        if len(rs) < 2:
            gaps.append(f"{ident}：只有 {len(rs)} 条可用记录 → 凑不出配对"
                        f"（角色卡最少 正视图 + 背视图）")

    # ---- 平衡性：各身份产出的对数不应差一个数量级 ----
    per_ident = Counter(p.identity for p in pairs)
    balanced = True
    if len(per_ident) > 1:
        lo, hi = min(per_ident.values()), max(per_ident.values())
        balanced = (lo * 5 >= hi)

    if undeclared:
        gaps.append(
            f"{len(undeclared)} 条记录**未声明**轴取值 → 不给它们配对"
            f"（缺列：{'/'.join(sorted({a for _, m in undeclared for a in m}))}）。"
            f"⚠️ 这不是要你手工标注：轴取值应由**渲染器 / 生成脚本**写进 manifest。")

    return PairReport(spec=spec, n_records=len(records), pairs=pairs,
                      coverage=coverage, undeclared=undeclared,
                      isolated=isolated, gaps=gaps, balanced=balanced)


def format_pair_report(rep: PairReport, *,
                       name_of: Optional[Dict[str, str]] = None,
                       max_pairs: int = 12) -> str:
    name_of = name_of or {}
    L = []
    L.append("=" * 74)
    L.append("  多视角配对（单旋钮变化）· 角色卡数据线")
    L.append("=" * 74)
    L.append(f"  配对规格：身份「{rep.spec.identity}」"
             f"｜**必须不同**的轴 {list(rep.spec.vary)}"
             f"｜**必须相同**的轴 {list(rep.spec.match) or '（无）'}")
    L.append(f"  记录 {rep.n_records} 条 → 可用配对 **{rep.n_pairs}** 对"
             f"｜已配对记录 {len(rep.paired_keys)} 条")
    L.append("")

    L.append("-" * 74)
    L.append("  各轴的取值覆盖")
    L.append("-" * 74)
    for ident, cov in rep.coverage.items():
        L.append(f"  [{ident}]")
        for axis, c in cov.items():
            items = "  ".join(f"{v}×{n}" for v, n in c.most_common())
            L.append(f"    {axis:<8s} {items or '（无）'}")
    L.append("")

    if rep.n_pairs:
        L.append("-" * 74)
        L.append(f"  配对样例（前 {min(max_pairs, rep.n_pairs)} 对）")
        L.append("-" * 74)
        for p in rep.pairs[:max_pairs]:
            vary = " ".join(f"{k}:{a}→{b}" for k, (a, b) in p.varied.items())
            L.append(f"    {name_of.get(p.anchor, p.anchor):<28s} ↔ "
                     f"{name_of.get(p.positive, p.positive):<28s} [{vary}]")
        if rep.n_pairs > max_pairs:
            L.append(f"    ... 另有 {rep.n_pairs - max_pairs} 对")
        L.append("")

    if rep.isolated:
        L.append("-" * 74)
        L.append(f"  声明齐全但**凑不出对**（{len(rep.isolated)} 条）")
        L.append("-" * 74)
        for k in rep.isolated[:10]:
            L.append(f"    {name_of.get(k, k)}")
        if len(rep.isolated) > 10:
            L.append(f"    ... 另有 {len(rep.isolated) - 10} 条")
            L.append("")

    if rep.gaps:
        L.append("-" * 74)
        L.append("  缺口（**该补拍什么**）")
        L.append("-" * 74)
        for g in rep.gaps:
            L.append(f"    ⚠️ {g}")
        L.append("")

    L.append("=" * 74)
    ok = rep.n_pairs > 0 and not rep.gaps and rep.balanced
    L.append("  " + ("✅ 配对充足" if ok else "❌ 配对不足或分布失衡"))
    if rep.n_pairs and not rep.balanced:
        L.append("     ⚠️ 各身份产出的对数差一个数量级以上（少数角色会欠拟合）")
    L.append("=" * 74)
    return "\n".join(L)


def load_records(csv_path: str, *, identity_col: str = "character",
                 axes: Sequence[str] = AXES) -> Tuple[List[Record], Dict[str, str]]:
    """从 manifest.csv 读记录。

    `identity_col` 缺省为 `character`；若表里没有该列，则退回用 `tag`
    （第一级交付只有 `file,tag,caption,source` 四列，此时 tag 就是角色分组）。
    """
    import csv as _csv
    import os as _os

    records: List[Record] = []
    names: Dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rd = _csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
        real_id = cols.get(identity_col) or cols.get("tag")
        for row in rd:
            key = (row.get(cols.get("file", "file")) or "").strip()
            ident = (row.get(real_id) or "default").strip() if real_id else "default"
            attrs = {}
            for a in axes:
                if a in cols:
                    attrs[a] = (row.get(cols[a]) or "").strip()
            records.append(Record(key=key, identity=ident, attrs=attrs))
            names[key] = _os.path.basename(key)
    return records, names


__all__ = ["AXES", "Record", "PairSpec", "Pair", "PairReport",
           "build_pairs", "format_pair_report", "load_records"]

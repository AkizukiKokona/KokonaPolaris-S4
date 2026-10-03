"""caption 语料审计 —— 语言配比 / 标签串检测 / 汉字覆盖（路线 P1.8）。

三个判据，对应三个**不可逆**的设计承诺：

  1. **语言配比**：中文为主 + 英文对齐。文本塔是 LLM ⇒ 中文原生，但占比要主动设计。
  2. **形态**：**自然语言句子**，不是逗号标签串（tag soup）。
     理由不只是"好听"——标签串把「用户的输入咒语」变成模型的隐式条件通路，
     从而丧失「显式条件轴」的可标定/可审计/可回滚特性（见设计稿三层控制栈 vs G3.5）。
  3. **汉字覆盖**：为文本塔 tokenizer 的词表覆盖把关（哪些字至今没出现过）。

⚠️ 本模块**不做**分词、**不依赖**任何第三方库；只用 Unicode 码点区间判定。
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import CAPTION

# ---------------------------------------------------------------------------
# 字符区间
# ---------------------------------------------------------------------------
# CJK 统一表意文字（含扩展 A）+ 兼容表意文字
CJK_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x3400, 0x4DBF),    # 扩展 A
    (0x4E00, 0x9FFF),    # 基本区
    (0xF900, 0xFAFF),    # 兼容表意文字
)
# 日文假名（出现即判 ja —— 对「日系画风」这批数据很重要，别把日文算成中文）
KANA_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x3040, 0x309F),    # 平假名
    (0x30A0, 0x30FF),    # 片假名
)
FULLWIDTH_PUNCT = "，。、；：？！（）【】《》「」『』“”‘’…—·％℃·,.;:?!)]}>"

# 标签串的分隔符（含中英文标点）
_TAG_SEP = re.compile(r"[,，、;；|/\\\n\t]+")
# 句末标点（自然语言的弱证据）
_SENT_PUNCT = re.compile(r"[。！？!?…]")
# 英文/半角收尾：以 . ! ? 结尾（允许后面跟引号/括号）
_TERMINAL = re.compile(r"[.!?][\"')\]}）】》」』”’]*\s*$")
# 自然语言的弱证据：常见虚词 / 连接 / 动作词
_GLUE_WORDS = re.compile(
    r"(的|了|在|是|和|与|及|把|被|正|着|有|一(个|只|位|名|片|张)|"
    r"正在|站在|坐在|看向|望着|背景|前景|阳光|光线|画面|视角|镜头|效果)")


def _has_sentence_punct(text: str) -> bool:
    """是否具备句末标点。

    ⚠️ 不能只看全角标点：英文自然语言以半角 `.` 收尾，
        但**标签串里的逗号**也常伴半角点（如 "8k." 结尾）。
        折中：全角句末标点直接算；半角只认**行尾**的 `.`/`!`/`?`。
    """
    if _SENT_PUNCT.search(text):
        return True
    return bool(_TERMINAL.search(text))


def _in_ranges(ch: str, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def _is_cjk(ch: str) -> bool:
    return _in_ranges(ch, CJK_RANGES)


def _is_kana(ch: str) -> bool:
    return _in_ranges(ch, KANA_RANGES)


# ---------------------------------------------------------------------------
# 1. 语言判定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LangMix:
    """一条 caption 的字符构成（**排除空白**后计数）。"""
    n_chars: int          # 非空白字符总数
    cjk: int
    kana: int
    latin: int            # 拉丁字母**个数**
    digit: int
    punct: int
    other: int
    latin_words: int = 0  # 拉丁**词**数（判混排用词数，不用字母数）

    @property
    def lang(self) -> str:
        """zh / ja / en / mixed / other。

        ⚠️ 判定用**词数**而非**字母数**：中文句里夹一个英文词
        （「穿着 white dress」）时，拉丁字母数很容易压过汉字数，
        但它在语义上仍是**中文 caption**。按字母数判会把这类句子
        误判成 mixed，从而错误触发"混排偏高"告警。
        """
        if self.n_chars == 0:
            return "other"
        if self.kana > 0:
            return "ja"                      # 先看假名：日系数据别算成中文
        if self.cjk == 0 and self.latin_words == 0:
            return "other"
        if self.latin_words == 0:
            return "zh"
        if self.cjk == 0:
            return "en"
        # 中英并存：少量英文词视为"夹词"（仍算中文）；少量汉字同理算英文
        if self.latin_words <= 2 and self.cjk >= 8:
            return "zh"
        if self.cjk <= 2 and self.latin_words >= 4:
            return "en"
        return "mixed"

    @property
    def zh_ratio(self) -> float:
        return self.cjk / max(1, self.n_chars)


_LATIN_WORD = re.compile(r"[A-Za-z]+")


def detect_language(text: str) -> str:
    """便捷入口：返回 zh / ja / en / mixed / other。"""
    return summarize_chars(text).lang


def summarize_chars(text: str) -> LangMix:
    c = Counter()
    for ch in text:
        if ch.isspace():
            continue
        c["n"] += 1
        if _is_kana(ch):
            c["kana"] += 1
        elif _is_cjk(ch):
            c["cjk"] += 1
        elif ch.isascii() and ch.isalpha():
            c["latin"] += 1
        elif ch.isdigit():
            c["digit"] += 1
        elif unicodedata.category(ch).startswith("P") or ch in FULLWIDTH_PUNCT:
            c["punct"] += 1
        else:
            c["other"] += 1
    return LangMix(c["n"], c["cjk"], c["kana"], c["latin"],
                   c["digit"], c["punct"], c["other"],
                   latin_words=len(_LATIN_WORD.findall(text)))


# ---------------------------------------------------------------------------
# 2. 标签串（tag soup）检测
# ---------------------------------------------------------------------------
def tag_soup_score(text: str) -> float:
    """0–1，越大越像「逗号标签串」而非自然语言句子。

    结构化判据（不硬编码任何具体标签词，避免过拟合到某个社区的写法）：
      · 被分隔符切出的片段**多**且**短**   → 像标签
      · 分隔符**密集**                     → 像标签
      · **没有句末标点**                   → 像标签
      · **没有虚词/连接词**                → 像标签
    """
    segs = [s.strip() for s in _TAG_SEP.split(text) if s.strip()]
    if len(segs) < 3:
        return 0.0                       # 一句话里有一两个逗号很正常

    n_sep = len(segs) - 1
    avg_len = mean(len(s) for s in segs)

    score = 0.0
    if avg_len <= 20:
        score += 0.4
    elif avg_len <= 30:
        score += 0.2
    if n_sep >= 4:
        score += 0.3
    if not _has_sentence_punct(text):
        score += 0.3
    if not _GLUE_WORDS.search(text):
        score += 0.1
    # 极短片段（≤ 8 字）占比高 —— 标签的强特征
    if sum(1 for s in segs if len(s) <= 8) / len(segs) >= 0.7:
        score += 0.2
    return min(1.0, score)


def is_tag_soup(text: str, threshold: Optional[float] = None) -> bool:
    t = CAPTION.tag_soup_threshold if threshold is None else threshold
    return tag_soup_score(text) >= t


# ---------------------------------------------------------------------------
# 3. 单条审计
# ---------------------------------------------------------------------------
@dataclass
class CaptionStat:
    text: str
    lang: str
    n_chars: int
    tag_soup: float
    has_sentence_punct: bool
    chars: Counter = field(default_factory=Counter)   # 汉字 → 次数

    def as_dict(self) -> dict:
        return {"lang": self.lang, "n_chars": self.n_chars,
                "tag_soup": round(self.tag_soup, 3),
                "has_sentence_punct": self.has_sentence_punct}


def analyze_caption(text: str) -> CaptionStat:
    t = (text or "").strip()
    mix = summarize_chars(t)
    chars = Counter(ch for ch in t if _is_cjk(ch))
    return CaptionStat(text=t, lang=mix.lang, n_chars=mix.n_chars,
                       tag_soup=tag_soup_score(t),
                       has_sentence_punct=_has_sentence_punct(t),
                       chars=chars)


# ---------------------------------------------------------------------------
# 4. 语料级审计
# ---------------------------------------------------------------------------
@dataclass
class CaptionAudit:
    n: int
    lang_counts: Dict[str, int]
    lang_share: Dict[str, float]
    violations: List[Tuple[int, str, str]]     # (行号, 类型, 摘要)
    length_stats: Dict[str, float]
    sentence_punct_rate: float
    char_set: Counter                          # 全部汉字 → 频次
    passed: bool
    issues: List[str]                          # 与目标配比的偏差说明

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "lang_counts": self.lang_counts,
            "lang_share": {k: round(v, 4) for k, v in self.lang_share.items()},
            "violations": [{"line": i, "kind": k, "detail": d}
                           for i, k, d in self.violations],
            "length_stats": {k: round(v, 2) for k, v in self.length_stats.items()},
            "sentence_punct_rate": round(self.sentence_punct_rate, 4),
            "unique_cjk_chars": len(self.char_set),
            "top_cjk_chars": self.char_set.most_common(40),
            "passed": self.passed,
            "issues": self.issues,
        }

    def format_report(self, *, max_violations: int = 20) -> str:
        L = []
        L.append("=" * 74)
        L.append("  caption 语料审计（P1.8 中文自然语言支持）")
        L.append("=" * 74)
        L.append(f"  样本数        : {self.n}")
        if self.n:
            ls = self.length_stats
            L.append(f"  字数          : 均值 {ls['mean']:.1f}｜中位 {ls['median']:.0f}"
                     f"｜min {ls['min']:.0f}｜max {ls['max']:.0f}")
            L.append(f"  含句末标点占比: {self.sentence_punct_rate:.1%}（自然语言的弱证据）")
            L.append(f"  出现的不同汉字: {len(self.char_set)} 个")
        L.append("")
        L.append("-" * 74)
        L.append("  语言配比")
        L.append("-" * 74)
        tgt_zh = CAPTION.zh_share_target
        for lang in ("zh", "en", "ja", "mixed", "other"):
            c = self.lang_counts.get(lang, 0)
            share = self.lang_share.get(lang, 0.0)
            mark = ""
            if lang == "zh":
                mark = f"   ← 目标 {tgt_zh:.0%}"
            elif lang == "en":
                lo, hi = CAPTION.en_share_band
                mark = f"   ← 目标 {lo:.0%}–{hi:.0%}"
            L.append(f"    {lang:<6s} {c:>6d}  {share:>7.1%}{mark}")
        L.append("")
        if self.issues:
            L.append("-" * 74)
            L.append("  与目标配比的偏差")
            L.append("-" * 74)
            for it in self.issues:
                L.append(f"    ⚠️ {it}")
            L.append("")
        if self.violations:
            L.append("-" * 74)
            L.append(f"  违规样本（共 {len(self.violations)} 条）")
            L.append("-" * 74)
            for i, kind, detail in self.violations[:max_violations]:
                L.append(f"    行{i:<5d} [{kind}] {detail}")
            if len(self.violations) > max_violations:
                L.append(f"    ... 另有 {len(self.violations) - max_violations} 条")
            L.append("")
        L.append("=" * 74)
        L.append("  " + ("✅ 通过：语料形态符合 P1.8 设计律" if self.passed
                         else "❌ 未通过：按上面修正后再定稿"))
        L.append("=" * 74)
        return "\n".join(L)


def audit_captions(captions: Sequence[str], *,
                   min_chars: Optional[int] = None,
                   max_chars: Optional[int] = None,
                   max_violation_rate: Optional[float] = None,
                   zh_band: Optional[Tuple[float, float]] = None,
                   en_band: Optional[Tuple[float, float]] = None) -> CaptionAudit:
    """对一批 caption 做语料级审计。返回 `CaptionAudit`。"""
    min_c = CAPTION.min_chars if min_chars is None else min_chars
    max_c = CAPTION.max_chars if max_chars is None else max_chars
    max_v = CAPTION.max_violation_rate if max_violation_rate is None else max_violation_rate
    zb = CAPTION.zh_share_band if zh_band is None else zh_band
    eb = CAPTION.en_share_band if en_band is None else en_band

    lang_counts: Counter = Counter()
    violations: List[Tuple[int, str, str]] = []
    lengths: List[int] = []
    char_set: Counter = Counter()
    n_punct = 0

    for i, raw in enumerate(captions):
        st = analyze_caption(raw)
        n = st.n_chars
        if n == 0:
            violations.append((i, "空", "caption 为空"))
            continue
        lang_counts[st.lang] += 1
        lengths.append(n)
        char_set.update(st.chars)
        if st.has_sentence_punct:
            n_punct += 1
        if n < min_c:
            violations.append((i, "过短", f"{n} 字 < {min_c}：{st.text[:40]}"))
        elif n > max_c:
            violations.append((i, "过长", f"{n} 字 > {max_c}：{st.text[:40]}"))
        if not CAPTION.allow_tag_soup and st.tag_soup >= CAPTION.tag_soup_threshold:
            violations.append(
                (i, "标签串",
                 f"标签串得分 {st.tag_soup:.2f}：{st.text[:48]}"))

    n = len(captions)
    lang_share = {k: (v / n if n else 0.0) for k, v in lang_counts.items()}

    # ---- 与目标配比的偏差 ----
    issues: List[str] = []
    zh = lang_share.get("zh", 0.0)
    en = lang_share.get("en", 0.0)
    ja = lang_share.get("ja", 0.0)
    mixed = lang_share.get("mixed", 0.0)
    if n:
        if zh < zb[0]:
            issues.append(f"中文占比 {zh:.1%} 低于下界 {zb[0]:.0%} —— "
                          f"中文能力会先天不足（文本塔是 LLM，这是可避免的损失）")
        elif zh > zb[1]:
            issues.append(f"中文占比 {zh:.1%} 高于上界 {zb[1]:.0%} —— "
                          f"英文对齐不足，跨语言泛化与英文 prompt 会退化")
        if en < eb[0]:
            issues.append(f"英文占比 {en:.1%} 低于 {eb[0]:.0%} —— 建议补英文对齐样本")
        elif en > eb[1]:
            issues.append(f"英文占比 {en:.1%} 高于 {eb[1]:.0%} —— 会挤压中文原生优势")
        if mixed > 0.25:
            issues.append(f"中英混排占比 {mixed:.1%} 偏高 —— 混排句子会让塔学到"
                          f"「同一句里切换语言」的伪相关，建议拆成单语样本")
        if ja > CAPTION.ja_share_max:
            issues.append(f"日文占比 {ja:.1%} 超过 {CAPTION.ja_share_max:.0%} —— "
                          f"日系画风需要日文，但不宜反客为主")

    viol_rate = len(violations) / n if n else 0.0
    passed = bool(n) and viol_rate <= max_v and not issues

    length_stats = {
        "mean": mean(lengths) if lengths else 0.0,
        "median": median(lengths) if lengths else 0.0,
        "min": float(min(lengths)) if lengths else 0.0,
        "max": float(max(lengths)) if lengths else 0.0,
    }
    return CaptionAudit(
        n=n, lang_counts=dict(lang_counts), lang_share=lang_share,
        violations=violations, length_stats=length_stats,
        sentence_punct_rate=(n_punct / n if n else 0.0),
        char_set=char_set, passed=passed, issues=issues,
    )


# ---------------------------------------------------------------------------
# 5. 反向用途：由目标配比反推「该准备多少条」
# ---------------------------------------------------------------------------
def suggest_language_plan(total: int) -> Dict[str, int]:
    """给定总条数，给出 zh / en / ja（可选）的建议配额。

    这是给用户看的「**还有多少缺口**」——数据池增量累积时最实用的一步。
    """
    total = int(total)
    lo, hi = CAPTION.zh_share_band
    elo, ehi = CAPTION.en_share_band
    zh = int(round(total * (lo + hi) / 2))
    en = int(round(total * (elo + ehi) / 2))
    ja = min(int(round(total * 0.08)), int(round(total * CAPTION.ja_share_max)))
    other = max(0, total - zh - en - ja)
    return {"zh": zh, "en": en, "ja": ja, "other": other, "total": total}


__all__ = [
    "CJK_RANGES", "LangMix", "CaptionStat", "CaptionAudit",
    "detect_language", "summarize_chars", "tag_soup_score", "is_tag_soup",
    "analyze_caption", "audit_captions", "suggest_language_plan",
]

"""Layout Planner —— 文本 → **JSON 版面结构**（确定性，T0 链路）。

设计稿 §4.9 / 补充 02：
  · 32× 压缩与文字**互斥**（40px 汉字只占 1.25 个 latent 格 → 编码阶段就被丢弃）
    ⇒ 文字必须走**独立的低压缩 ROI 分支**，而这条分支需要「字放在哪、多大」。
  · 本模块就是这个**规划器**：把文本规划成版面（文本框 + 行 + 字号 + 对齐），
    输出**纯 JSON**，不依赖任何模型权重 ⇒ **今天就能用、结果 100% 确定**。

⭐ 中日文排版要点（与西文不同）：
  · **可在任意字符间断行**（不需要空格），但**行首不能是标点**（避头尾）
  · 全角字符宽度 ≈ 1em，半角 ≈ 0.5em
  · 竖排：列从右往左推进
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# 行首禁则（避头尾）：这些字符不能出现在行首
NO_LINE_START = set("，。、；：？！）】》」』”’…—·%℃,.;:?!)]}>")
# 行尾禁则：这些字符不能出现在行尾
NO_LINE_END = set("（【《「『“‘([{<")


def char_width(ch: str, font_px: float) -> float:
    """粗略字宽：全角 ≈ 1em，半角 ≈ 0.55em，空格更窄。"""
    if ch.isspace():
        return font_px * 0.30
    return font_px if ord(ch) >= 0x2E80 else font_px * 0.55


def measure(text: str, font_px: float) -> float:
    return sum(char_width(c, font_px) for c in text)


def wrap(text: str, max_w: float, font_px: float) -> List[str]:
    """按宽度折行，带**避头尾**处理。适用于可任意断行的中日文。"""
    if not text:
        return [""]
    lines: List[str] = []
    cur, cur_w = "", 0.0
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur, cur_w = "", 0.0
            continue
        w = char_width(ch, font_px)
        if cur and cur_w + w > max_w:
            # 避头尾：若下一个字符不能在行首，则把它硬留在本行
            if ch in NO_LINE_START:
                cur += ch
                lines.append(cur)
                cur, cur_w = "", 0.0
                continue
            # 若本行末尾字符不能在行尾，把它推到下一行
            if cur and cur[-1] in NO_LINE_END:
                lines.append(cur[:-1])
                cur, cur_w = cur[-1], char_width(cur[-1], font_px)
            else:
                lines.append(cur)
                cur, cur_w = "", 0.0
        cur += ch
        cur_w += w
    if cur:
        lines.append(cur)
    return lines


@dataclass
class TextBlock:
    text: str
    size: int = 48                       # 字号（px）
    line_gap: float = 0.25               # 行距（相对字号）
    align: str = "left"                  # left | center | right
    direction: str = "h"                 # h 横排 | v 竖排

    def line_height(self) -> float:
        return self.size * (1.0 + self.line_gap)


@dataclass
class LayoutSpec:
    canvas: Tuple[int, int] = (1024, 1024)
    margin: int = 64
    blocks: List[TextBlock] = field(default_factory=list)


def plan(spec: LayoutSpec) -> Dict:
    """→ {"canvas":…, "boxes":[{"text","x","y","w","h","font_px","align","direction","lines"}]}"""
    W, H = spec.canvas
    avail_w = W - 2 * spec.margin
    avail_h = H - 2 * spec.margin
    boxes: List[Dict] = []
    warnings: List[str] = []
    y = float(spec.margin)

    for i, b in enumerate(spec.blocks):
        if b.direction == "v":
            # 竖排：列从右往左，列内字从上往下
            half = b.size * 0.5
            avail_col_h = avail_h
            per_col = max(1, int(avail_col_h // half))
            cols: List[str] = [b.text[j:j + per_col] for j in range(0, len(b.text), per_col)] or [""]
            col_w = b.line_height()
            w = min(avail_w, col_w * len(cols))
            h = min(avail_col_h, half * max(len(c) for c in cols))
            x = float(spec.margin)
            if b.align == "center":
                x = (W - w) / 2
            elif b.align == "right":
                x = W - spec.margin - w
            boxes.append({"text": b.text, "x": round(x, 2), "y": round(y, 2),
                          "w": round(w, 2), "h": round(h, 2), "font_px": b.size,
                          "align": b.align, "direction": "v", "columns": cols})
            y += h + b.size * 0.3
        else:
            lines = wrap(b.text, avail_w, b.size)
            lh = b.line_height()
            w = max(measure(ln, b.size) for ln in lines)
            h = lh * len(lines)
            x = float(spec.margin)
            if b.align == "center":
                x = (W - w) / 2
            elif b.align == "right":
                x = W - spec.margin - w
            boxes.append({"text": b.text, "x": round(x, 2), "y": round(y, 2),
                          "w": round(w, 2), "h": round(h, 2), "font_px": b.size,
                          "align": b.align, "direction": "h", "lines": lines})
            y += h + b.size * 0.3

        if y > H - spec.margin:
            warnings.append(f"第 {i} 个块溢出画布下边界（y={y:.0f} > {H - spec.margin}）")

    out = {"canvas": [W, H], "margin": spec.margin, "boxes": boxes, "warnings": warnings}
    out["valid"] = validate(out)
    return out


def validate(layout: Dict) -> bool:
    """所有框必须落在画布内，且两两不重叠。"""
    W, H = layout["canvas"]
    m = layout["margin"]
    boxes = layout["boxes"]
    for b in boxes:
        if b["x"] < -1e-6 or b["y"] < -1e-6 or b["x"] + b["w"] > W + 1e-6 or b["y"] + b["h"] > H + 1e-6:
            return False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, c = boxes[i], boxes[j]
            if (a["x"] < c["x"] + c["w"] and c["x"] < a["x"] + a["w"]
                    and a["y"] < c["y"] + c["h"] and c["y"] < a["y"] + a["h"]):
                return False
    return True


def roi_boxes(layout: Dict, pad: int = 8) -> List[Dict]:
    """给**低压缩 ROI 分支**用的 ROI 列表（含 padding，裁剪到画布内）。"""
    W, H = layout["canvas"]
    out = []
    for b in layout["boxes"]:
        x0 = max(0, int(b["x"]) - pad)
        y0 = max(0, int(b["y"]) - pad)
        x1 = min(W, int(b["x"] + b["w"]) + pad)
        y1 = min(H, int(b["y"] + b["h"]) + pad)
        out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "w": x1 - x0, "h": y1 - y0, "text": b["text"]})
    return out


__all__ = ["TextBlock", "LayoutSpec", "plan", "validate", "wrap",
           "measure", "char_width", "roi_boxes", "NO_LINE_START", "NO_LINE_END"]

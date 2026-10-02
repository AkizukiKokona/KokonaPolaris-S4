"""kp_schema.py —— 版面 JSON schema（严格对齐设计文档 §4.7.8 / 补充02 §4.1）+ 随机规划器。

设计原则（设计文档原话）：**版式规划与像素生成解耦**——
规划器只输出「符号 + 几何」，一个字像素都不碰；渲染由下游确定性地做。

======================================================================
字段语义（schema = {canvas, panels[], items[]}）
======================================================================

canvas : object  —— 画布定义
  .w, .h    : int    画布像素宽/高
  .reading  : str    阅读方向，"rtl"（右→左，日漫标准）| "ltr" | "ttb"
  .gutter   : int    面板之间的间隙（px）
  .dpi      : int    可选，印刷 dpi

panels : array  —— 分镜面板（可为空 = 单页无分镜）
  每项 { "poly": [[x,y],...] } —— 面板多边形（通常矩形 4 点，也可任意多边形）

items : array  —— 版面元素（文字框 / 气泡 / 拟声词等）——**规划器唯一的输出单元**
  公共字段：
    .type       : str   "bubble" 对话气泡 | "narration" 旁白框 | "sfx" 拟声词
                        | "text" 纯文本 | "caption" 图注
    .poly       : [[x,y],...]  文本框/气泡的多边形（顺时针）。文本在 poly 内排版。
    .text       : str   要渲染的文本内容（含标点）
    .font       : str   字体名（family，或 schema 别名如 SourceHanSans-Bold）
    .size       : int   字号（px，em 高度）
    .align      : str   "left" | "center" | "right" | "justify"（两端对齐，行内拉伸）
    .emphasis   : array | null  着重号/强调标注
                  形如 [ {"range":[s,e], "style":"dot"} ]，range 为 **半开区间 [s,e)**，
                  索引针对本 item 的 text（0-based，按字符计）。style: dot(着重号/傍点)
                  | sesame(胡麻点) | underline(旁线)。null/省略 = 无。
  气泡专有：
    .tail_to    : [x,y] | null  气泡「尾巴」指向的说话人锚点（绝对坐标）。
                                None = 无尾气泡（旁白/内心独白）。
  扩展（本项目渲染器支持，非文档强制）：
    .writing_mode : "horizontal"(默认) | "vertical"（縦書き）
    .color        : [r,g,b] | [r,g,b,a] | "#rrggbb"（文字色）
    .stroke_width : float  描边宽度（px），配合 stroke_color
    .stroke_color : [r,g,b] 描边色
    .line_height  : float  行高倍数（默认 1.2）
    .letter_spacing : float 字间距（px）
    .punct_squeeze : "none" | "compress" | "hang"  标点挤压策略
    .padding      : float  文本框内边距（px）
    .rot          : float  整体旋转角度（度，CCW），用于 sfx
    .style        : "fill" | "outline"
======================================================================
"""
from __future__ import annotations

import copy
import random
from typing import Optional

# 断言字段集合，便于外部校验
COMMON_FIELDS = {"type", "poly", "text", "font", "size", "align", "emphasis"}
BUBBLE_FIELDS = {"tail_to"}
EXT_FIELDS = {"writing_mode", "color", "stroke_width", "stroke_color", "line_height",
              "letter_spacing", "punct_squeeze", "padding", "rot", "style", "opacity",
              "valign"}

# ---------------------------------------------------------------------------
# 文本语料（合成抽样用；真实训练会换成 6763 常用字 × 语料）
# ---------------------------------------------------------------------------

DIALOGUE = [
    "你还活着啊。", "心夏北极星，终于亮起来了。", "别回头，它就在你身后。",
    "这就是……北极星的光？", "我说过，我会回来的。", "喂，你听见了吗？",
    "再往前走一步，就没有退路了。", "把星星摘下来，送给你。",
    "「答案一直都在这里。」", "你确定要这么做吗？", "风停了。",
    "我们只有一次机会。", "不要相信任何人。", "天亮之前必须离开。",
    "这条路，我走了十年。", "看清楚了——这才叫奇迹。",
    "四月的樱花，落了一地。", "第七次钟声响起的时候，门会开。",
    "把名字告诉我。", "疼吗？疼就对了。",
]
NARRATION = [
    "三个月后，世界安静得可怕。", "在那之前，没有人见过真正的夜空。",
    "所有的约定，都停在了那个夏天。", "船驶向没有地图的海域。",
    "这座城市从不下雪。", "档案编号：S4-0117，封存。",
]
SFX = ["咚", "轰——", "哗啦", "砰！", "滴答", "嗡——", "咔哒", "呼——", "轰隆", "啪"]
CAPTION = ["图注：北极星坐标", "第 4 话 / 心夏", "Fig.1 — 32× latent", "数据来源：合成"]

CHARSET_HINT = "心夏北极星abcdefghijklmnopqrstuvwxyz0123456789"


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def validate(layout: dict) -> list[str]:
    """返回错误列表（空 = 通过）。"""
    errs: list[str] = []
    if "canvas" not in layout:
        errs.append("缺少 canvas")
    else:
        for k in ("w", "h"):
            if k not in layout["canvas"]:
                errs.append(f"canvas 缺少 {k}")
    if "panels" not in layout:
        errs.append("缺少 panels")
    if "items" not in layout:
        errs.append("缺少 items")
        return errs
    W = layout.get("canvas", {}).get("w", 0)
    H = layout.get("canvas", {}).get("h", 0)
    for i, it in enumerate(layout["items"]):
        for k in ("type", "poly", "text"):
            if k not in it:
                errs.append(f"items[{i}] 缺少 {k}")
        poly = it.get("poly")
        if not poly or len(poly) < 3:
            errs.append(f"items[{i}].poly 至少 3 点")
        else:
            for p in poly:
                if not (0 <= p[0] <= W + 1 and 0 <= p[1] <= H + 1):
                    errs.append(f"items[{i}].poly 越界 {p}")
        if it.get("type") == "bubble" and it.get("tail_to") is not None:
            t = it["tail_to"]
            if not (isinstance(t, (list, tuple)) and len(t) == 2):
                errs.append(f"items[{i}].tail_to 应为 [x,y]")
        emph = it.get("emphasis")
        if emph:
            items = emph if isinstance(emph, list) else [emph]
            for e in items:
                if isinstance(e, dict) and "range" in e:
                    s, en = e["range"]
                    if not (0 <= s < en <= len(it.get("text", ""))):
                        errs.append(f"items[{i}].emphasis.range 越界 {e['range']}")
    return errs


# ---------------------------------------------------------------------------
# 随机规划器（= 未来 Layout Planner 的“教师/替身”；只出符号+几何）
# ---------------------------------------------------------------------------

FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",
         "DengXian", "Yu Gothic", "Microsoft JhengHei"]
ALIGNS = ["left", "center", "right", "justify"]


def _rect_poly(x, y, w, h):
    return [[int(x), int(y)], [int(x + w), int(y)],
            [int(x + w), int(y + h)], [int(x), int(y + h)]]


def _split_panels(x, y, w, h, depth, rng, gutter, out):
    """递归切分（几刀切法），生成 manga 式分镜。"""
    if depth <= 0 or (w < 260 and h < 260) or rng.random() < 0.25:
        out.append(_rect_poly(x, y, w, h))
        return
    vertical = (w >= h) if rng.random() < 0.8 else (h > w)
    # 竖切（左右分）或横切（上下分）
    if w >= h:
        f = rng.uniform(0.38, 0.62)
        lw = w * f
        _split_panels(x, y, lw - gutter / 2, h, depth - 1, rng, gutter, out)
        _split_panels(x + lw + gutter / 2, y, w - lw - gutter / 2, h, depth - 1, rng, gutter, out)
    else:
        f = rng.uniform(0.38, 0.62)
        th = h * f
        _split_panels(x, y, w, th - gutter / 2, depth - 1, rng, gutter, out)
        _split_panels(x, y + th + gutter / 2, w, h - th - gutter / 2, depth - 1, rng, gutter, out)


def plan_page(canvas_w: int = 1024, canvas_h: int = 1024, seed: int = 0,
              n_panels_hint: Optional[int] = None, margin: int = 40,
              gutter: int = 20) -> dict:
    """生成一页随机版面（纯符号+几何）。确定性：同 seed → 同结果。"""
    rng = random.Random(seed)
    canvas = {"w": canvas_w, "h": canvas_h,
              "reading": rng.choice(["rtl", "rtl", "ttb"]),
              "gutter": gutter, "dpi": 300}

    # 面板切分
    panels: list[dict] = []
    if rng.random() < 0.12:
        panels = []          # 无分镜整页
    else:
        depth = 1 if n_panels_hint else rng.choice([1, 2, 2, 3])
        _split_panels(margin, margin, canvas_w - 2 * margin, canvas_h - 2 * margin,
                      depth, rng, gutter, panels)
        # 过大面板再切
        panels = [p for p in panels]
    if not panels:
        panels = [_rect_poly(margin, margin, canvas_w - 2 * margin, canvas_h - 2 * margin)]

    items: list[dict] = []
    for pi, panel in enumerate(panels):
        xs = [p[0] for p in panel]
        ys = [p[1] for p in panel]
        px0, py0, px1, py1 = min(xs), min(ys), max(xs), max(ys)
        pw, ph = px1 - px0, py1 - py0

        # 每个面板 1–2 个元素
        n_items = rng.choices([1, 2, 3], weights=[55, 33, 12])[0]
        # 拟声词（大面积、旋转）
        if rng.random() < 0.22:
            sw = rng.uniform(0.35, 0.6) * min(pw, ph)
            sx = rng.uniform(px0 + 10, px1 - sw - 10)
            sy = rng.uniform(py0 + 10, py1 - sw - 10)
            items.append({
                "type": "sfx", "poly": _rect_poly(sx, sy, sw, sw),
                "tail_to": None, "text": rng.choice(SFX),
                "font": rng.choice(["SimHei", "Microsoft YaHei", "STXihei"]),
                "size": int(sw * 0.62), "align": "center",
                "emphasis": None, "style": "outline",
                "stroke_width": max(2, sw * 0.03), "stroke_color": [255, 255, 255],
                "color": [30, 30, 40], "rot": rng.uniform(-18, 18),
                "writing_mode": "horizontal",
            })
            n_items -= 1
        for _ in range(max(0, n_items)):
            bw = rng.uniform(0.46, 0.82) * pw
            bh = rng.uniform(0.30, 0.58) * ph
            bx = rng.uniform(px0 + 6, max(px0 + 7, px1 - bw - 6))
            by = rng.uniform(py0 + 6, max(py0 + 7, py1 - bh - 6))
            kind = rng.choices(["bubble", "narration", "caption"],
                               weights=[62, 30, 8])[0]
            if kind == "bubble":
                text = rng.choice(DIALOGUE)
                # 尾巴锚点：气泡近旁（距离 ≈ 0.9–1.45 倍半轴），夹在面板内
                bcx, bcy = bx + bw / 2, by + bh / 2
                ang = rng.uniform(0, 6.283)
                dist = rng.uniform(0.9, 1.45) * max(bw, bh) / 2
                import math
                tail = [int(min(px1, max(px0, bcx + math.cos(ang) * dist))),
                        int(min(py1, max(py0, bcy + math.sin(ang) * dist)))]
                pad = 12
                align, valign, vprob = "center", "middle", 0.10
            elif kind == "caption":
                text = rng.choice(CAPTION)
                tail = None
                pad = 6
                align, valign, vprob = "left", "top", 0.0
            else:
                text = rng.choice(NARRATION)
                tail = None
                pad = 10
                align, valign, vprob = rng.choice(["left", "center"]), "middle", 0.25
            vmode = "vertical" if rng.random() < vprob else "horizontal"
            size = 64   # 上界占位；实际字号由下游按框容量拟合
            emph = None
            # 偶发着重号
            if rng.random() < 0.18 and len(text) >= 3:
                s = rng.randint(0, len(text) - 2)
                emph = [{"range": [s, min(len(text), s + 2)],
                         "style": rng.choice(["dot", "sesame"])}]
            it = {
                "type": kind, "poly": _rect_poly(bx, by, bw, bh),
                "tail_to": tail, "text": text,
                "font": rng.choice(FONTS), "size": size,
                "align": align, "valign": valign,
                "emphasis": emph, "writing_mode": vmode,
                "line_height": round(rng.uniform(1.2, 1.4), 2),
                "letter_spacing": rng.choice([0, 0, 1, 2]),
                "punct_squeeze": rng.choice(["compress", "compress", "hang", "none"]),
                "padding": pad, "color": [20, 20, 25],
                "style": "fill",
            }
            if kind == "bubble":
                it["stroke_width"] = 0
            items.append(it)
    return {"canvas": canvas, "panels": panels, "items": items}


def _fit_size(text, max_w, max_h, size, mode, floor=10):
    """粗估字号使其放入框内（按字数近似，避免依赖引擎）。"""
    guard = 0
    while size > floor and guard < 40:
        guard += 1
        if mode == "vertical":
            rows = max(1, int(max_h // (size * 1.25)))
            cols = -(-len(text) // rows)
            if cols * size * 1.25 <= max_w and rows >= 1:
                # 检查竖排不溢出
                if (len(text) <= rows * max(1, int(max_w // (size * 1.25)))):
                    return size
        else:
            per = max(1, int(max_w // size))
            lines = -(-len(text) // per)
            if lines * size * 1.3 <= max_h:
                return size
        size -= 1
    return max(floor, size)

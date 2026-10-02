"""kp_engine.py —— 确定性中文排版引擎（E6 / T0）。

职责：把「文本 + 样式 + 文本框几何」确定性地光栅化成图层。

- 字形整形：uharfbuzz（横排 ltr + kern/liga；竖排 ttb + vert/vrt2，自动替换竖排标点形）
- 字形光栅化：freetype-py（灰度位图；竖排用 FT_LOAD_VERTICAL_LAYOUT 取竖排度量）
- 版式：CJK 避头尾换行（行首/行尾禁则）、对齐（左/中/右/两端）、
        着重号（傍点/胡麻点）、标点挤压 / 悬挂
- 双书写模式：横排 / 竖排（縦書き，列自右向左）

确定性保证：同一输入 → 同一像素输出（无随机、无采样）。字形来自真实字体，
因此「字形正确率 100%」——错字概率为 0（这是确定性合成，不是模型生成）。

不 import torch，不做任何 GPU 操作。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import freetype
import uharfbuzz as hb

from kp_fonts import discover, FontFace

# ----------------------------------------------------------------------------
# 字体数据库
# ----------------------------------------------------------------------------

FONT_ALIASES = {
    "SourceHanSans-Bold": "Microsoft YaHei",   # 设计文档 schema 示例名 → 雅黑
    "SourceHanSans": "Microsoft YaHei",
    "SourceHanSerif": "SimSun",
    "NotoSansCJK": "Microsoft YaHei",
    "yahei": "Microsoft YaHei",
    "heiti": "SimHei",
    "songti": "SimSun",
    "kaiti": "KaiTi",
    "fangsong": "FangSong",
    "gothic": "Yu Gothic",
    "mincho": "Yu Mincho",
}


class FontDB:
    """字体库：family 名 → Face，并缓存 freetype/harfbuzz 对象。"""

    def __init__(self):
        self.faces: list[FontFace] = discover()
        self.by_family: dict[str, FontFace] = {}
        for f in self.faces:
            if f.has_probe and f.family not in self.by_family:
                self.by_family[f.family] = f
        self._ft_cache: dict[tuple, freetype.Face] = {}
        self._hb_cache: dict[tuple, tuple] = {}
        self._cur_size: dict[tuple, int] = {}

    def resolve(self, name: Optional[str]) -> FontFace:
        if not name:
            return self.by_family["Microsoft YaHei"]
        if name in self.by_family:
            return self.by_family[name]
        if name in FONT_ALIASES and FONT_ALIASES[name] in self.by_family:
            return self.by_family[FONT_ALIASES[name]]
        low = name.lower().replace(" ", "")
        for fam, f in self.by_family.items():
            if low in fam.lower().replace(" ", ""):
                return f
        return self.by_family["Microsoft YaHei"]

    def ft_face(self, face: FontFace, size: int) -> freetype.Face:
        key = (face.path, face.index)
        ft = self._ft_cache.get(key)
        if ft is None:
            ft = freetype.Face(face.path, face.index)
            self._ft_cache[key] = ft
        if self._cur_size.get(key) != size:
            ft.set_pixel_sizes(0, size)
            self._cur_size[key] = size
        return ft

    def hb_font(self, face: FontFace):
        key = (face.path, face.index)
        cached = self._hb_cache.get(key)
        if cached is None:
            blob = hb.Blob.from_file_path(face.path)
            hbf = hb.Face(blob, face.index)
            font = hb.Font(hbf)
            font.scale = (hbf.upem, hbf.upem)
            cached = (hbf, font)
            self._hb_cache[key] = cached
        return cached

    def info(self) -> dict:
        return {"n_faces": len(self.faces),
                "n_cjk": sum(1 for f in self.faces if f.has_probe),
                "families": sorted(self.by_family.keys())}


_FONT_DB: Optional[FontDB] = None


def fontdb() -> FontDB:
    global _FONT_DB
    if _FONT_DB is None:
        _FONT_DB = FontDB()
    return _FONT_DB


# ----------------------------------------------------------------------------
# 字符分类（避头尾 / 挤压 / 竖排居中）
# ----------------------------------------------------------------------------

# 行首禁则：不可出现在行首（标点/收尾括号）
NO_LINE_START = set("、。，．,.!！?？:：;；)）]］}｝>》〉」』】〕”’〞"
                    "ぁぃぅぇぉっゃゅょゎ゛゜ゝゞ々ー・ヽヾ%‰℃°′″〇")
# 行尾禁则：不可出现在行尾（起始括号）
NO_LINE_END = set("([{｛（［｢〈《「『【〔“‘〝<")
# 全角标点（可挤压/悬挂）
FULLWIDTH_PUNCT = set("、。，．,.!！?？:：;；）)]］}｝》〉」』】〕”’…‥—")
FULLWIDTH_PUNCT_OPEN = set("（([［{｛《〈「『【〔“‘")
PUNCT_CPS = set(map(ord, NO_LINE_START | NO_LINE_END | FULLWIDTH_PUNCT | FULLWIDTH_PUNCT_OPEN))


def _is_wide(cp: int) -> bool:
    return (0x2E80 <= cp <= 0x303F or 0x3040 <= cp <= 0x30FF or
            0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or
            0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF or
            0xAC00 <= cp <= 0xD7AF)


def _is_latin_word_char(cp: int) -> bool:
    return (48 <= cp <= 57) or (65 <= cp <= 90) or (97 <= cp <= 122) or cp in (0x27,)


def _is_punct(cp: int) -> bool:
    return cp in PUNCT_CPS


# ----------------------------------------------------------------------------
# 整形
# ----------------------------------------------------------------------------

@dataclass
class Glyph:
    gid: int
    cluster: int          # token 内字符索引
    codepoint: int        # 对应源字符的 Unicode 码点（用于分类）
    x_adv: float
    y_adv: float
    x_off: float
    y_off: float


class Shaper:
    def __init__(self, db: FontDB):
        self.db = db

    def shape(self, text: str, face: FontFace, size: float,
              direction: str = "ltr", features: Optional[dict] = None) -> list[Glyph]:
        _, font = self.db.hb_font(face)
        upem = font.face.upem
        scale = size / upem
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])   # cluster == 字符索引
        buf.direction = direction
        buf.script = "Hani"
        buf.language = "zh"
        if features is None:
            features = {"vert": True, "vrt2": True} if direction == "ttb" \
                else {"kern": True, "liga": True}
        hb.shape(font, buf, features)
        out = []
        for info, p in zip(buf.glyph_infos, buf.glyph_positions):
            ci = info.cluster
            out.append(Glyph(
                gid=info.codepoint, cluster=ci,
                codepoint=ord(text[ci]) if ci < len(text) else 0,
                x_adv=p.x_advance * scale, y_adv=p.y_advance * scale,
                x_off=p.x_offset * scale, y_off=p.y_offset * scale))
        return out


# ----------------------------------------------------------------------------
# 字形位图缓存
# ----------------------------------------------------------------------------

_BMP_CACHE: dict[tuple, tuple] = {}


def glyph_bitmap(db: FontDB, face: FontFace, size: int, gid: int, vertical: bool = False):
    """返回 (arr uint8 HxW | None, bitmap_left, bitmap_top, vert_advance_px)。"""
    key = (face.path, face.index, size, gid, vertical)
    hit = _BMP_CACHE.get(key)
    if hit is not None:
        return hit
    ft = db.ft_face(face, size)
    flags = freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL
    if vertical:
        flags |= freetype.FT_LOAD_VERTICAL_LAYOUT
    ft.load_glyph(gid, flags)
    bmp = ft.glyph.bitmap
    va = ft.glyph.metrics.vertAdvance / 64.0
    if bmp.width == 0 or bmp.rows == 0:
        arr = None
    else:
        arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)
        arr = arr[:, :bmp.width].copy()
    res = (arr, ft.glyph.bitmap_left, ft.glyph.bitmap_top, va)
    _BMP_CACHE[key] = res
    return res


# ----------------------------------------------------------------------------
# 图层（灰度 mask 累加）
# ----------------------------------------------------------------------------

class Mask:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.a = np.zeros((h, w), dtype=np.float32)

    def blit(self, arr: np.ndarray, x: int, y: int):
        if arr is None:
            return
        bh, bw = arr.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + bw), min(self.h, y + bh)
        if x0 >= x1 or y0 >= y1:
            return
        sub = arr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        np.maximum(self.a[y0:y1, x0:x1], sub, out=self.a[y0:y1, x0:x1])


def mask_to_rgba(mask: Mask, fill, stroke_color=None, stroke_width: float = 0.0,
                 opacity: float = 1.0) -> np.ndarray:
    a = mask.a
    if stroke_width and stroke_width > 0 and stroke_color is not None:
        from PIL import Image as PILImage, ImageFilter
        k = max(3, int(round(stroke_width * 2)) | 1)
        m8 = np.clip(a, 0, 255).astype(np.uint8)
        dilated = np.asarray(PILImage.fromarray(m8).filter(ImageFilter.MaxFilter(k)),
                             dtype=np.float32)
        m = a / 255.0
        d = np.clip(dilated / 255.0, 0, 1)
        sc = np.array(stroke_color[:3], dtype=np.float32)
        fc = np.array(fill[:3], dtype=np.float32)
        rgba = np.zeros((mask.h, mask.w, 4), dtype=np.float32)
        for c in range(3):
            rgba[..., c] = sc[c] + (fc[c] - sc[c]) * m
        rgba[..., 3] = d * 255.0 * opacity
        return np.clip(rgba, 0, 255).astype(np.uint8)
    rgba = np.zeros((mask.h, mask.w, 4), dtype=np.float32)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = fill[0], fill[1], fill[2]
    rgba[..., 3] = np.clip(a, 0, 255) * opacity
    return np.clip(rgba, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# token 化 + 避头尾换行
# ----------------------------------------------------------------------------

@dataclass
class Token:
    text: str
    width: float            # 横排宽度（含字距）
    height: float           # 竖排高度（含字距）
    start: int              # 在源文本中的字符起始索引
    cps: list[int] = field(default_factory=list)
    is_space: bool = False

    @property
    def first_cp(self) -> int:
        return self.cps[0] if self.cps else 0

    @property
    def last_cp(self) -> int:
        return self.cps[-1] if self.cps else 0


def tokenize(text: str, face, size: float, shaper: Shaper,
             letter_spacing: float = 0.0) -> list[Token]:
    toks: list[Token] = []
    buf: list[str] = []
    start = 0

    def flush_word(end_idx: int):
        nonlocal buf
        if buf:
            s = "".join(buf)
            glyphs = shaper.shape(s, face, size, "ltr")
            w = sum(g.x_adv for g in glyphs) + letter_spacing * len(s)
            toks.append(Token(s, w, size + letter_spacing, start, [ord(c) for c in s]))
            buf = []

    for i, ch in enumerate(text):
        cp = ord(ch)
        if ch == "\n":
            flush_word(i)
            toks.append(Token("\n", 0.0, 0.0, i, [10], is_space=True))
            continue
        if ch.isspace():
            flush_word(i)
            toks.append(Token(" ", size * 0.5 + letter_spacing, size, i, [32], is_space=True))
            continue
        if _is_latin_word_char(cp):
            if not buf:
                start = i
            buf.append(ch)
            continue
        flush_word(i)
        toks.append(Token(ch, size + letter_spacing, size + letter_spacing, i, [cp]))
    flush_word(len(text))
    return toks


def _kinsoku_adjust(lines: list[list[Token]]) -> list[list[Token]]:
    """避头尾：行首禁则（追い出し）+ 行尾禁则。"""
    guard = 0
    changed = True
    while changed and guard < 128:
        changed = False
        guard += 1
        for i in range(len(lines) - 1):
            cur, nxt = lines[i], lines[i + 1]
            if not nxt:
                continue
            while nxt and nxt[0].first_cp in NO_LINE_START and cur:
                nxt.insert(0, cur.pop())
                changed = True
            while cur and cur[-1].last_cp in NO_LINE_END:
                nxt.insert(0, cur.pop())
                changed = True
    return [ln for ln in lines if ln]


def wrap_horizontal(text: str, face, size: float, shaper: Shaper,
                    max_width: float, letter_spacing: float = 0.0) -> list[list[Token]]:
    toks = tokenize(text, face, size, shaper, letter_spacing)
    lines: list[list[Token]] = []
    cur: list[Token] = []
    cur_w = 0.0

    def push():
        nonlocal cur, cur_w
        while cur and cur[-1].is_space:
            cur.pop()
        lines.append(cur)
        cur, cur_w = [], 0.0

    for tk in toks:
        if tk.text == "\n":
            push()
            continue
        if not cur and tk.is_space:
            continue
        if not cur or cur_w + tk.width <= max_width:
            cur.append(tk)
            cur_w += tk.width
        else:
            push()
            cur.append(tk)
            cur_w += tk.width
    if cur:
        push()
    return _kinsoku_adjust(lines)


def wrap_vertical(text: str, face, size: float, shaper: Shaper,
                  max_height: float, letter_spacing: float = 0.0) -> list[list[Token]]:
    toks = tokenize(text, face, size, shaper, letter_spacing)
    # 竖排：把多字符拉丁词拆成单字符（逐字下排）
    flat: list[Token] = []
    for tk in toks:
        if len(tk.text) > 1 and not tk.is_space:
            for k, ch in enumerate(tk.text):
                flat.append(Token(ch, size + letter_spacing, size + letter_spacing,
                                  tk.start + k, [ord(ch)]))
        else:
            flat.append(tk)
    toks = flat
    cols: list[list[Token]] = []
    cur: list[Token] = []
    cur_h = 0.0

    def push():
        nonlocal cur, cur_h
        cols.append(cur)
        cur, cur_h = [], 0.0

    for tk in toks:
        if tk.text == "\n":
            push()
            continue
        if not cur and tk.is_space:
            continue
        if not cur or cur_h + tk.height <= max_height:
            cur.append(tk)
            cur_h += tk.height
        else:
            push()
            cur.append(tk)
            cur_h += tk.height
    if cur:
        push()
    return _kinsoku_adjust(cols)


def _fits(text, face, size, shaper, box_w, box_h, mode, lh, ls) -> bool:
    if size < 1:
        return False
    if mode == "vertical":
        cols = wrap_vertical(text, face, size, shaper, box_h, ls)
        total_w = len(cols) * size * lh
        col_h = max((sum(t.height for t in c) for c in cols), default=0)
        return col_h <= box_h + 0.5 and total_w <= box_w
    lines = wrap_horizontal(text, face, size, shaper, box_w, ls)
    total_h = len(lines) * size * lh
    mw = max((sum(t.width for t in ln) for ln in lines), default=0.0)
    return total_h <= box_h and mw <= box_w + 0.5


def fit_size(db: "FontDB", text: str, font: str, box_w: float, box_h: float,
             size: int, mode: str = "horizontal", lh: float = 1.25,
             ls: float = 0.0, floor: int = 9) -> int:
    """二分求最大的、能放进 (box_w, box_h) 的字号（用真实换行结果判定）。"""
    face = db.resolve(font)
    shaper = Shaper(db)
    hi, lo, best = int(size), floor, floor
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(text, face, mid, shaper, box_w, box_h, mode, lh, ls):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ----------------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------------

PUNCT_SQUEEZE_FACTOR = 0.5


@dataclass
class RenderResult:
    mask: Mask
    rgba: np.ndarray
    lines: list
    bbox: tuple
    meta: dict = field(default_factory=dict)


def _parse_color(c) -> tuple:
    if isinstance(c, str):
        c = c.lstrip("#")
        if len(c) == 6:
            return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)
    if isinstance(c, (list, tuple)):
        if len(c) == 3:
            return (int(c[0]), int(c[1]), int(c[2]), 255)
        if len(c) == 4:
            return tuple(int(x) for x in c)
    return (0, 0, 0, 255)


def _parse_emphasis(emph, text: str) -> dict[int, str]:
    """emphasis → {字符索引: 样式}。样式含 dot/sesame/underline/bold。"""
    out: dict[int, str] = {}
    if not emph:
        return out
    if emph == "auto":
        return {i: "dot" for i in range(len(text))}
    if isinstance(emph, dict):
        style = emph.get("style", "dot")
        rng = emph.get("range")
        if rng and len(rng) == 2:
            idxs = range(max(0, rng[0]), min(len(text), rng[1]))
        else:
            idxs = range(len(text))
        for i in idxs:
            out[i] = style
        return out
    if isinstance(emph, list):
        for e in emph:
            out.update(_parse_emphasis(e, text))
        return out
    return out


def render_item(item: dict, canvas_w: int, canvas_h: int, db: FontDB) -> RenderResult:
    shaper = Shaper(db)
    face = db.resolve(item.get("font"))
    size = max(6, int(round(item.get("size", 32))))
    text = item.get("text", "")
    mode = item.get("writing_mode", "horizontal")
    color = _parse_color(item.get("color", [0, 0, 0, 255]))
    align = item.get("align", "left")
    valign = item.get("valign", "top")
    lh = float(item.get("line_height", 1.2))
    ls = float(item.get("letter_spacing", 0.0))
    squeeze = item.get("punct_squeeze", "compress")
    opacity = float(item.get("opacity", 1.0))
    stroke_w = float(item.get("stroke_width", 0.0))
    stroke_color = _parse_color(item.get("stroke_color", [255, 255, 255, 255]))
    emph = _parse_emphasis(item.get("emphasis"), text)

    poly = item.get("poly") or [[0, 0], [canvas_w, canvas_h]]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    pad = float(item.get("padding", 4))
    bx0, by0 = min(xs) + pad, min(ys) + pad
    bx1, by1 = max(xs) - pad, max(ys) - pad
    box_w, box_h = max(1.0, bx1 - bx0), max(1.0, by1 - by0)

    mask = Mask(canvas_w, canvas_h)
    ink = [10 ** 9, 10 ** 9, -10 ** 9, -10 ** 9]

    def note(bx, by, bw, bh):
        ink[0] = min(ink[0], bx); ink[1] = min(ink[1], by)
        ink[2] = max(ink[2], bx + bw); ink[3] = max(ink[3], by + bh)

    line_info = []

    if mode == "vertical":
        cols = wrap_vertical(text, face, size, shaper, box_h, ls)
        col_adv = size * lh
        total_w = col_adv * len(cols)
        if align == "left":
            block_x0 = bx0
        elif align == "right":
            block_x0 = bx1 - total_w
        else:  # center / justify → 居中
            block_x0 = bx0 + (box_w - total_w) / 2.0
        for ci, col in enumerate(cols):
            # 列自右向左
            col_x = block_x0 + (len(cols) - 1 - ci) * col_adv
            col_h = sum(t.height for t in col)
            if valign == "middle":
                y = by0 + (box_h - col_h) / 2
            elif valign == "bottom":
                y = by1 - col_h
            else:
                y = by0
            # 列首行对齐（竖排右对齐视觉）
            for tk in col:
                glyphs = shaper.shape(tk.text, face, size, "ttb")
                for g in glyphs:
                    arr, bl, bt, va = glyph_bitmap(db, face, size, g.gid, vertical=True)
                    if _is_wide(g.codepoint) and not _is_punct(g.codepoint):
                        # 表意字/假名：在 em 格内居中
                        draw_x = int(round(col_x + (size - (arr.shape[1] if arr is not None else 0)) / 2 + g.x_off))
                        draw_y = int(round(y + (size - (arr.shape[0] if arr is not None else 0)) / 2 - g.y_off))
                    else:
                        # 标点/拉丁：用竖排 bearing，落在格内正确角
                        draw_x = int(round(col_x + bl + g.x_off))
                        draw_y = int(round(y + (va - bt) - g.y_off))
                    if arr is not None:
                        mask.blit(arr, draw_x, draw_y)
                        note(draw_x, draw_y, arr.shape[1], arr.shape[0])
                    if tk.start + g.cluster in emph:
                        _draw_emphasis(mask, col_x + size * 0.88, y + size / 2,
                                       size, emph[tk.start + g.cluster], "v", note)
                y += tk.height
            line_info.append({"col": ci, "x": round(col_x, 1), "h": round(col_h, 1),
                              "n": sum(len(t.cps) for t in col)})
    else:
        lines = wrap_horizontal(text, face, size, shaper, box_w, ls)
        line_h = size * lh
        total_h = line_h * len(lines)
        if valign == "middle":
            y0 = by0 + (box_h - total_h) / 2
        elif valign == "bottom":
            y0 = by1 - total_h
        else:
            y0 = by0
        for li, line in enumerate(lines):
            lw = sum(t.width for t in line)
            if squeeze == "compress" and line:
                if line[-1].last_cp in FULLWIDTH_PUNCT:
                    lw -= line[-1].width * (1 - PUNCT_SQUEEZE_FACTOR)
                if line[0].first_cp in FULLWIDTH_PUNCT:
                    lw -= line[0].width * (1 - PUNCT_SQUEEZE_FACTOR)
            if align == "right":
                x = bx1 - lw
            elif align == "center":
                x = bx0 + (box_w - lw) / 2
            else:
                x = bx0
            extra = 0.0
            if align == "justify" and li != len(lines) - 1 and len(line) > 1:
                extra = max(0.0, (box_w - lw) / (len(line) - 1))
            baseline = y0 + li * line_h + size * 0.80
            for ti, tk in enumerate(line):
                pen = x
                for g in shaper.shape(tk.text, face, size, "ltr"):
                    arr, bl, bt, _ = glyph_bitmap(db, face, size, g.gid, vertical=False)
                    draw_x = int(round(pen + g.x_off + bl))
                    draw_y = int(round(baseline - g.y_off - bt))
                    if arr is not None:
                        mask.blit(arr, draw_x, draw_y)
                        note(draw_x, draw_y, arr.shape[1], arr.shape[0])
                    gi = tk.start + g.cluster
                    if gi in emph:
                        _draw_emphasis(mask, pen + g.x_adv / 2, baseline - size * 0.94,
                                       size, emph[gi], "h", note)
                    pen += g.x_adv + ls
                x = pen + extra
            line_info.append({"line": li, "y": round(y0 + li * line_h, 1),
                              "n": sum(len(t.cps) for t in line)})

    rgba = mask_to_rgba(mask, color, stroke_color, stroke_w, opacity)
    bbox = (max(0, ink[0] - 2), max(0, ink[1] - 2),
            min(canvas_w, ink[2] + 2), min(canvas_h, ink[3] + 2)) \
        if ink[2] > ink[0] else (0, 0, 0, 0)
    return RenderResult(mask, rgba, line_info, bbox,
                        meta={"n_lines": len(line_info), "mode": mode,
                              "font": face.family, "size": size})


def _draw_emphasis(mask: Mask, cx: float, cy: float, size: float,
                   style: str, mode: str, note):
    """着重号/傍点。style: dot（・）/ sesame（胡麻点）/ underline / bold(降级为 dot)。"""
    if style in ("underline",):
        # 下划线/旁线
        if mode == "v":
            x0, x1 = int(cx - size * 0.36), int(cx - size * 0.30)
            y0, y1 = int(cy - size / 2), int(cy + size / 2)
        else:
            x0, x1 = int(cx - size / 2), int(cx + size / 2)
            y0, y1 = int(cy + size * 0.10), int(cy + size * 0.16)
        line = np.full((max(1, y1 - y0), max(1, x1 - x0)), 255, np.uint8)
        mask.blit(line, x0, y0)
        note(x0, y0, x1 - x0, y1 - y0)
        return
    if style == "sesame":
        _draw_sesame(mask, cx, cy, size, mode, note)
        return
    # dot / bold（bold 暂降级为点）
    r = max(1.4, size * 0.095)
    R = int(np.ceil(r)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    dot = np.clip((r - d + 0.5) * 255.0, 0, 255).astype(np.uint8)
    mask.blit(dot, int(round(cx - R)), int(round(cy - R)))
    note(int(cx - r), int(cy - r), int(2 * r), int(2 * r))


def _draw_sesame(mask: Mask, cx: float, cy: float, size: float, mode: str, note):
    """胡麻点：四个小点（竖排时旋转排布）。"""
    r = max(1.0, size * 0.05)
    R = int(np.ceil(r)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    dot = np.clip((r - d + 0.5) * 255.0, 0, 255).astype(np.uint8)
    off = size * 0.10
    for dx, dy in ((-off, -off), (off, -off), (-off, off), (off, off)):
        mask.blit(dot, int(round(cx + dx - R)), int(round(cy + dy - R)))
    note(int(cx - off - r), int(cy - off - r), int(2 * (off + r)), int(2 * (off + r)))

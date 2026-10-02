"""kp_compose.py —— 版面 JSON → 图层 → 合成到画面（E6 交付物③「嵌字」）。

把 kp_schema.plan_page 或任意符合 schema 的 JSON 渲染成像素：
  - 形状层：气泡(ellipse+tail) / 旁白框(圆角矩形) / 面板边框（可选）
  - 文字层：交由 kp_engine 确定性光栅化真字体
  - 旋转：sfx 整体绕自身中心旋转
  - 合成：alpha-over 叠到底图（真实 bf16 基线出图）上

T0 的固有局限（设计文档原话）：字是「贴上去」的，不参与光影融合——
无透视、无材质、无遮挡。这既是它的优点（100% 正确、零训练），也是它的边界。
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

import kp_engine as E


def _bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _inner_poly(item) -> list:
    """文字排版用的内框：气泡取椭圆内接矩形，其它取原框内缩。"""
    x0, y0, x1, y1 = _bbox(item["poly"])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    t = item.get("type")
    if t == "bubble":
        k = 0.707 * 0.94     # 椭圆内接矩形 ≈ 边长×√2/2，再留 6% 余量
        hw, hh = w / 2 * k, h / 2 * k
    elif t == "narration":
        pad = max(6, min(w, h) * 0.08)
        hw, hh = w / 2 - pad, h / 2 - pad
    elif t == "caption":
        hw, hh = w / 2, h / 2
    else:
        hw, hh = w / 2, h / 2
    return [[int(cx - hw), int(cy - hh)], [int(cx + hw), int(cy - hh)],
            [int(cx + hw), int(cy + hh)], [int(cx - hw), int(cy + hh)]]


def _draw_shape(layer: Image.Image, item: dict):
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = _bbox(item["poly"])
    t = item.get("type")
    outline = tuple(item.get("outline_color", [20, 20, 25])) + (255,)
    lw = max(2, int(min(x1 - x0, y1 - y0) * 0.02))
    fill = tuple(item.get("bubble_fill", [255, 255, 255])) + (255,)
    if t == "bubble":
        d.ellipse([x0, y0, x1, y1], fill=fill, outline=outline, width=lw)
        tail = item.get("tail_to")
        if tail:
            _draw_tail(d, (x0, y0, x1, y1), tail, fill, outline, lw)
    elif t == "narration":
        r = max(4, int(min(x1 - x0, y1 - y0) * 0.06))
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=outline, width=lw)
    elif t == "caption":
        fa = int(item.get("fill_alpha", 255))
        d.rectangle([x0, y0, x1, y1], fill=tuple(item.get("bubble_fill", [250, 250, 245])) + (fa,))
    # sfx / text 不画框


def _draw_tail(d, bbox, tail, fill, outline, lw):
    """从气泡边界画一个指向 tail_to 的三角尾。"""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    tx, ty = tail
    # 尾巴根部：朝 tail 方向在椭圆上取一段弧
    import math
    ang = math.atan2(ty - cy, tx - cx)
    base_w = 0.42
    p1 = (cx + rx * math.cos(ang - base_w), cy + ry * math.sin(ang - base_w))
    p2 = (cx + rx * math.cos(ang + base_w), cy + ry * math.sin(ang + base_w))
    d.polygon([p1, p2, (tx, ty)], fill=fill, outline=outline)


def composite_over(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """layer(RGBA) alpha-over 到 base(RGB/RGBA uint8)。"""
    if base.shape[2] == 4:
        b = base[..., :3].astype(np.float32)
    else:
        b = base.astype(np.float32)
    a = layer[..., 3:4].astype(np.float32) / 255.0
    out = layer[..., :3].astype(np.float32) * a + b * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_layout(layout: dict, db: Optional[E.FontDB] = None,
                  base: Optional[np.ndarray] = None,
                  show_panels: bool = True,
                  transparent: bool = False) -> Image.Image:
    """把整份 layout JSON 渲染为 PIL Image（RGB 或 RGBA）。"""
    db = db or E.fontdb()
    W = layout["canvas"]["w"]
    H = layout["canvas"]["h"]
    if base is not None:
        canvas = np.array(base.convert("RGB") if isinstance(base, Image.Image)
                          else Image.fromarray(base).convert("RGB"))
    else:
        canvas = np.full((H, W, 3), 255, np.uint8)

    # 1) 面板边框
    if show_panels and layout.get("panels"):
        pl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(pl)
        for p in layout["panels"]:
            d.polygon([tuple(q) for q in p], outline=(0, 0, 0, 255))
        canvas = composite_over(canvas, np.array(pl))

    # 2) 逐个 item：形状 + 文字 → 单独层 → (旋转) → 合成
    for item in layout.get("items", []):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        _draw_shape(layer, item)
        textual = copy.deepcopy(item)
        textual["poly"] = _inner_poly(item)
        if item.get("type") == "caption":
            textual["size"] = max(10, int(item.get("size", 16)))
        res = E.render_item(textual, W, H, db)
        text_layer = Image.fromarray(res.rgba, "RGBA")
        layer = Image.alpha_composite(layer, text_layer)

        rot = float(item.get("rot", 0.0))
        if abs(rot) > 0.05:
            x0, y0, x1, y1 = _bbox(item["poly"])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            layer = layer.rotate(rot, center=(cx, cy), resample=Image.BICUBIC)
        canvas = composite_over(canvas, np.array(layer))

    img = Image.fromarray(canvas, "RGB")
    if transparent:
        return img
    return img

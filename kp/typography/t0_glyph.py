"""T0 · 像素域真字形合成 —— 把文字链路从「形状能对上」推到「字真的对」

═══ 为什么需要这个文件（缺口很具体）═══
`composite.py` 的 ROI 内容来自 `roi_patches_from_tokens`：
    「`dim → channels` 用的是**固定种子的高斯投影占位**（未训练）」
⇒ 链路能跑通、形状能对上，但**画出来的是噪声，不是字**。
它的 docstring 说得很清楚：字形正确性「由 T0 字形 provider 负责」—— **而 T0 当时不存在。**

═══ T0 做什么 ═══
在**像素域**用真实字库渲染汉字 → 缩到 ROI 大小 → 变成可喂给 ROI 分支的
`(B,C,side,side)` 张量。

⭐ **为什么不直接在 latent 域画字**：32×压缩会把 40px 汉字抹成 1.25 个格
（这正是项目要开独立低压缩分支的原因）。而 ROI 分支是**低压缩**的，
所以在 ROI 自己的**像素域**画字、再降采样到 ROI 尺寸是正确的层级。

⚠️ **诚实边界**：
    ① 需要系统有中文字体；找不到时**明确报错**并列出查找路径（不静默画方块）。
    ② 这是**真字形**，但 T0 只保证「字形被正确渲染进 latent」，
       **不保证主干能据此生成出可读的中文**（那取决于 P2 主干是否读了 ROI）。
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import LATENT

#: 常见中文字体的查找位置（Windows / Linux / macOS）
FONT_CANDIDATES: Tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",       # 黑体
    "C:/Windows/Fonts/simsun.ttc",       # 宋体
    "C:/Windows/Fonts/Deng.ttf",         # 等线
    "C:/Windows/Fonts/NotoSansCJK*.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    # 项目内自带（若放了）
    str(LATENT and ""),                 # 占位，下面会补项目内路径
)

FONT_GLOBS: Tuple[str, ...] = (
    "C:/Windows/Fonts/*.ttc",
    "C:/Windows/Fonts/*.ttf",
    "data/fonts/*.ttf",
    "data/fonts/*.ttc",
)


class FontNotFound(RuntimeError):
    """⛔ 找不到中文字体。**明确报错**，不静默画方块（那会产出"看似有字实则乱码"）。"""


def find_font(explicit: Optional[str] = None) -> str:
    """定位一个**能渲染中文**的字体文件；找不到就抛 `FontNotFound`。"""
    tried: List[str] = []
    if explicit:
        if os.path.exists(explicit):
            return explicit
        tried.append(explicit)
    for p in FONT_CANDIDATES:
        if p and p != str(LATENT and "") and os.path.exists(p):
            return p
    for pat in FONT_GLOBS:
        for p in sorted(glob.glob(pat)):
            tried.append(p)
            # ⭐ 验「能不能渲染汉字」：字体可能只有拉丁字形
            if _can_render_cjk(p):
                return p
    raise FontNotFound(
        "找不到可渲染中文的字体。\n"
        f"已尝试：{tried[:6]}{'…' if len(tried) > 6 else ''}\n"
        "⇒ 放一个 .ttf/.ttc 到 data/fonts/，或用 --font 指定路径。\n"
        "⛔ **不静默用方块代替**（那会产出「看着有字、实则乱码」的图）。")


def _can_render_cjk(font_path: str, ch: str = "心") -> bool:
    """字体真能画出这个汉字吗（而不是画成豆腐块）。"""
    try:
        from PIL import ImageFont
        f = ImageFont.truetype(font_path, 32)
        # `getmask` 对缺字形会返回空或豆腐框；用 bbox 粗判
        m = f.getmask(ch)
        return bool(m.getbbox())
    except Exception:                                          # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
@dataclass
class GlyphBox:
    """一个字的像素域渲染结果。"""
    ch: str
    rgba: np.ndarray          # (H,W,4) uint8
    width: int
    height: int
    baseline: int


def render_glyphs(text: str, font_px: int = 48, font_path: Optional[str] = None,
                  canvas: Optional[Tuple[int, int]] = None,
                  color: Tuple[int, int, int, int] = (255, 255, 255, 255),
                  ) -> List[GlyphBox]:
    """逐字渲染 → `List[GlyphBox]`（每个是独立的小图）。

    ⭐ **逐字**而不是整行：这样能对上 `LayoutSpec` 里的**逐字 box**，
    也便于塞进 19 类语义层那套分层表示。
    """
    from PIL import Image, ImageDraw, ImageFont
    fp = find_font(font_path)
    font = ImageFont.truetype(fp, font_px)
    w, h = canvas or (font_px * max(1, len(text)) + font_px,
                      int(font_px * 1.6))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    out: List[GlyphBox] = []
    x = font_px // 4
    for ch in text:
        if ch == " ":
            out.append(GlyphBox(ch, np.zeros((h, font_px, 4), np.uint8), font_px, h, h - 1))
            x += font_px // 2
            continue
        dr.text((x, font_px // 6), ch, font=font, fill=color)
        bb = dr.textbbox((x, font_px // 6), ch, font=font)
        gw = max(1, bb[2] - x)
        gh = max(1, bb[3] - bb[1])
        sub = img.crop((x, bb[1], x + gw, bb[1] + gh))
        out.append(GlyphBox(ch, np.array(sub), gw, gh, bb[3]))
        x += gw + max(2, font_px // 8)
    return out


def glyphs_to_roi_tensor(boxes: Sequence[GlyphBox], side: int,
                         channels: int = LATENT.total_ch) -> torch.Tensor:
    """`List[GlyphBox]` → `(N,C,side,side)`，值域 [-1,1]（与 VAE 输入同域）。

    ⚠️ 灰度字形被**复制到所有通道**：因为项目是混合 latent（8 语义 + 32 细节），
    单通道字形无法表达"哪个语义通道该放什么" ⇒ 复制是最诚实的等权假设，
    **不假装做了语义分层**。
    """
    if not boxes:
        raise ValueError("没有字形可转换")
    n = len(boxes)
    out = torch.zeros(n, channels, side, side, dtype=torch.float32)
    for i, b in enumerate(boxes):
        g = torch.from_numpy(_fit(b.rgba, side, side))     # (side,side) float32
        for c in range(channels):
            out[i, c] = g
    return out.mul_(2.0).sub_(1.0)                     # → [-1,1]


def _fit(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """把字形缩放到 (h,w) 并取**alpha 通道**作掩码（字在哪）。"""
    from PIL import Image
    im = Image.fromarray(rgba, "RGBA").convert("RGBA")
    # ⭐ 等比缩放 + 居中：**不拉伸字形**（拉伸会让汉字变形）
    scale = min(w / max(1, im.width), h / max(1, im.height))
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    can = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    can.paste(im, ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(can)[..., 3].astype(np.float32) / 255.0


def glyph_coverage(boxes: Sequence[GlyphBox], side: int) -> np.ndarray:
    """合成整行 → `(side,side)` alpha 覆盖率（判据用：字真的落在纸上）。"""
    can = np.zeros((side, side), np.float32)
    for b in boxes:
        g = _fit(b.rgba, side, side)
        can = np.maximum(can, g)
    return can


# ---------------------------------------------------------------------------
def render_line(text: str, roi_pixels: int, font_path: Optional[str] = None,
                font_px: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
    """一步到位：文字 → `(1,C,side,side)` ROI 张量 + 统计（便于验收）。"""
    side = int(roi_pixels)
    fp_px = font_px or max(8, int(side * 0.8))
    boxes = render_glyphs(text, font_px=fp_px, font_path=font_path,
                          canvas=(side * max(1, len(text)), side * 2))
    t = glyphs_to_roi_tensor(boxes, side)
    cov = glyph_coverage(boxes, side)
    stats = {"text_len": len(text), "side": side, "n_boxes": len(boxes),
             "ink_ratio": round(float(cov.mean()), 4),
             "glyphs": [b.ch for b in boxes],
             "font": find_font(font_path) if font_path else "auto"}
    return t.unsqueeze(0), stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="T0 · 像素域真字形合成")
    ap.add_argument("--text", default="心夏北极星")
    ap.add_argument("--size", type=int, default=64, help="ROI 边长（像素域）")
    ap.add_argument("--font", default=None)
    ap.add_argument("--out", default="out/typography/t0")
    a = ap.parse_args(argv)

    try:
        t, stats = render_line(a.text, a.size, font_path=a.font)
    except FontNotFound as e:
        print(f"⛔ {e}")
        return 1
    os.makedirs(a.out, exist_ok=True)
    # ⭐ t 是 **逐字** 的：(1, n_chars, C, side, side) ⇒ 逐字存，肉眼可核字形
    n = t.shape[1]
    for i, ch in enumerate(stats["glyphs"]):
        cov = t[0, i, 0].numpy()                     # (side, side) 单通道
        img = ((cov + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        safe = "space" if ch == " " else f"u{ord(ch):04x}"
        Image.fromarray(img).save(os.path.join(a.out, f"t0_{a.size}_{i}_{safe}.png"))
    # ⭐ 判据：墨水占比太低 = 字没画出来（不是"看起来像字"就算）
    cov_all = (t[0, :, 0].numpy() > 0).mean()
    print(f"✅ 字体：{stats['font']}")
    print(f"   文字：{a.text!r}（{n} 字）")
    print(f"   ROI 张量 {tuple(t.shape)} = (1, 每字1块, {t.shape[2]}ch, {a.size}²)")
    print(f"   墨水占比 {cov_all:.2%}")
    print(f"   → {a.out}/t0_{a.size}_*.png（逐字，可肉眼核）")
    if cov_all < 0.02:
        print("⛔ 墨水占比过低 ⇒ 字没画出来（检查字体/字号）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

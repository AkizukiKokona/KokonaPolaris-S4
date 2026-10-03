"""P1 · 训练数据扩增器 —— 用**四旋钮**从现有真图造出够训 VAE 的样本。

⭐ **为什么需要它**（实测驱动的）：
    P1 卡在**数据量**上 —— 全仓只有 **11 张**真图（`data/characters/kokona`），
    而 11 张连「过拟合都不稳」。而 VAE 只有 **4.4M 参数**（很小），
    理论上 11 张不该完全训不动，但**多样性**严重不足（1 个角色、2 个视图）。

═══ 四旋钮（来自《补充07》§数据供给）═══

    ① **视角**    —— 缩放 + 平移（`_augment` 已有：中心窗裁切再缩放回）
    ② **着色器**  —— 背景色 / 亮度 / 对比度 / 饱和度
    ③ **姿态**    —— 水平翻转 + 90° 旋转（⚠️ 角色图慎用旋转，见下）
    ④ **分层可见性** —— 半透明遮罩（模拟「被前景挡住一部分」）

⚠️ **三条必须守住的纪律**（否则会污染训练分布）：

① **⛔ 绝不用「合成图」冒充真图验证**。
    本模块产出的是**增广后的真图**（同一张图的不同取景/配色），
    **不是**新内容。它扩大的是**不变性**，不是**多样性**。
    ⇒ 用它训 VAE 可以，但**报告时必须说清「数据是增广来的」**。

② **🔴 旋转慎用**：角色立绘的**上下方向有语义**（头在上面）⇒ 90°/180° 旋转
    会产出「倒立的人」，污染姿态先验。
    ⇒ **默认只开水平翻转**（左右翻转对角色图是安全的），旋转必须显式开且默认关。

③ **⛔ 不做「无意义」的增广**：不翻转色相（会改变角色识别）、不加随机噪声
    （那是 VAE 该自己学的，不是数据该注入的）。

═══ 用法 ═══
    # 只扩增 kokona 那 11 张（示例：16 倍）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.augment \
        --image-dir data/characters/kokona --out out/aug/kokona_x16 --per-image 16

    # 看一眼增广统计（不写文件）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.data.augment \
        --image-dir data/characters/kokona --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------------------
# 旋钮 ①：视角（缩放 + 平移）
# ---------------------------------------------------------------------------
def knob_view(img: torch.Tensor, *, zoom: float = 1.0, dx: float = 0.0,
              dy: float = 0.0) -> torch.Tensor:
    """中心窗裁切（窗 = side/zoom）→ 缩放回。`zoom>1` 拉近、`dx/dy∈[-1,1]` 平移。"""
    s = img.shape[-1]
    cw = max(8, min(s, int(round(s / max(zoom, 1e-3)))))
    x0 = max(0, min(s - cw, int(round((s - cw) * (0.5 + dx)))))
    y0 = max(0, min(s - cw, int(round((s - cw) * (0.5 + dy)))))
    crop = img[:, y0:y0 + cw, x0:x0 + cw].unsqueeze(0)
    return torch.nn.functional.interpolate(crop, size=(s, s), mode="bilinear",
                                           align_corners=False)[0]


# ---------------------------------------------------------------------------
# 旋钮 ②：着色器（背景 / 亮度 / 对比 / 饱和）
# ---------------------------------------------------------------------------
def knob_shade(img: torch.Tensor, *, brightness: float = 1.0, contrast: float = 1.0,
               saturation: float = 1.0, bg_tint: Optional[float] = None) -> torch.Tensor:
    """亮度/对比/饱和 + 可选背景染色（`img` 已是 `[-1,1]` 的合成图）。

    ⚠️ 顺序固定为「先对比 → 后饱和 → 最后背景」：背景染色放最后，
    保证「只染背景、不染本体」的语义成立。
    """
    out = img
    if contrast != 1.0:
        m = out.mean()
        out = (out - m) * contrast + m
    if brightness != 1.0:
        out = out * brightness
    if saturation != 1.0:
        g = out.mean(0, keepdim=True)
        out = (out - g) * saturation + g
    if bg_tint is not None:
        # 背景染色：只影响「接近背景色」的那些像素（用亮度接近度做软掩码）
        lum = out.mean(0, keepdim=True)
        near_bg = (lum - bg_tint).abs() < 0.25
        out = torch.where(near_bg, torch.full_like(out, bg_tint), out)
    return out.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 旋钮 ③：姿态（⚠️ 默认只开水平翻转）
# ---------------------------------------------------------------------------
def knob_pose(img: torch.Tensor, *, hflip: bool = False,
              rot90: bool = False) -> torch.Tensor:
    """水平翻转 / 90° 旋转。

    ⚠️⚠️ **`rot90` 默认关、且强烈不建议对角色立绘开启** ——
    角色图的「上」有语义（头在上面）⇒ 旋转会造出「倒立的人」，
    污染姿态先验 ⇒ VAE 学到的「结构」会包含错误的朝向。
    ⇒ 这里**只提供能力、不做默认**；要开请显式传 `rot90=True` 并自行承担后果。
    """
    out = img
    if hflip:
        out = torch.flip(out, dims=[-1])
    if rot90:
        out = torch.rot90(out, 1, dims=(-2, -1))
    return out


# ---------------------------------------------------------------------------
# 旋钮 ④：分层可见性（半透明遮罩）
# ---------------------------------------------------------------------------
def knob_occlusion(img: torch.Tensor, *, ratio: float = 0.0,
                   seed: int = 0) -> torch.Tensor:
    """随机矩形遮挡（模拟「被前景挡住一部分」）。

    ⚠️ **默认关（`ratio=0`）**：遮挡会**移除信息**，训 VAE 时会逼它去猜
    ⇒ 只在「想让模型学会遮挡下的补全」时才开。P1 阶段**建议保持 0**。
    """
    if ratio <= 0:
        return img
    g = torch.Generator().manual_seed(seed)
    s = img.shape[-1]
    h = max(1, int(s * ratio))
    y = int(torch.randint(0, max(1, s - h), (1,), generator=g))
    x = int(torch.randint(0, max(1, s - h), (1,), generator=g))
    out = img.clone()
    out[:, y:y + h, x:x + h] = 0.0            # 置成中灰（0 in [-1,1]）
    return out


# ---------------------------------------------------------------------------
# 组合：一张图 → N 个增广样本
# ---------------------------------------------------------------------------
def augment_one(img: torch.Tensor, idx: int, *, size: int = 128,
                allow_rot90: bool = False, occlusion: float = 0.0) -> torch.Tensor:
    """用**确定性**参数（以 `idx` 为种子）把一张图变成一个增广样本。

    ⭐ **确定性很重要**：同一 `(图, idx)` 永远得到同一结果 ⇒ 可复现、可对照。
    """
    g = torch.Generator().manual_seed(hash((idx, img.shape[-1])) & 0xFFFFFFFF)
    r = lambda a, b: float(torch.rand(1, generator=g)) * (b - a) + a  # noqa: E731

    out = img
    # ① 视角：交替「拉近 + 平移」与「原图」，保证有一半样本是未动过的
    if idx % 2 == 1:
        out = knob_view(out, zoom=r(1.0, 1.25), dx=r(-0.35, 0.35), dy=r(-0.2, 0.2))
    # ③ 姿态：水平翻转（安全）；旋转只在显式允许时
    if idx % 4 >= 2:
        out = knob_pose(out, hflip=True, rot90=allow_rot90 and idx % 8 == 7)
    # ② 着色器：温和的亮度/对比/饱和扰动（⚠️ 不做色相旋转）
    out = knob_shade(out, brightness=r(0.88, 1.12), contrast=r(0.9, 1.1),
                     saturation=r(0.92, 1.08))
    # ④ 分层可见性（默认 0 = 关）
    if occlusion > 0 and idx % 5 == 4:
        out = knob_occlusion(out, ratio=occlusion, seed=idx)
    return out.clamp(-1.0, 1.0)


def list_images(dirs: Sequence[os.PathLike | str]) -> List[Path]:
    """递归收集图片路径。⛔ 只读，不移动不删除既有文件。"""
    out: List[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                out.append(f)
    return out


def build_dataset(image_dirs, *, per_image: int = 16, size: int = 128,
                  allow_rot90: bool = False, occlusion: float = 0.0,
                  out_dir: Optional[str] = None, dry_run: bool = False) -> dict:
    from ..character.dataset import load_image

    paths = list_images(image_dirs)
    if not paths:
        raise FileNotFoundError(f"没找到图片（扫了 {list(image_dirs)}）")
    total = len(paths) * per_image
    report = {"n_source": len(paths), "per_image": per_image, "total": total,
              "size": size, "allow_rot90": allow_rot90, "occlusion": occlusion,
              "⚠️_nature": "**这是增广数据，不是新内容** —— 扩大的是不变性不是多样性",
              "⚠️_warnings": []}
    if allow_rot90:
        report["⚠️_warnings"].append(
            "**已开启 90° 旋转** —— 角色立绘的「上」有语义，旋转会造出倒立的人，污染姿态先验")
    if occlusion > 0:
        report["⚠️_warnings"].append(
            f"**已开启遮挡（ratio={occlusion}）** —— 会移除信息，P1 阶段建议保持 0")
    if dry_run:
        return report

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for p in paths:
            base = load_image(str(p), size)
            for i in range(per_image):
                aug = augment_one(base, i, size=size, allow_rot90=allow_rot90,
                                  occlusion=occlusion)
                _save_png(aug, d / f"{p.stem}__a{i:03d}.png")
                n_written += 1
        report["n_written"] = n_written
        report["out_dir"] = str(d)
    return report


def _save_png(t: torch.Tensor, path: Path) -> None:
    """把 `[-1,1]` 的 (3,S,S) 存成 PNG（⛔ 不改动任何既有文件，只新建）。"""
    from PIL import Image
    import numpy as np
    a = ((t.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    img = Image.fromarray(a.permute(1, 2, 0).numpy(), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1 · 四旋钮数据扩增")
    ap.add_argument("--image-dir", action="append", default=[])
    ap.add_argument("--per-image", type=int, default=16, help="每张图扩几个")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default=None, help="输出目录（不给则只报统计）")
    ap.add_argument("--dry-run", action="store_true", help="只报统计，不写文件")
    ap.add_argument("--allow-rot90", action="store_true",
                    help="⚠️ 开启 90° 旋转（角色立绘**不建议**）")
    ap.add_argument("--occlusion", type=float, default=0.0,
                    help="遮挡比例（默认 0 = 关；P1 阶段建议 0）")
    a = ap.parse_args(argv)
    if not a.image_dir:
        print("用法：--image-dir <目录>（可多次）")
        return 2
    rep = build_dataset(a.image_dir, per_image=a.per_image, size=a.size,
                        allow_rot90=a.allow_rot90, occlusion=a.occlusion,
                        out_dir=a.out, dry_run=a.dry_run)
    print("=" * 64)
    print("P1 · 四旋钮数据扩增")
    print("=" * 64)
    for k, v in rep.items():
        if k.startswith("⚠️"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    if a.out and not a.dry_run:
        print(f"\n✅ 已写出 {rep.get('n_written')} 张到 {rep['out_dir']}")
        print("⭐ 接着训：")
        print(f"   .venv\\Scripts\\python.exe -m kp.train.vae_pretrain --image-dir {a.out} "
              f"--size {a.size} --steps 2000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

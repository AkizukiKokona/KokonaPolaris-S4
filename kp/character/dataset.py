"""多视角配对数据加载器 —— 把「**声明的**配对」变成 Character Fitter 的训练批次。

⭐ 配对来源**唯一** = `kp.character.pairing.build_pairs`（声明式）。
   本模块**不做任何推断**：没有声明 `view` 列 → 直接报缺口，绝不用文件名去凑。
   （理由见 `pairing.py` 顶部：猜出来的错误配对会**静默**污染 Fitter 训练目标。）

两条数据路径
------------
1. `PairViewLoader.from_batch()` —— 真实批次
     读 `data/characters/<batch>/manifest.csv`；图源优先取**管线归一化后**的
     `out/characters/<batch>/norm/`（单一投影尺度），退回 `images/`。
2. `PairViewLoader.synthetic()` —— 合成数据（供自检 / 冒烟）
     给训练机械一个**已知的可控信号**，使「学没学到」可证伪。

合成数据为什么这样构造
----------------------
    身份 = 低频随机模板（低分辨率上采样），**与视角无关**；
    视角 = 模板的**循环平移**（roll）+ 轻微噪声，**不改变身份**。

⇒ 理想 Fitter 应当：对「平移」不变（invariance）、对「模板」可区分（separation）。
   这正是身份 token 必须满足的两条性质。

⚠️ 刻意做的一步：模板**先减去逐通道均值**再使用。否则「全局平均池化」这种平凡解
   就能同时拿到不变性与可分性，测试会退化成「什么也没验证」。
   去均值后身份信息只活在**空间图案**里，必须真的学出平移不变的空间描述子。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# 训练时统一到方图（本体已在管线里归一化，这里只做尺寸/数值统一）
DEFAULT_SIZE = 256
DEFAULT_SPEC_VARY: Tuple[str, ...] = ("view",)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ViewSample:
    """一张已加载的视图。"""
    key: str                  # 文件名（配对键）
    identity: str
    view: Optional[str]       # 已声明的视角取值；None = 未声明
    attrs: Dict[str, str] = field(default_factory=dict)
    image: Optional[torch.Tensor] = None    # (3, H, W) float32，已归一化到 [-1, 1]


@dataclass
class FitPair:
    """一条「同身份 + 只动 vary 轴」的训练对。"""
    identity: str
    anchor_key: str
    positive_key: str
    varied: Dict[str, Tuple[str, str]]      # 轴 → (anchor 取值, positive 取值)
    anchor: torch.Tensor                    # (3, H, W)
    positive: torch.Tensor                  # (3, H, W)

    @property
    def varied_axis(self) -> str:
        return next(iter(self.varied)) if self.varied else ""

    def describe(self) -> str:
        v = " ".join(f"{k}:{a}→{b}" for k, (a, b) in self.varied.items())
        return f"{self.anchor_key} ↔ {self.positive_key} [{v}]"


# ---------------------------------------------------------------------------
# 图像 IO
# ---------------------------------------------------------------------------
def _require_pil():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("需要 Pillow：pip install Pillow") from e
    return __import__("PIL.Image", fromlist=["Image"])


def load_image(path: str, size: int = DEFAULT_SIZE) -> torch.Tensor:
    """读 RGBA 图 → (3, size, size) float32 ∈ [-1, 1]。

    · 用 **alpha 预乘**：去底后的透明区不参与身份（否则底色会漏进身份信号）。
    · 分辨率口径见设计稿：看「角色本体占多大」，归一化已在管线里做过。
    """
    Image = _require_pil()
    im = Image.open(path).convert("RGBA").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3:4]
    rgb = rgb * alpha                                   # 预乘：透明处 → 0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous() * 2.0 - 1.0


# ---------------------------------------------------------------------------
# 加载器
# ---------------------------------------------------------------------------
class PairViewLoader:
    """把一批图 + 声明式配对，装成 Fitter 的训练对。"""

    def __init__(self, pairs: List[FitPair], *, spec=None,
                 report=None, gaps: Optional[List[str]] = None,
                 samples: Optional[List[ViewSample]] = None,
                 source: str = ""):
        self.pairs = pairs
        self.spec = spec
        self.report = report                    # pairing.PairReport（真实批次才有）
        self.gaps = list(gaps or [])
        self.samples = list(samples or [])
        self.source = source

    # ---------------- 真实批次 ----------------
    @classmethod
    def from_batch(cls, batch: str = "kokona", *, data_root: str = "data/characters",
                   norm_root: str = "out/characters", size: int = DEFAULT_SIZE,
                   spec=None, target_views: Optional[Sequence[str]] = None,
                   identity_col: str = "character") -> "PairViewLoader":
        """读 manifest → 声明式配对 → 加载图。**未声明的轴一律不配对。**

        `norm_root` 若存在 `<norm_root>/<batch>/norm/` 则优先用它（归一化后的图）。
        """
        from .pairing import AXES, PairSpec, build_pairs, load_records

        batch_dir = os.path.join(data_root, batch)
        csv_path = os.path.join(batch_dir, "manifest.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"缺少 manifest.csv：{csv_path}")

        norm_dir = os.path.join(norm_root, batch, "norm")
        img_dir = os.path.join(batch_dir, "images")

        def _resolve(key: str) -> str:
            for d in (norm_dir, img_dir):
                p = os.path.join(d, key)
                if os.path.isfile(p):
                    return p
            raise FileNotFoundError(
                f"找不到 {key}（已试：{norm_dir}、{img_dir}）")

        records, names = load_records(csv_path, identity_col=identity_col, axes=AXES)
        spec = spec or PairSpec(vary=DEFAULT_SPEC_VARY, match=(), identity="character")
        tv = {"view": tuple(target_views)} if target_views else None
        rep = build_pairs(records, spec, target_values=tv, name_of=names)

        # 先把所有样本载成 ViewSample（顺带把"图读不出来"这种问题暴露在配对之前）
        by_key = {r.key: r for r in records}
        samples: List[ViewSample] = []
        img_cache: Dict[str, torch.Tensor] = {}
        for r in records:
            path = _resolve(r.key)
            img_cache[r.key] = load_image(path, size)
            samples.append(ViewSample(key=r.key, identity=r.identity,
                                      view=r.get("view"), attrs=dict(r.attrs),
                                      image=img_cache[r.key]))

        pairs: List[FitPair] = []
        for p in rep.pairs:
            # 优先取 vary 轴首轴的取值做描述
            varied = dict(p.varied)
            pairs.append(FitPair(identity=p.identity, anchor_key=p.anchor,
                                 positive_key=p.positive, varied=varied,
                                 anchor=img_cache[p.anchor],
                                 positive=img_cache[p.positive]))

        loader = cls(pairs, spec=spec, report=rep, gaps=list(rep.gaps),
                     samples=samples, source=f"batch:{batch}")
        if not pairs:
            loader.gaps.append(
                f"批次「{batch}」没有产出任何配对（默认需同身份 ≥2 视图且 view 已声明）。"
                f"⚠️ 不要用文件名去猜视角 —— 让渲染器/生成脚本把 view 写进 manifest。")
        return loader

    # ---------------- 合成数据 ----------------
    @staticmethod
    def _template(g: torch.Generator, low: int, size: int) -> torch.Tensor:
        tpl = torch.randn(3, low, low, generator=g)
        tpl = tpl - tpl.mean(dim=(-2, -1), keepdim=True)      # 去通道均值（排除平凡解）
        tpl = F.interpolate(tpl.unsqueeze(0), size=(size, size), mode="nearest")[0]
        return tpl / (tpl.std() + 1e-6)

    @classmethod
    def synthetic(cls, n_identities: int = 8, n_views: int = 3, *,
                  size: int = 48, low: int = 6, noise: float = 0.05,
                  shift_frac: float = 0.18, seed: int = 0,
                  names: Optional[Sequence[str]] = None) -> "PairViewLoader":
        """可控合成信号：身份=低频零均值模板，视角=循环平移。"""
        g = torch.Generator().manual_seed(seed)
        names = list(names) if names else [f"char{i:02d}" for i in range(n_identities)]
        n_identities = len(names)

        samples: List[ViewSample] = []
        pairs: List[FitPair] = []
        cache: Dict[str, torch.Tensor] = {}
        for i, ident in enumerate(names):
            tpl = cls._template(g, low, size)
            keys: List[str] = []
            for v in range(n_views):
                dy = int(round((v - (n_views - 1) / 2.0) * size * shift_frac))
                img = torch.roll(tpl, shifts=dy, dims=-2)
                img = img + noise * torch.randn(3, size, size, generator=g)
                key = f"{ident}_v{v}"
                cache[key] = img
                keys.append(key)
                samples.append(ViewSample(key=key, identity=ident, view=f"v{v}",
                                          attrs={"view": f"v{v}", "character": ident},
                                          image=img))
            for a in range(n_views):
                for b in range(a + 1, n_views):
                    pairs.append(FitPair(identity=ident, anchor_key=keys[a],
                                         positive_key=keys[b],
                                         varied={"view": (f"v{a}", f"v{b}")},
                                         anchor=cache[keys[a]], positive=cache[keys[b]]))
        return cls(pairs, spec=None, report=None, gaps=[], samples=samples,
                   source=f"synthetic:{n_identities}x{n_views}@{size}")

    @classmethod
    def synthetic_split(cls, n_identities: int = 8, n_train_views: int = 3,
                        n_holdout: int = 1, *, size: int = 48, low: int = 6,
                        noise: float = 0.05, shift_frac: float = 0.18, seed: int = 0,
                        names: Optional[Sequence[str]] = None):
        """合成数据 + **留出视角**：返回 `(train_loader, holdout, refs)`。

        `holdout` : {身份: [留出视角的图]} —— **完全不参与训练**
        `refs`    : {身份: 参考视角图（v0）}，用于量测「留出视角 ↔ 参考视角」的一致性

        ⭐ 为什么必须留出：注意力池化对 token 的**重排**本就不变，而合成视角变换近似一次
           重排 ⇒ 不变性有「平凡通过」成分；再加上只有二十几条训练对，模型可以**背下**
           训练样本把损失压到 0。**只看训练视角分不出「真学到不变」与「背下来」**。
           留出视角正是那个分辨器：记忆救不了没见过的输入。
        """
        g = torch.Generator().manual_seed(seed)
        names = list(names) if names else [f"char{i:02d}" for i in range(n_identities)]
        n_identities = len(names)
        n_total = n_train_views + n_holdout

        samples: List[ViewSample] = []
        pairs: List[FitPair] = []
        cache: Dict[str, torch.Tensor] = {}
        holdout: Dict[str, List[torch.Tensor]] = {}
        refs: Dict[str, torch.Tensor] = {}
        for ident in names:
            tpl = cls._template(g, low, size)
            keys: List[str] = []
            for v in range(n_total):
                dy = int(round((v - (n_total - 1) / 2.0) * size * shift_frac))
                img = torch.roll(tpl, shifts=dy, dims=-2)
                img = img + noise * torch.randn(3, size, size, generator=g)
                key = f"{ident}_v{v}"
                cache[key] = img
                keys.append(key)
                if v < n_train_views:
                    samples.append(ViewSample(key=key, identity=ident, view=f"v{v}",
                                              attrs={"view": f"v{v}", "character": ident},
                                              image=img))
            refs[ident] = cache[keys[0]]
            holdout[ident] = [cache[keys[v]] for v in range(n_train_views, n_total)]
            for a in range(n_train_views):
                for b in range(a + 1, n_train_views):
                    pairs.append(FitPair(identity=ident, anchor_key=keys[a],
                                         positive_key=keys[b],
                                         varied={"view": (f"v{a}", f"v{b}")},
                                         anchor=cache[keys[a]], positive=cache[keys[b]]))
        loader = cls(pairs, spec=None, report=None, gaps=[], samples=samples,
                     source=f"synthetic_split:{n_identities}x{n_train_views}+{n_holdout}@{size}")
        return loader, holdout, refs

    # ---------------- 视图 ----------------
    def identities(self) -> List[str]:
        out: List[str] = []
        for p in self.pairs:
            if p.identity not in out:
                out.append(p.identity)
        return out

    def group_by_identity(self) -> Dict[str, List[FitPair]]:
        d: Dict[str, List[FitPair]] = {}
        for p in self.pairs:
            d.setdefault(p.identity, []).append(p)
        return d

    def to_tensors(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """整批堆成张量（配对粒度，B = 配对数）。"""
        A = torch.stack([p.anchor for p in self.pairs])
        P = torch.stack([p.positive for p in self.pairs])
        if device:
            A, P = A.to(device), P.to(device)
        return {"anchor": A, "positive": P}

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def format_dataset_report(loader: PairViewLoader) -> str:
    L = []
    L.append("=" * 74)
    L.append("  多视角配对数据加载器 · Character Fitter 训练输入")
    L.append("=" * 74)
    L.append(f"  来源：{loader.source or '—'}")
    L.append(f"  样本 {len(loader.samples)} 张 → 配对 **{loader.n_pairs}** 对"
             f"｜身份 {len(loader.identities())} 个")
    if loader.pairs:
        per = {i: len(v) for i, v in loader.group_by_identity().items()}
        detail = "  ".join(f"{k}×{n}" for k, n in per.items())
        L.append(f"  各身份配对数：{detail}")
        L.append("  配对样例：")
        for p in loader.pairs[:6]:
            L.append(f"    {p.describe()}")
        if loader.n_pairs > 6:
            L.append(f"    ... 另有 {loader.n_pairs - 6} 对")
    else:
        L.append("  ⚠️ 0 对 → 无法训练（需要同身份 ≥2 视图，且 view 已声明）")
    if loader.gaps:
        L.append("-" * 74)
        L.append("  缺口")
        for gp in loader.gaps:
            L.append(f"    ⚠️ {gp}")
    L.append("=" * 74)
    return "\n".join(L)


__all__ = ["ViewSample", "FitPair", "PairViewLoader", "format_dataset_report",
           "load_image", "DEFAULT_SIZE"]

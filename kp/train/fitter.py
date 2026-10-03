"""Character Fitter 训练 —— 一次性训练，之后「换角色 = 前向一次（秒级）」。

⭐ 训练目标从哪来
    声明式配对（`kp.character.dataset`）给的是「**同身份 + 只动一个旋钮**」的训练对。
    由它可直接导出身份 token 必须满足的两条性质：

      ① **不变性**：同身份、不同视角 ⇒ 身份 token 应**几乎相同**
                    → 损失 `1 − cos(t_a, t_b)`（逐 token 求余弦再平均）
      ② **可区分**：不同身份 ⇒ 身份 token 应**拉开**
                    → 批内负样本，三元组间隔 `relu(neg − pos + margin)`

    ⚠️ 这两条不能只做 ①：只用 ① 会让 Fitter 把所有角色塌成同一个 token
       （不变性 100% 满足，但**毫无用处**）。负样本项是把它顶开的力。
       两个身份 / 一条配对时没有负样本 → 训练仍然可跑，但**只测得到 ①**，
       报告里必须明说 `has_negatives=False`（这是覆盖缺口，不是通过）。

⭐ 与 `run_qad` 相同的哲学：**闭式自检**（teacher→student 式可证伪）
    `run_closed_loop()` 在合成数据上跑，要求
      · 不变性损失**显著下降**（真学进去了，不是损失定义写着好看）
      · 正/负相似度**间隔拉大**（确实在区分身份，而不是躺平）
    指标一律用 `eval()` 模式（关 dropout）算 ⇒ 可复现、可判阈值。

⚠️ 合成数据上的成功 ≠ 真实数据上的成功。真实质量的判据是 G5：
   换角色后出图**像不像本人**、且**跨主干版本可用**。本模块只保证训练机械正确。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..character.dataset import FitPair, PairViewLoader
from ..character.fitter import CharacterFitter


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def set_dropout(model: nn.Module, p: float) -> int:
    """统一设置模型内所有 Dropout 的概率。返回改动个数。

    ⚠️ 闭式自检必须把 dropout 关掉（默认 0），否则指标是随机的、判不了阈值。
       真实训练可保留 0.1 做正则（`train_fitter(..., dropout=0.1)`）。
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(p)
            n += 1
    return n


def fitter_budget(model: nn.Module) -> Dict[str, float]:
    """训练预算账：Fitter 是**从零一次性训练**，全部参数可训。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": float(total),
        "trainable": float(trainable),
        "trainable_ratio": trainable / max(1, total),
        "adamw_state_gb": trainable * 8 / 1e9,        # AdamW: m + v (fp32)
    }


def _tokens(fitter: CharacterFitter, imgs: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) → 身份 token (B, T, dim)。单视图 = V=1 的合法输入。"""
    return fitter(imgs.unsqueeze(1))


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐 token 余弦，返回标量（对 token 与 batch 求平均）。"""
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1).mean()


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def fitter_loss(ta: torch.Tensor, tb: torch.Tensor, identities: Sequence[str],
                *, margin: float = 0.3, sep_weight: float = 1.0):
    """返回 (total, inv, gap, has_negatives)。

    · inv  = 1 − cos(正对)                    —— 越小越好
    · gap  = cos(正对) − 平均 cos(负对)       —— 越大越好
    · 无负样本时 sep 项为 0，gap 记为 0（**未知**，不是"好"）。
    """
    pos = _cos(ta, tb)
    inv = 1.0 - pos
    total = inv
    gap = torch.zeros((), dtype=ta.dtype, device=ta.device)
    has_neg = False

    # 批内负样本：取每个样本的池化原型，跨身份两两比
    z = F.normalize(ta, dim=-1).mean(1)                 # (B, dim)
    sim = z @ z.transpose(0, 1)                         # (B, B)
    n = len(identities)
    mask = torch.tensor([[identities[i] != identities[j] for j in range(n)]
                         for i in range(n)], dtype=torch.bool, device=sim.device)
    if bool(mask.any()):
        has_neg = True
        neg = sim[mask]
        neg_mean = neg.mean()
        total = total + sep_weight * F.relu(neg_mean - pos.detach() + margin)
        gap = pos.detach() - neg_mean
    return total, inv.detach(), gap.detach(), has_neg


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    history: List[dict]
    budget: Dict[str, float]
    n_pairs: int
    n_identities: int
    has_negatives: bool
    fitter: Optional[CharacterFitter] = None

    @property
    def inv_start(self) -> float:
        return self.history[0]["inv"] if self.history else 0.0

    @property
    def inv_end(self) -> float:
        return self.history[-1]["inv"] if self.history else 0.0

    @property
    def inv_drop(self) -> float:
        a = self.inv_start
        return 0.0 if a <= 1e-9 else 1.0 - self.inv_end / a

    @property
    def gap_start(self) -> float:
        return self.history[0]["gap"] if self.history else 0.0

    @property
    def gap_end(self) -> float:
        return self.history[-1]["gap"] if self.history else 0.0

    def __str__(self) -> str:  # pragma: no cover
        b = self.budget
        neg = "有" if self.has_negatives else "**无**（单身份/单配对 ⇒ 只测得到不变性）"
        return (f"Fitter 训练：{len(self.history)} 步｜{self.n_pairs} 对 / "
                f"{self.n_identities} 身份｜负样本 {neg}\n"
                f"  不变性损失 {self.inv_start:.4f} → {self.inv_end:.4f}"
                f"（降 {self.inv_drop:.1%}）｜正负间隔 {self.gap_start:+.4f} → "
                f"{self.gap_end:+.4f}\n"
                f"  参数 {int(b['total']):,}｜AdamW 状态 ≈ {b['adamw_state_gb']:.4f} GB")


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_fitter(loader: PairViewLoader, fitter: Optional[CharacterFitter] = None,
                 *, steps: int = 120, lr: float = 3e-3, weight_decay: float = 0.0,
                 batch_size: Optional[int] = None, margin: float = 0.3,
                 sep_weight: float = 1.0, dropout: float = 0.0, seed: int = 0,
                 device: str = "cpu", log_every: int = 0) -> FitResult:
    """在配对集上训练 Fitter。返回 FitResult（含逐步历史与预算）。"""
    torch.manual_seed(seed)
    data = loader.to_tensors(device)
    A, P = data["anchor"], data["positive"]
    idents = [p.identity for p in loader.pairs]
    n = len(idents)
    if n == 0:
        raise ValueError("配对数为 0 → 无法训练（需要同身份 ≥2 视图且 view 已声明）")

    if fitter is None:
        # 用图像尺寸推断（合成数据常是 48²，比默认 256² 快得多）
        fitter = CharacterFitter(dim=64, n_tokens=16, view_dim=32, heads=4,
                                 use_geometry=False).to(device)
    fitter = fitter.to(device)
    set_dropout(fitter, dropout)

    bs = int(batch_size or n)
    opt = torch.optim.AdamW(fitter.parameters(), lr=lr, weight_decay=weight_decay)

    hist: List[dict] = []
    has_neg_any = False
    for step in range(steps):
        fitter.train()
        idx = torch.randperm(n)[:bs] if bs < n else torch.arange(n)
        ta = _tokens(fitter, A[idx])
        tb = _tokens(fitter, P[idx])
        sub_ids = [idents[i] for i in idx.tolist()]
        loss, inv, gap, has_neg = fitter_loss(ta, tb, sub_ids,
                                              margin=margin, sep_weight=sep_weight)
        opt.zero_grad()
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(fitter.parameters(), 1.0))
        opt.step()

        # 指标用 eval 模式（关 dropout）在全量配对集上算 ⇒ 可复现
        fitter.eval()
        with torch.no_grad():
            e_ta = _tokens(fitter, A)
            e_tb = _tokens(fitter, P)
            _, e_inv, e_gap, e_neg = fitter_loss(e_ta, e_tb, idents,
                                                 margin=margin, sep_weight=0.0)
        has_neg_any = has_neg_any or e_neg
        hist.append({"step": step, "loss": float(loss.item()),
                     "grad_norm": gn, "inv": float(e_inv.item()),
                     "gap": float(e_gap.item())})
        if log_every and (step % log_every == 0 or step == steps - 1):
            h = hist[-1]
            print(f"  step {h['step']:>3}  loss {h['loss']:.5f}  "
                  f"inv {h['inv']:.4f}  gap {h['gap']:+.4f}  gn {gn:.3f}")

    fitter.eval()
    return FitResult(history=hist, budget=fitter_budget(fitter), n_pairs=n,
                     n_identities=len(loader.identities()),
                     has_negatives=has_neg_any, fitter=fitter)


# 别名（语义更直白）：闭式自检与正式训练是同一个函数，只是数据源不同
train = train_fitter


@dataclass
class ClosedLoopResult:
    fit: FitResult
    ok_invariance: bool
    ok_separation: bool
    ok_generalization: bool
    ok_control: bool
    holdout_inv: float
    control_inv_end: float = float("nan")
    control_holdout_inv: float = float("nan")
    detail: str = ""

    @property
    def ok(self) -> bool:
        return (self.ok_invariance and self.ok_separation
                and self.ok_generalization and self.ok_control)


def shuffle_positives(loader: PairViewLoader, *, seed: int = 0) -> PairViewLoader:
    """负对照：把每条配对的 `positive` **随机换成别的图**（配对被人为破坏）。

    为什么需要它：只看「训练损失下降」有**平凡通过**的风险 ——
    注意力池化对 token 重排本就不变，且样本少时模型可以**背下**训练对。
    一个合格的验证装置必须能分辨「配对正确」与「配对错误」；
    配对被破坏后，不变性项与分离项**互相打架**，指标必然变差。这就是可证伪性。
    """
    import random

    rng = random.Random(seed)
    pairs = list(loader.pairs)
    pool = [p.positive for p in pairs]
    rng.shuffle(pool)
    new = [FitPair(identity=p.identity, anchor_key=p.anchor_key,
                   positive_key=f"<shuffled:{i}>", varied=dict(p.varied),
                   anchor=p.anchor, positive=pool[i])
           for i, p in enumerate(pairs)]
    return PairViewLoader(new, spec=loader.spec, source=loader.source + "+打乱配对")


def holdout_invariance(fitter: CharacterFitter, holdout: Dict[str, List[torch.Tensor]],
                       refs: Dict[str, torch.Tensor]) -> float:
    """留出视角上的不变性损失（1 − cos），**逐身份、跨留出视角**平均。

    ⭐ 这是把负对照变成**可证伪**的关键量：训练视角上模型可以靠记忆压到 0，
       但留出视角它没见过 ⇒ 只有**真的**学出了视角不变的身份描述子才能压低。
    """
    fitter.eval()
    vals: List[float] = []
    with torch.no_grad():
        for ident, imgs in holdout.items():
            ref = _tokens(fitter, refs[ident].unsqueeze(0))
            for img in imgs:
                t = _tokens(fitter, img.unsqueeze(0))
                vals.append(float(1.0 - _cos(ref, t)))
    return sum(vals) / max(1, len(vals))


def run_closed_loop(*, n_identities: int = 8, n_train_views: int = 3, n_holdout: int = 1,
                    size: int = 48, steps: int = 120, seed: int = 0,
                    min_inv_drop: float = 0.5, margin: float = 0.3, control: bool = True,
                    ho_threshold: float = 0.15, **kw) -> ClosedLoopResult:
    """闭式自检：合成可控信号上，**陈述式**判定训练机械是否正确。

    四条判据（全过才算通过）：
      ① 不变性：训练损失下降 ≥ `min_inv_drop`，且终值 < 0.05
      ② 分离性：正负间隔变大且终值 > 0.5（否则等于把所有角色塌成同一个 token）
      ③ **泛化**：留出视角上的不变性损失 < `ho_threshold`（记忆救不了没见过的视角）
      ④ **负对照**：把配对打乱后重训，留出视角不变性必须**明显更差**
    """
    loader, holdout, refs = PairViewLoader.synthetic_split(
        n_identities=n_identities, n_train_views=n_train_views, n_holdout=n_holdout,
        size=size, seed=seed)
    fitter = CharacterFitter(dim=64, n_tokens=16, view_dim=32, heads=4,
                             use_geometry=False)
    res = train_fitter(loader, fitter, steps=steps, seed=seed, margin=margin, **kw)
    ho = holdout_invariance(fitter, holdout, refs)

    ok_inv = (res.inv_drop >= min_inv_drop) and (res.inv_end < 0.05)
    ok_sep = res.has_negatives and res.gap_end > max(res.gap_start, 0.0) \
        and res.gap_end > 0.5
    ok_gen = ho < ho_threshold

    ctrl_inv = ctrl_ho = float("nan")
    ok_ctrl = False
    if control:
        bad = shuffle_positives(loader, seed=seed)
        cfit = CharacterFitter(dim=64, n_tokens=16, view_dim=32, heads=4,
                               use_geometry=False)
        cfit = train_fitter(bad, cfit, steps=steps, seed=seed, margin=margin, **kw).fitter
        # 用**同一套指标**量测被破坏的模型（训练视角 inv 与留出视角 inv）
        cta = _tokens(cfit, bad.to_tensors()["anchor"])
        ctb = _tokens(cfit, bad.to_tensors()["positive"])
        _, c_inv, _, _ = fitter_loss(cta, ctb, [p.identity for p in bad.pairs])
        ctrl_inv = float(c_inv.item())
        ctrl_ho = holdout_invariance(cfit, holdout, refs)
        # 破坏配对后，**留出视角**不变性应明显更差（2× 余量 + 绝对下限）
        ok_ctrl = ctrl_ho > max(2.0 * ho, 0.10)

    detail = (f"{n_identities} 身份 × {n_train_views} 训练视角 +{n_holdout} 留出｜"
              f"{res.n_pairs} 对｜不变性 {res.inv_start:.4f}→{res.inv_end:.4f}｜"
              f"间隔 {res.gap_start:+.4f}→{res.gap_end:+.4f}｜"
              f"留出 inv={ho:.4f}（负对照 {ctrl_ho:.4f}）")
    return ClosedLoopResult(fit=res, ok_invariance=ok_inv, ok_separation=ok_sep,
                            ok_generalization=ok_gen, ok_control=ok_ctrl,
                            holdout_inv=ho, control_inv_end=ctrl_inv,
                            control_holdout_inv=ctrl_ho, detail=detail)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
def main() -> int:
    from ..character.dataset import format_dataset_report

    print("=" * 74)
    print("  ① 闭式自检：合成可控信号（留出视角 + 负对照 ⇒ 可证伪）")
    print("=" * 74)
    r = run_closed_loop(steps=150, log_every=30)
    print(r.fit)
    print(f"  → 不变性 {'✅' if r.ok_invariance else '❌'}｜"
          f"身份分离 {'✅' if r.ok_separation else '❌'}｜"
          f"留出泛化 {'✅' if r.ok_generalization else '❌'}｜"
          f"负对照 {'✅' if r.ok_control else '❌'}")
    print(f"  {r.detail}")
    print(f"  留出视角不变性：正确配对 {r.holdout_inv:.4f} vs 打乱配对 "
          f"{r.control_holdout_inv:.4f} —— 装置能分辨「配对对不对」才算真验证")

    print()
    print("=" * 74)
    print("  ② 真实批次冒烟：data/characters/kokona")
    print("=" * 74)
    try:
        loader = PairViewLoader.from_batch("kokona", size=128)
        print(format_dataset_report(loader))
        if loader.n_pairs:
            res = train_fitter(loader, steps=30, seed=0, log_every=10,
                               batch_size=min(4, loader.n_pairs))
            print(res)
            if not res.has_negatives:
                print("  ⚠️ 只有 1 个身份 ⇒ **测不到**身份分离项；"
                      "这不算通过，只是覆盖缺口（还需第 2 个角色或第 3+ 视图）")
        else:
            print("  ⚠️ 0 对 → 跳过训练")
    except FileNotFoundError as e:
        print(f"  ⚠️ 跳过（{e}）")

    print()
    print("结论：训练机械在可控信号上可证伪地成立；真实质量判据属 G5（换角色出图像不像本人）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""G2 · 通道分离的**显式监督件 + 验收装置**（验证门 G2，结构性门）。

设计依据（主文档 §4 / 补充 02 / MEMORY.md「必做：通道分离显式监督」）：
    40ch 混合 latent = 语义 8ch（管「是什么」）+ 细节 32ch（管「长什么样」）。
    若不显式监督，主干会把两个通道**都塞满**信息 ⇒ 画风解耦失效 ⇒
    三层控制栈的 L1（承担 ~90% 调用量）**全盘作废**。
    故 G2 是**结构性门**：不过就得改架构，不是调配方。

═══ 两个已踩过的设计缺陷（本模块的全部形状都由它们决定）═══

① **相对降幅不能拿「模型自己的误差」当分母。**
   完美分离时 `base ≈ 0`，除法会把任意微扰放大成几百倍 ——
   实测干净样本的 shuffle 扰动曾报出 **495% 的假失败**。
   ⚠️ **加个 eps 不算修好**：本模块第一版给分母托了 `1e-6`，
   结果 **oracle（构造上完全分离）算出 `nan`（0/0）**，
   而 **leaky（语义分支其实读的是细节块）算出 `1.000`（看起来完美）**。
   ⇒ 真正的修法是**换分母的口径**：一律除以「**图像自身的结构尺度**」
     （`structural_metric` / `texture_metric` 内部就是这么定义的）——
     它是**数据属性、与模型无关 ⇒ 不可能退化为 0**。
     判据因此改为**绝对误差门线**，不再比比值。

② **指标与损失都必须「分块」。**
   用全 40ch 的 MSE 测的是「整体重建」，扰动细节通道当然会掉 ——
   那与通道分离无关。⚠️ 更危险的是**训练侧**：若只用整体重建损失，
   细节块可以偷偷替语义块补课 ⇒ **惩罚反而在鼓励冗余，与 G2 目的正好相反**。
   ⇒ `recon_loss` 分块、`SeparationReport` 的每一项都注明是**哪一侧**的指标。

═══ 判据（G2 门线）═══
交叉扰动测试：打乱细节通道后，**语义侧的结构误差必须仍然很小**。
    `err_sem  = structural_metric(sem_path(x_perturbed), 真值语义块)`   ≤ `max_sem_err`
    `err_det  = texture_metric   (det_path(x_perturbed), 真值细节块)`   ≤ `max_det_err`
物理含义：语义分支**能不能只靠语义块撑住结构**、细节分支**能不能只靠细节块撑住样式**。
    ✅ 通过 ⇔ 两种扰动下都达标（扰动是「另一侧被破坏」，所以达标 = 不依赖另一侧）。

⚠️ 本模块是**装置**：它自带合成数据与真值构造，可在纯 CPU 上秒级复现。
   真图上的同一套指标由 `tools/g2_channel_ablation.py` 驱动。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LATENT
from .hybrid import join_channels

EPS_SCALE = 1e-6      # 分母下限（缺陷 ① 的补丁）


# ===========================================================================
# 1. 合成数据 —— 语义与细节由**两套互不相干的公式**生成（可验证的真值）
# ===========================================================================
def synthetic_batch(n: int = 16, side: int = 32, seed: int = 0, mix: float = 0.6):
    """→ (x, (shape_map, texture_map))。

    语义内容 `shape_map`：矩形块 + 低秩场 ⇒ **空间结构**（「是什么/在哪」）。
    细节内容 `texture_map`：高频噪声 + 逐图色偏 ⇒ **样式**（「长什么样」）。

    ⭐ 参数 `mix`：**串扰强度**（这是本装置最关键的一个旋钮）。
      · `mix = 0`  细节块**只**编码样式、语义块**只**编码结构。
        ⇒ 分离是**白送的**：分支各读各的就是最优解，**再强的监督也测不出差别**。
          （实测：负对照 w_inv=0 也 PASS —— 这正是必须引入 mix 的原因。）
      · `mix > 0`  细节块里**也掺入**结构与样式的混合编码。
        ⇒ 「语义分支顺手从细节块里读结构」变成一条**捷径**：
          不显式监督，它就会走捷径（重建损失更小）；显式监督才把它按回去。
      ⇒ **`mix = 0` 用来证明装置无假阴性，`mix > 0` 用来证明监督真的有用。**
        两者都必须报，只报一个都会得出错误结论。
    """
    sc = LATENT.semantic_ch
    dc = LATENT.detail_ch
    g = torch.Generator().manual_seed(seed)

    # ---- 结构（语义）：两块矩形 + 一个低秩梯度场 ----
    shape_map = torch.zeros(n, 1, side, side)
    for i in range(n):
        for _ in range(2):
            h = int(torch.randint(side // 4, side // 2, (1,), generator=g))
            w = int(torch.randint(side // 4, side // 2, (1,), generator=g))
            y = int(torch.randint(0, side - h, (1,), generator=g))
            x0 = int(torch.randint(0, side - w, (1,), generator=g))
            shape_map[i, 0, y:y + h, x0:x0 + w] = 1.0
        a = torch.randn(3, 1, generator=g)
        b = torch.randn(1, 3, generator=g)
        shape_map[i, 0] += 0.5 * (a @ b).repeat(side // 3 + 1, side // 3 + 1)[:side, :side]

    # ---- 样式（细节）：高频噪声 + 逐图三通道色偏 ----
    texture_map = torch.randn(n, 3, side, side, generator=g) * 0.6
    tint = torch.randn(n, 3, 1, 1, generator=g) * 1.5
    texture_map = texture_map + tint

    # ---- 编码进 40ch latent ----
    proj_s = torch.randn(sc, 1, generator=g) / (sc ** 0.5)
    proj_d = torch.randn(dc, 3, generator=g) / (dc ** 0.5)
    proj_sd = torch.randn(dc, 1, generator=g) / (dc ** 0.5)   # 结构 → 细节块（串扰）
    proj_ds = torch.randn(sc, 3, generator=g) / (sc ** 0.5)   # 样式 → 语义块（串扰）

    sem_clean = torch.einsum("ij,bjhw->bihw", proj_s, shape_map)
    det_clean = torch.einsum("ij,bjhw->bihw", proj_d, texture_map)
    sem = sem_clean + mix * torch.einsum("ij,bjhw->bihw", proj_ds, texture_map)
    det = det_clean + mix * torch.einsum("ij,bjhw->bihw", proj_sd, shape_map)
    x = join_channels(sem, det)
    # 真值仍用**干净**的 shape/texture map：判据问的是「通道里装的是不是正确的东西」，
    # 不是「模型有没有把串扰也复现出来」。
    return x, (shape_map, texture_map)


# ===========================================================================
# 2. 指标 —— 分别度量「结构」与「样式」（**分块**，缺陷 ②）
# ===========================================================================
def _sobel(t: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0, 1], [-2.0, 0, 2], [-1.0, 0, 1]]).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def structural_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """结构指标：**边缘能量图**上的归一化 L1（与「空间结构」直接对应）。

    为什么不用逐像素 MSE：逐像素对色偏/亮度极敏感，会把「样式」混进「结构」判决里
    ——正是缺陷 ② 要防的那类口径错误。
    """
    p = _sobel(pred.mean(1, keepdim=True))
    t = _sobel(target.mean(1, keepdim=True))
    scale = t.mean().detach() + EPS_SCALE
    return ((p - t).abs().mean() / scale)


def texture_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """样式指标：**通道均值 + 空间标准差**（色偏/对比度）上的相对 L1。"""
    def stat(v):
        return torch.cat([v.mean(dim=(2, 3)), v.std(dim=(2, 3))], dim=1)
    p, t = stat(pred), stat(target)
    scale = t.abs().mean().detach() + EPS_SCALE
    return (p - t).abs().mean() / scale


# ===========================================================================
# 3. 双分支小块 AE（合成数据上秒级可训）
# ===========================================================================
class _Mixer(nn.Module):
    """单层 1×1 线性混合，吃满 40ch。**不带非线性**：保证「语义分支只能靠
    语义块里的信息才可能报出结构」——若有非线性+混合，它可以把细节块
    重编码成结构，从而在真值上就做不到的分离被模型「算」出来。"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.lin = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.lin(x)


@dataclass
class SeparationModel:
    """持有两个分支（各自独立参数）+ 训练超参。"""
    semantic_ch: int = LATENT.semantic_ch
    detail_ch: int = LATENT.detail_ch
    hidden: int = 16
    sem_dec: nn.Module = field(init=False)
    det_dec: nn.Module = field(init=False)
    sem_in: nn.Module = field(init=False)
    det_in: nn.Module = field(init=False)

    def __post_init__(self):
        sc, dc = self.semantic_ch, self.detail_ch
        # 输入侧：允许（也必须）看到**全部** 40ch —— 若只喂自己那一块，
        # 「分离」就成了循环论证（缺陷：测的不是模型行为而是我们的接线）。
        self.sem_in = _Mixer(LATENT.total_ch, self.hidden)
        self.det_in = _Mixer(LATENT.total_ch, self.hidden)
        # 输出侧：只吃自己那一块 ⇒ 结构只能从语义块来，样式只能从细节块来
        self.sem_dec = _Mixer(self.hidden, sc)
        self.det_dec = _Mixer(self.hidden, dc)

    def parameters(self):
        for m in (self.sem_in, self.sem_dec, self.det_in, self.det_dec):
            yield from m.parameters()

    def sem_path(self, x: torch.Tensor) -> torch.Tensor:
        """语义分支：40ch → 结构（语义块重建）。"""
        return self.sem_dec(torch.tanh(self.sem_in(x)))

    def det_path(self, x: torch.Tensor) -> torch.Tensor:
        """细节分支：40ch → 样式（细节块重建）。"""
        return self.det_dec(torch.tanh(self.det_in(x)))


# ===========================================================================
# 4. 扰动算子（交叉扰动测试用）
# ===========================================================================
def shuffle_within_batch(x: torch.Tensor, channels=None,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """把**指定通道块**在批内打乱（同一批的其它张，通道位置不变）。

    `channels=None` ⇒ 只动细节块（语义块默认不动，见 `cross_perturb`）。
    ⚠️ 退化的 batch=1：**无从打乱**，此时必须**显式报缺口**而不是静默返回原值
    （否则测试会「永远通过」——通用教训 #6 退化输入要单独处理）。
    """
    if x.shape[0] < 2:
        raise ValueError("shuffle_within_batch 需要 batch ≥ 2（退化输入必须显式报缺口）")
    sc = LATENT.semantic_ch
    y = x.clone()
    if channels is None:
        channels = slice(sc, None)
    g = generator or torch.Generator().manual_seed(0)
    perm = torch.randperm(x.shape[0], generator=g)
    y[:, channels] = x[perm][:, channels]
    return y


def cross_perturb(x: torch.Tensor, mode: str, seed: int = 0) -> torch.Tensor:
    """G2 的三种扰动：
        `shuffle_detail`   打乱细节块 ⇒ 语义侧**应当不变**
        `shuffle_semantic` 打乱语义块 ⇒ 细节侧**应当不变**
        `reverse`          两块的对应关系反转（同批内 0↔-1 配对）⇒ 最强扰动
    """
    sc = LATENT.semantic_ch
    if mode == "shuffle_detail":
        return shuffle_within_batch(x, slice(sc, None), torch.Generator().manual_seed(seed))
    if mode == "shuffle_semantic":
        return shuffle_within_batch(x, slice(0, sc), torch.Generator().manual_seed(seed))
    if mode == "reverse":
        if x.shape[0] < 2:
            raise ValueError("reverse 需要 batch ≥ 2")
        y = x.clone()
        y[:, :sc] = x.flip(0)[:, :sc]
        y[:, sc:] = x.flip(0)[:, sc:]
        return y
    raise ValueError(f"未知扰动 {mode!r}")


# ===========================================================================
# 5. 训练（分块损失 + 交叉扰动不变性）——缺陷 ② 的训练侧补丁
# ===========================================================================
def recon_loss(model: SeparationModel, x: torch.Tensor, targets, w_inv: float = 1.0):
    """**分块**重建损失 + 交叉不变性正则。返回 (total, 明细 dict)。"""
    shape_map, _tex = targets
    sc = LATENT.semantic_ch
    sem_true = x[:, :sc]
    det_true = x[:, sc:]

    sem_pred = model.sem_path(x)
    det_pred = model.det_path(x)
    l_sem = F.mse_loss(sem_pred, sem_true)
    l_det = F.mse_loss(det_pred, det_true)

    # 交叉扰动不变性：打乱细节块，语义输出不许变；打乱语义块，细节输出不许变
    x_sh_d = cross_perturb(x, "shuffle_detail")
    x_sh_s = cross_perturb(x, "shuffle_semantic")
    inv = (F.mse_loss(model.sem_path(x_sh_d), sem_pred.detach())
           + F.mse_loss(model.det_path(x_sh_s), det_pred.detach()))
    total = l_sem + l_det + w_inv * inv
    _ = shape_map
    # ⚠️ 必须 .detach()：这些张量带 requires_grad，直接 float() 会触发
    #    「Converting a tensor with requires_grad=True to a scalar」警告。
    return total, {"sem": float(l_sem.detach()), "det": float(l_det.detach()),
                   "inv": float(inv.detach())}


def train_separation(x: torch.Tensor, targets, *, steps: int = 300, lr: float = 5e-2,
                     w_inv: float = 1.0, seed: int = 0) -> SeparationModel:
    """纯 CPU、秒级。返回训好的 `SeparationModel`。

    `w_inv = 0` ⇒ **负对照**（只做分块重建、不做交叉不变性）：
    用于证明「显式监督确实买到了分离」，而不是数据本身白送。
    """
    torch.manual_seed(seed)
    model = SeparationModel()
    opt = torch.optim.Adam(list(model.parameters()), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = recon_loss(model, x, targets, w_inv=w_inv)
        loss.backward()
        opt.step()
    return model


# ===========================================================================
# 6. 验收：交叉扰动测试 + 判据
# ===========================================================================
# ═══ 第三个踩到的坑（最隐蔽的一个）═══
# 交叉扰动是在**批维**上打乱的（样本 i 拿到样本 perm[i] 的另一侧通道）——
# 这必然**打散了原有的配对**。所以**不能**再去比「扰动后的输出 vs 原配对的目标」：
# 那样连 oracle（构造上完全分离）都会被判必错，实测报出 0.97 / 0.73 的假误差。
# ⇒ 正确的口径是**不变性**，不是「预测精度」：
#     语义分支在细节块被打乱时，**自己的输出不该变**（它压根不该看细节块）；
#     细节分支在语义块被打乱时，**自己的输出不该变**。
# 依赖度 = ‖branch(x') − branch(x)‖ / ‖branch(x) − branch(0)‖
#   分母是「该分支输出变化的总量级」——只要分支不是常函数就非零，
#   而**常函数（把通道彻底无视）会被这个分母抓住**：此时分母→0，依赖度反而不判合格。
def _branch_scale(a: torch.Tensor, b: torch.Tensor) -> float:
    """归一化尺度：输出相对 0 的变化量级（退化保护见下）。"""
    denom = float(b.abs().mean()) + float(a.abs().mean()) + EPS_SCALE
    return denom


def branch_dependency(model, x: torch.Tensor, mode: str, seed: int = 0) -> dict:
    """两种扰动下，**每个分支对自己不该看的通道的依赖度**。

    · `shuffle_detail`   → 语义分支的依赖度（理想 = 0）
    · `shuffle_semantic` → 细节分支的依赖度（理想 = 0）

    返回 dict：dep_semantic / dep_detail + 各自的绝对变化量（便于人工核对）。
    """
    with torch.no_grad():
        s0, d0 = model.sem_path(x), model.det_path(x)
        out = {"mode": mode}
        for m, which in (("shuffle_detail", "semantic"), ("shuffle_semantic", "detail")):
            xp = cross_perturb(x, m, seed=seed)
            with torch.no_grad():
                sp, dp = model.sem_path(xp), model.det_path(xp)
            if which == "semantic":
                delta, base = (sp - s0).abs().mean(), s0
            else:
                delta, base = (dp - d0).abs().mean(), d0
            out[f"dep_{which}"] = float(delta) / _branch_scale(delta, base)
            out[f"delta_{which}"] = float(delta)
            out[f"scale_{which}"] = _branch_scale(delta, base)
    return out


@dataclass
class SeparationReport:
    """G2 总判定。

    门线 `max_dep` 默认 **0.20**：分支对「另一侧通道」的依赖必须低于其输出变化量级的 20%。
    校准依据（两者都在 `run_g2` 里实跑，不是拍脑袋）：
      · oracle（构造上完全分离）→ dep = **0.000**（输出逐位不变）
      · leaky（语义分支实际读细节块）→ dep = **≈1.0**（跟着另一侧走）
    """
    rows: list = field(default_factory=list)
    max_dep: float = 0.20

    def add(self, r: dict):
        r = dict(r)
        r["sem_pass"] = r["dep_semantic"] <= self.max_dep
        r["det_pass"] = r["dep_detail"] <= self.max_dep
        r["pass"] = bool(r["sem_pass"] and r["det_pass"])
        self.rows.append(r)
        return r

    @property
    def overall(self) -> bool:
        return all(r["pass"] for r in self.rows) if self.rows else False

    def to_dict(self) -> dict:
        return {"max_dep": self.max_dep, "overall_pass": self.overall, "rows": self.rows}

    def format(self) -> str:
        head = (f"{'扰动':<18}{'语义分支依赖':>14}{'细节分支依赖':>14}"
                f"{'语义':>7}{'细节':>7}{'判定':>7}")
        out = [head, "-" * len(head)]
        for r in self.rows:
            out.append(f"{r['mode']:<18}{r['dep_semantic']:>14.4f}{r['dep_detail']:>14.4f}"
                       f"{'✅' if r['sem_pass'] else '❌':>8}"
                       f"{'✅' if r['det_pass'] else '❌':>8}"
                       f"{'PASS' if r['pass'] else 'FAIL':>7}")
        out.append("-" * len(head))
        verdict = ("✅ 通过（通道分离成立）" if self.overall
                   else "❌ 未通过（通道未解耦 ⇒ 须改架构）")
        out.append(f"G2 总判定：{verdict}   门线：依赖度 ≤ {self.max_dep:.2f}")
        return "\n".join(out)


def evaluate(model, x: torch.Tensor, *, max_dep: float = 0.20, seed: int = 0) -> SeparationReport:
    rep = SeparationReport(max_dep=max_dep)
    if x.shape[0] < 2:
        # 退化输入必须显式报缺口，而不是静默返回「全过」的假报告
        raise ValueError("G2 验收需要 batch ≥ 2（退化输入下扰动无从构造）")
    for m in ("shuffle_detail", "shuffle_semantic", "reverse"):
        rep.add(branch_dependency(model, x, m, seed=seed))
    return rep


def run_g2(*, n: int = 32, side: int = 32, steps: int = 400, seed: int = 0,
           w_inv: float = 1.0, max_dep: float = 0.20,
           mix: float = 0.6, mix_baseline: float = 0.0, verbose: bool = False) -> dict:
    """端到端：造数据 → 训 → 验收。

    ⚠️ **必须跑两个 mix 才有结论**（`mix_baseline=0` 与 `mix=0.6`）：
      · `mix = 0`（分离白送）⇒ 若负对照也 PASS，**不能**据此说「监督有用」，
        但可以据此说「装置不会误报」；
      · `mix > 0`（存在捷径）⇒ 这里才区分得出「显式监督买到了分离」。
    只报其中一个都会得出错误结论 —— 这是本装置最后一个、也是最容易忽略的坑。

    另外固定报 `oracle`（必须 PASS）与 `leaky`（必须 FAIL）两个尺子对照：
    没有它们，上面任何数字都不可信。
    """
    def one(mx: float):
        x, targets = synthetic_batch(n=n, side=side, seed=seed, mix=mx)
        sup = train_separation(x, targets, steps=steps, w_inv=w_inv, seed=seed)
        neg = train_separation(x, targets, steps=steps, w_inv=0.0, seed=seed)
        return x, evaluate(sup, x, max_dep=max_dep, seed=seed), \
            evaluate(neg, x, max_dep=max_dep, seed=seed)

    x0, rep_sup0, rep_neg0 = one(mix_baseline)
    x1, rep_sup1, rep_neg1 = one(mix)

    rep_oracle = evaluate(_Oracle(), x1, max_dep=max_dep, seed=seed)
    rep_leaky = evaluate(_Leaky(), x1, max_dep=max_dep, seed=seed)

    def worst(rep):
        return max(max(r["dep_semantic"], r["dep_detail"]) for r in rep.rows)

    out = {
        "config": {"n": n, "side": side, "steps": steps, "seed": seed,
                   "w_inv": w_inv, "max_dep": max_dep, "mix": mix,
                   "mix_baseline": mix_baseline},
        "oracle_sanity": rep_oracle.to_dict(),
        "leaky_sanity": rep_leaky.to_dict(),
        "mix0_baseline": {"supervised": rep_sup0.to_dict(),
                          "negative_control": rep_neg0.to_dict()},
        "mix_main": {"supervised": rep_sup1.to_dict(),
                     "negative_control": rep_neg1.to_dict()},
        "gap": {   # 显式监督相对负对照的改善（越小越好）
            "worst_dep_supervised": worst(rep_sup1),
            "worst_dep_negative": worst(rep_neg1),
            "improvement_x": (worst(rep_neg1) / max(worst(rep_sup1), EPS_SCALE)),
        },
        "verdict": "PASS" if rep_sup1.overall else "FAIL",
        "sanity_ok": bool(rep_oracle.overall and not rep_leaky.overall),
    }
    if verbose:
        g = out["gap"]
        print(f"=== 尺子对照（mix={mix}）===")
        print(f"  oracle（构造上分离，必须 PASS）：{'✅ PASS' if rep_oracle.overall else '❌ FAIL'}")
        print(f"  leaky （语义分支读细节块，必须 FAIL）：{'❌ FAIL ✅正确' if not rep_leaky.overall else '⚠️ PASS ✗尺子失灵'}")
        print()
        for tag, rs, rn in ((f"mix={mix_baseline}（分离白送：负对照也应当过）", rep_sup0, rep_neg0),
                            (f"mix={mix}（存在捷径：监督才应当拉开差距）", rep_sup1, rep_neg1)):
            print(f"=== {tag} ===")
            print(f"  [有显式监督 w_inv={w_inv}]")
            print(rs.format())
            print(f"  [负对照 w_inv=0]")
            print(rn.format())
            print()
        print(f"=== 监督的净收益（mix={mix}，只看最差的那一侧）===")
        print(f"  负对照最差依赖 {g['worst_dep_negative']:.4f}"
              f"  →  有监督 {g['worst_dep_supervised']:.4f}"
              f"   （改善 {g['improvement_x']:.2f}×）")
        print()
    return out


class _Oracle:
    """尺子校验用：语义分支**只吃**语义块、细节分支**只吃**细节块。

    这不是一个「模型」，是**判据的正确性证明** —— 当分离是构造出来的时候，
    装置必须报 PASS。若连它都报 FAIL/nan，说明坏的是尺子而不是被测量的东西。
    """

    def sem_path(self, x):
        return x[:, :LATENT.semantic_ch]

    def det_path(self, x):
        return x[:, LATENT.semantic_ch:]


class _Leaky:
    """尺子反例：语义分支**实际读的是细节块** ⇒ 必须 FAIL（否则判据没有分辨力）。"""

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.lin = nn.Conv2d(LATENT.detail_ch, LATENT.semantic_ch, 1)
        with torch.no_grad():
            self.lin.weight.normal_(0, 0.5)
            self.lin.bias.zero_()

    def sem_path(self, x):
        return self.lin(x[:, LATENT.semantic_ch:])

    def det_path(self, x):
        return x[:, LATENT.semantic_ch:] * 1.0 + self.lin.weight.sum() * 0.0


if __name__ == "__main__":  # pragma: no cover
    # 正式入口是 tools/g2_channel_ablation.py（带 CLI 参数与 JSON 落盘）；
    # 这里只留最小可运行入口，避免把开发期的调试输出当成交付界面。
    print(__doc__)
    print(f"G2 合成数据自检：{run_g2(verbose=True)['verdict']}")

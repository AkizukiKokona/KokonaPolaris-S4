"""M3「Latent 解耦」三条修法的可行性预实验（纯 CPU · 秒级）。

⭐ 设计要点（这一段是本文件的关键，别删）：
   不能在 `synthetic_batch(mix>0)` 上做这个实验 —— 那份数据每位置只有 2 个自由度，
   `sem` 与 `det` 都由同一对标量线性生成 ⇒ **逐位置 R² 必然饱和到 1.0**（数学必然，非数据冗余）。
   ⇒ 正确做法：**自己造一个「高冗余但自由度充足」的起点**（噪声副本），
     看惩罚能不能把它压下来。这样才有下降空间、才能分辨修法好坏。

跑法：  cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.probe.m3_fix
"""
from __future__ import annotations

import sys

import torch

from ..config import LATENT
from ..latent.real_separation import _r2 as _pos_r2

# 门线：由构造级已知答案校准（见 selftest §19「内容冗余测项先过已知答案」）
#   完全独立 0.0022 / 弱冗余 0.1044 / 精确拷贝 0.2517
# ⇒ 取 0.10 作为「有专属区」的判线：低于它算解耦良好。
MAX_CROSS_R2 = 0.10


def _blocks(z: torch.Tensor):
    return z[:, :LATENT.semantic_ch], z[:, LATENT.semantic_ch:]


def cross_r2(z: torch.Tensor) -> dict:
    """位置级跨块可预测性（口径唯一真源在 `real_separation._r2`）。"""
    sem, det = _blocks(z)
    return {"det_from_sem": _pos_r2(sem, det), "sem_from_det": _pos_r2(det, sem)}


def make_high_redundancy(n=16, side=16, noise=0.35, seed=0):
    """造一个**高冗余但自由度充足**的起点。

    做法：让细节块 = 语义块的**随机通道线性混合** + 独立噪声。
    `noise` 越小 ⇒ 冗余越强（noise=0 ⇒ 精确线性可预测 ⇒ R²=1）。
    ⭐ 与 `synthetic_batch` 的区别：**每位置的向量维度更高**（不是 2 个自由度），
       所以 R² 不会被结构性饱和到 1。
    """
    g = torch.Generator().manual_seed(seed)
    sc, dc = LATENT.semantic_ch, LATENT.detail_ch
    sem = torch.randn(n, sc, side, side, generator=g)
    W = torch.randn(sc, dc, generator=g) / (sc ** 0.5)          # 跨块线性混合
    det = torch.einsum("ncsw,cd->ndsw", sem, W)
    det = det + noise * torch.randn(n, dc, side, side, generator=g)
    return torch.cat([sem, det], 1)


# ---------------------------------------------------------------------------
# 修法 ①：跨块去相关惩罚（可微）
# ---------------------------------------------------------------------------
def penalty_cross_corr(z: torch.Tensor) -> torch.Tensor:
    """可微的跨块相关性惩罚（与 `cross_r2` 同族，但**可微**）。

    用「通道级 Gram 矩阵的非对角能量」而不是 R² —— R² 含 `lstsq`，
    对 z 不可微；Gram 非对角项是二次型，梯度稳定。

        目标：<sem_c, det_d> ≈ 0（对所有 c, d）

    ⚠️ **通道数从输入实测**（`a.shape[1]` / `b.shape[1]`），不能写死
       `LATENT.semantic_ch` / `detail_ch` —— 修法②的输出只有 `hidden` 个通道，
       写死会触发 einsum 广播错误（"subscript s has size 192 ... 256"）。
    """
    a = z[:, :LATENT.semantic_ch]
    b = z[:, LATENT.semantic_ch:]
    a = a.reshape(a.shape[0], a.shape[1], -1)
    b = b.reshape(b.shape[0], b.shape[1], -1)
    a = a - a.mean(2, keepdim=True)
    b = b - b.mean(2, keepdim=True)
    a = a / (a.norm(dim=2, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=2, keepdim=True) + 1e-6)
    g = torch.einsum("ncs,nds->ncd", a, b)          # (N, C_a, C_b) 相关系数
    return g.pow(2).mean()


def apply_fix_grad(z: torch.Tensor, *, steps=300, lr=0.05, w_corr=1.0) -> torch.Tensor:
    """在 latent 上直接优化去相关惩罚（**预演**「若 VAE 有这个约束会怎样」）。

    ⚠️ `z` 是叶子张量、没有 `.parameters()` —— 直接 `requires_grad_(True)` 后交给优化器。
    """
    zz = z.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([zz], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = w_corr * penalty_cross_corr(zz)
        loss.backward()
        opt.step()
    return zz.detach()


# ---------------------------------------------------------------------------
# 修法 ②：`patch_embed` 拆两路（不共享权重）
# ---------------------------------------------------------------------------
class SplitPatchEmbed:
    """语义块走一路、细节块走一路，**权重完全独立**。

    这是**结构保证**（推理期也成立），与修法 ① 的**训练期软约束**本质不同。
    量化参数量代价：两路独立 vs 一路 40→d：
        独立：(8+32)·d = 40d  （与一路相同）⇒ **参数量不变**，只多一次加法。
    """

    @staticmethod
    def forward(z: torch.Tensor, w_sem: torch.Tensor, w_det: torch.Tensor) -> torch.Tensor:
        sem, det = _blocks(z)
        return torch.einsum("ncsw,cd->ndsw", sem, w_sem) + \
               torch.einsum("ncsw,cd->ndsw", det, w_det)

    @staticmethod
    def trainable(z: torch.Tensor, *, steps=300, lr=0.05, hidden=32) -> torch.Tensor:
        """训两路独立投影，看「输出」是否比「输入」更解耦。"""
        d = hidden
        g = torch.Generator().manual_seed(0)
        w_sem = (torch.randn(LATENT.semantic_ch, d, generator=g) * 0.1).requires_grad_(True)
        w_det = (torch.randn(LATENT.detail_ch, d, generator=g) * 0.1).requires_grad_(True)
        zz = z.clone()
        for p in (w_sem, w_det):
            p.requires_grad_(True)
        opt = torch.optim.Adam([w_sem, w_det], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            out = SplitPatchEmbed.forward(zz, w_sem, w_det)
            loss = penalty_cross_corr(out)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = SplitPatchEmbed.forward(zz, w_sem, w_det)
        return out.detach()


def main() -> int:
    if not __debug__:
        raise RuntimeError("不要在 -O 下跑：断言会被移除")
    torch.manual_seed(0)
    print("=" * 72)
    print("M3「Latent 解耦」三条修法 · 可行性预实验（纯 CPU）")
    print("=" * 72)

    rows = []
    for noise, tag in ((0.00, "起点·精确冗余"), (0.20, "起点·强冗余"),
                       (0.50, "起点·中冗余"), (1.00, "起点·弱冗余")):
        z0 = make_high_redundancy(noise=noise)
        r0 = cross_r2(z0)
        rows.append((tag, r0["det_from_sem"], r0["sem_from_det"]))
        print(f"\n[{tag}]  cross_r2: det|sem={r0['det_from_sem']:.4f}  sem|det={r0['sem_from_det']:.4f}")

    # ---- 修法 ①：可微去相关惩罚 ----
    print("\n" + "-" * 72)
    print("修法 ①  VAE latent 上加可微跨块去相关惩罚（训练期约束）")
    print("-" * 72)
    for noise, tag in ((0.20, "强冗余"), (0.50, "中冗余")):
        z0 = make_high_redundancy(noise=noise)
        before = cross_r2(z0)["det_from_sem"]
        z1 = apply_fix_grad(z0)
        after = cross_r2(z1)["det_from_sem"]
        verdict = "✅ 压下来了" if after < before * 0.5 else "⚠️ 没压下来"
        print(f"  [{tag}] det|sem: {before:.4f} -> {after:.4f}   {verdict}")

    # ---- 修法 ②：patch_embed 拆两路 ----
    print("\n" + "-" * 72)
    print("修法 ②  patch_embed 拆两路、权重不共享（结构保证）")
    print("-" * 72)
    for noise, tag in ((0.20, "强冗余"), (0.50, "中冗余")):
        z0 = make_high_redundancy(noise=noise)
        before = cross_r2(z0)["det_from_sem"]
        out = SplitPatchEmbed.trainable(z0)
        after = cross_r2(out)["det_from_sem"]
        verdict = "✅ 压下来了" if after < before * 0.5 else "⚠️ 没压下来"
        print(f"  [{tag}] 输入 det|sem={before:.4f} -> 输出 {after:.4f}   {verdict}")
    print("  ⭐ 参数量代价：(8+32)·d = 40d，与单路 40→d **完全相同**（只多一次加法）")

    # ---- 结论 ----
    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    print("""
⚠️ **本实验能说的**（构造级，有已知答案支撑）：
   · 两条修法在「可微/可训练」的条件下**都能把跨块可预测性压下来** ⇒ 方向可行，不是空想。
   · 修法 ② **参数量代价为零**（40·d 两路独立 vs 一路 40→d 完全相同）。

⛔ **本实验不能说的**（务必别过度外推）：
   · **在真图上能不能压下来，本实验答不了** —— 这里的 latent 是人造的。
   · **压到多低才算够**，门线 0.10 只由构造级答案校准，真图上必须重新校准。
   · **压低 R² 会不会损失重建质量** —— 本实验完全没测这一项，
     而它恰恰是 DA-VAE 报告的核心矛盾（对齐/分离 vs 可建模性）。

⇒ **下一步必须是**：在真图上（`real_separation` 那条链路）重跑同一个对照，
   **同时记录 cross_r2 与重建误差两个量** —— 只看其中一个都会得出错误结论。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

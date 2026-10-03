"""单流 DiT 主干（3:1 混合注意力 / QK-Norm / Rectified Flow / Matryoshka）。

⚠️ 本文件属于 2026-10-03 **重写版**（原包因 `.gitignore` 未锚定被连带忽略而丢失）。
   依据：`design/` 主文档 §架构 + 补充01 §2.5（注意力修订）+ 补充04（身份走
   cross-attention）+ `kp/config.py` + `kp/selftest.py` / `kp/arch_report.py`
   / `kp/train/qad.py` 的实际调用面。

## 参数量结构（用来核对 arch_report 的参数量）
每个 block 恰好 **17·d²**：

    attention  qkv(3d²) + out(d²)              =  4·d²   （GatedLinear → buffer）
    mlp        fc1(d·2.5d) + fc2(2.5d·d)       =  5·d²   （GatedLinear → buffer）
    adaLN      Linear(d → 8d)                  =  8·d²   （nn.Linear → 真参数）

⇒ KP-S (d=1152, L=24)：17·1152²·24 ≈ 541M（+t-embed/文本路由/身份锚点 ⇒ 实测 ~0.58B）
⇒ KP-M (d=1792, L=32)：17·1792²·32 ≈ 1.75B（+其余 ⇒ 实测 1.875B，vs 标称 1.5B +25.0%）

## adaLN 的 8 段（⚠️ selftest 依赖第 8 段 = 身份门控）
    0:1:2 → attn  (shift, scale, gate)
    3:4:5 → mlp   (shift, scale, gate)
    6     → g_txt  文本流残差门控
    7     → g_id   **身份 cross-attention 门控** ← `adaLN[-1].bias[7d:8d]`

⚠️ adaLN 末层 **weight 全 0**（adaLN-Zero），但把 attn/mlp/txt 三个 gate 的
   **bias 初始化为 1.0** —— 否则整个 block 在初值是恒等映射，网络退化。
   身份门控（第 8 段）保持 0，满足「adaLN-Zero ⇒ 初始化时注入身份 token 不改输出」。

⭐ **为什么是 8 段不是 9 段（2026-10-03 瘦身）**
   原设计在 g_txt 与 g_id 之间留了一段 `g_geo`「几何分支门控（预留）」，但全代码库
   **从未有任何一处读取该段**（前向只消费段 6 与段 8）⇒ 它是**纯死参数**：
   每 block 白占 `d²`（KP-M 全模型 102.8M ≈ 5.2%）。
   ⚠️ FLOPs 两个分母别搞混：adaLN **自身** FLOPs `2·d·9d → 2·d·8d` = **−11.1%**；
   **全模型** FLOPs `18d² → 17d²` = **−5.6%**。两个都是真的，但别混着引。
   「预留接口」不该用「每层都付钱」的方式存在 —— 故删除，等几何分支真要用时再按需加回。
   ⚠️ 加回时注意：**几何分支走 CharaBridge 自己的门控**（`geo_kv` 在 K/V 内部融合 +
   `set_gate`），**从来不经过 adaLN** —— 段 7 从来就不是「几何分支的门」，只是个没接线的占位。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..capability.bus import CapabilityBus, GatedLinear
from ..config import DiTCfg, LATENT
from .common import RMSNorm, TimestepEmbedding, sincos_2d

__all__ = ["SingleStreamDiT", "DiTBlock", "build_attn_plan"]

# 注意力类型
LINEAR = "linear"      # gated linear attention（O(N)，省显存）
SIGMOID = "sigmoid"    # 非归一化 ⇒ 无 softmax 概率质量竞争 ⇒ 长提示不稀释
SOFTMAX = "softmax"    # 仅第 0 层保留一次（初始对齐锚点）


def build_attn_plan(layers: int, n_linear: int = 3, n_sigmoid: int = 1,
                    softmax_anchor: bool = True) -> List[str]:
    """按「每 4 层 = 3 线性 + 1 Sigmoid」铺注意力类型。

    ⚠️ 主文档补充01 §2.5：**全局不出现 softmax，仅在 patch embed 后的第 0 层保留一次**。
    ⇒ 第 0 层是 softmax 锚点，之后从 3:1 循环重新开始。
    """
    plan: List[str] = []
    if softmax_anchor and layers > 0:
        plan.append(SOFTMAX)
    cycle = [LINEAR] * max(1, n_linear) + [SIGMOID] * max(1, n_sigmoid)
    i = 0
    while len(plan) < layers:
        plan.append(cycle[i % len(cycle)])
        i += 1
    return plan[:layers]


def _gl(in_f: int, out_f: int, std: float = 0.02, bias: bool = False) -> GatedLinear:
    """建一个 GatedLinear（权重是 buffer ⇒ 冻结主干，可挂能力包）。"""
    w = torch.empty(out_f, in_f)
    nn.init.normal_(w, std=std / math.sqrt(max(1, in_f) / 64.0))
    b = torch.zeros(out_f) if bias else None
    return GatedLinear(w, b)


def _attn_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               kind: str, head_gate: Optional[torch.Tensor] = None,
               eps: float = 1e-6) -> torch.Tensor:
    """注意力内核。q/k/v: [B, H, N, Dh]（k/v 的 N 可与 q 不同 ⇒ 支持联合注意力）。"""
    if kind == SOFTMAX:
        out = F.scaled_dot_product_attention(q, k, v)
    elif kind == SIGMOID:
        # ⭐ 非归一化：每个 query-key 对独立打分，token 之间不抢概率质量
        #   （这就是「长提示不稀释」的机理）——softmax 把它们归一化到总和 1，
        #   提示越长每个词分到的份额越小。
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.sigmoid(q @ k.transpose(-1, -2) * scale)
        out = scores @ v
    else:  # LINEAR —— gated linear attention，(φ(q)·Σkᵀv) / (φ(q)·Σk)
        phi = F.elu
        qp = phi(q) + 1.0
        kp = phi(k) + 1.0
        kv = torch.einsum("bhmd,bhme->bhde", kp, v)          # Σ_n kₙ ⊗ vₙ
        num = torch.einsum("bhmd,bhde->bhme", qp, kv)        # φ(q)ᵀ (Σ k⊗v)
        ksum = kp.sum(dim=-2)                                # Σ_n φ(k)  →[B,H,D]
        den = torch.einsum("bhmd,bhd->bhm", qp, ksum).unsqueeze(-1)
        out = num / (den + eps)
    if head_gate is not None:
        out = out * torch.sigmoid(head_gate).reshape(1, -1, 1, 1)
    return out


class Attention(nn.Module):
    """注意力（kind 决定线性 / Sigmoid / softmax）。QK-Norm 由 cfg 控制。"""

    def __init__(self, dim: int, heads: int, kind: str, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} 必须能被 heads {heads} 整除"
        self.dim, self.heads, self.kind = dim, heads, kind
        self.head_dim = dim // heads
        self.qkv = _gl(dim, 3 * dim)
        self.out = _gl(dim, dim)
        # 「gated」的那个 gate：逐 head 的标量门（可学，参数量 O(heads)，可忽略）
        self.head_gate = nn.Parameter(torch.zeros(heads))
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:      # QK-Norm：稳定低比特下的注意力打分
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward(self, x: torch.Tensor,
                kv_x: Optional[torch.Tensor] = None,
                q_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`kv_x` 非空时做**联合注意力**：K/V 取自 `kv_x`，Q 取自 `q_x`（默认 x）。"""
        q = self._split(q_x if q_x is not None else x)[0]
        k, v = self._split(kv_x if kv_x is not None else x)[1:]
        o = _attn_core(q, k, v, self.kind, self.head_gate)
        B, H, N, Dh = o.shape
        return self.out(o.transpose(1, 2).reshape(B, N, H * Dh))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 2.5):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = _gl(dim, hidden)
        self.fc2 = _gl(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Q 来自主序列，K/V 来自外部 token（身份 token 用；**不拼主序列**）。"""

    def __init__(self, dim: int, heads: int, ctx_dim: int = None, qk_norm: bool = True):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.heads, self.head_dim = heads, dim // heads
        self.q = _gl(dim, dim)
        self.kv = _gl(ctx_dim, 2 * dim)
        self.out = _gl(dim, dim)
        self.norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        M = ctx.shape[1]
        q = self.q(self.norm(x)).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = k.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        # 身份 token 走 **非归一化** 打分：与主干注意力同族，避免身份浓度被稀释
        o = _attn_core(q, k, v, SIGMOID)
        return self.out(o.transpose(1, 2).reshape(B, N, self.heads * self.head_dim))


class TextLayoutRouter(nn.Module):
    """文本条件 → 固定 K 个 layout token（2D 网格位置编码）。

    ⭐ 主文档补充01 §2.5：「文本条件走独立的 **K=16 个 layout token 路由**」。
       好处：文本长度与图像 token 数解耦；matryoshka 变分辨率时文本侧不变；
       并且主序列长度可预测（便于低显存预算）。
    """

    def __init__(self, text_dim: int, dim: int, n_tokens: int = 16, heads: int = 8):
        super().__init__()
        heads = max(1, min(heads, dim // 8)) if dim >= 8 else 1
        while dim % heads:
            heads -= 1
        self.n_tokens, self.heads = n_tokens, heads
        self.head_dim = dim // heads
        self.in_proj = _gl(text_dim, dim)
        self.queries = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)
        self.q = _gl(dim, dim)
        self.kv = _gl(dim, 2 * dim)
        self.out = _gl(dim, dim)
        self.norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        # layout token 的 2D 网格位置（4×4）
        side = int(math.ceil(math.sqrt(n_tokens)))
        self.register_buffer("layout_pos",
                             sincos_2d(side, side, dim)[:n_tokens], persistent=False)

    def forward(self, text_ctx: torch.Tensor) -> torch.Tensor:
        B = text_ctx.shape[0]
        x = self.in_proj(text_ctx)                       # [B, L, dim] 变长文本
        M = x.shape[1]
        # layout query 自带 2D 网格位置（K=16 个固定槽位，与文本长度解耦）
        queries = self.queries + self.layout_pos
        q = self.q(self.norm(queries)).unsqueeze(0).expand(B, -1, -1)
        q = q.reshape(B, self.n_tokens, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        o = _attn_core(q, k, v, SIGMOID)
        o = o.transpose(1, 2).reshape(B, self.n_tokens, self.heads * self.head_dim)
        return self.out(o) + self.layout_pos.unsqueeze(0)


class DiTBlock(nn.Module):
    def __init__(self, cfg: DiTCfg, kind: str, double_stream: bool,
                 identity_ctx_dim: Optional[int] = None):
        super().__init__()
        d = cfg.dim
        self.dim, self.kind, self.double_stream = d, kind, double_stream
        self.norm1 = RMSNorm(d)
        # ⚠️ 8 段 adaLN —— 第 8 段（下标 7）是身份门控，selftest 直接写它的 bias
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 8 * d))
        self.attn = Attention(d, cfg.heads, kind, cfg.qk_norm)
        # 双流：文本流有**自己的一套 QKV/out**（共享 K/V 做联合注意力）
        self.txt_qkv = _gl(d, 3 * d) if double_stream else None
        self.txt_out = _gl(d, d) if double_stream else None
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, cfg.mlp_ratio)
        self.identity_cross = (CrossAttention(d, cfg.heads, identity_ctx_dim)
                               if identity_ctx_dim is not None else None)
        self._init_adaLN()

    def _init_adaLN(self) -> None:
        """adaLN-Zero：末层 weight 全 0；只把 attn/mlp/txt 三个 gate 的 bias 置 1。"""
        last = self.adaLN[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        d = self.dim
        with torch.no_grad():
            for seg in (2, 5, 6):                 # attn gate / mlp gate / g_txt
                last.bias[seg * d:(seg + 1) * d].fill_(1.0)
            # 段 7（身份门控）保持 0 ⇒ adaLN-Zero 保护
        self.register_buffer("_adaLN_zero", torch.tensor(True), persistent=False)

    def _txt_attn(self, txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """双流路径：文本流自己的 QKV/out，K/V 与图像流联合。"""
        B, Nt, d = txt.shape
        h = self.attn.heads
        dh = self.attn.head_dim
        q = self.txt_qkv(txt).chunk(3, dim=-1)[0]
        q = q.reshape(B, Nt, h, dh).transpose(1, 2)
        k, v = self.attn._split(img)[1:]
        if self.attn.q_norm is not None:
            q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        o = _attn_core(q, k, v, self.kind, self.attn.head_gate)
        return self.txt_out(o.transpose(1, 2).reshape(B, Nt, h * dh))

    def forward(self, img: torch.Tensor, txt: Optional[torch.Tensor],
                c: torch.Tensor, identity_ctx: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        d = self.dim
        mod = self.adaLN(c)                                  # [B, 8d]
        sa, ba, ga = mod[:, 0 * d:1 * d], mod[:, 1 * d:2 * d], mod[:, 2 * d:3 * d]
        sm, bm, gm = mod[:, 3 * d:4 * d], mod[:, 4 * d:5 * d], mod[:, 5 * d:6 * d]
        g_txt = mod[:, 6 * d:7 * d]
        g_id = mod[:, 7 * d:8 * d]

        h = self.norm1(img) * (1.0 + sa.unsqueeze(1)) + ba.unsqueeze(1)
        if txt is None:
            img = img + ga.unsqueeze(1) * self.attn(h)
        elif self.double_stream:
            img = img + ga.unsqueeze(1) * self.attn(h)
            txt = txt + g_txt.unsqueeze(1) * self._txt_attn(txt, h)
        else:
            # 单流：文本与图像拼进同一条序列，共享同一套注意力权重
            seq = torch.cat([txt, h], dim=1)
            out = self.attn(seq)
            n_txt = txt.shape[1]
            txt = txt + g_txt.unsqueeze(1) * out[:, :n_txt]
            img = img + ga.unsqueeze(1) * out[:, n_txt:]

        img = img + gm.unsqueeze(1) * self.mlp(self.norm2(img) * (1.0 + sm.unsqueeze(1))
                                               + bm.unsqueeze(1))
        if self.identity_cross is not None and identity_ctx is not None:
            img = img + g_id.unsqueeze(1) * self.identity_cross(img, identity_ctx)
        return img, txt


class SingleStreamDiT(nn.Module):
    """单流 DiT 主干（KP-S / KP-M 共用实现，差异只在 `DiTCfg`）。"""

    def __init__(self, cfg: Optional[DiTCfg] = None,
                 latent_ch: int = None,
                 identity_anchor_layers: Optional[Sequence[int]] = None,
                 text_dim: Optional[int] = None,
                 identity_dim: Optional[int] = None,
                 domain_dim: int = 16,
                 n_layout_tokens: int = 16,
                 softmax_anchor: bool = True):
        super().__init__()
        cfg = cfg or DiTCfg()
        self.cfg = cfg
        d = cfg.dim
        latent_ch = latent_ch or LATENT.total_ch
        identity_anchor_layers = ([] if identity_anchor_layers is None
                                  else list(identity_anchor_layers))
        self.latent_ch = latent_ch

        self._patch_embed_split = None
        self.patch_embed = None
        if cfg.split_patch_embed and latent_ch == LATENT.total_ch:
            # 🔴 M3 修法②：`patch_embed` 拆两路、**权重不共享**（结构保证，推理期也成立）
            #
            # 动机（2026-10-03 全局审查 + 真图实测）：
            #   原始 latent 测量到 `cross_r2_sem_from_detail = 0.988` ⇒ **两块内容高度冗余**，
            #   而 `branch_dependency = 0.0498` PASS ⇒ **分支行为干净但没有「专属区」**。
            #   ⇒ 写进语义通道的身份信号，在信息上等价于也写进了细节通道
            #      ⇒ 主文档 §6.3 的 M3「身份只走语义 ⇒ 不污染画风」不成立。
            #
            # 为什么拆两路就能修（`kp/probe/m3_fix.py` §修法② 实测）：
            #   两块**不共享权重** ⇒ 投影核对两块是**两个独立的线性函数**，
            #   输出无法互为线性重建 ⇒ 冗余度从 0.9666 掉到 0.0017。
            #
            # ⭐ **参数量恒等（已实测）**：`(8+32)·d = 46,080 ≡ 40·d`（单路 `40→d`）**完全相同**，
            #    只多一次加法 ⇒ **拆路在参数量上免费**。
            # ⚠️ **不要两处都建**（`self.patch_embed` 与 split 两路同时存在 ⇒ 多 46,080 **死参数**）。
            #    ⚠️ 这正是本项目教训「『预留接口』不该用『每层都付钱』的方式存在」的同类问题。
            # ⛔ **默认关闭**（`split_patch_embed=False`）：它改变架构行为，
            #    需在 P2 换主干时作为一道显式决策点拍板；此处先让能力可测。
            self._patch_embed_split = nn.ModuleDict({
                "sem": _gl(LATENT.semantic_ch, d),
                "det": _gl(LATENT.detail_ch, d),
            })
        else:
            self.patch_embed = _gl(latent_ch, d)
        self.t_embed = TimestepEmbedding(d)
        self.domain_embed = nn.Linear(domain_dim, d)
        self.text_dim = text_dim or d
        self.text_router = TextLayoutRouter(self.text_dim, d, n_layout_tokens, cfg.heads)
        if identity_dim is None or identity_dim == d:
            self.identity_proj = None
        else:
            self.identity_proj = _gl(identity_dim, d)

        plan = build_attn_plan(cfg.layers, cfg.ratio_gated_linear, cfg.ratio_sigmoid,
                               softmax_anchor)
        self.blocks = nn.ModuleList([
            DiTBlock(cfg, plan[i], double_stream=(i < cfg.double_stream_blocks),
                     identity_ctx_dim=(d if i in identity_anchor_layers else None))
            for i in range(cfg.layers)
        ])
        self.final_norm = RMSNorm(d)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = _gl(d, latent_ch)         # ⚠️ 名字须含 out_proj（QAD 跳过它）
        self.attn_plan = plan
        self._bus = CapabilityBus(list(self.gated_linears().values()))
        freeze_backbone_(self)

    # --- 能力总线接口（selftest / train.qad 依赖）---
    def gated_linears(self) -> Dict[str, GatedLinear]:
        return {n: m for n, m in self.named_modules() if isinstance(m, GatedLinear)}

    def capability_bus(self) -> CapabilityBus:
        return self._bus

    # --- 前向 ---
    def forward(self, x: torch.Tensor, t: torch.Tensor, *,
                text_ctx: Optional[torch.Tensor] = None,
                identity_ctx: Optional[torch.Tensor] = None,
                domain: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        d = self.cfg.dim
        if self._patch_embed_split is not None:
            # 🔴 M3 修法②：两路独立投影后相加（权重不共享 ⇒ 结构上阻断跨块重建）
            sc = LATENT.semantic_ch
            h = (self._patch_embed_split["sem"](x[:, :sc].flatten(2).transpose(1, 2))
                 + self._patch_embed_split["det"](x[:, sc:].flatten(2).transpose(1, 2)))
        else:
            h = self.patch_embed(x.flatten(2).transpose(1, 2))          # [B, N, d]
        h = h + sincos_2d(H, W, d).to(h.dtype)

        c = self.t_embed(t)
        if domain is not None:
            c = c + self.domain_embed(domain)

        txt = self.text_router(text_ctx) if text_ctx is not None else None
        idc = identity_ctx
        if idc is not None and self.identity_proj is not None:
            idc = self.identity_proj(idc)

        for blk in self.blocks:
            h, txt = blk(h, txt, c, idc)

        mod = self.final_adaLN(c)
        scale, shift = mod.chunk(2, dim=-1)
        h = self.final_norm(h) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        v = self.out_proj(h)                                        # [B, N, C]
        return v.transpose(1, 2).reshape(B, C, H, W)


def freeze_backbone_(model: nn.Module) -> int:
    """构造即冻结：主干是可训参数为 0 的「冻结底座」。

    ⚠️ 与 `train/qad.py::freeze_backbone` 同义 —— 那里是训练前显式再冻一遍。
       GatedLinear 的权重是 **buffer**，本来就不在 `parameters()` 里；
       这里冻的是 adaLN / t-embed / domain-embed / RMSNorm 这些**真参数**。
    """
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += p.numel()
    return n

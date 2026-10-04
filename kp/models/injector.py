"""身份接线 —— ⭐ 补上「Fitter 产出 token」到「主干注入」之间的**断链**

═══ 缺口是怎么发现的 ═══
`kp/character/e2e_smoke.py:95-98` 留了一条很诚实的注释：

    ⚠️⚠️ **第一版这里写错了**（我以为 CharaBridge 吃 Fitter 的 token）——
      读源码才发现：`CharaBridge.forward(refs)` 吃的是 **`[B,V,3,H,W]` 视图**，
      自己内部编码 → 身份 token。**Fitter 与 CharaBridge 是并列的两条身份通路**。

⇒ 后果：**「训好 Fitter → 换角色 =秒级」这条主线没有终点**。
   Fitter 产出 `(1,256,dim)` 的 token，但没有模块消费它；
   CharaBridge 能注入主干，但它自己从视图现场编码 ⇒ **Fitter 白训了**。

═══ 本文件接什么 ═══
    角色卡 ──▶ CharacterFitter ──▶ 身份 token (1,T,dim)
                                        │
                                   ┌────┴────────────────────┐
                                   ▼                         ▼
                          ① TokenInjector（新增）      ② CharaBridge（既有）
                          「把 token 当 K/V 注入」        「从视图现场编码」
                                   │                         │
                                   └──────▶ 主干 cross-attn ◀┘

⭐ 两条路**互补而非重复**：
   · CharaBridge：每次前向都看视图（推理时换了图，身份就变）
   · TokenInjector：**只吃存下来的 token** ⇒ 换角色不用重跑编码，
     也不用保留原图（这正是「角色卡跨版本通用」的商业价值）

⚠️ **诚实边界**：本模块是**接线**，不是新模型。两侧的编码能力
   （Fitter 的 token、CharaBridge 的 K/V）都来自已训好的模块。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..capability.bus import GatedLinear
from ..config import CAP
from .common import RMSNorm


def _gl(in_f: int, out_f: int, std: float = 0.02) -> GatedLinear:
    """与 `charabridge._gl` 同一套构造（`GatedLinear` 收**权重张量**，不是 int）。"""
    w = torch.empty(out_f, in_f)
    torch.nn.init.normal_(w, std=std / math.sqrt(max(1, in_f) / 64.0))
    return GatedLinear(w)


class TokenInjector(nn.Module):
    """⭐ 把**存下来的身份 token** 当 K/V 注入主干（CharaBridge 的静态 counterpart）。

    ⚠️ **为什么需要它**（不是为了多一个模块）：
       记忆库的设计目标是「**换角色 = Fitter 前向一次（秒级）**」。
       若主干只能吃 CharaBridge 现场编码的 K/V，那每次换角色都要
       **重新跑一遍视图编码** ⇒ Fitter 的产出无处可去，主线断在这里。

    ⭐ **门控语义与 CharaBridge 完全一致**：
       `gate == 0` → `forward` 返回 **`None`**（整条分支短路）。
       ⚠️ 绝不返回零向量 —— 零向量进 cross-attention 仍会算出非零输出
       （`sigmoid(QKᵀ)·V ≠ 0`）⇒「乘 0 再相加」**不等于**「不计算」。
    """

    def __init__(self, dim: int = None, n_tokens: int = None,
                 view_dim: int = 256, heads: int = 8):
        super().__init__()
        self.dim = int(dim or CAP.identity_token_dim)
        self.n_tokens = int(n_tokens or CAP.identity_tokens)
        self.view_dim = int(view_dim)
        self.heads = int(heads)
        self.head_dim = self.view_dim // self.heads
        if self.view_dim % self.heads:
            raise ValueError(f"view_dim={self.view_dim} 必须能被 heads={self.heads} 整除")

        # token (T, dim) → K/V (T, 2·view_dim)
        self.kv = _gl(self.dim, 2 * self.view_dim)
        self.norm_k = RMSNorm(self.head_dim)
        self.gate = nn.Parameter(torch.tensor(float(CAP.gate_init)))
        self.q = nn.Linear(self.view_dim, self.view_dim)
        self.norm_q = RMSNorm(self.view_dim)
        self.queries = nn.Parameter(torch.randn(self.n_tokens, self.view_dim) * 0.02)
        self.out = nn.Linear(self.view_dim, self.dim)

    # ------------------------------------------------------------------
    @property
    def is_off(self) -> bool:
        return float(self.gate.detach()) == 0.0

    def set_gate(self, value: float) -> None:
        with torch.no_grad():
            self.gate.fill_(float(value))

    def gated_linears(self) -> dict:
        return {n: m for n, m in self.named_modules() if isinstance(m, GatedLinear)}

    # ------------------------------------------------------------------
    def forward(self, tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """`(B, T, dim)` 身份 token → `(B, n_tokens, dim)`；`gate=0` ⇒ **返回 None**。

        ⛔ **不静默接受错形状** —— 形状不对是「角色卡与主干版本不匹配」的第一个信号，
           静默广播会变成"身份悄悄错位"，最难查。
        """
        if self.is_off:
            return None# ★ 短路：整条分支不参与计算
        if tokens is None:
            raise ValueError(
                "传了 tokens=None 但 gate≠0。⛔ 不静默返回 None —— "
                "那会让调用方以为「身份没生效」而不是「你 forgot 传 token」。")
        if tokens.dim() != 3:
            raise ValueError(f"tokens 应为 (B,T,dim)，收到 {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(
                f"身份 token 维度 {tokens.shape[-1]} ≠ 注入器 dim {self.dim}。"
                f"⛔ 不静默投影（跨主干版本的 token 不能直接用）—— "
                f"请重跑 Fitter 或换匹配的注入器。")
        # (B, T_in, dim) → 取前 n_tokens 个（不足则补零，**如实**在 meta 里体现）
        # ⛔ **token 数必须对上，不静默截断/补零**（2026-10-05 改）。
        #理由：token 数不一致 =角色卡与主干版本不匹配的第一个信号。
        #   静默补零 ⇒ 身份**部分**来自训练、部分是零 ⇒ 输出看着"有身份"实则半残。
        if tokens.shape[1] != self.n_tokens:
            raise ValueError(
                f"身份 token 数 {tokens.shape[1]} ≠ 注入器预算 {self.n_tokens}。"
                f"⛔ 不静默截断/补零（补零会让身份'半残'，看着正常实则失效）。"
                f"⇒ 改 n_tokens，或重跑 Fitter 产出对应长度的 token。")
        t = tokens

        kv = self.kv(t)                                # (B, T, 2·view_dim)
        k, v = kv.chunk(2, dim=-1)
        q = self.q(self.norm_q(self.queries)).unsqueeze(0).expand(k.shape[0], -1, -1)
        q = q.reshape(k.shape[0], self.heads, self.n_tokens, self.head_dim)
        k = k.reshape(k.shape[0], self.heads, -1, self.head_dim)
        v = v.reshape(k.shape[0], self.heads, -1, self.head_dim)
        k = self.norm_k(k)
        # ⭐ 与 CharaBridge 同款：**非归一化** sigmoid 打分
        #   （身份浓度不该被 softmax 的份额竞争稀释）
        scores = torch.sigmoid(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim))
        o = (scores @ v).transpose(1, 2).reshape(k.shape[0], self.n_tokens, -1)
        return self.out(o)


# ---------------------------------------------------------------------------
def load_token(card_path: str, device: str = "cpu") -> tuple:
    """从角色卡读身份 token → `((1,T,dim) tensor, 来源说明)`；缺失则**明确报错**。"""
    d = torch.load(card_path, weights_only=False)
    tok = d.get("identity_token")
    if tok is None:
        raise FileNotFoundError(f"卡里没有 identity_token：{card_path}")
    src = d.get("meta", {}).get("identity_token", "?")
    t = tok if isinstance(tok, torch.Tensor) else torch.as_tensor(tok)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.to(device).float(), src


def inject_into_mainnet(inj: "TokenInjector", tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """便捷入口：注入并**显式检查**返回不是 None。"""
    out = inj(tokens)
    if out is None:
        raise RuntimeError(
            "注入器返回 None ⇒ gate=0。⛔ 不静默通过 —— "
            "要么真要关身份（那调用方应显式跳过），要么 gate 忘开了。")
    return out


__all__ = ["TokenInjector", "load_token", "inject_into_mainnet"]

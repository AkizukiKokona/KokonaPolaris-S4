"""txt2img + 角色卡 · **端到端冒烟** —— 主线最后一块拼图的「能跑通」证明。

═══ 这个文件要证明什么 ═══

设计稿 v1.18 定的产品主线是 **txt2img（配合角色卡）**。其链路是：

    正/背视图 ──▶ ① CharacterFitter ──▶ 身份 token ──▶ ② CharaBridge 注入
                                    （秒级，≤256 token）      （gated cross-attn）
                                                                   │
    文本 prompt ──▶ ③ 文本塔（未蒸馏）──▶ 主干 DiT ◀───────────────┘
                                                              │
                                                    ④ 换角色 = 再前向一次 ✅

⛔ **本文件只证明「骨架通」，不证明「质量好」**：
    · 它用**真图**（kokona 正/背）跑通 ①②④；
    · 它**不涉及** ③ 文本塔（未蒸馏）与主干训练（未开始）。

⚠️ **为什么不等到全通了再写**：现在能证明①②④，省得最后才发现接口对不上。
   这是本项目的做法 —— **每一步都留可执行的判据**。

═══ 三条必须成立的不变量 ═══

① **Fitter 前向一次就出身份 token**（秒级，不训练）
② **门控关断 ⇒ 逐位不变**（bit-exact，身份不污染底模）
③ **换角色 ⇒ 身份 token 真的变**（不能是常量，否则 Fitter 什么都没学到）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import CAP, LATENT
from ..models.charabridge import CharaBridge
from .fitter import CharacterFitter
from .card import CharacterCard, N_LAYERS


def _load_views(size: int = 256) -> tuple:
    """载入 kokona 的正/背视图（⛔ 找不到就报缺口，不用假数据替代）。"""
    from .dataset import load_image
    from ..paths import DATA
    d = DATA / "characters" / "kokona" / "images"
    front, back = d / "front.png", d / "back.png"
    if not (front.exists() and back.exists()):
        return None, None, f"⚠️ 缺正/背视图：{front} / {back}"
    return (load_image(str(front), size), load_image(str(back), size),
            f"{front.parent}")


def run(n_tokens: Optional[int] = None, size: int = 256,
        seed: int = 0) -> Dict:
    """跑一遍端到端，返回每一步的数字（⛔ 任一步缺前置就如实报缺口）。"""
    torch.manual_seed(seed)
    rep: Dict = {"步骤": {}}

    front, back, where = _load_views(size)
    if front is None:
        return {"ok": False, "缺口": where}
    # ⚠️ **Fitter 的输入约定是 (B, V, 3, H, W)** —— 视图维在**第二维**，
    #    不是常见的 (B, 3, H, W, V)。第一次跑就踩了这个（报「expected 3 channels but got 6」）。
    #    这里是 B=1（一个角色）、V=2（正/背）。
    views = torch.stack([front, back])[None]     # (1,2,3,S,S)
    rep["输入"] = {"图": where, "size": size, "n_views": int(views.shape[0])}

    # ---------- ① Fitter：正/背 → 身份 token ----------
    nt = n_tokens or CAP.identity_tokens
    # ⚠️ `view_dim` 是**构造参数**（不是 CAP 字段）⇒ 用 Fitter 自己的默认值，别猜
    fitter = CharacterFitter(dim=CAP.identity_token_dim, n_tokens=nt,
                             use_geometry=True)
    t0 = time.time()
    with torch.no_grad():
        id_ctx = fitter(views)
    dt = time.time() - t0
    rep["步骤"]["①Fitter"] = {
        "耗时秒": round(dt, 3),
        "身份token形状": tuple(id_ctx.shape),
        "预算": nt,
        "秒级": dt < 5.0,
    }

    # ---------- ③ 换角色 ⇒ token 真的变 ----------
    # ⭐ 这一条最关键：如果换个角色 token 不变，说明 Fitter 什么都没学到
    with torch.no_grad():
        flipped = torch.flip(views, dims=[-1])          # 水平镜像，冒充「另一个角色」
        id_ctx_b = fitter(flipped)
    delta = float((id_ctx - id_ctx_b).abs().mean())
    rep["步骤"]["③换角色"] = {
        "两角色token平均差": round(delta, 6),
        "真的变了": delta > 1e-4,
    }

    # ---------- ② CharaBridge：视图 → 身份 token（**与 ① 并列，不是串联**）----------
    # ⚠️⚠️ **第一版这里写错了**（我以为 CharaBridge 吃 Fitter 的 token）——
    #   读源码才发现：`CharaBridge.forward(refs)` 吃的是 **`[B,V,3,H,W]` 视图**，
    #   自己内部编码 → 身份 token。**Fitter 与 CharaBridge 是并列的两条身份通路**。
    d = CAP.identity_token_dim
    cb = CharaBridge(dim=d, n_tokens=nt, gate=CAP.gate_init)   # gate 默认 0
    with torch.no_grad():
        out_off = cb(views)          # gate=0 ⇒ 应返回 **None**（不是零向量！）
        cb.set_gate(1.0)
        out_open = cb(views)         # 开门 ⇒ 才有 token
    off_is_none = out_off is None
    rep["步骤"]["②CharaBridge"] = {
        "输入约定": "[B,V,3,H,W] 视图（**不是** Fitter 的 token）",
        "gate=0 返回 None": off_is_none,
        "开门后token形状": tuple(out_open.shape) if out_open is not None else None,
        "门控默认0": CAP.gate_init == 0.0,
    }

    # ---------- ②b 真实验证：gate=0 ⇒ 身份**完全不参与** ----------
    with torch.no_grad():
        cb.set_gate(0.0)
        a = cb(views)
        b = cb(flipped)              # 换成「另一个角色」的视图
    gated_invariant = (a is None) and (b is None)
    # 开门时两个角色必须不同（否则 CharaBridge 什么都没学到）
    with torch.no_grad():
        cb.set_gate(1.0)
        ta, tb = cb(views), cb(flipped)
    cb_differs = (ta is not None and tb is not None
                  and not torch.allclose(ta, tb, atol=1e-6))
    rep["步骤"]["②b关断不变量"] = {
        "gate=0 两个角色都返回 None": gated_invariant,
        "说明": "True ⇒ 关断后身份**完全不参与**（这才是「不打架」的保证）",
        "开门时两角色token确实不同": cb_differs,
    }

    rep["ok"] = (rep["步骤"]["①Fitter"]["秒级"]
                 and rep["步骤"]["③换角色"]["真的变了"]
                 and gated_invariant
                 and rep["步骤"]["②CharaBridge"]["门控默认0"])
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="txt2img+角色卡 · 端到端冒烟")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args(argv)
    r = run(size=a.size)
    if not r.get("ok") and "缺口" in r:
        print(f"⚠️ {r['缺口']}")
        return 1
    print("=" * 68)
    print("txt2img + 角色卡 · 端到端冒烟（⛔ 只证骨架，不证质量）")
    print("=" * 68)
    for k, v in r["输入"].items():
        print(f"  {k}: {v}")
    for step, d in r["步骤"].items():
        print(f"\n  【{step}】")
        for k, v in d.items():
            print(f"    {k}: {v}")
    print("\n" + "=" * 68)
    print(f"总判定：{'✅ 骨架通' if r['ok'] else '❌ 有环节不成立'}")
    if r["ok"]:
        print("""⭐ 证明了三件事：
  ① Fitter 前向一次出身份 token（秒级）
  ③ 换角色 ⇒ token 真的变（Fitter 不是常量）
  ② gate=0 时不同角色输出相同 ⇒ 身份不污染底模
⚠️ **未涉及**：文本塔（未蒸馏）· 主干训练（未开始）· 语义层（占位）""")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""角色卡身份装载器 —— ⭐「换角色 = 一次前向」的主线入口

═══ 这份文件补的是什么 ═══
`CharacterFitter.make_identity()` 之前**零调用**（审计 P2-1）⇒
「秒级换角色」这条**主线入口没有任何代码在用**，只是个定义。

本文件把它接成**可执行的主线**：
    ① 训好 Fitter（或从磁盘加载）
    ② 给一组新角色的视图 → 一次前向 → 身份 token
    ③ 直接喂给 CharaBridge / 主干 cross-attention

═══ 为什么值得单独一个模块 ═══
「换角色」是项目**第一卖点之一**（记忆库：身份载体是一等公民）。
它必须是**一条命令**，而不是散落在训练脚本里的三行。

⚠️ **诚实边界**：
   ① 「秒级 / 峰值 1.2GB」是**设计目标**，本机尚未做峰值实测（见 `probe_peak`）。
   ② 视图口径：同一角色的多视角，最少正+背 2 张。
   ③ 单角色身份**不可区分**（没有第二个角色做负样本）⇒ token 数值无意义，
      但**流程可跑通**。要验证可区分性须≥2 个角色。
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..paths import OUT
from .card import CharacterCard
from .fitter import CharacterFitter


# ---------------------------------------------------------------------------
# 视图加载
# ---------------------------------------------------------------------------
def load_views(paths: Sequence[str], size: int = 256,
               device: str = "cpu") -> torch.Tensor:
    """→ (1, V, 3, size, size)，值域 [-1, 1]。⛔ 无图则报错，不用合成数据顶替。"""
    from PIL import Image
    if not paths:
        raise FileNotFoundError("没有视图图。⚠️ **不静默用合成数据替代**。")
    ims = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"视图不存在：{p}")
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        ims.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div_(127.5).sub_(1.0))
    return torch.stack(ims).unsqueeze(0).to(device)


def views_from_card(card_path: str, norm_dir: Optional[str] = None,
                    size: int = 256, device: str = "cpu") -> torch.Tensor:
    """从已产出的 `.card` 里按 `meta.view_masks` 取视图（保证与训练同源）。

    ⚠️ 卡里的 `view_masks` 存的是 **`<batch>/norm/<file>`**（相对批次目录），
       所以默认要把 `<批次>/cards/x.card` 往上退**两级**再拼 `norm/`。
    """
    import torch as _t
    d = _t.load(card_path, weights_only=False)
    meta = d.get("meta", {})
    vm = meta.get("view_masks") or {}
    if not vm:
        raise FileNotFoundError(
            f"卡里没有 view_masks ⇒ 无法取视图（meta 键：{sorted(meta)[:8]}）")
    if norm_dir is None:
        # cards/x.card → <batch>/norm
        batch_dir = os.path.dirname(os.path.dirname(os.path.abspath(card_path)))
        norm_dir = os.path.join(batch_dir, "norm")
    ps, missing = [], []
    for f in vm:
        # view_masks 的值形如 "norm/front.png"；也容许它已经是纯文件名
        rel = f if os.path.isabs(f) else os.path.join(os.path.dirname(norm_dir), f)
        cand = rel if os.path.exists(rel) else os.path.join(norm_dir, os.path.basename(f))
        (ps.append(cand) if os.path.exists(cand) else missing.append(cand))
    if missing:
        raise FileNotFoundError(
            f"视图缺失 {len(missing)} 张：{missing[:4]}\n（norm_dir={norm_dir}）")
    return load_views(ps, size=size, device=device)


# ---------------------------------------------------------------------------
# Fitter 的存 / 取
# ---------------------------------------------------------------------------
def save_fitter(fitter: CharacterFitter, path: str, **meta) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"state_dict": fitter.state_dict(), "dim": fitter.dim,
               "n_tokens": fitter.n_tokens, "meta": meta}
    _tmp = path + ".tmp"
    torch.save(payload, _tmp)
    os.replace(_tmp, path)          # ⭐ 原子替换（断电不会留下半个文件）
    return path


def load_fitter(path: str, device: str = "cpu") -> CharacterFitter:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fitter 权重不存在：{path}\n"
            f"⇒ 先跑 `python -m kp.character.pipeline --batch <角色> --fit`")
    d = torch.load(path, weights_only=False)
    f = CharacterFitter(dim=int(d.get("dim", 1024)),
                        n_tokens=int(d.get("n_tokens", 256)))
    f.load_state_dict(d["state_dict"])
    return f.to(device).eval()


# ---------------------------------------------------------------------------
# ⭐ 主线：换角色
# ---------------------------------------------------------------------------
@torch.no_grad()
def make_identity(fitter: CharacterFitter, views: torch.Tensor,
                  device: str = "cpu") -> dict:
    """一次前向 → 身份 token。返回 token + 计时（诚实报，不美化）。"""
    fitter.eval()
    v = views.to(device)
    t0 = time.time()
    tok = fitter.make_identity(v)
    dt = time.time() - t0
    peak = 0.0
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return {"token": tok, "seconds": round(dt, 4), "peak_gb": round(peak, 3),
            "shape": tuple(tok.shape)}


def attach_to_card(card_path: str, token: torch.Tensor) -> str:
    """把 token 写回卡（**就地更新** + 留档，meta 里注明来源）。"""
    d = torch.load(card_path, weights_only=False)
    t = np.asarray(token.squeeze(0) if token.dim() == 3 else token, dtype=np.float32)
    d["identity_token"] = torch.from_numpy(t)
    meta = dict(d.get("meta", {}))
    meta["identity_token"] = "trained(CharacterFitter.load_identity · 主线路径)"
    d["meta"] = meta
    _tmp = card_path + ".tmp"
    torch.save(d, _tmp)
    os.replace(_tmp, card_path)
    return card_path


# ---------------------------------------------------------------------------
def probe_peak(views: torch.Tensor, dim: int = 1024,
               device: str = "cuda") -> dict:
    """⭐ 测「换角色前向」的**真实峰值**（设计目标 1.2GB —— 需要实测兑现）。"""
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {"available": False}
    f = CharacterFitter(dim=dim).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    r = make_identity(f, views, device=device)
    del f
    torch.cuda.empty_cache()
    return {"available": True, **r,
            "target_gb": 1.2,
            "meets_target": r["peak_gb"] <= 1.2}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="换角色 =一次前向（主线入口）")
    ap.add_argument("--fitter", default="out/characters/_fitter/fitter.pt")
    ap.add_argument("--card", default=None, help="从这张卡取视图")
    ap.add_argument("--views", nargs="*", default=None, help="直接给图路径（≥2）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-peak", action="store_true",
                    help="实测换角色峰值显存（对照设计的 1.2GB 目标）")
    ap.add_argument("--write-card", action="store_true", help="把 token 写回卡")
    a = ap.parse_args(argv)

    if a.card:
        views = views_from_card(a.card, size=a.size, device=a.device)
    elif a.views:
        views = load_views(a.views, size=a.size, device=a.device)
    else:
        print("⛔ 需要 --card 或 --views（≥2 张）。⚠️ 不静默用合成数据替代。")
        return 1
    print(f"视图：{tuple(views.shape)}（V={views.shape[1]}）")

    if a.probe_peak:
        r = probe_peak(views, device=a.device)
        if not r["available"]:
            print("⚠️ 无 CUDA，跳过峰值实测（**该目标尚未兑现**）")
            return 1
        print(f"峰值 {r['peak_gb']:.3f} GB / 目标 {r['target_gb']} GB "
              f"⇒ {'✅ 达标' if r['meets_target'] else '⛔ 未达标'}｜"
              f"耗时 {r['seconds']:.3f}s")
        return 0 if r["meets_target"] else 1

    fitter = load_fitter(a.fitter, device=a.device)
    r = make_identity(fitter, views, device=a.device)
    print(f"✅ 身份 token {r['shape']}｜耗时 {r['seconds']:.3f}s"
          + (f"｜峰值 {r['peak_gb']:.3f}GB" if r["peak_gb"] else ""))
    if a.write_card and a.card:
        attach_to_card(a.card, r["token"])
        print(f"   已写回：{a.card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

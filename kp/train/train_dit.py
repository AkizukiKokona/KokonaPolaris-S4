"""P2 · 训练 KP 自己的单流 DiT 主干 —— ⭐ 用 DC-AE 的 latent（32ch, 32×）

═══ 为什么现在能做了 ═══
2026-10-05：VAE 改用 **DC-AE f32c32**（已训好，20.33dB，零成本）
⇒ **VAE 不再是卡点** ⇒ 可以开始训主干了。

═══ 这一步的目标（用户判据）═══
① **不 OOM**（8GB 铁律）  ② **图不崩**（用户看图）
⛔ 不用 rFID 门（那个参照系已被判定不成立）

═══ Rectified Flow（设计稿的配方）═══
    x_t = (1-t)·x0 + t·ε          （线性插值）
    v*  = ε - x0                  （目标速度）
    loss = ‖v_θ(x_t, t) - v*‖²
    t ~ logit-normal（设计稿 §4.1 指定，不是 uniform）

═══ ⚠️ 第一版先做「无条件」═══
文本塔还没训 ⇒ 先用**零条件**跑通**无条件生成**。
⭐ 这是最快到"第一张 KP 自己的图"的路径；有图之后再挂文本。
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"
sys.path.insert(0, str(KP_ROOT / "vendor" / "efficientvit"))


# ---------------------------------------------------------------------------
def load_dcae(dev: torch.device):
    """⭐ 用 MIT 原版 efficientvit 加载（⛔ diffusers 命名不兼容 ⇒ 会静默随机权重）。"""
    from efficientvit.models.efficientvit.dc_ae import dc_ae_f32c32, DCAE
    from safetensors.torch import load_file
    cfg = dc_ae_f32c32("dc-ae-f32c32-sana-1.0", None)
    m = DCAE(cfg)
    sd = load_file(str(KP_ROOT / "models" / "dc_ae_f32c32_sana_1.0.safetensors"))
    miss, unexp = m.load_state_dict(sd, strict=False)
    # ⛔ 必须检查！今天踩过"没报错但全是随机权重"
    if len(miss) > 10 or len(unexp) > 10:
        raise RuntimeError(f"DC-AE 权重没加载好: missing={len(miss)} unexp={len(unexp)}")
    return m.eval().to(dev).float(), float(cfg.scaling_factor)


def read_raw(shards: int, limit: int) -> List[bytes]:
    import pyarrow.parquet as pq
    out: List[bytes] = []
    for p in sorted(SHARD_DIR.glob("*.parquet"))[:shards]:
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=128, columns=["image"]):
            for r in b.to_pylist():
                out.append(r["image"])
                if len(out) >= limit:
                    return out
    return out


def prep(b: bytes, size: int):
    import numpy as np
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:                                              # noqa: BLE001
        return None
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im, dtype="uint8")
    return torch.from_numpy(a.copy()).float().permute(2, 0, 1) / 127.5 - 1.0


def make_backbone_trainable(model) -> int:
    """⭐ 把主干的 `weight`/`bias` 从 **buffer 提升为 Parameter**。

    ═══ 为什么必须做（2026-10-05 实测踩到）═══
    KP 的 `GatedLinear` 把权重注册成 **buffer** —— 那是为
    「**冻结主干** + 挂能力包」设计的。后果：
        `model.parameters()` **拿不到主干权重**
        ⇒ 实测 `可训参数量 = 0.00M`，backward 直接报
          `RuntimeError: element 0 of tensors does not require grad`
    ⭐ 预训练阶段必须先把它们提升为 Parameter。

    ⛔ 只动名字是 `weight`/`bias` 的 buffer（真权重）；
       `layout_pos` / RoPE 表这类**按名字注册的**不受影响。
    """
    import torch.nn as nn
    n = 0
    for mod in model.modules():
        for attr in ("weight", "bias"):
            if attr in mod._buffers and mod._buffers[attr] is not None:
                buf = mod._buffers.pop(attr)
                mod.register_parameter(attr, nn.Parameter(buf.detach().clone()))
                n += 1
    return n


def logit_normal_t(n: int, dev, m: float = 0.0, s: float = 1.0):
    """⭐ 设计稿指定：t ~ logit-normal（不是 uniform）。"""
    z = torch.randn(n, device=dev) * s + m
    return torch.sigmoid(z)


# ---------------------------------------------------------------------------
def train(a) -> int:
    from kp.models.dit import SingleStreamDiT, DiTCfg
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae, scale = load_dcae(dev)
    print(f"[*] DC-AE ok  scaling_factor={scale}", flush=True)

    # ⭐⭐ latent 缓存（关键提速）
    #   实测：解 JPEG 占了 1.35s/步 的绝大部分（GPU 在等 CPU）
    #   而 latent 只有 32×8×8×4B = 8KB/张 ⇒ 40000 张仅 328MB，RAM 装得下
    cache_path = KP_ROOT / "out" / "dit" / f"_latents_{a.size}_n{a.max_images}.pt"
    latents = None
    if a.latents and Path(a.latents).exists():
        latents = torch.load(a.latents, map_location="cpu", weights_only=False)["z"]
        print(f"[*] loaded cached latents {tuple(latents.shape)}", flush=True)
    elif cache_path.exists():
        latents = torch.load(cache_path, map_location="cpu",
                             weights_only=False)["z"]
        print(f"[*] loaded cached latents {tuple(latents.shape)}", flush=True)

    if latents is None:
        raw = read_raw(a.shards, a.max_images)
        print(f"[*] {len(raw)} raw imgs -> encoding latents...", flush=True)
        zs: List[torch.Tensor] = []
        t_enc = time.time()
        for i in range(0, len(raw), 16):
            chunk = [prep(b, a.size) for b in raw[i:i + 16]]
            chunk = [c for c in chunk if c is not None]
            if not chunk:
                continue
            xb = torch.stack(chunk).to(dev)
            with torch.no_grad():
                zz = vae.encode(xb)
                if not torch.is_tensor(zz):
                    zz = zz[0]
            zs.append((zz.float() * scale).cpu())
            if (i // 16) % 50 == 0:
                print(f"    {i}/{len(raw)}  {time.time()-t_enc:.0f}s", flush=True)
        latents = torch.cat(zs, dim=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"z": latents, "scale": scale, "size": a.size}, cache_path)
        print(f"[*] latents cached: {tuple(latents.shape)} "
              f"-> {cache_path.name} ({time.time()-t_enc:.0f}s)", flush=True)
        del raw, zs
        raw = None

    # ⭐ 文本条件（用户 2026-10-06 拍板：先挂文本塔）
    temb = tmask = None
    if a.text_emb and Path(a.text_emb).exists():
        _t = torch.load(a.text_emb, map_location="cpu", weights_only=False)
        temb = _t["emb"].to(torch.float32)              # [N, L, 2560]
        tmask = _t["mask"]
        print(f"[*] text emb {tuple(temb.shape)}  "
              f"(teacher={_t.get(chr(116)+'eacher')}, layers={_t.get('layers')})",
              flush=True)
    text_dim = int(temb.shape[-1]) if temb is not None else None

    cfg = DiTCfg(dim=a.dim, layers=a.layers, heads=a.heads)
    model = SingleStreamDiT(cfg, latent_ch=32, text_dim=text_dim).to(dev)
    # ⭐ 关键：把 buffer 权重提升为 Parameter，否则**梯度进不去**（实测踩过）
    _promoted = make_backbone_trainable(model)
    # ⭐ 断点续训（IDE 前台任务常被超时打断 ⇒ 没有它就全白跑）
    if a.resume and Path(a.resume).exists():
        _b = torch.load(a.resume, map_location="cpu", weights_only=False)
        _sd = _b.get("state_dict") or _b
        model.load_state_dict(_sd)
        print(f"[*] resumed from {a.resume} "
              f"(step={(_b.get('report') or {}).get('step')})", flush=True)
    else:
        print(f"[*] fresh init (no --resume)", flush=True)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"[*] promoted {_promoted} buffers -> Parameters", flush=True)
    assert n_par > 1e5, f"可训参数只有 {n_par} ⇒ 权重没被提升，训不了"
    print(f"[*] DiT params = {n_par/1e6:.1f}M  dim={a.dim} layers={a.layers}",
          flush=True)

    # ⭐⭐ **分组学习率**（2026-10-06 实测必需）
    # 【实测到的病理】训练后TextRouter 的 out/kv 权重 std 只有 **0.0095**
    #   —— 几乎等于它的**初始化值** `0.02/sqrt(dim/64)` ≈ 0.009
    #   ⛔ 也就是说「文本→输出」这条通路的权重**根本没长起来**。
    # 【根因】AdamW 是按参数的**相对幅度**更新，而 TextRouter 从零学，
    #   主干此时已收敛到loss 平台 ⇒ 用同一个 lr + 训6000 步仍没动。
    # 【业界标准做法】新模块（text_router / adaLN）用**更大的 lr**，
    #   预训练主干用较小的 lr 微调。
    _router, _backbone = [], []
    for _n, _q in model.named_parameters():
        (_router if ('text_router' in _n or 'domain_embed' in _n)
         else _backbone).append(_q)
    opt = torch.optim.AdamW(
        [{"params": _backbone, "lr": a.lr, "weight_decay": 0.0},
         {"params": _router, "lr": a.lr * a.router_lr_mult,
          "weight_decay": a.router_wd}],
    )
    print(f"[*] opt groups: backbone {len(_backbone)} tensors @lr={a.lr} "
          f"| router/adaLN {len(_router)} tensors @lr={a.lr * a.router_lr_mult:g} "
          f"(×{a.router_lr_mult})", flush=True)
    assert _router, "没找到 text_router 参数⇒ 分组没生效"
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lat_hw = a.size // 32
    print(f"[*] latent = 32ch × {lat_hw}×{lat_hw} = {lat_hw*lat_hw} tokens",
          flush=True)

    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    hist: List[float] = []
    t0 = time.time()
    step = 0
    # ⚠️ 两个缓存的索引必须**同一套顺序**（都取自 shard0 的前缀）⇒ 天然对齐
    N = latents.shape[0]
    if temb is not None:
        N = min(N, temb.shape[0])
        print(f"[*] usable pairs = {N} (latent ∩ text)", flush=True)
    while step < a.steps:
        idx = torch.randint(0, N, (a.batch,), generator=g)
        z0 = latents[idx.long()].to(dev).float()          # ⭐ 直接取 latent
        txt = temb[idx.long()].to(dev) if temb is not None else None
        b = z0.shape[0]
        # ⭐⭐ **条件 dropout**（2026-10-06 新增，实测是让文本生效的另一半关键）
        #   随机把 p_drop 比例的样本改成「无条件」⇒ 模型被迫学「有条件 vs 无条件」
        #   ⭐ 这是 **classifier-free guidance（CFG）** 的训练侧。
        #   为什么必需（实测）：
        #     只做对比损失 ⇒ 单步信号/偏移 209%，但采样信号/基线只有 6%
        #     ⇒ 原因：模型只在「匹配文本」这一个模态上学，没有「无文本」这条对照
        #     ⇒ 推理时模型永远在「有文本」侧，文本差异体现不出来。
        #   ⛔ 关掉它就退化成"常量偏置"（实测 0.3% 区分度）。
        if txt is not None and a.p_drop > 0:
            # ⚠️ 不能给 cuda 张量传 cpu generator（实测 RuntimeError）
            keep = (torch.rand(b, device=dev) >= a.p_drop)
            txt = torch.where(keep.view(-1, 1, 1), txt,
                              torch.zeros_like(txt))
            # ⚠️ 注意：这里给的是**零向量**而不是 None
            #    因为 None 会让模型走"完全不算文本"分支（见 design 的
            #    「可关断 = 返回 None」原则）；训练时要的是"零 embedding"
            #    —— 让模型见到「文本通道存在但内容为零」这种输入。
        noise = torch.randn_like(z0)
        t = logit_normal_t(b, dev)
        t_ = t.view(b, 1, 1, 1)
        xt = (1 - t_) * z0 + t_ * noise
        v_target = noise - z0

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            v_pred = model(xt, t, text_ctx=txt)      # ⭐ 带文本条件
            if not torch.is_tensor(v_pred):
                v_pred = v_pred[0]
            loss = F.mse_loss(v_pred.float(), v_target.float())

            # ═══════════════════════════════════════════════════════════════
            # ⭐⭐⭐ 条件对比损失（2026-10-06 新增，实测必需）
            # ═══════════════════════════════════════════════════════════════
            # 【实测到的病理现象】
            #   8000 对 / 80M 模型 / 5000 步之后：
            #       文本 vs 无文本 差异 = 0.212   （模型确实在用文本）
            #       文本A vs 文本B 差异 = 0.010   （⛔ 只占 1%！）
            #   ⛔ 模型学成了**常量偏置**：「有文本」⇒ 给一个固定偏移，
            #      但完全不知道「文本说了什么」。
            # 【根因】flow matching 的 L2/MSE 损失里，**图像项本身就能压低loss**
            #   ⇒ 走「忽略文本」这条捷径的代价很小
            #   ⇒ 条件信号在梯度里被淹没了。
            # 【业界标准解法】**classifier-free-style 条件对比**：
            #   把同一个 xt 配**错误的文本**再前向一次，
            #   要求「错配时的输出」离目标**更远**。
            #   ⇒ 显式给「文本是否匹配」一个梯度。
            if a.cls_w > 0 and temb is not None:
                # roll 一个固定偏移 ⇒ 拿到确定错配的文本（batch 必须足够大）
                shift = max(1, b // 4)
                perm = torch.roll(torch.arange(b, device=dev), shifts=shift)
                txt_bad = txt[perm]
                v_bad = model(xt, t, text_ctx=txt_bad)
                if not torch.is_tensor(v_bad):
                    v_bad = v_bad[0]
                # hinge：正确配对的前向要更接近目标（margin 拉开）
                d_pos = (v_pred.float() - v_target.float()).pow(2).mean()
                d_neg = (v_bad.float() - v_target.float()).pow(2).mean()
                cls_loss = F.relu(d_pos - d_neg + a.cls_margin)
                loss = loss + a.cls_w * cls_loss
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        hist.append(float(loss.detach()))
        step += 1
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t0
            sm = sum(hist[-50:]) / len(hist[-50:])
            mem = torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0
            print(f"[{step:5d}/{a.steps}] loss={sm:.4f} {el:.0f}s "
                  f"{mem:.2f}GB", flush=True)
        if a.save_every and step % a.save_every == 0 and step < a.steps:
            # ⭐⭐ 2026-10-06 修两个真bug（都导致「中间权重用不了」）：
            #   ① **缺 text_dim** ⇒ 有文本条件时 text_router 形状不匹配
            #      ⇒ load_state_dict 直接报 RuntimeError（实测踩过）
            #   ② **文件名不含规模** ⇒ 35M 和 80M 的中间权重互相覆盖
            #      ⇒ resume 时加载到错尺寸的权重（实测踩过）
            _out = (KP_ROOT / "out" / "dit"
                    / f"_ckpt_d{a.dim}L{a.layers}"
                      f"{'_t' + str(text_dim) if text_dim else ''}.pt")
            _out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "config": {"dim": a.dim, "layers": a.layers,
                                   "heads": a.heads, "latent_ch": 32,
                                   "size": a.size, "text_dim": text_dim},
                        "step": step,
                        "cls_w": a.cls_w}, _out)

    out = KP_ROOT / "out" / "dit" / f"d{a.dim}L{a.layers}_s{a.steps}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"dim": a.dim, "layers": a.layers, "heads": a.heads,
                           "latent_ch": 32, "size": a.size, "scale": scale,
                           "text_dim": text_dim},
                "report": {"steps": a.steps, "final_loss": hist[-1],
                           "best_loss": min(hist), "n_images": int(N),
                           "elapsed": round(time.time() - t0, 1),
                           "params_M": round(n_par / 1e6, 1),
                           "text_emb": a.text_emb, "text_dim": text_dim,
                           "⚠️": "已挂教师文本条件（文本塔本身待蒸馏）"}}, out)
    print(f"[OK] saved {out}  final={hist[-1]:.4f}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P2 · 训 KP 单流 DiT（无条件第一版）")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=40000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    # ⭐ 2026-10-08：默认改 **0.1**（业界标准：SD 默认 0.1，区间 0.1-0.2；
    #   4 个独立来源一致 + 我们实测 0.15 可用）。设为 0 会让 CFG 采样形同虚设。
    ap.add_argument("--p-drop", type=float, default=0.1,
                    help="条件 dropout 比例（⭐ CFG 训练侧，业界标准 0.1）")
    #⭐⭐ 2026-10-08 审查：默认关（原默认也是 0，这里写清原因）
    #   「条件对比损失」在本次联网检索的所有业界来源中**均无对应做法**；
    #   它确实有效（信号/基线 1%→6%）但会让 loss 从 1.37涨到 1.9。
    #   ⭐ 业界的标准解法是 **CFG dropout（--p-drop 0.1）**，见设计稿附录 N。
    #   ⇒ 留作实验开关，不当默认配方。
    ap.add_argument("--cls-w", type=float, default=0.0,
                    help="条件对比损失（非标准配方，实测有正向作用但会让 loss 变差；"
                         "默认关。标准解法是 --p-drop")
    ap.add_argument("--cls-margin", type=float, default=0.05,
                    help="hinge margin（正配对比错配至少要近这么多）")
    ap.add_argument("--router-lr-mult", type=float, default=1.0,
                    help="⭐ text_router/adaLN 的 lr 倍数（实测需要 >1，"
                         "因为它们的权重从 0.0095 几乎没长）")
    ap.add_argument("--router-wd", type=float, default=0.0,
                    help="text_router 的 weight decay（>0 抑制过拟合）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", default=None,
                    help="接着这个权重练（⛔ 不给就从零）")
    ap.add_argument("--text-emb", default=None,
                    help="教师文本 embedding 缓存（①挂文本塔用）")
    ap.add_argument("--latents", default=None,
                    help="已缓存的 latent 文件（省掉每次解 JPEG）")
    ap.add_argument("--seed", type=int, default=1)
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

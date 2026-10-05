"""P2 · 从训好的 KP 主干采样出图 —— ⭐ 出「第一张 KP 自己的图」

═══ 这一步意味着什么 ═══
⚠️ 前面所有"出图"都是**别人的模型**出的（Sana）⇒ 用户判定**效益≈0**。
本脚本用的是**我们自己训的 DiT** + DC-AE 解码 ⇒ 这才是 **L3 任务效益**。

═══ ⚠️ 第一版是无条件的 ═══
文本塔还没训 ⇒ 现在**没有文本条件** ⇒ 采样是**无条件生成**
（等于"随机抽一张二次元图"，不是"按你的话画"）。
⭐ 有图之后再挂文本塔，才有"夕阳海滩少女"。

═══ 采样器 ═══
Rectified Flow：`x_{t+dt} = x_t - v_θ(x_t, t)·dt`，t: 1→0
用 Euler 离散（设计稿的默认）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(KP_ROOT))
sys.path.insert(0, str(KP_ROOT / "vendor" / "efficientvit"))
OUT_DIR = KP_ROOT / "out" / "kp_samples"


def load_dcae(dev):
    from efficientvit.models.efficientvit.dc_ae import dc_ae_f32c32, DCAE
    from safetensors.torch import load_file
    cfg = dc_ae_f32c32("dc-ae-f32c32-sana-1.0", None)
    m = DCAE(cfg)
    sd = load_file(str(KP_ROOT / "models" / "dc_ae_f32c32_sana_1.0.safetensors"))
    miss, unexp = m.load_state_dict(sd, strict=False)
    if len(miss) > 10 or len(unexp) > 10:
        raise RuntimeError(f"DC-AE 没加载好: {len(miss)}/{len(unexp)}")
    return m.eval().to(dev).float(), float(cfg.scaling_factor)


def sample(model, shape, steps: int, dev, seed: int = 0):
    """Rectified Flow + Euler。t: 1 → 0。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(shape, generator=g, device=dev)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i]
        tb = t.expand(shape[0])
        with torch.no_grad():
            v = model(x, tb)
            if not torch.is_tensor(v):
                v = v[0]
        dt = ts[i + 1] - ts[i]
        x = x + v.float() * dt          # 负速度方向 ⇒ x 从噪声走向数据
    return x


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P2 · KP 主干采样")
    ap.add_argument("--ckpt", default="out/dit/_ckpt.pt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from kp.models.dit import SingleStreamDiT, DiTCfg
    from kp.train.train_dit import make_backbone_trainable

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = KP_ROOT / a.ckpt
    if not ck.exists():
        print(f"[X] no ckpt: {ck}", file=sys.stderr)
        return 1
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfgd = blob.get("config", {}) or {}
    cfg = DiTCfg(dim=int(cfgd["dim"]), layers=int(cfgd["layers"]),
                 heads=int(cfgd["heads"]))
    model = SingleStreamDiT(cfg, latent_ch=32)
    make_backbone_trainable(model)          # ⚠️ 权重是 buffer ⇒ 得先提升
    miss, unexp = model.load_state_dict(blob["state_dict"], strict=False)
    if len(miss) > 5:
        print(f"[!] missing={len(miss)}: {miss[:3]}", flush=True)
    model = model.eval().to(dev)
    print(f"[*] loaded {ck.name} dim={cfgd['dim']} L={cfgd['layers']} "
          f"step={blob.get('step')}", flush=True)

    vae, scale = load_dcae(dev)
    lat = 256 // 32
    t0 = time.time()
    z = sample(model, (a.n, 32, lat, lat), a.steps, dev, a.seed)
    print(f"[*] sampled in {time.time()-t0:.1f}s", flush=True)

    with torch.no_grad():
        img = vae.decode((z / scale).float())
    if not torch.is_tensor(img):
        img = img[0]
    # ⭐ 统一成 [B,3,H,W]（decode 可能返回 3 维或带额外包裹）
    if not torch.is_tensor(img):
        img = img[0]
    img = img.float()
    while img.dim() > 4:
        img = img[0]
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.shape[1] != 3 and img.shape[-1] == 3:
        img = img.permute(0, 3, 1, 2)
    img = (img.clamp(-1, 1) + 1) / 2
    print(f"[*] image tensor {tuple(img.shape)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from PIL import Image
    arr = (img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")  # B,C,H,W -> B,H,W,C
    for i in range(arr.shape[0]):
        p = OUT_DIR / f"kp_{i:02d}.png"
        Image.fromarray(arr[i]).save(p)
        print(f"  -> {p}", flush=True)
    # 拼一张总览
    cols = min(4, arr.shape[0])
    rows = (arr.shape[0] + cols - 1) // cols
    h, w = arr.shape[1], arr.shape[2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype="uint8")
    for i in range(arr.shape[0]):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = arr[i]
    sp = OUT_DIR / "kp_sheet.png"
    Image.fromarray(sheet).save(sp)
    print(f"[OK] {sp}")
    print("⚠️ 这是**无条件**生成（文本塔未训）⇒ 不能指定内容；用户看图判断")
    return 0


if __name__ == "__main__":
    sys.exit(main())

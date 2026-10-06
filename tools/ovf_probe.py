"""⭐ 提示敏感性探针（一个脚本量化「模型听不听提示」）

═══ 为什么需要它 ═══
之前的判断是「肉眼看着三行差不多」⇒ 靠眼睛分辨不出。
⭐ 但**模型听不听提示**是个可量化的量：
   - 固定 seed ⇒ 不同提示的像素差 = **文本信号**
   - 同提示不同 seed 的像素差 = **随机基线**
   - ⭐ **信号 / 基线** = 文本条件「学到了几成」
   昨天实测：过拟合模型 0.09-0.19 vs 随机基线 0.21 ⇒ 信号 ≈ 基线 ⇒ 学到了
   4000 步真训：0.007 vs 0.21 ⇒ 只占 3% ⇒ 没学到

## ⚠️ 踩过的坑（保留在这防止重犯）
DC-AE decode 输出固定 (B,3,H,W)；PIL 只吃 **HWC**
⇒ 漏 permute 会报 `Cannot handle (1,1,256)`（PIL 内部再 squeeze 一次，报错信息完全误导）
"""
import sys, os, itertools, argparse
sys.path.insert(0, '.')
import torch
import numpy as np
from PIL import Image, ImageDraw
from kp.models.dit import SingleStreamDiT, DiTCfg
from kp.train.train_dit import make_backbone_trainable
from kp.train.sample_dit import sample, load_dcae, encode_text


def norm01(im):
    """DC-AE 输出 -> HWC uint8（⚠️ permute 不能省，见模块 docstring）"""
    if im.dim() == 4:
        im = im[0]
    arr = ((im.clamp(-1, 1) + 1) * 127.5).clip(0, 255).byte()
    return arr.permute(1, 2, 0).numpy() if arr.dim() == 3 else arr.numpy()


def build(ck, dev):
    blob = torch.load(ck, map_location='cpu', weights_only=False)
    c = blob['config']
    m = SingleStreamDiT(DiTCfg(dim=c['dim'], layers=c['layers'], heads=c['heads']),
                        latent_ch=c.get('latent_ch', 32),
                        text_dim=c.get('text_dim'))
    make_backbone_trainable(m)
    m.load_state_dict(blob['state_dict'])
    return m.eval().to(dev), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--prompts', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--steps', type=int, default=32)
    ap.add_argument('--cfg', type=float, default=0.0)
    ap.add_argument('--out', default='out/ovf_probe')
    a = ap.parse_args()

    dev = torch.device('cuda')
    model, cfg = build(a.ckpt, dev)
    lat = cfg.get('size', 256) // 32
    os.makedirs(a.out, exist_ok=True)

    # 缓存 VAE（别每张都重载）
    vae, sc = load_dcae(dev)

    # ---- 信号：不同提示（同 seed）----
    sig = {}
    for p in a.prompts:
        txt = encode_text([p], dev)
        with torch.no_grad():
            z = sample(model, (1, 32, lat, lat), a.steps, dev, seed=7, txt=txt,
                       cfg=a.cfg)
            r = vae.decode((z / sc).float())
        sig[p] = norm01(r.float().cpu()).astype('float32') / 255.0
        print('[sig] %r' % p, flush=True)

    # ---- 基线：同提示不同 seed ----
    txt0 = encode_text([a.prompts[0]], dev)
    base = []
    for s in range(a.seeds + 1):
        with torch.no_grad():
            z = sample(model, (1, 32, lat, lat), a.steps, dev,
                       seed=100 + s, txt=txt0, cfg=a.cfg)
            r = vae.decode((z / sc).float())
        base.append(norm01(r.float().cpu()).astype('float32') / 255.0)
    base_d = float(np.mean([np.abs(base[i] - base[j]).mean()
                            for i, j in itertools.combinations(range(len(base)), 2)]))
    print('[baseline] 同提示不同seed = %.4f' % base_d, flush=True)

    # ---- 汇总 ----
    sigs = []
    for p, q in itertools.combinations(a.prompts, 2):
        d = float(np.abs(sig[p] - sig[q]).mean())
        sigs.append(d)
        print('[pair] %.4f  %r | %r' % (d, p[:40], q[:40]), flush=True)
    sig_d = float(np.mean(sigs))
    print()
    print('========== 提示敏感性 ==========')
    print('  信号（不同提示）  = %.4f' % sig_d)
    print('  基线（同提示种子）= %.4f' % base_d)
    print('  ⭐ 信号/基线 = %.0f%%' % (100 * sig_d / max(base_d, 1e-6)))
    print('  (>30%% 算学到了；<10%% 等于没学)')

    # ---- 拼图 ----
    S, pad, lab = 256, 8, 24
    W = len(a.prompts) * (S + pad) + pad
    H = S + pad * 2 + lab
    sheet = Image.new('RGB', (W, H), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)
    x = pad
    for p in a.prompts:
        dr.text((x + 2, pad + 4), p[:44], fill=(255, 235, 180))
        sheet.paste(Image.fromarray((sig[p] * 255).astype('uint8')), (x, pad + lab))
        x += S + pad
    out = os.path.join(a.out, 'sheet.png')
    sheet.save(out)
    print('[OK] %s' % out)


if __name__ == '__main__':
    main()

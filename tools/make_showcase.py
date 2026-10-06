"""⭐ 出成果图：多提示 × 多种子，网格排版"""
import sys, os, argparse
sys.path.insert(0, '.')
import torch
import numpy as np
from PIL import Image, ImageDraw
from kp.models.dit import SingleStreamDiT, DiTCfg
from kp.train.train_dit import make_backbone_trainable
from kp.train.sample_dit import sample, load_dcae, encode_text

P = [
    ('1girl, misaka mikoto, toaru kagaku no railgun', '御坂美琴'),
    ('1girl, shirai kuroko, toaru kagaku no railgun', '白井黑子'),
    ('1girl, alters \\(fate\\), fate/extra', 'Alter 角色'),
    ('3girls, outdoors, cherry blossoms', '三人·樱花'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', type=float, default=8.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=48)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--out', default='out/showcase.png')
    a = ap.parse_args()

    dev = torch.device('cuda')
    blob = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    c = blob['config']
    m = SingleStreamDiT(DiTCfg(dim=c['dim'], layers=c['layers'],
                              heads=c['heads']),
                        latent_ch=32, text_dim=c.get('text_dim'))
    make_backbone_trainable(m)
    m.load_state_dict(blob['state_dict'])
    m.eval().to(dev).float()
    vae, sc = load_dcae(dev)
    lat = a.size // 32

    rows = []
    for prompt, tag in P:
        txt = encode_text([prompt] * a.seeds, dev)
        with torch.no_grad():
            z = sample(m, (a.seeds, 32, lat, lat), a.steps, dev,
                       seed=1234, txt=txt, cfg=a.cfg)
            r = vae.decode((z / sc).float())
        im = r.float().cpu().clamp(-1, 1)
        im = ((im + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
        rows.append((prompt, tag, im))
        print('[row] %s' % tag, flush=True)

    S, pad, lab, hdr = a.size, 10, 22, 26
    W = a.seeds * (S + pad) + pad
    H = len(rows) * (S + pad + lab) + pad
    sheet = Image.new('RGB', (W, H), (18, 18, 22))
    dr = ImageDraw.Draw(sheet)
    y = pad
    for prompt, tag, ims in rows:
        dr.text((pad + 2, y + 4), '%s  |  %s' % (tag, prompt[:52]),
                fill=(255, 232, 170))
        y += lab
        for j, im in enumerate(ims):
            sheet.paste(Image.fromarray(im), (pad + j * (S + pad), y))
        y += S + pad
    sheet.save(a.out)
    print('[OK] %s  cfg=%.1f seeds=%d steps=%d'
          % (a.out, a.cfg, a.seeds, a.steps))


if __name__ == '__main__':
    main()

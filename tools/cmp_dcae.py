"""对照：原图 / 我们的VAE / DC-AE 现成32x —— 让用户直接看差别"""
import sys, os, io, math, types
sys.path.insert(0, '.')
sys.path.insert(0, 'vendor/efficientvit')
import torch, numpy as np, pyarrow.parquet as pq
from PIL import Image, ImageDraw

from kp.models.vae import HybridVAE
from efficientvit.models.efficientvit.dc_ae import dc_ae_f32c32, DCAE
from safetensors.torch import load_file

dev = torch.device('cuda')
# ours
_ours_blob = torch.load('out/vae/b32_256_s4000.pt', map_location='cpu', weights_only=False)
_ocfg = _ours_blob.get('config', {})
ours = HybridVAE(base=int(_ocfg.get('base', 16)),
                 res_blocks=int(_ocfg.get('res_blocks', 0)))
ours.load_state_dict(_ours_blob['state_dict']); ours.eval().to(dev)

# dcae
_dcfg = dc_ae_f32c32('dc-ae-f32c32-sana-1.0', None)
dcae = DCAE(_dcfg)
_sd = load_file('models/dc_ae_f32c32_sana_1.0.safetensors')
dcae.load_state_dict(_sd, strict=False)
dcae = dcae.eval().to(dev).float()
print('[*] both loaded', flush=True)

pf = pq.ParquetFile('out/data/curated_danbooru/_shards/data_shard_00000.parquet')
idx = [500, 501, 502, 503]
raw = []
for b in pf.iter_batches(batch_size=64, columns=['image']):
    for r in b.to_pylist():
        if len(raw) > max(idx):
            break
        raw.append(r['image'])
    if len(raw) > max(idx):
        break

def prep(b):
    im = Image.open(io.BytesIO(b)).convert('RGB')
    w, h = im.size; s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((256, 256), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, dtype='uint8').copy()).float().permute(2, 0, 1)/127.5 - 1.0

xs = [prep(raw[i]) for i in idx]
rows = [('ORIGINAL', xs)]
for tag, m, is_dc in (('OURS (17.7dB)', ours, False), ('DC-AE 32x (20.3dB)', dcae, True)):
    out = []
    for x in xs:
        with torch.no_grad():
            z = m.encode(x.unsqueeze(0).to(dev))
            if not torch.is_tensor(z):
                z = z[0] if isinstance(z, (tuple, list)) else z
            r = m.decode(z)
            if not torch.is_tensor(r):
                r = r[0] if isinstance(r, (tuple, list)) else r
        rc = r.float().cpu()
        if rc.dim() == 4:
            rc = rc[0]
        out.append(rc.clamp(-1, 1))
    rows.append((tag, out))

pad, lab, S = 6, 24, 256
W = len(idx)*(S+pad)+pad
H = len(rows)*(S+pad+lab)+pad
sheet = Image.new('RGB', (W, H), (24, 24, 28))
dr = ImageDraw.Draw(sheet)
y = pad
for tag, imgs in rows:
    dr.text((pad+2, y+5), tag, fill=(255, 235, 180))
    y += lab
    for j, t in enumerate(imgs):
        arr = ((t.permute(1, 2, 0).numpy()+1)*127.5).clip(0, 255).astype('uint8')
        sheet.paste(Image.fromarray(arr), (pad+j*(S+pad), y))
    y += S+pad
os.makedirs('out/compare', exist_ok=True)
sheet.save('out/compare/ours_vs_dcae.png')
print('[OK] out/compare/ours_vs_dcae.png')

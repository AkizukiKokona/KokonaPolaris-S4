import sys, os, types
sys.path.insert(0, 'vendor/efficientvit')
# ⚠️ efficientvit 的 __init__ 会连带 import SAM（要 segment_anything）
#   ⇒ 我们只要 dc_ae，没必要装无关依赖 ⇒ **建一个假的包 + 子模块**占位
def _stub_pkg(name, submods):
    mod = types.ModuleType(name)
    mod.__path__ = []                      # 让它像个包
    sys.modules[name] = mod
    for sub, attrs in submods.items():
        sm = types.ModuleType(name + "." + sub)
        for a in attrs:
            setattr(sm, a, type(a, (), {}))
        sys.modules[name + "." + sub] = sm
        setattr(mod, sub, sm)
    return mod


_sa = _stub_pkg("segment_anything", {
    "modeling": ["MaskDecoder", "PromptEncoder", "TwoWayTransformer"],
    "build_sam": ["build_sam", "sam_model_registry"],
})
for _a in ("SamAutomaticMaskGenerator", "SamPredictor", "sam_model_registry"):
    setattr(_sa, _a, type(_a, (), {}))
_stub_pkg("onnxsim", {})
sys.modules["onnxsim"].simplify = lambda *a, **k: (None, True)
import torch, io, math
import numpy as np, pyarrow.parquet as pq
from PIL import Image
from efficientvit.models.efficientvit.dc_ae import dc_ae_f32c32, DCAE
from safetensors.torch import load_file

# ⚠️ dc_ae_f32c32 返回的是 **config**，不是模型；要 DCAE(cfg) 才实例化
# ⚠️ 且它的 load_model 只认 {"state_dict": ...} 的 .pt
#   ⇒ 我们的资产是 raw safetensors ⇒ 自己 load_state_dict
_cfg = dc_ae_f32c32('dc-ae-f32c32-sana-1.0', None)
vae = DCAE(_cfg)
_sd = load_file('models/dc_ae_f32c32_sana_1.0.safetensors')
_miss, _unexp = vae.load_state_dict(_sd, strict=False)
print('[*] missing=%d unexpected=%d' % (len(_miss), len(_unexp)), flush=True)
if len(_miss) > 10 or len(_unexp) > 10:
    raise SystemExit('weights did not load properly')
vae = vae.cuda().eval().float()
print('[*] params %.1fM' % (sum(p.numel() for p in vae.parameters()) / 1e6), flush=True)

pf = pq.ParquetFile('out/data/curated_danbooru/_shards/data_shard_00000.parquet')
imgs = []
for b in pf.iter_batches(batch_size=64, columns=['image']):
    for r in b.to_pylist():
        im = Image.open(io.BytesIO(r['image'])).convert('RGB')
        w, h = im.size; s = min(w, h)
        im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((256, 256), Image.LANCZOS)
        imgs.append(torch.from_numpy(np.asarray(im, dtype='uint8').copy()).float().permute(2, 0, 1) / 127.5 - 1.0)
        if len(imgs) >= 4:
            break
    if len(imgs) >= 4:
        break

os.makedirs('out/dcae_recon', exist_ok=True)
tot = 0.0
for i, x in enumerate(imgs):
    xb = x.unsqueeze(0).cuda()
    with torch.no_grad():
        z = vae.encode(xb)
        r = vae.decode(z)
    rc = r.float().cpu()[0].clamp(-1, 1)
    mse = float(((rc - x) ** 2).mean())
    psnr = 10 * math.log10(4 / max(1e-8, mse))
    tot += psnr
    pair = torch.cat([x, rc], dim=2)
    Image.fromarray(((pair.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8')).save(
        'out/dcae_recon/dcae_%02d.png' % i)
    print('  [%d] latent%s PSNR=%.2fdB' % (i, tuple(z.shape), psnr), flush=True)
print('  DC-AE real PSNR = %.2f dB   (ours best: 17.7)' % (tot / len(imgs)))

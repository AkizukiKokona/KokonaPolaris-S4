"""Near-duplicate detection across all repo images (read-only)."""
import os

import numpy as np
from PIL import Image

ROOT = r"D:/model"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXCLUDE = {".git", ".venv", "site-packages", "node_modules", "__pycache__"}

paths = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in EXCLUDE]
    for f in fn:
        if os.path.splitext(f)[1].lower() in EXTS:
            paths.append(os.path.join(dp, f))

# 16x16 grayscale signature
sigs = {}
for p in paths:
    try:
        with Image.open(p) as im:
            im = im.convert("L").resize((16, 16), Image.BILINEAR)
            sigs[p] = np.asarray(im, dtype=np.float32).ravel()
    except Exception as e:
        print("skip", p, e)

keys = list(sigs)
parent = list(range(len(keys)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


# dHash 64-bit for exact-dup detection
def dhash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


hs = {p: dhash(p) for p in keys}
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        dist = int((hs[keys[i]] ^ hs[keys[j]]).sum())
        if dist <= 6:  # hamming <=6 => visually same
            union(i, j)

clusters = {}
for i, p in enumerate(keys):
    clusters.setdefault(find(i), []).append(p)

print(f"total images: {len(keys)}")
print(f"unique (dhash<=6) clusters: {len(clusters)}")
print(f"redundant copies: {len(keys) - len(clusters)}")
print("\n--- largest clusters ---")
for r, mem in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:12]:
    dirs = sorted({os.path.relpath(os.path.dirname(m), ROOT) for m in mem})
    print(f"  n={len(mem):3d}  dirs={dirs}")
    for m in sorted(mem)[:6]:
        print(f"        {os.path.relpath(m, ROOT)}")
    if len(mem) > 6:
        print(f"        ... +{len(mem)-6} more")

# unique count restricted to >=512px and to the two default dirs
def ok(p, thresh):
    try:
        with Image.open(p) as im:
            w, h = im.size
        return min(w, h) >= thresh
    except Exception:
        return False


for thresh in (512, 1024):
    tot = [p for p in keys if ok(p, thresh)]
    cl = {find(i) for i, p in enumerate(keys) if ok(p, thresh)}
    print(f"\nmin(w,h)>={thresh}: {len(tot)} images, {len(cl)} unique clusters")

DEF = {"data/characters/kokona/images", "out/e5b/g1/bf16"}
d = [p for p in keys if os.path.relpath(os.path.dirname(p), ROOT).replace("\\", "/") in DEF]
dc = {find(i) for i, p in enumerate(keys) if p in set(d)}
print(f"\nreal_separation.py default 2 dirs: {len(d)} images, {len(dc)} unique")

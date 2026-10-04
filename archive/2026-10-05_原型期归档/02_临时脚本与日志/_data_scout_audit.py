"""Read-only image audit for VAE training data gap analysis. Does not modify anything."""
import os
import random
from collections import Counter, defaultdict

from PIL import Image

ROOT = r"D:/model"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXCLUDE = {".git", ".venv", "site-packages", "node_modules", "__pycache__"}

DIRS = [
    "data/characters/kokona/images",
    "data/characters/kokona/raw",
    "out/e5b/g1/bf16",
    "out/e5b/g1_images/bf16",
    "out/e5b/g1_images/PTQ-W4A8",
    "out/e5b/g1_images/PTQ-W4A4",
    "out/e5b/g1_cmp",
    "out/e5/images",
    "out/e4b_bf16",
    "out/e6",
    "out/e6/synth",
    "out/characters/kokona",
    "out/characters/kokona/norm",
    "out/characters/kokona/layers",
    "out/typography_chain",
    ".",
]

MAX_SAMPLE = 30


def audit(rel):
    d = os.path.join(ROOT, rel)
    if not os.path.isdir(d):
        print(f"[MISSING] {rel}")
        return
    files = sorted(
        f for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in EXTS
        and os.path.isfile(os.path.join(d, f))
    )
    if not files:
        print(f"[EMPTY]   {rel}")
        return
    total_bytes = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    rnd = random.Random(0)
    sample = files if len(files) <= MAX_SAMPLE else rnd.sample(files, MAX_SAMPLE)

    sizes, modes, alpha, extreme, minpx = Counter(), Counter(), 0, [], []
    for f in sample:
        try:
            with Image.open(os.path.join(d, f)) as im:
                w, h = im.size
                sizes[f"{w}x{h}"] += 1
                modes[im.mode] += 1
                has_a = im.mode in ("RGBA", "LA") or (
                    im.mode == "P" and "transparency" in im.info
                )
                alpha += int(has_a)
                ar = w / h if h else 0
                if ar > 3 or ar < 1 / 3:
                    extreme.append(f"{f}({w}x{h})")
                minpx.append(min(w, h))
        except Exception as e:
            print(f"  !! {f}: {e}")

    print(f"\n=== {rel} ===")
    print(f"  files={len(files)}  sampled={len(sample)}  avg={total_bytes/len(files)/1024:.0f}KB")
    top = sizes.most_common(6)
    print(f"  sizes: {top}")
    if len(sizes) > 6:
        ws = [int(s.split("x")[0]) for s in sizes.elements()]
        hs = [int(s.split("x")[1]) for s in sizes.elements()]
        print(f"  w range {min(ws)}-{max(ws)}  h range {min(hs)}-{max(hs)}")
    print(f"  modes: {dict(modes)}")
    print(f"  has_alpha: {alpha}/{len(sample)}")
    print(f"  min(w,h): min={min(minpx)} median={sorted(minpx)[len(minpx)//2]} max={max(minpx)}")
    print(f"  extreme_ar(<1:3 or >3:1): {len(extreme)} {extreme[:5]}")


for r in DIRS:
    audit(r)

# full-repo totals
print("\n\n########## FULL REPO ##########")
total = Counter()
per_dir = defaultdict(int)
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
    for f in filenames:
        if os.path.splitext(f)[1].lower() in EXTS:
            per_dir[os.path.relpath(dirpath, ROOT)] += 1
            total[os.path.splitext(f)[1].lower()] += 1
print("by ext:", dict(total))
print("grand total:", sum(total.values()))
print("dirs with images:", len(per_dir))

# view naming convention check
print("\n########## MULTI-VIEW NAMING ##########")
import re
views = ["front", "back", "left", "right", "side", "f", "b"]
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
    imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in EXTS]
    tagged = [f for f in imgs if any(re.search(rf"(^|[_\-]){v}([_\-.]|$)", os.path.splitext(f)[0]) for v in views)]
    if tagged:
        print(f"  {os.path.relpath(dirpath, ROOT)}: {len(tagged)} -> {sorted(tagged)[:12]}")

# character grouping by stripping view suffix
print("\n########## CHARACTER SETS ##########")
groups = defaultdict(list)
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
    for f in filenames:
        stem, ext = os.path.splitext(f)
        if ext.lower() not in EXTS:
            continue
        base = re.sub(r"(_v\d+|_s\d+|_1024|_whitebg|_fullscreen_ref|_ref|_bf16|_ptw\d+_\w+)$", "", stem, flags=re.I)
        base = re.sub(r"[_\-]?(front|back|left|right|side)$", "", base, flags=re.I)
        base = re.sub(r"^(cut|cutrembg|raw|clusters|norm)_?", "", base, flags=re.I)
        groups[base or stem].append(os.path.join(os.path.relpath(dirpath, ROOT), f))
for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(v) > 1:
        print(f"  {k}: {len(v)}")

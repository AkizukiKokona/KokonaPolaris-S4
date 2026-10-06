"""P1.8 · 缓存**教师文本 embedding**（给 DiT 训练用）

═══ 为什么缓存 ═══
DiT 训练每步都要文本条件。若每步现跑教师（4bit 4B）⇒ 每步 +0.5s ⇒ 训练慢 5 倍。
⇒ 教师**离线跑一次**，把 embedding 落盘（设计稿也说「教师是离线工具，不是常驻依赖」）。

═══ 两种 caption 来源 ═══
① `--captions out/captions/zh_captions.jsonl`：用已改写的**中文** caption
② 不给就用数据里的 `prompt`（**英文 Danbooru 标签**）
   ⚠️ 设计稿要求「caption 中文优先」⇒ ②只是**快速验证用**，不是最终形态

⚠️ **多层特征聚合**（设计稿 §4.1）：不只取最后一层，取 TEACHER_LAYERS 的均值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"
OUT_DIR = KP_ROOT / "out" / "text_emb"
TEACHER = KP_ROOT / "models" / "Qwen3.5-4B-Base"
#: 设计稿 §4.1：多层特征聚合（32 层取 4 层）
TEACHER_LAYERS = (8, 16, 24, 28)
MAX_LEN = 96


def read_captions(shards: int, limit: int, captions_file: Optional[str]):
    """→ (texts, tags)"""
    if captions_file and Path(captions_file).exists():
        rows = []
        with Path(captions_file).open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rows.append((d["zh"], d["tags"]))
                if len(rows) >= limit:
                    break
        print(f"[*] using {len(rows)} ZH captions from {captions_file}")
        return [r[0] for r in rows], [r[1] for r in rows]
    import pyarrow.parquet as pq
    tags: List[str] = []
    for p in sorted(SHARD_DIR.glob("*.parquet"))[:shards]:
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=256, columns=["prompt"]):
            for r in b.to_pylist():
                tags.append((r.get("prompt") or "").strip()[:400])
                if len(tags) >= limit:
                    break
            if len(tags) >= limit:
                break
        if len(tags) >= limit:
            break
    print(f"[*] using {len(tags)} EN tags (非设计要求的 中文优先)")
    return tags, tags


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="缓存教师文本 embedding")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--captions", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=list(TEACHER_LAYERS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    texts, tags = read_captions(a.shards, a.limit, a.captions)
    if not texts:
        print("[X] no captions", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(str(TEACHER))
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(TEACHER), quantization_config=bnb, device_map="cuda",
        trust_remote_code=True).eval()
    n_lay = model.config.num_hidden_layers
    use = [i for i in a.layers if i < n_lay]
    print(f"[*] teacher layers={n_lay}, using {use}", flush=True)

    embs: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    t0 = time.time()
    for i in range(0, len(texts), a.batch):
        chunk = texts[i:i + a.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_len).to("cuda")
        with torch.no_grad():
            o = model(enc["input_ids"], attention_mask=enc["attention_mask"],
                      output_hidden_states=True, use_cache=False)
        # ⭐ 多层聚合：取 use 层的均值（设计稿 §4.1）
        hs = torch.stack([o.hidden_states[j + 1] for j in use], dim=0).mean(0)
        embs.append(hs.float().cpu().to(torch.float16))
        masks.append(enc["attention_mask"].cpu().bool())
        if (i // a.batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{len(texts)}  {time.time()-t0:.0f}s",
                  flush=True)

    E = torch.cat(embs, 0)
    M = torch.cat(masks, 0)
    outp = OUT_DIR / (a.out or ("zh" if a.captions else "entags") + f"_{len(texts)}.pt")
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": E, "mask": M, "texts": texts, "tags": tags,
                "layers": use, "teacher": str(TEACHER.name)}, outp)
    print(f"[OK] {outp}  emb={tuple(E.shape)}  {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

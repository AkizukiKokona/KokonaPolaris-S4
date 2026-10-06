"""P1.8 第一步 · **把 Danbooru 标签改写成中文自然语言 caption**

═══ 为什么必须做（设计稿的要求）═══
`design/补充11 §4.8` + `补充11 第 300-314 行`：
> 文本塔的教师是多语言语言模型（Qwen3.5-4B-Base），中文是它的**原生能力**；
> **蒸馏时刻意保留**即可。**数据侧对应件：caption 中文优先**（10–20% 英文对齐）。
> 条件轴接受**中文自然语言描述**（不是标签串）。

⚠️ 而我们的数据（curated-danbooru）里 `prompt` 是**英文 Danbooru 标签串**
⇒ 直接用会**违背设计的 caption 语言分布** ⇒ 必须改写。

═══ 怎么改写（实测可行）═══
用教师自己做 **few-shot 标签→中文**：
    `1girl, beach, sunset` → `一个少女，海滩，夕阳`
⚠️ 教师是 **Base 版**（不是 instruct）⇒ 不能靠"指令"，要靠**少样本续写**。
✅ 实测（tools/probe_teacher.py）：few-shot 有效。

═══ 输出 ═══
`out/captions/zh_captions.jsonl`：每行 `{"i": 序号, "tags": 原标签, "zh": 中文}`

⛔ 只做改写，**不在这里跑蒸馏**（蒸馏要教师 hidden states，是下一步）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

KP_ROOT = Path(os.environ.get("KP_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = KP_ROOT / "out" / "data" / "curated_danbooru" / "_shards"
OUT_DIR = KP_ROOT / "out" / "captions"
TEACHER = KP_ROOT / "models" / "Qwen3.5-4B-Base"

# ⭐ few-shot 示范（Base 模型靠这个"学会"格式）
FEWSHOT = [
    ("1girl, beach, sunset", "一个少女在海滩上看夕阳"),
    ("2girls, school uniform, classroom", "两个穿校服的女孩在教室里"),
    ("1girl, long hair, smile, cherry blossoms", "一个长发少女微笑着站在樱花树下"),
    ("1boy, sword, night, rain", "一个少年在雨夜中持剑"),
    ("no humans, landscape, mountain, clouds", "一片没有人的山峦云海风景"),
]


def read_tags(shards: int, limit: int) -> List[str]:
    import pyarrow.parquet as pq
    out: List[str] = []
    for p in sorted(SHARD_DIR.glob("*.parquet"))[:shards]:
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=256, columns=["prompt"]):
            for r in b.to_pylist():
                out.append((r.get("prompt") or "").strip())
                if len(out) >= limit:
                    return out
    return out


def build_prompt(tags: str) -> str:
    lines = ["把 Danbooru 标签改写成一句自然的中文描述。"]
    for t, z in FEWSHOT:
        lines.append(f"{t} -> {z}")
    lines.append(f"{tags} ->")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="标签 → 中文 caption（few-shot）")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--in-len", type=int, default=384,
                    help="输入截断长度（⭐ 实测 few-shot prompt ~260 token，"
                         "1024 是浪费 3-4 倍）")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default="zh_captions.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="接着已有文件继续（支持断点）")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(str(TEACHER))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(TEACHER), quantization_config=bnb, device_map="cuda",
        trust_remote_code=True).eval()
    print(f"[*] teacher loaded {time.time()-t0:.0f}s "
          f"VRAM={torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)

    tags = read_tags(a.shards, a.limit)
    print(f"[*] {len(tags)} tags", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outp = OUT_DIR / a.out
    done = 0
    if a.resume and outp.exists():
        with outp.open(encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[*] resume from {done}", flush=True)

    t1 = time.time()
    with outp.open("a" if done else "w", encoding="utf-8") as fo:
        for i in range(done, len(tags), a.batch):
            chunk = tags[i:i + a.batch]
            prompts = [build_prompt(t) for t in chunk]
            # ⭐ 2026-10-06 修性能：实测 few-shot prompt 只有 **260 token**
            #   （见 docstring）而这里写 1024 ⇒ tokenizer 会按最长补齐，
            #   而且 generate 每步都过 1024 长度的 KV ⇒ **白白慢 ~3-4 倍**。
            #   ⇒ 按真实长度 + 输出余量设上限。
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=a.in_len).to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for j, (tg, tx) in enumerate(zip(chunk, texts)):
                # ⭐ 只取第一行（few-shot 续写可能带出下一例）
                zh = tx.strip().split("\n")[0].strip()
                fo.write(json.dumps({"i": i + j, "tags": tg, "zh": zh},
                                    ensure_ascii=False) + "\n")
            fo.flush()
            if (i - done) % (a.batch * 20) == 0:
                el = time.time() - t1
                sp = (i + a.batch - done) / max(1e-6, el)
                eta = (len(tags) - i - a.batch) / max(1e-6, sp)
                print(f"  {i+a.batch}/{len(tags)}  {sp:.1f}/s  "
                      f"eta {eta/60:.0f}min", flush=True)
                print(f"     样例: {chunk[0][:50]} -> {texts[0].strip()[:60]}",
                      flush=True)
    print(f"[OK] {outp}  用时 {time.time()-t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

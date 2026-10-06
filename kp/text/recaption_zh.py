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
import re
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


def build_prompt(tags: str, max_chars: int = 24) -> str:
    """⭐⭐ 长度约束是**必须显式写出来**的（2026-10-06 实测）。

    【实测】few-shot 示例本来就只有 12-15 字，但教师输出**平均 87 字**、
    最长 356 ⇒ ⛔ 它**没有在模仿示例长度**，只是在「翻译完整」。
    ⇒ 结果 89% 被 max_new 截断成残句（实测 1272 条里 1131 条残）。
    ✅ 解法：把「不超过 N 字」写进指令 + 换一个明确讲规则的句式。
    """
    lines = [
        f"把 Danbooru 标签改写成**一句简短的中文画面描述，不超过 {max_chars} 个字**。",
        "要求：",
        f"1. 必须**不超过 {max_chars} 个汉字**，超长算错",
        "2. 只写画面：人物数量 / 发色 / 动作 / 场景 / 主要服饰",
        "3. ⛔ 不要评价、不要解释、不要背景细节、不要心理描写",
        "4. ⛔ 结尾不要句号",
        "示例：",
    ]
    for t, z in FEWSHOT:
        lines.append(f"{t} -> {z}")
    lines.append(f"{tags} ->")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="标签 → 中文 caption（few-shot）")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-offset", type=int, default=0,
                    help="⭐ 从第几个标签开始（配合多进程并行切分）")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--keep-truncated", action="store_true", default=True,
                    help="⭐ 保留超长句（默认开）—— 丢弃等于白生成已算的 token；"
                         "事后用 tools/clean_captions.py 按长度清洗更划算")
    ap.add_argument("--no-keep-truncated", dest="keep_truncated",
                    action="store_false",
                    help="生成时就丢弃超长句（会浪费已算的 token）")
    ap.add_argument("--max-chars", type=int, default=24,
                    help="⭐ caption 目标字数（写进指令；教师不会自动模仿示例长度）")
    ap.add_argument("--in-len", type=int, default=384,
                    help="输入截断长度（⭐ 实测 few-shot prompt ~260 token，"
                         "1024 是浪费 3-4 倍）")
    # ⭐ 2026-10-06 提高：实测中文 caption 平均 **87 字**、最长 356
    #   而 72 token ≈ 72 汉字⇒ 大量残句（已加截断检测丢弃）
    # ⭐ 2026-10-06：设 64 足够（约束让输出 ~28 字）；
    #   ⛔ 设太大反而慢（generate 按 max_new 逐步解码）且产出长句
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

    tags = read_tags(a.shards, a.limit + a.shard_offset)[a.shard_offset:]
    print(f"[*] {len(tags)} tags", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outp = OUT_DIR / a.out
    done = 0
    if a.resume and outp.exists():
        with outp.open(encoding="utf-8") as f:
            done = sum(1 for _ in f)
        print(f"[*] resume from {done}", flush=True)
    elif outp.exists():
        # ⚠️ 2026-10-06：多进程并行切分时**不能续写同一个文件**
        #   （进程 A 在写，进程 B 也在写 ⇒ 行数不是"已完成数"）
        #   ⇒ 每个 shard-offset 用**自己的文件**（见 tools/gen_captions.sh）
        print(f"[*] fresh write (shard_offset={a.shard_offset})", flush=True)

    n_trunc = 0
    n_lost_chars = 0
    t1 = time.time()
    with outp.open("a" if done else "w", encoding="utf-8") as fo:
        for i in range(done, len(tags), a.batch):
            chunk = tags[i:i + a.batch]
            prompts = [build_prompt(t, a.max_chars) for t in chunk]
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
                # ⭐⭐ **截断检测**（2026-10-06 修）
                #   实测：max_new=72 token ≈ 72 汉字，而实际 caption 平均 **87 字**
                #   ⇒ 大量 caption **在句子中间被砍**（最长的被砍到 356 字残句）
                #   ⇒ 被截断的句子 = 语法不完整 = 噪声标签
                #   ✅ 修：**只保留以句末标点结尾的**（说明自然结束）
                # ⭐ 2026-10-06 修正判据：**不能只靠句末标点**
                #   实测：加上「不超过 24 字」约束后，教师输出**不带句号**
                #   （指令里写了「不要句号以外的标点」⇒ 它干脆不加标点）
                #   ⇒ 纯标点判据会把**全部**输出丢掉（实测 0 条通过）
                #✅ 现在：**够短**就算完整（本来就不会被截断）
                #   否则要求句末标点（说明自然结束）
                if not a.keep_truncated and len(zh) > a.max_chars + 8:
                    n_trunc += 1
                    n_lost_chars += len(zh)
                    continue
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
    print(f"[OK] {outp}  用时 {time.time()-t1:.0f}s"
          + (f"  skipped-truncated={n_trunc}" if n_trunc else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

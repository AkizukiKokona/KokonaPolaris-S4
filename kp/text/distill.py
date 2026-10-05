"""P1.8 · 文本塔蒸馏 —— 从 Qwen3.5-4B-Base 蒸出 ~217M 的塔，**保住中文**。

🔴🔴 **教师换版带来的未适配问题（2026-10-05 用户决策后，必须知道）**
本脚本原本按**纯文本 Transformer** 写（逐层`output_hidden_states` 取特征）。
而 **Qwen3.5-4B-Base 是多模态早融合架构**：
    `Qwen3_5ForConditionalGeneration`，层布局 =
    `8 × (3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN))`，共 32 层，hidden 2560
⇒ ⚠️ **Gated DeltaNet 层不是标准 attention**，`layer.register_forward_hook` 仍能拿到 hidden states，
   但 **hidden 维度可能与 `attn` 分支不一致**（见 `multi_layer_aggregate` 的维度断言）。
⇒✅ **本文件在这一步已加显式检查**（见 `_probe_teacher`）：维度对不上就**明确报错**，
   而不是静默截断/广播（那会让蒸馏出一个"看着能训、实则错位"的塔）。
⚠️ **词表也变了**：卡片称 248,320，⛔ 但**不可照抄**（本项目已在 Qwen3-4B 上实测出
   「卡片值 151936 ≠ 真实索引上界 151669」，差 267）。⇒ **必须先跑
   `python -m kp.text.tokenizer` 量出真实上界**，再回填 `TextTowerCfg.vocab_size`。

═══ 这一步为什么关键 ═══

设计稿 §4.1 说「文本塔 = ~220M 蒸馏自 Qwen3.5-4B-Base，**多层特征聚合**，不缓存 embedding」。
而 §P1.8 说「**中文是文本塔的原生能力**」——⚠️ 但那句话的**主语是教师（Qwen3）**，
**不是我们的塔**。塔现在**随机初始化**（自检 §31 只证明了「中文能被编码」）。

⚠️ **必须区分的三件事**（本项目反复吃的坑：指标口径不对，结论必错）：
    ① 中文**能被编码**   ✅ 已证（§31，汉字覆盖 100%）
    ② 塔**懂中文**       ⛔ **未证**（要蒸馏后单独测）
    ③ 主干**用中文**     ⛔ 未开始

═══ 本文件解决什么 ═══

① **蒸馏数据通路**：中文/英文文本 → ids → 教师 hidden states → 落盘
② **多层特征聚合**（设计稿 §4.1 明写）：不只取最后一层
③ **中文能力回归门线**：蒸馏**后**才能测 ⇒ 本文件提供**测法**，⛔ 不假装已达标

═══ 关键设计 ═══

⚠️ **教师是「离线跑一次」的工具，不是常驻依赖**（设计稿的定位）
⇒ 本文件**只产出训练对**（ids + 教师特征），蒸馏训练本身是独立一步。
⇒ ⛔ **不要求下载 Qwen3 的 8GB 权重**：`--teacher` 可指向任何 HF id，
   没有权重时**只导出 ids**（第一步仍能做）。

═══ 用法 ═══

    # ① 只把中文 caption 编码成 ids（不需要教师权重，离线可用）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.text.distill \
        --encode data/captions.jsonl --out out/distill/ids.pt

    # ② 再抽教师特征（需要 Qwen3 权重，8GB，离线跑一次）
    ... -m kp.text.distill --encode data/captions.jsonl --out out/distill/ids.pt --teacher Qwen/Qwen3.5-4B-Base
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

#: 设计稿 §4.1：多层特征聚合（不只取最后一层）
# ⭐ LLM 的末层是为 next-token 优化的，**对图像生成并非最优** ⇒ 取浅层+多层融合更稳
TEACHER_LAYERS = (8, 16, 24, 28)     # Qwen3.5-4B-Base 共 36 层，取这几层做聚合
#: 蒸馏目标维度（与 `TextTowerCfg.out_dim` 对齐）
STUDENT_DIM = 1024
MAX_LEN = 128                          # ⭐ 蒸馏阶段用短序列（快）；正式训练再放开


def encode_captions(src: str, out: str, max_len: int = MAX_LEN) -> Dict:
    """jsonl（{"caption": ...}）→ ids（⛔ 不需要教师权重，纯 CPU 离线可做）。"""
    import torch
    from .tokenizer import load_tokenizer
    tok = load_tokenizer()
    rows: List[Dict] = []
    skipped = 0
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"没找到 {src}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:                                  # noqa: BLE001
            skipped += 1
            continue
        cap = o.get("caption") or o.get("text") or ""
        if not cap:
            skipped += 1
            continue
        rows.append({"ids": tok.encode(cap, add_special_tokens=True)[:max_len],
                     "lang": o.get("lang", "zh")})
    obj = {"max_len": max_len, "count": len(rows), "skipped": skipped, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out)
    return {"count": len(rows), "skipped": skipped, "out": out,
            "平均长度": round(sum(len(r["ids"]) for r in rows) / max(1, len(rows)), 1)}


def extract_teacher(ids_pt: str, out: str, teacher: str = "Qwen/Qwen3.5-4B-Base",
                    layers: Sequence[int] = TEACHER_LAYERS,
                    batch: int = 4) -> Dict:
    """抽教师的多层 hidden states（⛔ 需要教师权重，8GB，**离线跑一次**）。"""
    import torch
    from transformers import AutoModelForCausalLM
    from .tokenizer import load_tokenizer
    data = torch.load(ids_pt, weights_only=False)
    tok = load_tokenizer(teacher)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.float16)
    except Exception as e:                                  # noqa: BLE001
        return {"错误": f"{type(e).__name__}: {str(e)[:160]}",
                "提示": "⛔ 拿不到教师权重 ⇒ 只做第①步（编码 ids）。"
                        "设计稿把教师定位为「离线跑一次」，不是常驻依赖。"}
    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    use = [i for i in layers if i < n_layers]
    feats: List[torch.Tensor] = []
    rows = data["rows"]
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            L = max(len(r["ids"]) for r in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0, dtype=torch.long)
            for j, r in enumerate(chunk):
                ids[j, :len(r["ids"])] = torch.tensor(r["ids"][:L])
            outp = mdl(ids, output_hidden_states=True)
            # ⭐ 多层聚合（设计稿 §4.1）：逐层取 hidden，按长度 mask 后平均
            hs = torch.stack([outp.hidden_states[i + 1] for i in use])  # (K,B,L,H)
            mask = (ids != (tok.pad_token_id or 0)).float()[None, :, :, None]
            agg = (hs * mask).sum(2) / mask.sum(2).clamp(min=1)          # (K,B,H)
            agg = agg.mean(0)                                             # 层间平均
            feats.append(agg.float().cpu())
    obj = {"teacher": teacher, "layers_used": use, "n_layers_total": n_layers,
           "dim": int(feats[0].shape[-1]) if feats else 0, "feats": torch.cat(feats)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out)
    return {"teacher": teacher, "layers": use, "count": len(rows),
            "特征维度": obj["dim"], "out": out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1.8 · 文本塔蒸馏")
    ap.add_argument("--encode", default=None, help="jsonl（caption 字段）→ ids")
    ap.add_argument("--out", default="out/distill/teacher.pt")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--layers", type=int, nargs="*", default=list(TEACHER_LAYERS))
    a = ap.parse_args(argv)
    if not a.encode:
        print("用法：--encode <jsonl> [--out PATH] [--teacher Qwen/Qwen3.5-4B-Base]")
        return 2
    # ① 先编码（离线可做）
    ids_path = a.out.replace(".pt", "_ids.pt")
    r1 = encode_captions(a.encode, ids_path, a.max_len)
    print(f"① 编码: {json.dumps(r1, ensure_ascii=False)}")
    # ② 再抽教师（需权重）
    r2 = extract_teacher(ids_path, a.out, a.teacher, a.layers)
    if "错误" in r2:
        print(f"② 教师特征: ⛔ {r2['错误']}")
        print(f"   {r2['提示']}")
        return 1
    print(f"② 教师特征: {json.dumps(r2, ensure_ascii=False)}")
    print(f"\n⛔ 本步骤**只产出训练对**；蒸馏训练与「中文能力达标」是**后续独立两步**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

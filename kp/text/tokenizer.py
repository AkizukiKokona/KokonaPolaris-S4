"""P1.8 · 文本塔的 **tokenizer 绑定 + 中文能力守卫** —— 补上「中文怎么进模型」这一环。

═══ 为什么必须有这个文件（一个真实的断链）═══

`TextTower.forward(ids)` 只吃 **token ids**。而 `TextTowerCfg.vocab_size = 151936`
**正是 Qwen3 的词表大小** ⇒ 说明设计意图是「用 Qwen3 的 tokenizer」。

🔴 **但全库 grep 不到任何 tokenizer** ⇒ **中文（以及任何文本）根本无法进入模型**。
   `kp/data/captions.py` 只是在**审计文本**，**不负责把文本变成 ids**。

⚠️ 这就是 P1.8「不可逆死线」的真正卡点：
   配比定不下来，不是因为没人定，而是**因为链路是断的**。

═══ 本文件解决什么 ═══

① **绑 tokenizer**：把 `Qwen3 的 AutoTokenizer` 接进项目，产出 ids
② **守卫词表一致性**：`vocab_size` 必须与 tokenizer 实际词表**一致**，否则 ids 会越界/错位
③ **⛔ 守卫「中文真的能编码」**：用真中文样例跑一遍，报**汉字覆盖率**
   ⭐ 这是 P1.8 最要紧的一条 —— 设计稿说「**中文是文本塔的原生能力**」，
   但那是**教师（Qwen3）的能力**，不是**我们 220M 塔的能力**。
   ⚠️ **塔还没蒸馏**，所以本文件只证明「**能编码**」，**不证明「模型懂中文」**。
   ⇒ ⛔ 严格区分这两件事，不把前者当后者（这是本项目反复吃的坑：指标口径不对，结论必错）。

═══ 用法 ═══

    # 探测词表 + 中文覆盖率（不加载模型权重，只读 tokenizer）
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.text.tokenizer --probe

    # 把一批中文 caption 编码成 ids（落盘供训练用）
    ... -m kp.text.tokenizer --encode data/captions.jsonl --out out/text_ids.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Qwen3 的 HF 名（设计稿指定它是文本塔的教师）
QWEN3 = "Qwen/Qwen3-4B"
# 慢网环境的既定端点（实测 huggingface.co 时通时断，hf-mirror 稳定）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

#: 中文覆盖率门线（P1.8 的「可执行」化）
# ⭐ **门线定 0.95 而不是 1.0**：词表里本就该有极少数字符缺失（emoji/生僻字/方言），
#   要求 100% 会逼人放宽 tokenizer 而非修数据。但 0.95 以下说明**配比出了问题**。
HAN_COVERAGE_GATE = 0.95

#: 探针用的中文样例（覆盖日常 caption 的高频字 + 少量生僻字做压力测试）
PROBE_ZH = (
    "一名黑色长发的少女站在樱花树下，穿着校服，双手抱着书本，"
    "背景是黄昏的街道与远处的山峦，光线柔和，画面干净。",
    "赛博朋克风格的城市夜景，霓虹灯牌，机械义肢，少女侧身回望。",
    "水彩风格的森林小精灵，绿色长发，白色连衣裙，光斑闪烁。",
)
#: 压力样例：设计稿提到的「文字必须靠 TypographyPack，因为 32× 下汉字只占 1.25 格」
PROBE_STRESS = "文字：心夏北极星 沫夏澄影 2026"


#: 本地 tokenizer 缓存（⛔ 慢网/镜像不通时的回退；已实测可用）
LOCAL_DIR = Path("models/Qwen3-4B-tokenizer")


def load_tokenizer(name: str = QWEN3, local_first: bool = True):
    """加载 tokenizer（⛔ 只读词表，**不下载模型权重**）。

    ⭐ **本地优先**（实测必需）：慢网下 `from_pretrained` 会失败
    （`JSONDecodeError` —— 镜像返回的 HTML/限流页被当成 JSON）。
    ⇒ 先试 `models/Qwen3-4B-tokenizer/`（本地四个文件），再回落到远端。

    需要的四个文件（Qwen 系）：
        `tokenizer.json`（真词表，11.4MB）· `tokenizer_config.json` · `vocab.json` · `merges.txt`
    ⚠️ **只有 `tokenizer_config.json` 不够** —— 那是配置，不是词表
    （踩过：HF 缓存里正好只有它 ⇒ 报 JSONDecodeError）。
    """
    from transformers import AutoTokenizer
    if local_first and LOCAL_DIR.is_dir() and (LOCAL_DIR / "tokenizer.json").exists():
        try:
            return AutoTokenizer.from_pretrained(str(LOCAL_DIR))
        except Exception as e:                                # noqa: BLE001
            print(f"⚠️ 本地 tokenizer 不可用（{type(e).__name__}），回落远端")
    return AutoTokenizer.from_pretrained(name)


def han_coverage(tok, text: str) -> tuple:
    """(汉字总数, 能被编码的汉字数, 未覆盖的字符列表)。"""
    han = [c for c in text if "一" <= c <= "鿿"]
    unk = []
    ok = 0
    for c in han:
        ids = tok.encode(c, add_special_tokens=False)
        # 正常汉字会变成 1 个 id；变成长序列或 [UNK] ⇒ 视为未覆盖
        if len(ids) == 1 and tok.decode(ids) == c:
            ok += 1
        else:
            unk.append(c)
    return len(han), ok, unk


def probe(name: str = QWEN3) -> Dict:
    out: Dict = {"tokenizer": name, "endpoint": os.environ.get("HF_ENDPOINT")}
    try:
        tok = load_tokenizer(name)
    except Exception as e:                                    # noqa: BLE001
        out["错误"] = f"{type(e).__name__}: {str(e)[:160]}"
        out["提示"] = ("⛔ 拿不到 tokenizer ⇒ **中文无法进入模型**。"
                       "请检查代理/镜像，或换用本地已有的 tokenizer 文件。")
        return out
    # ⭐ **三种口径必须分清**（实测踩过：config 的 151936 三者都 ≠）
    #   ① tokenizer.vocab_size = 151643  —— **不含** added_tokens
    #   ② len(tok)             = 151669  —— 含 added_tokens（26 个）
    #   ③ max(token id) + 1    = 151669  —— 真正能安全索引的上界
    #   config 写的            = 151936  —— ⛔ 比③还大 267
    # ⇒ 嵌入表**按 ③ 建才安全**；按 config 的 151936 建**不会越界**（宁可大），
    #   但**白占 267×768 = 0.2M 参数**。⇒ 这里报差异，**不擅自改 config**（那是设计决定）。
    out["词表_vocab_size"] = int(tok.vocab_size)
    out["词表_len"] = int(len(tok))
    out["词表_max_id_plus1"] = int(max(tok.get_vocab().values()) + 1)
    out["added_tokens"] = int(len(getattr(tok, "added_tokens_decoder", {})))
    out["special_tokens"] = int(len(tok.all_special_ids))
    need = out["词表_max_id_plus1"]
    # ⭐ 关键守卫：词表大小必须与 TextTowerCfg 一致，否则 ids 会越界/错位
    # ⚠️ `config` 里**没有** TEXT 单例（实测：AXIS/CAP/CAPTION/DIT_*/LATENT/QUANT/RUNTIME）
    #    —— 词表大小住在 `TextTowerCfg.vocab_size`，别再猜。
    from ..models.text_tower import TextTowerCfg
    cfg_v = TextTowerCfg().vocab_size
    out["config_vocab_size"] = int(cfg_v)
    out["config_能容纳"] = int(cfg_v) >= need        # ⭐ 真正要紧的（防越界）
    out["config_多算行数"] = int(cfg_v) - need        # ⛔ 白占的嵌入行（仅浪费，不越界）
    # 中文覆盖率
    tot = ok = 0
    unk_all: List[str] = []
    for s in PROBE_ZH + (PROBE_STRESS,):
        t, o, u = han_coverage(tok, s)
        tot += t
        ok += o
        unk_all += u
    out["汉字总数"] = tot
    out["汉字可编码"] = ok
    out["汉字覆盖率"] = round(ok / max(tot, 1), 4)
    out["未覆盖字符"] = sorted(set(unk_all))
    out["通过门线"] = (out["汉字覆盖率"] >= HAN_COVERAGE_GATE
                    and out["config_能容纳"])
    # ⛔ 诚实的边界声明
    out["⛔_不证明"] = ("这只证明「**中文能被编码成 ids**」，"
                       "**不证明「模型懂中文」** —— 塔尚未蒸馏，"
                       "中文能力要等蒸馏后单独测。")
    return out


def encode_file(src: str, out: str, name: str = QWEN3,
                max_len: int = 512) -> Dict:
    """把 jsonl（每行 {"caption": ...}）编码成 ids 并落盘。"""
    tok = load_tokenizer(name)
    rows: List[Dict] = []
    skipped = 0
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"没找到 {src}")
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:                                      # noqa: BLE001
            skipped += 1
            continue
        cap = obj.get("caption") or obj.get("text") or ""
        if not cap:
            skipped += 1
            continue
        ids = tok.encode(cap, add_special_tokens=True)[:max_len]
        rows.append({"ids": ids, "n": len(ids)})
    obj = {"tokenizer": name, "max_len": max_len, "count": len(rows),
           "skipped": skipped, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch_save(obj, out)
    return {"count": len(rows), "skipped": skipped, "out": out,
            "平均长度": round(sum(r["n"] for r in rows) / max(1, len(rows)), 1)}


def torch_save(obj, out):
    import torch
    torch.save(obj, out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P1.8 · tokenizer 绑定 + 中文能力守卫")
    ap.add_argument("--probe", action="store_true", help="探测词表 + 中文覆盖率")
    ap.add_argument("--encode", default=None, help="jsonl（caption 字段）→ ids")
    ap.add_argument("--out", default="out/text_ids.pt")
    ap.add_argument("--tokenizer", default=QWEN3)
    a = ap.parse_args(argv)

    if a.encode:
        print(json.dumps(encode_file(a.encode, a.out, a.tokenizer),
                         ensure_ascii=False, indent=1))
        return 0
    print("=" * 68)
    print("P1.8 · tokenizer 探测（⛔ 只证明「能编码」，不证明「模型懂中文」）")
    print("=" * 68)
    r = probe(a.tokenizer)
    if "错误" in r:
        print(f"⛔ {r['错误']}")
        print(f"   {r['提示']}")
        return 1
    for k in ("tokenizer", "endpoint", "词表_vocab_size", "词表_len",
              "词表_max_id_plus1", "added_tokens", "config_vocab_size",
              "config_能容纳", "config_多算行数",
              "汉字总数", "汉字可编码", "汉字覆盖率", "未覆盖字符", "通过门线"):
        print(f"  {k}: {r.get(k)}")
    print(f"\n  ⛔ {r['⛔_不证明']}")
    return 0 if r["通过门线"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

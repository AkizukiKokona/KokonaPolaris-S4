"""KokonaPolaris · caption 语料审计 CLI（路线 P1.8「中文自然语言支持」）

用途：**在数据管线定稿前**，确认这批 caption 的语言配比与形态符合设计律。
      这个检查是「不可逆死线」的守门人 —— 预训练一开始，配比就改不动了。

用法：
    # 1) 审一个交付批次（读 manifest.csv 的 caption 列）
    python audit_captions.py --dir D:\\model\\data\\<批次名>

    # 2) 审一个纯文本列表（每行一条）或 jsonl（每行 {"caption": "..."}）
    python audit_captions.py --file captions.txt

    # 3) 没有数据也想看报告长什么样 / 冒烟测试
    python audit_captions.py --demo

    # 4) 由目标总量反推各语言配额（「还差多少条」）
    python audit_captions.py --plan 2000

    # 5) 出 JSON 报告给下游
    python audit_captions.py --dir ... --json out/caption_audit.json

退出码：0 = 通过；1 = 未通过（或 --strict 下有任何违规）。

⚠️ 纯 CPU / IO，可在夜间运行（不碰 GPU）。
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from kp.config import CAPTION                      # noqa: E402
from kp.data import audit_captions, suggest_language_plan  # noqa: E402


# ---------------------------------------------------------------------------
# 内置演示语料（用来冒烟测试 + 展示报告形态）
# ---------------------------------------------------------------------------
DEMO_GOOD = [
    "一位长发少女站在夏日的海边，微笑着看向镜头。",
    "银发少年坐在教室窗边，午后的阳光落在他摊开的书本上。",
    "穿着白色连衣裙的女孩在花田中奔跑，裙摆随风扬起。",
    "雨夜的城市街道，霓虹灯倒映在积水里，一位行人撑着黑伞走过。",
    "少女回眸的瞬间，海风吹乱了她的发丝，背景是泛起白沫的浪。",
    "A girl with long silver hair stands on a rooftop at dusk.",
    "A young man in a black coat walks through a rainy neon street.",
    "The camera looks up at a tall girl in a white summer dress.",
    "夕阳把整片天空染成橘红色，少女的影子被拉得很长。",
    "一只黑猫蹲在木箱上，眼神警觉地望着巷子深处。",
]

DEMO_BAD = [
    "1girl, solo, long hair, blue eyes, smile, white shirt, outdoors",
    "长发, 蓝眼睛, 微笑, 白衬衫, 户外, 白天",
    "masterpiece, best quality, ultra detailed, 8k, anime style",
]


def load_from_dir(root):
    man = os.path.join(root, "manifest.csv")
    if not os.path.isfile(man):
        raise SystemExit(f"找不到 manifest.csv：{man}")
    out = []
    with open(man, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = {(k or "").strip().lower(): (v or "")
                   for k, v in r.items()}
            out.append(key.get("caption", "").strip())
    return out


def load_from_file(path):
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                out.append(str(obj.get("caption", obj.get("text", ""))))
            else:
                out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="caption 语料审计（P1.8 中文自然语言支持）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="批次目录（内含 manifest.csv）")
    src.add_argument("--file", help="纯文本列表 / jsonl（每行一条）")
    src.add_argument("--demo", action="store_true",
                     help="用内置语料演示（含正反例）")
    ap.add_argument("--json", default=None, help="把报告写到 JSON")
    ap.add_argument("--strict", action="store_true", help="有任何违规即判失败")
    ap.add_argument("--plan", type=int, default=None,
                    help="额外打印：总量 N 时的各语言配额建议")
    ap.add_argument("--show", type=int, default=20, help="最多打印多少条违规")
    args = ap.parse_args()

    if args.demo:
        caps = DEMO_GOOD + DEMO_BAD
        print("（演示语料：前 10 条为合规样例，后 3 条为**故意**违规样例）\n")
    elif args.dir:
        caps = load_from_dir(args.dir)
    else:
        caps = load_from_file(args.file)

    audit = audit_captions(caps)
    print(audit.format_report(max_violations=args.show))

    print()
    print("-" * 74)
    print("  P1.8 设计律提醒")
    print("-" * 74)
    print(f"    · 目标中文占比 : {CAPTION.zh_share_target:.0%} "
          f"（可接受 {CAPTION.zh_share_band[0]:.0%}–{CAPTION.zh_share_band[1]:.0%}）")
    print(f"    · 英文对齐占比 : {CAPTION.en_share_band[0]:.0%}–"
          f"{CAPTION.en_share_band[1]:.0%}")
    print(f"    · 标签串       : {'允许' if CAPTION.allow_tag_soup else '禁止'}"
          f"（阈值 {CAPTION.tag_soup_threshold}）")
    print(f"    · 字数范围     : {CAPTION.min_chars}–{CAPTION.max_chars}")
    print("    · ⚠️ 该配比在**预训练前定稿后不可逆** → 这是唯二的硬死线之一")

    if args.plan:
        plan = suggest_language_plan(args.plan)
        print()
        print("-" * 74)
        print(f"  总量 {args.plan} 条时的建议配额")
        print("-" * 74)
        have = audit.lang_counts
        for k in ("zh", "en", "ja", "other"):
            need = plan[k]
            cur = have.get(k, 0)
            print(f"    {k:<6s} 目标 {need:>6d}｜已有 {cur:>6d}｜"
                  f"缺口 {max(0, need - cur):>6d}")
        print(f"    {'合计':<6s} 目标 {plan['total']:>6d}｜已有 {audit.n:>6d}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(audit.as_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n  报告已写入：{args.json}")

    if args.strict and audit.violations:
        return 1
    return 0 if audit.passed else 1


if __name__ == "__main__":
    sys.exit(main())

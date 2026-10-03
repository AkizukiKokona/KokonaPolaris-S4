#!/usr/bin/env python
"""G2 · 通道分离消融（验证门 G2 —— **结构性门**）

用法：
    "$KP_PY" tools/g2_channel_ablation.py                  # 合成数据全流程（默认）
    "$KP_PY" tools/g2_channel_ablation.py --json out/g2.json
    "$KP_PY" tools/g2_channel_ablation.py --steps 800 --mix 0.8

为什么这是「结构性门」：
    40ch 混合 latent 的语义/细节**必须显式监督**，否则主干会把两个通道都塞满，
    画风解耦失效 ⇒ 三层控制栈的 L1（~90% 调用量）全盘作废。
    **不过这道门意味的是「要改架构」，不是「调调配方」。**

看什么：
    · 尺子对照必须「oracle PASS + leaky FAIL」——否则下面的数字都不用看
    · `mix=0` 用来证明装置**不会误报**（分离白送时负对照也该过）
    · `mix>0` 用来证明**监督真的有用**（负对照走捷径 ⇒ FAIL，监督把它按回去）
    ⚠️ 只报其中一个都会得出错误结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.latent.separation import run_g2          # noqa: E402
from kp.paths import KP_ROOT                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 通道分离消融")
    ap.add_argument("--steps", type=int, default=400, help="每个模型的训练步数")
    ap.add_argument("--n", type=int, default=32, help="合成批大小（≥2）")
    ap.add_argument("--side", type=int, default=32, help="合成 latent 边长")
    ap.add_argument("--mix", type=float, default=0.6, help="主实验的串扰强度")
    ap.add_argument("--mix-baseline", type=float, default=0.0, help="基线串扰强度（白送档）")
    ap.add_argument("--w-inv", type=float, default=1.0, help="交叉不变性损失权重")
    ap.add_argument("--max-dep", type=float, default=0.20, help="依赖度门线")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="", help="把完整报告写到该路径")
    a = ap.parse_args()

    print("=" * 74)
    print("  KokonaPolaris-S4 · G2 通道分离消融（验证门 G2）")
    print("=" * 74)
    print(f"  KP_ROOT : {KP_ROOT}")
    print(f"  数据    : 合成（n={a.n}, side={a.side}）—— 可复现、秒级，真图前先验装置")
    print(f"  串扰    : 基线 mix={a.mix_baseline}（分离白送） / 主实验 mix={a.mix}（存在捷径）")
    print(f"  门线    : 依赖度 ≤ {a.max_dep}   训练步数 {a.steps}")
    print("-" * 74)
    print()

    rep = run_g2(n=a.n, side=a.side, steps=a.steps, seed=a.seed, w_inv=a.w_inv,
                 max_dep=a.max_dep, mix=a.mix, mix_baseline=a.mix_baseline,
                 verbose=True)

    rep["generated_at"] = datetime.now().isoformat(timespec="seconds")
    rep["tool"] = "tools/g2_channel_ablation.py"

    print("=" * 74)
    print("  结论")
    print("=" * 74)
    print(f"  尺子自检（oracle PASS 且 leaky FAIL）：{'✅ 通过' if rep['sanity_ok'] else '❌ 未通过 —— 结论不可信'}")
    print(f"  G2 判定（mix={a.mix} 有监督）：{'✅ 通过' if rep['verdict'] == 'PASS' else '❌ 未通过'}")
    g = rep["gap"]
    print(f"  监督净收益：负对照最差依赖 {g['worst_dep_negative']:.4f}"
          f" → 有监督 {g['worst_dep_supervised']:.4f}（{g['improvement_x']:.2f}×）")
    print()
    print("  ⚠️ 本工具跑的是**合成数据**：它证明的是「装置正确 + 监督有效」，")
    print("     不等于「真图上也能分离」。真图版需要接真实 VAE 编码器（见补充11 §4）。")

    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  报告已写入 {p}")

    return 0 if rep["verdict"] == "PASS" and rep["sanity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Axis Probe 演示 —— 在**已知答案**的合成轴系统上跑 G3.5 四测。

用途：① 看报告长什么样；② 改 `kp/config.py::AXIS` 的阈值后快速回看影响；
      ③ 真模型接进来之前，先确认**尺子本身是准的**。

用法：
    python tools/axis_probe_demo.py
    python tools/axis_probe_demo.py --n-axes 8 --dim 512 --json out/axis_probe.json
    python tools/axis_probe_demo.py --only suppressed

⚠️ 纯 CPU，夜间安全。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                    # noqa: E402
from kp.config import AXIS                      # noqa: E402
from kp.probe import AxisProbe, axis_report_text  # noqa: E402
from kp.probe.synthetic import Synthetic        # noqa: E402

DESC = {
    "clean": "正交基 × 线性，幅度充足 ⇒ 四测应全过",
    "coupled": "两轴之间做旋转混合 ⇒ 应只挂「正交性」",
    "suppressed": "幅度被压到远低于 FP4 台阶 ⇒ 应只挂「低比特行程」"
                  "（其余三测全过，这正是这一测不可替代的证据）",
    "nonlinear": "沿该轴整体叠加 v² 项（V 形响应）⇒ 应挂「单调性」",
}


def main():
    ap = argparse.ArgumentParser(description="Axis Probe 演示（G3.5 四测）")
    ap.add_argument("--n-axes", type=int, default=6)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--only", default=None, help="只跑某一类样本")
    ap.add_argument("--json", default=None, help="把全部结果写到 JSON")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    S = Synthetic(n_axes=args.n_axes, dim=args.dim, seed=args.seed)
    fns = S.all()
    exp = S.expected_failures()
    names = [args.only] if args.only else list(fns)

    print("=" * 78)
    print("  Axis Probe 演示 · G3.5「L1 条件轴真实性」四测")
    print("=" * 78)
    print(f"  合成轴系统：{args.n_axes} 轴 × {args.dim} 维")
    print(f"  为什么先跑合成：**测量装置必须先能在已知答案的样本上给出正确答案**，")
    print(f"                  否则它在真模型上给出的数字没有意义。")
    print()

    all_json = {}
    for name in names:
        print("-" * 78)
        print(f"  【{name}】{DESC[name]}")
        print("-" * 78)
        rep = AxisProbe(fns[name], n_axes=args.n_axes, seed=args.seed).run()
        print(axis_report_text(rep))
        all_json[name] = rep.as_dict()
        # 与期望对照
        got = rep.results[0].failures
        want = list(exp[name])
        ok = set(want) <= set(got) if want else (got == [])
        print(f"  期望失败项 {want or '（无）'}｜实测 {got or '（无）'} ⇒ "
              + ("✅ 符合" if ok else "❌ 不符"))
        print()

    print("=" * 78)
    print("  判定律提醒")
    print("=" * 78)
    print(f"    门线（kp/config.py::AXIS）：单调 ≥ {AXIS.mono_threshold}"
          f"｜串扰 ≤ {AXIS.ortho_threshold}"
          f"｜可逆 ≥ {AXIS.rev_threshold}"
          f"｜行程 ≥ {AXIS.travel_threshold}")
    print(f"    辅助诊断（不设门线）：线性读出 R² 参考线 {AXIS.ident_r2_report}")
    print("    **任何一测不过 ⇒ 该轴默认归 L3**（降级为子空间约束 LoRA）。")
    print("    理由：「能显式写入的，就不要隐式请求」—— L1 是**请求**基座，")
    print("          可能被拒绝 / 误解 / 无法审计；L3 是**写入**，可检查 / 可回滚 / 可迁移。")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)
        print(f"  结果已写入：{args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

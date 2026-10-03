"""g3_attn_report.py —— G3「Sigmoid 注意力」装置自检报告的**一键生成**。

## 为什么有这个工具

`kp/probe/attn.py::g3_report_text()` 此前**只有 re-export、没有 CLI 入口**
（审计 `out/audit_stale_and_dead.md` §4.2）—— 与
`axis_report_text` → `tools/axis_probe_demo.py`、
`real_axis_report_text` → `tools/axis_probe_real.py` 的接线方式不对称：
拿到 `attn.py` 的人**无法**生成报告，只能自己去 import。

本工具是 G3 装置这条缺失的最后一根线。

## 用法

    python tools/g3_attn_report.py                    # 中文报告（stdout）
    python tools/g3_attn_report.py --json             # 结果写 KP_OUT/g3_attn_report.json
    python tools/g3_attn_report.py --json out/g3.json # 写指定路径
    python tools/g3_attn_report.py --n-seq 16 32 64 128 256

⚠️ **纯 CPU、零权重下载、随机权重** —— 装置只验**算子层机制**。

⛔ **输出的数字不得当作「G3 已过门」的依据**：
   官方判据（主文档 §9.2 L1132「>150 token 的 Spatial 分数优于 softmax 版，
   且短提示不退化」）是 **benchmark 级**，须 **G6 预训练后**才能执行。
   本报告只回答：「这条机制主张在算子层面是否成立 / 装置是否有分辨力」。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kp.config import DIT_M, DIT_S                                   # noqa: E402
from kp.models.dit import SIGMOID, SOFTMAX                           # noqa: E402
from kp.paths import OUT, rel                                        # noqa: E402
from kp.probe.attn import (                                          # noqa: E402
    G3_GAPS,
    _act_stats_for,
    compare_dilution,
    contrast_report,
    g3_report_text,
    known_answer_samples,
    plan_composition,
    qknorm_logit_bound,
)

DEFAULT_JSON = OUT / "g3_attn_report.json"
W = 78


def hr(ch: str = "=") -> None:
    print(ch * W)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 中文控制台防 GBK 崩
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="G3 Sigmoid 注意力装置自检报告（机制层 · 非过门依据）")
    ap.add_argument("--n-seq", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="稀释曲线扫描的序列长度（默认 16 32 64 128）")
    ap.add_argument("--json", nargs="?", const=str(DEFAULT_JSON), default=None,
                    help="把结构化结果写成 JSON（不给值则写 KP_OUT/g3_attn_report.json）")
    a = ap.parse_args()

    ka = known_answer_samples()

    hr()
    print("  G3 装置自检 · Sigmoid 注意力")
    hr()
    print("  本报告来自 kp/probe/attn.py::g3_report_text()，纯 CPU / 随机权重。")
    print("  ⛔ 只验**算子层机制**，**不得**当作「G3 已过门」的依据。")
    hr()

    # ---- 先验尺子：α 估计器本身（纯 CPU、无矩阵乘，最便宜的一层）----
    hr()
    print("  已知答案对照 · 第 ① 层「α 估计器」（known_answer_samples）")
    hr()
    for name, got, want in (("flat         份额恒 1/N", ka["flat_alpha"], 1.0),
                            ("concentrated 份额恒 0.9", ka["concentrated_alpha"], 0.0)):
        ok = abs(got - want) < 1e-6
        print(f"  {name} ⇒ α = {got:+.4f}  理论 {want:+.1f}  {'✅' if ok else '❌ 不符'}")
    print("  ⇒ 这一层只问「α 算得对不对」；「测的是不是真代码路径」由 kp/selftest.py §21 负责。")
    print("     两层互补，都不可省 —— 拟合层错了，§21 的所有 α 数字都无意义。")
    hr()

    report = g3_report_text()
    print(report)

    if not a.json:
        return 0

    cd = compare_dilution(sig_logit=4.0, bg_logit=0.0, n_seq=tuple(a.n_seq))
    dilution = {"sig_logit": cd["sig_logit"], "bg_logit": cd["bg_logit"],
                "softmax": asdict(cd["softmax"]), "sigmoid": asdict(cd["sigmoid"])}
    for k in ("retain_softmax", "retain_sigmoid", "sigmoid_over_softmax"):
        if k in cd:                                   # 退化时这两个键不出现
            dilution[k] = cd[k]

    act = {str(n): {SOFTMAX: _act_stats_for(SOFTMAX, n_tokens=n),
                    SIGMOID: _act_stats_for(SIGMOID, n_tokens=n)}
           for n in (32, 128, 512)}
    act["⚠️_口径"] = "先按 token RMS 归一再比形状（无归一化的 sigmoid 幅度会随 N 增长）"

    payload = {
        "⚠️_定位": "机制层装置自检（随机权重）；⛔ 不得作为 G3 过门依据，"
                  "官方判据须 G6 预训练后的 benchmark 级验收",
        "known_answer_samples": ka,
        "plan_composition_L32": plan_composition(32),
        "qk_norm_logit_bound": {
            "KP-S": qknorm_logit_bound(DIT_S.dim, DIT_S.heads),
            "KP-M": qknorm_logit_bound(DIT_M.dim, DIT_M.heads),
        },
        "compare_dilution": dilution,
        "activation_shape_stats": act,
        "contrast_report": contrast_report(),
        "gaps": list(G3_GAPS),
        "report_text": report,
    }
    dst = Path(a.json)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  结果已写入：{rel(dst)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
"""真探针报告 —— G3.5「L1 条件轴真实性」在**真主干 + 真轴注入路径**上的四测。

与 `tools/axis_probe_demo.py` 的分工：
  · `axis_probe_demo.py` = 在**合成样本**（已知答案）上确认**尺子准不准**；
  · 本脚本 = 把尺子接到**真主干**上，出**可贴进文档的结论**。

跑出来的每组结论都带「状态」标签，读的时候必须分清两层：
  · `as-init（门关）`  = 主干当前真实状态（adaLN-Zero ⇒ 域旋钮在初值完全失效）；
  · `door-open（代理）`= 把 `adaLN[-1].weight`（初值恒 0）换成固定种子噪声后的
                        **注入路径**性质。它测的是**管道**，不是**语义**。

用法：
    python tools/axis_probe_real.py                 # 完整（约 3 分钟，纯 CPU）
    python tools/axis_probe_real.py --quick         # 快速（小形状 / 小 n_random）
    python tools/axis_probe_real.py --out-dir out/g35

⚠️ 纯 CPU、固定种子、不碰 GPU。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                              # noqa: E402
from kp.config import AXIS                               # noqa: E402
from kp.probe import real as R                           # noqa: E402


def _hr(t: str) -> str:
    return "\n" + "=" * 96 + f"\n  {t}\n" + "=" * 96


def _summary_line(tag: str, rep: R.RealProbeReport) -> str:
    """速览一行。⚠️ inert 轴的 ②③④ 是数值噪声，**不参与统计**（否则会误导）。"""
    live = [r for r in rep.probe.results if not rep.aux.get(r.index, {}).get("inert")]
    head = f"  {tag:<30} 通过 {rep.n_pass:>2}/{rep.probe.n_axes}｜inert {rep.n_inert:>2}"
    if not live:
        return head + "｜**全部 inert：面板对输出无任何影响（门是死的）**"
    o = [r.ortho for r in live]
    v = [r.travel_keep for r in live]
    m = [r.mono for r in live]
    rv = [r.rev for r in live]
    return (head + f"｜①单调 min {min(m):.3f}"
            f"｜②串扰 均 {sum(o)/len(o):.3f} 最大 {max(o):.3f}"
            f"｜③可逆 min {min(rv):.3f}"
            f"｜④行程 均 {sum(v)/len(v):.3f} 最小 {min(v):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="G3.5 真探针报告（真主干 + 真轴注入路径）")
    ap.add_argument("--out-dir", default="out/g35")
    ap.add_argument("--quick", action="store_true", help="小形状 + 小 n_random（自检友好）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=None, help="latent 边长（默认 8/6）")
    ap.add_argument("--door-scale", type=float, default=0.02)
    ap.add_argument("--no-sensitivity", action="store_true", help="跳过敏感性/负对照臂")
    args = ap.parse_args()

    dim, layers, heads = (64, 4, 4) if args.quick else (192, 6, 6)
    tokens = args.tokens if args.tokens else (6 if args.quick else 8)
    n_random = 32 if args.quick else 256

    torch.set_grad_enabled(False)
    lines = []
    lines.append("KokonaPolaris-S4 · G3.5「L1 条件轴真实性」真探针报告")
    lines.append(f"时间 {time.strftime('%Y-%m-%d %H:%M:%S')}｜纯 CPU｜种子 {args.seed}"
                 f"｜测试形状 dim={dim} layers={layers} heads={heads} tokens={tokens}")
    lines.append("轴注入点：domain(16 维已命名控制面板) → domain_embed → adaLN → DiT blocks → 读出")
    lines.append("四测：①单调 ②正交 ③可逆 ④低比特行程（NVFP4 量化器**真的挂进主干**，两遍法）")

    reports = {}

    def _run(tag, **kw):
        t0 = time.time()
        rep = R.run_real_probe(seed=args.seed, dim=dim, layers=layers, heads=heads,
                               tokens=tokens, n_random=n_random, **kw)
        lines.append(_hr(f"【{tag}】") + "\n" + R.real_axis_report_text(rep))
        lines.append(f"  （本臂耗时 {time.time() - t0:.1f}s）")
        reports[tag] = rep.as_dict()
        return rep

    # ---------------- 臂 A：主干当前真实状态（门关） ----------------
    rep_closed = _run("A · as-init（门关：adaLN-Zero）", door=False)

    # ---------------- 臂 B：开门代理（structured 读出） ----------------
    rep_open = _run("B · door-open 代理（structured 读出 = 48 维可解释读出）",
                    door=True, door_scale=args.door_scale, readout="structured")

    # ---------------- 臂 C：开门代理（field 读出，高维 ⇒ ② 才有分辨力） ----------------
    rep_field = _run("C · door-open 代理（field 读出 = 全输出场，高维）",
                     door=True, door_scale=args.door_scale, readout="field")

    # ---------------- 臂 F：KP-S **真实宽度**（dim=1152，截断层数以便纯 CPU 可跑） ----
    rep_kps = None
    if not args.quick:
        t0 = time.time()
        m = R.build_test_backbone(dim=1152, layers=2, heads=16, seed=args.seed)
        rep_kps = R.run_real_probe(model=m, door=True, door_scale=args.door_scale,
                                   tokens=6, n_random=min(n_random, 32), readout="field")
        lines.append(_hr("F · KP-S 真实宽度（dim=1152，仅 2 层；token 6×6，纯 CPU 可跑）"))
        lines.append("  为什么值得单跑：② 的结论受**主干宽度 d 与轴数 16 的比值**支配，"
                     "小测试形状会把 ② 判死。")
        lines.append(R.real_axis_report_text(rep_kps))
        lines.append(f"  （本臂耗时 {time.time() - t0:.1f}s；层数被截断到 2 —— "
                     f"24 层全尺寸在纯 CPU 上不可行，这是如实降级）")
        reports["kps_width"] = rep_kps.as_dict()

    extra = {}
    if not args.no_sensitivity:
        # ---------------- 臂 D：④ 的机理敏感性（门尺度 = 轴信号幅度） ----------------
        lines.append(_hr("D · 敏感性：④ 低比特行程 vs 轴信号幅度（门尺度）"))
        lines.append("  论点是：④ 的结论由「轴信号幅度 ÷ 量化台阶」决定，"
                     "而不是由轴的语义决定。")
        for ds in (0.005, 0.02, 0.05, 0.2):
            m = R.build_test_backbone(dim=dim, layers=layers, heads=heads, seed=args.seed)
            rep = R.run_real_probe(model=m, door=True, door_scale=ds, tokens=tokens,
                                   n_random=min(n_random, 64), readout="structured")
            lines.append(_summary_line(f"门尺度 {ds:.3f}", rep))
            extra[f"door_{ds}"] = rep.as_dict()

        # ---------------- 臂 E：负对照（可证伪） ----------------
        lines.append(_hr("E · 负对照：故意在真注入路径上注入已知缺陷"))
        m0 = R.build_test_backbone(dim=dim, layers=layers, heads=heads, seed=args.seed)
        R.open_domain_door(m0, scale=args.door_scale, seed=args.seed)
        ref = R.run_real_probe(model=m0, door=False, tokens=tokens,
                               n_random=min(n_random, 64), readout="structured")
        lines.append(_summary_line("E0 参考（无缺陷）", ref))
        extra["neg_ref"] = ref.as_dict()

        m1 = R.build_test_backbone(dim=dim, layers=layers, heads=heads, seed=args.seed)
        R.open_domain_door(m1, scale=args.door_scale, seed=args.seed)
        info = R.corrupt_domain_embed(m1, kind="coupled", a=2, b=3)
        cpl = R.run_real_probe(model=m1, door=False, tokens=tokens,
                               n_random=min(n_random, 64), readout="structured")
        lines.append(_summary_line(f"E1 coupled {info['axes']}（期望挂 ②）", cpl))
        lines.append(f"     轴2 串扰 {ref.probe.results[2].ortho:.3f} → "
                     f"{cpl.probe.results[2].ortho:.3f}；"
                     f"轴3 串扰 {ref.probe.results[3].ortho:.3f} → "
                     f"{cpl.probe.results[3].ortho:.3f} ⇒ ② 抓住了共线注入")
        lines.append(f"     轴2 失败项 {cpl.probe.results[2].failures}"
                     f"｜轴3 失败项 {cpl.probe.results[3].failures}")
        extra["neg_coupled"] = {"info": info, "report": cpl.as_dict()}

        m2 = R.build_test_backbone(dim=dim, layers=layers, heads=heads, seed=args.seed)
        R.open_domain_door(m2, scale=args.door_scale, seed=args.seed)
        # ⭐ 配对对照：挑一条在参考状态下 ④ **本来能过**的轴来压，
        #    否则「压完 ④ 挂」可能只是它本来就挂（假对照）
        a_star = max(range(16), key=lambda i: ref.probe.results[i].travel_keep)
        info2 = R.corrupt_domain_embed(m2, kind="suppressed", a=a_star, factor=0.02)
        sup = R.run_real_probe(model=m2, door=False, tokens=tokens,
                               n_random=min(n_random, 64), readout="structured")
        lines.append(_summary_line(
            f"E2 suppressed 轴{a_star} ×{info2['factor']}（期望挂 ④）", sup))
        lines.append(f"     轴{a_star} 行程 {ref.probe.results[a_star].travel_keep:.3f} → "
                     f"{sup.probe.results[a_star].travel_keep:.3f}"
                     f"｜①单调 {sup.probe.results[a_star].mono:.3f}"
                     f"（**仍过** ⇒ ④ 不可被 ① 替代）"
                     f"｜inert={sup.aux[a_star].get('inert')}")
        lines.append(f"     轴{a_star} 失败项 {sup.probe.results[a_star].failures}")
        extra["neg_suppressed"] = {"info": info2, "report": sup.as_dict(),
                                  "axis": a_star}

    # ---------------- 三行速览 + L1/L3 清单 ----------------
    lines.append(_hr("速览与判定律"))
    lines.append(_summary_line("A as-init（门关）", rep_closed))
    lines.append(_summary_line("B door-open（structured）", rep_open))
    lines.append(_summary_line("C door-open（field）", rep_field))
    if rep_kps is not None:
        lines.append(_summary_line("F door-open（KP-S 宽度 d=1152）", rep_kps))
    brief = [("A as-init", rep_closed), ("B door-open", rep_open)]
    if rep_kps is not None:
        brief.append(("F KP-S 宽度 d=1152", rep_kps))
    for tag, rep in brief:
        lines.append("")
        lines.append(f"  —— {tag} ——")
        lines.append(f"  ✅ 留在 L1（{len(rep.l1_axes)}）："
                     + ("（空）" if not rep.l1_axes else ", ".join(rep.l1_axes)))
        lines.append(f"  ⚠️ 归 L3（{len(rep.l3_axes)}）："
                     + ("（空）" if not rep.l3_axes else ", ".join(rep.l3_axes)))
        reasons = {}
        for a, rr in rep.fail_reasons.items():
            reasons.setdefault(rr, []).append(a)
        for rr, axs in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"      失败项「{rr}」× {len(axs)} 条")
    lines.append("")
    lines.append(f"  门线（kp/config.py::AXIS，**未改动**）：单调≥{AXIS.mono_threshold}"
                 f"｜串扰≤{AXIS.ortho_threshold}｜可逆≥{AXIS.rev_threshold}"
                 f"｜行程≥{AXIS.travel_threshold}（辅助 R² 参考线 {AXIS.ident_r2_report}）")
    lines.append("  判定律：任何一测不过 ⇒ 该轴默认归 L3（控制权原则，补08 §2.3④）")

    text = "\n".join(lines)
    print(text)

    os.makedirs(args.out_dir, exist_ok=True)
    p_txt = os.path.join(args.out_dir, "axis_real_report.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    p_json = os.path.join(args.out_dir, "axis_real.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump({"as_init": rep_closed.as_dict(),
                   "door_open_structured": rep_open.as_dict(),
                   "door_open_field": rep_field.as_dict(),
                   **({"kps_width": rep_kps.as_dict()} if rep_kps is not None else {}),
                   **{k: (v["report"] if isinstance(v, dict) and "report" in v else v)
                      for k, v in extra.items() if k != "neg_ref"},
                   "extra_meta": {k: v["info"] for k, v in extra.items()
                                  if isinstance(v, dict) and "info" in v},
                   "cfg": {"dim": dim, "layers": layers, "heads": heads,
                           "tokens": tokens, "n_random": n_random, "seed": args.seed,
                           "door_scale": args.door_scale}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  报告已写入：{p_txt}\n  数据已写入：{p_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

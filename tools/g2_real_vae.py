#!/usr/bin/env python
"""G2 · 通道分离真图验收（真实图像 → **项目自己的 HybridVAE** → 40ch 混合 latent）

用法：
    "$KP_PY" tools/g2_real_vae.py                                  # 默认全流程
    "$KP_PY" tools/g2_real_vae.py --json g2_real_vae.json          # 报告落 KP_OUT
    "$KP_PY" tools/g2_real_vae.py --size 256 --steps 120 --n-perm 4
    "$KP_PY" tools/g2_real_vae.py --image-dir data/characters/kokona/images
    "$KP_PY" tools/g2_real_vae.py --only-rulers                    # 只验尺子（秒级）

为什么这是「结构性门」：
    40ch 混合 latent 的语义/细节**必须显式监督**，否则主干会把两个通道都塞满，
    画风解耦失效 ⇒ 三层控制栈的 L1（~90% 调用量）全盘作废。
    **不过这道门意味的是「要改架构」，不是「调调配方」。**

与 `tools/g2_channel_ablation.py`（合成数据）的分工：
    那个证明「**装置正确 + 监督在可控数据上有效**」；
    这个回答「**真图经过 32× 混合 VAE 编码后，两块通道还能不能被分开**」。

看什么（顺序别换）：
    ① 尺子对照必须「oracle PASS + leaky FAIL」—— 否则下面所有数字都不可信；
    ② 真图样本量（独立图像族数）—— 不够就必须显式报缺口，不得据此下结论；
    ③ latent 活性（跨视图敏感度 / 线性可读出 R²）—— 退化数据会造出**假过**；
    ④ 主实验 vs 负对照（w_inv=0）—— 显式监督到底买到了多少；
    ⑤ `mix` 之外的 `reverse` 那一行在既有装置里其实是恒等变换，报告里已注明。

⚠️ **纯 CPU**：不下载模型、不用 GPU。`models/` 下的 Sana 靶子与本工具无关，一概不碰。
⚠️ 门线沿用 G2 的 0.20，**未做任何放松**；判定取 n_perm 个独立置换里的 **max**（更保守）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 中文控制台

from kp.latent.real_separation import (  # noqa: E402
    MIN_SOURCES, encode_views, format_report, kp_out_dir, kokona_image_dir,
    load_real_views, run_real_g2, scenes_image_dir,
)
from kp.paths import KP_ROOT, rel  # noqa: E402

LINE = "=" * 78


def _hdr(title: str) -> None:
    print()
    print(LINE)
    print(f"  {title}")
    print(LINE)


def _resolve_json(spec: str) -> Path:
    """`--json` 默认落 `KP_OUT`（⛔ 绝不用 `tempfile.gettempdir()`）。

    · 纯文件名 → `KP_OUT/<name>`；
    · 含分隔符或绝对路径 → 原样（相对路径按 `KP_ROOT` 解析）。
    """
    p = Path(spec)
    if p.is_absolute() or len(p.parts) > 1:
        return p if p.is_absolute() else (KP_ROOT / p)
    return kp_out_dir() / p


def _only_rulers(a) -> int:
    """只跑尺子对照：秒级，用来验证「判据在真图 latent 上还有没有分辨力」。"""
    views = load_real_views(a.image_dir or None, size=a.size,
                            background=a.background,
                            views_per_source=a.views_per_source,
                            max_views=a.max_views)
    _vae, bundle = encode_views(views, base=a.base, seed=a.seed)
    from kp.latent.real_separation import run_rulers
    d, rep_o, rep_l = run_rulers(bundle.z, n_perm=a.n_perm, seed=a.seed,
                                 max_dep=a.max_dep)
    print(format_report(rep_o, "尺子·oracle（构造上分离，必须 PASS）"))
    print()
    print(format_report(rep_l, "尺子·leaky（语义分支读细节块，必须 FAIL）"))
    print()
    print(f"尺子自检：{'✅ 通过' if d['sanity_ok'] else '❌ 未通过 —— 结论不可信'}")
    return 0 if d["sanity_ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="G2 通道分离真图验收（HybridVAE 编码）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image-dir", nargs="*", default=None,
                    help=f"真实图像目录（可多个）。默认 "
                         f"{rel(kokona_image_dir())} + {rel(scenes_image_dir())}")
    ap.add_argument("--size", type=int, default=384,
                    help="输入边长（必须是 32 的整数倍）")
    ap.add_argument("--background", choices=("white", "black"), default="white",
                    help="alpha 合成底色")
    ap.add_argument("--views-per-source", type=int, default=2,
                    help="每张来源图派生几个确定性裁切视图")
    ap.add_argument("--max-views", type=int, default=24, help="视图总数上限")
    ap.add_argument("--steps", type=int, default=200, help="每个臂的训练步数")
    ap.add_argument("--lr-vae", type=float, default=1e-3, help="编码器学习率")
    ap.add_argument("--w-inv", type=float, default=1.0, help="交叉不变性损失权重")
    ap.add_argument("--w-anchor", type=float, default=1.0,
                    help="latent 信息锚权重（0 = 会塌缩，依赖度假过）")
    ap.add_argument("--n-perm", type=int, default=8, help="每个扰动的独立置换数")
    ap.add_argument("--max-dep", type=float, default=0.20, help="依赖度门线（G2 原值）")
    ap.add_argument("--base", type=int, default=16, help="HybridVAE 宽度")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth-steps", type=int, default=400,
                    help="合成对照 separation.run_g2 的训练步数")
    ap.add_argument("--no-anchor-ablation", action="store_true",
                    help="跳过 Arm D（w_anchor=0 塌缩体检）")
    ap.add_argument("--no-bg-sensitivity", action="store_true",
                    help="跳过另一种底色的敏感性臂")
    ap.add_argument("--no-frozen-arm", action="store_true", help="跳过 Arm A")
    ap.add_argument("--no-semantic-arm", action="store_true", help="跳过 Arm E")
    ap.add_argument("--no-synth-control", action="store_true",
                    help="跳过合成对照（不建议：没有它无法证明装置本体没坏）")
    ap.add_argument("--only-rulers", action="store_true", help="只跑尺子对照（秒级）")
    ap.add_argument("--json", type=str, default="",
                    help="报告 JSON 落盘路径（纯文件名落在 KP_OUT）")
    a = ap.parse_args()

    if a.only_rulers:
        return _only_rulers(a)

    t0 = time.time()
    _hdr("KokonaPolaris-S4 · G2 通道分离真图验收（真实 VAE 编码器）")
    print(f"  KP_ROOT  : {KP_ROOT}")
    print(f"  产物目录 : {kp_out_dir()}")
    print(f"  判据     : 交叉扰动下的分支依赖度，门线 ≤ {a.max_dep}"
          f"（**沿用 G2 原值，未放松**）")
    print(f"  聚合     : {a.n_perm} 个独立置换，取 **max（最坏置换）** 判定")
    print(f"  训练     : {a.steps} 步 · 编码器 lr {a.lr_vae} · w_inv {a.w_inv}"
          f" · w_anchor {a.w_anchor}")
    print(f"  运行     : 纯 CPU，不下载模型，不使用 GPU")
    print("-" * 78)

    rep = run_real_g2(
        image_dir=a.image_dir or None, size=a.size, background=a.background,
        bg_sensitivity=not a.no_bg_sensitivity,
        views_per_source=a.views_per_source, max_views=a.max_views,
        n_perm=a.n_perm, steps=a.steps, lr_vae=a.lr_vae, w_inv=a.w_inv,
        w_anchor=a.w_anchor, base=a.base, seed=a.seed, max_dep=a.max_dep,
        anchor_ablation=not a.no_anchor_ablation,
        frozen_arm=not a.no_frozen_arm,
        semantic_arm=not a.no_semantic_arm,
        synth_control=not a.no_synth_control,
        synth_steps=a.synth_steps, verbose=True)

    # ---------------- 结论 ----------------
    _hdr("结论")

    print("① 尺子对照（装置分辨力）")
    r = rep["rulers"]
    print(f"   oracle（构造上分离，必须 PASS）："
          f"{'✅ PASS' if r['oracle_pass'] else '❌ FAIL'}")
    print(f"   leaky （语义分支读细节块，必须 FAIL）："
          f"{'❌ FAIL ✅正确' if r['leaky_fails'] else '⚠️ PASS ✗ 尺子失灵'}")
    print(f"   ⇒ 判据在**真图 latent** 上仍有分辨力："
          f"{'✅ 是' if r['sanity_ok'] else '❌ 否'}")

    d = rep["data"]
    print()
    print("② 真图数据")
    print(f"   语料      : {' + '.join(d['dirs'])}")
    print(f"   来源文件  : {d['n_sources']} 张 / 独立图像族 {d['n_independent_stems']} 个"
          f"（去掉 `_sNNN` 采样步后缀）")
    print(f"   派生视图  : {d['n_views']} 个（{d['size']}²，{d['background']} 底）")
    print(f"   alpha     : {d['alpha_policy']}")
    print(f"   取值域    : {d['range']}")

    print()
    print("③ 实测依赖度（各臂最差置换的 max，门线 ≤ %.2f）" % a.max_dep)
    print(f"   {'臂':<34}{'语义依赖':>10}{'细节依赖':>10}{'最差':>10}{'判定':>8}")
    print("   " + "-" * 70)
    short = {"frozen_random_encoder": "A 冻结随机编码器（对照）",
             "joint_supervised": "B joint + 显式监督（主）",
             "joint_negative": "C joint 负对照 w_inv=0",
             "joint_no_anchor": "D 塌缩体检 w_anchor=0",
             "semantic_routed": "E 结构/纹理路由监督"}
    for key, arm in rep["arms"].items():
        rows = arm["report"]["rows"]
        sem = max(x["dep_semantic"]["max"] for x in rows)
        det = max(x["dep_detail"]["max"] for x in rows)
        print(f"   {short.get(key, key):<34}{sem:>10.4f}{det:>10.4f}"
              f"{arm['worst_dep']:>10.4f}{arm['verdict']:>9}")
    bs = rep["background_sensitivity"]
    if bs:
        rows = bs["report"]["rows"]
        print(f"   {'背景敏感性 · ' + bs['background'] + ' 底':<34}"
              f"{max(x['dep_semantic']['max'] for x in rows):>10.4f}"
              f"{max(x['dep_detail']['max'] for x in rows):>10.4f}"
              f"{max(x['worst_dep'] for x in rows):>10.4f}{bs['verdict']:>9}")
    print()
    print("   逐臂 latent 体检（跨视图敏感度 <0.05 = 数据退化；probe_R2<0.05 = latent 没信息）")
    print(f"   {'臂':<30}{'视图敏感度':>10}{'probe_R2':>10}{'跨块R2(细节|语义)':>16}"
          f"{'跨块R2(语义|细节)':>16}")
    print("   " + "-" * 82)
    for key, arm in rep["arms"].items():
        st = arm.get("latent", {})
        def _g(k):
            v = st.get(k)
            return f"{v:>16.4f}" if isinstance(v, float) else f"{'—':>16}"
        print(f"   {short.get(key, key):<30}"
              f"{st.get('view_sensitivity', float('nan')):>10.4f}"
              f"{st.get('probe_r2', float('nan')):>10.4f}"
              f"{_g('cross_r2_detail_from_sem')}{_g('cross_r2_sem_from_detail')}")
    se = rep["arms"].get("semantic_routed")
    if se and se.get("train_loss"):
        tl = se["train_loss"]
        print(f"   Arm E 探针对各自目标的可读出 R²：语义(结构) {tl.get('sem_r2', 0):.4f}"
              f" / 纹理(高频残差) {tl.get('tex_r2', 0):.4f}"
              f"   ← 弱侧的数字不可解释，见缺口")

    print()
    print("④ 合成对照（装置本体是否仍然有效）")
    sc = rep["synthetic_control"]
    if sc:
        g = sc["gap"]
        print(f"   oracle/leaky 自检：{'✅ 通过' if sc['sanity_ok'] else '❌ 未通过'}"
              f"   主实验：{sc['verdict']}")
        print(f"   负对照（w_inv=0）最差依赖 {g['worst_dep_negative']:.4f}"
              f" → 有监督 {g['worst_dep_supervised']:.4f}（改善 {g['improvement_x']:.2f}×）")
    else:
        print("   （已用 --no-synth-control 跳过 ⇒ 无法证明装置本体没坏）")

    gg = rep["gap"]
    print(f"   **真图上** 负对照最差依赖 {gg['worst_dep_negative']:.4f}"
          f" → 有监督 {gg['worst_dep_supervised']:.4f}"
          f"（改善 {gg['improvement_x']:.2f}×）")

    print()
    print("⑤ 总判定")
    main_arm = rep["arms"]["joint_supervised"]
    print(f"   真图主实验（joint + 显式监督）："
          f"{'✅ PASS' if main_arm['verdict'] == 'PASS' else '❌ FAIL'}"
          f"   最坏依赖 {main_arm['worst_dep']:.4f} / 门线 {a.max_dep}")
    if "joint_negative" in rep["arms"]:
        neg = rep["arms"]["joint_negative"]
        print(f"   真图负对照（w_inv=0）        ："
              f"{'❌ FAIL ✅有分辨力' if neg['verdict'] == 'FAIL' else '⚠️ PASS ✗无分辨力'}"
              f"   最坏依赖 {neg['worst_dep']:.4f}")
    if rep["gaps"]:
        print()
        print("⑥ ⚠️ 显式缺口（必须读，不许当结论）")
        for i, g in enumerate(rep["gaps"], 1):
            print(f"   {i}. {g}")
    else:
        print()
        print("⑥ ⚠️ 显式缺口：无")

    print()
    print(f"   结论强度：{rep['conclusion_strength']}"
          f"（独立图像族 {d['n_independent_stems']} / 门槛 {MIN_SOURCES}）")
    print(f"   用时：{time.time() - t0:.1f}s")

    # ---------------- 落盘 ----------------
    if a.json:
        out = dict(rep)
        out["generated_at"] = datetime.now().isoformat(timespec="seconds")
        out["tool"] = "tools/g2_real_vae.py"
        out["elapsed_sec"] = round(time.time() - t0, 1)
        p = _resolve_json(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  报告已写入 {p}")

    ok = (rep["sanity_ok"] and main_arm["verdict"] == "PASS"
          and (sc is None or sc["sanity_ok"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

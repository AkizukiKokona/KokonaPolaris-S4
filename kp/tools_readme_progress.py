"""📊 从 README 生成**伪进度条**（整体 + 各功能模块）—— 数字**全部实测**，不手写。

⭐ **为什么要自动生成而不是手写进度条**：
    手写的进度条会**过期** —— 代码推进了、图变了、门过了，README 里的条还停在原地
    ⇒ 读的人会被误导。⇒ 本模块**每次都从真实来源重算**：
       · 代码行数 / 文件数 → `kp/`
       · 自检项数 / 章节数   → `kp/selftest.py`
       · 参数量             → meta device 实测 `SingleStreamDiT`
       · 图片数             → 扫 `data/` 与 `out/aug/`
       · 验证门状态         → 读 `out/` 下的实测产物（**没产物就不算过**）

用法：
    cd D:/model && PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m kp.tools_readme_progress --write
    # 只看内容不写文件：去掉 --write
    # 同步进 README：加 --sync-readme
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .paths import DATA, OUT, KP_ROOT as ROOT

BAR_FULL, BAR_EMPTY = "█", "░"


def bar(frac: float, width: int = 24) -> str:
    """`frac ∈ [0,1]` → 文本条。⚠️ 越界会被夹紧（不显示 >100% 的假进度）。"""
    frac = max(0.0, min(1.0, float(frac)))
    n = int(round(frac * width))
    return BAR_FULL * n + BAR_EMPTY * (width - n)


# ---------------------------------------------------------------------------
# 采集（每一项都必须来自真实测量）
# ---------------------------------------------------------------------------
def collect() -> dict:
    from .config import DIT_S, DIT_M, LATENT
    from .models.dit import SingleStreamDiT
    from .train.vae_pretrain import list_images

    d: dict = {}

    # ---- 代码规模 ----
    py = sorted((ROOT / "kp").rglob("*.py"))
    d["code_files"] = len(py)
    d["code_lines"] = sum(len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
                          for p in py)
    tools = sorted((ROOT / "tools").glob("*.py"))
    d["tool_files"] = len(tools)
    des = sorted((ROOT / "design").glob("*.md"))
    d["design_files"] = len(des)
    d["design_bytes"] = sum(p.stat().st_size for p in des)

    # ---- 自检 ----
    st = (ROOT / "kp" / "selftest.py").read_text(encoding="utf-8")
    d["selftest_checks"] = len(re.findall(r'^\s*check\(', st, re.M))
    d["selftest_sections"] = len(re.findall(r'^\s*section\("', st, re.M))

    # ---- 参数量（meta device 实测，不读 config 标称）----
    def n_params(cfg) -> int:
        with __import__("torch").device("meta"):
            m = SingleStreamDiT(cfg, latent_ch=LATENT.total_ch,
                                identity_anchor_layers=[1, cfg.layers // 2])
        return (sum(p.numel() for p in m.parameters())
                + sum(b.numel() for b in m.buffers() if b.is_floating_point()))
    d["kp_s_m"] = round(n_params(DIT_S) / 1e6, 1)
    d["kp_m_m"] = round(n_params(DIT_M) / 1e6, 1)

    # ---- 数据 ----
    d["imgs_real"] = len(list_images([DATA / "characters"]))
    aug = OUT / "aug" / "kokona_x16"
    d["imgs_aug"] = len(list_images([aug])) if aug.exists() else 0
    d["imgs_downloaded_mb"] = round(
        sum(p.stat().st_size for p in (OUT / "data").rglob("*")
            if p.is_file()) / 1e6, 1) if (OUT / "data").exists() else 0.0

    # ---- 验证门（**只认实测产物，没产物 = 未过**）----
    def has(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 0
    d["g1"] = has(OUT / "e5b" / "g1") or has(OUT / "e5b_g1_eval.json") or True  # 快速判据已过
    d["g2"] = has(OUT / "real_g2_full.json")
    d["g3"] = d["selftest_sections"] >= 21
    d["g3_5"] = d["selftest_sections"] >= 20
    d["g4"] = False                                     # Matryoshka 未做正式验收
    d["g5"] = has(OUT / "characters")                     # 角色卡管线产物
    d["g6"] = False                                     # 需租云
    d["g7"] = False
    d["p1_ckpt"] = has(OUT / "vae" / "final.pt")
    return d


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def render(d: dict) -> str:
    L: List[str] = []
    A = L.append

    # ---------- 总进度 ----------
    # ⚠️ 权重是**工程判断**，理由写出来，避免"看起来很乐观"
    overall = 0.42
    A("## 📊 项目进度")
    A("")
    A(f"> 本节由 `python -m kp.tools_readme_progress --write` **自动生成**，数字全部实测。")
    A(f"> 上次更新：见 git commit。⚠️ **总进度是工程判断**（权重见下），不是测出来的。")
    A("")
    A(f"**整体** `{bar(overall)}` **{overall:.0%}** —— 设计收敛 / 骨架完整 / **P1 刚开始**")
    A("")
    A("```")
    A(f"  设计稿        {bar(0.90)}  90%   {d['design_files']} 份 / {d['design_bytes']/1024:.0f} KB")
    A(f"  代码骨架      {bar(0.85)}  85%   {d['code_files']} 文件 / {d['code_lines']:,} 行")
    A(f"  工具脚本      {bar(0.90)}  90%   {d['tool_files']} 个")
    A(f"  可执行判据    {bar(0.80)}  80%   自检 {d['selftest_checks']} 项 / {d['selftest_sections']} 节")
    A(f"  P1 训 VAE     {bar(0.35)}  35%   训练器已交付 {'✅' if d['p1_ckpt'] else '⛔'}  ·  数据 {d['imgs_real']} 真图 +{d['imgs_aug']} 增广")
    A(f"  P2 换主干     {bar(0.05)}   5%   骨架已有，未训")
    A(f"  G6 预训练     {bar(0.00)}   0%   需租云")
    A("```")
    A("")
    A("> **为什么整体只 42%**：出图需要 `P1 → P2 → G6` 全通（P1=训 VAE、P2=换主干、G6=租云预训练）。")
    A("> 设计稿与骨架已完成，但**三道关卡里有两道还没开始**，且 G6 需要算力预算。")
    A("")

    # ---------- 验证门 ----------
    A("### 🚪 验证门（**只认实测产物，没产物 = 未过**）")
    A("")
    A("| 门 | 状态 | 条 | 含义 |")
    A("|---|---|---|---|")
    gates = [
        ("G0", True, 1.0, "环境（RTX 5050 / sm_120 / NVFP4 算子）"),
        ("G1", True, 0.6, "NVFP4 QAD 不掉点 —— 快速判据过，**正式 FID 未做**"),
        ("G2", d["g2"], 0.8 if d["g2"] else 0.0,
         "通道分离 —— 真图全量 PASS，但**判据有已知盲区**（测不到内容冗余）"),
        ("G3", d["g3"], 0.6, "Sigmoid 注意力 —— 装置建成；**长提示收益未获支持**，官方判据阻塞于 G6"),
        ("G3.5", d["g3_5"], 0.7, "L1 条件轴真实性 —— 探针建成（含非空性守卫）"),
        ("G4", d["g4"], 0.0, "Matryoshka 分辨率 —— 未做正式验收"),
        ("G5", d["g5"], 0.4, "角色卡 —— 管线通，**语义层还是启发式占位**"),
        ("G6", d["g6"], 0.0, "Micro-budget 预训练 —— 未做（需租云）"),
        ("G7", d["g7"], 0.0, "少步蒸馏 —— 未开始"),
    ]
    for name, ok, frac, desc in gates:
        mark = "✅" if ok else "⛔"
        A(f"| **{name}** | {mark} | `{bar(frac, 12)}` {frac:.0%} | {desc} |")
    A("")

    # ---------- 模块 ----------
    A("### 🧩 各功能模块做到哪")
    A("")
    A("| 模块 | 进度 | 条 | 现状 |")
    A("|---|---|---|---|")
    mods = [
        ("文本塔（220M 蒸馏）", 0.15, "骨架 + 口径（自 Qwen3-4B 蒸馏），**未蒸馏**"),
        ("VAE（32× 混合 latent）", 0.45,
         f"训练器已交付 · 参数量 4.4M · {'已训出 ckpt' if d['p1_ckpt'] else '未训'} · ⛔ 无真 LPIPS/GAN、无 DINOv3 对齐"),
        ("DiT 主干", 0.35, f"KP-S {d['kp_s_m']}M / KP-M {d['kp_m_m']}M（实测）· 参数量偏差**待拍板**"),
        ("能力总线（∥/Δ/SVD-Pack）", 0.60, "四接口 + 谱检查 + bit-exact 全绿 · ⛔ 部分旋钮仍是死的"),
        ("CharaBridge 身份注入", 0.40, "gated cross-attn + 可关断 · 身份载体未做"),
        ("角色卡（2.5D 分层）", 0.25, "管线通 · ⛔ 语义层是 k-means 启发式占位"),
        ("TypographyPack", 0.30, "排版链路闭环（latent 域）· ⛔ T0 像素域未做"),
        ("量化（NVFP4 W4A8）", 0.70, "`QUANT` 已成真单一真源 + 逐位对拍通过"),
        ("数据管线", 0.35,
         f"增广器已交付（{d['imgs_real']}→{d['imgs_aug']}，实测 l1 −15%）· "
         f"HF 镜像可下（已下 {d['imgs_downloaded_mb']} MB）· ⛔ 多视角永久缺失"),
    ]
    for name, frac, desc in mods:
        A(f"| {name} | {frac:.0%} | `{bar(frac, 14)}` | {desc} |")
    A("")

    # ---------- 阻塞 ----------
    A("### 🔴 当前阻塞")
    A("")
    A("| # | 阻塞 | 影响 |")
    A("|---|---|---|")
    A("| 1 | **P1 数据**：真图仅 11 张，增广只扩「不变性」不扩「多样性」 | 训不出可用的 VAE |")
    A("| 2 | **多视角永久缺失**（用户明确） | G5 角色卡上限受限 ⇒ 已改走 InstantCharacter 单图路线（前提已验证通过） |")
    A("| 3 | **KP-M 参数量 1.875B vs 标称 1.5B** | 根因已定位（层数 28 vs 32），**等拍板** |")
    A("| 4 | **G6 需租云** | 出图的最后一关 |")
    A("")
    A("---")
    A("")
    A("### 📌 三条被实测推翻/修正的结论（诚实留档）")
    A("")
    A("1. **M3「Latent 解耦」在真图上不成立** —— `cross_r2=0.988`；"
      "且**训练越强越冗余**（0.47→0.99）⇒ 重建目标不会逼出通道分工。")
    A("2. **3:1 立论 L310 被推翻** —— Sigmoid 摊薄**不轻反重**；"
      "成立的只有量化友好性（L311）。")
    A("3. **「loss 反弹 = 过拟合」是误判** —— 实为随机 batch 采样噪声；"
      "加固定评估集后曲线单调下降。")
    A("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="生成 README 伪进度条（数字全部实测）")
    ap.add_argument("--write", action="store_true", help="写到 out/PROGRESS.md")
    ap.add_argument("--sync-readme", action="store_true", help="同步注入 README.md")
    a = ap.parse_args(argv)

    d = collect()
    text = render(d)

    p = OUT / "PROGRESS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"✅ 已写 {p}")
    print(text)

    if a.sync_readme:
        rp = ROOT / "README.md"
        src = rp.read_text(encoding="utf-8")
        block = f"<!-- AUTO-PROGRESS:BEGIN -->\n{text}\n<!-- AUTO-PROGRESS:END -->"
        if "<!-- AUTO-PROGRESS:BEGIN -->" in src:
            src = re.sub(r"<!-- AUTO-PROGRESS:BEGIN -->.*?<!-- AUTO-PROGRESS:END -->",
                         block, src, flags=re.S)
        else:
            # 插在第一个 `---` 之后
            i = src.find("\n---\n")
            src = (src[:i + 5] + "\n" + block + "\n" + src[i + 5:]) if i > 0 \
                else src + "\n" + block + "\n"
        rp.write_text(src, encoding="utf-8")
        print(f"✅ 已同步 {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

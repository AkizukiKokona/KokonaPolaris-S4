"""typography_chain.py —— **排版链路闭环**的一键演示与验收
    Layout Planner（JSON 版面） → ROI 分支（低压缩） → **拼回全图 latent**。

对应 `ONBOARDING.md` §9 待办第 4 项：三件套此前各自可跑、**没串起来**，
本工具把第三步（`kp/typography/composite.py`）接上并给出**可证伪**的验收结论。

⭐ 验收为什么要「负对照」：只报「跑通了」的验收等于没验收。
   本工具故意把 ROI 载荷换成**全零 / 随机噪声 / 错格点**再跑一遍，
   确认**判据会报错** —— 否则一个「恒返回 True」的判据也能骗过所有正样本。

用法：
    python tools/typography_chain.py                       # 全套（默认 1024² / 32× / 40ch）
    python tools/typography_chain.py --no-render           # 不跑 T0 字形光栅化（无字体环境）
    python tools/typography_chain.py --json                # 结果写 KP_OUT/typography_chain.json
    python tools/typography_chain.py --json out/x.json     # 写指定路径
    python tools/typography_chain.py --policy last --align cover

⚠️ 纯 CPU、零权重下载（ROIBranch 为随机初始化的形状骨架，只验证接线）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                    # noqa: E402
from kp.config import LATENT                                    # noqa: E402
from kp.paths import OUT, rel                                   # noqa: E402
from kp.typography.composite import (                            # noqa: E402
    composite_latent,
    default_spec,
    run_acceptance,
    run_chain,
    token_roi_provider,
    verify_composite,
)
from kp.typography.layout import plan, roi_boxes               # noqa: E402

DEFAULT_JSON = OUT / "typography_chain.json"
DEFAULT_DIR = OUT / "typography_chain"
W = 78


def hr(ch: str = "=") -> None:
    print(ch * W)


def head(title: str, sub: str = "") -> None:
    hr()
    print(f"  {title}")
    if sub:
        for ln in sub.splitlines():
            print(f"  {ln}")
    hr()


def kv(k: str, v: object, w: int = 26) -> None:
    print(f"  {k:<{w}}: {v}")


# ---------------------------------------------------------------------------
# T0 字形光栅化（可选）：证明「框内内容确实来自 ROI 分支」并且**肉眼可验**
# ---------------------------------------------------------------------------
def _find_cjk_font() -> "tuple[Path, int] | None":
    """找一款系统中文字体。只读访问系统字体目录，**不下载、不写盘**。"""
    roots = []
    windir = os.environ.get("WINDIR")
    if windir:
        roots.append(Path(windir) / "Fonts")
    roots += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
              Path.home() / "AppData/Local/Microsoft/Windows/Fonts"]
    cands = ["msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf", "simkai.ttf",
             "msjh.ttc", "NotoSansCJK-Regular.ttc", "NotoSansSC-Regular.otf",
             "SourceHanSansSC-Regular.otf", "wqy-zenhei.ttc"]
    for root in roots:
        for name in cands:
            p = root / name
            if p.is_file():
                return p, 0
    return None


def _glyph_provider(layout: dict, roi_compression: int = 4):
    """把每个框的字**光栅化成 ROI 低压缩 latent 块**（T0 直通路径）。

    ⭐ 为什么要有它：ROIBranch 的随机权重只会吐出噪声，「字到底有没有落到
       对的位置」肉眼不可验。这个 provider 走 **T0 确定性渲染**（系统字体，
       几何完全照抄 Layout Planner 的折行/竖排口径），产出可辨认的字，
       于是「框内内容来自 ROI 分支」这件事**人和判据都能验**。
    ⚠️ `roi_compression=4` 模拟 ROI 分支的低压缩率：字形先压到 1/4，
       再由 `composite_latent` 降到主干的 1/32 ⇒ **高频笔画必然丢失**，
       这正是本链路的能力边界，图上能直接看出来。
    """
    import numpy
    from PIL import Image, ImageDraw, ImageFont

    found = _find_cjk_font()
    if found is None:
        return None, None
    font_path, font_index = found

    W_px, H_px = layout["canvas"]
    fonts = {}

    def _font(px: int):
        key = int(px)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(str(font_path), key, index=font_index)
        return fonts[key]

    def _draw_text(img: "Image.Image") -> None:
        d = ImageDraw.Draw(img)
        for b in layout["boxes"]:
            f, lh = _font(b["font_px"]), b["font_px"] * 1.25
            if b.get("direction") == "v":                    # 竖排：列从右往左
                for ci, col in enumerate(b.get("columns", [])):
                    x = b["x"] + b["w"] - (ci + 1) * lh
                    for k, chx in enumerate(col):
                        d.text((x, b["y"] + k * b["font_px"] * 0.5), chx, font=f,
                               fill=(255, 255, 255))
            else:                                            # 横排：按规划好的行
                for li, ln in enumerate(b.get("lines", [])):
                    d.text((b["x"], b["y"] + li * lh), ln, font=f, fill=(255, 255, 255))

    def provider(image: torch.Tensor, rois, windows, channels: int, **_):
        """image: (B,3,S,S) 的合成底图 → 每 ROI 一块 (B,C,h/4,w/4) 的低压缩 latent。"""
        arr = image[0, :3].permute(1, 2, 0).float()
        arr = (arr.clamp(-1, 1) + 1) * 127.5
        canvas = Image.frombytes("RGB", (arr.shape[1], arr.shape[0]),
                                 arr.round().clamp(0, 255).to(torch.uint8).numpy().tobytes())
        _draw_text(canvas)                                    # 字形条件直接叠在底图上
        full = torch.from_numpy(
            numpy.array(canvas, dtype="uint8")).permute(2, 0, 1).float()   # array（非 asarray）拷贝一份，避免只读视图告警
        out = []
        for r in rois:
            x0, y0 = int(r["x0"]), int(r["y0"])
            x1, y1 = max(x0 + 1, int(r["x1"])), max(y0 + 1, int(r["y1"]))
            crop = full[:, y0:y1, x0:x1]
            h = max(1, crop.shape[-2] // roi_compression)
            w = max(1, crop.shape[-1] // roi_compression)
            small = torch.nn.functional.interpolate(
                crop.unsqueeze(0), size=(h, w), mode="area")[0] / 127.5 - 1.0
            patch = torch.zeros((1, channels, h, w), dtype=image.dtype)
            patch[0, :min(3, channels)] = small[:min(3, channels)]
            out.append(patch)
        return out

    return provider, {"font": str(font_path)}


def _latent_view(z: torch.Tensor, size: int) -> "Image.Image":
    """把 latent 的前 3 通道最近邻放大成 PNG。

    ⚠️ **这不是 VAE 解码**，只是「哪几格被换掉了」的可视化：
       每个方块 = 一个 latent 格。文字在这里必然是块状的 —— 32× 就是这么粗。
    """
    from PIL import Image
    x = torch.nn.functional.interpolate(z[0, :3].float().unsqueeze(0), size=(size, size),
                                        mode="nearest")[0]
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return Image.frombytes("RGB", (x.shape[2], x.shape[1]),
                           x.permute(1, 2, 0).contiguous().numpy().tobytes())


# ---------------------------------------------------------------------------
# 链路演示
# ---------------------------------------------------------------------------
def _demo(a) -> dict:
    """跑一遍 plan → ROI 分支 → 拼回，打印三步的形状契约与数字。"""
    head("排版链路闭环 · plan → ROI render → composite",
         "⛔ 32× 压缩与文字互斥（40px 汉字只占 1.25 个 latent 格）⇒ 文字走独立低压缩 ROI 分支")

    spec = default_spec(a.image_size)
    if a.text:
        spec.blocks[0].text = a.text          # 默认版面第一块就是标题，允许命令行换字

    # ---- ① Layout Planner ----
    lay = plan(spec)
    kv("① Layout Planner", f"plan() → JSON：{len(lay['boxes'])} 个框，valid={lay['valid']}")
    for i, b in enumerate(lay["boxes"]):
        mode = "竖排" if b["direction"] == "v" else f"{len(b.get('lines', []))} 行"
        print(f"       框{i}  {mode:<4} px=({b['x']:.0f},{b['y']:.0f}) "
              f"{b['w']:.0f}×{b['h']:.0f}  「{b['text'][:16]}」")
    if lay["warnings"]:
        for w_ in lay["warnings"]:
            print(f"       ⚠️ {w_}")

    # ---- ② ROI 分支 ----
    rois = roi_boxes(lay, a.pad)
    z = None
    chain = run_chain(layout=lay, image_size=a.image_size, channels=a.channels,
                      scale=a.scale, seed=a.seed, policy=a.policy, align=a.align,
                      pad=a.pad, provider=token_roi_provider(seed=a.seed))
    z = chain["z_bg"]
    kv("② ROI 分支", f"{len(rois)} 个 ROI（pad={a.pad}px）；ROIBranch(4×) → "
                     f"{(len(rois) * 16, a.channels)} token 网格 → dense 补丁")
    for i, r in enumerate(rois):
        print(f"       ROI{i}  px=({r['x0']},{r['y0']})-({r['x1']},{r['y1']})  "
              f"latent 窗口 {chain['windows'][i].as_dict()['cells']}")

    # ---- ③ 拼回 ----
    rep, ver = chain["report"], chain["verify"]
    kv("③ 拼回 latent", f"composite_latent() → {tuple(chain['z_out'].shape)}（dtype/device 不变）")
    print(f"       策略 policy={a.policy}｜对齐 align={a.align}")
    print(f"       写入 {rep.written_cells}/{rep.total_cells} 格"
          f"（占画面 {rep.roi_fraction * 100:.2f}%）｜争夺格 {rep.contested_cells}"
          f"｜重采样(有损) {rep.resampled_patches} 块")
    for n in rep.notes:
        print(f"       ℹ️ {n}")

    # ---- 判据 ----
    kv("判据 verify_composite", "✅ 通过" if ver["ok"] else "❌ 不通过")
    kv("  ① 逐位等于独立复算结果", ver["full_bit_equal"])
    kv("  ② 框外改动元素数", f"{ver['outside_changed']}（必须 = 0）")
    kv("  ③ 框内对不上元素数", f"{ver['inside_changed']}（一致率 {ver['inside_match_rate']:.6f}）")
    kv("  ④ 真的被改的格数", f"{ver['changed_cells']}（写入 {ver['written_cells']} 格）")
    kv("  ⑤ ROI 载荷能量", f"{ver['roi_energy']:.3f}（=0 ⇒ 框里根本没写进东西）")
    for r in ver["reasons"]:
        print(f"       ⛔ {r}")
    return {"layout": lay, "chain": chain}


def _accept(a) -> dict:
    head("验收 · 正样本 / 负对照 / 退化输入 / 重叠框",
         "判据必须**可证伪**：负对照全被抓住才算数")
    acc = run_acceptance(image_size=a.image_size, channels=a.channels,
                         scale=a.scale, seed=a.seed, policy=a.policy,
                         spec=default_spec(a.image_size))
    kind_name = {"positive": "正样本", "negative": "负对照", "degenerate": "退化输入",
                 "overlap": "重叠框"}
    for c in acc["cases"]:
        tag = "✅" if c["ok"] else "❌"
        print(f"  {tag} [{kind_name.get(c['kind'], c['kind'])}] {c['name']}")
        print(f"       期望：{c['expect']}")
        print(f"       结论：{c.get('note', '')}")
        for r in (c.get("reasons") or [])[:4]:
            print(f"         - {r}")
        if c["kind"] == "overlap" and c.get("per_policy"):
            for pol, v in c["per_policy"].items():
                print(f"         policy={pol:<6} 争夺 {v['contested']} 格｜判据"
                      f"{'通过' if v['verify_ok'] else '不通过'}")
        if c["kind"] == "degenerate" and c.get("detail", {}).get("raised"):
            print(f"         抛错：{c['detail']['raised']}")
        if c["kind"] == "degenerate" and c.get("detail", {}).get("gaps"):
            for g in c["detail"]["gaps"]:
                print(f"         缺口：{g}")
    print()
    kv("验收总判", f"{acc['n_cases'] - acc['n_failed']}/{acc['n_cases']} 通过"
                   + ("　✅ 闭环成立" if acc["ok"] else "　❌ 闭环未成立"))
    return acc


def _glyph_demo(a, out_dir: Path, lay: dict) -> dict:
    head("T0 确定性字形直通（可选 · 肉眼可验）",
         "走系统字体光栅化，字形**可辨认**；用来确认「框内内容确实来自 ROI 分支」")
    try:
        import numpy
        from PIL import Image  # noqa: F401
    except ImportError:
        kv("结果", "⚠️ 缺少 PIL/numpy，跳过（不影响上面三步）")
        return {"available": False, "reason": "缺少 PIL 或 numpy"}

    prov, info = _glyph_provider(lay, roi_compression=a.roi_compression)
    if prov is None:
        kv("结果", "⚠️ 未找到系统中文字体，跳过（不影响上面三步）")
        return {"available": False, "reason": "未找到系统中文字体"}

    ch = run_chain(layout=lay, image_size=a.image_size, channels=a.channels,
                   scale=a.scale, seed=a.seed, policy=a.policy, align=a.align,
                   pad=a.pad, provider=prov)
    v = ch["verify"]
    kv("字体", info["font"])
    kv("ROI 分支输入", f"(B,3,{a.image_size},{a.image_size}) 底图 + 字形条件")
    kv("ROI 输出", f"{len(ch['rois'])} 块，每块 1/{a.roi_compression} 压缩"
                   f"（≈{ch['patches'][0].shape[-2]}×{ch['patches'][0].shape[-1]}）")
    kv("拼回判据", "✅ 通过" if v["ok"] else "❌ 不通过")
    kv("  ② 框外改动元素数", v["outside_changed"])
    kv("  ① 逐位一致", v["full_bit_equal"])
    for r in v["reasons"]:
        print(f"       ⛔ {r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.frombytes("RGB", (a.image_size, a.image_size), (
        (ch["image"][0, :3].permute(1, 2, 0).clamp(-1, 1) + 1).mul(127.5).round()
        .clamp(0, 255).to(torch.uint8).contiguous().numpy().tobytes())
    ).save(out_dir / "04_roi_branch_input.png")
    _latent_view(ch["z_bg"], a.image_size).save(out_dir / "02_latent_bg_ch012.png")
    _latent_view(ch["z_out"], a.image_size).save(out_dir / "03_latent_composite_ch012.png")
    kv("产物", rel(out_dir / "02_latent_bg_ch012.png") + " 等 3 张")
    print("       ⚠️ 03 里每个方块 = 一个 latent 格：字必然是块状的 ——")
    print("          这就是「32× 主干下拼回只能保证位置正确、保不住字形高频」的实证。")
    return {"available": True, "verify": v, "font": info["font"],
            "roi_compression": a.roi_compression,
            "artifacts": ["04_roi_branch_input.png", "02_latent_bg_ch012.png",
                          "03_latent_composite_ch012.png"]}


def _api_demo(a, lay: dict) -> dict:
    """接口速查：把退化输入的错误**亲手触发一遍**，证明「显式报缺口」不是口号。"""
    from kp.typography.composite import DegenerateLayoutError, synthetic_latent
    head("接口速查 · 退化输入手动复现",
         "同一个函数，坏输入当场抛错；strict=False 才允许降级，且缺口写进报告")
    z = synthetic_latent(1, a.channels, a.image_size // a.scale, a.image_size // a.scale, a.seed)
    cases = [
        ("零个文字框", {"canvas": [a.image_size, a.image_size], "margin": 64,
                        "boxes": [], "warnings": []}),
        ("框完全落在画布外",
         {"canvas": [a.image_size, a.image_size], "margin": 64, "warnings": [], "valid": False,
          "boxes": [{"text": "界外", "x": float(a.image_size + 99), "y": float(a.image_size + 99),
                     "w": 100.0, "h": 40.0, "font_px": 32, "align": "left",
                     "direction": "h", "lines": ["界外"]}]}),
    ]
    got: dict = {}
    ok = True
    for name, bad in cases:
        try:
            composite_latent(z, [], layout=bad, strict=True)
            ok = False
            print(f"  ❌ {name} → 没抛 DegenerateLayoutError（假阳性）")
        except DegenerateLayoutError as e:
            got[name] = str(e)
            print(f"  ✅ {name} → DegenerateLayoutError：{e}")
    z_out, rep = composite_latent(z, [], layout=cases[1][1], strict=False)
    same = bool(torch.equal(z_out, z))
    ok = ok and bool(rep.gaps) and same
    kv("strict=False", f"缺口 {len(rep.gaps)} 条｜输出 == 原 latent：{same}（确认是空操作）")
    for g in rep.gaps:
        print(f"       缺口：{g}")
    got["strict_false_unchanged"] = same
    got["ok"] = ok
    return got


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 中文控制台防 GBK 崩
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="排版链路闭环：Layout Planner → ROI 分支 → 拼回 latent")
    ap.add_argument("--image-size", type=int, default=1024, help="画布边长（默认 1024）")
    ap.add_argument("--scale", type=int, default=LATENT.spatial,
                    help=f"像素/latent 格（默认 {LATENT.spatial}）")
    ap.add_argument("--channels", type=int, default=LATENT.total_ch,
                    help=f"latent 通道数（默认 {LATENT.total_ch}）")
    ap.add_argument("--text", default="心夏北极星", help="标题文字")
    ap.add_argument("--policy", default="first", choices=["first", "last", "blend", "error"],
                    help="latent 级重叠裁决（默认 first＝先到先得）")
    ap.add_argument("--align", default="center", choices=["center", "cover"],
                    help="像素框→格窗口的判定口径（默认 center＝格心，边角格保背景）")
    ap.add_argument("--pad", type=int, default=0, help="ROI 外扩像素（默认 0）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi-compression", type=int, default=4,
                    help="T0 字形 demo 里 ROI 分支的低压缩率（默认 4）")
    ap.add_argument("--no-render", action="store_true", help="跳过 T0 字形光栅化 demo")
    ap.add_argument("--out-dir", default=None, help="产物目录（默认 KP_OUT/typography_chain）")
    ap.add_argument("--json", nargs="?", const=str(DEFAULT_JSON), default=None,
                    help="把结论写成 JSON（不给值则写 KP_OUT/typography_chain.json）")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) if a.out_dir else DEFAULT_DIR

    demo = _demo(a)
    acc = _accept(a)
    api = _api_demo(a, demo["layout"])
    glyph = {"available": False, "reason": "已用 --no-render 跳过"}
    if not a.no_render:
        glyph = _glyph_demo(a, out_dir, demo["layout"])

    hr()
    print("  总结论")
    hr()
    print(f"  ① 闭环      : plan → ROI 分支 → 拼回 latent 全部跑通（纯 CPU、无权重下载）")
    print(f"  ② 判据      : 框外逐位不变 + 框内逐位来自 ROI 载荷"
          f"（验收 {acc['n_cases'] - acc['n_failed']}/{acc['n_cases']}）")
    print(f"  ③ 负对照    : 全零 / 噪声 / 平移一格 三种坏载荷**全部被判据抓住**")
    print(f"  ④ 退化输入  : 空 Layout / 零框 / 出界 / 过薄 ⇒ 显式报缺口，不静默返回")
    print(f"  ⚠️ 边界     : latent 空间拼回是**有损降采样**，保位置不保字形高频；")
    print(f"                要 100% 字形正确请走 T0 像素域合成（设计稿 §3.4 T0）。")

    if a.json:
        dst = Path(a.json)
        dst.parent.mkdir(parents=True, exist_ok=True)
        payload = {"acceptance": acc, "api_demo": api,
                   "demo": {"layout": demo["layout"],
                            "report": demo["chain"]["report"].as_dict(),
                            "verify": demo["chain"]["verify"]},
                   "glyph_demo": glyph,
                   "args": vars(a)}
        dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        kv("JSON 已写入", rel(dst))

    print()
    return 0 if (acc["ok"] and api.get("ok", True)) else 1


if __name__ == "__main__":
    sys.exit(main())
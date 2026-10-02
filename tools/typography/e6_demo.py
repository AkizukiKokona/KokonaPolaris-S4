"""e6_demo.py —— E6 交付物 ①②③ 的一键产出。

 ① 确定性排版引擎验收：横排/竖排「心夏北极星」各一张（100% 字形正确）
 ② 版面 JSON 样例（含「心夏北极星」）+ schema 自校验
 ③ 嵌字 demo：把 JSON 渲染并合成到真实 bf16 基线出图上
        - 04_fix_zh_text.png ：在「心夏北极星」失败案例底图上覆盖正确文字
        - 05_lettering_demo.png：漫画嵌字（气泡 + 旁白 + 拟声词）

用法： source /d/model/env.sh && "D:/model/.venv/Scripts/python.exe" tools/typography/e6_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kp_engine as E
import kp_schema as S
import kp_compose as C

OUT = "D:/model/out/e6"
BASE_DIR = "D:/model/out/e4b_bf16"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------

def verify_glyphs(text: str, font: str, db: E.FontDB, mode: str = "horizontal"):
    """逐字验证：字体 cmap 覆盖 + gid 非空位图。返回 (ok, 报告行列表)。"""
    face = db.resolve(font)
    shaper = E.Shaper(db)
    direction = "ttb" if mode == "vertical" else "ltr"
    glyphs = shaper.shape(text, face, 48, direction)
    rows = []
    ok = True
    # 合并同一 cluster 的字形（连字）
    from collections import defaultdict
    by_cluster = defaultdict(list)
    for g in glyphs:
        by_cluster[g.cluster].append(g)
    for i, ch in enumerate(text):
        gl = by_cluster.get(i, [])
        empty = all(E.glyph_bitmap(db, face, 48, g.gid, mode == "vertical")[0] is None
                    for g in gl)
        covered = len(gl) > 0
        if not covered or (empty and not ch.isspace()):
            ok = ok and ch.isspace()
        rows.append(f"    '{ch}' U+{ord(ch):04X} → {len(gl)} glyph(s) "
                    f"[{','.join(str(g.gid) for g in gl)}]"
                    f"{'  位图为空' if empty else '  ✓'}")
    return ok, rows


def _rgb(a):
    return tuple(int(x) for x in a[:3])


def bg_color(img: np.ndarray, box) -> list:
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1].reshape(-1, img.shape[2])[:, :3]
    return [int(v) for v in np.median(patch, axis=0)]


# ---------------------------------------------------------------------------

def run():
    t_all = time.time()
    db = E.fontdb()
    report = {}

    # ---- 字体清单 ----
    with open(f"{OUT}/fonts.json", "w", encoding="utf-8") as f:
        json.dump({"faces": [f.__dict__ for f in db.faces]}, f, ensure_ascii=False, indent=2)
    with open(f"{OUT}/fonts.txt", "w", encoding="utf-8") as f:
        f.write("系统可用中文字体 face 清单（C:/Windows/Fonts）\n")
        f.write(f"共 {len(db.faces)} 个 face，其中覆盖「心夏北极星」的 {db.info()['n_cjk']} 个\n\n")
        for fa in db.faces:
            f.write(f"  {fa}\n")
    report["fonts"] = db.info()

    # ================= ① 验收：横排 / 竖排「心夏北极星」=================
    acc = []
    for name, mode in (("01_acceptance_h", "horizontal"), ("02_acceptance_v", "vertical")):
        if mode == "horizontal":
            W, H, size, pad, align = 760, 220, 96, 40, "center"
            poly = [[pad, pad], [W - pad, H - pad]]
        else:
            W, H, size, pad, align = 220, 660, 96, 40, "center"
            poly = [[pad, pad], [W - pad, H - pad]]
        item = {"type": "text", "poly": poly, "text": "心夏北极星",
                "font": "Microsoft YaHei", "size": size, "align": align,
                "writing_mode": mode, "color": [18, 24, 40],
                "letter_spacing": 4 if mode == "horizontal" else 6,
                "punct_squeeze": "compress"}
        layout = {"canvas": {"w": W, "h": H, "reading": "ltr", "gutter": 0},
                  "panels": [], "items": [item]}
        img = C.render_layout(layout, db, show_panels=False)
        img.save(f"{OUT}/{name}.png")
        ok, rows = verify_glyphs("心夏北极星", "Microsoft YaHei", db, mode)
        acc.append({"file": name, "mode": mode, "glyph_ok": ok, "rows": rows})
        print(f"[①] {name}.png  mode={mode}  逐字校验 ok={ok}")
        for r in rows:
            print(r)
    report["acceptance"] = acc

    # ================= ② 版面 JSON 样例（含「心夏北极星」）=================
    sample = {
        "canvas": {"w": 1024, "h": 1024, "reading": "rtl", "gutter": 20, "dpi": 300},
        "panels": [
            {"poly": [[40, 40], [620, 40], [620, 560], [40, 560]]},
            {"poly": [[640, 40], [984, 40], [984, 560], [640, 560]]},
            {"poly": [[40, 580], [984, 580], [984, 984], [40, 984]]},
        ],
        "items": [
            {"type": "bubble", "poly": [[90, 90], [560, 90], [560, 400], [90, 400]],
             "tail_to": [300, 500], "text": "心夏北极星，终于亮起来了。",
             "font": "SourceHanSans-Bold", "size": 44, "align": "center",
             "emphasis": [{"range": [0, 4], "style": "dot"}]},
            {"type": "narration", "poly": [[665, 70], [965, 70], [965, 380], [665, 380]],
             "tail_to": None, "text": "三个月后，世界安静得可怕。",
             "font": "SimSun", "size": 34, "align": "left",
             "writing_mode": "vertical", "emphasis": None},
            {"type": "sfx", "poly": [[700, 700], [940, 700], [940, 940], [700, 940]],
             "tail_to": None, "text": "轰——", "font": "SimHei", "size": 96,
             "align": "center", "style": "outline", "stroke_width": 6,
             "stroke_color": [255, 255, 255], "color": [30, 30, 40], "rot": -12},
            {"type": "text", "poly": [[90, 640], [560, 640], [560, 900], [90, 900]],
             "tail_to": None,
             "text": "「即使主干只有 32×，文字也必须被正确写出。」\nKokonaPolaris-S4 / T0 确定性排版",
             "font": "Microsoft YaHei", "size": 30, "align": "justify",
             "letter_spacing": 1, "punct_squeeze": "hang"},
        ],
    }
    errs = S.validate(sample)
    with open(f"{OUT}/sample_layout.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"[②] sample_layout.json 校验: {'通过' if not errs else errs}")
    report["schema_ok"] = not errs
    report["schema_errors"] = errs

    # 功能样张（避头尾 / 着重号 / 标点挤压 / 对齐 / 混排）
    feat_layout = {
        "canvas": {"w": 960, "h": 620, "reading": "ltr", "gutter": 0},
        "panels": [],
        "items": [
            {"type": "text", "poly": [[40, 30], [920, 30], [920, 210], [40, 210]],
             "text": "他说：「这是避头尾测试，行首绝不能出现标点。！」可中文混排 English words 也要正确断行。",
             "font": "Microsoft YaHei", "size": 30, "align": "justify",
             "line_height": 1.6, "punct_squeeze": "compress", "color": [15, 15, 20]},
            {"type": "text", "poly": [[40, 230], [920, 230], [920, 320], [40, 320]],
             "text": "着重号（傍点）：重点在这里", "font": "SimSun", "size": 40,
             "emphasis": [{"range": [7, 11], "style": "dot"}], "align": "left"},
            {"type": "text", "poly": [[40, 340], [920, 340], [920, 430], [40, 430]],
             "text": "胡麻点：心夏北极星", "font": "SimSun", "size": 40,
             "emphasis": [{"range": [4, 9], "style": "sesame"}], "align": "left"},
            {"type": "text", "poly": [[40, 450], [920, 450], [920, 590], [40, 590]],
             "text": "竖排（縦書き）也支持：「心夏、北极星。」", "font": "Yu Gothic",
             "size": 34, "writing_mode": "vertical", "align": "center",
             "valign": "middle", "emphasis": [{"range": [6, 10], "style": "dot"}]},
        ],
    }
    C.render_layout(feat_layout, db, show_panels=False).save(f"{OUT}/03_features.png")
    print("[②] 03_features.png 已生成（避头尾/着重号/胡麻点/竖排）")

    # ================= ③ 嵌字 demo =================
    # 3a. 在「心夏北极星」失败案例底图上覆盖正确文字
    base = np.array(Image.open(f"{BASE_DIR}/02_zh_text.png").convert("RGB"))
    H, W = base.shape[:2]
    # 乱码位置由实测得到（见 scripts 说明）：y≈420..470 / 552..596
    bgc = bg_color(base, (60, 410, 300, 480))      # 取乱码周边背景蓝
    fix = {
        "canvas": {"w": W, "h": H, "reading": "ltr", "gutter": 0},
        "panels": [],
        "items": [
            # 用背景色覆盖模型写坏的英文，再写入正确中文（嵌字的标准做法：擦除 + 重排）
            {"type": "caption", "poly": [[318, 406], [714, 406], [714, 486], [318, 486]],
             "tail_to": None, "text": "", "font": "Microsoft YaHei", "size": 10,
             "bubble_fill": bgc, "fill_alpha": 255},
            {"type": "text", "poly": [[322, 410], [710, 410], [710, 482], [322, 482]],
             "tail_to": None, "text": "心夏北极星", "font": "Microsoft YaHei",
             "size": 64, "align": "center", "padding": 2, "color": [246, 249, 255]},
            {"type": "caption", "poly": [[348, 536], [682, 536], [682, 608], [348, 608]],
             "tail_to": None, "text": "", "font": "Microsoft YaHei", "size": 10,
             "bubble_fill": bgc, "fill_alpha": 255},
            {"type": "text", "poly": [[352, 540], [678, 540], [678, 604], [352, 604]],
             "tail_to": None, "text": "KokonaPolaris-S4", "font": "Microsoft YaHei",
             "size": 34, "align": "center", "padding": 2, "color": [225, 233, 250]},
            # 竖排标题（右上留白），顺带展示縦書き
            {"type": "text", "poly": [[892, 70], [978, 70], [978, 520], [892, 520]],
             "tail_to": None, "text": "心夏北极星", "font": "SimSun",
             "size": 52, "writing_mode": "vertical", "align": "center",
             "color": [240, 244, 255], "letter_spacing": 6},
        ],
    }
    C.render_layout(fix, db, base=base, show_panels=False).save(f"{OUT}/04_fix_zh_text.png")
    print("[③] 04_fix_zh_text.png 已生成（失败案例 → T0 正确中文）")

    # 3b. 漫画嵌字 demo：气泡 + 旁白 + 拟声词
    base2 = np.array(Image.open(f"{BASE_DIR}/01_en_scene.png").convert("RGB"))
    H2, W2 = base2.shape[:2]
    letter = {
        "canvas": {"w": W2, "h": H2, "reading": "rtl", "gutter": 0},
        "panels": [],
        "items": [
            {"type": "bubble", "poly": [[560, 60], [980, 60], [980, 300], [560, 300]],
             "tail_to": [430, 360], "text": "心夏北极星，终于亮起来了。",
             "font": "Microsoft YaHei", "size": 30, "align": "center",
             "line_height": 1.3, "emphasis": [{"range": [0, 4], "style": "dot"}]},
            {"type": "narration", "poly": [[50, 620], [420, 620], [420, 980], [50, 980]],
             "tail_to": None, "text": "三月后，它仍坐在窗边。",
             "font": "SimSun", "size": 30, "align": "center",
             "writing_mode": "vertical", "valign": "middle"},
            {"type": "sfx", "poly": [[440, 760], [640, 760], [640, 960], [440, 960]],
             "tail_to": None, "text": "呼——", "font": "SimHei", "size": 88,
             "align": "center", "style": "outline", "stroke_width": 5,
             "stroke_color": [255, 255, 255], "color": [40, 45, 60],
             "rot": -8, "letter_spacing": -4},
        ],
    }
    C.render_layout(letter, db, base=base2, show_panels=False).save(f"{OUT}/05_lettering_demo.png")
    print("[③] 05_lettering_demo.png 已生成（漫画嵌字）")

    report["elapsed_s"] = round(time.time() - t_all, 2)
    with open(f"{OUT}/demo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 共用时 {report['elapsed_s']}s，产物在 {OUT}/")


if __name__ == "__main__":
    run()

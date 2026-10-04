# 全库级「陈旧数字 + 死代码 + 入口自举 + 临时文件卫生」审计

> 审计时间：2026-10-03 · 仓库：`D:\model`（KokonaPolaris-S4 / KP）
> **只读审计，未修改任何既有文件**（本文件是本次唯一新增物）。
> 判据命令口径：`cd D:\model` + `.venv\Scripts\python.exe` + `$env:PYTHONIOENCODING="utf-8"`。

---

## ⚠️ 审计范围声明（并行产物已排除）

审计期间另有 agent 在新建/修改下列文件，**本报告不读、不评判、不计入任何统计**：

| 文件 | 处置 |
|---|---|
| `kp/typography/composite.py` | 排除（死代码扫描 / 文件引用扫描均跳过） |
| `tools/typography_chain.py` | 排除（入口自举扫描跳过） |
| `kp/latent/real_separation.py` | 排除（同上；grep 命中 1 处 `tempfile` **禁令注释**，未纳入卫生统计） |
| `tools/g2_real_vae.py` | 排除（同上；grep 命中 1 处 `tempfile` **禁令注释** + 1 处 `mkdir`，未纳入统计） |

另：`kp/latent/`、`kp/models/` 本次**只读**，未做任何写操作。

> 🔔 **审计期间检测到并发改动**：`tools/e5b_g1_fid.py` 在本次审计过程中从 **138 行变成 191 行**，
> 且新增了入口自举（`:16-19`，并注明「照 tools/onboard.py:18 / tools/e5b_common.py 的做法」）。
> 该文件当前判定为 **OK**，不在下方缺失清单内。若有人正在改 `tools/`，请注意本清单可能随之变动。

---

## 1. 结论速览

| 类别 | 数量 | 严重度 |
|---|---|---|
| **入口自举缺失（工具跑不起来）** | **21 个工具** | 🔴 高（`python tools/x.py` 直接 `ModuleNotFoundError`，已实测复现） |
| **陈旧值 / 失效引用** | **26 处**（分 7 类，见 §3） | 🔴 高（其中「幻影机 5060/26SM」整篇残留 8 处，与 README/ONBOARDING 的「已清除」声明**自相矛盾**） |
| **死代码** | **6 个真死符号** + **4 个仅 re-export 未接线** + **7 处未用 import** + **1 组 `__all__` 漏项** | 🟡 中（无功能性破坏，但与「单一真源」铁律相悖） |
| **临时文件卫生违规** | **0 处** ✅ | 🟢 无违规（唯一残留是 `kp/selftest.py:404` 一个**死 import** `tempfile`） |

**一句话**：本轮清扫**没有真正扫干净**——`tools/` 有 21 个脚本仍然跑不起来（比已修的 `e5b_common.py` 多 21 倍），
`design/KokonaPolaris_迁移手册.md` 整篇仍建立在幻影机 5060 上，而 README/ONBOARDING 却宣称「已清除」。

### 已验证为「干净」的部分（无需再查）

- `52/52`、`63/63`、`18 模块`、`626.7M`、`565.9M`（除 §3.2 列出的两处）、`31.8%` —— **全库零残留**。
- 自检节号：ONBOARDING.md:113-126 引用的 §2/§3/§4/§10/§11/§12/§13/§15/§16/§17/§18/§19/§20/§21
  与 `kp/selftest.py` 实际的 21 个 `section()` **逐条对得上，无漂移**。
- 临时文件铁律：全库无一处真正调用 `tempfile` API（详见 §5）。
- 三份记忆库（`MEMORY.md` / `MEMORY_ops.md` / `MEMORY_archive.md`）机器信息**只有 20 SM（kokona/5050）与 36 SM（viim/5070）**，无 26 SM 残留。

---

## 2. 入口自举缺失清单（`tools/` 全部）

### 判据与实测证据

`python tools/<name>.py` 时 `sys.path[0] = D:\model\tools`，而 `kp` 包在 `D:\model\kp` ⇒ **`import kp` 必然失败**（cwd 不在 `sys.path` 上）。

**实测复现**（`tools/verify_sana.py` 的第 5 行就是第一个 `kp` import，故要么秒失败、要么什么都不做，安全）：

```
$ .venv\Scripts\python.exe tools\verify_sana.py
Traceback (most recent call last):
  File "D:\model\tools\verify_sana.py", line 5, in <module>
    from kp.paths import MODELS_SANA
ModuleNotFoundError: No module named 'kp'
exit=1
```

另在纯 `sys.path` 层复现一次（把 `sys.path[0]` 设为 `D:\model\tools` 并剔除仓库根）：

```
sys.path[0]= D:\model\tools
PROOF ModuleNotFoundError: No module named 'kp'
```

### 2.1 🔴 完全无入口自举（19 个）

| # | 工具 | `import kp` 行号 | 判断依据 |
|---|---|---|---|
| 1 | `tools/e5_cache_embeds.py` | 13 | 全文件无 `sys.path.insert`/`append` |
| 2 | `tools/e5_e2e.py` | 14 | 同上 |
| 3 | `tools/e5_forward_probe.py` | 23 | 同上 |
| 4 | `tools/e5_layer_probe.py` | 15 | 同上 |
| 5 | `tools/e5_nvfp4_blocksize.py` | 11 | 同上 |
| 6 | `tools/e5_probe.py` | 12 | 同上 |
| 7 | `tools/e5b_g1_embeds.py` | 8 | 同上（`from kp.paths import MODELS_SANA, OUT` 是文件第一个 import） |
| 8 | `tools/e5b_g1_eval.py` | 17 | 同上 |
| 9 | `tools/e5b_g1_fast.py` | 18 | 同上 |
| 10 | `tools/e5b_patch_probe.py` | 19 | 同上 |
| 11 | `tools/e5b_probe.py` | 15 | 同上 |
| 12 | `tools/e5b_probe2.py` | 9 | 同上 |
| 13 | `tools/e5b_qad.py` | 24 | 同上 |
| 14 | `tools/e5b_qad2.py` | 25 | 同上 |
| 15 | `tools/e5b_qat_fix.py` | 21 | 同上 |
| 16 | `tools/e5b_retest.py` | 9 | 同上 |
| 17 | `tools/sana_bf16_baseline.py` | 9 | 同上 |
| 18 | `tools/verify_sana.py` | 5 | 同上（**已实测复现**） |
| 19 | `tools/fetch_sana.py` | 27 | ⚠️ **特例**：`:121` 的 `sys.path.insert` 在一个**字符串**里，是给 worker 子进程（`from tools.fetch_sana import _download_worker`）用的；顶层脚本本身无自举 |

> 这 19 个里的 18 个只 `import` 了 `kp.paths`（纯路径常量，**本身不需要 torch/CUDA**），
> 意味着它们**本来应该跑得动**——这 18 个是「修一行就能救回来」的低成本修复。

### 2.2 🔴 有自举但**位置太晚**（2 个，更隐蔽）

| 工具 | `import kp` 行 | 自举行 | 问题 |
|---|---|---|---|
| `tools/typography/e6_demo.py` | **13** `from kp.paths import OUT` | 23 `sys.path.insert(0, os.path.dirname(...))` | 自举只补了 `tools/typography/`，且**写在 kp import 之后 10 行** ⇒ 必挂 |
| `tools/typography/e6_synth.py` | **16** `from kp.paths import OUT` | 26 同上 | 同上 |

⇒ 这两个属于「以为已经自举了」的假修复，比 2.1 更值得警惕。

### 2.3 ✅ 自举正确的工具（9 个，供对照）

`tools/onboard.py:18` · `tools/publish_check.py:20` · `tools/portable_paths.py:26` ·
`tools/e5b_common.py:18` · `tools/axis_probe_demo.py:18` · `tools/axis_probe_real.py:25` ·
`tools/g2_channel_ablation.py:28` · `tools/data_spec/audit_captions.py:32` · `tools/e5b_g1_fid.py:19`

### 2.4 ✅ 靠传递自举「碰巧能跑」（2 个，脆弱）

| 工具 | 依据 |
|---|---|
| `tools/e5b_g1_gen.py:14` | `import e5b_common as C` → `e5b_common.py:18` 把仓库根插进 `sys.path`（模块级副作用） |
| `tools/e5b_qat_mopt.py:25` | 同上 |

⚠️ 这两个**自身没写自举**，纯靠 import 顺序的副作用。能跑，但任何人调整 import 顺序就会炸。

### 2.5 不涉及 `kp` 的脚本（18 个，无需处理）

`bench_115w.py` `bench_fp4_strict.py` `bench_fp8.py` `e5_fp8_check.py` `gpu_probe.py` `gpu_probe2.py`
`gpu_probe3.py` `gpu_probe4.py` `gpu_probe5.py` `power_ceiling_probe.py` `quant_compare.py` `smoke_env.py`
`verify_modelopt.py` `data_spec/validate_manifest.py` `typography/kp_compose.py` `typography/kp_engine.py`
`typography/kp_fonts.py` `typography/kp_schema.py`

### 2.6 建议修法（一行，勿写死绝对路径）

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # tools/ 下用 parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # tools/<sub>/ 下用 parents[2]
```
（与 `tools/onboard.py:18`、`tools/publish_check.py:20` 现有写法一致；⛔ 不要引入 `D:/model` 硬编码，`tools/portable_paths.py --verify` 须恒为 0。）

---

## 3. 陈旧值 / 失效引用清单

### 3.1 🔴 幻影机「5060 / 26 SM」整篇残留 —— `design/KokonaPolaris_迁移手册.md`

**背景冲突**：`README.md:159` 与 `ONBOARDING.md:144` 都写着「库里曾记的『5060 / 26 SM』是**幻影机**……**已清除**」，
`tools/publish_check.py:34-35` 也把它列为守门对象。但**迁移手册这一整篇仍然以 5060 为主角**。

| 位置 | 现值 | 应改为 |
|---|---|---|
| `design/KokonaPolaris_迁移手册.md:1` | `# KokonaPolaris-S4 · 迁移手册（5050 → 5060）` | 实际是 5050(kokona) → **5070(viim)**；标题整体重写 |
| `…:11` | `5060 Laptop 与 5050 Laptop 同架构…只有 SM 数 20→26 算力 +30%` | **无实测依据**。实测机 = viim / RTX 5070 / **36 SM**（`MEMORY_ops.md:13`、`MEMORY_archive.md:11`） |
| `…:20` | `design/（md + html，主文档 v1.14 + 补充 01–11）` | 主文档 **v1.15**（正文已有 v1.16 标注）+ 补充 **01–12**；且「md + html」不全（补充10/12/迁移手册 无 html） |
| `…:21` | `kp/（17 模块）` | `kp/` 实测 **42** 个 `.py`（见 §3.3） |
| `…:23` | `data/characters/kokona/（角色卡素材 3.7MB + manifest）` | 现状见 `data/characters/kokona/{images,raw,manifest.csv}`，体积已变（2026-10-03 日志记录已加 `raw/`） |
| `…:70` | `3. **骨架自检 52 项**（纯 CPU，1 分钟内跑完）` | **70 项** |
| `…:95` | `## 五、GPU 差异：5060 vs 5050（结论：不用改配置）` | 整节作废（无 5060 实测） |
| `…:97` | 表头 `**5060 Laptop（新机）**` | 作废 |
| `…:101` | `\| SM 数 \| 20 \| **26** \| ⭐ **+30% 算力** |` | 作废；库内合法 SM 数只有 **20 / 36** |
| `…:102` | `\| TGP \| 45–100 W \| **45–115 W** \| ⭐ 上限 +15W |` | 作废（5060 档位无实测） |
| `…:104` | `\| CUDA Core \| 2560 \| **3328** \| ⭐ +30% |` | 作废；viim 实测 4608 core（`.workbuddy/memory/2026-10-03.md:1164`） |
| `…:111` | `新机 5060 的 TGP 上限也是 115W` | 作废；viim 持续负载下热限制到 ~90 W（`MEMORY_archive.md:38`） |
| `…:113-115` | `「实测 120.5 TFLOPS 已逼近 20 SM 理论峰值 114.6」…绑在 5050 上` | 口径本身正确（✅ 保留），但整段的迁移前提已作废 |
| `…:123` | `python -m kp.selftest # ② 自检 52 项（应全绿）` | **70 项** |
| `…:130` | `**日常写设计/跑自检不需要下 Sana** —— 自检 52 项全是纯 CPU。` | **70 项** |

> ⚠️ `design/KokonaPolaris_迁移手册.md` 是**唯一**还把 5060 当主角的文档；建议整篇重写或标注「⛔ 全文作废，换机意向从未实测」。
> `.workbuddy/memory/2026-10-03.md:714/749` 也留有 5060 记录，但那是**当日日志的历史流水**，同文件 `:1276` 已自我更正为「新机 = 5070 / 36 SM（不是 5050/5060）」—— **日志可保留，不算残留**。

### 3.2 🔴 参数量陈旧值（1.806B / 20.4% / 565.9M）

现状真值（依据 `ONBOARDING.md:138` + `kp/models/dit.py:16`）：
**KP-M = 1.875B（+25.0%）**、**KP-S = 594.8M ≈ 0.6B ✅**、已删 adaLN 段 7 `g_geo` 死参数 **−5.2%**。

| 位置 | 现值 | 应改为 |
|---|---|---|
| `design/KokonaPolaris_架构设计方案.md:10` | `KP-M 主干实测 1.806B 比标称 1.5B 大 20.4%` | `1.875B … 大 25.0%` |
| `design/KokonaPolaris_架构设计方案.md:10` | `（KP-S 565.9M ≈ 0.6B ✅）` | `（KP-S 594.8M ≈ 0.6B ✅）` |
| `design/KokonaPolaris_迁移手册.md:109` | `KP-M 实测 1.806B vs 标称 1.5B（+20.4%）` | `1.875B vs 1.5B（+25.0%）` |

> ✅ 已确认干净：`design/*.html` 全部 11 份**均无** `1.806B / 565.9M / 20.4%` 残留；`kp/models/dit.py:16` 已是 `17·1792²·32 ≈ 1.75B（+其余 ⇒ 实测 1.875B，vs 标称 1.5B +25.0%）`。

### 3.3 🟡 自检项数陈旧

| 位置 | 现值 | 应改为 |
|---|---|---|
| `design/KokonaPolaris_架构设计方案.md:8` | `自检 **57/57** 通过` | **70/70**（`kp/selftest.py` 实测 70 个 `check()` 调用） |
| `design/KokonaPolaris_迁移手册.md:70` | `骨架自检 52 项` | **70 项** |
| `design/KokonaPolaris_迁移手册.md:123` | `自检 52 项（应全绿）` | **70 项** |
| `design/KokonaPolaris_迁移手册.md:130` | `自检 52 项全是纯 CPU` | **70 项** |
| `design/KokonaPolaris_补充12_G2通道分离装置与判据.md:126` | `自检总数 **52 → 57**（原 52 项无回归）` | 改写为 `52 → 57 → 70`（否则读者会以为当前是 57） |

### 3.4 🟡 模块 / 文件计数陈旧

实测：`kp/` 下 **44** 个 `.py`，扣除本次并行在建的 2 个（`kp/typography/composite.py`、`kp/latent/real_separation.py`）
⇒ **稳定 42 个**（比已修的「41」多 1，多出来的是 2026-10-03 新增的 `kp/probe/attn.py`）。

| 位置 | 现值 | 应改为 |
|---|---|---|
| `README.md:126` | `kp/ 参考实现骨架（41 个 Python 文件，全部纯 CPU 可跑）` | **42**（或写成「42 个（不含在建的 composite.py / real_separation.py）」） |
| `ONBOARDING.md:33` | `**四件套**（≈8650 行 Python，41 个模块，纯 CPU 可跑）` | **42**；`≈8650 行` 亦建议按当前 `.py` 行数复核 |
| `design/KokonaPolaris_迁移手册.md:21` | `kp/（17 模块）` | **42** |

### 3.5 🟡 文档版本号漂移（md 与 html 孪生不同步）

| 位置 | 现值 | 应改为 |
|---|---|---|
| `design/KokonaPolaris_架构设计方案.html:123` | kicker `Architecture Design · v1.13` | md 头部是 **v1.15**，且正文 `:315` / `:1154` 已出现 **v1.16** 标注 ⇒ html 落后 ≥3 个版本 |
| `design/KokonaPolaris_迁移手册.md:20` | `主文档 v1.14 + 补充 01–11` | `主文档 v1.15（正文含 v1.16 标注）+ 补充 01–12` |
| `design/KokonaPolaris_补充11_…html:73` | `配套：主文档 v1.14` | 与 `补充11.md` 头部的配套声明对齐 |

> ⚠️ 同类问题历史上已发生过一次并被记入 `design/KokonaPolaris_补充09_…md:311-312`
> （「主文档 html 顶部 kicker 仍写 v1.9 → 已改为 v1.11」「补充 08 声明配套 v1.10 → 已改为 v1.11」），
> **说明这是复发型缺陷**，建议加进 `tools/publish_check.py` 的守门项。

### 3.6 🟡 失效的文件引用

| 位置 | 引用 | 实际 |
|---|---|---|
| `.workbuddy/memory/2026-10-03.md:1279` | `design/补充12_G2通道分离装置与判据.md` | 实际文件名是 `design/KokonaPolaris_补充12_G2通道分离装置与判据.md`（**少了 `KokonaPolaris_` 前缀**） |
| `design/KokonaPolaris_补充12_G2通道分离装置与判据.md:6` | `out/g2/g2_report.json` | 该文件**与 `out/g2/` 目录均不存在**（`out/` 下只有 `characters e4b_bf16 e5 e5b e6 kp tc_test typography_chain`）⇒ 属「尚未产出的目标路径」，建议加「（待跑 `tools/g2_channel_ablation.py` 后生成）」标注 |
| `design/KokonaPolaris_架构设计方案.md:8` | `D:\model\kp\` | ⛔ 与「⛔ 不写死绝对路径」铁律冲突（README.md:146、`tools/portable_paths.py`）；同页其他文档已改用相对写法 |

### 3.7 🟡 已完成但未勾掉的待办

| 位置 | 内容 | 现状 |
|---|---|---|
| `.workbuddy/memory/2026-10-03.md:1325-1327` | 「`MEMORY.md` 里还写着『5050/5060 三代全同』… ⇒ 待整理：把 MEMORY.md 的硬件节同步到 5070」 | **已做**：`MEMORY.md:64` 已是「kokona=5050/20SM ｜ viim=5070/36SM」两机口径 ⇒ 该待办应勾掉，否则下次有人照着重复劳动 |

### 3.8 ✅ 自检节号引用 —— 无漂移（逐条核对过）

`ONBOARDING.md:113-126` 共引用 14 个节号：`§2 §3 §4 §10 §11 §12 §13 §15 §16 §17 §18 §19 §20 §21`。
`kp/selftest.py` 实际 `section()` 调用共 21 个（`:83`–`:1172`），编号连续无缺口。
`design/KokonaPolaris_架构设计方案.md:316` 写「自检第 21 节（`kp/probe/attn.py`…）」—— 与 `:1172` 的 `section("21. G3 Sigmoid 注意力装置…")` 一致 ✅。
`§19 = G2 通道分离`（`:923`）、`§20 = G3.5 真探针`（`:1014`）、`§21 = G3 Sigmoid`（`:1172`）—— 与 ONBOARDING 的描述全部吻合 ✅。
**「文档说『自检 §19』但节号变了」这类漂移，本轮不存在。**

---

## 4. 死代码清单

**方法**：AST 解析全部 42 个 `kp/**/*.py`（排除 2 个并行产物）提取模块级公开符号（`def` / `class` / 赋值），
在 `kp/` + `tools/` + 全部 `.md`/`.html`/`.sh` 中按词边界统计引用；再按审计规则剔除
`__init__.py` 的 re-export 行、各模块 `__all__` 字面量行、定义行本身，最后人工复核每一条。

### 4.1 🔴 真死符号：全库零引用（6 个）

| # | 位置 | 符号 | 为什么判定是死的 | 影响 |
|---|---|---|---|---|
| 1 | `kp/paths.py:59` | `ensure(*p) -> Path` | 全库出现 **2 次**：定义行 + `kp/paths.py:25` 的 `__all__` 字面量。**0 次调用**。而 `kp/` 全库唯一的 `mkdir(` 就在它自己体内（`paths.py:62`） | 🔴 **自述规则与实现矛盾**：docstring 写「唯一允许创建目录的入口」，但所有代码都不走它。目录创建约定实际无守门人 |
| 2 | `kp/character/card.py:34` | `N_LAYERS = len(SEMANTIC_LAYERS)` | 出现 4 次：定义 + `card.py:116 __all__` + `kp/character/__init__.py:8` re-export + `:19 __all__`。**0 次读取** | 🟡 角色卡层数有「单一真源」外观，实际无人消费；改 `SEMANTIC_LAYERS` 时不会报错 |
| 3 | `kp/config.py:170` / `:180` | `class RuntimeCfg` / `RUNTIME = RuntimeCfg()` | `RuntimeCfg` 2 次（类定义 + 实例化）；`RUNTIME` 5 次全为 re-export/文档：`kp/__init__.py:24,28`、`ONBOARDING.md:66`、`.workbuddy/memory/2026-10-03.md:459`。**0 次读属性** | 🔴 **文档仍在推销它**：`ONBOARDING.md:66` 把 `RUNTIME` 列为「全部可调旋钮」之一 ⇒ 读者会以为有运行时旋钮可调，实际是空壳 |
| 4 | `kp/data/captions.py:190` | `is_tag_soup(text, threshold=None)` | 出现 2 次：定义 + `captions.py:407 __all__`。**0 次调用**，且**未**进 `kp/data/__init__.py`（该文件 re-export 的是 `tag_soup_score`，见 `:17,26`） | 🟡 薄包装（`:192` 转发给 `tag_soup_score`）。自检 §15（`kp/selftest.py:660,690-691`）只测 `tag_soup_score` ⇒ 阈值分支 `threshold` 参数**从未被任何测试覆盖** |
| 5 | `kp/probe/attn.py:305` | `known_answer_samples(...)` | 出现 2 次：定义 + `attn.py:66 __all__`。**0 次调用**，且**未**进 `kp/probe/__init__.py` | 🔴 **装置与自检两套「已知答案」并存、其一未接线**：该函数 docstring（`:308`）自称「照抄『装置层必须先过已知答案』这条规矩（G3.5 §16 的做法）」，但自检 §21 **另写了三个** known-answer 检查（`kp/selftest.py:1192` / `:1202` / `:1211`）而完全没用它 ⇒ 装置自带的这套样本是孤儿 |
| 6 | `kp/probe/real.py:728` | `DEFAULT_SHAPE = dict(dim=192, layers=6, heads=6)` | **全库仅 1 次命中 = 它自己**。不在任何 `__all__` | 🟡 与 `build_test_backbone` 的默认参数（`kp/probe/real.py:731`，同为 `dim=192, layers=6, heads=6`）**完全重复** ⇒ 改形状时改一处会漏另一处，是个静默陷阱 |

### 4.2 🟡 「仅 re-export、无调用点」—— 按规则不算死，但是未接线（4 个）

> 审计规则要求排除「被 `__init__.py` re-export 的」，故这 4 条**不算死代码**。
> 但它们和「已接线的同类」并排时暴露出**接线不对称**，单独列出。

| 位置 | 符号 | 现状 | 对照的「已接线」同类 |
|---|---|---|---|
| `kp/probe/attn.py:326` | `g3_report_text()` | 被 `kp/probe/__init__.py:33` re-export，但**没进 `kp.probe.__all__`（`:55-61`）**；全库 **0 次调用** | `axis_report_text` → `tools/axis_probe_demo.py:63` + `kp/selftest.py:774`；`real_axis_report_text` → `tools/axis_probe_real.py:80,106`。**G3 装置没有对应的 CLI 入口** ⇒ 拿到 `kp/probe/attn.py` 的人无法生成报告 |
| `kp/probe/real.py:339` | `gate_ranges(...)` | re-export（`kp/probe/__init__.py:46`）+ 在 `__all__`（`:59`）；**0 次调用** | 同上，`real.py` 的其他 `*_report_text` 都有 CLI |
| `kp/quant/nvfp4.py:117` | `precision_table(w)` | re-export（`kp/quant/__init__.py:11`）+ `__all__`（`:17`、`:130`）；**0 次调用** | `DESIGN_SPEC` / `OFFICIAL_DEFAULT_SPEC` 都被 `kp/probe/real.py:69,465,561` 实际消费 |
| `kp/train/qad.py:48` | `clear_quant(model)` | re-export（`kp/train/__init__.py:4`）+ `__all__`（`:28`）；**0 次调用** | 对称的 `set_quant` 有 3 处调用（`kp/train/qad.py:198`、`kp/selftest.py:474`、`kp/models/dit.py` 侧 `GatedLinear.set_quant`）⇒ 对外 API 成对但只有一半能用 |

### 4.3 🟡 `__all__` 漏项（1 组，接线完整性的隐性坑）

| 位置 | 问题 |
|---|---|
| `kp/probe/__init__.py:21-36` | 从 `.attn` re-export 了 **14** 个名字，但同文件 `__all__`（`:55-61`）**一个 attn 名字都没有** ⇒ `from kp.probe import *` 拿不到 G3 装置的任何东西 |
| `kp/probe/__init__.py:21-36` | 反向也漏：`kp/probe/attn.py:54-68` 的 `__all__` 里的 `activation_shape_stats` 与 `known_answer_samples` **未**在包级 re-export（对比 `contrast_report` / `contrast_vs_offset` / `qknorm_logit_bound` 都 re-export 了） |

> ⇒ 「模块 `__all__`」与「包 `__all__`」两份清单已经不同步，任何新增符号都可能只落一边。

### 4.4 🟡 未使用的 import（7 处）

| 位置 | 符号 | 依据 |
|---|---|---|
| `kp/selftest.py:404` | `import tempfile` | 全文件对 `tempfile` 的另外两次出现（`:49`、`:319`）**都是禁令注释**；本节临时产物实际落 `OUT/"_kp_adapter_selftest.pt"`(`:426`) 与 `OUT/"_kp_adapter_dit.pt"`(`:454`) ⇒ 这个 import 是上一版实现（用 `tempfile`）的残留 |
| `kp/latent/separation.py:47` | `from .hybrid import split_channels, join_channels` 的 `split_channels` | `split_channels` 全库仅此 1 处；同行的 `join_channels` 有用 |
| `kp/models/common.py:11` | `from typing import Tuple` | `Tuple` 在该文件仅此 1 处 |
| `kp/models/common.py:15` | `import torch.nn.functional as F` | 该文件无任何 `F.` 调用 |
| `kp/probe/real.py:70` | `from .axis import AxisProbe, AxisProbeReport, AxisResult` 的 `AxisResult` | `AxisResult` 全库仅此 1 处 |
| `tools/e5b_retest.py:99` | `import torch.autograd as _ta` | 别名 `_ta` 全库仅此 1 处；同文件 `:101` 直接用 `torch.autograd.…`，根本不需要别名 |
| `tools/e5_probe.py:54` | `import gc; gc.collect()` | ⛔ **不是问题**（同行使用），列在此仅为闭环，见 §6 |

---

## 5. 临时文件卫生违规清单

### ✅ 结论：**0 处违规**

| 检查项 | 结果 |
|---|---|
| `tempfile.gettempdir()` | **零调用**。全库 `tempfile` 只出现 6 处：3 处禁令注释（`kp/selftest.py:49`、`:319`；另 2 处在并行产物里、已排除）+ 1 处死 import（`kp/selftest.py:404`）+ 2 处在并行产物里、已排除 |
| `tempfile.NamedTemporaryFile` | 零命中 |
| `tempfile.mkdtemp` / `tempfile.TemporaryFile` | 零命中 |
| 写 `/tmp` / `"tmp"` / `TMPDIR` | 零命中（唯一近似命中是 `.workbuddy/memory/2026-10-03.md:989`「从 `/tmp` 里 source 也能正确解出路径」—— 那是**测 shell 的 cwd 行为**，不是写临时产物，不算违规） |

**自检的临时产物合规链**（可作为其余脚本的模板）：
- 落盘位置一律 `OUT / "…"`：`kp/selftest.py:426`（`_kp_adapter_selftest.pt`）、`:454`（`_kp_adapter_dit.pt`）
- 清理：`_rm()` 定义在 `kp/selftest.py:46-58`，两处都在 `finally` 里调用（`:431-432`、`:458-459`）
- 禁令写在最显眼处：`kp/selftest.py:49-52` 明确写了「本机实测 `gettempdir()` 返回**工作区根目录** ⇒ 临时产物一律落 `KP_OUT` 并清理」

**唯一需要清理的残留**：`kp/selftest.py:404` 的 `import tempfile` —— 不是违规（没调用），是**上一版实现的死 import**，建议一并删掉（见 §4.4）。

---

## 6. ⚠️ 误报排除说明（查过、判定没问题，别重复查）

### 6.1 「疑似死代码」实为已接线

| 符号 | 位置 | 为什么不是死的 |
|---|---|---|
| `CapabilityBus` | `kp/capability/bus.py:164` | 被 `kp/models/dit.py:47`（import）、`:354`（`self._bus = CapabilityBus(...)`）、`:361`（`-> "CapabilityBus"`）使用 |
| `freeze_backbone_` | `kp/models/dit.py:393` | 被同文件 `:355` 在 `__init__` 内调用 |
| `shape_of` | `kp/latent/hybrid.py:52` | 被同文件 `:89` 调用 |
| `CapabilityPack` | `kp/capability/bus.py:29` | 类型标注用：`bus.py:118,210,224` |
| `Responder` | `kp/probe/axis.py:51` / `kp/probe/synthetic.py:23` | 类型别名，两文件互用（`axis.py:222`、`synthetic.py:42`） |
| `KMEANS_K` / `CANVAS` / `BODY_RATIO` / `MAX_FIT_PIXELS` | `kp/character/pipeline.py:35/33/34/36` | 作为默认参数值被 `:198` / `:160` / `:210-211` 使用 |
| `kmeans` / `band_share` / `stage_normalize` / `stage_pack` / `stage_report` / `run_batch` | `kp/character/pipeline.py:69/87/159/373/426/500` | `run_batch` 链式调用前四者（`:508,519,521`），且 `run_batch` 是 `argparse` CLI 入口（`:564`）⇒ 属「被 CLI 调用」，按规则排除 |
| `axis_report_text` / `real_axis_report_text` | `kp/probe/axis.py:189` / `kp/probe/real.py:644` | 分别被 `tools/axis_probe_demo.py:63` + `kp/selftest.py:774`、`tools/axis_probe_real.py:80,106` 调用 |
| `set_quant` | `kp/train/qad.py:38` | `kp/train/qad.py:198` + `kp/selftest.py:474` |
| `LINEAR` / `SIGMOID` / `SOFTMAX` | `kp/models/dit.py:54/55/56` | `kp/probe/attn.py:52`、`kp/selftest.py:1174`、`dit.py:68-69,89,91,98` |
| `DEFAULT_BLOCK` / `FP4_MAX` / `DESIGN_SPEC` / `OFFICIAL_DEFAULT_SPEC` | `kp/quant/nvfp4.py:28/29/109/111` | `nvfp4.py:47-48,62,83` + `kp/probe/real.py:465,561` |
| `NO_LINE_START` / `NO_LINE_END` | `kp/typography/layout.py:20/22` | `layout.py:50,56` + `kp/selftest.py:613,631` |
| `FULLWIDTH_PUNCT` | `kp/data/captions.py:38` | `captions.py:145` |
| `EPS_SCALE` / `DIT_S` / `DIT_M` / `CANVAS` 等 | 各处 | 均有实际读取点，见上 |
| `LatentCfg` / `QuantCfg` / `CapabilityCfg` / `AxisCfg` | `kp/config.py:14/71/90/144` | 模块级即实例化（`LATENT = LatentCfg()` 等）⇒ 「只有定义和实例化两处命中」是**正常**的，不是死代码 |
| `tag_soup_score` | `kp/data/captions.py:157` | `captions.py:192,218` + `kp/data/__init__.py:17,26` + `kp/selftest.py:660,690,691` |
| `import gc`（`tools/e5_probe.py:54`） | — | 同行 `gc.collect()`，AST 扫描器把「定义行」排除后误判 |
| `import torch.autograd as _ta`（`tools/e5b_retest.py:99`） | — | ⚠️ **这条是真发现**（别名 `_ta` 确实没用），已列入 §4.4；但若只看 `torch.autograd` 会误以为在用 |

> ⚠️ `tools/typography/kp_engine.py:119,121,123` 里的 `NO_LINE_START` / `NO_LINE_END` / `FULLWIDTH_PUNCT`
> 是该文件**自己定义的同名副本**，**不是**从 `kp/` import 的 ⇒ 统计符号引用时会产生假连接，别当重复定义 bug。

### 6.2 「疑似陈旧数字」实为**正确的历史记录**（不要改）

| 位置 | 内容 | 为什么不改 |
|---|---|---|
| `kp/models/__init__.py:12-13` | `2026-10-03 由 18·d² 改为 17·d² —— 删掉 adaLN 里全代码库无人读取的段 7「g_geo」死参数` | 这是**变更说明**（记录「曾是 18·d²」），不是当前值。改它反而丢失审计线索 |
| `kp/models/dit.py:28` | `⭐ 为什么是 8 段不是 9 段（2026-10-03 瘦身）` | 同上，标题里的「9 段」是历史指称 |
| `design/KokonaPolaris_架构设计方案.md:57` | `自检 52 → 57 项` | 历史变更记录，保留有价值 |
| `design/KokonaPolaris_补充09_…md:316` | 「补充 01–06 的『配套主文档 vX』停在旧版 —— **非错误**，那是撰写时的版本，属历史事实；保留（不追改，避免失真）」 | 文档里**已经明文裁定**过这条政策。⇒ 建议把同一政策套用到 §3.5 的 html/md 版本漂移上，而不是无脑追改 |
| `design/KokonaPolaris_补充07_环境基线报告.md` 全部 SM 数 20 的内容 | `SM 数 20` / `20 SM` | 5050（kokona）的**实测**值，有效。⛔ 不要因为「换机到 5070」就改这里——记录本身没错，只需在读的时候带机器名 |
| `.workbuddy/memory/2026-10-03.md:714,749` 等 5060 记录 | 日志流水 | 同日 `:1276` 已自我更正；日志是时间线快照，不应回改 |

### 6.3 「疑似漂移」实为无问题

| 检查项 | 结论 |
|---|---|
| 自检节号 §2–§21（ONBOARDING.md:113-126） | ✅ 与 `kp/selftest.py` 的 21 个 `section()` **逐条吻合**，无漂移 |
| `52/52`、`63/63`、`18 模块`、`626.7M`、`31.8%`、`9 段`、`18·d²` | ✅ 全库零残留（唯一出现是 §6.2 里两处**正确的变更说明**） |
| 三份记忆库的机器信息 | ✅ `MEMORY.md` / `MEMORY_ops.md` / `MEMORY_archive.md` 只有 20 SM（kokona/5050）与 36 SM（viim/5070），**无 26 SM** |
| README/ONBOARDING/publish_check 里的「5060 已清除」表述 | ✅ **表述本身正确**（问题在于 `design/KokonaPolaris_迁移手册.md` 根本没清） |
| `design/*.html` 是否残留 1.806B / 565.9M / 20.4% | ✅ 11 份 html 全部干净（只有 `.md` 有） |
| README / ONBOARDING 引用的工具名 | ✅ `tools/fetch_sana.py` `gpu_probe.py` `onboard.py` `portable_paths.py` `publish_check.py` **全部存在** |
| 文档引用的裸文件名（`MEMORY.md` / `audit_captions.py` / `validate_manifest.py` / `separation.py` / `dit.py` …） | ✅ 全部是**正文行文里的简称**，同句或同节均给出完整路径（如 `补充11.md:10` 同时写了 `tools/data_spec/audit_captions.py`）⇒ 非失效引用 |
| `design/KokonaPolaris_补充04_…md:319` 引 `comfy_extras/nodes_model_merging_model_specific.py` | ✅ **外部项目**（ComfyUI）的文件，不在本仓库 ⇒ 非失效引用 |
| 自检项数 70 | ✅ 已核：`kp/selftest.py` 恰有 70 个模块级 `check(` 调用，与 README/ONBOARDING/`tools/onboard.py:75,108` 一致 |
| `kp/character/pipeline.py` 的入口自举 | ✅ 它是**包内模块**，按 `python -m kp.character.pipeline` 跑（cwd=仓库根即可），不属 `tools/` 直跑场景 ⇒ 不在 §2 清单内 |
| 临时文件铁律 | ✅ 全库零违规（详见 §5） |

### 6.4 需要人工拍板、本报告不擅自判定的两处

| 事项 | 现状 | 为什么不下结论 |
|---|---|---|
| `.workbuddy/memory/MEMORY.md:64` 写 `kokona = RTX 5050 Laptop(20 SM，**本会话所在机**，D:\model)`；而 `MEMORY_ops.md:13` 写「**实测机是 5070 / 36 SM**」 | 两个「当前机」声明并存，且当前工作目录确实是 `D:\model` | 「本会话所在机」是**会话相关**的相对表述，随人随机变化，本审计无法判定哪台是对的。**请机主确认 `D:\model` 这份工作副本究竟在哪台机上**，然后把 MEMORY.md:64 的「本会话所在机」改成绝对表述 |
| `.workbuddy/memory/2026-10-03.md:1327` 的待办「把 MEMORY.md 硬件节同步到 5070」 | MEMORY.md:64 已同步 | 按 §3.7 处理即可（勾掉待办），无歧义 |

---

## 7. 建议修复优先级

| 优先级 | 动作 | 影响面 |
|---|---|---|
| **P0** | 给 §2 的 **19 个**无自举工具各加 1 行 `sys.path.insert` | 一行一个，`ModuleNotFoundError` 立即消失 |
| **P0** | 修 §2.2 的 2 个「自举太晚」（把 `sys.path.insert` 提到 `import kp` 之前） | 2 个文件 |
| **P0** | `design/KokonaPolaris_迁移手册.md` 整篇标注作废/重写（§3.1，15 处） | 与 README:159 / ONBOARDING:144 的「已清除」声明**必须自洽** |
| **P1** | 刷新参数量 3 处（§3.2）+ 自检项数 5 处（§3.3）+ 模块计数 3 处（§3.4） | 纯文案 |
| **P1** | 删死符号：`kp/paths.py:59 ensure`、`kp/character/card.py:34 N_LAYERS`、`kp/probe/real.py:728 DEFAULT_SHAPE`、`kp/data/captions.py:190 is_tag_soup`、`kp/config.py:170/180 RuntimeCfg` | ⚠️ `RUNTIME` 删前先改 `ONBOARDING.md:66`（文档在推销它） |
| **P2** | 决定 `kp/probe/attn.py:305 known_answer_samples` 的去留：接进自检 §21，或删 | 「装置自带已知答案」的设计意图目前落空 |
| **P2** | 给 G3 装置补 CLI 入口（对齐 `axis_report_text` / `real_axis_report_text` 的 `tools/*_probe*.py` 模式） | §4.2 |
| **P2** | 同步 `kp/probe/__init__.py` 的 `__all__`（§4.3）+ 清理 7 处未用 import（§4.4） | 预防复发 |
| **P3** | 修 `.workbuddy/memory/2026-10-03.md:1279` 的坏文件名（§3.6）+ 勾掉 `:1327` 待办（§3.7） | 记忆库卫生 |
| **P3** | 把「md / html 孪生版本号必须同步」加进 `tools/publish_check.py` 守门项 | §3.5 是复发型缺陷（补充09:311-312 记过前科） |

---

*（本报告由只读审计生成，未对仓库任何既有文件做创建、修改、删除或 git 操作。）*

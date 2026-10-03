# 🔄 移交文档 · KokonaPolaris-S4（KP / 心夏北极星）

> **给下一个智能体 / 新的 IDE。** 生成时间：2026-10-03 19:4x · 对应 commit `e46efa2` · 自检 **73/73**
> 前置阅读：`ONBOARDING.md`（项目全貌）→ 本文（**怎么干活**）→ `.workbuddy/memory/MEMORY.md`（决策与硬约束）
>
> ⚠️ **本文档只讲「怎么高效干活」，不复述项目是什么。** 项目是什么看 `ONBOARDING.md`。

---

## 0. 三十秒版

| 问 | 答 |
|---|---|
| 现在到哪了？ | 设计层收敛 · **自检 73/73 全绿** · 12 条架构不变量**审计 0 偏离** · 两个 commit 已推云端 |
| 工作区干净吗？ | ✅ 干净，= `cloud/main` = `e46efa2` |
| 有什么半成品？ | ⚠️ **有 2 处**（见 §4），接手第一件事是收尾它们 |
| 能停吗？ | **不能。** 每轮结束时必须**已经在推进**，而不是把「等回答」当默认动作 |

---

## 1. 🚀 开工序列（照做，5 分钟内进入正题）

```bash
cd D:\model
$env:PYTHONIOENCODING="utf-8"      # ⚠️ 中文控制台 GBK 陷阱，见 §5
.venv\Scripts\python.exe tools/onboard.py                    # ① 环境体检（路径/依赖/缺口）
.venv\Scripts\python.exe -m kp.selftest                       # ② 必须 73/73，这是「验收单」
.venv\Scripts\python.exe tools/portable_paths.py --verify    # ③ 必须恒为 0
git status -sb && git log --oneline -3                       # ④ 确认起点
```

**基线数字（用来判断你有没有搞坏东西）**：

| 项 | 值 |
|---|---|
| 自检 | **73/73**（22 节：§21 G3 装置 · §22 排版闭环） |
| KP-S / KP-M 参数量 | **594.8M / 1.875B**（`python -m kp.arch_report`） |
| 路径硬编码 | **0** |
| 仓库体积 | 工作树 5.8 MB + .git 5.0 MB |

⚠️ **改了主干/VAE/量化后必须核对参数量数字不变** —— 这是「行为等价」最便宜的证据。

---

## 2. ⭐ 并行作业纪律（这一节是本文档的重点）

### 2.1 什么时候可以并行
本项目**只有一张 8GB GPU**，所以判据是**资源类型**：

| 任务类型 | 可否并行 | 说明 |
|---|---|---|
| 纯 CPU（审计/溯源/文档/纯函数实现） | ✅ 随便派 | 本项目绝大多数任务是这类 |
| 读设计稿做调研 | ✅ 随便派 | 但见 §2.3 的教训 |
| GPU 出图 / 训练 / 压测 | ⛔ **必须串行** | 派新的 GPU 任务前先查孤儿进程（§5.3） |

### 2.2 派 agent 的**文件边界**是硬要求
> 多个 agent **共享同一个工作树**（没有各自的 checkout）⇒ **两个 agent 改同一个文件 = 互相踩**。

派活时必须写死：
```
✅ 可改：<明确列出的文件>
⛔ 绝对不要改：<其余全部>（别人正在并行改）
⛔ 不要 git commit / add / checkout   ← agent 改了 git 索引，主线程会误提交别人的半成品
```

**分工原则：按「文件」切，不按「功能」切。** 功能相关但文件不同的可以并行；功能无关但文件相同的必须串行。

### 2.3 ⚠️ 三个实测踩过的坑

| 坑 | 代价 | 正确做法 |
|---|---|---|
| **调研任务给太宽** | 派去读 14 份设计稿的 agent **跑了 20+ 分钟没出稿、最后在写报告前挂掉，零产出** | **先窄后宽**：先派一个只挖「官方结论是什么」的窄任务，几分钟出结果；宽任务自己来 |
| **要求「先收集完再写」** | 上面那个 agent 死前说「证据已收齐但还没落盘」⇒ 全部白干 | 要求**边查边落盘到 `out/`**，别让它攒到最后 |
| **agent 改 git 索引** | 会把别人的半成品一起提交 | 每个 agent 都要写「⛔ 不要 git commit/add/checkout」 |

### 2.4 效率上限在哪（我的实测结论）
这一轮 **12 个 agent** 跑下来，真实瓶颈**不是并行度，是任务拆分质量**：
- ✅ 有效的：窄任务（15 分钟内出结论）、机械批量修复（21 个文件一次性修好）
- ❌ 无效的：宽调研任务、「全量对拍」类任务（挂了 2 次）
> ⇒ **压榨效率的正确姿势：多派窄任务，主线程留最关键的活，且每派一个都写死文件边界。**

---

## 3. 每轮收尾流程（照做就不会出事）

```bash
.venv\Scripts\python.exe -m kp.selftest                     # ① 必须全绿
.venv\Scripts\python.exe tools/portable_paths.py --verify   # ② 必须 0
git add -A && git commit -F <msgfile>                       # ③ ⚠️ 用文件传 message，反引号会被 bash 吃掉
git push origin main && git push cloud main                 # ④ 双推
# ⑤ 在 .workbuddy/memory/当天日志末尾追加「做了什么 + 关键数字 + **下一步**」
```

> ⑤里「**下一步**」那行最重要 —— 它是下一轮判断「是否中断未完成」的依据。
> ⚠️ `git push` 到 `cloud` 会用凭证，**如果报 403 是账号问题不是网络问题**（记忆库有完整记录）。

---

## 4. ⚠️ 移交时的半成品（接手第一件事）

### 4.1 🔴 半成品 A：`kp/latent/real_separation.py` + `tools/g2_real_vae.py`（G2 接真实 VAE）
- **状态**：文件已生成（已入库），但 agent **被中断，没交付报告、没验证过**
- ⚠️ **不要直接信任它**。接手后第一件事：`python tools/g2_real_vae.py` 看它跑不跑通，
  然后**人工核对它的结论**（尤其是「真图能不能分离」这个结论 —— G2 是**结构性门**，
  数字不好看也要如实报，⛔ 不许为了好看调门线）
- 价值：把 G2 从「合成数据」推进到「真实图片 + 项目自己的 HybridVAE」

### 4.2 🔴 半成品 B：7 个「死的旋钮」（接线死旋钮任务被中断）
审计发现 `kp/config.py` 里有 **7 个旋钮改了不生效**（只有 `arch_report` 的 print 在读）：

| 旋钮 | 位置 | 状态 |
|---|---|---|
| `quantize_injection_point` | config.py:49 | 待标注（行为由 `bus.py` 结构保证） |
| `gate_dtype` | config.py:92 | 待接线（**安全相关**，应能生效） |
| `matryoshka_tokens` | config.py:47 | 待接线（`sample.py` 两处硬编码） |
| `param_count()` | config.py:55 | 待修（公式 `9d²` **与 17d² 口径矛盾**，会误导人） |
| `identity_via_cross_attention` | config.py:95 | 待标注 |
| `identity_token_dim` | config.py:94 | 待接线（`fitter.py:52` 硬编码 1024） |
| `QuantCfg` 全 8 字段 | config.py:71-80 | 待接线（`nvfp4.py` **从不 import config**） |

> ⚠️ **这类东西比 bug 危险**：文档说「全部旋钮都在 config.py」，于是改 config 却不生效、
> **零报错** —— 和项目记忆里「注入点被 NVFP4 量化器吃掉 = 静默失效」是同一类病。
> 完整清单见 `out/audit_invariants.md` §4。

### 4.3 ✅ 已完成但**文档还没更新**的
- `design/KokonaPolaris_迁移手册.md` **整篇 15 处**仍以幻影机 5060/26SM 为主角（README/ONBOARDING 已清除，这篇漏了）
- `kp/config.py` 的 `RUNTIME` 已在 ONBOARDING 标注为「预留空壳」

---

## 5. ⚠️ 本机陷阱（Windows 中文控制台，全部实测踩过）

### 5.1 GBK 编码 —— 一个 bug 类，打了 4 个
| 症状 | 根因 |
|---|---|
| `UnicodeEncodeError: 'gbk' codec can't encode '\xb2'` | 打印 `1024²` 撞 GBK |
| `AttributeError: 'NoneType' object has no attribute 'split'` | `subprocess(text=True)` 用 GBK 解码 git 的 **UTF-8** 输出 ⇒ `r.stdout=None` |

**修法**：模块入口 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；
subprocess 一律显式 `encoding="utf-8"`。⚠️ **报错文本完全指不到根因**（教训 #2）。

### 5.2 `python tools/xxx.py` 的 `sys.path` 陷阱
`sys.path[0]` 是 **`tools/`** 不是仓库根 ⇒ `import kp.paths` 必崩。
**本轮修了 21 个工具**，现在统一写法：
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 子目录用 parents[2]
from kp.paths import ...  # noqa: E402
```
⛔ 永远用 `Path(__file__).resolve().parents[N]`，**不写死绝对路径**。

### 5.3 孤儿进程（**派 GPU 任务前必查**）
agent 被中断后子进程**会继续存活**并占着 GPU：
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select ProcessId,CommandLine
# 确认是孤儿后再 kill，别误杀正在跑的
```
本轮清理时发现 `.venv` 和系统 Python 是**同一份**，每个逻辑进程显示两条。

### 5.4 机器身份
- **本机 = kokona = RTX 5050 Laptop / 20 SM / `D:\model`**（实测确认）
- 档案里的基线机 = **viim = RTX 5070 / 36 SM**（在 `D:\kokonapolaris-s4`）
- ⛔ **本机测的 FLOPS/功耗不可回写 viim 基线**（纯 CPU 结论不受限）
- ⚠️ `tempfile.gettempdir()` 在**受沙箱限制的机器**上会退化成 cwd ⇒ 临时产物一律落 `KP_OUT` 并清理

---

## 6. ⏳ 等用户拍板的事（⛔ 别擅自决定）

| # | 事项 | 现状与选项 |
|---|---|---|
| 1 | **KP-M 尺寸**：实测 1.875B vs 标称 1.5B（+25%） | 选**改结构**（`dim≈1664/L32`，≈1.52B）还是**改标称**（改成 2.0B）？<br>倾向改结构（KP-L 3B 仅 teacher 不发布 ⇒ KP-M 是发布最高档） |
| 2 | **文本塔尺寸**：220M 蒸馏 vs 蒸馏到 0.6B | ⭐ 建议**蒸馏到 0.6B**：同参数量下多一份「为图像条件化任务定制」的收敛。<br>⚠️ 踩死线：中文配比在预训练前定稿后**不可逆** |
| 3 | **教师模型**：建议用 **`Qwen3-4B-Base`** 而非 Instruct | Base 无 chat 对齐层 ⇒ **根本不需要擦除**，且 Apache 2.0、sha 可钉死、任何人能复现 |
| 4 | **G3 判据缺口**：「量化友好性」那半边**无通过判据** | 要不要补阈值？（该半边实测**成立**，且是 NVFP4 硬约束下更值钱的那半边） |
| 5 | `补充11.md:56` 的污染文本如何订正 | 证据表明应为「220M 文本塔」（其 HTML 孪生版本就是这样） |

---

## 7. 🔥 下一步优先级（建议顺序）

1. **收尾 §4.1 的 G2 半成品**（先验证再采信）—— G2 是**结构性门**，最高价值
2. **收尾 §4.2 的 7 个死旋钮** —— 静默失效类，最危险
3. **G5 语义层**：把角色卡管线里的启发式占位换成 **See-through 自举**
   （用户明确说过「**先把角色卡线做完**，做完立刻回主线」）
4. **G1 正式 FID**：⚠️ 本轮实测结论是 **N=24 测不出东西**（噪声地板 189.7 已淹没两臂的 107/157）
   ⇒ 要么堆到大批量样本，要么换个更适合小样本的分布级指标
5. **排版链路补 T0 像素域合成**（latent 空间拼回是有损降采样，保位置不保字形高频）
6. **G6 租云** Micro-budget 预训练 —— 它同时解锁 **G3 官方判据**（benchmark 级，当前结构性阻塞）

---

## 8. ⛔ 三条铁律（别重犯）

1. **不推半成品** —— 没跑通 `kp.selftest` 的代码**不要提交**
2. **不猜** —— 没声明就说没有并报缺口。设计稿没写的阈值**不许自己编**
3. **别把机器实测当永久事实** —— 跨机结论一律作废

---

## 9. 📁 本轮新增的审计报告（`out/`，gitignored）

| 文件 | 内容 |
|---|---|
| `out/audit_invariants.md` | **12 条架构不变量逐条对拍：0 偏离**；附 7 个死旋钮 + 5 处写死魔数 |
| `out/audit_stale_and_dead.md` | 陈旧值 26 处 / 死代码 6 个 / 入口自举缺失 21 个（**已修**） |
| `out/texttower_trace.md` | 文本塔溯源：**KP 从头到尾是 Qwen3-4B，Gemma2-2B 是 Sana 靶子的** |
| `out/g3_official_criteria.md` | G3 官方判据溯源 + **判据缺口清单** |

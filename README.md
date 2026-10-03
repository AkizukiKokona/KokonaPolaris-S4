# KokonaPolaris-S4 · 心夏北极星（KP）

> **自研文生图架构**（非 SDXL / Flux 修补）。主打二次元 / 日系，写实是**主干内生的 domain 旋钮**。
> **无底模，一个权重都不继承。**

⚠️ **本仓库是「设计 + 参考实现骨架」，不是成品模型。** 骨架自检 **52/52** 通过（纯 CPU）。

---

## 三个必须先澄清的误解

| 误解 | 事实 |
|---|---|
| 「教师」=「底模」 | ❌ KP **无底模**。教师/工具（Sana、Z-Image）只是**试件**，用来验证设计，不被继承权重 |
| 「我不生成图像内容」⇒「这不是视觉模型」 | ❌ KP **是**文生图视觉生成模型；「不生成」指的是我（AI）不亲自画图，我只是造机器 |
| 「接了两个生图模型」 | ❌ 只是**工具链**。产品 = ① 文本塔 + ② 自研 DiT 主干 + ③ VAE + ④ 可选能力挂件 |

---

## 产品 = 四件套

| 件 | 规模 | 作用 |
|---|---|---|
| **文本塔** | ~220M LLM（自 Qwen3-4B 蒸馏） | ⭐ **天生支持中文**（不是 CLIP） |
| **自研 DiT 主干** | KP-S 0.6B / KP-M 1.5B | 单流 + 混合注意力 3:1 + QK-Norm + RF |
| **VAE** | 32× 压缩 | 语义 8ch（DINOv3）+ 细节 32ch（DC-AE 式）= **40ch 混合 latent** |
| **能力挂件** | 100M–1B | ∥-Pack / Δ-Pack / SVDPack / 擦除算子 `E` |

**1024² 仅 1024 个图像 token**（显存第一因是 token 数，不是参数量）。

---

## 四条硬约束（不可动摇）

1. **NVFP4 原生**，兼容 Blackwell **SM120**（RTX 50 系，最低 8GB）
2. **低显存优先** —— 显存效率是第一设计变量，不是「为 8GB 定制」
3. **弃用 UNet**，主干走 DiT
4. **LoRA 训练极简易** —— 角色正视图 + 背视图即可训好，不与底模打架

---

## 核心设计决策（勿重复推翻）

- **能力总线三不变量**：① 门控全 0 ⇒ 与裸模型 **bit-exact**（靠**整条旁路短路**，不是乘 0）；② 门控全程 fp32；③ **注入点必须绕过 NVFP4 量化器**（否则量化器会把增量吃掉 = 静默失效）。
- **⭐「禁止入侵维度」**：LoRA 会产生**正交于 `W₀` 奇异向量的高排名分量**（arXiv 2410.21228）⇒ Δ-Pack 建在 `W₀` 的 SVD 子空间内，训练后做**谱检查**（cos < 0.1 判不合格）。**能力越大越域外，越不能住 ΔW 里。**
- **能力包成本三档**：秒级·闭式（恢复被抑制能力）/ 小时级（长尾 Δ-Pack）/ 天级（跨域 ∥-Pack）。
- **可逆擦除账本**：合规抑制实现为显式低秩算子 `E = Σαᵢuᵢvᵢᵀ` ⇒ **`Pack_recover ≡ E⁻¹`**（加能力 = 减能力取反）。**因为我们自己过滤数据，所以我们知道擦了什么。**
- **身份载体是一等公民**：2.5D 分层（19 类语义层，1–5MB，**不依赖底模坐标系 → 跨版本通用**）⇒ 换角色 = Fitter 前向一次（**秒级**）。
- **三层控制栈**：L1 主干内生条件轴（~90% 调用量，零成本）/ L2 外置条件编码器（~9%，**身份必须外置**）/ L3 子空间约束 LoRA（~1%）。**让 LoRA 只做长尾。**
- ⚠️ **90/9/1 是调用量分布，不是能力分层** —— L3 承担 **100% 兜底责任**。故设 **G3.5 门**（Axis Probe 四测：单调/正交/可逆/**低比特行程**），不过则该轴降级 L3。
- **32× 压缩与文字互斥**（40px 汉字仅占 1.25 latent 格）⇒ **主干不动**，文字走独立低压缩 ROI 分支（TypographyPack）+ 确定性 Layout Planner（纯 JSON）。

---

## 快速开始

> ### 🤖 给接手这个仓库的 AI / 新同事
>
> **第一件事不是跑代码，是读记忆库** —— 里面记录了「已定决策、硬约束、以及踩过的坑」，
> 这些**不在代码注释里**，也不在 git 历史里。跳过它最容易重犯已经解决过的问题。
>
> ```bash
> ① .workbuddy/memory/MEMORY.md        # 决策 + 硬约束 + 协作约定（**先读这个**）
> ② .workbuddy/memory/MEMORY_ops.md    # 本机环境细则、数据交付规范
> ③ .workbuddy/memory/MEMORY_archive.md# 实测数字（跑分、量化、显存）
> ④ .workbuddy/memory/2026-10-*.md     # 逐日日志，末尾有「**下一步**」
> ```
>
> 读完直接看 `design/KokonaPolaris_架构设计方案.md`（项目的真相）。
> 代码是**骨架**，不跑训练；它承载的是「可验收的不变量」，自检 52 项就是验收单。

```bash
git clone https://github.com/AkizukiKokona/KokonaPolaris-S4.git
cd KokonaPolaris-S4

# 新环境第一条命令：路径体检 → 依赖检查 → 跑自检 → 缺口清单
python tools/onboard.py
```

**只需要 `torch` / `numpy` / `pillow` 就能跑全部 52 项自检**（纯 CPU，无需下载模型）。

| 常用命令 | 作用 |
|---|---|
| `python -m kp.selftest` | **52 项不变量自检**（纯 CPU，夜间安全） |
| `python -m kp.arch_report` | 架构速览 + 真实参数量核对 |
| `python -m kp.character.pipeline` | 角色卡数据管线 |
| `python -m kp.train.fitter` | Character Fitter 训练闭环 |
| `python -m kp.paths` | 路径解析体检 |
| `python tools/publish_check.py` | 发布前体检（体积/权重/密钥） |
| `python tools/portable_paths.py --verify` | 路径硬编码审计（应恒为 0） |

📖 **换机 / 新环境** → 看 [`design/KokonaPolaris_迁移手册.md`](design/KokonaPolaris_迁移手册.md)

---

## 可执行判据（本仓库的「不是纸面」部分）

| 判据 | 守住什么 |
|---|---|
| **谱检查**（`spectral_check`） | 禁止入侵维度 ⇒ LoRA 不与底模打架 |
| **`E⁻¹∘E` 可逆性**（KL ≈ 0） | 擦除可逆 ⇒ 加能力 = 减能力取反 |
| **门控全 0 ⇒ bit-exact** | 能力包全关时与裸模型逐位相同 |
| **Axis Probe 四测** | L1 条件轴真实性（G3.5），**低比特行程不可被前三测替代** |
| **caption 语料审计** | P1.8 中文自然语言支持（**配比进预训练后不可逆**） |
| **声明式多视角配对** | 角色卡 Fitter 的训练对，**不猜、不静默出错配** |
| **Fitter 闭式四判据** | 留出泛化 + 负对照，**无负样本时显式报缺口** |
| **通道分离监督（G2）** | 交叉/反向扰动 + MI 惩罚 ⇒ 画风解耦不失效 |

---

## 验证门

**G0 环境 🟢 → G1 ✅ 初步通过（W4A8）→ G2 通道分离 → G3 Sigmoid 注意力 → 🔴 G3.5 L1 条件轴真实性 → G4 Matryoshka → G5 角色卡 → G6 Micro-budget（需租云）→ G7 少步蒸馏**

当前：**设计层收敛（~90%）/ 假设层未就绪 / 实现层骨架已落地（52/52）**。
唯一遗留：**G1 正式 FID（≥50 张/臂，需白天解锁功耗）**。

---

## 仓库结构

```
design/     设计稿（主文档 v1.14 + 补充 01–11 + 迁移手册）★ 项目的真相
kp/         参考实现骨架（18 模块，全部纯 CPU 可跑）
  config.py     全部可调旋钮（QUANT / CAP / CAPTION / AXIS / LATENT / DIT）
  paths.py      路径唯一真源（三级探测，跨机可移植）
  latent/       40ch 混合 latent + 打包纯函数
  capability/   bus / delta_pack / svd_pack / parallel_pack / erase
  models/       dit / vae / text_tower / charabridge
  character/    card / fitter / pairing / dataset / pipeline
  quant/        NVFP4（与参考实现逐位对拍）
  train/        qad（adapter 式量化训练）/ fitter
  typography/   layout（避头尾/竖排）/ typography_pack（低压缩 ROI）
  probe/        axis（G3.5 四测）/ synthetic（对照样本）
  data/         captions（P1.8 语料审计）
tools/       探针与体检脚本（纯 CPU）
data/        小体积数据（角色卡素材 + manifest）
.workbuddy/  记忆库（决策 / 实测数字 / 本机环境 / 逐日日志）★
```

---

## 版本控制策略

**只入库** 设计稿 / 源码 / 配置 / 小体积数据 / 记忆库（合计 **5.5 MB**）。
**不入库** `models/`（4.6GB，`tools/fetch_sana.py` 重下）/ `out/`（163MB，结论已入库）/ `.venv` / `.cache`。

```bash
# 换机后重下 G1 靶子（约 4.6GB，可断点续传）
source env.sh && "$KP_PY" tools/fetch_sana.py
```

---

## 开发环境

RTX 5060 Laptop / 8GB / **sm_120** / 26 SM · torch 2.8.0+cu128 · Python 3.12
**环境入口** `source env.sh` → 用 `"$KP_PY"` 调解释器（缓存全落项目根，不写 C 盘）

⚠️ **夜间禁压测**：整机切安静模式时 GPU 被压到约 40W ⇒ 夜间只做 CPU / IO / 文档 / 代码。

---

## 许可与合规

- 代码与设计稿：本仓库自有。
- **G1 靶子 Sana 1.6B**（Apache 2.0）仅作**验证试件**，不随仓库分发，需自行从 HF 下载。
- 角色卡数据见 `data/characters/kokona/`（`source` 列标注来源），**不含解包商业游戏资产**。

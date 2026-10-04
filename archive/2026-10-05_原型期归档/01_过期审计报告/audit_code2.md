# KP 代码实现审查（code-audit）

审查对象：`D:/model/kp/`（44 个 .py / 11289 行）
方法：AST 静态扫描 + 逐文件精读 + 实跑验证（CPU，未跑 GPU，未用 `python -O` 跑业务）
只读审查：**未修改 `kp/` 下任何文件**，未 git 操作。

> ⚠️ **方法论披露（先自证）**：本次审查第一版的死旋钮扫描脚本有 bug——跳过目录集合里
> 误含 `"models"`/`"data"`，导致 `kp/models/`、`kp/data/` 整个被跳过（`dit.py` 未被扫），
> 一度误报 `ratio_gated_linear` 等为死旋钮。修正后重跑，并加了「关键文件必须在扫描清单内」
> 的自检（97 文件 / 9 个关键文件全部命中）。**下文所有结论基于修正后的结果。**
> 这条披露本身也是对「审计工具需要负对照」的应用。

---

## 一、旋钮接线状态表

`config.py` 共 6 个 frozen 架构 dataclass + 1 个 `RuntimeCfg`。
方法：对每个字段名全库 AST 扫 `Attribute` 读取，并区分**行为消费**与**仅报告消费**。

### 1.1 完全未接线（0 次读取，7 个）

| 字段 | 所属类 | 默认值 | 引用位置 | 优先级 |
|---|---|---|---|---|
| `DiTCfg.matryoshka_tokens` | DiTCfg | `(256,1024)` | **无** | 🔴 高 |
| `DiTCfg.quantize_injection_point` | DiTCfg | `False` | **无** | 🟡 中 |
| `CapabilityCfg.gate_dtype` | CapabilityCfg | `"float32"` | **无** | 🟡 中 |
| `CaptionCfg.require_sentence_punct` | CaptionCfg | `False` | **无** | 🔵 低 |
| `RuntimeCfg.vram_target_infer_gb` | RuntimeCfg | `2.5` | **无** | 🔵 低 |
| `RuntimeCfg.vram_target_lora_gb` | RuntimeCfg | `4.5` | **无** | 🔵 低 |
| `RuntimeCfg.tags` | RuntimeCfg | `{"stage":"skeleton"}` | **无** | 🔵 低 |

外加 **`RUNTIME` 单例本身 0 引用**（`RuntimeCfg` 5 字段全死，与 `config.py:174-186`
自检文档的自我说明一致，属**已知且已标注**的死代码，不算新发现）。

### 1.2 ⚠️ 只被 `arch_report.py` 读——「报告活、行为死」（12 个，交接文档未列）

这 12 个字段**确实被读取**，所以朴素 grep 会判它们"已接线"；但唯一的读者是
`kp/arch_report.py`（纯打印档案），**没有任何运行时行为消费它们**。改这些值 =
只改一行打印输出，模型行为零变化。这比 1.1 的"死旋钮"更隐蔽。

| 字段 | 所属类 | 唯一读者 |
|---|---|---|
| `QuantCfg.weight_bits` / `weight_fmt` / `act_fmt` / `act_bits` / `scale_fmt` / `block_size` / `quantize_proj_out` / `sm_arch` | QuantCfg（**全 8 字段**） | `arch_report.py:85-88` |
| `CapabilityCfg.identity_via_cross_attention` | CapabilityCfg | `arch_report.py:98` |
| `CapabilityCfg.identity_token_dim` | CapabilityCfg | `arch_report.py:97` |
| `CapabilityCfg.delta_pack_bytes` | CapabilityCfg | `arch_report.py:99` |
| `CapabilityCfg.erase_default_rank` | CapabilityCfg | `arch_report.py:102` |

**核实结论：交接文档 HANDOFF §4.2 的 7 个旋钮清单基本准确**（`quantize_injection_point`、
`gate_dtype`、`matryoshka_tokens` 三项逐条复核为真；`param_count()` 见 §四）。
**新增发现是上面这张 1.2 表**——文档说"QuantCfg 全 8 字段"是**报告层死**而非**读取层死**，
这个区别很重要：它意味着 `QUANT` 这个单例虽然有 8 次引用，但**它对量化行为零影响**。

### 1.3 已正确接线（抽查确认）

`LATENT.*`（86 引用）、`CAP.gate_init`（`bus.py:36` 真消费）、`CAP.spectral_*`
（`delta_pack.py:91,93`）、`AXIS` 四门线（`axis.py:436-439`）、
`DiTCfg.dim/heads/layers/mlp_ratio/qk_norm/double_stream_blocks/ratio_*`（`dit.py` 真消费）。

---

## 二、不变量断言状态表

| # | 不变量 | 断言位置 | 方向对不对 | 负对照 |
|---|---|---|---|---|
| 1 | 能力包全关 ⇒ bit-exact（返回 None 而非零向量） | `selftest.py:134,263,273,537,541,545` / `:556,581,588` | ✅ 对 | ✅ 有（`_gate_on` 断言开后**必须变** `:151,269`） |
| 2 | 禁止入侵维度（ΔW 建在 W₀ 的 SVD 子空间） | `selftest.py:164`（子空间初始化→合格）、`:179`（正交扰动→**不合格**） | ✅ 对 | ✅ 有（`_orthogonal_fail` 构造 Utail 正交扰动必须判不合格） |
| 3 | 可逆擦除 `E⁻¹∘E` ≈ 恒等 | `selftest.py:197-198`（`w_roundtrip_err<1e-5`、`kl<1e-6`） | ✅ 对 | ⚠️ 弱（见 🔴-4） |
| 4 | SVDPack 按构造无入侵维度 + 跨版本可迁移 | `selftest.py:508,512`（`min_cos>0.99`）、`:526`（换 W₀ 后 ΔW 不变） | ✅ 对 | ✅ 有（`:515` 与 Δ-Pack 用同一把尺子对比） |
| 5 | CharaBridge 可关断 | `selftest.py:556`（`cb(refs) is None`）、`:583,588,605,606` | ✅ 对 | ✅ 有（`:606` 开门后必须变） |
| 6 | G3.5 四测（单调/正交/可逆/低比特行程） | `probe/axis.py:436-439` + `selftest.py:1066-1076,1085` | ✅ 对 | ✅ 强（死轴/门关/共线注入/分辨力基线，4 类负对照） |

**结论：6 条核心不变量全部有真断言，方向全部正确，未发现"断言写反"。**
G3.5 一节的负对照密度尤其高（`_rp_nonvacuous` 钉死"死轴不许假通过 ①"，
并显式记录了 `argsort` 对常量返回原序导致 ρ=1.00 的真实踩坑史）——
这一节符合教训 #5 的要求。

---

## 三、逐条发现

### 🔴-1 `python -O` 下 73/73 检查全部变成空断言，**自检 100% 失效**
**位置**：`kp/selftest.py`（171 处裸 `assert`）+ `check()` 骨架 `:61-69`

实测：
```
.venv/Scripts/python.exe -O -m kp.selftest   →  结果：73/73 通过  全部通过 ✅
```
`check()` 靠捕获 `AssertionError` 判失败，而 `-O` 会把 `assert` 语句整体移除。
于是每个 check 的函数体变成"只算不判"，**返回空串 → 被记为 True**。
AST 逐个解析 73 个 check 的函数体（含一层嵌套 def 解析）：

| 类别 | 数量 |
|---|---|
| `-O` 下**完全**变成 no-op（无任何存活的 raise/守卫） | **71 / 73** |
| 仍有显式 `raise` 存活 | 2（`selftest.py:984`、`:1318`） |

**这不是"可能的风险"，是已复现的事实**：73/73 这个数字在 `-O` 下仍然成立，
但它不再证明任何东西。任何 CI/部署若带 `-O`（`PYTHONOPTIMIZE` 环境变量、
`python -O -m`、某些打包器），验收基线就静默失效。

**建议**：把 `check()` 改成不依赖 `assert` 的门函数，例如
```python
def check(name, fn):
    try:
        detail = fn() or ""
    except Exception as e: ...
```
并要求每个 check 内部用 `if not cond: raise CheckFailed(...)`，
或至少在 `kp/__init__.py` / `paths.py` 入口处显式检测
`not __debug__` 并直接拒绝运行自检（`__debug__` 在 `-O` 下为 `False`，是零成本的可靠探针）。

---

### 🔴-2 `DiT` 的 3:1 混合注意力实际是 **1:3.43（32 层）/ 1:3.60（24 层）**，且断言被放宽到刚好盖住
**位置**：`kp/models/dit.py:59-74`（`build_attn_plan`）、`kp/config.py:42-44`、`selftest.py:1258-1259`

实算：
```
DIT_S (L=24): {'softmax':1,'sigmoid':5,'linear':18}  → 1:3.60
DIT_M (L=32): {'softmax':1,'sigmoid':7,'linear':24}  → 1:3.43
设计目标 ratio_sigmoid/ratio_gated_linear = 1/3 = 0.3333
```
**根因**（不是随手写错，是结构性冲突）：第 0 层被 softmax 锚点占用后，
剩余槽位为 `L-1`，而 `23 mod 4 = 3`、`31 mod 4 = 3` ⇒ **3:1 循环在两个真实配置上都无法整除**，
尾部被 `plan[:layers]` 截断，sigmoid 层数被迫少于理想值。

**断言方向问题**：`selftest.py:1259` 写 `assert 0.2 <= ratio <= 0.35`，
实测 `ratio = 0.2917` 落在**贴近上沿**的位置。也就是说
**断言的容差带 (0.20~0.35) 是围绕"实际发生的 1:3.43"设计的，而不是围绕设计目标 1:3(=0.3333)**。
按记忆教训 #5 的判据（"断言要检验设计想表达的不变量，不是代码当前的行为"），
**这一条正是典型的"测当前行为"**：它把偏离固化成了合格。

**建议**：二选一，并让断言表达设计意图——
- 若"每 4 层 3 linear + 1 sigmoid"是硬约束：`layers` 必须取 `4k+1`（如 KP-S=25、KP-M=33），
  断言改为 `assert ratio == 1/3` 附近的极窄带（如 `abs(ratio-1/3) < 0.02`）。
- 若 softmax 锚点优先：把 `DiTCfg` 显式加一个 `n_sigmoid_override` 字段算准数量，
  断言改为「sigmoid 层数 == 期望整数」而非比例区间。
- 无论哪种，**当前这条比例区间断言应标为"测行为"并降级为报数**，
  因为它无法区分"符合设计"和"偏离设计但落在宽容带内"。

---

### 🔴-3 latent 40ch 在主干里**只是通道拼接**，没有真正分开处理
**位置**：`kp/latent/hybrid.py:141-145`、`kp/models/dit.py:331`

```python
def split_channels(x):        # hybrid.py:141
    sc = LATENT.semantic_ch
    return x[:, :sc], x[:, sc:]        # ← 纯切片
```
主干侧：
```python
self.patch_embed = _gl(latent_ch, d)  # dit.py:331，latent_ch=40
```
`patch_embed` 是**一个 40→1152 的稠密矩阵**，40 个通道在同一组权重下线性混合。
即：**语义 8ch 与细节 32ch 在 patch embedding 处就被同一套权重混合**，
后续 24/32 层 block 也完全不看通道身份。

代码里唯一让两块"分开"的机制是：
- `channel_mi_penalty`（`hybrid.py:174`）—— 只在 `separation.py` 的**训练/验收装置**里用；
- `swap_channels`（`hybrid.py:159`）—— 交叉扰动探针。

这两者都在 `kp/latent/` 与 `probe/` 域内，**DiT 主干前向路径完全不感知通道划分**。

**与设计稿的偏差**：设计稿要求"句法/细节通道分离"（`hybrid.py:1-13` 自述
"通道分离**必须显式监督**，否则主干会把两个通道都塞满信息"）。
代码把"显式监督"实现为**一个可微惩罚项**，但**没有任何机制阻止 `patch_embed`
在推理时把两块混在一起**——惩罚只在训练期生效，且不约束 `patch_embed` 本身。
换句话说：**"分离"在代码里是训练目标，不是结构保证。**

**建议**：若设计意图是结构保证（推理期也分开），`patch_embed` 应拆成
`_gl(8,d)` + `_gl(32,d)` 后相加，或对语义块单独加一路投影；
若设计意图就是"训练期软约束"，**应在 `hybrid.py` 的 docstring 里把话说死**
（现在 §设计要点 与 §"必须显式监督"读起来像结构保证，实际是软约束），
并补一条断言：训练后 `patch_embed` 权重在两块上的 Frobenius 能量比有下界。

---

### 🟡-4 可逆擦除的断言方向对，但**测不出设计想表达的东西**
**位置**：`kp/selftest.py:188-203`、`kp/capability/erase.py:190-207`

`erasure_roundtrip` 测的是 `recover(apply(W)) == W`。但 `apply` 是 `W - E`、
`recover` 是 `W + E`，**这是浮点加减法，必然回到原点**——它验证的是
"减法再加法可逆"，而非"擦除算子设计正确"。

`E⁻¹∘E ≈ 恒等` 这个设计不变量的**真正风险**不在往返误差，而在：
`U`/`V` 是否正交归一、`alpha` 是否被正确记账、`retained` 保留集是否真的没被擦到。
当前断言一条都没覆盖。

**建议**：补三条真不变量——
1. `delta_w()` 落在 `span(U)⊗span(V)` 内（构造性，应恒为 0 残差）；
2. `apply_output` 与 `apply` 在**同一 x 上自洽**：`(W-E)x == Wx - E x`；
3. 保留集边界：`retained` 中任一方向 `r` 应满足 `⟨E, r⟩ ≈ 0`（设计意图"不该被擦到的方向"）。

---

### 🟡-5 `real_separation.py` **不是半成品，能完整跑通**（交接文档结论需更正）
**位置**：`kp/latent/real_separation.py`（997 行）

交接文档称其为"被中断的半成品，没验证过"。实跑结果（CPU，缩小参数）：
```
verdict FAIL  sanity_ok True  strength OK
arms: frozen_random_encoder / joint_supervised / joint_negative / joint_no_anchor / semantic_routed
worst_dep: sup=0.2148  neg=0.2976  improvement_x=1.385
GAP: 输入分辨率 128² 过低，latent 仅 4² 格，细节通道几乎没有空间可分
GAP: [Arm E] 探针 R² 相差过大（语义 0.960 vs 纹理 0.454）⇒ 弱侧数字不可解释
```
**五个实验臂全跑通、尺子自检（oracle 过 / leaky 挂）通过、缺口显式报出、
`conclusion_strength` 机制生效**。代码完整度远超"半成品"。

（注：`verdict FAIL` 是因为我把 `size` 降到 128²、latent 只剩 4² 格——模块**自己
正确地报出了这个缺口**。这恰恰证明判据方向是对的，不是 bug。）

**唯一实质问题**：无任何读者。`grep` 全库，只有 `tools/g2_real_vae.py:42,77` 引用，
**`kp/selftest.py` 完全不碰它** ⇒ 这 997 行（含 5 个对照臂、活性守卫、
多置换聚合、显式缺口机制）**全部不在 73 项自检覆盖范围内**。

**建议**：把 `run_real_g2` 的**小参数冒烟**（如 `size=128, steps=20, n_perm=3`）
加进 selftest，断言的是**结构性不变量**而非判定结果：
`rulers.sanity_ok is True`、`len(arms)==5`、`conclusion_strength` 有值、
且"低分辨率时必须报出分辨率缺口"。这样既纳入覆盖，又不会因算力/数据波动而假失败。

---

### 🟡-6 `param_count()` 与真实参数量口径不符，且无人消费
**位置**：`kp/config.py:55-59`

```python
per_block = 4*d*d + 2*d*int(d*mlp_ratio) + 4*d     # 少了 adaLN 的 8d²
return L*per_block + 2*d*LATENT.total_ch
```
`dit.py:9-16` 的自述是每 block **17·d²**（含 `adaLN = Linear(d→8d) = 8d²`），
而 `param_count()` 的 `4d²+5d² = 9d²` **完全没有 adaLN 项**，且 `2*d*int(d*mlp_ratio)`
在 `mlp_ratio=2.5` 时因 `int()` 截断成 `2*1152*2880` 而非精确 `5d²`。

**核实**：`param_count()` 无任何调用方（AST 扫描无读取）。
**建议**：删掉，或修正为 `17·d²·L` 并明确它是"粗略估算"（docstring 已说"粗略"，
但误差来源是**漏项**而非"粗略"，性质不同）。

---

### 🵦-7 其他代码质量项（均已核实为**非问题**，记录以免重复排查）

- **异常吞掉**：全库仅 3 处 `except: pass`，全在 `_rm()` 临时文件清理与
  `paths.py:85` 的 fallback，**语义正确、无掩盖**。无 `warnings.filterwarnings` 掩盖。
- **硬编码魔数**：`nvfp4.py:28 DEFAULT_BLOCK=16`、`FP4_MAX=6.0` 与
  `QuantCfg.block_size=16` 数值一致但**未走 config**（属 1.2 表的延伸）；
  `real_separation.py` 的 `max_dep=0.20` 同样硬编码但与 G2 门线一致且有 docstring 说明。
- **`dit.py` 的 `assert dim % heads == 0`（`:117`）** 是**唯一一处放在运行路径上的
  业务断言**，`-O` 下会失效；但同一约束由 `head_dim = dim//heads` 在类型/形状上
  间接保障，风险可接受。
- **CharaBridge 关断返回 `None` 而非零向量**：`selftest.py:556` 用 `is None` 严格断言，
  写法正确（用 `== None` 或 falsy 判断会放过零张量）。

---

## 四、对交接文档 §4.2 七旋钮清单的逐条核实结论

| 交接文档声称 | 核实结果 |
|---|---|
| `quantize_injection_point` 死 | ✅ 确认 0 读取 |
| `gate_dtype` 死 | ✅ 确认 0 读取（fp32 门控实际由 `bus.py:36-39` 硬编码 `dtype=torch.float32` 保证，行为正确但旋钮无效） |
| `matryoshka_tokens` 死 | ✅ 确认 0 读取。`selftest.py:230` 仍传入该参数，属"传了但无人读" |
| `param_count()` 死 | ✅ 确认无调用方，且公式漏 adaLN 项（见 🟡-6） |
| `identity_via_cross_attention` 死 | ✅ 确认**仅 `arch_report.py:98` 读**（报告层死）。实际行为由 `dit.py` 的 `CrossAttention` 硬编码保证 |
| `identity_token_dim` 死 | ✅ 确认**仅 `arch_report.py:97` 读**（报告层死） |
| `QuantCfg` 全 8 字段死 | ✅ 确认**全 8 字段仅 `arch_report.py:85-88` 读**（报告层死）。`nvfp4.py` 硬编码 `DEFAULT_BLOCK=16` 与 spec 字面量 |

**交接文档结论正确**，但建议补上本次的**结构性补充**：
「死旋钮」应分两类——**0 读取**（7 个，改了完全无反应）与
**仅报告层读取**（12 个，改了只改打印）。后者更隐蔽，因为 grep 得到的是"有引用"。

---

## 五、一句话总结

自检的**断言质量很高**（6 条核心不变量方向全对、G3.5 负对照密度好、无自证式空断言），
主要问题不在"测什么"而在**两个结构性漏洞**：
**`-O` 下 73/73 全部失效（🔴-1，已复现）**、**3:1 比例断言围绕实际行为而非设计目标（🔴-2）**；
另有 latent 通道分离实为通道拼接（🔴-3）与 `real_separation` 被误判为半成品且无自检覆盖（🟡-5）。
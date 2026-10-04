# KP 承重墙 12 条不变量 · 只读审计

> 审计对象：`D:\model`（KokonaPolaris-S4 / KP）
> 快照时间：**2026-10-03 19:04**（所有引用均为此刻的文件内容）
> 方法：`grep` 找读者 → 读实现 → `python -m kp.arch_report` / `python -m kp.selftest` 实跑交叉验证 → meta device 逐 block 拆账
> ⛔ 本次审计**未修改任何文件**（唯一写入为本文件）；未执行任何 git 写操作。
> ⚠️ 并行修改提示：`kp/config.py`(18:55:03)、`kp/models/common.py` / `kp/latent/separation.py`(18:56:28)、`kp/models/vae.py`、`kp/quant/nvfp4.py`、`kp/capability/bus.py`、`kp/capability/delta_pack.py`、`kp/train/qad.py`、`kp/sample.py`、`kp/arch_report.py`、`kp/models/dit.py` 的 mtime **均早于**本次读取，**读到的是最新版本**。
> ⚠️ **读取之后又被并行改动**（本次核验于 19:08 复查，引用行号仍全部有效）：`kp/selftest.py`(19:06:48)、`kp/probe/attn.py`(19:07:18)、`kp/latent/real_separation.py`(19:07:41)。复查确认本报告引用的 `selftest.py:119/600/1241/1242/1245/1246` 与 `probe/attn.py:242/255` 行号未移位。若这些文件后续继续变动，§5 与 §4 的相关行号需重核。
> ✅ 硬约束遵守：本次**唯一写入**为 `out/audit_invariants.md`；`kp/`、`tools/`、`design/` 下无任何文件被本会话改动。

---

## 1. 结论速览

| 判定 | 条数 | 条目 |
|---|---|---|
| ✅ **完全一致**（config 有值 + 代码真的按它执行） | **8** | #1 32× 压缩 · #2 40ch · #3 1024 token · #5 adaLN 8 段 · #7 softmax 锚点 · #9 身份 256 + cross-attn · #12 谱检查 · #10 前两半（gate 0.0 / fp32） |
| ⚠️ **有条件成立**（值对，但表述/口径需要限定） | **3** | #4 17·d² · #6 3:1 · #11 NVFP4 |
| ❌ **偏离** | **0** | — |
| 🚨 **附带发现（不属于 12 条，但属同一类问题）** | **7 个死的旋钮 + 5 处写死的魔数** | 见 §4 |

**一句话**：12 条里**没有一条是错的**——代码跑出来的行为与设计一致，`arch_report` 与 `selftest`（**70/70 全通过**）都能复现。但**「值对」不等于「旋钮活着」**：#4/#6 的表述需要限定口径，#10/#11 里**最关键的两个旋钮（`quantize_injection_point`、`quantize_proj_out`）是死的**——行为靠硬编码保证，不靠 config 保证。

---

## 2. 对拍表

| # | 不变量 | 代码实况（文件:行号） | 一致? | 备注 |
|---|---|---|---|---|
| 1 | latent 空间压缩率 **32×** | `kp/config.py:15` `spatial: int = 32`；`kp/models/vae.py:40` `STRIDES = 5`、`:93 compression() = 2**5`；实跑 `HybridVAE().encode(1,3,1024,1024)` → `(1,40,32,32)` | ✅ | **两个真源但同值**。`vae.py` **不读** `LATENT.spatial`，靠写死的 `STRIDES=5`；改 `config` 不会改 VAE |
| 2 | 语义 8ch + 细节 32ch = **40ch** | `kp/config.py:16,17,20`；`vae.py:39 LATENT_CH`、`vae.py:86-91 split/merge`；`kp/latent/hybrid.py:141-153 split_channels/join_channels`；`dit.py:326` `latent_ch = latent_ch or LATENT.total_ch` | ✅ | 通道边界在代码里处处显式可见；实跑输出通道 40 |
| 3 | 1024² → **恰好 1024** token（patch_size=1） | `config.py:22-25 tokens()`（`side=1024//32=32`，`32²=1024`）；`dit.py:371` `patch_embed` 是纯通道线性（**无空间 patchify ⇒ 结构上 patch_size=1**）；forward-hook 实测 `tokens = 1024`，`out=(1,40,32,32)` | ✅ | 无硬编码 1024/32；全链路由 `LATENT.spatial` 推导 |
| 4 | 每 block **17·d²**（attn 4d² + MLP 5d² + adaLN 8d²） | adaLN 确为 8 段：`dit.py:244 nn.Linear(d, 8*d)`（实测 out_features = 8d）。attn：`dit.py:120-121` `qkv(3d²)+out(d²)`；MLP：`dit.py:156-157` `fc1/fc2` 各 2.5d²。**但**：attn+MLP 是 **buffer 不是 Parameter**（`dit.py:77-82 _gl` → `bus.py:97 register_buffer`）；且 **5 个 block 是 double-stream（+4d²，`dit.py:247-248`）、2 个 block 带身份 cross-attn（+4d²，`dit.py:251`）**；17d² 也漏了 adaLN bias(8d) 与 norm 参数 | ⚠️ | 见 §3.1。**18→17 的改动是真的**（`g_geo` 段确已删除，实测 `adaLN` 输出 8d 宽）；但「每 block = 17d²」只对**名义 block**成立。KP-S 实测全模型 594.8M vs 名义 `17d²×24`=541.5M，**+9.9%** |
| 5 | adaLN **8 段**，第 8 段（下标 7）= 身份门控 | `dit.py:244` 8 段；`dit.py:284-288` 切片 `g_txt = mod[:,6d:7d]`、`g_id = mod[:,7d:8d]`；`dit.py:306-307` 唯一的消费者就是 `identity_cross`；`dit.py:262` 段 7 的 bias 保持 0（adaLN-Zero 保护）；`selftest.py:600` 直接写 `blk.adaLN[-1].bias[7d:8d]` 并断言切片非空 | ✅ | 前向**只消费段 2/5/6/7**（+0-5 的 shift/scale），无第 9 段残留 |
| 6 | 混合注意力 **3:1**（每 4 层 3 linear + 1 sigmoid） | `config.py:43-44` `ratio_gated_linear=3 / ratio_sigmoid=1` → `dit.py:341-342` 传入；`dit.py:59-74 build_attn_plan` 循环 `[L,L,L,S]` | ⚠️ | 循环本身严格 3:1，但**全局计数不是 3:1**：softmax 锚点占了循环外的一个槽位。KP-S(L=24) 实测 `linear=18 / sigmoid=5`（=3.6:1）；selftest 用 L=32：`linear=24 / sigmoid=7`（=3.43:1），其断言 `0.2 ≤ sigmoid/linear ≤ 0.35`（`selftest.py:1245`）实测 0.292 ✅ 通过。见 §3.2 |
| 7 | softmax 锚点 **恰好 1 次，在第 0 层** | `dit.py:59-74`：`softmax_anchor=True` 时 `plan.append(SOFTMAX)` 先于循环；实测 24 层 plan 首元素为 `softmax` 且 `plan.count('softmax')==1`；`selftest.py:1242` `assert c["softmax"]==1` | ✅ | 只在主干注意力计划内计数 1 次。⚠️ 全库仍有其它 softmax：`character/fitter.py:80`（身份池化）、`dit.py:90`（SOFTMAX 分支本身）——**不属主干**，但字面上「全库恰好 1 次 softmax」不成立 |
| 8 | QK-Norm **开**，对 q/k 各做 RMSNorm | `config.py:45 qk_norm=True` → `dit.py:245 Attention(..., cfg.qk_norm)` → `dit.py:124-128, 136-138` 对 q/k 各建 `RMSNorm(head_dim)`；实测 `q_norm/k_norm = RMSNorm/RMSNorm`。双流路径 `dit.py:275-276` 也对 txt-q / img-k 施加。`selftest.py:1246` 用 `qknorm_logit_bound(1792,16)=10.58` 复核 Cauchy–Schwarz 上界 | ✅ | ⚠️ 小瑕疵：`dit.py:251` 构造 `CrossAttention` 时**没传 `cfg.qk_norm`**，靠 `CrossAttention.__init__` 的默认 `qk_norm=True`（`dit.py:166`）。⇒ 把 `cfg.qk_norm` 设为 False，身份 cross-attn 的 QK-Norm 仍会开 |
| 9 | 身份 token 预算 **256**，走 **cross-attention 不拼主序列** | 预算：`config.py:93 identity_tokens=256` → **有真读者** `kp/character/fitter.py:56 n_tokens = CAP.identity_tokens`，`:65` 建 `(256, view_dim)` query。接线：`dit.py:306-307` 身份只作为 K/V 进 `identity_cross`。全 DiT 前向**唯一**的 `torch.cat` 是 `dit.py:298 torch.cat([txt, h])`（拼的是**文本**）；`identity_ctx` 在 `dit.py:379-381` 只做投影后外传 | ✅ | 「不拼主序列」是**结构性保证**（grep 全 `kp/` 的 `torch.cat` 无第二处主干序列拼接）。⚠️ 需注意 `identity_anchor_layers` 默认为空 ⇒ 不显式指定时身份 cross-attn **不建**（`dit.py:327-328, 345`）；`arch_report.py:31` / `qad.py:180` 传 `[1, layers//2]` |
| 10a | 门控初始 **0.0** | `config.py:91 gate_init=0.0` → 真读者 `bus.py:36` `CAP.gate_init if gate is None else gate`、`kp/typography/typography_pack.py:42` 同款 | ✅ | 实测 `CAP.gate_init = 0.0` |
| 10b | 门控 dtype **fp32** | `bus.py:35-38 nn.Parameter(torch.tensor(..., dtype=torch.float32), requires_grad=False)`；`typography_pack.py:41-42` 用默认 fp32 | ✅（值对） | 🚨 **旋钮是死的**：`config.py:92 gate_dtype="float32"` **零读者**（全库仅出现在 config 自身）。fp32 由 `bus.py:37` 硬编码保证 |
| 10c | 注入点 `quantize_injection_point=False`（绕过量化器） | `config.py:49 quantize_injection_point: bool = False` | 🚨 **旋钮是死的** | **零读者**（全库只出现在 config 自身）。行为本身**确实正确**且是结构性的：`bus.py:142-153` 的 `y = F.linear(q(x), q(W)) + Σ pack(x)` —— 包拿到的是**未量化**的 `x`，其输出也不被量化；`bus.py:107-115 set_quant` 只挂在主干算子上。见 §4 |
| 11 | NVFP4：W**4bit/E2M1** · A**8bit/E4M3** · block **16** · scale **E4M3** · `quantize_proj_out=False` · sm **120** | 行为侧全部走 `kp/quant/nvfp4.py`：`nvfp4.py:28 DEFAULT_BLOCK=16`、`:29 FP4_MAX=6.0`（E2M1 精确上界）、`:27 _E4M3 = torch.finfo(torch.float8_e4m3fn)`、`:47` scale `.to(torch.float8_e4m3fn)`、`:109 DESIGN_SPEC=QuantSpec(weight="fp4", act="fp8", block=16)`；`quantize_proj_out=False` 的执行点是 `qad.py:28 SKIP_DEFAULT=("out_proj",)` → `:31-35 iter_gated` 跳过。sm120 仅 `arch_report.py:88` 打印 | ⚠️ | **数值全部吻合**，selftest 第 9 节「与 `tools/e5b_qad.py` 逐位一致」✅、「block16=9.98% < block32=11.30%」✅。但 **`kp/quant/nvfp4.py` 从不 import `QUANT`**：8 个 `QuantCfg` 字段的**唯一读者是 `arch_report.py:85-88` 的打印**。见 §3.3、§4 |
| 12 | 谱检查：阈值 **cos < 0.1**，只看前 **50%** 排名 | `config.py:98-99` → 真读者 `capability/delta_pack.py:90-93`（`cos_threshold = CAP.spectral_cos_threshold`、`rank_ratio = CAP.spectral_rank_ratio`）→ `:114 k0 = max(1, int(rank_ratio*min(m,n)))`、`:140 passed = bool(min_cos >= cos_threshold)`；`svd_pack.py:94` 复用默认。实测零扰动 `threshold=0.1` | ✅ | 判据方向也正确（`min_cos < 0.1` ⇒ 判**不合格**）。selftest 双向尺子：子空间初始化 PASS(cos=1.000) / 正交扰动 FAIL(cos=0.000) ✅。⚠️ `:73 energy_floor: float = 0.1` 是**另一个**写死的 0.1（「高排名」的相对奇异值门），不在 config 中，与本条阈值无关 |

---

## 3. 偏离项详述

### 3.1 #4 —— 「每 block 17·d²」是名义账，不是实测常数

**设计说**：每 block 恰好 17·d² = attn 4d² + MLP 5d² + adaLN 8d²（2026-10-03 从 18·d² 改来，删了死的 `g_geo` 段）。

**代码是**：
1. **18→17 的改动属实**。`dit.py:244` 确为 `nn.Linear(d, 8*d)`，实测 `out_features = 8d`；前向 `dit.py:288` 取 `mod[:,7d:8d]` 作 `g_id`；没有第 9 段。`arch_report.py:114-115` 声称的「省 102.8M ≈ 5.2%」与「删一段省 d²×L=1792²×32=102.8M」算术自洽 ✅。
2. **但 17d² 漏了两类小项**。对 d=1152 的名义 block，实测 `params + buffers = 22,572,448`，而 `17d² = 22,560,768`，**差 +11,680** = adaLN bias `8d`(=9,216) + `norm1/norm2` 各 `d`(=2,304) + `q_norm/k_norm` 各 `head_dim`(=144) + `head_gate`(=16)。
3. **更关键：不是每个 block 都是 17d²**。KP-S 实测逐 block 分类（`arch_report.py:31` 传 `identity_anchor_layers=[1,12]`）：

   | block 类型 | 数量 | GatedLinear buffer | nn.Parameter | 合计 | 相对 17d² |
   |---|---|---|---|---|---|
   | 名义单流 | 18（blk5-11,13-23） | 9d² = 11,943,936 | 10,628,512 | 22,572,448 | **17d² + 11,680** |
   | 双流（i<`double_stream_blocks`=5） | 4（blk0,2,3,4） | 13d² = 17,252,352 | 10,628,512 | 27,880,864 | **21d² + 11,680** |
   | 双流 + 身份 cross-attn | 1（blk1） | 17d² = 22,560,768 | 10,629,808 | 33,190,576 | **25d² + 12,976** |
   | 单流 + 身份 cross-attn | 1（blk12） | 13d² = 17,252,352 | 10,629,808 | 27,882,160 | **21d² + 12,976** |

   ⇒ 全模型实测 **594,837,168 (594.8M)** vs 名义 `17d²×24 = 541,458,432`，**+53.4M (+9.9%)**。KP-M：**1,874,834,592 vs 1,746,927,616，+127.9M (+7.3%)**。
4. **attn 4d² + MLP 5d² 不在 `parameters()` 里**。`dit.py:77-82 _gl` 建的 `GatedLinear` 把 `weight` 注册为 **buffer**（`bus.py:97-99`），所以单 block 的 `sum(p.numel() for p in blk.parameters())` 只有 **10,628,512 ≈ 8d²+11,680（47% of 17d²）**。`arch_report.py:17-25 _count` 特意把 buffer 单列才避免低估——这本身说明「17d²」是**跨 params+buffers 的加总口径**。

**影响**：不改行为，但**对外引用会失真**。`dit.py:15-16` 的 docstring 写「KP-S ≈ 541M / KP-M ≈ 1.75B（+其余 ⇒ 实测 ~0.58B / 1.875B）」——docstring 自己已经用「+其余」打了补丁，但「每个 block 恰好 17d²」（`dit.py:9`）这句话仍字面不成立。若有人拿 17d² 做显存/FLOPs 外推，会低估双流与身份锚点层的开销。

**建议**（不在本次授权内，仅记录）：
- 把 `dit.py:9` 的措辞改为「**名义 block** 17·d²（attn 4d² + MLP 5d² 存于 buffer + adaLN 8d² 参数）；实际 = 名义 + 8d（adaLN bias）+ O(d) 归一化项，双流层再 +4d²，身份锚点层再 +4d²」。
- 或者在 `arch_report.py` 里加一行「按 block 类型分列的账」，让 541M→594.8M 的 +9.9% 有出处。

### 3.2 #6 —— 3:1 是**循环**比例，不是**全局**比例

**设计说**：每 4 层 3 层 linear + 1 层 Sigmoid，比例 3:1。

**代码是**：`dit.py:66-74` 先无条件追加第 0 层 softmax（`dit.py:67-68`），**然后**才开始 `[L,L,L,S]` 循环（`dit.py:69-73`，`i` 从 0 重新起）。因此第 0 层占掉了循环外的一个槽位：

- KP-S L=24 → 实测 `softmax=1, linear=18, sigmoid=5` ⇒ **3.60:1**（不是 3:1）
- KP-M / selftest 用 L=32 → 实测 `softmax=1, linear=24, sigmoid=7` ⇒ **3.43:1**
- `selftest.py:1244-1245` 的断言是 `0.2 ≤ sigmoid/linear ≤ 0.35`（即 2.86:1 ~ 5:1），实测 7/24=0.292 ✅ 通过

**影响**：极小。任何**连续 4 层**的窗口（从第 1 层起）都严格是 3L+1S；只有「全模型总数」这一口径不等于 3:1，且**偏差随层数减小而增大**（L=4 时会退化到 2L+1S+1softmax）。

**建议**：把 #6 的表述写成「**锚点之后**每 4 层 3 linear + 1 sigmoid」，或在 `build_attn_plan` 的 docstring（`dit.py:61-65`）里把这点写死。

### 3.3 #11 —— NVFP4 的行为是对的，但**完全不读 config**

**设计说**：`QuantCfg` 是「单一真源」，8 个字段描述 NVFP4 数值规格。

**代码是**：`kp/quant/nvfp4.py` **完全没有 `from ..config import QUANT`**。全库 `QUANT` 的导入点只有一处：`kp/arch_report.py:12`，且只用于 `:85-88` 的四行 `print`。真正的执行路径全部是写死的常量：

| config 字段 | 实际执行点 | 是否读 config |
|---|---|---|
| `weight_bits=4` / `weight_fmt="E2M1"` | `nvfp4.py:29 FP4_MAX=6.0` + `:48 round().clamp(-6,6)` | ❌ 硬编码 |
| `act_bits=8` / `act_fmt="E4M3"` | `nvfp4.py:53-58 quant_fp8` + `:27 _E4M3` | ❌ 硬编码 |
| `block_size=16` | `nvfp4.py:28 DEFAULT_BLOCK=16`、`:109 DESIGN_SPEC(block=DEFAULT_BLOCK)` | ❌ 硬编码 |
| `scale_fmt="E4M3"` | `nvfp4.py:47 .to(torch.float8_e4m3fn)` | ❌ 硬编码 |
| `quantize_proj_out=False` | `qad.py:28 SKIP_DEFAULT=("out_proj",)` | ❌ **硬编码字符串元组** |
| `sm_arch=120` | 无执行点（纯环境标注） | ❌ 仅打印 |

**影响**：
- **当前安全**：8 个值与硬编码常量**全部一致**，改任何一边都不会立刻出错。
- **未来危险**：把 `QUANT.block_size` 改成 8，**什么都不会发生**——`quant_fp4` 仍按 16 分块，而 `arch_report` 会打印 `block=8`。这是典型的「静默陷阱」：**报告说 A，代码做 B**。
- `quantize_proj_out` 尤其危险：它是**行为**旋钮（决定哪些层被量化），却被写死成 `"out_proj"` 这个**子串匹配**（`qad.py:33 any(s in name for s in skip)`）。`dit.py:352` 的注释「⚠️ 名字须含 out_proj（QAD 跳过它）」坦承了这种字符串耦合——**重命名该属性会静默改变量化范围**。

**建议**：让 `nvfp4.py` 顶部 `from ..config import QUANT`，用 `DESIGN_SPEC = QuantSpec(weight="fp4", act="fp8", block=QUANT.block_size)`；让 `qad.py:28` 的 `SKIP_DEFAULT` 由 `QUANT.quantize_proj_out` 推导（`()` 或 `("out_proj",)`）。**注意 `nvfp4.py` 当前不 import config 可能是刻意的解耦**（避免量化层依赖架构常量），改动前应先定这个边界。

---

## 4. 死的旋钮 / 写死的魔数

### 4.1 🚨 配置里有，但**没有任何代码读它**（7 个）

| 旋钮 | 位置 | 唯一「读者」 | 实际行为由谁保证 |
|---|---|---|---|
| `DiTCfg.quantize_injection_point = False` | `kp/config.py:49` | **无**（全库仅 config 自身） | `bus.py:142-153` 的前向结构（正确，但与旋钮无关） |
| `CapabilityCfg.gate_dtype = "float32"` | `kp/config.py:92` | **无** | `bus.py:37` `dtype=torch.float32` 硬编码 |
| `CapabilityCfg.identity_via_cross_attention = True` | `kp/config.py:95` | `arch_report.py:98`（**只 print**） | `dit.py:306-307` 的 cross-attn 接线（结构性，旋钮无关） |
| `CapabilityCfg.identity_token_dim = 1024` | `kp/config.py:94` | `arch_report.py:97`（**只 print**） | `character/fitter.py:52 dim: int = 1024` 硬编码默认值 |
| `QuantCfg` 全部 8 字段 | `kp/config.py:71-80` | `arch_report.py:85-88`（**只 print**） | `quant/nvfp4.py:27-29,109` + `train/qad.py:28` 硬编码（见 §3.3） |
| `DiTCfg.matryoshka_tokens = (256, 1024)` | `kp/config.py:47` | **无**（只在 `selftest.py:230` / `probe/real.py:749` / `train/qad.py:196` 被**重新赋值**成 `(16,64)`，从未被读取） | `kp/sample.py:19` `make_matryoshka_schedule(lo_tokens=256, hi_tokens=1024)` 与 `sample.py:41 matryoshka=(256,1024)` 两处硬编码默认值 |
| `DiTCfg.param_count()`（方法） | `kp/config.py:55-59` | **无**（`arch_report.py:79` 调的是 `TextTowerCfg.param_count`，不是这个） | 该方法公式为 `4d² + 2d⌊d·mlp_ratio⌋ + 4d = 9d²+4d`，**与 17d² 口径矛盾**——它是一个会误导人的陈旧方法 |

> 参照：`RuntimeCfg` / `RUNTIME`（`config.py:171-196`）的死旋钮问题**已被前一轮审计登记并自行标注为「预留（空壳）」**（见其 docstring 与 `ONBOARDING.md §3`）。本次新发现的 7 个**尚未被标注**。

### 4.2 写死的魔数，绕过了 config

| 魔数 | 位置 | 绕过了什么 | 风险 |
|---|---|---|---|
| `DEFAULT_BLOCK = 16` | `quant/nvfp4.py:28` | `QUANT.block_size` | 改 config 无效（见 §3.3） |
| `FP4_MAX = 6.0` | `quant/nvfp4.py:29` | `QUANT.weight_fmt="E2M1"` | 换 FP 格式（如 E3M2）需改代码 |
| `SKIP_DEFAULT = ("out_proj",)` | `train/qad.py:28` | `QUANT.quantize_proj_out` | **字符串耦合**：重命名 `dit.py:352` 的 `out_proj` 属性会静默改变量化范围 |
| `energy_floor: float = 0.1` | `capability/delta_pack.py:73` | 无对应 config 字段 | 与 #12 的 `cos_threshold=0.1` 是**两个不同的 0.1**，易混淆；改「高排名」定义只能改代码 |
| `STRIDES = 5` | `models/vae.py:40` | `LATENT.spatial`（=32） | 改压缩率需同时改 config、VAE、`hybrid.py:207`、`real_separation.py:310-311`、`typography/composite.py` 共 5 处 |
| `CrossAttention(..., qk_norm=True)` 默认值 | `models/dit.py:166`，**`dit.py:251` 调用时未传 `cfg.qk_norm`** | `cfg.qk_norm` | 把 `qk_norm=False` 时，身份 cross-attn 的 QK-Norm **仍会开**（行为不一致，但偏安全方向） |
| `identity_anchor_layers=()` | `probe/real.py:739`（默认空） | — | G3.5 真探针默认**不建**身份 cross-attn ⇒ 探到的主干结构与 `arch_report` 报的（`[1, layers//2]`）不同 |

### 4.3 ✅ 确认**活着**的旋钮（对照组，供后续审计复用）

`CAP.gate_init`（`bus.py:36`、`typography_pack.py:42`）· `CAP.identity_tokens`（`character/fitter.py:56`）· `CAP.spectral_cos_threshold` / `CAP.spectral_rank_ratio`（`delta_pack.py:91,93`）· `LATENT.semantic_ch` / `detail_ch` / `total_ch`（`vae.py`、`hybrid.py`、`dit.py:326`、`separation.py`）· `LATENT.spatial`（`hybrid.py:207`、`real_separation.py`、`typography/composite.py`）· `DiTCfg.qk_norm` / `ratio_gated_linear` / `ratio_sigmoid` / `double_stream_blocks` / `mlp_ratio` / `heads` / `dim` / `layers`（`dit.py:245,250,341-344`）· `AXIS.*`（`probe/axis.py:48`、`probe/real.py:67`）· `CAPTION.*`（`data/captions.py:22`）。

---

## 5. ⚠️ 无法判定的（⛔ 不猜）

1. **`identity_anchor_layers` 在生产路径上的取值** —— `arch_report.py:31`、`qad.py:180` 传 `[1, layers//2]`，`probe/real.py:739` 默认为 `()`，`selftest.py:231` 传 `[1]`。**没有任何一处代码规定「生产环境应当挂几层身份锚点」**。`CAP` 里也没有对应字段。因此 #9 的「身份走 cross-attention」可判 ✅，但「身份在第几层注入」**无设计真源可比对，无法判定**。

2. **`double_stream_blocks = 5` 的来源** —— `config.py:41` 注释写「浅层前 N 个 block 保留 double-stream」，但这 5 条 12 条不变量里没有它，我也没有找到 design/ 里的推导。**不判定对错**。

3. **`sm_arch = 120` 是否真匹配目标硬件** —— 该字段零执行点（只 print），且本次审计**未做 GPU 探测**。配置文件里的「Blackwell 消费卡；无 tcgen05/TMEM」是注释而非实测。**无法判定**。

4. **`LatentCfg.spatial = 32` 与 `vae.py::STRIDES = 5` 双真源是否会漂** —— 当前同值（2⁵=32）。**无法判定**这是「有意解耦」还是「未清理的重复」，因为没有找到任何说明这两者应当独立的文档。

5. **`quant_fp4` 与官方 NVFP4 的**逐位**一致性** —— selftest 第 9 节只与**本仓库的** `tools/e5b_qad.py` 对拍（「逐位一致」✅），**没有与 TensorRT ModelOpt 官方实现对拍**（`quant/nvfp4.py:3-7` 的 docstring 说手写是因为 ModelOpt 不提供反向，但没说数值布局是否等价）。**本条超出 12 条范围，仅登记为未验证项。**

6. **`arch_report` 打印的「冻结W0 / 可训练」两栏是否互斥完备** —— `arch_report.py:23-24` 把 `parameters()` 与 floating-point `buffers()` 相加。我核对过 `GatedLinear.weight` 确实进 buffer、`head_gate`/`adaLN`/`RMSNorm` 确实进 parameters，合计 594,837,168 与 `dit.py:15` docstring 的量级自洽。但 **`dit.py:265 _adaLN_zero` 这类 bool buffer 被 `is_floating_point()` 过滤掉**——这是有意为之还是遗漏，**无法从代码判定**。

---

## 附：实跑命令与关键输出

```
cd D:\model; $env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m kp.arch_report     # exit 0
.venv\Scripts\python.exe -m kp.selftest        # exit 0 —— 结果：70/70 通过，全部通过 ✅
```

`arch_report` 实测：KP-S 1152/24/16/2.5 → 可训 264.3M + 冻结W0 330.6M = **594.8M**（vs 标称 0.6B，−0.9% ✅）；KP-M → **1.875B**（vs 标称 1.5B，+25.0% ⚠️，已登记为待拍板）；1024² → **1024 token** ✅；谱检查 `cos < 0.1` / 前 50% ✅。

`selftest` 中与本 12 条直接相关的通过项：
- §1「token 数自检（256/512/1024）」→ `1024x1024→1024tok` ✅
- §3「Δ-Pack 谱检查」→ 子空间 PASS(cos=1.000) / 正交 FAIL(cos=0.000) ✅
- §6「HybridVAE 32× 编解码形状」→ `256²→latent(1,40,8,8)→256²（32×）` ✅
- §7「Fitter 输出身份 token … （走 cross-attention，不拼主序列）」✅
- §9「与 `tools/e5b_qad.py` 的 quant_fp4 逐位一致」/「block16=9.98% < block32=11.30%」✅
- §13「与主干联动：adaLN-Zero 下身份门控为 0，注入 token 不应改变输出」✅（切片 `adaLN[-1].bias[7d:8d]` 非空即证 8 段）
- §21「3:1 计划构成正确 + QK-Norm logit 上界」→ `{'layers':32,'softmax':1,'sigmoid':7,'linear':24} ｜ 3:1 ≈ 1:3.4 ｜ logit 上界 10.58` ✅

⛔ 本报告未对 `kp/typography/__init__.py`、`kp/probe/*`、`kp/paths.py`、`tools/*` 做覆盖审计（超出 12 条范围，且与并行修改者重叠）。

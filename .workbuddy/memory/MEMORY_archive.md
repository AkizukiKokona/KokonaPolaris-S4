# KokonaPolaris 记忆·全量档案（不注入，仅按需查阅）

> 本文件保存 MEMORY.md 精简前的完整实测细节与历史快照。
> MEMORY.md 只保留「决策 + 硬约束 + 结论 + 协作约定」；具体数字/表格放这里。

## ⭐⭐ 两台开发机（**2026-10-03 用户明确要求分名：禁止混记**）

| 机器名 | 硬件 | 状态 | 说明 |
|---|---|---|---|
| **kokona** | RTX **5050** Laptop GPU / 20 SM / 8GB / sm_120 / 驱动 610.74 / 标称 115W(max) | **已迁出** | 本档案里所有标「kokona」的数字**只作历史快照** |
| **viim** | RTX **5070** Laptop GPU / **36 SM** / 4608 core / 8151 MiB / sm_120 / 驱动 591.91 / CUDA UMD 13.1 / VBIOS 98.06.2a.80.08 | **当前工作副本所在机** | PCI ID `0x2D58`；宿主名 `LAPTOP-SQ8UPAUC`；序列号 `T5NRKD02K501217` |

⚠️ **nvidia-smi 报的是「RTX 5070 Laptop GPU」，不是 5070 Ti**（用户口头称 5070ti）——
记数字时以 nvidia-smi 为准，免得以后对不上。
✅ **两台机架构参数完全相同**（sm_120 + 8GB + 384GB/s）⇒ `kp/config.py` 架构常量零改动；
❌ 但 **FLOPS / 功耗 / SM 数一律不可跨机比较或拼接**（`env.sh` 已导出 `KP_MACHINE` 标身份）。
🔴 **规则**：任何实测数字写下时必须带机器名；发现没带机器名的数字，**先怀疑是串味的**。

## ⭐【viim】实测基线（2026-10-03 白天 · 满血窗口 · `tools/bench_115w.py`）
> 原始日志 `out/g0_baseline/bench_5070_run1.log`。起因：`MEMORY_ops.md` 明写「新机跑 GPU 前必须重测并回写」，
> 且 **FP4 算力此前是 archive 的空白**。

| 项 | 【viim】实测 | 对照【kokona】 |
|---|---|---|
| GPU / 显存 | RTX **5070 Laptop** / 8151 MiB / sm_120 / 36 SM | 5050 / 8151 MiB / sm_120 / 20 SM |
| **bf16** | **44.2 TFLOPS**（8192³×30）· 42.8（12288³×12） | 25.8 @80W ⇒ **1.71×** |
| **FP4（block-scaled）** | **238.8 TFLOPS**（4096³×40）· 148.1（2048³）· 150.9（8192³） | 120.5 @113W ⇒ **1.98×** |
| 功耗 | soak 起始冲 **115.9 W**，30s 后**稳态掉到 ~90 W / P2** | 113.8 峰 @82°C |
| 温度 | soak 末 **86–87°C**（起始 53°C） | 82°C |
| 显存 | 结束后空闲 **6.57 GiB / 7.96 GiB** | 空闲 3.2–6.3 GiB |

🔴 **三条必须记住的口径警告（【viim】）**：
1. **FP4 的 238.8 偏乐观**：4096³ 时 0.575 ms/iter，A+B+C 驻留量按 bf16 已 >0.2GB、8192³ 时 >2GB，
   **很可能被 L2(32MB)/带宽/缓存效应放大**；而且 8192³ 一档只有 150.9 TFLOPS。
   ⇒ **不要拿 238.8 做规划**；要正式数字须重做（多档 + 冷缓存 + 最小二乘）。
2. **功耗不是恒 115W**：30s soak 即掉到 ~90W（P2，86°C）。与【kokona】「静音/满血两档」不同，
   【viim】是**持续负载下回落** ⇒ 长任务预算按 **90W 档**更稳。
3. **`bench_115w.py` Phase D 失败**：`PermissionError: [WinError 5] C:\Users\TX\.triton`
   ⇒ 见「本机环境新事实」的 `TRITON_CACHE_DIR` 缺口。

## ⭐【viim】实测基线 · 早先那次（**同为 viim，不是 kokona —— 别误读成两台机**）
> ⚠️ 本条与上面「白天那次」**是同一台机（viim）的两次测量**，差异来自**矩阵档位**：
> 本条是 **4096³**（compute-bound 档，缓存友好）⇒ 47.4；白天那次主档是 **8192³ / 12288³**
> （显存流量大）⇒ 44.2 / 42.8。**不是机器变慢，也不是两台机。**
> 🔴 **待办**：用**同一脚本同一档位**把 4096³ 补齐，才能做真正的同条件对照（勿直接比较不同档位）。

| 项 | 实测值 |
|---|---|
| GPU / 显存 | RTX 5070 Laptop GPU / 8151 MiB（7.96 GiB） |
| **SM 数** | **36**（×128 = 4608 CUDA core） |
| compute_cap / L2 | **sm_120** / **32 MB** |
| 驱动 / CUDA UMD | 591.91 / 13.1 |
| 功耗上限 | **Current & Requested 115 W**（Default 55 W / Min 5 W） |
| **bf16 算力** | **47.4 TFLOPS**（**4096³ ×300**，CUDA event 纯计时） |
| 满载工况 | **114.8 W 峰值 / 中位 114.4 W**、SM **2722 MHz**、占用 **100%**、**76°C** |
| P-state / 降频 | **P0**；SW Power Cap / HW Slowdown / SW Thermal **全 Not Active** |
| 每瓦算力 | 0.41 TFLOPS/W（对照【kokona】0.32） |

**对照【kokona】（已迁出）**：20 SM / bf16 25.8 TFLOPS @80W（静音 20.9 @45.7W）/ FP4 120.5 @113W。
⇒ 【viim】的 bf16 是【kokona】满血档的 1.84×，SM 数 1.8×，**且功耗上限同为 115W、架构同为 sm_120、显存同为 8GB**
⇒ **`kp/config.py` 架构常量一个都不用改**，只是算力更宽裕。


🔴 **测量方法论（踩过两次，务必照做）**：
1. **绝对不要把 `nvidia-smi` 子进程放进 CUDA event 的计时区间** —— 每次调用 ~150ms，会把
   GPU 时间虚高成「4.6 TFLOPS / 21W」这种假数字（实测虚低 **10×**）。
2. **算力与功耗分两阶段测**：阶段 A 纯 matmul + event 计时；阶段 B 另开线程采样
   `nvidia-smi`，只看功耗不看时间。
3. **预热必须充分**：4096³ 冷启动单发只有 7.7 TFLOPS，预热 30–40 次后才是 47 TFLOPS。
4. 采样要读 `clocks_event_reasons.active`（本机恒 `0x0` ⇒ 无任何降频）。

**迁出机**：RTX 5050 Laptop / 8151 MiB / compute_cap 12.0（sm_120 ✓）/ 20 SM / 驱动 610.74（CUDA UMD 13.3）。
- 功耗标称 5/55/**115W(max)**。实测峰值 **113.8W 可达（=上限 98.1%）**，82°C，`clocks_event_reasons.active` 恒 `0x0`。
- 五形态实测：D2D 拷贝 68.4 W / fp32 elementwise 66.9 W / bf16 tensor 79.7 W / 三流并发 79.8 W / **FP4 tensor 113.0 W（峰 113.8）**。
- 两档算力：静音 45.7 W = bf16 20.9 / FP4 99.2 TFLOPS；满血 = bf16 **25.8 @80W** / FP4 **120.5 @113W**。
  - FP4 功耗 ×2.47 而算力仅 ×1.21 → 静音档已拿满血 **82.3%**；bf16 ×1.74 而算力 ×1.23 → **81.0%**。
  - **FP4/bf16 比值稳定 4.7×**（静音 4.75 / 满血 4.67）。FP4 每瓦 1.07 TFLOPS/W，是 bf16（0.32）的 3.3 倍。
  - FurMark 到 109 W 是因它是「power virus」（烧光栅/ROP，不产算力）。
- 20 SM FP4 理论峰值 ≈ 20×1024 MAC×2×2.797 GHz ≈ **114.6 TFLOPS**，实测 120.5 已在该量级 → 再给瓦数也买不到 FLOPS。
- 显存：桌面自占 1.6–4.7GB，实测空闲 3.2–6.3GB。
- 环境：nvcc 13.2，`CUDA_PATH`=v13.2，三套工具链并存（12.4/13.2/13.3）→ flashinfer 失败原因。
- Python：AI 环境在系统 Python 3.12.10（`C:/Users/Akizuki/AppData/Local/Programs/Python/Python312/python.exe`）；WorkBuddy 托管 3.13 是空环境。
- torch 2.8.0+cu128，`arch_list` 含 sm_120。已装：triton-windows 3.6.0 / transformers 5.10.1 / diffusers 0.34.0 / accelerate 1.14.0 / peft 0.17.1 / nvidia-modelopt 0.46.0 / cuda-tile / cupy-cuda12x / pytorch-fid / torchmetrics / clip-anytorch。未装：bitsandbytes / transformer-engine / flash_attn / xformers。
- 3 个环境缺陷：transformers 5.10.1 太新（HybridCache ImportError → 打坏 modelopt diffusers/peft 插件）；nunchaku 需 diffusers≥0.35.1；flashinfer 是 cu12 构建。修复原则：项目独立 venv `D:\model\.venv`。
- 性能定位：本机 bf16 25.8 / FP4 120.5 TFLOPS vs 8×H100≈7900 → **验证机不是训练机**。G6 全量预训练必须租云。
- 磁盘：D: 可用 180GB，新增大文件一律落 `D:\model\`，不碰 C 盘。入口 `source /d/model/env.sh` 后用 `"$KP_PY" xxx.py`；版本定格 `D:\model\requirements.lock.txt`。
- 探针脚本在 `D:/model/tools/`：`gpu_probe*.py` / `bench_115w.py` / `power_ceiling_probe.py` / `quant_compare.py`。

## G1 靶子：Sana 1.6B（完整）
`Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`（Apache 2.0），落 `D:\model\models\`（11GB，含官方 int4 参照）。
- 选它因与 KP 同构：1024² → latent **(1,32,32,32)**；`patch_size=1` + `sample_size=32` ⇒ **token 数 = 1024 ＝ KP 目标**；VAE = DC-AE f32c32（32×）；主干 **1.6045B / 20 层 / hidden 2240 / mlp_ratio 2.5 / 线性注意力**。
- 显存账：transformer 3.2G + Gemma2-2B 5.2G + vae 0.62G ≈ 9.0G > 可用 6.84G → 必须 `enable_model_cpu_offload()`。实测加载 ~3s、1024²×20 步 13–22s、峰稳定 **4.96GB**。
- 脚本：`tools/{fetch_sana,verify_sana,sana_bf16_baseline}.py`。

## E5 量化实测（真模型 PTQ，15 例平均；详见补充09）
| 配置 | 相对误差 | cos |
|---|---|---|
| W4A4（官方默认） | 29.72% | 0.9542 |
| W4A4 + 4/6 + MSE 校准 | 29.57%（改善 0.5%，耗时 47×） | 0.9552 |
| **W4A8（设计档）** | **9.98%** | 0.9947 |
| W4A16 | 7.74% | 0.9968 |
- 误差几乎全来自「激活」量化（权重降 4bit 只花 7.74%；激活降 4bit 再加 21.98，3.84×）。端到端 PSNR：W4A16 16.87 · W4A8 16.28 · W4A4 13.90。
- 误差随 timestep 倒 U 形：t=999 19.5%，t=500 35.4%。
- 官方 `NVFP4_DEFAULT_CFG` 真身 = weight+input quantizer（E2M1/block16/E4M3 scale，effective 4.5bit），其余 23 条 enable:false（含 proj_out 默认不量化）⇒ 即 W4A4 + 朴素 max 校准。
- ModelOpt 行为边界：NVFP4 fake-quant 忽略 block_sizes；手写量化器时 block 真实生效（16 优于 32，低 9–25%；proj_out 最敏感）；FP8 模拟走 `_fp8_eager` 回退，确实生效。
- 永久规范：① 逐像素 PSNR/SSIM 只用于同轨迹复现性，不可判画质 → G1 主判据须分布级；② PTQ 是模拟量化，其峰值显存不能论证 NVFP4 显存收益。

## G1 快速判据（`out/e5b/g1_fast.json`）
方法：`d_quant`（同 prompt 同 seed，量化臂 vs bf16 特征距离）÷ `d_ref`（bf16 同 prompt 不同 seed 自然离散度，282 对，**d_ref = 14.576**）。
| 臂 | d_quant | d_quant/d_ref | PSNR |
|---|---|---|---|
| **W4A8** | 7.799 | **0.535** | 16.28 |
| W4A16 | 7.680 | **0.527** | 16.87 |
| W4A4 | 12.960 | **0.889** | 13.90 |
三臂全 <1；W4A4 达 88.9% ⇒ 激活降到 FP8 几乎不额外损失。**G1 初步通过（W4A8）**。局限：每 prompt 仅 3 对 → 正式过门仍建议补 ≥50 张/臂 FID。

## E5b 可训性定案
- blocker：ModelOpt 0.46.0 的 NVFP4/FP8 fake-quant 不提供反向 → 量化后 Sana 第一步 backward 就崩（六配方全灭）。
- 真根因（抓调用栈）：插件把 SDPA 换成内部 `FP8SDPA`（只实现 forward、无 backward；`plugins/diffusion/diffusers.py:214`）⇒ 与量化器开关/CUDA 扩展均无关。修：patch `md.FP8SDPA` → 原版 SDPA（`tools/e5b_patch_probe.py`）→ W4A8/W4A4 全参 backward OK。
- 容量结论：全参 QAT 本机不可行（AdamW 状态 ≈ **12.8GB**）⇒ 必须 adapter 式 QAD（冻结主干 + 只训低秩旁路，5.99M=0.37%，峰 3.41GB）——与设计稿 Δ-Pack 同构 ⇒ 提前实证。
- 自研可微 NVFP4 fake-quant（`tools/e5b_qad.py`）：手写 E2M1+block16+E4M3 scale + STE，替换 Sana 165 Linear（proj_out 不量化），独立复现倒 U 形。bf16 全量微调对照：1024² 12.7 s/step / 峰 6.84GB；开梯度检查点 5.66 s/step / 6.3GB。

## E6 / P2′ T0 排版链路
`tools/typography/` → `out/e6/`。「心夏北极星」5 字全部 1 glyph 精确命中（横排+竖排），3.79s 出全套 ⇒ T0「正确率 100%」已实证。机器可用中文字体 37 faces / 34 CJK（含日文 Yu Gothic → 竖排日文有依托）。

## 数据线（现成）
NOVA-Human（10.2k VRM，16 随机 + 4 正交视角，163.2k 图，512²，Fitter 完美配对；⚠️ 仅研究用）；See-through（9,102  Live2D 自举 19 类语义层，开源）。标准正交视图由我们用渲染器生产，不指望用户提供。

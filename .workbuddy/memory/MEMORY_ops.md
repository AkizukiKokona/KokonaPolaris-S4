# KokonaPolaris · 本机操作手册（不注入，按需查阅）

> 从 MEMORY.md 切出：**环境 / 数据交付 / 本机协作 / 网络** 的操作性细节。
> 其中的**铁律**在 MEMORY.md 保留一行摘要；此处是完整背景与数字。

## 开发机基线（完整数字见 MEMORY_archive.md）
> ⚠️ **2026-10-03 换机**：kokona = RTX **5050** Laptop(20 SM) → **viim = RTX 5070 Laptop(36 SM)**。
> 架构/显存/带宽全同（sm_120 / 8GB / 384GB/s）⇒ **`kp/config.py` 架构常量零改动**；
> ⚠️ 但 **FLOPS / 功耗 / SM 数一律绑死具体机器、换机即作废**（kokona 的数字只作历史快照）。
> ✅ viim 已按此重测并回写：sm_120 实算 bf16 GEMM 2048³×50 = **39.4 TFLOPS**（36 SM vs 20 SM = 1.8×）；
> ⚠️ 再上任何 GPU 任务前先 `tools/gpu_probe.py` 重测，且**记数字必须带机器名**。
> 📌 **同日第二次订正**：本块初版写「5060 Laptop(26SM)」——那只是用户口头的换机意向（`2026-10-03.md` 续跑四），
>   **实测机是 5070 / 36 SM**，全库无 26 SM 记录；一切以 `nvidia-smi` 与「当前工作副本所在机」小节为准。

**迁出机（历史）**：RTX 5050 Laptop / 8151 MiB / sm_120 ✓ / 20 SM / 驱动 610.74。
- 功耗标称 5/55/**115W(max)**；实测峰值 **113.8W 可达**，`clocks_event_reasons.active` 恒 `0x0`。
- **静音档已拿满血 82%**（FP4 99.2 → 120.5 TFLOPS）⇒ **G1 不必等解锁功耗**。FP4/bf16 稳 **4.7×**。
- ⭐ **永久规范**：功耗**必须按负载形态分别测**（D2D 68.4 / fp32 66.9 / bf16 tensor 79.7 / 三流 79.8 / **FP4 tensor 113.0**）+ **必须同时读 `clocks_event_reasons.active`**。
- 环境：nvcc 13.2（12.4/13.2/13.3 三套并存）；**AI 环境 = 系统 Python 3.12.10**
  （`C:/Users/Akizuki/AppData/Local/Programs/Python/Python312/python.exe`）；torch 2.8.0+cu128。
- 已装：triton-windows / transformers 5.10.1 / diffusers 0.34.0 / peft / **nvidia-modelopt 0.46.0** / cupy / pytorch-fid。
  未装：bitsandbytes / transformer-engine / flash_attn / xformers。
- ⚠️ 3 个环境缺陷：transformers 5.10.1 太新（HybridCache ImportError，打坏 modelopt 插件）/ nunchaku 需 diffusers≥0.35.1 / flashinfer 是 cu12。
  → **修复原则：项目独立 venv `D:\model\.venv`，不改全局。**
- 入口 `source /d/model/env.sh` 后用 `"$KP_PY" xxx.py`；版本定格 `D:\model\requirements.lock.txt`。
- 探针脚本 `D:/model/tools/`：`gpu_probe*.py` / `bench_115w.py` / `power_ceiling_probe.py` / `quant_compare.py`。

## 环境铁律（MEMORY.md 只留摘要）
- 🖥️ **夜间禁压测**：夜间整机安静模式，GPU 被压到 ~40 W ⇒ **禁止任何压力测试 / 满载 soak / 基准 / 大批量出图**。
- ⛔ **单卡独占**：只有一个 8GB GPU，**子代理/后台进程会在宿主失联后继续存活** ⇒
  每次派新 GPU 任务前先列 python 进程并清掉同名孤儿：
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` → `taskkill /F /PID <pid>`。
- ⛔ **长任务监控**：**崩溃是瞬时事件，不能「跑完再查」**。① 每步极小增量；② `-u` 实时输出、每产出立刻打一行；③ **几十秒级**读日志；④ 支持**断点续跑**。
- 💾 **不要动 C 盘**。🧹 **清理/删除类操作前必须先问**：只做只读扫描 + 报告。
- 🌐 **网络**：回环 **7897** 有代理。**PyPI → 国内镜像**；⭐ **HuggingFace → 必须走代理直连**
  （hf-mirror 192KB/s vs 代理 86MB/s，快 26×）→ 用 `env.sh` 的 `kp_hf`。⚠️ **响应快 ≠ 传输快。**
- 🖥️ **功耗档位只能用户在奥创控制中心手改**（`nvidia-smi` 推不动）。
- 🔌 **ComfyUI 已于 2026-10-03 深夜经用户同意关闭**（端口 8188 释放）→ 显存基线回 ~830 MiB。
  ⚠️ 若用户重开，显存预算扣 ~0.9–1.5GB，**绝对不要杀**。

## 数据交付（细则）
- ⭐ **格式通用、不针对特定内容**；用户只交 **3 项** —— 图片（短边 ≥1024、无水印）+ caption + ⭐**`tag`（擦除账本定位的唯一依据）**；
  **其余全自动**（姿态/深度/法线/掩码/DINOv3/质量/去重/分桶）→ **严禁派人工标注**。
  组织 `data/<批次>/{images/, manifest.csv}`（列 `file,tag,caption,source`）。
- **试点分两级**：50–100 张（打通管线）→ 500–2000 张 → 再定总量。**原「一上来 500 张」门槛定高了。**
- **「预训练」是窗口期不是瞬间**：准确说法「**越早进越便宜越稳**」。数据池增量累积，缺的标「待补」，**不为凑齐卡住项目**。
- **分工**：**我 = 通用接口与算子**（过滤接口 / 擦除算子 `E` / 账本 / 验证脚本）；**用户 = 数据池内容与合法性**。
- ⚠️ 这类数据**不是「根基」**，是池子里**普通一份子**；**给特殊地位反而更坏**（擦除更难、更伤主干）。

## 角色卡数据线（现状）
- ⭐ 角色 **Kokona · Summer Splash**（项目中文名「心夏」来源）。采纳 `raw/{f,b}.png`（游戏内截图，400×961 / 385×961，本体 309×943 / 299×935）
  → **rembg(U2Net) 去底** → `images/{front,back}.png`(RGBA) + `manifest.csv`。
- ⚠️ **角色已占满画面（943/961）⇒「拉近相机」已到头**；单眼仅 ~20–25 px 是最紧处。
  ⛔ **禁用生成式超分**（编造发丝/瞳纹 → 把"猜的"当"事实"）；可加无损裁切 + Lanczos。
- ⭐ **抠图工具**：洪水填充在本案**失败**（背景浅色光弧成屏障）→ 改用 **`rembg` 2.0.81 + U2Net + alpha matting**
  （onnxruntime 回落 CPU，**夜间安全**，~15s/张）。
- ⭐ **两个判据**：
  ①「要不要头部特写」看「**尺度一致性**」而非「谁切图」（角色卡是单一对象、单一投影尺度）；角色已占满画面 ⇒ 头特写只能作**补充**
  （全身=结构基准、特写=头部层高清参考；前提：同姿势 + **同视角、不移动相机只放大画面** + 拍头肩）。
  ② **LoRA「脸不像」归因** —— 分辨率通常**非主因**：ⓐ 脸部**像素量**太少 ⓑ 脸视角/表情单一
  ⓒ **caption 把角色独有特征也写进去**（正解：独有特征不写、只写变化项）ⓓ 过拟合/欠拟合 ⓔ 底模先验压制。
  ⭐「加人脸特写」是 LoRA 标准解法，**但对 KP 角色卡不适用**。

## 用户已有资产
- ⭐ 本机 **WAI 系模型**（ComfyUI）+ **用户自训的角色 LoRA**。⚠️ **LoRA 不能迁移到 KP**（绑死旧 `W₀`）——
  **但证明用户已在做「能力挂件」**，且它是角色卡数据线的**现成引擎**（同角色 + 只换视角/姿势 → 多视角配对数据 → 喂 P2.6/G5 Fitter）。

## ⭐ 当前工作副本所在机（2026-10-03 10:3x 实测；**与上面「迁出机」不是同一台**）
仓库新克隆到 **`D:\kokonapolaris-s4`**（孤儿目录，**先于克隆为空**）。实测：

| 项 | 实测值 |
|---|---|
| GPU | **NVIDIA GeForce RTX 5070 Laptop GPU** |
| 显存 | 8151 MiB（≈8GB） |
| compute_cap | **12.0（sm_120 ✓ 与硬约束一致）** |
| 驱动 / CUDA UMD | **591.91 / 13.1** |
| 功耗上限 | 115 W |
| 系统 Python | 3.13.14（`.../Microsoft/WindowsApps`）+ 3.14.6；另有 uv 托管 **3.12.15** |
| 磁盘 | C: 剩 142GB / D: 剩 196GB |

### 环境（2026-10-03 已装齐 ✅）
- ✅ **`.venv` = `D:\kokonapolaris-s4\.venv`**，**Python 3.12.15**（uv 托管，MSC v.1944 → 对上本机 MSVC 14.44）。**自包含**（不再依赖什么系统 site-packages）。
- ✅ torch **2.8.0+cu128** / torchvision 0.23.0+cu128；sm_120 实算 **bf16 GEMM 2048³×50 = 39.4 TFLOPS**（cap=(12,0)）。
- ✅ ModelOpt **0.46.0**，NVFP4 配方 **20 个**（⚠️ 旧记忆写 23，差异待核）。
- ✅ 全量清单见重写后的 **`requirements.lock.txt` v2**（含两步安装顺序 + 版本钉子）。装法：`pip install -r requirements.lock.txt -i 清华`，但 **torch 必须先单独走 `--index-url https://download.pytorch.org/whl/cu128`（需代理）**。
- ⚠️ **`rembg` 必须装 `rembg[cpu]` 或 `[gpu]`** —— 只装 `rembg` 会 import 即打印提示并退出（缺 onnxruntime）。
- ⚠️ modelopt 会对 transformers 4.55.4 打「not tested」警告 —— **故意的钉子**（5.x 会打坏插件），不是问题。
- ✅ git 身份：**全局 + 本仓库**均已设为 `AkizukiKokona <139216879+AkizukiKokona@users.noreply.github.com>`（用户明确指定；**所有 commit 都用它**，且**禁止 PR、直接推**）。
- ✅ **CUDA Toolkit 13.4 已装**（2026-10-03，winget `Nvidia.CUDA`）：
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`，**nvcc V13.4.59**。
  已写入 Machine 级 `CUDA_PATH` + `PATH`（`...\v13.4\bin\x64` 与 `...\bin` 两条）
  ⇒ **新开终端**才生效。⚠️ **占 C 盘 7.9GB**（C 盘可用 142GB → 121GB）——
  与「不碰 C 盘」的约定冲突，是用户明确要求安装的；如需挪到 D 盘得用官方
  `cuda_*.exe` 重装时改路径（winget 装法不给选路径）。
  ⚠️ 驱动机报的是 CUDA UMD 13.1，工具链 13.4 属同大版本 ⇒ 可用（minor version compatibility）。
- ⚙️ **本仓库 `.git/config` 推送链路**（2026-10-03 排查后写入）：
  - `url.https://github.com/AkizukiKokona/.insteadOf = https://github.com/AkizukiKokona/`
    —— **自映射抵消全局的 gh-proxy 重写**（git 取最长匹配前缀）。原因见下。
  - `http.proxy` / `https.proxy = http://127.0.0.1:7897`（仓库级）⇒ 无需环境变量即可联网。
  - 🔴 **gh-proxy 不支持 push**：`git-receive-pack` 返回 **405 Method Not Allowed**。
    而本机全局配了 `url.https://gh-proxy.com/https://github.com/.insteadof = https://github.com/`
    ⇒ 默认所有操作都会被导到 gh-proxy，**push 必失败**。fetch/clone 正常。
  - ⚠️ `git remote -v` 显示 gh-proxy 地址是**显示层重写**的假象，`remote.origin.url` 实为
    `https://github.com/AkizukiKokona/KokonaPolaris-S4`。**别被它骗了。**
  - ⚠️ 本副本**没有**旧机的本地裸镜像 `repos/KokonaPolaris-S4.git`，所以 `origin`
    直接就是云端真地址 ⇒ **本机只需推 `origin`，不需要旧机的「双推」**。
  - 🔴 **仍缺凭证**：Windows 凭据管理器无 github 条目、无 `.git-credentials`、无 `gh`、无 token。
    `credential.helper=helper-selector`（来自 PortableGit 的 **system** gitconfig）是个 GUI 程序，
    无头环境下刷 libpng 警告后挂死 ⇒ **必须用户手动 `git push` 一次做浏览器登录**。
- ❌ ~~CUDA Toolkit 未安装~~（已解决，见上）。
- ⚠️ `tools/e5b_msvc_env.bat` 里 `TORCH_EXTENSIONS_DIR` 也写死了 `D:/model/.cache/torch_ext`（`env.sh` 已提供正确值）。
- ⚠️ `env.sh` 已改为**自定位**（不再写死 `D:/model`）；未写注册表级持久变量（`kp.paths` 本就自动探测，写死反而多副本误伤）。

## G2 通道分离线（2026-10-03 建成 · 结构性门）
- ⭐ **装置**：`kp/latent/separation.py`（监督件 + 验收）+ `tools/g2_channel_ablation.py`（CLI，出 JSON）。
  自检第 19 节 5 项。**判据 = 交叉扰动下的「分支依赖度」**（不是误差比值，见下）。
- 🔴 **三个踩过的坑（都是口径问题，不是代码问题）**：
  1. **相对降幅不能拿「模型自己的误差」当分母** —— 完美分离时 base≈0 ⇒ 除法放大成假失败
     （历史实测 495%）。**加 eps 不算修好**：本模块第一版加了 `1e-6`，结果
     **oracle（构造上完全分离）算出 `nan`（0/0）**，而 **leaky 算出 `1.000`（看起来完美）**。
  2. **扰动是在批维上做的 ⇒ 必然打散配对**，所以**不能**比「扰动后输出 vs 原配对目标」——
     那样连 oracle 都会被判必错（实测报出 0.97 / 0.73 的假误差）。
     ⇒ 正解是**不变性口径**：`依赖度 = ‖branch(x') − branch(x)‖ / ‖branch(x) − branch(0)‖`
     （分母非零且能抓住「把通道彻底无视」的退化解）。
  3. ⭐ **合成数据必须给出「白送 vs 有捷径」两档对照**：`mix=0`（语义块只含结构）时分离是**白送**的，
     **负对照 w_inv=0 也会 PASS** ⇒ 只看这一档会得出「显式监督没用」的错误结论。
     引入 `mix=0.6`（细节块里掺结构 = 捷径）之后判据才有分辨力。
- ✅ **实测（合成，纯 CPU，~78s）**：尺子 oracle 依赖 **0.0000 PASS** / leaky **0.592 FAIL**；
  `mix=0` 负对照也 PASS（**证明装置无假阴性**）；`mix=0.6` 负对照 **0.5340 FAIL**
  vs 有监督 **0.0654 PASS** ⇒ **显式监督净收益 8.17×**。
- ⚠️ **仍是合成数据**：它证明的是「装置正确 + 监督有效」，**不等于真图能分离**；真图版需接真实 VAE 编码器。

## 🔴 本机环境新事实（2026-10-03 白天补测）
- 🔴 **HF 直连不可用（不是网的问题，是本机 TLS 凭证栈）**：
  `curl` → `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS (0x8009030e)`；
  PowerShell `Invoke-WebRequest` → `The SSL connection could not be established`（0.1s 即失败）。
  **只有 Python/certifi 栈 + 代理 7897 能通**（测速 1.4 MB/s；实际拉模型约 **0.6 MB/s**）。
  ⇒ **下 HF 必须 `kp_hf` / 走代理** —— 本机上「直连更快」是假的。
- ⚠️ **`hf_xet` 未装** → 每个文件都刷 `Xet Storage is enabled ... Falling back to regular HTTP download`。
  HF 官方称 xet 对大文件显著更快 ⇒ **建议先装 `hf_xet` 再拉大模型**（**待实测，先别当结论**）。
- 🔴 **`TRITON_CACHE_DIR` 未设** → Triton 缓存落到 `C:\Users\TX\.triton`，权限/沙箱下报
  `PermissionError: [WinError 5]` ⇒ **ModelOpt NVFP4 端到端量化（`bench_115w.py` Phase D）跑不完**。
  修法：`env.sh` 增加 `export TRITON_CACHE_DIR="$KP_ROOT/.cache/triton"`。
- 🔴 **`tempfile.gettempdir()` 实测返回「工作区根目录」** ⇒ 临时产物会掉进仓库根、污染 `git status`
  （本机曾出现 `_kp_card_selftest.json`）。**规则：临时文件一律落 `KP_OUT` 并清理**，
  禁止用 `tempfile.gettempdir()`（已改掉 `selftest` 第 7 节那处）。
- ⚠️ **中文控制台下 `python -m kp.selftest` 会因 emoji 报 GBK `UnicodeEncodeError` 而 exit 1**
  （看起来像自检挂了）。已在 `selftest.py` 入口 `reconfigure(encoding="utf-8")` 兜住。

### 已有、可直接复用
- ✅ **MSVC 14.44.35207**（`C:\Program Files\Microsoft Visual Studio\2022\Community\...`）+ **Windows SDK 10.0.26100.0**。
  ⚠️ `tools/e5b_msvc_env.bat` 默认路径找的是 `2022\BuildTools\...` 与 `D:\vc2022` / `D:\vs` ⇒ **本机三处都不匹配**，需设 `MSVC_ROOT` 指向 Community 版。
- ✅ 代理 `127.0.0.1:7897` 连通（github 200 / 6.6s）；PyPI 清华镜像 200 / 3s。
- ✅ pip 26.1.2 可用（WorkBuddy 托管 python 3.13）。

### 待装清单（按「项目代码实际 import」+ onboard 判定）
- **必需**：`torch`(cu128, 需含 sm_120)、`numpy`、`pillow`
- **可选但代码已用到**：`transformers`(⚠️ 钉 4.55.4，**5.10.1 会打坏 modelopt 插件**)、`diffusers`、`tokenizers`、`huggingface_hub`、`torchvision`、`nvidia-modelopt`、`scipy`、`scikit-image`、`pytorch-fid`、`fontTools`、`uharfbuzz`、`freetype-py`、`rembg`、`triton-windows`、`peft`
- ⚠️ **`requirements.lock.txt` 只有 4 行**（diffusers / hf_hub / tokenizers / transformers）——那是旧机 **venv 覆盖层**，其余靠「系统 Python 3.12.10 的 site-packages」。**本机照它装不够**。

---

## 🔴🔴 venv 必须放在【工作区外】（2026-10-04 事故 + 定案）

**症状**：`D:\kokonapolaris-s4\.venv\Scripts\python.exe` 一律报
`Cannot read 'D:\kokonapolaris-s4\.venv\pyvenv.cfg'`（此前一天还完全正常）。

**真因**：**工作区根 `D:\kokonapolaris-s4` 被加了硬化的 ACL**，并**向下继承**：
```
Everyone:(CI)(DENY)(DC)
LAPTOP-...\TX:(OI)(CI)(WO)                      ← 只有「写所有者」
S-1-4-635281079-250895385:(OI)(CI)(W,D,DC)      ← 疑似沙箱 AppContainer 能力 SID
```
⇒ **从工作区内启动的可执行文件（python.exe）读不了工作区内的文件**。
✅ **对照组**：同字节的 venv 复制到工作区外（`D:\kp-venv`）**立刻正常**；工作区内新建 venv 同样失败。

**已排除的假设**（都做过实验）：沙箱运行时限制 ❌（关沙箱照样失败）· 文件损坏 ❌（`head`/基础 python 都能读）·
权限不足 ❌（`icacls /grant TX:(R)` 成功仍失败）· venv 内容坏 ❌（工作区外同一份正常）。

**定案**：
1. **venv 位置 = `<仓库父目录>/kp-venv`** ⇒ 本机是 `D:\kp-venv`
2. `env.sh` 已改为**优先探测 `<仓库父目录>/kp-venv`**，回退工作区内 `.venv`；支持 `KP_VENV` 覆盖
3. 坏掉的旧 venv **改名留档** `.venv.broken-20261004`（未删，~4GB）
4. `.gitignore` 加 `/.venv*/` 与 `/kp-venv/`
5. ⚠️ 该硬化 ACL 覆盖**整个工作区根**（`.git` / `kp` / `out` 全一样），但 **git / grep / head 等普通读写不受影响**，
   **只有「从工作区内启动的 python.exe」会撞上** —— 排查时容易误判成权限问题。

⭐ **教训**：`Cannot read 'pyvenv.cfg'` 这种报错**看起来像文件/权限问题，实际是位置问题**。
**做对照实验（工作区外 vs 内）比查文档快得多** —— 又一次「工具报错文本 ≠ 根因」。

## 📦 本机 tokenizer（2026-10-04 补齐）
`models/Qwen3-4B-tokenizer/` 需 **4 个文件**（`kp/text/tokenizer.py` 点名）：
`tokenizer.json`(11.4MB **真词表**) · `tokenizer_config.json` · `vocab.json` · `merges.txt`。
⚠️ **只放 `tokenizer_config.json` 不够**（那是配置，HF 缓存里常年只有它 ⇒ 报 `JSONDecodeError`）。
**镜像可达性实测**：`hf-mirror.com` **307 可用**；`huggingface.co` **直连失败**。取法：
```bash
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  curl -sL --retry 2 -o "models/Qwen3-4B-tokenizer/$f" \
      "https://hf-mirror.com/Qwen/Qwen3-4B/resolve/main/$f"; done
```

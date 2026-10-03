# KokonaPolaris-S4 · 迁移手册（5050 → 5060）

> 面向两方：**你**（建云端仓库 + 在新机拉取）与 **我**（在新环境继续推进）。
> 目标：换机后 **1 小时内恢复工作**，且**不丢任何结论、不改任何架构常量**。

---

## 一句话结论

**代码全带走（5.5MB），模型别带走（4.6GB 按需重下），出图产物不带走（163MB 但结论已入库）。**
5060 Laptop 与 5050 Laptop **同架构（sm_120）、同显存（8GB）**，只有 SM 数 20→26 算力 +30% ⇒
**`kp/config.py` 里的架构常量一个都不用改**。

---

## 一、你会推的东西（我这边已就绪）

| 类别 | 内容 | 体积 | 必需性 |
|---|---|---|---|
| 设计稿 | `design/`（md + html，主文档 v1.14 + 补充 01–11） | ~0.8 MB | ✅ **项目的真相** |
| 源码 | `kp/`（17 模块）+ `tools/`（探针脚本） | ~0.8 MB | ✅ |
| 记忆库 | `.workbuddy/memory/`（MEMORY / archive / ops + 当日日志） | ~0.3 MB | ✅ **别漏这个** |
| 小数据 | `data/characters/kokona/`（角色卡素材 3.7MB + manifest） | 3.7 MB | ✅ |
| 配置 | `requirements.lock.txt` / `env.sh` / `.gitignore` / `.gitattributes` | 小 | ✅ |
| **模型权重** | `models/Sana_1600M_…`（11GB 目录，实际精选 4.6GB） | **4.6 GB** | ❌ **不入库**，新环境跑 `tools/fetch_sana.py` 重下 |
| **出图产物** | `out/`（e5/e5b/e4b/e6 图片与 embeds） | **163 MB** | ❌ **不入库**，但**结论数字已写进设计稿与记忆库** |
| 环境 | `.venv/` / `.cache/` / `repos/` | — | ❌ 不入库，新环境重建 |

**仓库总量 ≈ 5.5 MB**（`.git` 4.8 MB）。推到 GitHub / Gitee / 任意 Git 服务都毫无压力。

> ⚠️ **`out/` 不入库但结论不丢**：所有实测数字都已落进
> `.workbuddy/memory/MEMORY_archive.md` 与设计稿补充 09（量化）。
> 想留原始 json 的话，跑 `git add -f out/**/*.json`（30 个文件共 75KB，很便宜）。

---

## 二、建云端仓库（你这边，3 分钟）

```bash
# 1. 在 GitHub/Gitee 上新建一个**空**仓库，名字必须是 KokonaPolaris-S4
#    （不要勾 README/.gitignore/LICENSE，否则首次 push 会冲突）

# 2. 本机加第二个远端（保留本地裸镜像做兜底）
cd /d/model
git remote add cloud <你的仓库 URL>
git push -u cloud main

# 3. 验证
git remote -v          # 应看到 origin(本地裸镜像) + cloud(云端)
git push cloud --dry-run   # dry-run 通过就算成功
```

**推荐平台**：国内直连用 **Gitee**（免代理、克隆快）；需要 Actions/免费算力用 **GitHub**（需代理）。

---

## 三、新机拉取（3 分钟）

```bash
git clone <你的仓库 URL> kp
cd kp

# 环境体检（回答：路径对吗？依赖齐吗？自检过吗？缺什么？）
python tools/onboard.py
```

`onboard.py` 会依次告诉你：
1. **路径解析**（自动探测项目根，不依赖任何绝对路径）
2. **依赖检查**（必需/可选分开列）
3. **骨架自检 52 项**（纯 CPU，1 分钟内跑完）
4. **缺口清单**（该补什么、怎么补）

---

## 四、四个环境变量（可选但推荐）

`kp/paths.py` 的探测优先级：**`KP_ROOT` 环境变量 > 按 `__file__` 向上定位 > 从 CWD 向上找**。
所以默认**零配置**即可用。若想放到别处：

| 变量 | 作用 | 默认 |
|---|---|---|
| `KP_ROOT` | 项目根 | 自动探测 |
| `HF_HOME` | HuggingFace 缓存 | `$KP_ROOT/.cache/huggingface` |
| `TORCH_HOME` | torch 缓存 | `$KP_ROOT/.cache/torch` |
| `http_proxy` / `https_proxy` | HF 下载加速 | 关（走镜像） |

Windows 用 Git Bash 时 `source env.sh` 即可；Linux/macOS 需把 `Scripts/python.exe` 换成 `bin/python`：

```bash
sed -i 's#/Scripts/python#/bin/python#g' env.sh   # 或按 tools/onboard.py 提示手改
```

---

## 五、GPU 差异：5060 vs 5050（结论：不用改配置）

| 项 | 5050 Laptop（本机） | **5060 Laptop（新机）** | 影响 |
|---|---|---|---|
| 架构 | Blackwell sm_120 | **Blackwell sm_120** | ✅ 相同，NVFP4 路线**不受影响** |
| 显存 | 8 GB GDDR7 | **8 GB GDDR7** | ✅ 相同，「~2.5GB 推理 / 4–5GB 训练」目标**照旧** |
| SM 数 | 20 | **26** | ⭐ **+30% 算力**（更宽松，不是更紧） |
| TGP | 45–100 W | **45–115 W** | ⭐ 上限 +15W，更能跑到高功耗档 |
| 显存带宽 | 384 GB/s | **384 GB/s** | ✅ 相同 |
| CUDA Core | 2560 | **3328** | ⭐ +30% |

**⇒ 对 KP 的三条影响：**

1. **架构常量零改动** —— `kp/config.py` 里 `DIT_S/DIT_M/LATENT/QUANT` 全部不动。
   ⚠️ 唯一仍待你拍板的老问题依旧：**KP-M 实测 1.806B vs 标称 1.5B（+20.4%）**。
2. **G1 正式 FID 更可行** —— 那 50 张/臂的正式 FID 之前要等白天解锁 115W 才能跑；
   新机 5060 的 TGP 上限也是 115W，**但白天窗口一样要看厂商功耗设置**，晚上仍是 ~40W 静音档。
   ⇒ **夜间禁压测铁律在新机同样适用**（安静模式会把 GPU 压到约 40W）。
3. **算力预算类结论需重标** —— 记忆库里那些「实测 120.5 TFLOPS 已逼近 20 SM 理论峰值 114.6」的
   结论是**绑在 5050 上的**。新机首次跑 GPU 任务时，先用 `tools/gpu_probe.py` 重测一遍并更新记忆库，
   **别把旧机的 FLOPS 当作 KP 的永久事实**。

---

## 六、换机后第一件事（建议顺序）

```bash
python tools/onboard.py          # ① 体检（上面已说）
python -m kp.selftest            # ② 自检 52 项（应全绿）
python tools/publish_check.py    # ③ 再确认仓库干净
python tools/gpu_probe.py        # ④ 重测新机 GPU 基线 → 更新记忆库
# ⑤ 只有要跑 G1 量化实验时才需要：
source env.sh && "$KP_PY" tools/fetch_sana.py     # 4.6GB，可断点续传
```

**日常写设计/跑自检不需要下 Sana** —— 自检 52 项全是纯 CPU。

---

## 七、我在新环境看到的第一样东西

`.workbuddy/memory/MEMORY.md`（决策 + 铁律）、`MEMORY_archive.md`（实测数字）、
`MEMORY_ops.md`（本机环境细则）、以及 `2026-10-03.md` 当日日志末尾的「**下一步**」。
那几个「下一步」就是续跑的起点 —— 与自动化任务「先判断有没有中断未完成的活」是同一套逻辑。

---

## 八、常见坑（都已踩过并修好，附位置）

| 坑 | 后果 | 修法 | 位置 |
|---|---|---|---|
| 硬编码 `D:/model` | 换机后静默指错位置 | 全部收敛到 `kp.paths` | `kp/paths.py` + `tools/portable_paths.py` |
| `.gitignore` 写行尾注释 | 模式全失效，误暂存 3770 文件 | 注释必须**整行独立** | `.gitignore` 头部 |
| `D:/model/.venv` 解释器路径 | 新机不存在 | `env.sh` 派生 `$KP_PY` | `env.sh` |
| Windows↔Linux 行尾 | 整文件假 diff | `.gitattributes` `text=auto eol=lf` | `.gitattributes` |
| 依赖只装在 `.venv`（system-site-packages） | 新机缺 torch | `tools/onboard.py --install` | `tools/onboard.py` |
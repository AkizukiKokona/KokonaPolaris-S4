# artifacts/ · 基线产物（不可复现，故入库）

> **为什么有这个目录**：项目里绝大部分数据是**可从上游重新下载的**（模型权重）或
> **可由代码重新生成的**（临时输出）—— 那些不入库。
> 但有一类东西**既下不来、也跑不出来** ⇒ **实验结论的原始数据**。它们才是本目录的_contents。

## 判定标准（三条都满足才入库）

| # | 条件 | 说明 |
|---|---|---|
| ① | **不可重新生成** | 跑一遍要几小时，且**结果会随环境漂移** |
| ② | **承载已得结论** | 后续判据要拿它当**基线**比对 |
| ③ | **体积可控** | 单文件 < 20MB，目录 < 100MB |

⛔ **不满足 ⇒ 不入库**：
- **模型权重**（`models/`，> 250MB）⇒ **可从 ModelScope/HF 重下**（实测 6.14 MB/s）
  ⇒ 用 `kp/data/fetch.py` 与 `kp/probe/m3_attribution.py` 记录的方式复现即可
- **中间输出**（`out/` 其余部分）⇒ 可由代码重跑
- **增广图**（`out/aug/`）⇒ 可由 `python -m kp.data.augment` 重生成（确定性，同 seed 同结果）

---

## 📄 实验结论数据（**最重要，不可复现**）

| 文件 | 体积 | 承载什么结论 |
|---|---|---|
| `m3_attribution.json` | 2KB | 🔴 **M3 归因的最终结论**：随机 encoder **0.2876** → 训练后 **0.9997**；DC-AE（训练充分）**0.4375** ⇒ 判决「**训练把冗余做满**」⇒ 修法②升为首选 |
| `real_g2_full.json` | 40KB | 真图全量 G2（384px/5 臂）**CPU 版**结果：`verdict PASS`、监督净收益 **4.45×** |
| `real_g2_full_gpu.json` | 40KB | 同上 **GPU 版**（568s vs CPU 651s ⇒ 只快 13%，瓶颈在训练不在编码） |
| `typography_chain.json` | 15KB | 排版链路闭环的判据结果 |

⚠️ **这三份是本项目所有「已得结论」的原始凭证** ——
没有它们，那些数字**无法复核**（重跑会因环境/随机性得到不同值）。

---

## 💾 checkpoint（可作基线，但体积较小才留）

| 文件 | 体积 | 用途 |
|---|---|---|
| `vae_trained_11img.pt` | 17MB | 🔴 **M3 归因的关键**：`cross_r2 = 0.9997`（**训练后**的 encoder）⇒ 没有它就证不了"训练做满冗余" |
| `vae_trained_176aug.pt` | 17MB | 同上，增广数据版（同样 0.9997 ⇒ 排除"是数据太少"的可能） |
| `fitter_kokona.pt` | 8MB | 角色卡 Fitter 基线（换角色 0.064s / 换角色 token Δ=0.0103 的出处） |

⚠️ **只留了 2 个 VAE checkpoint**（其余 `out/vae/*.pt` 是同一次实验的中间产物，删了不影响任何结论）。

---

## ⚙️ `dc_ae_config/`

`config.json`（1.1KB）—— DC-AE f32c32 的配置（`latent_channels = 32` 已从这里确认）。

⛔ **权重不在这里**（1.25GB，**可重下**）：
```bash
# ModelScope 直连 ~6 MB/s；用 GET+stream（它不支持 HEAD）
# 见 kp/probe/m3_attribution.py::load_dcae 的加载姿势
```
⭐ **加载必须用** `from_single_file(ckpt, config=<此目录>, local_files_only=True)`
—— 用 `from_pretrained` 会**键名重叠 0**（ckpt 是 Sana 官方 `stages/op_list` 命名）⇒ 全 meta ⇒ 崩。

---

## 🔁 怎么重建这些产物

```bash
# M3 归因（~3 分钟）
python -m kp.probe.m3_attribution_fix --size 256 --ckpts out/vae/*.pt

# 真图全量 G2（CPU 651s / GPU 568s）
python tools/g2_real_vae.py --size 384 --steps 200

# 两个 VAE checkpoint（各约 4 分钟）
python -m kp.train.vae_pretrain --image-dir data/characters/kokona --size 128 --steps 200 \
    --out out/vae/overfit11.pt
python -m kp.train.vae_pretrain --image-dir out/aug/kokona_x16 --size 128 --steps 200 \
    --out out/vae/with_eval.pt
```
⚠️ **重跑得到的数字可能与本目录的略有差异**（浮点累积顺序 + 环境差异）⇒
**本目录是"当时那次的凭证"，不是"标准答案"**。
自检 §34 的判据用**相关系数**而非逐位相等，正是为了容忍这一点。

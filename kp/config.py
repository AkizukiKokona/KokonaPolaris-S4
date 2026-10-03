"""KP 架构常量（单一真源）。

⚠️ 这些数字全部来自设计文档，改动即架构改动，必须同步更新 design/。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 混合 Latent（补充：32× 空间压缩）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatentCfg:
    spatial: int = 32          # 32× 空间压缩 → 1024² 图 → 32×32 格
    semantic_ch: int = 8       # 语义通道（对齐 DINOv3）
    detail_ch: int = 32        # 细节通道（DC-AE 式）
    @property
    def total_ch(self) -> int:
        return self.semantic_ch + self.detail_ch          # = 40

    def tokens(self, image_size: int) -> int:
        """图像边长 → token 数（patch_size=1，1 token = 1 latent 格）。"""
        side = image_size // self.spatial
        return side * side


LATENT = LatentCfg()


# ---------------------------------------------------------------------------
# DiT 主干
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiTCfg:
    dim: int = 1152
    layers: int = 24
    heads: int = 16
    mlp_ratio: float = 2.5
    # 浅层前 N 个 block 保留 double-stream
    double_stream_blocks: int = 5
    # 混合注意力比例：每 4 层里 3 层 gated-linear、1 层 Sigmoid/Softpick
    ratio_gated_linear: int = 3
    ratio_sigmoid: int = 1
    qk_norm: bool = True
    # 分辨率 Matryoshka：早期步在低 token 上跑
    matryoshka_tokens: tuple = (256, 1024)
    # 注入点必须在量化器之外
    quantize_injection_point: bool = False
    # 🔴 M3 修法②（2026-10-03）：`patch_embed` 拆成「语义 8ch 走一路 + 细节 32ch 走一路」，
    #    两路**权重不共享** ⇒ 结构上阻断跨块线性重建（M3「专属区」的结构保证）。
    #    ⭐ 参数量恒等：(8+32)·d ≡ 40·d，与单路 40→d **完全相同** ⇒ 零代价。
    #    ⛔ **默认关闭**：它改变架构行为，需在 P2 换主干时显式拍板；此处先让能力可测。
    #       依据：`kp/probe/m3_fix.py` §修法②（构造级 0.9666→0.0017）、
    #             `kp/probe/m3_fix_real.py`（真图 0.2632→0.0054，重建 L1 未测）。
    split_patch_embed: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def param_count(self) -> int:
        """粗略参数量（单流主干估算，用于档案核对）。"""
        d, L = self.dim, self.layers
        per_block = 4 * d * d + 2 * d * int(d * self.mlp_ratio) + 4 * d
        return L * per_block + 2 * d * LATENT.total_ch


# KP-S（主力 0.6B）/ KP-M（质量档 1.5B）
DIT_S = DiTCfg(dim=1152, layers=24, heads=16)
DIT_M = DiTCfg(dim=1792, layers=32, heads=16)


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantCfg:
    weight_bits: int = 4
    weight_fmt: str = "E2M1"        # 16 级
    act_fmt: str = "E4M3"           # 设计档 W4A8（256 级）；官方默认是 W4A4=FP4 激活
    act_bits: int = 8
    block_size: int = 16
    scale_fmt: str = "E4M3"
    # 官方 NVFP4_DEFAULT_CFG 里 proj_out 默认不量化
    quantize_proj_out: bool = False
    sm_arch: int = 120


QUANT = QuantCfg()


# ---------------------------------------------------------------------------
# 能力总线
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilityCfg:
    gate_init: float = 0.0                   # ★ 门控初始 0 ⇒ 接口成本严格为零
    gate_dtype: str = "float32"              # 门控全程 fp32（不参与量化）
    identity_tokens: int = 256               # 身份 token 预算（走 cross-attention）
    identity_token_dim: int = 1024
    identity_via_cross_attention: bool = True   # 禁止拼主序列
    delta_pack_bytes: int = 27 * 1024 * 1024    # 27MB
    # 谱检查：与 W0 全奇异向量 cos < 阈值 的高排名分量判不合格
    spectral_cos_threshold: float = 0.1
    spectral_rank_ratio: float = 0.5          # 只看前 50% 排名
    erase_default_rank: int = 6               # 安全子空间有效秩 k≈6


CAP = CapabilityCfg()


# ---------------------------------------------------------------------------
# Caption / 文本塔语言配比（路线 P1.8 —— 「中文自然语言支持」）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaptionCfg:
    """⭐ 这是一组**旋钮**，不是常量。

    为什么单独成节：文本塔是 ~220M **LLM**（自 Qwen3-4B 蒸馏），中文是**原生能力**；
    但「中文够不够好」完全由**训练数据里中文 caption 的占比与形态**决定，
    而该占比在**预训练前定稿后不可逆**（换主干/重跑预训练才有第二次机会）。

    设计律（补充 11 §4.8）：
      · 以**自然语言句子**为主，**禁止退回纯英文标签串**（tag soup）；
      · 中文为主 + 一定比例英文对齐（避免英文能力塌陷、保跨语言泛化）；
      · `tag` 的作用从「用户写的输入咒语」退到「**内部条件轴的训练标签**」。
    """
    # ---- 语言配比（按 caption 条数计）----
    zh_share_target: float = 0.80
    zh_share_band: tuple = (0.70, 0.90)
    en_share_band: tuple = (0.08, 0.22)
    ja_share_max: float = 0.15          # 日系画风需要日文 caption，但不宜反客为主
    # ---- 形态约束 ----
    allow_tag_soup: bool = False
    tag_soup_threshold: float = 0.60
    min_chars: int = 8
    max_chars: int = 300
    require_sentence_punct: bool = False   # 弱证据：不强制，但计入报告
    # ---- 语料审计告警线 ----
    max_violation_rate: float = 0.05    # 违规（标签串/过短/过长）占比上限


CAPTION = CaptionCfg()


# ---------------------------------------------------------------------------
# Axis Probe（验证门 G3.5 —— L1 条件轴真实性的四测阈值）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AxisCfg:
    """⭐ L1「主干内生条件轴」承担 ~90% **调用量**，却有一个致命的验证缺口：
    轴**存在吗、可靠吗**，只能靠出图**间接看**；出错**只能重训主干**。
    这是全案第三个隐藏关键假设 —— 本组阈值就是它的**测量判据**。

    设计律：**未通过任何一测的轴，默认归 L3**（降级为子空间约束 LoRA）。
    """
    n_axes: int = 16                 # 已标定控制面板的维数（逐维命名）
    sweep_steps: int = 9             # 单轴扫描步数（奇数 ⇒ 有中心点 v=0）
    n_random: int = 256              # 辅助指标（线性可辨识）的随机采样数
    # ---- 四测通过线 ----
    mono_threshold: float = 0.90     # ① 单调性：Spearman ρ
    ortho_threshold: float = 0.30    # ② 正交性：跨轴最大 |cos|（串扰）
    rev_threshold: float = 0.90      # ③ 可逆性：奇对称度（1 − 最大不对称误差）
    travel_threshold: float = 0.80   # ④ 低比特行程：NVFP4 后仍可区分的比例
    # ---- 辅助诊断（不设门线，只报数）----
    ident_r2_report: float = 0.90    # 线性读出 R² 的参考线


AXIS = AxisCfg()


# ---------------------------------------------------------------------------
# 超参
# ---------------------------------------------------------------------------
# ⚠️ 预留旋钮 —— **当前是空壳**（详见 `RuntimeCfg` 的 docstring）
@dataclass
class RuntimeCfg:
    """⭐ ⚠️ **预留（当前空壳）** —— 这不是一组**生效中**的旋钮。

    审计（`out/audit_stale_and_dead.md` §4.1）查明：`RuntimeCfg` / `RUNTIME`
    **没有任何字段被 `kp/` 消费** —— 全库只出现在 `kp/__init__.py:24,28` 的 re-export 上，
    `dtype` / `device` / `seed` / `vram_*` / `tags` 都没有读者。

    ⇒ **改这里不会改变任何行为**。`ONBOARDING.md` §3 的文件地图曾把它和
       QUANT/CAP/CAPTION/AXIS/LATENT/DIT 并列为「可调旋钮」，现已改标为「预留（空壳）」。

    ⛔ **不要因为零消费就删掉**：删不删是**架构决策**（要不要引入统一运行时配置层），
       需用户拍板，不在死代码清理的授权范围内。
    ✅ 将来要接线时，两个字段组的语义已经写在这里：运行时 dtype/设备/种子走 `RuntimeCfg`，
       **架构**（层数/头数/通道/门线）一律留在 `DiTCfg` / `LatentCfg` / `QuantCfg` 等
       frozen 架构常量里 —— 后者是「改动即架构改动」，不该被运行期参数覆盖。
    """
    dtype: str = "bfloat16"
    device: str = "cpu"                       # 骨架默认 CPU（夜间安全）
    seed: int = 0
    # 目标显存
    vram_target_infer_gb: float = 2.5
    vram_target_lora_gb: float = 4.5
    tags: dict = field(default_factory=lambda: {"stage": "skeleton"})


RUNTIME = RuntimeCfg()

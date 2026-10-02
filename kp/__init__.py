"""KokonaPolaris-S4（心夏北极星 / KP）· 参考实现骨架

设计稿见 D:/model/design/KokonaPolaris_*.md（主文档 v1.13 / 补充 01–12）。

分层：
    kp.latent      混合 latent（32× 压缩，8ch 语义 + 32ch 细节 = 40ch）
    kp.models      单流 DiT 主干（3:1 混合注意力 / QK-Norm / Rectified Flow）
    kp.capability  能力总线（∥-Pack / Δ-Pack / 擦除算子 E / 门控 bit-exact）
    kp.character   角色卡数据对象 + Character Fitter

硬约束（不可动摇）：
    1. NVFP4 原生（SM120 兼容）
    2. 低显存优先（显存第一因是 token 数）
    3. 弃用 UNet
    4. 能力包全关时必须与裸模型 bit-exact
"""
from . import config  # noqa: F401

__version__ = "0.1.0-skeleton"
__all__ = ["config"]

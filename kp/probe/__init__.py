"""kp.probe —— 验证门的**测量装置**（不是模型，是尺子）。

三件：
  · `AxisProbe`（`kp/probe/axis.py`）—— G3.5「L1 条件轴真实性」的四测装置；
  · `Synthetic`（`kp/probe/synthetic.py`）—— **已知答案的对照样本**：
    测量装置必须先能在已知答案的样本上给出正确答案，否则它在真模型上
    给出的数字没有意义；
  · `RealAxisProbe`（`kp/probe/real.py`）—— **真探针**：responder 接在
    **真主干 + 真轴注入路径**（`domain → domain_embed → adaLN → blocks`）上，
    ④ 低比特行程走**真量化回路**（`kp/quant/nvfp4.py` 挂在主干 `GatedLinear` 上）；
  · `kp/probe/attn.py` —— **G3 Sigmoid 注意力**的机制层装置（长提示收益 / 量化友好性）。
    ⚠️ 只验**算子层**（随机权重），**不能**替代 G6 后的 benchmark 级验收。
"""
from .axis import (  # noqa: F401
    AxisResult,
    AxisProbeReport,
    AxisProbe,
    axis_report_text,
)
from .synthetic import Synthetic  # noqa: F401
from .attn import (  # noqa: F401
    G3_GAPS,
    AttentionWeights,
    DilutionPoint,
    DilutionResult,
    attention_weights,
    compare_dilution,
    contrast_report,
    contrast_vs_offset,
    dilution_curve,
    dilution_exponent,
    dilution_share,
    g3_report_text,
    plan_composition,
    qknorm_logit_bound,
)
from .real import (  # noqa: F401
    DOMAIN_AXES,
    DOMAIN_AXIS_NAMES,
    RealAxisProbe,
    RealProbeReport,
    BackboneDomainResponder,
    build_test_backbone,
    corrupt_domain_embed,
    domain_response_dev,
    gate_ranges,
    latent_readout,
    noise_floor,
    open_domain_door,
    quantized_backbone,
    real_axis_report_text,
    run_real_probe,
)

__all__ = ["AxisResult", "AxisProbeReport", "AxisProbe", "axis_report_text",
           "Synthetic",
           "DOMAIN_AXES", "DOMAIN_AXIS_NAMES", "RealAxisProbe", "RealProbeReport",
           "BackboneDomainResponder", "build_test_backbone", "corrupt_domain_embed",
           "domain_response_dev", "gate_ranges", "latent_readout", "noise_floor",
           "open_domain_door", "quantized_backbone", "real_axis_report_text",
           "run_real_probe"]

"""kp.probe —— 验证门的**测量装置**（不是模型，是尺子）。

现在只有一件：`AxisProbe` —— G3.5「L1 条件轴真实性」的四测。

配套 `kp.probe.synthetic` 提供**对照样本**：测量装置必须先能在已知答案的
样本上给出正确答案，否则它在真模型上给出的数字没有意义。
"""
from .axis import (  # noqa: F401
    AxisResult,
    AxisProbeReport,
    AxisProbe,
    axis_report_text,
)
from .synthetic import Synthetic  # noqa: F401

__all__ = ["AxisResult", "AxisProbeReport", "AxisProbe", "axis_report_text",
           "Synthetic"]

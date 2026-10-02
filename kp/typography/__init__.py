"""kp.typography —— 文字渲染链路（低压缩 ROI 分支 + Layout Planner）。

    Layout Planner（确定性、无权重、今天可用）
    TypographyPack（ROI 分支的模型侧骨架，详见 typography_pack.py）
"""
from .layout import (  # noqa: F401
    TextBlock,
    LayoutSpec,
    plan,
    validate,
    wrap,
    measure,
    char_width,
    roi_boxes,
    NO_LINE_START,
    NO_LINE_END,
)
from .typography_pack import ROIBranch  # noqa: F401

__all__ = ["TextBlock", "LayoutSpec", "plan", "validate", "wrap", "measure",
           "char_width", "roi_boxes", "NO_LINE_START", "NO_LINE_END", "ROIBranch"]

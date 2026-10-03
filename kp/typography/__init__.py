"""kp.typography —— 文字渲染链路（低压缩 ROI 分支 + Layout Planner）。

    Layout Planner（确定性、无权重、今天可用）
    TypographyPack（ROI 分支的模型侧骨架，详见 typography_pack.py）
    Composite（**拼回 latent**，2026-10-03 落地 ⇒ 三件套闭环）

⭐ 为什么要单独一块「拼回」：32× 压缩下 40px 汉字只占 1.25 个 latent 格，
   所以文字必须走**独立的低压缩 ROI 分支**；但 ROI 出图后要**贴回**主干 latent，
   这一步的判据（框外必须 bit-exact、框内必须逐位等于独立复算）是纯 CPU 可验的。
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
from .composite import (  # noqa: F401
    CellWindow,
    CompositeReport,
    DegenerateLayoutError,
    ROIConflictError,
    TypographyChainError,
    composite_latent,
    default_spec,
    roi_windows,
    run_acceptance,
    run_chain,
    verify_composite,
    windows_from_layout,
)

__all__ = ["TextBlock", "LayoutSpec", "plan", "validate", "wrap", "measure",
           "char_width", "roi_boxes", "NO_LINE_START", "NO_LINE_END", "ROIBranch",
           # 排版链路闭环（plan → ROIBranch → composite）
           "CellWindow", "CompositeReport", "TypographyChainError",
           "DegenerateLayoutError", "ROIConflictError",
           "composite_latent", "verify_composite", "roi_windows",
           "windows_from_layout", "run_chain", "run_acceptance", "default_spec"]

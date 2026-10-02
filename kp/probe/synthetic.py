"""合成 responder —— **Axis Probe 的对照样本**（测试夹具，不是模型）。

存在的理由：一个**测量装置**必须先能在**已知答案**的样本上给出正确答案，
否则它在真模型上给出的数字没有意义。这里造 4 个可控的「轴系统」：

| 名字 | 构造 | 应当触发哪一测失败 |
|---|---|---|
| `clean` | 正交基 × 线性，幅度充足 | —— （四测全过） |
| `coupled` | 两轴之间做旋转混合 | ② 正交性 |
| `suppressed` | 幅度被压到极弱 | ④ 低比特行程（**其余三测全过**） |
| `nonlinear` | 沿该轴整体叠加 v² 项 | ① 单调性 |

⭐ `suppressed` 是这组样本里**最有价值**的一个：它证明「低比特行程」这一测
**不可能**被前三条替代 —— 轴依然单调、依然可辨识、也不串扰，
**只有在量化回路里才看得见它已经废了**。
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

Responder = Callable[[torch.Tensor], torch.Tensor]


def _basis(n_axes: int, dim: int, seed: int = 0) -> torch.Tensor:
    """(dim, n_axes) 列正交基。"""
    g = torch.Generator().manual_seed(seed)
    A, _ = torch.linalg.qr(torch.randn(dim, n_axes, generator=g))
    return A


class Synthetic:
    """一组可控的合成轴系统。"""

    def __init__(self, n_axes: int = 6, dim: int = 256, scale: float = 8.0,
                 seed: int = 0):
        self.n_axes, self.dim, self.scale = n_axes, dim, scale
        self.A = _basis(n_axes, dim, seed)

    # ---- 四类样本 ----
    def clean(self) -> Responder:
        A, s = self.A, self.scale
        return lambda V: (V @ A.T) * s

    def coupled(self, i: int = 0, j: int = 1, k: float = 0.9) -> Responder:
        A, s, n = self.A, self.scale, self.n_axes
        M = torch.eye(n)
        M[i, j] = k
        M[j, i] = k
        return lambda V: (V @ M.T @ A.T) * s

    def suppressed(self, factor: float = 0.002) -> Responder:
        """幅度被压得远低于 FP4 的台阶 ⇒ 相邻档位量化后落在同一格。"""
        A, s = self.A, self.scale
        return lambda V: (V @ A.T) * s * factor

    def nonlinear(self, i: int = 0, quad: float = 6.0) -> Responder:
        """沿轴 i 的整体方向叠加二次项 ⇒ 响应呈 V 形，不再单调。

        ⚠️ 关键：二次项必须沿**整个方向 A[:,i]** 叠加，不能只改某一个分量
        —— 只改 1/dim 个分量时，投影被其余维度的线性项淹没，
        测出来仍"单调"，这不是装置的问题而是样本没造对。
        """
        A, s, n = self.A, self.scale, self.n_axes
        dir_i = A[:, i].clone()

        def f(V):
            Y = (V @ A.T) * s
            Y = Y + torch.outer(V[:, i] ** 2, dir_i) * (quad * s)
            return Y
        return f

    # ---- 便捷：一次性拿到全部 ----
    def all(self) -> Dict[str, Responder]:
        return {"clean": self.clean(), "coupled": self.coupled(),
                "suppressed": self.suppressed(), "nonlinear": self.nonlinear()}

    def expected_failures(self) -> Dict[str, Tuple[str, ...]]:
        # nonlinear：V 形响应既破坏单调，也让「线性读出」失效 ⇒ 同时踩两测
        return {"clean": (), "coupled": ("正交",),
                "suppressed": ("低比特行程",), "nonlinear": ("单调", "可逆")}


__all__ = ["Synthetic", "Responder"]

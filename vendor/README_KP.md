# vendor/ —— 借来的零件（不是我们的代码）

> 借零件不借整体（PROJECT_RULES 第九条）。

## sd-scripts（kohya-ss）

- 来源：https://github.com/kohya-ss/sd-scripts
- 许可：**Apache-2.0**（2026-09-24 核实）
- 借的用途：**训练循环 + 单流 block 参考**
- ⛔ **不借它的权重**

### 实际读了哪段（不黑箱调用）
`library/flux_models.py:762` `class SingleStreamBlock` —— 逐行读完，理解三处：
1. `linear1: hidden → 3*hidden + mlp_hidden`（⭐ **QKV 与 MLP 融合成一个矩阵**）
2. `linear2: hidden + mlp_hidden → hidden`（投影也融合）
3. `modulation: double=False`（⭐ 单次调制，双流是双倍）

### ⭐ 读完后发现的真差异（对我们有利，**不改**）
| | FLUX 单流 | **我们的 KP** |
|---|---|---|
| QKV / MLP | ⛔ **融合**成一个 `7d` 矩阵 | ✅ **分开**（`3d²` + `5d²`）|
| 能力包换 attn 权重 | 得从 7d 矩阵里切（脆弱）| ✅ 直接换 `qkv` 那一块（清晰）|

⇒ **结论：我们的"分开"是能力总线的结构性优势，不改成他们的形状。**
   这正是"借零件 ≠ 借架构"的实际含义。

## 清理
若不再使用：`rm -rf vendor/sd-scripts`

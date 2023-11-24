# 折叠常数

## 简介

此示例首先生成一个带有多个操作的模型，这些操作可以在推理之前进行评估，然后折叠这些操作并导出一个新模型。

常数折叠涉及预先计算不依赖于运行时信息的表达式。实际上，这意味着在 ONNX GraphSurgeon 图中仅依赖于 `Constant` 的任何节点都可以被折叠。

ONNX GraphSurgeon 内置常数折叠的一个限制是它不会旋转节点。所以，假设 `x` 是图的输入，`c0`、`c1` 和 `c2` 是图中的常数：

- `x + (c0 + (c1 + c2))` **将** 被折叠
- `((x + c0) + c1) + c2` **不会** 被折叠，即使从数学上看它等效于前一个表达式（不考虑浮点数舍入误差）。

## 先决条件

1. ONNX GraphSurgeon 使用 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 来评估图中的常数表达式。您可以使用以下命令安装它：

   ```bash
   python3 -m pip install onnxruntime
   ```

## 运行示例

1. 通过运行以下命令生成一个带有多个节点的模型并保存为 `model.onnx`：

   ```bash
   python3 generate.py
   ```

   生成的模型计算 `output = input + ((a + b) + d)`，其中 `a`、`b` 和 `d` 都是常数，均设置为 `1`：

   ![../resources/05_model.onnx.png](./assets/05_model.onnx.png)

2. 通过运行以下命令在图中折叠常数，并将其保存为 `folded.onnx`：

   ```bash
   python3 fold.py
   ```

   这将用一个单一的常数张量（全部为 `3`）替换表达式：`((a + b) + d)`。生成的图将计算 `output = input + e`，其中 `e = ((a + b) + d)`：

   此脚本还会显示 `Graph.fold_constants()` 的帮助输出。

   ![../resources/05_folded.onnx.png](./assets/05_folded.onnx.png)

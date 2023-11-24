# 使用图层 API 创建模型

## 简介

此示例使用 `Graph.layer()` 函数与 `Graph.register()` 结合使用，演示了如何更轻松地构建复杂的 ONNX 模型。

`Graph.layer()` API 允许您更轻松地向 `Graph` 添加 `Node`。除了创建节点外，此函数还可以创建输入和输出张量，并自动将节点插入图中。有关详细信息，请查看 `Graph.layer()` 的 `help` 输出。

**注意**：您仍然需要自己设置 `Graph` 的输入和输出！

`Graph.layer()` 可以用来实现自己的函数，这些函数可以注册到 `Graph.register()` 中。例如，要实现一个 `graph.add` 函数，该函数将 "Add" 操作插入到图中，可以编写如下代码：

```python
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out"])
```

然后可以像这样调用它：

```python
[Y] = graph.add(*graph.add(X, B), C)
```

这将在图中添加一组节点，用于计算 `Y = (X + B) + C`（假设 X、B、C 是图中的某些张量），而无需手动创建涉及的中间张量。

## 运行示例

1. 通过运行以下命令生成一个复杂度适中的模型并保存为 `model.onnx`：

   ```bash
   python3 generate.py
   ```

   此脚本还会显示 `Graph.layer()` 的 `help` 输出。

   生成的模型如下所示：

   ![../resources/07_model.onnx.png](./assets/07_model.onnx.png)

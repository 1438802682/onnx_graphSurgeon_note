# 替换子图

## 简介

此示例首先生成一个包含 `Min` 操作和 `Max` 操作的模型，然后使用在 [示例 07](../07_creating_a_model_with_the_layer_api) 中看到的 `graph.layer()` 和 `graph.register()` API 创建一个函数，该函数可以用于替换此子图为一个 `Clip` 操作。

这可以用于启用 TensorRT 插件与 ONNX，这样可以很有用。

## 子图替换基础知识

替换子图的过程包括三个步骤。例如，对于具有以下结构的图：

         Tensor0
            |
          Node0
            |
         Tensor1    Tensor2
               \    /
                Node1
                  |
               Tensor3
                  |
                Node2

为了替换由 [`Node0`, `Node1`] 组成的子图，我们需要执行以下操作：

1. 断开子图输入的**输出**：`Tensor0` 和 `Tensor2`

   这意味着我们需要删除 `Tensor0` 和 `Node0` 之间的边缘，以及 `Tensor2` 和 `Node1` 之间的边缘。

2. 断开子图输出的**输入**：`Tensor3`

   这意味着我们需要删除 `Node1` 和 `Tensor3` 之间的边缘。

这将使我们得到以下图：

         Tensor0     Tensor2
    
               Tensor3
                  |
                Node2

以及现在断开的子图：

          Node0
            |
         Tensor1
               \
                Node1

3. 最后，我们需要插入我们的节点，使其具有输入：[`Tensor0`, `Tensor2`] 和输出：[`Tensor3`, ]。

完成此步骤后，我们得到了最终的图（`cleanup()` 将删除悬挂的子图）：

         Tensor0     Tensor2
               \     /
              MyNewNode0
                  |
               Tensor3
                  |
                Node2

## 运行示例

1. 通过运行以下命令生成包含 `Min` 和 `Max` 操作的模型：

   ```bash
   python3 generate.py
   ```

   生成的模型将计算 `max(min(x, 6), 0)`，并且如下所示：

   ![../resources/08_model.onnx.png](./assets/08_model.onnx.png)

2. 通过运行以下命令使用 `Clip` 操作替换子图：

   ```bash
   python3 replace.py
   ```

   最终的模型将包括一个 `clip(x, min=0, max=6)`，并且如下所示：

   ![../resources/08_replaced.onnx.png](./assets/08_replaced.onnx.png)

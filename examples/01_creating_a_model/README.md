# 创建一个 ONNX 模型

## 简介

ONNX GraphSurgeon 包括一个[中间表示（IR）](../../README.md#ir)，可以导出为 ONNX 格式。此外，IR 还提供了一个简单的 API，使得可以手动构建图。

此示例创建一个包含单个 GlobalLpPool 节点的 ONNX 模型。

## 运行示例

通过运行以下命令生成模型并保存为 `test_globallppool.onnx`：

```bash
python3 example.py
```

生成的模型如下所示：

![../resources/01_test_globallppool.onnx.png](./assets/01_test_globallppool.onnx.png)

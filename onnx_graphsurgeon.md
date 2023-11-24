# ONNX GraphSurgeon

## 目录

- [介绍](#介绍)
- [安装](#安装)
- [示例](#示例)
- [理解基础知识](#理解基础知识)
  - [导入器](#导入器)
  - [IR](#ir)
    - [Tensor（张量）](#tensor张量)
    - [Node（节点）](#node节点)
    - [关于修改输入和输出的注意事项](#关于修改输入和输出的注意事项)
    - [Graph（图）](#graph图)
  - [导出器](#导出器)
- [高级用法](#高级用法)
  - [处理具有外部数据的模型](#处理具有外部数据的模型)

## 介绍

ONNX GraphSurgeon 是一个工具，可以让您轻松生成新的 ONNX 图或修改现有图。

## 安装

### 使用预构建的 Wheel 包

```bash
python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

### 从源代码构建

#### 使用 Make 目标

```
make install
```

#### 手动构建

1. 构建 Wheel 包：

```
make build
```

2. 从 **仓库外部** 手动安装 Wheel 包：

```
python3 -m pip install onnx_graphsurgeon/dist/onnx_graphsurgeon-*-py2.py3-none-any.whl
```

## 示例

[示例](./examples) 目录包含了 ONNX GraphSurgeon 常见用例的几个示例。提供的可视化是使用 [Netron](https://github.com/lutzroeder/netron) 生成的。

## 理解基础知识

ONNX GraphSurgeon 由三个主要组件组成：导入器、IR 和导出器。

### 导入器

导入器用于将图导入到 ONNX GraphSurgeon IR 中。导入器接口定义在 [base_importer.py](./onnx_graphsurgeon/importers/base_importer.py) 中。

ONNX GraphSurgeon 还提供了[高级导入器 API](./onnx_graphsurgeon/api/api.py) 以提高易用性：

```python
graph = gs.import_onnx(onnx.load("model.onnx"))
```

### IR

中间表示（IR）是进行对图的所有修改的地方。它还可用于从头开始创建新图。IR 包括三个组件：[Tensor（张量）](./onnx_graphsurgeon/ir/tensor.py)、[Node（节点）](./onnx_graphsurgeon/ir/node.py) 和 [Graph（图）](./onnx_graphsurgeon/ir/graph.py)。

几乎所有组件的成员变量都可以自由修改。有关这些类的各种属性的详细信息，可以在交互式 shell 中使用 `help(<class_or_instance>)` 查看帮助输出，或在脚本中使用 `print(help(<class_or_instance>))`，其中 `<class_or_instance>` 是 ONNX GraphSurgeon 类型或该类型的实例。

#### Tensor（张量）

张量分为两个子类：`Variable` 和 `Constant`。

- `Constant` 是其值在一开始就已知的张量，可以检索为 NumPy 数组并进行修改。*注意：`Constant` 的 `values` 属性是按需加载的。如果不访问该属性，值将不会作为 NumPy 数组加载。*
- `Variable` 是其值在推断时才会知道的张量，但可能包含有关数据类型和形状的信息。

张量的输入和输出始终是节点。

**来自 ResNet50 的示例常量张量：**

```
>>> print(tensor)
Constant (gpu_0/res_conv1_bn_s_0)
[0.85369843 1.1515082  0.9152944  0.9577646  1.0663182  0.55629414
 1.2009839  1.1912311  2.2619808  0.62263143 1.1149117  1.4921428
 0.89566356 1.0358194  1.431092   1.5360111  1.25086    0.8706703
 1.2564877  0.8524589  0.9436758  0.7507614  0.8945271  0.93587166
 1.8422242  3.0609846  1.3124607  1.2158023  1.3937513  0.7857263
 0.8928106  1.3042281  1.0153942  0.89356416 1.0052011  1.2964457
 1.1117343  1.0669073  0.91343874 0.92906713 1.0465593  1.1261675
 1.4551278  1.8252873  1.9678202  1.1031747  2.3236883  0.8831993
 1.1133649  1.1654979  1.2705412  2.5578163  0.9504889  1.0441847
 1.0620039  0.92997414 1.2119316  1.3101407  0.7091761  0.99814713
 1.3404484  0.96389204 1.3435135  0.9236031 ]
```

**来自 ResNet50 的示例变量张量：**

```
>>> print(tensor)
Variable (gpu_0/data_0): (shape=[1, 3, 224, 224], dtype=float32)
```

#### Node（节点）

`Node` 定义图中的操作。节点可以指定属性；属性值可以是任何 Python 基本类型，以及 ONNX GraphSurgeon 的 `Graph` 或 `Tensor`。

节点的输入和输出始终是张量。

**来自 ResNet50 的示例 ReLU 节点：**

```
>>>

 print(node)
 (Relu)
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

在这种情况下，节点没有属性。否则，属性将显示为 `OrderedDict`。

#### 关于修改输入和输出的注意事项

节点和张量的 `inputs`/`outputs` 成员具有特殊逻辑，当您进行更改时，会更新所有受影响的节点/张量的输入/输出。这意味着，例如，当您更改输入张量的 `outputs` 时，不需要更新节点的 `inputs`。

考虑以下节点：

```
>>> print(node)
 (Relu).
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

可以这样访问输入张量：

```
>>> tensor = node.inputs[0]
>>> print(tensor)
Tensor (gpu_0/res_conv1_bn_1)
>>> print(tensor.outputs)
[ (Relu).
	Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
	Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

如果我们从张量的输出中删除节点，这也会反映在节点的输入中：

```
>>> del tensor.outputs[0]
>>> print(tensor.outputs)
[]
>>> print(node)
 (Relu).
    Inputs: []
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

#### Graph（图）

`Graph` 包含零个或多个 `Node` 和输入/输出 `Tensor`。

中间张量不会被明确跟踪，而是从图中包含的节点中检索的。

`Graph` 类公开了几个函数。以下是其中的一小部分：

- `cleanup()`: 删除图中未使用的节点和张量。
- `toposort()`: 对图进行拓扑排序。
- `tensors()`: 返回一个 `Dict[str, Tensor]`，将张量名称映射到张量，通过遍历图中的所有张量进行操作。这是一个 `O(N)` 操作，因此对于大型图可能较慢。

要查看完整的 Graph API，可以在交互式 Python shell 中使用 `help(onnx_graphsurgeon.Graph)`。

### 导出器

导出器用于将 ONNX GraphSurgeon IR 导出到 ONNX 或其他类型的图中。导出器接口定义在 [base_exporter.py](./onnx_graphsurgeon/exporters/base_exporter.py) 中。

ONNX GraphSurgeon 还提供了[高级导出器 API](./onnx_graphsurgeon/api/api.py) 以提高易用性：

```python
onnx.save(gs.export_onnx(graph), "model.onnx")
```

## 高级用法

### 处理具有外部数据的模型

使用具有外部存储数据的模型与处理没有外部数据的 ONNX 模型几乎相同。有关如何加载此类模型的详细信息，请参阅[官方 ONNX 文档](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#loading-an-onnx-model-with-external-data)。要将模型导入到 ONNX-GraphSurgeon 中，可以像平常一样使用 `import_onnx` 函数。

在导出时，只需要多执行一步：

1. 从 ONNX-GraphSurgeon 导出模型如常：

   ```python
   model = gs.export_onnx(graph)
   ```

2. 更新模型，使其将数据写入外部位置。如果未指定位置，默认为与 ONNX 模型相同的目录：

   ```python
   from onnx.external_data_helper import convert_model_to_external_data
   
   convert_model_to_external_data(model, location="model.data")
   ```

3. 然后，您可以像往常一样保存模型：

   ```python
   onnx.save(model, "model.onnx")
   ```

### ONNX GraphSurgeon

文档地址：https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html

本页面包含 ONNX GraphSurgeon 的 Python API 文档。ONNX GraphSurgeon 提供了一种创建和修改 ONNX 模型的便捷方法。

有关安装说明和示例，请参阅 [此页面](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)。

**API参考**

- [Export](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/exporters/toc.html)
- [Import](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/importers/toc.html)
- Intermediate Representation
  - [Graph](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/graph.html)
  - [Node](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/node.html)
  - [Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/toc.html)
    - [Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html)
    - [Variable](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/variable.html)
    - [Constant](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/constant.html)
- [Exception](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/exception/toc.html)

## Export

```py
onnx_graphsurgeon.export_onnx(
    graph: onnx_graphsurgeon.ir.graph.Graph, 
    do_type_check=True, 
    **kwargs
)→ onnx.onnx_ml_pb2.ModelProto
```

- **功能**：将onnx-graphsurgeon图导出为ONNX模型。
- **参数**：
  -  `graph (Graph)` - 要导出的图
  - `do_type_check (bool)` - 是否检查输入和输出张量是否定义了数据类型，并在没有定义的情况下失败。默认为True。
  - `kwargs` - onnx.helper.make_model的附加参数
- **返回**： 一个相应的ONNX模型。
- **返回类型**： onnx.ModelProto

## Import

```py
onnx_graphsurgeon.import_onnx(
    onnx_model: onnx.onnx_ml_pb2.ModelProto
) → onnx_graphsurgeon.ir.graph.Graph
```

- **功能**： 从提供的ONNX模型中导入一个onnx-graphsurgeon图。
- **参数**： `onnx_model (onnx.ModelProto)` - ONNX模型。
- **返回**： 一个相应的onnx-graphsurgeon图。
- **返回类型**： Graph

## Intermediate Representation

### Graph

```py
class onnx_graphsurgeon.Graph(
    nodes: Sequence[onnx_graphsurgeon.ir.node.Node] = None,
    inputs: Sequence[onnx_graphsurgeon.ir.tensor.Tensor] = None,
    outputs: Sequence[onnx_graphsurgeon.ir.tensor.Tensor] = None,
    name: str = None,
    doc_string: str = None,
    opset: int = None,
    import_domains: Sequence[str] = None,
    producer_name: str = None,
    producer_version: str = None,
    functions: Sequence[Function] = None
)
```

**功能**：代表包含节点和张量的图。

**参数**：

- nodes (Sequence[Node]) - 图中节点的列表。
- inputs (Sequence[Tensor]) - 图输入张量的列表。
- outputs (Sequence[Tensor]) - 图输出张量的列表。
- name (str) - 图的名称。默认为“onnx_graphsurgeon_graph”。
- doc_string (str) - 图的文档字符串。默认为“”。
- opset (int) - 导出此图时使用的ONNX操作集。
- producer_name (str) - 用于生成模型的工具的名称。默认为“”。
- producer_version (str) - 生成工具的版本。默认为“”。





#### `staticregister(opsets=None)`

向Graph类注册指定操作集群的函数。注册函数后，可以像访问普通成员函数一样访问它。

##### **例如**：

```py
@Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])

graph.add(a, b)
```

##### **参数**：

- opsets (Sequence[int]) - 要为其注册函数的一组操作集。如果它们为不同的操作集注册，可以同时注册具有相同名称的多个函数。为相同操作集注册具有重复名称的函数将覆盖之前为那些操作集注册的任何函数。默认情况下，函数被注册到所有操作集。



#### `node_ids()`

返回一个上下文管理器，用于为图中的节点提供唯一的整数ID。

##### 例如：

```py
with graph.node_ids():
    assert graph.nodes[0].id != graph.nodes[1].id
```

**返回**：	

- 一个上下文管理器，用于为节点提供唯一的整数ID。


**返回类型**：  

- NodeIDAdder




#### `subgraphs(recursive=False)`

方便函数，用于遍历此图中包含的所有子图。子图位于 ONNX 控制流节点的属性中，例如“If”和“Loop”。

##### 	参数：

- recursive （bool） - 是否递归搜索此图的子图以查找更多子图。默认值为 False。


##### 	返回：

- 一个生成器，它循环访问此图中包含的子图。




#### `cleanup(remove_unused_node_outputs=False,recurse_subgraphs=True,remove_unused_graph_inputs=False, recurse_functions=True`

从图形中删除未使用的节点和张量。如果节点或张量对任何图形输出没有贡献，则认为该节点或张量未使用。

此外，图输入张量的任何生产者节点，以及不在图中的图输出张量的使用者节点，都会从图中删除。

> 注意：此函数绝不会修改图形输出张量。
>

##### 参数：

- remove_unused_node_outputs （bool） - 是否移除节点未使用的输出张量。这永远不会删除空张量（即可选但省略）输出。默认值为 False。
- recurse_subgraphs （bool） - 是否递归清理子图。默认值为 True。
- remove_unused_graph_inputs （bool） - 是否删除未使用的图形输入。默认值为 False。
- recurse_functions （bool） - 是否同时清理此图的局部函数。默认值为 True。		

##### 返回：

- self




#### `toposort(recurse_subgraphs=True,recurse_functions=True,mode='full')`	

对图形进行拓扑排序。

##### 参数：

- recurse_subgraphs （bool） - 是否对子图进行递归拓扑排序。仅当 mode=“full” 或 mode=“nodes” 时适用。默认值为 True。
- recurse_functions （bool） - 是否对该图函数的节点进行拓扑排序。仅当 mode=“full” 或 mode=“nodes” 时适用。默认值为 True。
- mode （str） - 是否重新排序此图的节点列表和/或函数列表。可能的值： - “full”：对节点列表和函数列表进行拓扑排序。- “nodes”：仅对节点列表进行排序。
- “functions”：仅对函数列表进行排序。默认为“full”。

##### 返回：

- self




#### `tensors(check_duplicates=False)`

通过遍历所有节点来创建此图使用的所有张量的张量图。此映射中省略了空张量。

张量保证按图中节点的顺序排列。因此，如果图形是拓扑排序的，张量图也会是拓扑排序的。

##### 参数：

​	check_duplicates （bool） - 遇到多个同名张量时是否失败。

##### 提高：

​	OnnxGraphSurgeonException - 如果 check_duplicates为 True，并且图中的多个不同张量共享相同的名称。

##### 返回：

​	张量名称到张量的映射。

##### 返回类型：

​	OrderedDict[str， [Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)]          



#### `fold_constants(fold_shapes=True,recurse_subgraphs=True,partitioning=None,error_ok=True,flatten_subgraphs=True,size_threshold=None,should_exclude_node=None,recurse_functions=True)`

在图形中就地折叠常量。在调用此函数之前，必须对图形的节点和函数进行拓扑排序（参见 `toposort()`）。

此函数在折叠常量后不会删除常量。为了摆脱这些挂起的节点，您可以运行 `cleanup()` 函数。

注意：由于此函数的实现方式，图形必须可导出到 ONNX，并在 ONNX-Runtime 中进行评估。此外，还必须安装 ONNX-Runtime。

##### 参数：

- fold_shapes （bool） - 是否折叠图中的 Shape 节点。这需要在图形中推断形状，并且只能折叠静态形状。默认值为 True。

- recurse_subgraphs （bool） - 是否递归折叠子图中的常量。默认值为 True。

- partitioning （Union[str， None]） –是否/如何对图形进行分区，以便折叠模型的一部分时的错误不会影响其他部分。可用模式包括：

  - None：不对图形进行分区。如果推理失败，则不会折叠任何常量。

  - `”basic”: Partition the graph. If inference fails in one partition, other partitions will`

    ​	不受影响

  - `”recursive”: Parition the graph recursively. If inference fails in a partition, the partition`

    ​	将进一步分摊

  默认值为 None

- error_ok （bool） - 是否应抑制推理错误。当此值为 False 时，将重新引发推理过程中遇到的任何错误。默认值为 True。
- flatten_subgraphs （bool） - 是否在可能的情况下展平子图。例如，如果具有恒定条件的节点可以展平到父图中。
- size_threshold （int） – 要折叠常量的最大大小阈值（以字节为单位）。任何大于此值的张量都不会被折叠。设置为“无”可禁用大小阈值并始终折叠常量。例如，某些模型可能会将 Tile 或 Expand 等运算应用于常量，这可能会导致非常大的张量。与其预先计算这些常量并增加模型大小，不如跳过折叠它们并允许在运行时计算它们。默认值为 None
- should_exclude_node （Callable[[gs.Node]， bool]） - 一个可调用对象，它接受图中的 onnx-graphsurgeon 节点，并报告是否应将其从折叠中排除。这仅适用于其他可折叠的节点。请注意，防止节点被折叠也会防止其使用者被折叠。默认为始终返回 False 的可调用对象。


- recurse_functions （bool） - 是否在此图的函数中折叠常量。默认值为 True。

##### 返回：

- self





#### `layer(inputs=None,outputs=None,args,kwargs)`

创建一个节点，将其添加到此图中，并选择性地创建其输入和输出张量。

输入和输出列表可以包括各种不同的类型：

- Tensor：

  ​	提供的任何张量都将在所创建节点的输入/输出中按原样使	用。因此，您必须确保提供的张量具有唯一的名称。

- str：

  ​	如果提供了字符串，则此函数将使用该字符串生成新的张量以生成名称。它将在提供的字符串末尾附加一个索引，以保证名称的唯一性。

- numpy.ndarray：

  ​	如果提供了 NumPy 数组，则此函数将使用名称前缀“onnx_graphsurgeon_constant”生成常量张量，并在前缀末尾附加索引以保证名称唯一。

- Union[List[Number], Tuple[Number]]：

  ​	如果提供了数字列表或元组（int 或 float），则此函数将使用名称前缀“onnx_graphsurgeon_lst_constant”生成常量张量，并在前缀末尾附加索引以保证名称唯一。张量的值将是包含指定值的一维数组。数据类型将为 np.float32 或 np.int64。



##### 参数：

- inputs （List[Union[[*Tensor*](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)， str， numpy.ndarray]]） - 输入列表
- outputs （List[Union[[*Tensor*](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)， str， numpy.ndarray]]） - 输出列表
- args/kwargs – 这些直接传递给 Node 的构造函数

##### 返回：

- 节点的输出张量


##### 返回类型：

- List[[Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)]






#### `copy(tensor_map:Optional[collections.OrderedDict[str，[onnx_graphsurgeon.ir.tensor.Tensor]] = None)`

复制图形。

这将复制图中的所有节点和张量，但不会对权重或属性进行深度复制（Graph 属性除外，它将使用其复制方法进行复制）。

##### 参数：

- tensor_map （OrderedDict[str，[*Tensor*](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)]） - 张量名称到外图张量的映射。如果这是最外层的图形，则应为 None。


##### 返回：

- Graph的副本


##### 返回类型：

- [Graph](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/graph.html#onnx_graphsurgeon.Graph)



### Node

```py
classonnx_graphsurgeon.Node(op: str, name: Optional[str] = None, 
                            attrs: Optional[Dict[str, object]] = None, 
                            inputs: Optional[List[onnx_graphsurgeon.ir.tensor.Tensor]] = None, 
                            outputs: Optional[List[onnx_graphsurgeon.ir.tensor.Tensor]] = None, 
                            domain: Optional[str] = None)
```

**Bases**: `object`

一个节点在图中表示一个操作，消耗零个或多个张量，并产生零个或多个张量。

**参数**：

- op （str） - 此节点执行的操作。
- name （str） - 此节点的名称。
- attrs （Dict[str， object]） - 将属性名称映射到其值的字典。
- inputs （List[[*Tensor*](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)]） - 零个或多个输入张量的列表。


- outputs （List[[*Tensor*](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)]） - 零个或多个输出张量的列表。


- domain （str） - 该节点的域，



#### `classAttributeRef(name:str,type:type)`

Bases: `object`

AttributeRef 是一个属性值，它引用父函数中的属性。只有当节点位于 Function 中时，节点的属性才能是 AttributeRef。

##### 参数：	

- name （str） – 父函数中引用属性的名称。


- type （type） - 属性的类型。



#### `i(tensor_idx=0,producer_idx=0)`

方便函数，用于获取该节点的输入张量之一的生产者节点。请注意，与 o（） 函数相比，参数是交换的;这是因为张量可能只有一个生产者

##### 例如：

```
assert node.i() == node.inputs[0].inputs[0]
assert node.i(1, 2) == node.inputs[1].inputs[2]
```

##### 参数：

- tensor_idx （int） - 该节点的输入张量索引。默认值为 0。
- producer_idx （int） - 输入张量的生产者的索引，如果张量有多个生产者。默认值为 0

##### 返回

- 指定的生产者（输入）节点


##### 返回类型

- 节点 Node





#### `o(consumer_idx=0,tensor_idx=0)`

便利函数，用于获取该节点的输出张量之一的消费者节点。

##### 例如：

```
assert node.o() == node.outputs[0].outputs[0]
assert node.o(2, 1) == node.outputs[1].outputs[2]
```

##### 参数：

- consumer_idx （int） - 输入张量的消费者索引。默认值为 0。
- tensor_idx （int） - 如果该节点有多个输出，则此节点的输出张量的索引。默认值为 0。

##### 返回：

- 指定的使用者（输出）节点。


##### 返回类型：

- Node




#### `subgraphs(recursive=False)`

方便函数，用于遍历此节点中包含的所有子图。节点子图位于 ONNX 控制流节点的属性中，例如“If”和“Loop”。

##### 参数：

- recursive （bool） - 查找子图时是否递归到子图节点中。默认值为 False。


##### 返回：

- 一个生成器，它遍历此节点的子图。




#### `copy(inputs:Optional[List[onnx_graphsurgeon.ir.tensor.Tensor]]=None,outputs:Optional[List[[onnx_graphsurgeon.ir.tensor.Tensor]]]=None,tensor_map=None)`

创建此节点的浅表副本，覆盖输入和输出信息。

> 注意：通常，您应该只制作 Graph 的副本。
>





###  Tensor

#### Tensor

```py
class onnx_graphsurgeon.Tensor
```

**Bases**: `object`

图中张量的抽象基类

> **此类是抽象的，不能直接构造。**



##### `is_empty()`

返回此张量在图中是否被视为空。

> 注意：这里的“空”是指张量的名称，可选张量省略了该名称，而不是张量的形状
>

##### 返回：

- 张量是否为空，这意味着它用于省略的可选输入或输出。


##### 返回类型：

- bool




##### `to_constant(values:numpy.ndarray,data_location: Optional[int] = None)`

就地修改此张量以将其转换为常量。这意味着张量的所有使用者/生产者都将看到更新。

##### 参数：

- values （np.ndarray） - 此张量中的值


- data_location （int） - 一个枚举值，指示张量数据的存储位置。通常，这将来自 onnx。TensorProto.DataLocation。

##### 返回：

- self




##### `to_variable(dtype:Union[numpy.dtype,onnx.TensorProto.DataType]=None,shape:Sequence[Union[int, str]] = [])`

就地修改此张量以将其转换为变量。这意味着张量的所有使用者/生产者都将看到更新。

##### 参数：

- dtype （Union[numpy.dtype， onnx.TensorProto.DataType]） - 张量的数据类型。
- shape （Sequence[int]） - 张量的形状

**返回**

- self




##### `i(tensor_idx=0,producer_idx=0)`

方便函数，用于获取该张量的输入节点之一的输入张量。请注意，与 o（） 函数相比，参数是交换的;这是因为张量可能只有一个生产者

##### 例如：

```
assert tensor.i() == tensor.inputs[0].inputs[0]
assert tensor.i(1, 2) == tensor.inputs[2].inputs[1]
```

##### 参数：

- tensor_idx （int） - 输入节点的输入张量的索引。默认值为 0。

- producer_idx （int） - 输入张量的生产者节点的索引（如果张量有多个生产者）。默认值为 0。

##### 返回：

- 指定的生产者（输入）张量


##### 返回类型：

- [Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)



##### `o(consumer_idx=0,tensor_idx=0)`

方便函数，用于获取该张量的输出节点之一的输出张量。

##### 例如：

```
assert tensor.o() == tensor.outputs[0].outputs[0]
assert tensor.o(2, 1)==tensor.outputs[2].outputs[1]
```

##### 参数：

- consumer_idx （int） - 输入张量的消费者索引。默认值为 0。


- tensor_idx （int） - 如果节点有多个输出，则节点的输出张量的索引。默认值为 0。

##### 返回：

- 指定的使用者（输出）tensor


##### 返回类型：

- [Tensor](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)





#### Variable

```py
class onnx_graphsurgeon.Variable(name:str,
                                 dtype:Union[numpy.dtype,onnx.TensorProto.DataType]=None,
                                 shape:Sequence[Union[int,str]]=None)
```

**Bases**: [`onnx_graphsurgeon.ir.tensor.Tensor`](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)

表示其值已知的张量。

**参数**：

- name （str） - 张量的名称。


- values （numpy.ndarray） - 此张量中的值，采用 NumPy 数组的形式。


- data_location （int） - 一个枚举值，指示张量数据的存储位置。通常，这将来自 onnx。TensorProto.DataLocation。



##### `to_constant(values: numpy.ndarray)`

就地修改此张量以将其转换为常量。这意味着张量的所有使用者/生产者都将看到更新。

**参数**：

- values （np.ndarray） - 此张量中的值

- data_location （int） - 一个枚举值，指示张量数据的存储位置。通常，这将来自 onnx。TensorProto.DataLocation。

**返回**：

- self




##### `copy()`

制作此张量的浅拷贝，省略输入和输出信息。

注意：通常，您应该只制作 Graph 的副本。



#### Constant

```py
class onnx_graphsurgeon.Constant(name:str,
                                 values:Union[numpy.ndarray,onnx_graphsurgeon.ir.tensor.LazyValues],
                                 data_location: Optional[int] = None)
```

**Bases**: [`onnx_graphsurgeon.ir.tensor.Tensor`](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/ir/tensor/tensor.html#onnx_graphsurgeon.Tensor)

表示其值已知的张量。

##### 参数：

- name （str） - 张量的名称。


- values （numpy.ndarray） - 此张量中的值，采用 NumPy 数组的形式。


- data_location （int） - 一个枚举值，指示张量数据的存储位置。通常，这将来自 onnx。TensorProto.DataLocation。



##### `to_variable(dtype: Optional[numpy.dtype] =None,shape: Sequence[Union[int, str]] = [])`

就地修改此张量以将其转换为变量。这意味着张量的所有使用者/生产者都将看到更新。

##### 参数：

- dtype （Union[numpy.dtype， onnx.TensorProto.DataType]） - 张量的数据类型。

- shape （Sequence[int]） - 张量的形状。

##### 返回：

- self




##### `copy()`

制作此张量的浅拷贝，省略输入和输出信息。

> 注意：通常，您应该只制作 Graph 的副本。
>



## Exception

```
class onnx_graphsurgeon.OnnxGraphSurgeonException
```

**基类**：Exception

由ONNX-GraphSurgeon引发的异常。


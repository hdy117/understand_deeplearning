# `nn.Module.train()` 与 `nn.Module.eval()` 的区别

> 文档状态：代码解释  
> 更新时间：2026-11-16  
> 关联代码：[`train_flow_matching.py`](./train_flow_matching.py)、[`train_vae.py`](./train_vae.py)、[`flow_matching.py`](./flow_matching.py)

`nn.Module.train()` 和 `nn.Module.eval()` 切换的是模块树中的**运行模式**，不是“是否允许梯度计算”。理解它们时，最重要的一条关系是：

```python
model.eval() == model.train(False)
```

`eval()` 并不是一个独立的复杂流程，而是 `train(False)` 的便捷写法。两者都会修改模块及其所有子模块的 `training` 标志。

## 1. 底层只做模式传播

PyTorch 中这两个方法的核心逻辑可以概括为：

```python
def train(self, mode: bool = True):
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self

def eval(self):
    return self.train(False)
```

因此，调用：

```python
model.train()
```

会把当前模块以及所有子模块的 `training` 设置为 `True`。

调用：

```python
model.eval()
```

则把它们统一设置为 `False`。

这种设置是持久状态。调用 `eval()` 后，不会在某次 `forward()` 结束或某个 batch 结束后自动恢复为训练模式。训练代码必须在下一次训练前显式切回：

```python
model.train()
```

## 2. 哪些层会响应模式切换

`training` 只是一个通用信号，具体层可以决定是否响应它。最典型的行为差异如下：

| 层或操作 | `train()` 模式 | `eval()` 模式 |
|---|---|---|
| `nn.Dropout` | 随机置零，并缩放保留值 | 不丢弃，原样输出 |
| `nn.BatchNorm` | 使用当前 batch 统计量，更新 running stats | 使用保存的 running stats，不更新 |
| `nn.LayerNorm` | 行为不变 | 行为不变 |
| `nn.GroupNorm` | 行为不变 | 行为不变 |
| `nn.Linear` | 行为不变 | 行为不变 |
| `nn.Softmax` | 行为不变 | 行为不变 |
| Transformer 中的 dropout | 生效 | 关闭 |
| 普通 self-attention 权重 | 行为不变 | 行为不变 |

因此，`eval()` 不是让所有层“变成另一种数学”。对 Linear、LayerNorm、GroupNorm、Softmax 这类没有训练/推理差异的操作，切换模式不会改变它们的输出。真正改变行为的通常是有随机性、会更新统计量，或者显式检查 `self.training` 的层。

## 3. 当前 FlowMatching 中的实际影响

当前模型构造如下：

```python
self.dit = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=self.d_model,
        nhead=self.n_head,
        dim_feedforward=4 * self.d_model,
        activation="gelu",
        batch_first=True,
    ),
    num_layers=self.n_layer,
)
```

代码没有显式传入 `dropout`。在当前 PyTorch 版本中，`nn.TransformerEncoderLayer` 的默认值是 `dropout=0.1`，内部 `MultiheadAttention` 也会使用相应的 dropout 设置。因此：

```python
flow_net.train()
```

与：

```python
flow_net.eval()
```

在当前模型中会改变数值行为。模型没有 `BatchNorm`，但这不代表两种模式完全一样，因为 Transformer 内部仍可能包含 dropout。

训练脚本在训练路径中切换到：

```python
self.flow_net.train()
```

采样路径则切换到：

```python
flow_net.eval()
```

这两个调用属于标准的模式分工：训练时需要随机正则化，推理时希望同一组输入产生稳定输出。

## 4. `eval()` 不会关闭梯度

这是一个非常常见的误解。即使执行：

```python
model.eval()
```

模型参数依然默认保留：

```python
requires_grad=True
```

因此，下面的计算仍然可以建立 autograd 图并计算梯度：

```python
model.eval()
output = model(x)
loss = loss_fn(output, target)
loss.backward()
```

`eval()` 只负责把 Dropout、BatchNorm 等层切换到推理语义，它不负责修改 `requires_grad`，也不负责关闭 autograd。

关闭自动求导需要另一个工具：

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

或者：

```python
model.eval()
with torch.inference_mode():
    output = model(x)
```

可以用下面的职责表区分：

| 目的 | 使用的 API |
|---|---|
| 让 Dropout 停止随机丢弃 | `model.eval()` |
| 让 BatchNorm 使用 running stats | `model.eval()` |
| 不建立 autograd 图，减少内存 | `torch.no_grad()` |
| 在推理场景中不记录梯度且减少额外记录开销 | `torch.inference_mode()` |
| 让参数本身不需要梯度 | `parameter.requires_grad_(False)` |
| 真正更新参数 | `optimizer.step()` |

验证或推理时通常把前两项一起使用：

```python
model.eval()
with torch.inference_mode():
    predictions = model(inputs)
```

## 5. 为什么验证循环应切回训练模式

模式具有持续性。一个常见错误是验证结束后直接进入下一轮训练，却没有重新执行：

```python
model.train()
```

这样下一轮训练中的 Dropout 仍然是关闭的，BatchNorm 仍然使用推理统计量，训练和验证的语义就会串线。

典型训练结构应当明确写出两个阶段：

```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = ...
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        for batch in validation_loader:
            ...
```

`train()` 和 `eval()` 是模式边界；`zero_grad()`、`backward()`、`step()` 是参数更新边界；`no_grad()` 或 `inference_mode()` 是梯度记录边界。三者相关，但不是一回事。

## 6. 当前项目中的命名陷阱

`train_flow_matching.py` 中有一个需要特别说明的实现细节：`TrainDit` 本身继承自 `nn.Module`，但它重写了 `train()`：

```python
def train(self):
    # 整个训练流程入口
    ...
```

因此：

```python
self.train()
```

不一定表示 PyTorch 原生的 `nn.Module.train()`，而可能是在启动一个完整的训练流程。

训练脚本又在内部显式调用：

```python
self.flow_net.train()
```

这才是对被包装的 `FlowMatching` 子模块执行 PyTorch 的模式切换。

同样的名字问题也出现在采样路径上。当前代码使用：

```python
@torch.inference_mode()
def sample(self, flow_net: FlowMatching):
    flow_net.eval()
```

这里的 `flow_net.eval()` 是 PyTorch 的模式切换，而如果外部再调用 `TrainDit.eval()`，方法名的语义就可能与 PyTorch 原生 `eval()` 混淆。

更清晰的设计是避免在 `nn.Module` 子类中重写 `train()` 和 `eval()`：

```python
class Trainer:
    def fit(self):
        ...

    def evaluate(self):
        ...
```

如果确实要继承 `nn.Module`，也要让 `train()` 和 `eval()` 保持 PyTorch 的标准含义，把训练入口命名为 `fit()`、`run_training()` 等。这样，子模块和父模块的模式传播不会因为同名方法被意外截断。

## 7. 一句话总结

`train()` 与 `eval()` 的区别只在模块的运行模式：前者开启训练行为，后者开启评估行为；`eval()` 等价于 `train(False)`，但不关闭梯度。要节省 autograd 开销，必须另外使用 `torch.no_grad()` 或 `torch.inference_mode()`；要真正停止参数更新，则要处理优化器和 `requires_grad`，而不是只调用 `eval()`。

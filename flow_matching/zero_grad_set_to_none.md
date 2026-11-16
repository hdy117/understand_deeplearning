# `optimizer.zero_grad(set_to_none=True)` 详解

> 文档状态：代码解释  
> 更新时间：2026-11-16  
> 关联代码：[`train_flow_matching.py`](./train_flow_matching.py#L157)

在 [`train_flow_matching.py`](./train_flow_matching.py#L157) 的训练循环中有这样三行：

```python
self.optimizer.zero_grad(set_to_none=True)
loss.backward()
self.optimizer.step()
```

它们分别承担三个不同职责：清掉上一轮梯度、根据当前 loss 计算新梯度、使用新梯度更新参数。`set_to_none=True` 只影响第一步：清梯度时，它把参数的 `.grad` 设为 `None`，而不是写成一张全零 Tensor。

## 1. 这个参数控制什么

PyTorch 默认会让参数梯度在多次 `backward()` 之间累加。如果不清理，上一次迭代留下的梯度会和当前 loss 产生的梯度相加，导致一次参数更新混入多个 batch 或步骤的信息。

`optimizer.zero_grad()` 会遍历该优化器管理的参数，清理已有梯度。`set_to_none` 决定使用哪种清理方式：

```python
# 放入一张全零梯度
optimizer.zero_grad(set_to_none=False)

# 直接删除梯度引用
optimizer.zero_grad(set_to_none=True)
```

简化后的行为差异可以写成：

```python
# set_to_none=False
if p.grad is not None:
    p.grad.detach_()
    p.grad.zero_()

# set_to_none=True
if p.grad is not None:
    p.grad = None
```

两种方式都能防止旧梯度继续参与当前这一步更新，区别在于旧梯度是被“原地清零”还是“直接丢弃”。

在当前使用的 PyTorch 版本中，`zero_grad()` 的默认参数已经是 `set_to_none=True`。训练代码显式写出来，可以让这个语义一眼可见，而不必依赖读者记住当前版本的默认值。

## 2. 为什么通常更省资源

设为 `None` 的常见收益有两层。

首先，`set_to_none=True` 会解除对旧梯度 Tensor 的引用。如果没有其他对象继续持有它，这块梯度存储就可以被释放。设成 `False` 时，代码执行的是 `p.grad.zero_()`，梯度表仍留在原处，只是所有数值变成零。

其次，某些参数在当前一步可能根本没有参与计算图。清成 `None` 后，如果 `backward()` 没有为该参数产生梯度，它的 `.grad` 可以继续是 `None`；这比保留一张全是零的梯度表更准确，也减少无意义的存储和优化器处理。

因此，常规训练通常写成：

```python
optimizer.zero_grad(set_to_none=True)
```

它不会改变模型参数本身，也不会释放 optimizer 里的动量、二阶矩等状态，只影响参数梯度当前的表示。

## 3. `None` 梯度与全零梯度并不完全相同

从数值上看，“没有梯度”和“梯度全是零”似乎差不多；对 PyTorch 的状态管理来说，它们并不一样。

| 状态 | `.grad` | 含义 |
|---|---|---|
| `None` | 不存在梯度 Tensor | 当前步骤没有为该参数产生梯度 |
| `torch.zeros_like(p)` | 存在一张全零梯度 | 产生了一张数值为零的梯度 |

一个关键差异在优化器行为上：多数优化器会跳过 `.grad is None` 的参数，但面对全零梯度时仍可能执行权重衰减、动量更新或其他优化逻辑。

可以用 AdamW 观察这个区别：

```python
import torch
from torch import nn

# 梯度为 None：优化器跳过更新
p1 = nn.Parameter(torch.tensor([1.0]))
opt1 = torch.optim.AdamW([p1], lr=0.1, weight_decay=0.1)
p1.grad = None
opt1.step()

print(p1.item())  # 1.0


# 梯度为零 Tensor：优化器仍可能执行 step 中的处理
p2 = nn.Parameter(torch.tensor([1.0]))
opt2 = torch.optim.AdamW([p2], lr=0.1, weight_decay=0.1)
p2.grad = torch.zeros_like(p2)
opt2.step()

print(p2.item())  # 约 0.99
```

第二个参数发生变化，是因为 AdamW 的 weight decay 逻辑仍会处理这张零梯度 Tensor。这个例子不是说 `set_to_none=False` 一定错误，而是说明两种状态在少数场景下可以被优化器区分。

对于多任务模型、条件分支或只使用部分参数的训练，`None` 更接近“这一步完全不知道该参数该往哪里更新”；全零梯度则更像“算出了梯度，碰巧梯度为零”。大多数全参数训练不会依赖这个区别，但理解它能避免把两种状态混为一谈。

## 4. 在代码中的控制流

当前代码的内层训练步骤是：

```text
zero_grad 清旧梯度
    ↓
forward 计算当前预测
    ↓
loss 衡量预测与目标
    ↓
backward 产生新梯度
    ↓
step 用新梯度更新参数
```

把 `zero_grad()` 放在 `backward()` 前面，意味着每次新时间点的监督都拥有独立的一次参数更新。`set_to_none=True` 不改变这个控制顺序，只把清理步骤变轻。

当前 Flow Matching 的训练循环在 `total_steps=24` 的内层循环中每次都会执行：

```python
self.optimizer.zero_grad(set_to_none=True)
loss.backward()
self.optimizer.step()
```

因此，同一个 image batch 的 24 个时间点抽样会分别清梯度、反向传播和更新参数。若要做 gradient accumulation，则应调整这一控制流：先累积若干次 loss 的梯度，再在真正准备更新时清理一次，而不是每一步都清成 `None`。

## 5. 它不会做什么

`set_to_none=True` 不是模型或训练器的“万能清理”：

- 它不会把模型权重清零。
- 它不会清除 optimizer 的 momentum、exp_avg 等内部状态。
- 它不会替代 `optimizer.step()`。
- 它不会阻止同一次迭代内的多个 `backward()` 按设计累加。
- 它不会减少模型参数、激活值或 VAE 的显存。

它清理的范围是优化器管理的参数梯度，作用点是每一轮参数更新之前的状态边界。

## 6. 推荐写法

普通训练推荐保留这个参数：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

如果代码需要手动读取或修改 `.grad`，就要明确自己是否需要一张持续存在的零梯度表。例如，有些自定义逻辑会假设每个参数都有 `.grad`，这时全零 Tensor 可能更方便；但只要优化器能正确处理 `None`，通常没有必要为了“看起来整齐”而额外分配全零梯度。

一句话总结：`set_to_none=True` 表示**丢弃上一轮的梯度引用，而不是创建一张全零梯度表**。它通常更省内存，也能保留“这个参数当前没有梯度”这一状态信息。

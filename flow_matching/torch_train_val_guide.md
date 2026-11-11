# PyTorch 训练、验证与可变学习率：一个最小闭环

> 目标：用一份能直接跑通的代码，同时讲清训练、验证和 learning-rate schedule（学习率时间表）。
> 记住一句话：**训练是五动作循环；验证复用循环但关闭训练行为；scheduler 只负责在正确的时刻改学习率。**

## 1. 先看数据和控制流

```text
Dataset -> DataLoader -> batch -> model -> loss -> backward -> optimizer.step()
                                                    |
每个 epoch 结束                                      v
                              evaluate() -> scheduler.step() -> 下一轮
```

最小系统只有五件套：

| 部件 | 负责什么 |
|---|---|
| `Dataset` / `DataLoader` | 取样本、组 batch、训练集打乱 |
| `model` | 把输入 `x` 变成预测 `logits` |
| `loss_fn` | 用一个标量衡量预测有多差 |
| `optimizer` | 根据梯度改参数 |
| `lr_scheduler` | 按时间或验证指标改 optimizer 的学习率 |

## 2. 完整代码

下面把 scheduler 直接放进主循环，不另拆一份“加了 scheduler 的版本”。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class ToyDataset(Dataset):
    def __init__(self, num_samples=2000):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(num_samples, 20, generator=generator)
        weight = torch.randn(20, 3, generator=generator)
        self.y = (self.x @ weight).argmax(dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_data():
    ds = ToyDataset()
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(
        ds,
        [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0),
    )
    return (
        DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0),
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)  # 1. 清梯度
        logits = model(x)                      # 2. 前向
        loss = loss_fn(logits, y)              # 3. 算 loss
        loss.backward()                        # 4. 反向
        optimizer.step()                       # 5. 更新参数

        batch_n = len(x)
        total_loss += loss.item() * batch_n    # 只保存标量
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += batch_n

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        batch_n = len(x)
        total_loss += loss.item() * batch_n
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += batch_n

    return total_loss / total_n, total_correct / total_n


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl = make_data()

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 3),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)

        scheduler.step()  # optimizer 已更新完；为下一轮推进时间表
        next_lr = scheduler.get_last_lr()[0]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best.pt")
            mark = " <- best"
        else:
            mark = ""

        print(
            f"epoch {epoch:>2} | train {train_loss:.4f}/{train_acc:.3f} "
            f"| val {val_loss:.4f}/{val_acc:.3f} "
            f"| next lr {next_lr:.1e}{mark}"
        )


if __name__ == "__main__":
    main()
```

换成自己的任务通常只改三处：`Dataset`、模型结构、损失函数。分类用 `CrossEntropyLoss`，输入必须是未经过 softmax 的 logits；回归用 `MSELoss`；多标签分类用 `BCEWithLogitsLoss`。

## 3. Dataset 与 DataLoader

`Dataset` 描述“第 `i` 条样本长什么样”，只需要实现 `__len__` 和 `__getitem__`。示例里的 `ToyDataset` 返回 `(x, y)`；真实任务可以在 `__getitem__` 中读图片、tokenize 或做增强。

`DataLoader` 负责把样本组成 batch。关键参数可以用一句话记住：

| 参数 | 作用 | 常见选择 |
|---|---|---|
| `batch_size` | 每个 batch 多少条样本 | 先填满 GPU / CPU 吞吐，再看 loss 是否稳定 |
| `shuffle` | 每个 epoch 是否重新打乱顺序 | train 为 `True`，val/test 为 `False` |
| `num_workers` | 用几个子进程并行取数据 | 先在 Windows 用 `0`；CPU 数据读得慢再增加 |
| `drop_last` | 丢弃最后一个不满的 batch | 少量任务需要固定形状时再用 |
| `pin_memory` | 尝试把 batch 放进 pinned memory | CUDA 训练且数据加载是瓶颈时可设 `True` |

`drop_last` 默认是 `False`，因为多训练几条样本通常比丢掉它们划算。验证时不要打乱顺序；但指标仍要按样本数加权，不能直接平均各 batch 的指标。

## 4. 五动作，以及学习率藏在哪里

固定学习率时，最后一步是

$$
\text{optimizer.step()}: \qquad
\theta_{t+1}=\theta_t-\eta_t\nabla_\theta\mathcal{L}(\theta_t).
$$

其中 $\theta_t$ 是当前参数，$\nabla_\theta\mathcal{L}$ 是梯度。固定学习率时 $\eta_t=\eta$；scheduler 把 $\eta_t$ 变成随训练进程变化的函数。`optimizer.zero_grad()` 负责清掉上一步的梯度，因为 PyTorch 默认会累加梯度。

为什么需要它？固定学习率要同时服务两个互相拉扯的阶段：早期适合大一步探路，后期适合小一步精修。常见做法因此是前期较大、后期渐小；训练不稳定的大模型还会先做 **warmup**，让学习率从很小逐渐升到主学习率。

但这不代表“加 scheduler 必然涨点”。在 10 个 epoch 的玩具任务上，`CosineAnnealingLR(T_max=10)` 会让最后一轮的学习率接近零，验证准确率反而从约 0.94 掉到 0.89。调度器的收益通常来自足够长的训练：**固定 lr 已经出现平台，再降低 lr 精修**。

## 5. scheduler 怎么接，何时推进

先创建 `optimizer`，再创建 `scheduler`；后者持有前者的引用，并修改其中各个 `param_group["lr"]`。典型 epoch 时序如下：

```text
model.train()
for batch in train_loader:
    zero_grad -> forward -> loss -> backward -> optimizer.step()

model.eval() + no_grad()
evaluate()
scheduler.step()       <- 推进时间表，不是更新模型参数
```

大多数基于 epoch 的 scheduler 都在这里每轮调用一次；`OneCycleLR` 通常按 batch 调用；`ReduceLROnPlateau` 则要用 `scheduler.step(val_loss)` 提供反馈信号。

## 6. scheduler 参数与选择

两个余弦 scheduler 的签名只差时间参数：

```python
CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)
CosineAnnealingWarmRestarts(
    optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1
)
```

| 参数 | 含义 | 使用要点 |
|---|---|---|
| `optimizer` | scheduler 控制哪个 optimizer | 必须先后构造；optimizer 初始 `lr` 就是 $\eta_{\max}$ |
| `T_max` | 一次余弦下降覆盖多少步 | 常用作训练总 epoch / step 数；不会自动停止训练 |
| `T_0` | 第一次 restart 前经过多少步 | 每个 `step()` 前进一格，不是固定代表一个 epoch |
| `T_mult` | 每次 restart 后周期扩大倍数 | 整数且 $\ge 1$；`2` 表示周期按 2 倍增长 |
| `eta_min` | 学习率绝对下限 | 它是数值，不是比例；例如可写 `1e-5` |
| `last_epoch` | 时间表从哪个进度开始 | 续训优先恢复 `scheduler.state_dict()` |

两者共用余弦形状

$$
\eta_t=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}
\left(1+\cos\frac{\pi T_{\mathrm{cur}}}{T_i}\right),
$$

其中 $T_{\mathrm{cur}}$ 是当前周期内的进度，$T_i$ 是这一周期的长度。`CosineAnnealingLR` 只走一次，常用终点是 $T_{\max}$；Warm Restarts 在 $T_{\mathrm{cur}}$ 到达 $T_i$ 后把时钟归零，让学习率重新升高：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-5
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5
)
```

若每个 epoch 调用一次 `step()`，`T_0=10, T_mult=2` 的周期长度是 `10, 20, 40, 80...`，对应的 restart 时间边界约为 epoch `10, 30, 70, 150...`，具体叫第 10 还是第 11 个完成轮取决于你的计数方式。100 epoch 的训练不会正好在最低点收尾，周期末尾仍会处在较高学习率。restart 只重置学习率，**不会重置模型参数或 optimizer state**；它也不是 warmup。

简单选择：

| scheduler | 改 lr 的依据 | 适合的直觉 |
|---|---|---|
| `StepLR` | 每隔固定 epoch 乘一个系数 | 简单、可预测，常见基线 |
| `CosineAnnealingLR` | 一次余弦曲线从大降到小 | 总预算明确、希望平稳收尾 |
| `CosineAnnealingWarmRestarts` | 多次余弦下降并周期性回热 | 长训练、希望重复探索与精修 |
| `ReduceLROnPlateau` | `val_loss` 停滞才降 | 用模型表现当反馈，但更受指标噪声影响 |
| `OneCycleLR` | 一个 cycle 内先升后降 | 短周期充分探索，通常按 batch 推进 |

Cosine、Step、OneCycle 都是 **open-loop**：不观察模型表现，时间到了就调整；Plateau 是 **closed-loop**，需要 `scheduler.step(val_loss)`。开环更容易复现，闭环更贴合实际训练，但也更容易被验证指标噪声带着抖。

## 7. 训练与验证的边界

| | 训练 | 验证 |
|---|---|---|
| 模式 | `model.train()` | `model.eval()` |
| autograd | 开启，执行 `backward()` | `@torch.no_grad()` |
| 参数 | `optimizer.step()` 会修改 | 完全不修改 |
| Dropout / BN | 训练行为 | 推理行为 |
| DataLoader | 通常 `shuffle=True` | `shuffle=False` |

一句话：**`train()` 用来学，`eval()` + `no_grad()` 用来考。** 忘记 `eval()` 时，Dropout 仍会随机丢弃，BatchNorm 仍会使用当前 batch 的统计量，验证分数会偏低且抖动；忘记 `no_grad()` 虽不一定报错，却会白白保留计算图，增加显存和时间开销。

## 8. 最容易踩的坑

| 症状 | 原因 | 修法 |
|---|---|---|
| loss 先降后爆 | 忘了清梯度，梯度跨 batch 累加 | 每次 `backward()` 前 `zero_grad()` |
| val 分数低且随机波动 | 没切到 `eval()` | 验证函数第一行调用 `model.eval()` |
| 显存越来越大 | 把带计算图的 tensor 累计进 `total_loss` | 只累计 `loss.item() * batch_n` |
| 平均指标轻微失真 | 每个 batch 先平均，再对 batch 平均 | 按样本数累计，最后除以 `total_n` |
| lr 很快变成 0，训练卡死 | 基于 epoch 的 scheduler 被每个 batch 调用 | 在 epoch 末尾调用；`OneCycleLR` 除外 |
| Plateau 不降 lr | 没传验证指标 | 调用 `scheduler.step(val_loss)` |
| 断点续训后 lr 突然回到初值 | 只保存了 model / optimizer | 同时保存并恢复 `scheduler.state_dict()` |

指标和 loss 记账不只会影响打印。最后一个 batch 往往不满，直接平均 batch 指标会给它过高权重；只保存 loss tensor 则会把整张计算图一起留住，最后可能 OOM。

## 9. 记忆锚点

> **一条闭环：清、前、算、反、更。**
> **验证：切 `eval()`、关 autograd、别改参数。**
> **可变学习率：optimizer 用当前 $\eta_t$ 更新参数，scheduler 按时间表推进下一轮学习率。**
> **调度器不是自动涨点器：短训练先固定 lr；长训练后期压平，再考虑 cosine、step、Plateau 或 warmup。**

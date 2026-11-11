# 从零理解 Flow Matching：一条从噪声走到数据的概率路径

> 目标：读完以后，你能从概率路径和速度场推导出 Conditional Flow Matching（CFM）的训练目标，解释为什么回归的是 $\mathbf{x}_1-\mathbf{x}_0$，并独立写出一个能运行的二维 sampler。
>
> 约定：$\mathbf{x}_0\sim p_0$ 是简单噪声，$\mathbf{x}_1\sim p_{\mathrm{data}}$ 是数据；时间 $t\in[0,1]$ 从噪声走向数据。

## 0. 先抓住一件小事：样本会走路

生成模型最朴素的需求是：给它一堆独立的高斯噪声，它要长成数据分布。

普通神经网络一次映射就能做这件事，但它不必告诉我们“中间是怎么走的”。Flow matching 换了一种组织方式：先设计一条从噪声到数据的连续路径，再学一个**速度场**，告诉每个位置上的粒子下一步往哪里移动。

```text
noise x0             learned velocity v_theta             data x1
p0 (Gaussian)  -->  integrate  dx/dt = v_theta(x,t)  -->  p_data

t = 0                 every step increases t                t = 1
```

如果把数据点想成河流里的叶子，velocity field 就是整片水域的速度计。训练时看一眼“这批叶子从哪来，到哪去”；采样时只看当前位置和时间，不知道最终目的地，也能顺着流走完。

这个视角带来三个面向：

| 面向 | Flow matching 要回答的问题 | 本文对应的部分 |
|---|---|---|
| 路径 | 噪声到数据可以怎样走？ | §1–§2 |
| 学习 | 网络怎样从样本对上学会整片速度场？ | §3–§4 |
| 采样 | 有一个速度场后，怎样真正生成？ | §5 |

## 1. 概率路径：先修桥，再学车

直接从 $p_0$ 飞到 $p_1$ 太难：我们只见到两边的一批点，不知道中间该怎么安排。于是先构造一个**概率路径** $p_t$，让 $p_{t=0}=p_0$、$p_{t=1}=p_{\mathrm{data}}$。

先看一个常用的 Gaussian 路径模板：

$$
p_t(\mathbf{x}\mid\mathbf{x}_1)
=\mathcal N\!\left(\mathbf{x};\alpha_t\mathbf{x}_1,\sigma_t^2\mathbf{I}\right).
$$

这里 $\mathbf{x}_1$ 是固定的数据点，$\alpha_t$ 控制它保留多少信号，$\sigma_t$ 控制还剩多少噪声。常见的 diffusion 时间方向是

$$
\alpha_t=1-t,\qquad \sigma_t=t,
$$

这会让 $t=0$ 对应数据、$t=1$ 对应噪声。通常也可以用独立噪声 $\boldsymbol{\epsilon}\sim\mathcal N(0,I)$ 写成

$$
\mathbf{x}_t=\alpha_t\mathbf{x}_1+\sigma_t\boldsymbol{\epsilon}.
$$

为了保持“从噪声到数据”的方向，本文采用对称版本：

$$
\alpha_t=t,\qquad \sigma_t=1-t.
$$

若独立取 $\mathbf{x}_0\sim\mathcal N(0,I)$，同一族 Gaussian 路径也可以写成非常具体的线性插值：

$$
\boxed{
\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1,
\qquad \mathbf{x}_0\sim\mathcal N(\mathbf{0},\mathbf{I}),
\quad \mathbf{x}_1\sim p_{\mathrm{data}}.
}
$$

第 0 步是纯噪声，第 1 步是数据，中间是两者的加权混合。这个公式之所以重要，不只是因为简单，而是它对每个固定样本对 $(\mathbf{x}_0,\mathbf{x}_1)$ 都是确定的。

### 1.1 一个点怎么走

先只看一条路径。取

$$
\mathbf{x}_0=(-1,0),\qquad \mathbf{x}_1=(1,2).
$$

对 $t$ 求导：

$$
\mathbf{u}_t
=\frac{d\mathbf{x}_t}{dt}
=\mathbf{x}_1-\mathbf{x}_0.
$$

所以在这个最简单的 straight-line path 上，速度是一个常数：

$$
\mathbf{u}_t=(2,2)
\quad\text{for every }t\in[0,1].
$$

这是一辆不踩刹车、不拐弯的车：

```text
(-1,0) -- (+2,+2) -- (+2,+2) -- (+2,+2) --> (1,2)
  t=0                 t=.5                    t=1
```

样本不同，目标速度也不同。训练数据里某对 $(\mathbf{x}_0,\mathbf{x}_1)$ 的“金标准答案”，居然就是终点减起点。这个小小的观察是后面整个 CFM loss 的种子。

### 1.2 点会走路，不等于分布会正确变形

一个 subtle 却关键的转折：**路径的每一对端点都可以随意配对**。同样是噪声和高斯数据，它们可以配对成很直的路，也可以交叉、绕弯。

而且数据端是许多样本，噪声端也是许多样本。只修好了一条条路径，并不自动保证中间的 $p_t$ 就是想要的边缘分布。Flow matching 要做的是：让一整片速度场把整个 $p_0$ 推成 $p_1$。

## 2. 速度场：函数的连续版本

真实的目标速度场

$$
\mathbf{v}(\mathbf{x},t)
=\mathbb E[\mathbf{u}_t\mid \mathbf{x}_t=\mathbf{x}]
$$

表示：当粒子恰好出现在 $\mathbf{x}$、时间为 $t$ 时，平均该往哪里走。它不是“对某个特定样本的固定答案”，而是把许多可能路径在同一位置的速度平均起来。

连续性方程给出更底层的约束：

$$
\frac{\partial p_t}{\partial t}
+\nabla\cdot\bigl(p_t(\mathbf{x})\,\mathbf{v}(\mathbf{x},t)\bigr)=0.
$$

它不是额外要求，而是在表达一句朴素的话：**流不能凭空创造或消灭概率质量**。左下角看得见的粒子，必须沿着速度流出；某个区域新增的概率，必须来自别处流进来的粒子。

用不同角度重复一遍：

- 从物理看，velocity field 是一张随地点和时间变化的箭头地图。
- 从统计看，它把 $p_0$ 通量地搬运到 $p_1$。
- 从网络看，它只是输入 $(\mathbf{x},t)$、输出同样维度速度的 regression 模型。
- 从采样看，它不直接生成 $\mathbf{x}_1$，而是给一段 ODE 提供 slope。

## 3. 为什么不能直接回归目标速度

理论上最想要的 loss 是

$$
\mathcal L_{\mathrm{FM}}(\theta)
=\mathbb E_{t,\mathbf{x}\sim p_t}
\left[
\left\|\mathbf{v}_\theta(\mathbf{x},t)
-\mathbf{v}(\mathbf{x},t)\right\|^2
\right].
$$

问题在 $\mathbf{v}(\mathbf{x},t)$：它依赖整个边际分布，训练时我们只拿到有限样本，更不知道在某个位置有多少条隐藏路径经过。直接计算它，不可行。

Flow matching 的关键招法是改变监督对象。对每个采样对 $(\mathbf{x}_0,\mathbf{x}_1)$，我们不仅知道起点和终点，还知道直线路径在任意 $t$ 上的位置 $\mathbf{x}_t$ 和速度 $\mathbf{u}_t=\mathbf{x}_1-\mathbf{x}_0$。于是把不可见的 marginal target 换成可见的 conditional target：

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=\mathbb E_{t\sim\mathcal U[0,1],\,
\mathbf{x}_0\sim p_0,\,
\mathbf{x}_1\sim p_{\mathrm{data}}}
\left[
\left\|\mathbf{v}_\theta
\bigl((1-t)\mathbf{x}_0+t\mathbf{x}_1,t\bigr)
-(\mathbf{x}_1-\mathbf{x}_0)\right\|^2
\right].
}
$$

这里每个样本提供三元组

$$
(\mathbf{x}_t,\ t,\ \mathbf{x}_1-\mathbf{x}_0).
$$

网络学习输入位置和时间、输出对应速度。训练时 loss 不平凡是好事；具体数值并不是一个“越小就越像图”的分数，下面会专门拆开这个容易误读的点。

### 3.1 条件目标为什么能替掉条件分布

记条件随机变量 $\mathbf{y}=(\mathbf{x}_0,\mathbf{x}_1)$，条件速度为 $\mathbf{u}(\mathbf{y})=\mathbf{x}_1-\mathbf{x}_0$。条件分布下的平方回归，其最优预测正是条件均值：

$$
\mathbf{v}^*(\mathbf{x},t)
=\arg\min_{\mathbf{g}}
\mathbb E\left[\|\mathbf{g}(\mathbf{x},t)-\mathbf{u}\|^2
\mid\mathbf{x}_t=\mathbf{x}\right]
=\mathbb E[\mathbf{u}\mid\mathbf{x}_t=\mathbf{x}].
$$

这恰好就是前面定义的目标 marginal velocity。因此我们不需要显式知道 $\mathbf{v}(\mathbf{x},t)$，只要大量采样 $(\mathbf{x}_0,\mathbf{x}_1)$ 并对条件速度做 regression，最优解就会在条件均值意义下靠近它。

这里的“靠近”还有一个容易忽略的层次：**CFM loss 的 minimizer 与直接 regression 的 minimizer 匹配，并不意味着单个有限网络必然恢复完美的几何形状**。采样数、网络容量、时间条件、积分误差和 coupling 都会产生影响。

## 4. 从公式到训练代码

下面是一个能独立运行的二维 toy。它不是生产级图像模型，但把训练与采样所需的每个 tensor 形状都暴露出来。

```python
import math
import torch
from torch import nn

torch.manual_seed(0)


def two_moons(n, noise=0.06):
    n1 = (n + 1) // 2
    n2 = n - n1
    a1 = math.pi * torch.rand(n1)
    a2 = math.pi * torch.rand(n2)
    moon1 = torch.stack((a1.cos(), a1.sin()), dim=-1)
    moon2 = torch.stack((1 - a2.cos(), 0.5 - a2.sin()), dim=-1)
    x = torch.cat((moon1, moon2), dim=0)[:n]
    x = (x - torch.tensor([0.5, 0.25])) * 0.85
    return x + noise * torch.randn_like(x)


class TimeMLP(nn.Module):
    def __init__(self, time_dim=32, hidden=128):
        super().__init__()
        self.time_net = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.velocity_net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        time_features = self.time_net(t[:, None])
        return self.velocity_net(torch.cat((x, time_features), dim=-1))


model = TimeMLP()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1, 3001):
    x1 = two_moons(256)                    # data
    x0 = torch.randn_like(x1)             # noise
    t = torch.rand(len(x1))               # time
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    target_velocity = x1 - x0

    predicted_velocity = model(xt, t)
    loss = (predicted_velocity - target_velocity).square().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step == 1 or step % 500 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")
```

这段循环故意没有 `DataLoader`，因为它需要同步抽两类随机变量：噪声 $\mathbf{x}_0$ 和数据 $\mathbf{x}_1$。在真实图像任务中，你通常从图像 loader 取 $\mathbf{x}_1$，在 GPU 上直接采样同形状的 $\mathbf{x}_0$ 和 $t$。

### 4.1 形状走一遍

设 batch size 为 $B$，二维向量维度为 $2$：

| tensor | 形状 | 含义 |
|---|---|---|
| `x1` | `[B, 2]` | batch 个数据点 |
| `x0` | `[B, 2]` | batch 个高斯噪声点 |
| `t` | `[B]` | 每个样本独立时间 |
| `t[:, None]` | `[B, 1]` | 便于广播 |
| `xt` | `[B, 2]` | 每个样本路径上的点 |
| `target_velocity` | `[B, 2]` | 每个点各自向量速度，不是二维标量 |
| `predicted_velocity` | `[B, 2]` | 网络输出 |

最容易错的是把“二维空间”的速度误做成一个标量。数据有 $d$ 维，velocity field 的输出也必须有 $d$ 维；图像 latent 是 `[B, C, H, W]`，速度就同样是 `[B, C, H, W]`。

另一处常见错误是把时间做成一个全局 `t = rand(1)`。做 conditional flow matching 时，batch 中每个样本可以拥有不同时间；用 `[B]` 的时间更有利于网络学会整条时间轴。

## 5. 采样：把 velocity field 积回去

网络给出的是局部斜率，不是最终答案。采样要从噪声出发，按

$$
\frac{d\mathbf{x}_t}{dt}=\mathbf{v}_\theta(\mathbf{x}_t,t)
$$

一直走到 $t=1$。最朴素的 Euler ODE solver 是：

$$
\mathbf{x}_{t+\Delta t}
\approx
\mathbf{x}_t+\Delta t\,\mathbf{v}_\theta(\mathbf{x}_t,t).
$$

用代码把这句话逐行写出来：

```python
@torch.no_grad()
def sample(model, n=4000, steps=100):
    x = torch.randn(n, 2)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + dt * model(x, t)

    return x


generated = sample(model)

print("generated mean:", generated.mean(dim=0).tolist())
print("generated std: ", generated.std(dim=0).tolist())
```

这段代码回答了一个核心问题：为什么训练时要输入时间？因为真实采样不是从任意位置一次性猜答案，而是沿着不断移动的位置、在不同的 $t$ 上重复调用 velocity field。

如果把 Euler 的步数减少，帧和帧之间的曲线就变成一条条小直线，轨迹会更粗；增加步数会让折线逐渐贴近连续轨迹。ODE solver 越精细，采样成本也越高。这里先只掌握 Euler，不展开 Runge–Kutta 或高阶 solver。

## 6. 结果怎样判读：统计量不是全部

用上面的网络短训练时，loss 可以下降、生成分布的一阶和二阶统计量也会接近数据，但二维小网络仍可能把双月牙糊成一片云。可以把它当作三个层次：

| 检查 | 怎么看 | 不等于什么 |
|---|---|---|
| 是否接线正确 | 训练 loss 能前降、生成均值和标准差接近数据 | 不说明形状对 |
| 分布是否大致接近 | 分位数、二维直方图、nearest-neighbor 距离 | 不说明视觉质量够好 |
| 几何是否清楚 | 画数据与生成点的二维散点图 | 不能只靠单一平均 loss 判断 |

这也是为什么“CFM loss 从 0.94 降到 0.82”不能直接当成“画质变好了”。以当前 straight-line path 为例，想要的条件速度是 $\mathbf{x}_1-\mathbf{x}_0$；同一空间位置的路径可能有不同速度方向，网络给出的是它们的平均，单个样本的最佳答案与平均预测之间天然存在方差。

一个简单固定的 bad case 值得记住：如果噪声点和数据点没有结构地配对，许多路径会交叉；即使 $t=0$ 和 $t=1$ 的边缘分布都正确，中间仍在“搅拌”很多可能性。这就是 coupling 与训练预算会影响困难程度的地方，也是 Rectified Flow 想进一步改善的对象。

## 7. 和 Rectified Flow 的关系

上面的路径是

$$
\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1.
$$

如果固定一对 $(\mathbf{x}_0,\mathbf{x}_1)$，路径本身是直的，但大量配对起来的 global flow 未必直。Rectified Flow 也使用回归速度场与 ODE 采样，强调通过重新配对与 refinement，让学到的路径更接近直线，从而用更少步数采样。

可以把两者放在同一张地图上：

| 问题 | Flow matching 的视角 | Rectified Flow 的具体做法 |
|---|---|---|
| 目标对象 | 从噪声到数据的 probability path | 同样有 probability path 与 velocity field |
| 训练条件 | 条件速度可以回归 | 同样可以做直线插值的 velocity regression |
| 如何采样 | 积分 ODE | 同样积分 ODE |
| 如何变快 | 网络、solver、路径都有关 | 重配对可以做 rectification，让轨迹更直 |

所以不要把它们理解成“一个是 flow，一个不是”。更准确的说法是：Rectified Flow 是 flow 这一思想的一种具体配方，重点在于用直线化与重配对改善少步采样。

## 8. 和其他生成路线的接缝

Diffusion 与 flow matching 都学“时间上的变化”，但参数化和连接方式不同：

| 视角 | Diffusion（常见形式） | Flow matching |
|---|---|---|
| 基础过程 | 多个小噪声步骤，前向加噪、反向去噪 | 连续 ODE 的速度场 |
| 训练目标 | 常预测噪声、score 或与之等价的量 | 直接回归速度 |
| 采样 | 反向 diffusion sampler / probability-flow ODE | 积分学到的 ODE |
| 核心直觉 | 逐步剥去噪声 | 沿速度场搬运概率质量 |

它们不是非黑即白的两个世界。某些扩散概率流 ODE 与 flow 的连续时间视角可以互相转换；真正决定实现差异的是路径、目标、时间采样和 solver。

## 9. 常见坑与调试顺序

先调数据路径，再调网络；先验证小公式，再放大模型。

| 症状 | 优先检查 |
|---|---|
| loss 几乎不变 | `t` 是否送进网络；`xt` 是否正确插值；目标速度是否形状相同 |
| 生成点整体跑到一个地方 | 采样时 `t` 是否逐步增加；是否误对数据额外做一次插值 |
| 均值和方差像数据，但结构模糊 | 网络容量、训练步数、coupling 与 solver，不应只盯 loss 数字 |
| 每个 batch 变成相同生成 | 检查噪声是否每个样本独立，sample 中是否错误地复用了同一个 `x` |
| 增加步数反而更怪 | 检查网络训练区间是否覆盖 `[0,1]`，以及 solver 是否真的在增加时间 |

一个可靠的实验纪律：

1. 先打印 `x0`、`x1`、`xt`、`target_velocity` 的形状，确认没有悄悄广播。
2. 用极小的 network 与 batch 过拟合 32 个固定样本，确认 tricks 之外的基本管道能跑。
3. 再检查 $t=0$ 的生成分布是否仍像噪声，$t=1$ 是否进入数据支撑集。
4. 最后才比较不同 coupling、solver 与训练预算；不要一边修 bug 一边调五个超参数。

## 10. 一页记忆模型

> **Flow matching 用一个 velocity field 搬运概率质量。**
>
> **路径**：$\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1$。
>
> **目标速度**：沿这条条件直线，答案是 $\mathbf{x}_1-\mathbf{x}_0$。
>
> **训练**：从 $(\mathbf{x}_0,\mathbf{x}_1)$ 造出 $\mathbf{x}_t$，监督 $\mathbf{v}_\theta(\mathbf{x}_t,t)$。
>
> **采样**：从 $\mathbf{x}_0\sim\mathcal N(0,I)$ 出发，沿 $d\mathbf{x}/dt=\mathbf{v}_\theta(\mathbf{x},t)$ 积到 $t=1$。
>
> **最重要的一句**：学到的不是“从噪声直接跳到某张图”的映射，而是一张让整片概率分布沿着时间流动的方向图。

自检一下：如果我们把 $\mathbf{x}_0$ 和 $\mathbf{x}_1$ 的顺序互换，直线插值会怎样变化？答案是路径方向反转，条件速度变成 $\mathbf{x}_0-\mathbf{x}_1$，ODE 的采样方向也必须相应调整。Flow matching 可以有多种时间约定，但路径、目标速度与采样三者的方向必须一致。

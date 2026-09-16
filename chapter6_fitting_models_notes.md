# 第 6 章：拟合模型（Fitting Models）

> 书：《Understanding Deep Learning》Ch.6  
> 前置：第 5 章把“拟合得好不好”压成标量 \(L[\phi]\)。  
> 本章：在参数空间里怎样一步步把这个标量变小？  
> 后续：第 7 章才回答“梯度从哪来、从哪出发”；本章假定 \(\nabla_\phi L\) 能算。

---

## 0. 标量已经在手，还不会走

第 3–4 章：网络是一族函数。第 5 章：损失函数给出一个数。训练在这一章变成一句极短的话：

\[
\hat\phi=\arg\min_\phi L[\phi].
\]

标准做法是迭代，不是闭式求根：

```text
启发式初始化 φ
        │
        ▼
算 ∇_φ L          ← 本章假设能算；第 7 章讲怎么算
        │
        ▼
沿使损失下降的方向改 φ    ← GD / SGD / Momentum / Adam
        │
        └──────── 重复
```

母问题：

> **拿到梯度之后，参数该怎么更新，才有机会走到低损失，而不是在曲面上乱弹？**

调查链：

```text
梯度指向最陡上山方向 → 减号就是下山（GD）
        │
        ▼
线性+MSE 是凸的，随便下山都能到
非线性损失通常非凸：局部最小、鞍点，终点依赖初始化
        │
        ▼
全数据梯度太贵，也太“干净” → SGD 用 batch 噪声
        │
        ▼
方向来回摆 → Momentum 把历史方向做成惯性
        │
        ▼
不同坐标曲率差很多 → Adam 按坐标自适应步长
        │
        ▼
带 L2 时 Adam 会扭曲衰减 → AdamW 把衰减解耦
```

本章方法对任意可微模型都适用。神经网络特有的“怎么高效算 \(\nabla L\)”和“初始化坏了激活会炸”，留给第 7 章。

---

## 1. 梯度下降：减号是下山

参数写成向量 \(\phi=[\phi_0,\ldots,\phi_N]^\top\)。每一步两件事：先算梯度

\[
\frac{\partial L}{\partial\phi}
=
\begin{bmatrix}
\partial L/\partial\phi_0\\
\vdots\\
\partial L/\partial\phi_N
\end{bmatrix},
\]

再更新

\[
\phi \leftarrow \phi - \alpha\cdot\frac{\partial L}{\partial\phi}.
\]

梯度指向最陡**上山**的方向。减号把它翻成下山。\(\alpha>0\) 是步长：固定时叫 learning rate；也可以在该方向上试几个 \(\alpha\)，叫 line search。最小值附近曲面变平，梯度 \(\approx 0\)，参数几乎不动。实践中常盯梯度模长，太小就停。

**两参数走一步。** 取 \(\phi=(\phi_0,\phi_1)^\top=(1.0,\,2.0)^\top\)，假装刚算出

\[
\nabla_\phi L=\begin{bmatrix} 0.4 \\ -0.8 \end{bmatrix},\qquad \alpha=0.1.
\]

下山一步：

\[
\phi'
=
\begin{bmatrix}1.0\\2.0\end{bmatrix}
-
0.1\begin{bmatrix}0.4\\-0.8\end{bmatrix}
=
\begin{bmatrix}1.0-0.04\\2.0+0.08\end{bmatrix}
=
\begin{bmatrix}0.96\\2.08\end{bmatrix}.
\]

\(\phi_0\) 沿梯度正方向减小；\(\phi_1\) 的梯度是负的，减号把它变成**增大**。步长向量是 \(\alpha\nabla L=(0.04,-0.08)\)，欧氏长度 \(0.0894\)，不是 \(\alpha\) 本身。\(\alpha\) 只管“沿这个方向走多远”，不保证每维都动 \(0.1\)。

先看一个会成功的例子，再看它为什么不能代表神经网络。

线性回归 \(f[x,\phi]=\phi_0+\phi_1 x\)，平方损失

\[
L[\phi]
=
\sum_{i=1}^{I}(\phi_0+\phi_1 x_i-y_i)^2.
\]

总梯度是各样本贡献之和。单样本：

\[
\frac{\partial\ell_i}{\partial\phi}
=
\begin{bmatrix}
2(\phi_0+\phi_1 x_i-y_i)\\
2x_i(\phi_0+\phi_1 x_i-y_i)
\end{bmatrix}.
\]

线性模型加平方损失是 **convex**：任意两点的连弦都在曲面上方。凸意味着没有“看起来像谷底、其实旁边还有更深的坑”——从哪出发一直下山，都能到同一个全局最小。训练**不会因为初始化而失败**。

神经网络的损失通常 **non-convex**。书用两参数 Gabor 模型把曲面画出来：

\[
f[x,\phi]
=
\sin(\phi_0+0.06\cdot\phi_1 x)
\cdot
\exp\Bigl(-\frac{(\phi_0+0.06\cdot\phi_1 x)^2}{32}\Bigr).
\]

\(\phi_0\) 移动波形中心，\(\phi_1>0\) 沿 \(x\) 拉伸。平方损失的曲面上会出现：

| 对象 | 含义 | 下山时会怎样 |
|------|------|----------------|
| Global minimum | 全局最低 | 理想终点 |
| Local minimum | 梯度为 0，四周都更高，但不是全局最低 | GD 会停，不知别处是否更好 |
| Saddle | 梯度为 0；有的方向像谷，有的像脊 | 附近很平，容易误判“已经收敛” |

结论先说清楚，免得后面把 SGD 神话化：

> **非线性模型上，标准梯度下降的终点强烈依赖初始化。没有实用办法保证全局最优。**

全数据 GD 还有两个更具体的麻烦：每步要用完全部样本，贵；梯度太干净，一旦掉进错误山谷或贴在鞍点上，没有力量把它抖出去。下一节把随机性请进来，不是为了“看起来更现代”，是为了这两件事。

---

## 2. SGD：用 batch 噪声换便宜，也换逃逸

高维网络有数百万参数，穷举或无限随机重启都不现实。SGD 每次只用训练集的一个随机子集——minibatch \(B_t\)——来估梯度：

\[
\phi_{t+1}
\leftarrow
\phi_t
-
\alpha
\sum_{i\in B_t}
\frac{\partial\ell_i[\phi_t]}{\partial\phi}.
\]

| 词 | 含义 |
|----|------|
| Batch \(B_t\) | 这一步用到的样本下标 |
| Epoch | 不放回地走完整个训练集一遍 |
| Full-batch GD | batch = 全体数据，就是上一节 |
| Learning rate \(\alpha\) | 固定或按 schedule 衰减 |

实现上通常 **without replacement** 扫完再打乱，保证每个 epoch 里样本贡献次数大致相同。

**同一组数：full-batch 对 minibatch。** 三条样本，当前 \(\phi\) 上单样本梯度（标量参数，方便看账）是

\[
g^{(1)}=2.0,\quad g^{(2)}=-1.0,\quad g^{(3)}=0.4.
\]

全数据一步用平均梯度（有人用求和，差一个常数，可并进 \(\alpha\)）：

\[
\bar g=\frac{2.0-1.0+0.4}{3}=0.4667,\qquad
\phi\leftarrow\phi-\alpha\bar g.
\]

\(\alpha=0.1\) 时全数据位移 \(-0.04667\)。

Batch \(\{1,2\}\)：\(\bar g_B=(2-1)/2=0.5\)，位移 \(-0.05\)。  
Batch \(\{3\}\)：\(\bar g_B=0.4\)，位移 \(-0.04\)。

```text
full batch     :  always  -0.04667   (clean, expensive)
batch {1,2}    :          -0.05      (a bit too far)
batch {3}      :          -0.04      (a bit too short)
```

噪声是零均值的：三个等可能 batch 的位移平均仍是全数据那一步。贵的那一步每步看完全部 \(I\) 条；便宜的那一步只看 \(|B|\) 条，方向会抖，正是上一节要的“不太干净”。

另一种读法有时更清楚：每个 batch 定义一张稍微不同的损失曲面。SGD 是在**不断变化的损失**上做确定性下降。对当前 batch 是下山，对全局损失却可能暂时上山——这正是跳出错误山谷的机制。期望梯度和期望损失仍与全数据一致，噪声是零均值的，不是乱走。

它同时办成几件事：每步至少改善当前 batch；不放回扫描所以贡献公平；子集算梯度更省；原则上可逃局部最小、降低贴死在鞍点上的概率；实践中常走到泛化更好的区域（第 9 章会把这叫 implicit regularization）。

“收敛”在传统意义上不必严格发生。靠近好最小时，各 batch 都拟合得差不多，梯度都小，参数几乎不动。常用 learning rate schedule：早期 \(\alpha\) 大，负责探索和跳山谷；后期变小，负责精调。

SGD 的噪声是各向同性的乱抖。山谷里左右来回摆时，我们其实希望横着的分量互相抵消、顺着谷底的分量留着。这就不是再加噪声，而是给更新加惯性。

---

## 3. Momentum：方向一致就加速，来回摆就对消

用“当前 batch 梯度”和“上一步方向”的加权组合：

\[
\begin{aligned}
m_{t+1}
&\leftarrow
\beta m_t
+(1-\beta)
\sum_{i\in B_t}
\frac{\partial\ell_i[\phi_t]}{\partial\phi},\\
\phi_{t+1}
&\leftarrow
\phi_t-\alpha\, m_{t+1}.
\end{aligned}
\]

\(m_t\) 是动量向量，\(\beta\in[0,1)\) 控制历史保留多少。递归展开后，\(m\) 是所有历史梯度的无穷加权和，越久远权重越小。

连续多步方向一致时，有效步长变大，加速穿过缓坡。方向在峡谷里左右摆时，横向分量对消，轨迹更直。惯性不是永远向前冲：它过滤的是高频抖动，留下低频的一致方向。

**两步手算。** \(\beta=0.9\)，\(m_0=0\)，\(\alpha=0.1\)，两步梯度同号：\(g_0=1.0\)，\(g_1=1.0\)。

\[
\begin{aligned}
m_1&=0.9\cdot 0+(1-0.9)\cdot 1.0=0.10,\\
\phi_1&=\phi_0-0.1\cdot 0.10=\phi_0-0.010,\\
m_2&=0.9\cdot 0.10+0.1\cdot 1.0=0.19,\\
\phi_2&=\phi_1-0.1\cdot 0.19=\phi_0-0.029.
\end{aligned}
\]

Vanilla GD 两步各走 \(0.1\times 1=0.10\)，合计 \(0.20\)。动量前两步合计只走 \(0.029\)——因为 \(m\) 从 0 暖机。若方向一直是 \(+1\)，\(m\) 会趋近 \(1\)，有效步长趋近 \(\alpha\)，暖机过后才“加速”。

再换成峡谷：\(g_0=+1\)，\(g_1=-1\)（左右摆）。

\[
m_1=0.10,\quad m_2=0.9\cdot 0.10+0.1\cdot(-1)=-0.01.
\]

横向几乎对消。Vanilla 则是 \(+0.10\) 再 \(-0.10\)，原地踏步但振幅大。动量把高频左右摆滤掉，留下接近 0 的净位移。

Nesterov 把“先走再看”说得更明确。普通动量在当前位置算梯度，再叠惯性。Nesterov 先沿动量预瞄到 \(\phi_t-\alpha\beta m_t\)，**在预瞄点**算梯度，用这块梯度修正路径：

\[
m_{t+1}
\leftarrow
\beta m_t
+(1-\beta)
\sum_{i\in B_t}
\frac{\partial\ell_i[\phi_t-\alpha\beta m_t]}{\partial\phi}.
\]

直观：动量先预测下一步会到哪，梯度再在那个未来点上做修正，少走一点弯路。

动量让“同一方向”的坐标走得更顺。它还没回答另一个几何问题：有的坐标很陡、有的很缓，一个全局 \(\alpha\) 无法同时伺候两者。

---

## 4. Adam：每个坐标自己有一把尺子

损失在不同参数方向上曲率可以差几个数量级。\(\alpha\) 按陡的方向设，缓的方向爬不动；按缓的方向设，陡的方向过冲。深层网络里，不同层的梯度尺度差尤其大——这和第 7 章的反传尺度是同一件事。

先看一个极端的按坐标归一化。记 \(m\) 为梯度、\(v\) 为逐元平方：

\[
\phi
\leftarrow
\phi
-
\alpha\cdot\frac{m}{\sqrt{v}+\epsilon}.
\]

除法和开方都是 pointwise。结果几乎只保留每个坐标的符号，每维大约走固定距离 \(\alpha\)。问题是最小值附近会来回振荡，停不精确。

Adam 对 \(m\) 和 \(v\) 都加动量，再做偏差修正——因为初期历史全是 0，原始估计偏小：

\[
\begin{aligned}
m_{t+1}
&\leftarrow
\beta m_t+(1-\beta)g_t,\\
v_{t+1}
&\leftarrow
\gamma v_t+(1-\gamma)g_t^{\odot 2},\\
\tilde m_{t+1}
&=
\frac{m_{t+1}}{1-\beta^{t+1}},
\qquad
\tilde v_{t+1}
=
\frac{v_{t+1}}{1-\gamma^{t+1}},\\
\phi_{t+1}
&\leftarrow
\phi_t
-
\alpha\cdot\frac{\tilde m_{t+1}}{\sqrt{\tilde v_{t+1}}+\epsilon}.
\end{aligned}
\]

随机版里 \(g_t\) 来自 mini-batch。谱系是 AdaGrad → RMSProp / AdaDelta → Adam（动量 + 自适应二阶矩）。它在深度学习里常用，是因为：各向异性曲率下更稳；能平衡不同层的更新尺度；对初始学习率相对不敏感，初期进展快。SGD 与 Adam 谁泛化更好有过争论；仔细调参后 Adam 往往能追上 SGD，并且收敛更快。这不是定理，是经验。

**第一步偏差修正（tiny numbers）。** \(\beta=0.9\)，\(\gamma=0.999\)，\(m_0=v_0=0\)，\(g_0=0.5\)，\(\alpha=0.001\)，\(\epsilon=0\)（手算先关掉）。

\[
\begin{aligned}
m_1&=0.1\cdot 0.5=0.05,\\
v_1&=0.001\cdot 0.25=0.00025.
\end{aligned}
\]

若不修正，步长分子是 \(0.05\)，分母 \(\sqrt{0.00025}\approx 0.01581\)，比值 \(\approx 3.16\)——被初始化成 0 的 EMA 压扁了 \(m\)，却几乎没压扁第一步的 \(v\)（\(1-\gamma\) 极小），尺度乱。修正：

\[
\tilde m_1=\frac{0.05}{1-0.9}=0.5,
\qquad
\tilde v_1=\frac{0.00025}{1-0.999}=0.25,
\qquad
\frac{\tilde m_1}{\sqrt{\tilde v_1}}=\frac{0.5}{0.5}=1.
\]

第一步等价于“只看符号、步长 \(\alpha\)”。偏差修正修的就是 **EMA 从全零起步造成的低估**，不是修梯度算错。

带显式 L2 时，还有一个 Adam 自己制造的坑。对 SGD，L2 penalty 和 weight decay 几乎是同一件事；对 Adam 不是。

| 写法 | 做法 | 后果 |
|------|------|------|
| Adam + L2（耦合） | \(g=\nabla L+\lambda\phi\)，再走 \(m/\sqrt{v}\) | 有效衰减 \(\propto 1/\sqrt{v}\)，正则被梯度尺度扭曲 |
| AdamW（解耦） | \(m,v\) 只用 \(\nabla L\)；额外 \(\phi\leftarrow\phi-\alpha\lambda\phi\) | 按比例缩权重，衰减不再被 \(\sqrt{v}\) 打折 |

误解常写成“参数大所以衰减慢”。真正被打折的是 **\(\sqrt{v}\) 大的那些维**。完整手算和 `adam_vs_adamw_demo.py` 对照见 `adam_vs_adamw_guide.md`。第 9 章会从 regularization 再碰到这件事。

**一维对照（数字）。** 当前 \(W=2.0\)，数据梯度 \(\nabla L=0.5\)，\(\lambda=0.1\)，Adam 的自适应因子假装已经暖机成 \(P=1/\sqrt{v}=0.25\)（这维最近很陡，\(\sqrt{v}\) 大），\(\alpha=0.1\)。

Adam+L2 把 \(2\lambda W\) 塞进梯度再乘 \(P\)：

\[
g=0.5+2\cdot 0.1\cdot 2.0=0.9,
\qquad
W\leftarrow 2.0-0.1\cdot 0.25\cdot 0.9=2.0-0.0225=1.9775.
\]

其中来自衰减的位移只有 \(0.1\cdot 0.25\cdot 0.4=0.010\)。同一维若 \(\sqrt{v}\) 很小、\(P=2\)，衰减位移会变成 \(0.1\cdot 2\cdot 0.4=0.08\)。**陡的维衰减被打折。**

AdamW：\(m,v\) 只用 \(0.5\)，衰减单独乘 \(W\)：

\[
W\leftarrow 2.0-0.1\cdot 0.25\cdot 0.5-0.1\cdot 0.1\cdot 2.0
=2.0-0.0125-0.020=1.9675.
\]

衰减 \(0.020\) 不经过 \(P\)。两维梯度尺度差十倍时，Adam+L2 会对它们施加差十倍的“L2”；AdamW 仍按同一比例缩 \(W\)。

---

## 5. 这些都是超参数，不是 \(\phi\)

优化算法、batch size、learning rate 及 schedule、动量系数 \(\beta,\gamma\)，全部是 hyperparameters。它们不在 \(\nabla L\) 里被学习，要靠 search 来选——第 8 章会把验证集请进来，否则你会在训练集上把 \(\alpha\) 调到“看起来最小”，再在新数据上后悔。

一句话对照：

| 算法 | 它补上的缺口 |
|------|----------------|
| GD | 全数据、最陡下山 |
| SGD | 更便宜，带噪声，可逃局部最小 / 鞍点 |
| Momentum | 一致方向加速，峡谷振荡对消 |
| Adam | 按坐标自适应，方向更均衡 |
| AdamW | Adam 上把 weight decay 解耦 |

脚注：第 20 章会再问“局部最小和鞍点在深度学习里到底有多致命”——实践中深度网往往比二维 Gabor 图看起来好训。本章先把走法立住即可。

---

## 6. 和 `gradient_descent.py` 对一下

`ManualGradientDescent` 是 **full-batch 梯度下降 + 手工链式法则**：更新规则属于本章，梯度怎么拆开属于第 7 章。

| 书中对象 | 代码里 |
|----------|--------|
| \(f[x,\phi]\) | 三次多项式 \(a+bx+cx^2+dx^3\) |
| \(L[\phi]\) | MSE：`mean((y_pred-y_label)**2)` |
| \(\partial L/\partial f\) | `loss_2_f = 2*(y_pred-y_label)` |
| 更新 \(\phi\leftarrow\phi-\alpha\nabla L\) | 各参数分别减 |
| 全数据一步 | 每 epoch 用全部 1000 点 |

代码里对不同参数用了不同有效学习率倍数：

```python
parameter_a -= 1e1 * lr * l_2_a
parameter_b -= 1e3 * lr * l_2_b
parameter_c -= lr * l_2_c
parameter_d -= lr * l_2_d
```

这正是 Adam 要自动办的事。\(\partial f/\partial d=x^3\)，\(|x|\) 到 10 时量级可达 \(10^3\)，而 \(\partial f/\partial a=1\)。手工按坐标调 \(\alpha\)，Adam 用 \(\sqrt{v}\) 做自动按坐标归一化。

若要对齐本章，可以试：同一 \(\alpha\) 的 vanilla GD（看 \(c,d\) 是否难训）；mini-batch SGD；Momentum / Adam；对照曲线——SGD 更抖，Momentum 更平滑，Adam 初期更快。

---

## 7. 自测

1. 梯度指向哪，为什么更新要减号？  
2. 从平方损失推出线性回归单样本梯度。  
3. 凸损失为什么“从哪出发都能到”？Gabor 曲面上 local min 和 saddle 各会骗过 GD 什么？  
4. 固定学习率的非随机 GD 能否逃出局部最小？  
5. 数据 100 条、batch=20、1000 次迭代，训了几个 epoch？  
6. 把动量 \(m_t\) 展开成历史梯度的无穷加权和，直观上它在过滤什么？  
7. Nesterov 相对普通动量，梯度是在哪一点算的？  
8. 为什么一个全局 \(\alpha\) 伺候不了又陡又缓的坐标？Adam 的 \(\sqrt{v}\) 做了什么？  
9. 偏差修正 \(1-\beta^{t+1}\) 在修哪一件初始化造成的事？  
10. Adam+L2 与 AdamW 差在哪？被打折的是“大参数”还是“大 \(\sqrt{v}\) 的维”？  
11. Learning rate、batch size 为什么不能靠最小化训练损失来选？这把哪一章的问题提前预告了？

---

## 8. 合上书再看一眼

第 5 章给出下山的高度计 \(L[\phi]\)。本章给出走法：顺着 \(-\nabla L\) 走。线性加 MSE 的世界是凸的，走法几乎无关紧要；神经网络的世界有坑有鞍，全数据 GD 又贵又干净，于是 SGD 用 batch 噪声换便宜和逃逸，Momentum 把一致方向做成惯性，Adam 给每个坐标一把自己的尺子，AdamW 再把权重衰减从那把尺子里拆出来。

这些更新都假定梯度已经在手。下一章问两件神经网络特有的事：怎样不高价地算出 \(\nabla_\phi L\)，以及从怎样的 \(\phi_0\) 出发，前向激活和后向梯度才不会逐层消失或爆炸。

# 第 7 章：梯度与初始化（Gradients and Initialization）

> 书：《Understanding Deep Learning》Ch.7  
> 前置：第 6 章假定 \(\nabla_\phi L\) 能算，并给出 GD / SGD / Adam。  
> 本章：神经网络里，这个梯度怎样不高价地算出来？从怎样的 \(\phi_0\) 出发，信号才不会逐层消失或爆炸？  
> 后续：第 8 章衡量走得怎样；第 11 章会用 residual / normalization 再帮深度网络把尺度稳住。

---

## 0. 会走还不够：要知道往哪走，也要从能走的地方出发

第 6 章的循环是：

```text
初始化 φ → 算 ∇_φ L → 按规则更新 → 重复
```

对神经网络，这两头都变得特别：

| 缺口 | 为什么难 | 本章的答案 |
|------|----------|------------|
| 算梯度 | 参数可以到 \(10^{12}\) 量级，每步、每个 batch 样本都要 \(\partial\ell/\partial\beta_k,\partial\ell/\partial\Omega_k\) | 反向传播：按计算图链式法则，复用中间量 |
| 出发点 | 初始权重方差错一点，前向激活和后向梯度会逐层指数消失或爆炸 | He 初始化等：让每层方差稳住 |

```text
第 5 章：优化什么（L）
第 6 章：怎么走（更新规则）
第 7 章：往哪走（反传）+ 从哪出发（初始化）
第 8 章：走得怎样（性能与泛化）
```

母问题：

> **怎样让一次前向加一次后向，就得到所有参数的梯度；以及怎样选初始方差，让这两次传播的尺度都不崩？**

调查链：

```text
权重对损失的影响 ∝ 源激活 → 前向必须缓存 h
        │
        ▼
参数改动向后涟漪 → 从损失往回走，复用 ∂ℓ/∂f
        │
        ▼
标量玩具看清单；向量版只是同一套规则换成矩阵
        │
        ▼
框架里的 loss.backward() 就是 reverse-mode AD
        │
        ▼
即使梯度算对，σ_Ω 过小/过大 → 消失/爆炸
        │
        ▼
ReLU 下要方差稳定 → He：σ² = 2 / fan-in
```

先把网络写成后面要用的符号。三层隐层：

\[
\begin{aligned}
\mathbf h_1 &= a[\boldsymbol\beta_0+\boldsymbol\Omega_0\mathbf x],\\
\mathbf h_2 &= a[\boldsymbol\beta_1+\boldsymbol\Omega_1\mathbf h_1],\\
\mathbf h_3 &= a[\boldsymbol\beta_2+\boldsymbol\Omega_2\mathbf h_2],\\
f[\mathbf x,\phi] &= \boldsymbol\beta_3+\boldsymbol\Omega_3\mathbf h_3.
\end{aligned}
\]

\(a[\cdot]\) 本章主用 ReLU。参数 \(\phi=\{\boldsymbol\beta_k,\boldsymbol\Omega_k\}\)。总损失 \(L[\phi]=\sum_i\ell_i\)，例如平方损失。SGD 要的就是每个层、batch 内每个样本上的 \(\partial\ell_i/\partial\boldsymbol\beta_k\) 和 \(\partial\ell_i/\partial\boldsymbol\Omega_k\)。

若把 \(\partial\ell/\partial\omega_0\) 从输出一路手工展开，会得到一张极长、大量重复的式子。反传要消灭的就是这份重复。

---

## 1. 两观察：为什么必须前向存、后向走

**观察 1。** 每个权重把源隐单元的激活乘到目标隐单元。因此 \(\partial\ell/\partial\omega\) 与源激活成正比。源激活只在前向出现，所以 **forward pass 必须把各层 \(h\)（以及 pre-activation \(f\)）存下来**。没有这份缓存，后向没有“乘在哪根绳子上”的那根绳子。

**观察 2。** 改一层参数，会改该层隐单元，再改下一层，一直改到输出和损失。后面层已经算好的“损失对该层输出有多敏感”，前面层可以原样复用。所以梯度从后往前走：

```text
forward:   x → h1 → h2 → h3 → f → ℓ
           存储各层 f_k, h_k

backward:  ∂ℓ/∂f ← ∂ℓ/∂h ← … 从输出端往输入端
           复用已算好的 ∂ℓ/∂(更后层)
```

反传 = **按计算图链式法则 + 共享中间结果**。不是一种新的微分学。下一节用标量玩具把链条写到能手算，然后再升维。

---

## 2. 标量玩具：一次看清单

逐层不同激活的标量网络：

\[
f
=
\beta_3+\omega_3\cdot
\cos\Bigl(
\beta_2+\omega_2\cdot
\exp\bigl(
\beta_1+\omega_1\cdot
\sin[\beta_0+\omega_0 x]
\bigr)
\Bigr).
\]

\(\ell_i=(f-y_i)^2\)。前向先拆成中间量并存储：

\[
\begin{aligned}
f_0&=\beta_0+\omega_0 x_i,\\
h_1&=\sin[f_0],\\
f_1&=\beta_1+\omega_1 h_1,\\
h_2&=\exp[f_1],\\
f_2&=\beta_2+\omega_2 h_2,\\
h_3&=\cos[f_2],\\
f_3&=\beta_3+\omega_3 h_3,\\
\ell_i&=(f_3-y_i)^2.
\end{aligned}
\]

后向从损失开始。\(\partial\ell_i/\partial f_3=2(f_3-y_i)\)。再

\[
\frac{\partial\ell_i}{\partial h_3}
=
\omega_3\cdot\frac{\partial\ell_i}{\partial f_3}.
\]

更前一层把已经算好的部分放进括号里复用：

\[
\frac{\partial\ell_i}{\partial f_{k-1}}
=
\frac{\partial h_k}{\partial f_{k-1}}
\cdot
\frac{\partial f_k}{\partial h_k}
\cdot
\frac{\partial\ell_i}{\partial f_k}.
\]

参数的导数只是再乘一个局部因子：

\[
\frac{\partial\ell_i}{\partial\beta_k}
=
\frac{\partial f_k}{\partial\beta_k}\cdot\frac{\partial\ell_i}{\partial f_k},
\qquad
\frac{\partial\ell_i}{\partial\omega_k}
=
\frac{\partial f_k}{\partial\omega_k}\cdot\frac{\partial\ell_i}{\partial f_k}.
\]

当 \(f_k=\beta_k+\omega_k h_k\) 时，\(\partial f_k/\partial\beta_k=1\)，\(\partial f_k/\partial\omega_k=h_k\)。输入层则是 \(x_i\) 代替 \(h_k\)。观察 1 在式子里出现了：\(\partial\ell/\partial\omega\) 正比于前向已经存下的源激活。

**一套具体数字走完前向+后向。** 为了能手算，把玩具压成两层线性+一层平方损失（激活先拿掉，链式法则更干净；ReLU 只是再乘一个 \(0/1\) 门）：

\[
f_0=\beta_0+\omega_0 x,\qquad
h_1=f_0,\qquad
f_1=\beta_1+\omega_1 h_1,\qquad
\ell=(f_1-y)^2.
\]

取 \(x=2\)，\(y=1\)，\(\beta_0=0.5\)，\(\omega_0=0.3\)，\(\beta_1=-0.2\)，\(\omega_1=0.4\)。

前向：

\[
\begin{aligned}
f_0&=0.5+0.3\cdot 2=1.1,\\
f_1&=-0.2+0.4\cdot 1.1=0.24,\\
\ell&=(0.24-1)^2=0.5776.
\end{aligned}
\]

后向：

\[
\frac{\partial\ell}{\partial f_1}=2(0.24-1)=-1.52.
\]

\[
\frac{\partial\ell}{\partial\beta_1}=-1.52,
\quad
\frac{\partial\ell}{\partial\omega_1}=h_1\cdot(-1.52)=1.1\cdot(-1.52)=-1.672.
\]

\[
\frac{\partial\ell}{\partial h_1}=\omega_1\cdot(-1.52)=0.4\cdot(-1.52)=-0.608
=\frac{\partial\ell}{\partial f_0}.
\]

\[
\frac{\partial\ell}{\partial\beta_0}=-0.608,
\quad
\frac{\partial\ell}{\partial\omega_0}=x\cdot(-0.608)=2\cdot(-0.608)=-1.216.
\]

核对：\(\omega_0\) 的梯度确实 \(\propto x\)（源激活）。\(\alpha=0.1\) 时 \(\omega_0\leftarrow 0.3-0.1\cdot(-1.216)=0.4216\)。没有前向缓存的 \(h_1=1.1\) 和 \(x=2\)，后向乘不上。

标量把机制说完了。深度网络只是把 \(\omega\) 换成矩阵、把乘法换成矩阵乘，规则不变。

---

## 3. 向量版：同一套规则，主成本是矩阵乘

把网络写成序列：

\[
\begin{aligned}
\mathbf f_0&=\boldsymbol\beta_0+\boldsymbol\Omega_0\mathbf x_i,\\
\mathbf h_k&=a[\mathbf f_{k-1}],\\
\mathbf f_k&=\boldsymbol\beta_k+\boldsymbol\Omega_k\mathbf h_k,\\
\ell_i&=l[\mathbf f_K,y_i].
\end{aligned}
\]

\(\mathbf f_{k-1}\) 是 pre-activation，\(\mathbf h_k\) 是激活后。后向仍是两段。先回传 \(\partial\ell_i/\partial\mathbf f_k\)：

- 最后一层依赖损失：MSE 下是 \(2(\mathbf f-y)\)；交叉熵加 softmax 常会简化；
- \(\partial\mathbf f_K/\partial\mathbf h_K=\boldsymbol\Omega_K^\top\)；
- ReLU 的 \(\partial\mathbf h/\partial\mathbf f\) 是对角门控 \(I[f>0]\)，实践中用逐元素相乘，不必真建对角矩阵。

回传交替做两件事：

```text
乘 Ωᵀ  →  按 ReLU 门控（⊙ I[f>0]） →  再乘上一层 Ωᵀ → …
```

再落到参数：

\[
\frac{\partial\ell_i}{\partial\boldsymbol\beta_k}
=
\frac{\partial\ell_i}{\partial\mathbf f_k},
\qquad
\frac{\partial\ell_i}{\partial\boldsymbol\Omega_k}
=
\frac{\partial\ell_i}{\partial\mathbf f_k}\,\mathbf h_k^\top.
\]

偏置的梯度就是对 pre-activation 的梯度；权重的梯度是外积，形状与 \(\boldsymbol\Omega_k\) 相同，仍然 \(\propto\mathbf h_k\)。

算法摘要。前向存中间量；后向从 \(k=K\) 走到 \(1\)：

\[
\begin{aligned}
\frac{\partial\ell_i}{\partial\boldsymbol\beta_k}
&=
\frac{\partial\ell_i}{\partial\mathbf f_k},\\
\frac{\partial\ell_i}{\partial\boldsymbol\Omega_k}
&=
\frac{\partial\ell_i}{\partial\mathbf f_k}\,\mathbf h_k^\top,\\
\frac{\partial\ell_i}{\partial\mathbf f_{k-1}}
&=
\mathbf I[\mathbf f_{k-1}>0]
\odot
\bigl(\boldsymbol\Omega_k^\top\tfrac{\partial\ell_i}{\partial\mathbf f_k}\bigr).
\end{aligned}
\]

输入层用 \(\mathbf x_i\) 替换 \(\mathbf h_0\)。Batch 内对 \(i\) 求和，就得到 SGD 用的梯度。

计算上，前向和后向的主成本都是矩阵乘 \(\boldsymbol\Omega,\boldsymbol\Omega^\top\)，所以高效。内存上，前向激活全存，于是 batch 和模型规模会被显存卡住。缓解手段包括梯度检查点、micro-batch、可逆网络。这些是工程，不改变链式法则。

现代框架做的是 algorithmic differentiation：每个算子带着自己的局部 Jacobian，记录计算图，自动 forward + reverse-mode backward。用户看见的是 `loss.backward()`。Batch 把向量升成 tensor（图像常是 \(N\times C\times H\times W\)）。计算图只需是 DAG，允许分支合流；不限于一条链。

Reverse-mode 适合“标量损失 → 大量参数”，正是深度学习的形状。Forward-mode 适合少量输入、多输出。选错模式，复杂度会差一个参数量那么多。

梯度现在可以算了。若初始化把每层的数乘得越来越小或越来越大，算对的梯度也会是 0 或 NaN。下一节把尺度当成一个概率问题，而不是再调一个 learning rate。

---

## 4. 初始化：让每层的方差活下去

前向是

\[
\mathbf f_k=\boldsymbol\beta_k+\boldsymbol\Omega_k a[\mathbf f_{k-1}].
\]

偏置常初始化为 0，权重 \(\Omega_{ij}\sim\mathcal N(0,\sigma_\Omega^2)\)。\(\sigma_\Omega^2\) 过小（如 \(10^{-5}\)），激活逐层缩小，ReLU 再砍掉一半，后向 **vanishing**；过大（如 \(10^5\)），激活逐层放大，后向 **exploding**。两种都会让更新无效或浮点崩溃。

一层的方差账如下。设 \(\boldsymbol\beta=\mathbf 0\)，\(\Omega_{ij}\) 与 \(\mathbf h\) 独立，则 \(\mathbb E[f'_i]=0\)，且

\[
\mathrm{Var}(f'_i)
=
\sigma_\Omega^2\sum_{j=1}^{D_h}\mathbb E[h_j^2].
\]

若前层 pre-activation 关于 0 对称，经 ReLU 后 \(\mathbb E[h_j^2]=\sigma_f^2/2\)（负半轴被置零，能量剩一半）。于是

\[
\sigma_{f'}^2
=
D_h\,\sigma_\Omega^2\cdot\frac{\sigma_f^2}{2}.
\]

要 \(\sigma_{f'}^2=\sigma_f^2\)，必须

\[
\boxed{
\sigma_\Omega^2=\frac{2}{D_h}
}
\quad\text{（He initialization）}.
\]

\(D_h\) 是该矩阵**输入侧**维度（fan-in）。那个 2 来自 ReLU 砍半；Xavier / Glorot 没乘这个 1/2，大约差两倍，更适合近似线性的激活。LeCun 更早，常配合 sigmoid。

后向乘的是 \(\boldsymbol\Omega^\top\)，对称论证给出 \(\sigma_\Omega^2=2/D_{h'}\)（fan-out）。非方阵无法同时满足两边，折中是

\[
\sigma_\Omega^2=\frac{4}{D_h+D_{h'}}.
\]

\(D_h=100\) 时 He 对应 \(\sigma_\Omega^2=0.02\)。图上前向激活方差和后向梯度方差都会稳住。

**He 的算术。** \(D_h=100\)，ReLU 砍半：

\[
\sigma_{f'}^2=100\cdot\sigma_\Omega^2\cdot\frac{\sigma_f^2}{2}
=50\,\sigma_\Omega^2\,\sigma_f^2.
\]

要 \(\sigma_{f'}^2=\sigma_f^2\)，必须 \(\sigma_\Omega^2=1/50=0.02\)，即 \(2/D_h\)。权重标准差 \(\sigma_\Omega=\sqrt{0.02}\approx 0.1414\)。若误用 Xavier 风格 \(\sigma_\Omega^2=1/D_h=0.01\)，每层方差乘 \(50\cdot 0.01=0.5\)，三层之后前向能量大约剩 \(0.5^3=0.125\)。

**三层：消失对爆炸。** 忽略 ReLU 砍半，只看线性放大因子 \(\rho=D_h\sigma_\Omega^2\)（数量级故事）。\(D_h=100\)。

| \(\sigma_\Omega\) | \(\sigma_\Omega^2\) | 一层 \(\rho\) | 三层 \(\rho^3\) | 名字 |
|---|---|---|---|---|
| \(0.01\) | \(10^{-4}\) | \(0.01\) | \(10^{-6}\) | vanishing |
| He \(\approx 0.141\) | \(0.02\) | 量级 \(1\)（再配 ReLU 的 2） | 量级 \(1\) | 稳住 |
| \(10\) | \(100\) | \(10^{4}\) | \(10^{12}\) | exploding |

```text
sigma=0.01 :  1  →  0.01  →  1e-4  →  1e-6     (signal dies)
sigma=10   :  1  →  1e4   →  1e8   →  1e12     (signal explodes)
```

后向乘的是同一串 \(\Omega^\top\)，前向消失则梯度也消失，前向爆炸则梯度也爆炸。\(\sigma\) 错一个数量级，三层已经足够让 `float32` 变成全 0 或 Inf；学习率再怎么拧也救不了尺度。

全零初始化是另一类失败：对称破坏失败，同层神经元收到相同梯度，学不到不同特征。通常 **权重随机、偏置可以是 0**。

BatchNorm 等会在训练中持续归一化（第 11 章），减轻对初始 \(\sigma\) 的苛刻依赖，但不等于“初始化不再重要”——网络一开始仍要能给出非零、非 Inf 的梯度。

---

## 5. 实现可以很短，机制不能短

书中训练循环的精神：

```text
定义 Linear–ReLU–Linear–ReLU–Linear
  → He（kaiming_normal_）
  → 损失
  → 优化器
  → 循环: zero_grad → forward → loss → backward → step
```

`loss.backward()` 把本节所有链式复用藏进一行。会写这一行不够；要能解释它在算 \(\partial\ell/\partial\beta=\partial\ell/\partial f\) 和 \(\partial\ell/\partial\Omega=(\partial\ell/\partial f)h^\top\)，以及初始化坏了时这两个量会怎样集体消失。

和 `gradient_descent.py` 的对照：多项式 `forward` 是前向；`loss_2_f=2*(y_pred-y_label)` 是 \(\partial\ell/\partial f\)；`l_2_a/b/c/d` 是手工链式法则。多项式层数浅、没有 ReLU 堆叠，不容易出现“逐层方差指数发散”，所以参数从 0 起步也能训。深度全连接或卷积才会强烈依赖 He 这种方差账。

---

## 6. 自测

1. 为什么 \(\partial\ell/\partial\omega\) 正比于源激活，从而前向必须缓存？  
2. “从后往前走”复用的是什么中间量？若每条路径都从头展开，重复在哪？  
3. 标量网络里 \(\partial\ell/\partial\beta_k\) 和 \(\partial\ell/\partial\omega_k\) 怎样用 \(\partial\ell/\partial f_k\) 表示？  
4. MSE 下 \(\partial\ell_i/\partial f=\)？  
5. 向量版为什么 \(\partial\ell/\partial\boldsymbol\Omega_k=(\partial\ell/\partial\mathbf f_k)\mathbf h_k^\top\)？形状为什么和权重一样？  
6. ReLU 的门控在后向做什么？Heaviside / 矩形激活为什么不利于梯度训练？  
7. 为什么深度网用 reverse-mode 而不是 forward-mode？  
8. \(\sigma_\Omega^2\) 过小和过大，前向和后向各会发生什么？  
9. 从“ReLU 砍掉一半能量”推出 He 的 2。Xavier 少了这个 2，它默认更接近哪种激活？  
10. 权重和偏置全部初始化为 0 会怎样？这和“方差过小”是同一件事吗？  
11. 为什么多项式实验从 0 起步能训，深度 ReLU 网通常不能？

---

## 7. 合上书再看一眼

第 6 章给了下山规则，本章给了地图和出发点。地图是反传：前向把源激活存下来，后向从 \(\partial\ell/\partial f\) 往回走，权重梯度是外积，偏置梯度就是 \(\partial\ell/\partial f\)。出发点是方差账：ReLU 会砍半，要用 \(\sigma_\Omega^2=2/D_h\) 把每层的信号尺度接住，否则算对的梯度也会消失或爆炸。

模型、损失、更新、反传、初始化到此齐了。下一章问一个到现在都被躲开的问题：训练损失往下走，是否等于模型真的学会了？

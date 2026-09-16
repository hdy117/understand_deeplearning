# 第 17 章：变分自编码器（Variational Autoencoders, VAE）

> 书：《Understanding Deep Learning》Ch.17  
> 前置：第 16 章用可逆变换换精确 \(p(x)\)，层必须是双射。Ch.14 把 VAE 放在「有 \(p(x)\)、但通常只能优化 lower bound」那一格。  
> 本章：怎样训练一个带 latent variable 的 probabilistic generator，而不必精确计算那个积不出来的 \(p(x)\)？  
> 后续：Ch.18 Diffusion 可以读成「encoder 被预先固定的 VAE」——先把本章的 ELBO 走通。

---

## 0. 生成很容易，卡住的是那个积分

第 14 章已经把生成写成：

\[
z\sim p(z),
\qquad
x\sim p_\theta(x\mid z).
\]

Prior 常取最省事的那个：

\[
p(z)=\mathcal N(0,I).
\]

抽一个 \(z\)，送进 decoder，就能得到新样本。这一步没有新困难。真正的训练目标却是第 5 章那套最大似然，只是现在没有标签 \(y\)，似然写在数据自己身上：

\[
p_\theta(x)
=
\int p_\theta(x\mid z)\,p(z)\,dz.
\]

Decoder 是非线性网络，这个积分通常没有闭式，也不能靠把 latent space 网格化来穷举。于是整章只追问一件事：

> **怎样绕过不可 tractable 的 latent integral，仍然把 decoder 往“提高 data likelihood”的方向推？**

答案不是再发明一种 generator，而是给这个积不出来的 \(\log p_\theta(x)\) 找一个可算、可采样、可反传的下界。三步就够陈述，后面每一节只负责让其中一步变得不可避免：

```text
引入一个可算的 qφ(z|x) 去近似真正的 posterior
        │
        ▼
得到恒等式：log pθ(x) = ELBO + KL(qφ || true posterior)
        │
        ▼
KL ≥ 0，所以最大化 ELBO 是在推高 log-likelihood 的下界
        │
        ▼
用 reparameterization 让「从 q 里抽样」也能把梯度送回 encoder
```

原书有一句几乎总被读滑的话，先钉在这里：

> **VAE 不是 \(p(x)\) 这个生成模型本身。它是用来学习那个模型的神经网络架构。**

训完之后，真正的 generative model 只剩 \(p(z)\,p_\theta(x\mid z)\)。Encoder、ELBO、reparameterization 都是训练时的脚手架。从 prior 生成新样本时，encoder 不在场。

三条路径不要混，后面所有“重建好却生成差”的故事都从这里长出来：

```text
Generative model：   z ~ p(z)        → x ~ pθ(x|z)     （生成只走这条）
Inference model：    看见 x          → qφ(z|x)         （训练和编码才需要）
Training estimator： x 与 ε ~ N(0,I) → z=μ+σ⊙ε → 随机 ELBO
```

| 对象 | 属于哪条路径 | 单样本 ELBO 里是否随机 |
|------|--------------|------------------------|
| \(x\) | 已观察到的 training sample | 条件固定 |
| \(z\) | latent variable | 随 \(q_\phi(z\mid x)\) 抽 |
| \(p(z)\) | 生成用的 prior | 固定，通常 \(\mathcal N(0,I)\) |
| \(p_\theta(x\mid z)\) | decoder / 观测似然 | 参数由 \(\theta\) 决定 |
| \(p_\theta(z\mid x)\) | true posterior | 想要但通常算不出 |
| \(q_\phi(z\mid x)\) | encoder 给出的近似 posterior | 可算、可抽 |
| \(\epsilon\) | reparameterization 的独立噪声 | 随 \(\mathcal N(0,I)\) 抽 |
| \(\theta,\phi\) | decoder / encoder 参数 | 优化变量，不是 sample |

下一个问题不是“encoder 怎么设计”，而是：为什么要绕到 \(z\) 上去描述 \(p(x)\)？直接写一个复杂密度，不行吗？

---

## 1. 为什么间接描述 \(p(x)\)：从混合高斯到无限混合

Latent variable model 不直接写 \(p(x)\)。它先写一个更容易下手的联合分布，再把没看见的 \(z\) 积掉：

\[
p_\theta(x,z)
=
p_\theta(x\mid z)\,p(z),
\qquad
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
\]

一个 observed \(x\) 可能由很多不同的 \(z\) 生成。要问“这个 \(x\) 有多可能”，必须把所有能生出它的 \(z\) 的贡献加起来。这就是 marginalization：不是丢掉 \(z\)，而是把 \(z\) 的不确定性算进 \(p(x)\)。

离散版就是 mixture of Gaussians。\(z\) 是成分编号：

\[
p(x)
=
\sum_{k=1}^{K}
p(z=k)\,p(x\mid z=k)
=
\sum_{k=1}^{K}
\lambda_k\,\mathcal N(x;\mu_k,\sigma_k^2).
\]

每个 \(z=k\) 只负责一块简单 Gaussian；加权求和之后，\(p(x)\) 可以多峰、可以歪。简单的 \(p(x\mid z)\) 和 \(p(z)\)，拼出复杂的 \(p(x)\)。这就是“间接描述”的全部好处。

VAE 把成分编号换成连续向量 \(z\)，把每个成分的均值换成神经网络：

```text
离散混合：K 个 Gaussian，均值是表格里的 μ_k
连续混合：无穷多个 Gaussian，均值是 fθ(z)
```

于是 \(p(x)\) 变成一张被 \(p(z)\) 加权的、无穷薄片 Gaussian 的叠加。Decoder 的工作不是“画出 \(x\)”，而是告诉你：给定这个 \(z\)，观测分布的中心在哪。

---

## 2. Nonlinear latent variable model：生成为什么容易

标准设定：

\[
p(z)=\mathcal N(z;0,I),
\qquad
p_\theta(x\mid z)
=
\mathcal N\bigl(x;\,f_\theta(z),\,\sigma_x^2 I\bigr).
\]

\(f_\theta\) 是 decoder 网络，输出的是 Gaussian mean，不是像素本身。\(\sigma_x^2 I\) 把“没被 \(z\) 解释掉的残差”记成各向同性噪声。\(z\) 通常比 \(x\) 低维：重要结构走 \(z\)，剩下的算观测噪声。

生成是 ancestral sampling，两步：

```text
z ~ N(0,I)
   ↓ decoder
mean fθ(z)
   ↓ 可选的 observation noise
x ~ pθ(x|z)
```

展示时若直接贴 \(f_\theta(z)\)，用的是 conditional mean。这和“从 \(p_\theta(x\mid z)\) 再抽一次”不是同一件事：后者会把观测噪声也画出来，前者更干净，也更容易把多峰细节平均成糊。

到这里，生成模型已经完整了。还没有 encoder。Encoder 是因为下面这个积分积不出来，才不得不请进来的。

---

## 3. 同一个分母卡住两件事

训练要最大化

\[
\log p_\theta(x)
=
\log\int p_\theta(x\mid z)\,p(z)\,dz.
\]

积分跨过整个 latent space，\(f_\theta\) 又非线性，没有闭式。这叫 evidence / marginal likelihood，是第一个难解对象。

第二个看起来像另一件事：看见 \(x\) 以后，哪些 \(z\) 最可能生出它？Bayes 给出 true posterior

\[
p_\theta(z\mid x)
=
\frac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)}.
\]

分母正是那个积不出来的 \(p_\theta(x)\)。所以：

> **Evidence 难算和 true posterior 难算，是同一道题的两个面。**

Posterior 就是 encoding 想要的对象：把一张脸压回内部坐标。我们算不出它，才需要下一节那个可算的替身 \(q_\phi(z\mid x)\)。

---

## 4. 变分思想：用一个可算的 \(q\) 去换下界

引入 encoder，输出的不是一个确定的 \(z\)，而是一整个分布的参数：

\[
q_\phi(z\mid x)
=
\mathcal N\bigl(\mu_\phi(x),\,\operatorname{diag}(\sigma_\phi^2(x))\bigr).
\]

```text
x
 ↓ encoder
μ(x), log σ²(x)
 ↓ 定义 qφ(z|x)
这个 x 对应的一整团可能 latent
```

选 diagonal Gaussian 是工程上的自洽，不是真理：抽样简单，对标准正态的 KL 有闭式，神经网络也好参数化。代价也很具体——true posterior 若多峰或维度间高度相关，一个对角高斯会近似得很差，ELBO 再怎么优化也紧不上去。

为什么不给每个训练样本单独优化一套 variational parameters？传统 VI 就是这么干的。VAE 改成一个共享 encoder：

```text
旧做法：每来一个 x_i，再单独迭代求它的 q_i(z)
VAE：   一次训练 encoder；以后一次 forward 就为任意 x 输出 qφ(z|x)
```

Inference 的成本被摊到整个 dataset 上，所以叫 amortized inference。摊销不是免费的：共享 encoder 受容量限制，即使每个样本单独优化能得到更好的 \(q\)，一次 forward 也不一定够到，这部分差距叫 amortization gap。

Encoder 再好，它也只是 inference 工具。生成模型的联合分布仍然是 \(p(z)p_\theta(x\mid z)\)。下一节要回答的是：有了 \(q\)，那个积不出来的 \(\log p_\theta(x)\) 到底怎么被卡住一个下界。

---

## 5. ELBO：先看恒等式，再看 Jensen

对任意满足 support 条件的 \(q_\phi(z\mid x)\)——凡是 \(p_\theta(x,z)\) 有贡献的地方，\(q\) 不能为 0——恒等式不是另造的界，只是把 \(\log p_\theta(x)\) 按 \(q\) 取期望再拆开。\(p_\theta(x)\) 对积分变量 \(z\) 是常数，所以

\[
\log p_\theta(x)
=
\mathbb E_q[\log p_\theta(x)]
=
\mathbb E_q
\left[
\log\frac{p_\theta(x,z)}{p_\theta(z\mid x)}
\right].
\]

再乘除 \(q_\phi(z\mid x)\)：

\[
=
\mathbb E_q
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
+
\mathbb E_q
\left[
\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right].
\]

第一项就是 ELBO，第二项就是 \(D_{\mathrm{KL}}(q_\phi\Vert p_\theta(z\mid x))\)。因此

\[
\boxed{
\log p_\theta(x)
=
\mathcal L_{\mathrm{ELBO}}(x;\theta,\phi)
+
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\bigr)
}
\]

其中

\[
\mathcal L_{\mathrm{ELBO}}(x;\theta,\phi)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
\]

KL 非负，所以右边第二项 \(\ge 0\)，第一项永远不超过 \(\log p_\theta(x)\)。这就是 Evidence Lower Bound 这个名字：它是 **log evidence** 的下界，不是 \(p_\theta(x)\) 本身的下界。

恒等式把整章的优化目标说完了：

```text
想推高 log pθ(x)
        │
        ├── 推高 ELBO（可算、可采样）
        └── 缩小 qφ 到 true posterior 的 KL（紧度）
```

当 \(q_\phi(z\mid x)=p_\theta(z\mid x)\) 时 KL 为 0，bound 才 tight。否则 ELBO 上升，并不保证每一次参数更新都让 true log-likelihood 同步上升——variational gap 本身也随 \(\theta\) 变。最大化 ELBO 是 exact log-likelihood 的 surrogate，不是它的别名。

离散、两值 \(z\)，可以把三件事同时算完。固定一个 \(x\)，令

\[
p(z=0)=p(z=1)=\tfrac12,
\qquad
p(x\mid z=0)=0.8,\quad p(x\mid z=1)=0.2.
\]

Marginal

\[
p(x)=0.5\cdot 0.8+0.5\cdot 0.2=0.5,
\qquad
\log p(x)=\log 0.5\approx -0.693.
\]

True posterior：\(p(z=0\mid x)=0.8,\ p(z=1\mid x)=0.2\)。Encoder 若给出同样的 \(q=(0.8,0.2)\)，则 posterior KL 为 0。ELBO 是

\[
\begin{aligned}
\mathbb E_q[\log p(x,z)-\log q]
&=
0.8\bigl(\log(0.5\cdot 0.8)-\log 0.8\bigr)
+
0.2\bigl(\log(0.5\cdot 0.2)-\log 0.2\bigr)\\
&=
\log 0.5
\approx -0.693.
\end{aligned}
\]

ELBO \(=\log p\)，界贴死。若 encoder 偷懒输出均匀 \(q=(0.5,0.5)\)：

\[
\begin{aligned}
\mathrm{ELBO}
&=
0.5\log\frac{0.5\cdot 0.8}{0.5}
+
0.5\log\frac{0.5\cdot 0.2}{0.5}
=
0.5\log 0.8+0.5\log 0.2
\approx -0.916,
\end{aligned}
\]

\[
D_{\mathrm{KL}}(q\Vert p(\cdot\mid x))
=
0.5\log\frac{0.5}{0.8}+0.5\log\frac{0.5}{0.2}
\approx 0.223.
\]

相加：\(-0.916+0.223=-0.693=\log p(x)\)。**恒等式是算术，不是口号：松掉的那截正好是 KL。** 训练最大化 ELBO 时，若 \(q\) 停在均匀，你会少记 \(0.223\) 的 log-likelihood，decoder 收到的梯度也掺了错的 \(z\) 权重。

同一条界也可以从 Jensen 走出来，几何图像更直。对数是凹函数：先取期望再取对数，一定不低于先取对数再取期望，

\[
\log\mathbb E_q[Y]
\ge
\mathbb E_q[\log Y].
\]

从

\[
\log p_\theta(x)
=
\log\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
\]

直接得到 ELBO。Jensen 解释“为什么是下界”；上面的恒等式解释“界松在哪”。两段路通向同一个对象。

ELBO 里那个 expectation 用 \(q_\phi\) 抽样就能 Monte Carlo 估计，不必穷举 latent space。还差一件事：抽样怎么把梯度送回 encoder。那是第 8 节的事。先把 ELBO 拆成训练时真正对打的两股力。

---

## 6. 两股相反的力：reconstruction 与 prior KL

把 \(p_\theta(x,z)=p_\theta(x\mid z)p(z)\) 代进去：

\[
\boxed{
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\Vert\,p(z)\bigr)
}
\]

第一项问：从 encoder 给出的 \(z\)，decoder 能不能给原始 \(x\) 较高 likelihood？这是 reconstruction / data fit。第二项问：每个样本的 \(q_\phi(z\mid x)\) 离大家共用的 prior 有多远？这是 latent regularization。

训练最小化 negative ELBO：

\[
\boxed{
\mathcal J
=
-\mathbb E_q[\log p_\theta(x\mid z)]
+
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\Vert\,p(z)\bigr).
}
\]

```text
Reconstruction pressure：z 必须携带这个样本的信息
KL pressure：            各样本的 posterior 必须和共同 prior 兼容
```

两股力方向相反。KL 改善 prior compatibility，却不保证消灭 latent holes，也不保证每个 prior sample 都能 decode 成高质量数据。Reconstruction 想把 \(q(z\mid x)\) 拉得又尖又远离原点，好记住这张图；KL 想把它们按回 \(\mathcal N(0,I)\)。VAE 的性格就是这场拉锯，不是“加一层噪声的 autoencoder”。

“Reconstruction loss”也不是凭空出现的 MSE。它就是 \(-\log p_\theta(x\mid z)\)，由你选的观测似然决定。

若

\[
p_\theta(x\mid z)=\mathcal N(f_\theta(z),\sigma_x^2 I),
\]

则

\[
-\log p_\theta(x\mid z)
=
\frac{1}{2\sigma_x^2}\|x-f_\theta(z)\|^2
+\text{constant}.
\]

MSE 是 fixed-variance Gaussian 的 NLL；\(\sigma_x^2\) 还顺便决定这项相对 KL 的权重。若每个维度真是 binary、decoder 输出 \(\hat p_j\)，同一项变成 BCE。把连续自然图像叫 Bernoulli，只是 modeling approximation，不是像素天生服从 Bernoulli。换 categorical、discretized logistic、learned-variance Gaussian，惩罚误差的方式、锐度和 likelihood 数字都会变。

一个把两项真正加在一起的标量例子。设 \(\mu=2,\sigma=0.5\)，于是对标准正态

\[
D_{\mathrm{KL}}(q\Vert p)
=
\frac12\sum_j\bigl(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\bigr)
\approx 2.318.
\]

再设 \(x=1.2,\,f_\theta(z)=1.0,\,\sigma_x^2=1\)，Gaussian reconstruction 去掉常数后是 \(0.02\)。这个 toy sample 的 negative ELBO 约 \(2.338\)。数字不证明“KL 永远更大”——相对尺度取决于观测维度怎么求和、\(\sigma_x^2\)、latent 维数和后面的 \(\beta\)。它只说明两股力会同时反传，并且经常对着干。

---

## 7. 为什么普通 Autoencoder 不够

普通 autoencoder 把 \(x\) 映成一个确定的点 \(z\)，再映回来。它可以把训练数据放到 latent space 里零散的孤岛上：重建很好，从 \(\mathcal N(0,I)\) 随手一抽却落在孤岛之间，decoder 输出荒谬样本。第 14 章说过：reconstruction 好，不等于 generation 好。

```text
ordinary AE latent：  ●●       ●       ●●
                     中间是 prior 会抽到的空洞

VAE：每个 x 对应一团 q(z|x)，KL 把这些团拉向共同 prior
                     更连续、较易采样
```

代价是 latent 不能任意分散去记忆数据，重建往往变差。VAE 用概率结构买的就是这块“prior 抽到的地方 decoder 也见过”的权利，不是更花哨的压缩。

---

## 8. Reparameterization：抽样怎样把梯度送回 encoder

Encoder 输出 \(\mu_\phi(x),\sigma_\phi(x)\) 之后，我们需要 \(z\sim\mathcal N(\mu,\sigma^2)\)。目标 expectation 对 \(\phi\) 的梯度通常存在；坏在不能把“从一个依赖 \(\phi\) 的分布里抽样”当成普通确定性节点，去做常规 pathwise backprop。Score-function / REINFORCE 是另一条路，方差通常更高。

改写成：

\[
\epsilon\sim\mathcal N(0,I),
\qquad
\boxed{
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
}
\]

```text
random ε ~ N(0,I) ──────┐
                         ▼
x → encoder → μ,σ → μ + σ⊙ε → z → decoder
```

随机性挪到与 \(\phi\) 独立的 \(\epsilon\) 分支。给定一次 \(\epsilon\)，\(z\) 对 \(\mu,\sigma\) 可微：

\[
\frac{\partial z}{\partial\mu}=1,
\qquad
\frac{\partial z}{\partial\sigma}=\epsilon.
\]

Decoder loss 对 \(z\) 的梯度可以继续传到 encoder。Reparameterization 没有删除随机性，只是把随机源从“依赖 \(\phi\) 的分布节点”移到“标准噪声 \(\epsilon\)”。

\(\mu=2,\sigma=0.5,\epsilon=-1\) 时 \(z=\mu+\sigma\epsilon=2+0.5\cdot(-1)=1.5\)。换 \(\epsilon=+1\) 得 \(z=2.5\)；换 \(\epsilon=0\) 得 \(z=2\)。三个数都来自同一个 \(q=\mathcal N(2,0.5^2)\)，差别只在独立噪声。梯度：\(\partial z/\partial\mu=1\)，\(\partial z/\partial\sigma=\epsilon\)，所以 \(\epsilon=-1\) 那一次对 \(\sigma\) 的 pathwise 梯度与 \(\epsilon=+1\) 符号相反——Monte Carlo 有方差，不是 \(z\) 变成了确定的编码。同一个 \(x\) 换一个 \(\epsilon\) 就换一个 \(z\)，但都服从 encoder 定义的 Gaussian。这正是 Monte Carlo ELBO 所要的：一次前向，一份随机下界估计。

---

## 9. 一次训练怎么走

对每个 \(x\)：

```text
1. Encoder(x) → μφ(x), log σ²φ(x)
2. ε ~ N(0,I)
3. z = μ + σ⊙ε
4. Decoder(z) → pθ(x|z) 的参数
5. loss = reconstruction NLL + KL(qφ(z|x) || p(z))
6. 反传过 decoder 和 encoder
```

```python
mu, logvar = encoder(x)
std = exp(0.5 * logvar)
eps = normal_like(std)
z = mu + std * eps
x_params = decoder(z)
loss = reconstruction_nll(x, x_params) + kl_to_standard_normal(mu, logvar)
```

Network 常输出 `logvar` 而不是 variance：任意实数都合法，\(\sigma=\exp(\tfrac12\mathrm{logvar})\) 自动为正，KL 公式本身也要用 \(\log\sigma^2\)。

Batch 上的 shape 只是把上面这条路径铺开：

```text
x                               [B,C,H,W]
μ, logvar, ε, z                 [B,L]
decoder likelihood parameters   与 x 同 shape
per-sample recon NLL、KL        [B]
batch reduction                 scalar
```

ELBO 里的 reconstruction 是 expectation。训练常用 \(L\) 个 latent samples 估计；标准对角高斯 VAE 的 KL 用闭式精确算，随机性主要来自 reconstruction。实践里 \(L=1\) 很常见。Mini-batch 又是整个 dataset 的随机估计，所以训练同时有 data sampling 和 latent sampling 两层噪声。

---

## 10. 生成、重建、aggregated posterior 不是同一条路

| 任务 | \(z\) 从哪来 | 要不要 encoder |
|------|----------------|----------------|
| Training | \(z\sim q_\phi(z\mid x)\) | 要 |
| Deterministic reconstruction | \(z=\mu_\phi(x)\) | 要 |
| Stochastic reconstruction | \(z\sim q_\phi(z\mid x)\) | 要 |
| Prior generation | \(z\sim p(z)\) | **不要** |
| Resynthesis / editing | encode 再改 / interpolate | 要 |

把所有真实数据的 posterior 混在一起，得到 aggregated posterior

\[
q_\phi(z)
=
\mathbb E_{x\sim p_{\mathrm{data}}}[q_\phi(z\mid x)]
\approx
\frac1N\sum_{i=1}^N q_\phi(z\mid x_i).
\]

```text
重建用的 z：来自某个 qφ(z|x)
真实数据的 latent 云：来自 aggregated qφ(z)
纯生成用的 z：来自 prior p(z)
```

若 \(q_\phi(z)\neq p(z)\)，decoder 在重建常见的区域可以很好，prior 仍会采到它很少见过的洞。Per-sample KL 鼓励二者靠近，不保证重合。从 aggregated posterior 采样往往更像训练数据，但那已经在用真实数据和 encoder，不是纯 prior generation。

编辑时可以把 \(x\) 投到 \(z=\mu_\phi(x)\)，或找一个 MAP-like 的 \(z^*\)。两个 latent 之间插值：接近球形 Gaussian 时，沿球面的 Slerp 有时比直线更少穿过 typical set 的低密度内部。语义插值不是自动保证——第 14 章的 disentanglement 警告在这里继续有效。

---

## 11. 三件不会自动发生的事：模糊、collapse、可解释因子

高斯似然加 MSE 时，若同一个 \(z\) 附近有多种合理细节，decoder 的 conditional mean 会把它们平均掉，样本发糊。这是常见机制，不是定理。质量还取决于 decoder 结构、likelihood 选择、latent 维数、ELBO 权重、是否分层、怎么训练。现代 VAE 用更强 likelihood、hierarchical / discrete latents，可以显著改观。

另一头是 posterior collapse。若 decoder 自己就能把 \(x\) 解释掉——尤其是带 autoregressive context、teacher forcing 的强 decoder——它可以不使用 \(z\)：

\[
q_\phi(z\mid x)\approx p(z),
\qquad
I_q(X;Z)\approx 0,
\qquad
p_\theta(x\mid z)\approx p_\theta(x).
\]

KL 接近 0 看起来像“正则化很成功”，其实是 latent 没干活。简单 feedforward decoder 的唯一输入就是 \(z\) 时，完全忽略 \(z\) 往往只能输出全局平均，data fit 会垮，所以 collapse 风险依赖结构。干预诊断：在 batch 内 shuffle \(z\)，或用 prior sample 替换 posterior sample；若重建几乎不变，decoder 就没在看 latent。缓解手段（KL warm-up、free bits、限制 decoder 绕过能力）是 heuristics，削弱 decoder 也可能把 likelihood 一起打下去。

\(\beta\)-VAE 把 KL 乘上 \(\beta>1\)，强化“靠近 prior”那一股，有时看到更分离的因子，通常牺牲重建。没有 supervision 或额外假设，真实 generative factors 并不保证可唯一识别。Disentanglement 是希望和经验现象，不是 ELBO 的定理。

这三件事回到第 14 章同一句话：VAE 在「概率故事清楚、latent 较规则」上下注，用 sample 偏平滑和 exact likelihood 换来的。没有额外压力，它不会再送你锐图和可拧旋钮。

---

## 12. 应用里真正用到的是哪条路径

Vanilla VAE 重建的是它看见的输入，不会因为有 bottleneck 就自动变成 denoiser。要 noisy-to-clean，必须显式把 noisy input 和 clean target 写进协议。

异常检测可以用重建误差、ELBO 或近似 likelihood 当信号；异常有时也能拿到高 likelihood，必须在目标 domain 验证。

压缩是 encoder 产出 code、decoder 重建；真要压体积还得 quantization 和 entropy coding。

ELBO 本身是 log-likelihood 下界。若想更紧的近似，importance sampling 用

\[
p_\theta(x)
=
\mathbb E_{z\sim q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right],
\qquad
\widehat p_\theta(x)
=
\frac1K\sum_{k=1}^K
\frac{p_\theta(x,z_k)}{q_\phi(z_k\mid x)}.
\]

Encoder 把样本集中到对当前 \(x\) 重要的区域，比从 prior 做 naive Monte Carlo 高效得多。注意 \(\mathbb E[\log w]\neq\log\mathbb E[w]\)：ELBO 与 log importance average 不能交换，实现要用 `logsumexp`。有限 \(K\) 仍是近似。

---

## 13. 自测

1. 为什么说 VAE 是学习 \(p(x)\) 的架构，而生成模型本身不含 encoder？  
2. 混合高斯怎样用简单的 \(p(x\mid z)p(z)\) 拼出复杂 \(p(x)\)？VAE 把其中什么换成了神经网络？  
3. Decoder 输出的是 \(x\) 还是 \(p_\theta(x\mid z)\) 的参数？展示 \(f_\theta(z)\) 时用的是哪一个？  
4. Evidence 与 true posterior 为什么是同一道题？  
5. 写出 \(\log p_\theta(x)\)、ELBO 与 posterior KL 的恒等式。紧度在什么时候达到？  
6. Reconstruction term 与 prior KL 各施加什么压力？为什么会对着干？  
7. Gaussian likelihood 怎样导出 MSE-like reconstruction？\(\sigma_x^2\) 还控制什么？  
8. 普通 autoencoder 的 latent 孤岛怎样让 prior generation 失败？  
9. Reparameterization 把随机性移到了哪里？它有没有让 \(z\) 变成确定性？  
10. Encoder 为什么常输出 `mu` 与 `logvar`？  
11. Aggregated posterior 与 prior 不一致时，重建好为什么仍可能生成差？  
12. Posterior collapse 的 KL 很小，为什么反而是失败？怎样做干预诊断？  
13. \(\beta\) 增大可能带来什么、牺牲什么？它能把 disentanglement 变成定理吗？  
14. Amortized inference 摊销了什么成本，又引入什么 gap？  
15. ELBO 的 \(\mathbb E[\log w]\) 与 importance sampling 的 \(\log\mathbb E[w]\) 为什么不能交换？  
16. 为什么 vanilla VAE 不会仅凭 bottleneck 自动完成 denoising？  
17. 两值 \(z\)、\(p(x\mid z)=(0.8,0.2)\)、均匀 prior：\(\log p(x)\)、tight ELBO、均匀 \(q\) 的 ELBO 与 KL 各是多少？三者怎样加回去？  
18. \(\mu=2,\sigma=0.5,\epsilon=-1\) 的 \(z\) 是多少？\(\partial z/\partial\sigma\) 为什么带着 \(\epsilon\)？

---

## 14. 合上书再看一眼

VAE 的生成假设极短：先从简单 prior 抽 \(z\)，再由 decoder 给出 \(p_\theta(x\mid z)\)。这一步容易。训练难，因为 \(p(x)\) 要把所有能解释这个 \(x\) 的 \(z\) 积在一起，而 true posterior 的分母就是这个积分。

于是请一个 encoder 来给出可算的 \(q_\phi(z\mid x)\)，把目标换成 ELBO。恒等式说，ELBO 与 \(\log p(x)\) 之间只隔着 \(q\) 到 true posterior 的 KL。把 ELBO 展开，又看见两股力：reconstruction 要 \(z\) 记住这个样本，KL 要各样本的 \(q\) 挤回共同 prior，好让生成时的随机 \(z\) 落在 decoder 见过的地方。Reparameterization 只做一件工程上致命的小事：让“抽样”不再挡住回 encoder 的梯度。

训的时候 encoder 和 decoder 都在；从 prior 生成时只留下 decoder。重建好、KL 小、因子可拧，没有一件是这份 objective 自动配送的。下一章会把“学一个 encoder 去近似 posterior”换成“把 encoder 写成固定加噪”，只学反向的逐步去噪。ELBO 还在，只是 \(q\) 不再是网络。

# 第 18 章：扩散模型（Diffusion Models）

> 书：《Understanding Deep Learning》Ch.18  
> 前置：Ch.17 的 latent model、ELBO、Gaussian 与 reparameterization。  
> 本章：怎样把“一步从噪声跳到图像”拆成许多容易学的小去噪步骤？  
> 一句话先记住：forward 加噪是固定协议，不学习；reverse 去噪才是模型。

---

## 0. 把 VAE 的 encoder 换成一条固定的加噪路

第 15 章的 GAN 会造图，却没有 \(p(x)\)。第 16 章的 Flow 有精确 likelihood，却被可逆性捆住。第 17 章的 VAE 有完整概率故事，却必须学一个 encoder 去近似 posterior，样本还常发糊。

Diffusion 站在这两边中间。原书的定位几乎是一句定义：

> **它像 Flow，观测和 latent 同维；它像 VAE，用 ELBO 逼近 likelihood。但 encoder 被预先写死——逐步把数据混进白噪声——我们只学那个反向的 decoder。**

一步从 Gaussian 跳到人脸仍然很难：

```text
white noise ──one huge jump──► realistic image
```

可如果每次只去掉一点点噪声，每一步都是“稍微干净一点”的回归，问题可能突然好做：

```text
noise → slightly less noisy → layout → objects → data
```

核心策略因此极短：

> **先设计一条固定的逐步加噪过程，把数据变成（接近）标准 Gaussian；再学习这条过程的逐步反向。**

```text
Forward / diffusion： x = z₀ → z₁ → … → z_T ≈ noise
                      固定协议，没有 learned parameters

Reverse / generation： noise = z_T → … → z₁ → z₀ = x
                      这才是 learned model
```

为什么这招有机会成立？加噪完全已知；每一步移动很小，小反向常可用 Gaussian 近似；训练时 clean \(x\) 在手，可以构造监督 target。后面所有公式都在把这三句话变成可算的算法。

调查链：

```text
VAE 要学 encoder 去近似 q(z|x)
        │
        ▼
Diffusion 把 encoder 写成固定加噪：x → z₁ → … → z_T
        │
        ▼
训练不必真走完前 t 步：diffusion kernel 一步跳到任意 z_t
        │
        ▼
真正要的 q(z_{t-1}|z_t) 未知；训练时的 q(z_{t-1}|z_t,x) 却是闭式 Gaussian
        │
        ▼
ELBO 在 Gaussian 假设下变成加权 noise regression；实践再丢掉权重得到 L_simple
        │
        ▼
训练随机抽一个 t；生成必须从 T 串行走回 0
```

三条 data flow 必须先分开。同一个张量 \(z_t\) 在三条路上身份不同，混用一次，后面的 \(\epsilon\)-prediction、sampling、guidance 会全部错位。

---

## 1. 三条路：加噪、训练、生成

Forward 的目的只有一个：构造 noisy state \(z_t\)。

```text
clean x → add a little noise → add more → … → almost standard Gaussian
```

它是预先规定的 corruption，不训练 network。

Training 的目的是：给 \(z_t\) 和当前噪声等级 \(t\)，让网络预测当初加进去的 noise。

```text
x, random t, random ε
 → 直接构造 z_t
 → network εθ(z_t, t)
 → MSE against ε
```

Sampling 的目的是从纯噪声造新样本。此时没有 ground-truth \(x\) 可偷看：

```text
z_T ~ N(0,I)
 → 在 T 处预测并去掉噪声
 → z_{T-1} → … → z₀ ≈ data
```

基础 pixel-space diffusion 里，\(x,z_t,\epsilon,\epsilon_\theta\) 都是与图像同 shape 的张量。这里称 \(z_t\) 为 latent，只表示它未被直接观测；它不是 VAE 那种已经压短的语义坐标。只有 latent diffusion 才会先用 autoencoder 把图压成较小的 spatial tensor。

| 符号 | 含义 | 是否学习 |
|------|------|----------|
| \(x=z_0\) | clean sample | 来自 dataset |
| \(z_t\) | 第 \(t\) 级 noisy state | 由 fixed forward kernel 构造 |
| \(t\) | noise-level index | 抽来当条件，不是网络参数 |
| \(\beta_t\) | 这一步加入的 noise variance | 通常预设 |
| \(\alpha_t=1-\beta_t\) | 单步保留 signal 的比例 | 由 schedule 决定 |
| \(\bar\alpha_t=\prod_{s=1}^t\alpha_s\) | 从 0 到 \(t\) 的累计保留 | 由 schedule 决定 |
| \(\epsilon\) | 构造 \(z_t\) 时的标准噪声 | 每个例子重新抽 |
| \(\epsilon_\theta(z_t,t)\) | 网络预测的 noise | \(\theta\) 学习的对象 |

下一节把 forward 写成公式。公式看起来像在“学加噪”，其实每条等号右边都没有 \(\theta\)。

---

## 2. Forward：每一步只做两件事

取 \(\beta_t\in(0,1)\)，\(\alpha_t=1-\beta_t\)。一步转移：

\[
\boxed{
q(z_t\mid z_{t-1})
=
\mathcal N\bigl(\sqrt{\alpha_t}\,z_{t-1},\,(1-\alpha_t)I\bigr)
}
\]

抽样形式更直白：

\[
\boxed{
z_t
=
\sqrt{\alpha_t}\,z_{t-1}
+
\sqrt{1-\alpha_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I).
}
\]

第一项衰减已有 signal，第二项加入新鲜噪声。所有 \(\beta_t\) 排成 noise schedule，决定信号消失的速度。这是 Markov 链：给定 \(z_{t-1}\)，下一步不再需要更早历史。

若训练每次都从 \(x\) 一步步加到第 \(t\) 步，会把计算浪费在已经规定好的随机性上。Gaussian 的线性性送出一个救命的闭式。

---

## 3. Diffusion kernel：训练为什么不必走完前面所有步

定义累计保留

\[
\bar\alpha_t=\prod_{s=1}^t\alpha_s.
\]

任意时刻都可以从 \(x\) 一步跳到：

\[
\boxed{
q(z_t\mid x)
=
\mathcal N\bigl(\sqrt{\bar\alpha_t}\,x,\,(1-\bar\alpha_t)I\bigr)
}
\]

\[
\boxed{
z_t
=
\sqrt{\bar\alpha_t}\,x
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
\]

这就是 diffusion kernel。它把“顺序加噪”换成一次代数。

\(\bar\alpha_t=0.64,\,x=1,\,\epsilon=-0.5\) 时，\(z_t=0.8\cdot 1+0.6\cdot(-0.5)=0.5\)。Signal 系数 0.8，noise 系数 0.6。小 \(t\) 时 \(\bar\alpha_t\) 接近 1，\(z_t\) 仍像 \(x\)；大 \(t\) 时 \(\bar\alpha_t\) 接近 0，\(z_t\) 接近标准正态。

有限 \(T\) 时只能说“接近”，不要写成“等于”。因为

\[
q(z_T\mid x)
=
\mathcal N\bigl(\sqrt{\bar\alpha_T}\,x,\,(1-\bar\alpha_T)I\bigr),
\]

只有 \(\bar\alpha_T=0\) 时 mean 才不含 \(x\)。若 \(\bar\alpha_T=0.1\)，mean 仍是 \(\sqrt{0.1}\,x\)，残留信号还在。终点 prior matching 的好坏取决于 schedule 和 \(T\)。

Kernel 解决了训练时怎么得到 \(z_t\)。它没有解决生成时真正需要的那个反向分布。

---

## 4. 真正要的反向为什么难，训练时的反向为什么容易

生成需要

\[
q(z_{t-1}\mid z_t).
\]

它依赖真实 \(p_{\mathrm{data}}\)。一个 noisy point 可能来自许多不同的 clean 图，这个 reverse 可以复杂、多峰、未知。

训练时多知道一个条件 \(x\)：

\[
q(z_{t-1}\mid z_t,x).
\]

Forward 是线性高斯，clean \(x\) 又已知，这个条件分布是闭式 Gaussian：

\[
q(z_{t-1}\mid z_t,x)
=
\mathcal N(\widetilde\mu_t,\widetilde\beta_t I),
\]

\[
\widetilde\beta_t
=
\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t},
\qquad
\widetilde\mu_t
=
\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,z_t.
\]

两项各带一块已知信息：

```text
clean x：数据最终从哪来
noisy z_t：当前轨迹走到哪
```

```text
q(z_{t-1}|z_t)      生成需要，未知且复杂
q(z_{t-1}|z_t,x)    训练时 x 已知，精确可算
```

整章的学习问题可以收成一句：

> **用大量“知道 \(x\)”的例子，学会在不知道 \(x\) 时把噪声往回推。**

Network 因此只看见 \(z_t\) 和 \(t\)，输出与 \(\widetilde\mu_t\) 等价的信息。下一节把这个 decoder 写成一个共享网络。

---

## 5. Learned reverse：一小步、一个高斯、一个共享 U-Net

定义

\[
p_\theta(z_{t-1}\mid z_t,t)
=
\mathcal N\bigl(\mu_\theta(z_t,t),\,\sigma_t^2 I\bigr).
\]

为什么是 Gaussian？\(\beta_t\) 很小时，反向只移动一点，单峰高斯通常够用。这和 VAE 用对角高斯去近似 posterior 是同一类“用简单分布换可算性”的赌注，只是这里每一小步都更有理由相信局部高斯。

不为每个 \(t\) 存一套独立网络。一个共享 U-Net 吃两个输入：

```text
noisy z_t ─────────┐
                    ├→ shared U-Net → prediction
 time embedding t ─┘
```

\(t\) 不是装饰。同一个 noisy tensor 在 \(t=3\) 和 \(t=900\) 意味着完全不同的任务：一个是微调纹理，一个是从雪花里找布局。没有 time embedding，共享网络不知道该去多强的噪。

概率故事到这里已经闭环：forward 固定，reverse 学习，ELBO 可以对这一长串 latent 写下来。下一节是本章最容易“公式一多就迷路”的地方：ELBO 怎样一步步变成大家真正在最小化的 \(\|\epsilon-\epsilon_\theta\|^2\)。

---

## 6. 从 ELBO 到预测噪声：三次简化，不是一次恒等

第 17 章用 encoder 网络去近似 \(q(z\mid x)\)。这里 encoder 已经是固定加噪，ELBO 仍然适用，只是 \(q\) 不再含学习参数。Diffusion 是带许多中间变量的层次模型：\(x=z_0,z_1,\ldots,z_T\)。Data likelihood

\[
p_\theta(x)=\int p_\theta(x,z_{1:T})\,dz_{1:T}
\]

积不出来，于是像 VAE 一样优化 ELBO。展开后是三组项：

```text
Reconstruction：  z₁ 能否解释 clean x
Transition KLs： learned reverse 是否匹配可计算的 q(z_{t-1}|z_t,x)
Terminal prior： forward 终点是否接近选定的 Gaussian prior
```

当 reverse variance 固定时，Gaussian transition KL 里与 \(\theta\) 有关的部分可以改写成带时间权重的噪声平方误差：

\[
\mathcal L_{\mathrm{weighted}}
=
\mathbb E_{x,t,\epsilon}
\bigl[
w_t\|\epsilon-\epsilon_\theta(z_t,t)\|^2
\bigr]
+\text{与 }\theta\text{ 无关的项}.
\]

\(w_t\) 来自 schedule、posterior variance 和 parameterization，不是所有 \(t\) 天然同权。

实践再走一步：让网络预测构造 \(z_t\) 时加入的 \(\epsilon\)，并丢掉 \(w_t\)，

\[
\boxed{
\mathcal L_{\mathrm{simple}}(\theta)
=
\mathbb E_{x,t,\epsilon}
\bigl[
\|\epsilon-\epsilon_\theta(z_t,t)\|^2
\bigr],
\qquad
z_t=\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\epsilon.
}
\]

必须把三层分开，否则会把工程简化说成定理：

```text
Exact ELBO
    ↓ Gaussian 代数 + 固定 variance
Weighted noise-prediction objective
    ↓ 丢掉与 t 有关的权重（实践简化）
Unweighted L_simple
```

\(\mathcal L_{\mathrm{simple}}\) 往往很好用，它不是与原始 ELBO 恒等。“从 ELBO 推出来”不等于“中间没有 approximation”。

预测噪声并不是与图像无关的辅助任务。Kernel 是 \(z_t=\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\epsilon\)。网络给出 \(\hat\epsilon=\epsilon_\theta(z_t,t)\) 之后，移项就是 clean 估计：

\[
\boxed{
\hat x_0
=
\frac{z_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}.
}
\]

知道哪一部分是噪声，就知道剩下的 signal 该是什么。沿用前面的标量：\(z_t=0.5,\bar\alpha_t=0.64\)，若 \(\hat\epsilon=-0.5\) 则 \(\hat x_0=1\)；若误成 \(-0.4\)，则 \(\hat x_0=0.925\)。Noise error 通过这组系数变成 clean-sample error，不同 \(t\) 的系数不同，所以 schedule 和 loss weighting 会改变各噪声等级的学习难度。

同一个 reverse 还可以让网络预测别的、可互相转换的对象：

| Parameterization | 网络目标 | 直觉 |
|------------------|----------|------|
| \(\epsilon\)-prediction | 加入的噪声 | 最常见的 DDPM 简化目标 |
| \(x_0\)-prediction | clean sample | 直接输出去噪估计 |
| \(v\)-prediction | signal 与 noise 的线性组合 | 改善不同 noise levels 的数值平衡 |

一种常见 velocity：

\[
v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x.
\]

给定 \(z_t,t\) 和 schedule，三者可代数转换；它们对各个 \(t\) 的 implicit weighting 和优化尺度并不相同。“表示同一信息”不等于“训练行为一样”。

再往下看一层：条件 noisy Gaussian 的 score 是

\[
\nabla_{z_t}\log q(z_t\mid x)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
\]

最优 noise predictor 学到 \(\mathbb E[\epsilon\mid z_t]\)，与 noisy marginal 的 score 成比例。这就是 denoising diffusion 和 score-based modeling 能写在同一套语言里的原因。第一遍读可以跳过；它解释的是“为什么预测噪声不是拍脑袋”。

---

## 7. 训练：随机抽一个 \(t\)，一次前向就有 loss

```text
1. Sample x
2. Sample t ~ Uniform{1,…,T}
3. Sample ε ~ N(0,I)
4. z_t = √ᾱ_t x + √(1-ᾱ_t) ε
5. ε̂ = εθ(z_t, t)
6. loss = ‖ε − ε̂‖²
```

```python
x = sample_data_batch()
t = sample_timesteps()
eps = normal_like(x)
z_t = sqrt(alpha_bar[t]) * x + sqrt(1 - alpha_bar[t]) * eps
pred = model(z_t, t)
loss = mse(pred, eps)
```

不必从 \(z_1\) 顺序走到 \(z_t\)。同一个 \(x\) 搭配不同 \(t\) 和 \(\epsilon\)，本身就是数据增强。

Noise schedule \(\{\beta_t\}\) 和训练时的 \(p(t)\) 是两件事：前者说每个 \(t\) 腐蚀多强，后者说这个 \(t\) 被抽到的频率。基础 DDPM 常令 \(t\) 均匀。按难度做 importance sampling 是扩展，不要说成“按 schedule 抽 \(t\)”。

训练便宜：一次只调用网络一次。生成没有这个福气。

---

## 8. 生成：必须从 \(T\) 串行走回 \(0\)

```text
1. z_T ~ N(0,I)
2. for t = T, …, 1:
       预测 εθ(z_t, t)
       转成 reverse mean / denoised estimate
       sample 或确定地得到 z_{t-1}
3. 返回 z_0
```

```text
pure noise → coarse layout → objects → texture → sample
```

和训练的差别就是第 14 章那根“采样速度”轴：训练随机单步，生成多步串行。下一步的输入 \(z_{t-1}\) 要等这一步算完才存在，时间轴上无法完全并行。

常见 fixed-variance DDPM 里，noise 预测转成 reverse mean：

\[
\boxed{
\mu_\theta(z_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
z_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(z_t,t)
\right)
}
\]

然后

\[
\boxed{
z_{t-1}
=
\mu_\theta(z_t,t)+\sigma_t\eta,
\qquad
\eta\sim\mathcal N(0,I).
}
\]

```text
z_t, t  →  U-Net  →  εθ
                ↓ schedule 代数
               μθ
                + σ_t η（t>1）
                ↓
              z_{t-1}
```

最后一步 \(t=1\) 通常不再加新噪声。\(\sigma_t^2\) 可以取 forward / posterior variance，也可以学；上式是共同接口，不是所有 sampler 的同一轨迹。

到这里算法已经能跑。下面两个问题是工程后果，不是新的概率对象：图像上每一步都是同尺寸的 image-to-image，所以 U-Net 会来；共享网络必须知道噪声等级，所以 \(t\) 会来。

---

## 9. 为什么是 U-Net，为什么必须把 \(t\) 喂进去

每一步：

```text
noisy image → predicted noise / denoised image
```

既要 high-resolution 细节，又要大感受野，输出还得和输入同空间尺寸。U-Net 的 skip 把 encoder 的细纹理直接送回 decoder，正好干这件事。实际 diffusion U-Net 再叠 residual、attention、timestep embedding、normalization——第 10–12 章的零件，第一次被装进一个生成系统。

Time embedding 经 MLP 注入多个 stage，去调各通道的 scale / shift。没有 \(t\)，同一套权重无法同时胜任“擦一点噪”和“从纯噪声里找结构”。

---

## 10. 条件生成：guidance 是旋钮，不是越大越好

希望样本满足条件 \(c\)（class、text、图、segmentation、低分辨率图）时，网络变成 \(\epsilon_\theta(z_t,t,c)\)。

Classifier guidance 额外训一个能看 noisy image 的 \(p(c\mid z_t)\)，用 \(\nabla_{z_t}\log p(c\mid z_t)\) 把 reverse mean 推向更符合条件的方向。Gradient 按 reverse variance 缩放后修正更新，不是随手加在任意位置。代价是多一个必须适应各种噪声等级的分类器。

Classifier-free guidance 把分类器丢掉。同一个模型同时学 conditional 与 unconditional，训练时随机把 \(c\) 换成 null。采样时组合

\[
\boxed{
\hat\epsilon
=
\epsilon_{\mathrm{uncond}}
+
w(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
}
\]

- \(w=0\)：无条件；
- \(w=1\)：普通条件；
- \(w>1\)：沿条件方向外推，**不是**两个概率模型的凸插值。

\(w\) 大则更贴 prompt、更典型；过大则多样性下降，颜色和细节过曝，出现 artifacts。另有一种写法 \(\epsilon_{\mathrm{cond}}+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})\)，此时 \(s=0\) 才是普通条件。不同实现的数字不能脱离公式比较。

Guidance 是第 14 章 fidelity / 条件一致性 与 diversity 的控制旋钮。它不创造数据里没有的信息，只是在已学到的条件方向上加力。

---

## 11. 为什么慢，以及用什么换速度

慢的原因已经在第 8 节：大约 \(T\) 次串行网络评价。加速走三条常见路。

DDIM 证明同一套 noise predictor 可以对应一族 reverse process，其中包含确定性路径，并允许跳过许多 \(t\)。不必为了少步数而重训一个完全不同的模型。

另一路是更好的 ODE / SDE solver、progressive distillation、consistency-style 方法：用更复杂的训练或近似换更快 inference。

第三路是 latent diffusion：先用 autoencoder 把图像压到较小 latent，再在那里 diffusion。

```text
image x → encoder → compact y → diffusion in y → decoder → image
```

Compute 下降，质量上限受 autoencoder 重建能力约束。这里的 latent 主要为了压缩计算，不保证是 VAE 那种可解释因子。

---

## 12. 回到第 14 章的交换表

| 方面 | 下注 | 代价 |
|------|------|------|
| 训练 | noise MSE，通常稳定 | 大模型和大开销仍在 |
| 样本 | 高 fidelity，coverage 往往较好 | 多步 sampling 慢 |
| 概率 | 有 ELBO 故事 | exact likelihood 不直接 |
| 条件 | class / text / image 灵活 | guidance 与 diversity 对打 |
| Latent | 固定噪声路径，数学清楚 | 基础 pixel \(z_t\) 没有语义旋钮 |
| 结构 | 共享 U-Net + time embedding | 显存和算力大 |

这就是第 14 章给 Diffusion 的那一格：用采样步数换质量和训练稳定性。Encoder 被写死成加噪以后，我们不再像 VAE 那样同时搏一套 amortized posterior——这是它比 VAE 好训的重要原因，也是它的 \(z_t\) 不像 VAE 的 \(z\) 那样可拧的原因。

---

## 13. 自测

1. 原书说 Diffusion“像 VAE 用 ELBO，但 encoder 预先固定”。固定的是什么，学的是什么？  
2. 为什么把一步生成拆成许多小 reverse steps 会更容易？  
3. \(\beta_t,\alpha_t,\bar\alpha_t\) 分别管哪一段 signal / noise？  
4. Diffusion kernel 怎样让训练避免顺序走完前 \(t-1\) 步？  
5. 有限 \(T\) 时 \(z_T\) 为什么通常只是接近、而不是等于标准正态？  
6. 为什么 \(q(z_{t-1}\mid z_t)\) 难，而 \(q(z_{t-1}\mid z_t,x)\) 可算？这句话怎样概括整章的学习问题？  
7. Exact ELBO、weighted noise loss、unweighted \(L_{\mathrm{simple}}\) 是什么关系？  
8. 为什么预测 \(\epsilon\) 等价于在估计 \(\hat x_0\)？写出恢复公式。  
9. \(\epsilon\)-、\(x_0\)-、\(v\)-prediction 为何信息可转换但训练行为不完全一样？  
10. 写出一次 training sample 的完整 data flow。它和 sampling 的 data flow 差在哪？  
11. Noise schedule 与训练时的 \(p(t)\) 有什么区别？  
12. DDPM reverse mean 怎样把 \(\epsilon_\theta\) 转成 \(z_{t-1}\)？为什么必须串行？  
13. 为什么图像 diffusion 常用 U-Net？Time embedding 在告诉网络什么？  
14. CFG 公式里 \(w=0,1,>1\) 分别是什么？为什么说 \(w>1\) 不是凸插值？  
15. DDIM 为什么常常不必重训？Latent diffusion 用什么代价换速度？  
16. 基础 pixel-space 的 \(z_t\) 和 VAE 的 \(z\) 差在哪一件关于“压缩 / 语义”的事？

---

## 14. 合上书再看一眼

Diffusion 先规定一条路：数据逐步失去 signal，最后接近标准高斯。这条路不学习。训练时甚至不必真的一步步走——随机抽 \(x,t,\epsilon\)，kernel 直接构造 \(z_t\)，网络根据 \(z_t\) 和 \(t\) 预测 \(\epsilon\)。生成才反过来：从 \(z_T\sim\mathcal N(0,I)\) 出发，一步步把噪声变成图。

真正难的反向 \(q(z_{t-1}\mid z_t)\) 依赖未知的数据分布；训练时多给的那个 \(x\) 把它变成闭式高斯。ELBO 在一串高斯假设下变成加权噪声回归，实践再丢掉权重。所以你在代码里看到的 MSE，是三次简化之后的对象，不是 likelihood 本身。

Forward 固定，Reverse 学习；Training 单个随机 \(t\)，Sampling 多步串行。U-Net 负责同尺寸去噪，time embedding 声明当前噪声等级。它用慢采样换来稳定训练、高质量和灵活条件——第 14 章那一组轴上，这是一种很明确的赌法。

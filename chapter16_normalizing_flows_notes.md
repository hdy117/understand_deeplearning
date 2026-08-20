# 第 16 章：归一化流（Normalizing Flows）

> 书：《Understanding Deep Learning》Ch.16  
> 前置：第 15 章的 GAN 会生成，通常不会给样本写 \(p(x)\)。  
> 本章：若坚持要精确 likelihood，映射必须可逆；层的设计被这四个字卡住。  
> 后续：可逆太苛刻。第 17 章用 ELBO 换回不可逆的 decoder。

---

## 0. 要打分，就得能把密度跟着走

GAN 把 \(z\) 送进任意深层网就出图。密度被弄丢了：非线性一堆，不知道体积被拉伸了多少，\(p(x)\) 写不出来。Normalizing flow 把这句话反过来：

> **从一个好算的 base density \(p(z)\) 出发，用可逆变换把它拉成数据的形状；体积变化由 Jacobian 精确记账。**

采样走正向 \(x=f(z,\phi)\)。打分走反向 \(z=f^{-1}(x,\phi)\)，再补上拉伸因子。两个方向都要快，所以每一层都不是普通的全连接+ReLU。

母问题：

> **怎样把「任意灵活的深层映射」收成「可逆、可算行列式」的一层层变换，仍然能拟合复杂密度？**

调查链：

```text
GAN 不会打分
        │
        ▼
可逆变换把简单密度拉成复杂密度
        │
        ▼
概率质量守恒：拉伸则变矮（|∂x/∂z|）
        │
        ▼
正向采样、反向算 likelihood，所以 f 必须是双射
        │
        ▼
层要同时满足：够表达、可逆、逆要快、det 要快
        │
        ▼
linear / elementwise 是积木；coupling / AR / residual 才真正弯
        │
        ▼
多尺度：一部分维度提前变成 z，计算量下降
        │
        ▼
样本往往不如 GAN / Diffusion → 第 17 章放弃精确可逆，改用下界
```

---

## 1. 一维：面积不变，高度必须跟着斜率走

Base density \(p(z)\) 常取标准正态。变换 \(x=f(z,\phi)\) 把 \(z\) 轴拉成 \(x\) 轴。把 base 切成等宽小条：每条里的概率质量变换后必须还在，所以被拉宽的地方高度下降，被压窄的地方高度上升，曲线下面积始终是 1。

精确写出来：

\[
p(x\mid\phi)
=
\left|
\frac{\partial f(z,\phi)}{\partial z}
\right|^{-1}
p(z),
\qquad
z=f^{-1}(x,\phi).
\]

导数绝对值大于 1，密度变矮；小于 1，密度变高。\(p(z)\) 好算，难的是 \(z\) 和那个导数。于是 \(f\) 必须可逆。正向叫 **generative direction**（从正态造数据），反向叫 **normalizing direction**（把复杂的 \(x\) 变回正态的 \(z\)）。「Normalizing flow」这个名字来自后一个方向。

学习仍是第 5 章的最大似然。\(I\) 个独立样本：

\[
\hat\phi
=
\arg\min_\phi
\sum_{i=1}^{I}
\left(
\log\left|\frac{\partial f(z_i,\phi)}{\partial z_i}\right|
-
\log p(z_i)
\right),
\qquad
z_i=f^{-1}(x_i,\phi).
\]

第一项惩罚「把体积拉得太大」：乱拉伸会把密度摊薄，数据点上的 likelihood 就掉。第二项要 \(z_i\) 落在 base 的高密度区。两头一起夹，变换才会既把数据送到正态里，又记住路上体积怎么变。

---

## 2. 多维：一行变一张 Jacobian

\(z,x\in\mathbb{R}^D\)，\(x=f(z,\phi)\)。一维的绝对值导数换成 Jacobian 行列式的绝对值：

\[
p(x\mid\phi)
=
p(z)\,
\left|
\det\frac{\partial f(z,\phi)}{\partial z}
\right|^{-1}.
\]

\(D\times D\) 的行列式一般是 \(O(D^3)\)，每个样本、每个训练步都要算，普通深层网直接不可用。深度网是一层层叠的 \(f=f_K\circ\cdots\circ f_1\)，链式法则把总 Jacobian 写成各层 Jacobian 的乘积，行列式变成各层行列式的乘积。于是问题下放到每一层：

**一层合格的 flow，要同时做到四件事：**

1. 全体叠起来，要能把多元正态拉成足够任意的密度；
2. 每一层是双射：一个输入对一个输出，否则反向说不清；
3. 反向要快（闭式或廉价算法）——训练每一步都要对数据做 \(z=f^{-1}(x)\)；
4. 正向或反向的 \(\log|\det J|\) 要快。

普通 ReLU 网四条里一条都不好说。下面那些「奇奇怪怪」的层，不是审美，是这四条逼出来的。

---

## 3. 积木：线性与逐点

**Linear flow** \(f(h)=\Omega h+\beta\)。可逆要求 \(\Omega\) 可逆；\(\det J=\det\Omega\)。把 \(\Omega\) 参数化成 LU 分解，行列式是对角元的乘积，\(O(D)\) 而不是 \(O(D^3)\)。线性本身不够弯，只能旋转、拉伸、平移这团正态。

**Elementwise flow** 对每个坐标单独套一个单调非线性。Jacobian 是对角的，行列式又是对角元的乘积，反向可以逐点做。仍然不够：坐标之间没有混合，学不会「第 2 维该跟着第 1 维弯」。

真正能用的层，都是在「一部分算得起 det」的前提下，让坐标互相说话。

---

## 4. Coupling：一半先不动

把向量切成两半 \(h=(h_a,h_b)\)。\(h_a\) 原样通过；\(h_b\) 被一个以 \(h_a\) 为条件的逐点变换 \(g(\cdot,\phi(h_a))\) 修改。\(g\) 对 \(h_b\) 是 elementwise 且可逆（常见：仿射 \(h_b\mapsto h_b\odot s(h_a)+t(h_a)\)，RealNVP）。

反向：先拿到 \(h_a\)（它没变），再解开 \(h_b\)。Jacobian 是下三角（或置换后下三角），行列式只看 \(g\) 对 \(h_b\) 的那些对角导数，\(s(h_a)\) 再复杂也不进行列式——这是 coupling 能把普通深度网塞进 \(s,t\) 的原因：那两张网不必可逆。

表达力的缺口也清楚：一次 coupling 只动一半。所以要交错切开、叠很多层，让每个坐标都有机会当过 \(h_b\)。

---

## 5. Autoregressive：每次只多解一维

第 \(d\) 维只依赖前 \(d-1\) 维：

\[
x_d
=
g\bigl(z_d;\,\phi(x_{1:d-1})\bigr).
\]

Jacobian 又是三角的。正向（或反向，取决于你把 AR 写在哪一头）必须串行：先 \(x_1\)，再 \(x_2\)，……。**Masked autoregressive flow (MAF)** 让 likelihood（反向）并行、采样串行；**Inverse autoregressive flow (IAF)** 反过来，采样快、打分慢。同一套三角结构，把「哪一头并行」当成超参数。第 14 章的采样速度轴，在这里变成层的朝向。

---

## 6. Residual：加残差，再把可逆买回来

\(f(h)=h+g(h,\phi)\)。第 11 章的残差在这里出现，是因为恒等已经可逆，只要 \(g\) 不太猛，整体仍是双射。

**iRevNet** 把通道切开做 additive coupling，逆是闭式的。**iResNet** 要求 \(g\) 是压缩映射（Lipschitz \(<1\)，常靠 clip 权重），逆用不动点迭代；\(\log|\det(I+\partial g/\partial h)|\) 没有闭式，用

\[
\log|I+A|
=
\mathrm{tr}\log(I+A)
=
\sum_{k=1}^{\infty}
\frac{(-1)^{k-1}}{k}\mathrm{tr}(A^k)
\]

截断，再用 Hutchinson 估计 \(\mathrm{tr}(A)\approx\mathbb{E}_\epsilon[\epsilon^\top A\epsilon]\)。精确 likelihood 在这里已经开始掺近似——缺口指向第 17 章：若反正要近似，何必坚持可逆。

---

## 7. Multi-scale：不必所有维都走完全程

图像有几万维。每一层都对全维做可逆变换，又慢又浪费：很多维度其实是纹理噪声，早就可以当成已经「正态」的 \(z\) 抽走。Multi-scale flow（RealNVP / Glow 一路）在若干深度把一部分通道直接接到最终的 \(z\)，剩下的继续变。总 Jacobian 仍是分块三角，行列式好乘；采样时这些提前抽出的维从正态补回来。

这和第 18 章 diffusion 的多尺度不是同一件事，但直觉类似：不必用同一套容量伺候所有频率。

---

## 8. 能做什么，以及精确 likelihood 买不到的东西

**密度模型。** 异常检测、给样本打分、和别的模型比 test NLL：这是 GAN 直接做不了的。比较时必须对齐预处理和测量约定（第 14 章）：连续密度可以大于 1，换一套 dequantization，数字就能换一名。

**合成。** 从正态抽 \(z\)，走正向。也可以像 GAN 那样做 truncation，从更靠近均值的 \(z\) 抽，换质量丢多样性。原书写得很干脆：样本质量通常不如 GAN 和 diffusion。不知道是可逆层的根本限制，还是这条线投的研究更少。Glow 的插值和第 1、15 章同一套：两张真图反向得到 \(z\)，中间线性插值，再正向走回来。

**去逼近另一个好打分、不好采样的密度。** Flow 当 student，目标密度当 teacher。自己造的样本知道 \(z\)，不必反向，于是可以用「逆很慢」的 MAF。损失是 reverse KL，逼两头的 likelihood 对齐。第 17 章的 ELBO 也是「用一个好采样的 \(q\) 去追一个不好算的 \(p\)」，亲戚关系在这一节已经能看见。

---

## 9. 自测

1. GAN 丢掉了哪一句，flow 用 Jacobian 把它买回来？  
2. 一维公式里，\(|\partial f/\partial z|>1\) 时密度升高还是降低？面积为什么必须守恒？  
3. 为什么采样看正向、likelihood 看反向？「normalizing」指哪一个方向？  
4. NLL 里 \(\log|\partial f/\partial z|\) 和 \(-\log p(z)\) 各自在惩罚什么？  
5. 多维时普通深层网的 \(\det J\) 贵在哪？深度网怎样把问题下放到每一层？  
6. 合格的一层要同时满足哪四条？缺了「逆要快」会在训练的哪一步爆掉？  
7. 线性流的 LU 技巧省的是什么？它为什么单独不够？  
8. Coupling 里 \(s(h_a),t(h_a)\) 为什么可以是普通不可逆网络？行列式只看到谁？  
9. MAF 和 IAF 在「采样 vs 打分」上怎样对调？这和第 14 章哪一条轴是同一句话？  
10. iResNet 已经在近似 \(\log|\det|\)。这件事怎样预支第 17 章的口子？  
11. Multi-scale 提前抽出一些通道，为什么总 Jacobian 仍然好算？  
12. 精确 likelihood 为什么仍不能保证样本比 GAN 好看？比较两个 flow 的 NLL 之前要先对齐什么？

---

## 10. 合上书再看一眼

Flow 把生成写成一次可逆的换坐标：简单密度走正向变成数据，数据走反向变回简单密度，体积由 \(\lvert\det J\rvert\) 记账。于是同一套参数既能采样又能打精确的 \(p(x)\)。代价是层必须是双射，而且逆和行列式都得便宜——coupling、自回归、残差流全是被这四条逼出来的形状，不是新的生成哲学。

精确打分换不来 GAN 那种锐度，也换不来后面 diffusion 那种稳。可逆本身就是很强的结构先验。下一章把这根弦放松：decoder 可以任意弯、不必可逆，似然改成一个可算的下界。那个下界叫 ELBO，生成模型本身仍然只有 \(p(z)p_\theta(x\mid z)\)，encoder 只是训练时的客人。

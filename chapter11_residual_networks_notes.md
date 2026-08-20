# 第 11 章：残差网络（Residual Networks）

> 书：《Understanding Deep Learning》Ch.11  
> 前置：第 10 章把局部性与权值共享写进 convolution；VGG 说明多层小 kernel 能扩大 receptive field，但更深的 plain stack 更难训。  
> 本章：怎样让新层默认「先别破坏已有表示」，只学习必要修正？  
> 后续：第 12 章 Transformer。残差让复杂 block 可以反复堆叠之后，还缺一件事：token 怎样按内容决定读谁。  
> 对照实现：`resnet50/resnet_50.py`（canonical ResNet-50，post-activation）。

---

## 0. 更深，按理不该更差

第 10 章把网络加深，是为了让简单的局部探测器组合成 part，再组合成 object。VGG 已经把这句话做成了工程：多叠几层 $3\times 3$，感受野会变大，中间还能多插几次非线性。那再叠下去——三十层、五十层、一百层——至少不该比浅网络差。多出来的层如果暂时没用，做成 identity 就行：输入原样出去，浅网络已经会的事情，深网络照样会。

实践里却经常看见相反的事。He et al. 2015 把它叫做 **degradation problem**：更深的 plain network，training error 更高，test error 也更高。

```text
deeper plain net: higher train error AND higher test error
```

这和 overfitting 不是同一张病历。过拟合是「训练集上更好、测试集上更差」，你该去怀疑 generalization。Degradation 是两边一起变差，首先该怀疑的是 optimization 和 architecture：好答案未必不存在，而是 optimizer 很难顺着一长串 nonlinear transformations 把它找出来。

| 现象 | Train | Test | 首先怀疑 |
|------|-------|------|----------|
| Overfitting | 更好 | 更差 | generalization |
| Degradation | 更差 | 更差 | optimization / architecture |

所以本章只追一个母问题：

> **怎样重新参数化每一层，使「什么也不做」变成容易表达、容易优化的默认状态，从而让深度增加时好解仍能被 SGD 找到？**

后面每一节只补这条链上的一个缺口：

```text
plain deep net: worse train error (not overfitting)
        │
        ▼
each layer rewrites the whole representation;
identity is hard to hit inside nonlinear composition
        │
        ▼
reparameterize: h ← h + F(h)   (learn a perturbation of identity)
        │
        ▼
add is an interface: shapes must match;
else projection — not a true identity
        │
        ▼
backward: I + J_F  (a bypass, not a free pass)
        │
        ▼
magnitudes can accumulate → BatchNorm / LayerNorm
        │
        ▼
BN/ReLU before vs after Add: is the identity highway clean?
        │
        ▼
ResNet-50: counting, Bottleneck, stage shifts, code
        │
        ▼
still missing: content-dependent routing  → Ch.12
```

Residual 的设计目标可以先记成一句，后面所有公式都在把它变成可加、可反传、可堆叠的计算图：

> 新 block 默认接近 identity，只负责学习「应该在原表示上改多少」。

---

## 1. Plain network 为什么连 identity 都难？

若多出来的层可以做成 identity，理论上深网可以复现浅网，train error 不该更差。那 SGD 为什么找不到这条路？因为 plain composition 问错了问题：每一层都必须交出**整张**下一表示，而不是「保留还是修改」。

### 1.1 前向只有一条重写链

\[
h_1=f_1(x),\qquad h_2=f_2(h_1),\qquad h_3=f_3(h_2).
\]

Data flow 没有旁路：

```text
x → f1 → f2 → f3 → ... → output
```

$f_\ell$ 一旦学歪，后面所有层只能在被破坏的表示上继续算。想让某一层「什么也不做」，等于要求一个带 ReLU、convolution、随机初始化的非线性映射精确地学会 $f(h)=h$。这在函数类里往往做得到，在 optimization landscape 里却很贵：你是在所有可能的重写里，搜那一张恰好等于原文的纸。

### 1.2 反向必须穿过整条 Jacobian 连乘

Loss $\mathcal{L}$ 对早期表示的导数，是一串局部 Jacobian 的乘积：

\[
\frac{\partial h_L}{\partial h_0}
=
\frac{\partial h_L}{\partial h_{L-1}}
\cdots
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_1}{\partial h_0}
=
\prod_{\ell=0}^{L-1} J_{f_\ell}.
\]

每一项 $J_{f_\ell}$ 的奇异值只要经常小于 1，连乘就会 **vanishing gradient**：早期层几乎收不到更新。经常大于 1，就 **exploding**。还有第三种更阴的病，叫 **shattered gradient**：幅度未必归零，但参数或输入稍变，gradient 的方向就像噪声一样跳——深层 plain net 的反向信号不只是变弱，还会变乱。

这三件事是同一条结构的不同症状：forward information 和 backward gradient 都必须穿过所有 transformations，没有旁路。第 7 章用 initialization 控制单层的尺度，第 9 章用 regularization 管过拟合；它们都没有改计算图的拓扑。Degradation 逼我们改拓扑。

---

## 2. 换一个问法：只学对 identity 的扰动

Plain layer 问的是「下一张表示完整是什么」：

\[
h_{\ell+1}=F_\ell(h_\ell).
\]

Residual block 改成：

\[
h_{\ell+1}=h_\ell+F_\ell(h_\ell;\phi_\ell).
\]

```text
                 ┌──────── identity path ────────┐
h_l ─────────────┤                               + ──→ h_{l+1}
                 └→ learned branch F_l(h_l) ─────┘
```

$h_\ell$ 是已经有的 representation，原样保留；$F_\ell$ 是需要学习的 correction / residual；相加之后才是下一表示。层不再被要求发明一个新世界，只被要求回答：在现有表示上，这一步该补多少、删多少。

用改文章来看同一件事。Plain 思路是每一轮从空白纸重写整篇；residual 思路是保留当前稿，只写修改意见——补一个例子、删一句废话、调一段顺序。如果这一轮不知道怎么改，只要

\[
F_\ell(h_\ell)\approx 0,
\]

就有 $h_{\ell+1}\approx h_\ell$。「什么也不做」从「在所有非线性映射里命中恒等」变成了「让残差支路输出接近 0」。对随机初始化的卷积来说，后者便宜得多。

同一个式子也可以读成表示空间里的一小步：

\[
h_{\ell+1}=h_\ell+\Delta h_\ell,
\]

很像 Euler 离散

\[
h(t+\Delta t)=h(t)+F(h(t))\,\Delta t.
\]

加深不再 magically 多几级「全新坐标系」，而更像沿同一条 representation trajectory 多走几步可优化的局部修正。这条视角后来连到 Neural ODE，第一遍不必展开；记住「depth ≈ 修正步数」就够。Stage 之间的 downsample 仍会换分辨率——那是后面 ResNet-50 要走的换挡，不是对这句话的否定。

这里有一个必须钉死的分工。Plain deep net 在表达能力上往往已经能表示类似函数；degradation 时 **train error 也更高**，说明瓶颈通常不在「能不能表示」，而在「SGD 容不容易走到」。Residual 首先是 **再参数化**：把目标从「学整个映射」改成「学残差」，使 $F\approx 0$ 成为廉价默认。Architecture 的杠杆是扩大「容易找到的好解集合」，而不只是扩大「可表示函数集合」。起步时还可以把这个默认写进初始化——那要等到 BatchNorm 的 $\gamma$ 出场，才知道旋钮在哪。

加法立刻带来一个新约束，上一节的公式假装它不存在：两项必须能加在一起。

---

## 3. 加法是接口，不是一根随便拉的线

\[
\operatorname{shape}(h_\ell)=\operatorname{shape}\bigl(F_\ell(h_\ell)\bigr)
\]

不成立就不能 element-wise add。Skip connection 看起来只是「跨层拉一根线」，实际上它是一条 tensor 契约。

Shape 不变时，shortcut 可以是真 identity：

```text
h:    [N, 64, 56, 56]
F(h): [N, 64, 56, 56]
h + F(h)：合法，S(h) = h
```

Channel 数或空间分辨率一变，直接相加非法：

```text
h:    [N, 64, 56, 56]
F(h): [N,128, 28, 28]
直接相加：非法
```

这时改成 **projection shortcut**

\[
h_{\ell+1}=P(h_\ell)+F_\ell(h_\ell),
\]

$P$ 常用 stride-2 的 $1\times 1$ convolution，同时改 channels 与 spatial size。代码里这个分支往往仍叫 `identity`，但数学上已经不是 $S(x)=x$。Projection 是换坐标系的工具，不是高速公路本身。

He et al. 2016 后来强调：真正帮深层优化的，是**尽量多的真 identity shortcut**。Projection 块把 $I$ 换成了 $J_P$，直接通路上多了一层线性变换。ResNet-50 里 16 个 Bottleneck 只有 4 个走 projection（每个 stage 的首块），其余 12 个才是干净旁路——比例不是偶然。

U-Net、DenseNet 也有人随口叫 skip，契约却不同。那是走完 ResNet-50 之后的对照；眼下只需承认：没有 shape 对齐，就没有 $h+F(h)$ 这件事。

前向有了旁路。反向呢？梯度会不会自动沿 identity 回家？

---

## 4. 反向：多出来的是 $I$，不是免死金牌

先看没有额外非线性挡在加法之后的干净残差：

\[
h_{\ell+1}=h_\ell+F_\ell(h_\ell).
\]

对 $h_\ell$ 求导：

\[
\frac{\partial h_{\ell+1}}{\partial h_\ell}
=I+J_{F_\ell}.
\]

Plain layer 只有 $J_F$；residual block 多了 identity term $I$。Backward 可以走 learned branch，也可以走一条不依赖 $F$ 学得好不好的贡献：

```text
Backward gradient
   ├── learned branch J_F
   └── identity contribution I
```

叠 $L$ 层，chain rule 变成

\[
\frac{\partial h_L}{\partial h_0}
=\prod_{\ell=0}^{L-1}\bigl(I+J_{F_\ell}\bigr).
\]

对比 plain 的 $\prod J_{f_\ell}$：即使某个 $J_F$ 的奇异值很小，$I+J_F$ 仍然靠近 $I$，连乘不会仅仅因为「残差支路暂时没用」而塌成 0。这是 residual 改善可训练性时，梯度层面最硬的一句。

另一个视角是把网络展开。两个 block：

\[
h_2=h_0+F_0(h_0)+F_1(h_1).
\]

原始表示 $h_0$ 对后面有直接的加性贡献。一般地

\[
h_L=h_\ell+\sum_{i=\ell}^{L-1}F_i(h_i),
\]

于是

\[
\frac{\partial\mathcal{L}}{\partial h_\ell}
=
\frac{\partial\mathcal{L}}{\partial h_L}
\left(
I+\sum_{i=\ell}^{L-1}\frac{\partial F_i}{\partial h_\ell}
\right).
\]

$\partial\mathcal{L}/\partial h_L$ 乘在 $I$ 上的那一项，就是从 loss 直达早期表示的通路。展开后你会看见许多不同长度的 computation path：有的几乎不经过 residual branch，有的叠了很多 $F$。浅路径帮助 forward 信号和 backward gradient 活着。

但这里必须刹车。Veit et al. 把 ResNet 读成「许多深浅不同的路径的集合」，这是有用的图景，**不是**说一个 ResNet 等于许多个彼此独立训练的 ensemble。那些 path 共享 parameters 和 representations，梯度是耦合的。更不是：有了 $I$，gradient 就永远不消失、不爆炸。

三件仍然成立的限制：

1. 公式写的是 $h_{\ell+1}=h_\ell+F_\ell(h_\ell)$。若加法之后还有 ReLU（经典 post-activation ResNet），旁路会被非线性拧一次，$F\to 0$ 时输出接近 $\operatorname{ReLU}(h)$ 而不是 $h$。§7 专门处理这个顺序问题。
2. Projection shortcut 把 $I$ 换成 $J_P$。Highway 的干净程度取决于有多少块是真 identity。
3. $I+J_F$ 的特征值仍可能落在危险区；learning rate、初始化、残差支路的尺度都还在。Architecture 让 optimization 更友好，不是取消问题。

旁路还有另一面。信号容易活，幅度也容易涨。

---

## 5. 旁路修好了，尺度可能炸

假设两项近似独立、均值为零：

\[
\operatorname{Var}(h_{\ell+1})
\approx
\operatorname{Var}(h_\ell)+\operatorname{Var}\bigl(F_\ell(h_\ell)\bigr).
\]

若 residual branch 的方差和输入相当，理想化情况下每个 block 都可能近似把幅度翻一倍。连续十次是 1024 倍。真实网络并不满足独立、等方差，这个思想实验只说明结构本身带了一股 **magnitude accumulation**：skip 改善了信息传递，也给了 activation 一条可以一路加下去的通道。

常见稳定手段都在对付这股力：normalization、residual branch scaling、更谨慎的 initialization、pre-activation、以及后来 Transformer 里的 LayerNorm / RMSNorm。原书这一章把镜头对准 Batch Normalization——它是 ResNet 能堆起来的另一半契约。没有它，只谈 $I+J_F$ 是半截高速公路。

---

## 6. BatchNorm：稳住尺度，但不要只背「均值 0、方差 1」

上一节要的是：在不断 $h+F(h)$ 的同时，别让表示的数值漂走。BatchNorm 的手段是：用当前 mini-batch 的统计量把 activation 拉回标准尺度，再允许网络用两个学出来的数把尺度拿回来。

### 6.1 Training 时对一组 activation 做什么

对 $x_1,\ldots,x_M$：

\[
\mu_B=\frac1M\sum_i x_i,
\qquad
\sigma_B^2=\frac1M\sum_i(x_i-\mu_B)^2,
\]

\[
\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},
\qquad
y_i=\gamma\hat x_i+\beta.
\]

$\epsilon$ 防止方差太小除零。$\gamma,\beta$ 是 learned scale 与 shift：normalize 若永远锁死 mean 0、variance 1，会把网络能表达的仿射自由度砍掉；$\gamma,\beta$ 把这自由度还回去。所以「BN 之后永远是标准正态」是错的——那只是 $\hat x$，不是 $y$。把 residual 支路最后一层 BN 的 $\gamma$ 初始化为 0，起步时 $F(x)\approx 0$，整个 block 接近 identity：本仓库的 `zero_init_residual` 就是这个旋钮。结构和初始化在做同一件事。

手算一个最小 batch $[1,3]$：mean 为 2，variance 为 1，忽略 $\epsilon$ 后 $\hat x=[-1,1]$。之后 $\gamma,\beta$ 仍可再缩放和平移。

### 6.2 CNN 里在哪些轴上统计

输入 shape $[N,C,H,W]$。典型 BatchNorm2d 对**每个 channel 单独**统计，在 batch 与全部空间位置上汇总：

```text
channel c 的 mean / variance
= all N examples × all H × W positions
```

不同 channels 不混在同一对 $(\mu,\sigma^2)$ 里。这和 convolution 的「一个 channel 一种探测器」一致：你标准化的是同一种 feature detector 在整张图、整个 batch 上的响应，而不是把边缘探测器和颜色探测器搅在一起。

### 6.3 Training 与 inference 是两套 data flow

Training：

```text
current mini-batch
   → batch mean / variance
   → normalize
   → affine with γ, β
   → update running statistics
```

Inference：

```text
single example or arbitrary batch
   → stored running mean / variance
   → normalize
   → same γ, β
```

Inference 不能继续依赖「当前 batch 的偶然同伴」。否则同一样本会因和谁一起推理而改变结果——这不是随机正则，是 bug。Train 很好、`model.eval()` 突然崩，第一嫌疑常常是 BN 的 mode 和 running statistics，而不是 residual 本身。

### 6.4 它帮了什么，边界在哪

可能的收益：控制 activation scale，从而允许更积极的 optimization；mini-batch 统计量带来一点 stochastic regularization；经验上会改变 loss geometry（原始论文讲 internal covariate shift，后来有工作更强调平滑 landscape——「为什么有效」至今不是一句口号能盖住的）。

代价同样具体。小 batch 时 $(\mu_B,\sigma_B^2)$ 很吵。Distributed training 要处理跨设备 statistics（SyncBN 一类）。对 variable-length sequence 和 autoregressive inference，一个 token 往往没有「同一时刻的一大群空间位置」可平均，BatchNorm 的轴就不自然。这时更常见的是 **LayerNorm**：对**单个** token 的 feature 维做归一化，不看 batch 里别的例子。第 12 章的 Transformer 基本走这条路；本章只要记住：normalization 家族的职责是尺度稳定，统计轴必须匹配数据的独立性结构。

和 residual 默认 identity 接在一起，训练不稳时不必先怀疑整套哲学。更短的检查顺序是：

| 症状 | 先查 |
|------|------|
| 加深后 train 也变差 | degradation / 是否真的走了 $h+F$ / 初始化 |
| Add 时报 shape error | channels 或 H/W 没对齐 |
| 训练好、eval 突然差 | BN 的 train/eval mode、running statistics |
| Activation 随 depth 变大 | residual 支路是否过强、缺 normalization |
| 小 batch 波动大 | BN statistics 噪声；考虑换 GroupNorm / LayerNorm |
| Train 好、test 差 | generalization，不只是 residual path |

尺度稳住了，还剩一个顺序问题：BN 和 ReLU 相对加法点放在哪，identity 还是不是 identity。

---

## 7. Add 前后放什么：高速公路上有没有收费站

「Pre / post-activation」说的不是网络的前后端，而是：

> **相对 residual 加法点 Add，BN/ReLU 放在 Conv 的前面还是后面。**

标准 ResNet-50（He et al. 2015，torchvision 默认，本仓库 `resnet_50.py`）是 **post-activation**。He et al. 2016 *Identity Mappings in Deep Residual Networks* 把 BN→ReLU 挪到 Conv 之前，叫做 **pre-activation**。§4 写干净公式 $I+J_F$ 时，其实已经在假设加完之后不再拧一把；经典 ResNet-50 并没有完全满足这个假设。

### 7.1 Post-activation：加完再 ReLU

以 Bottleneck、进出 shape 相同（真 identity）为例：

```text
x ──────────────────────── identity path ─────────────────┐
                                                          │
  F(x): Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU         │
        → Conv1x1 → BN ───────────────────────────────────+
                                                          │
                                                        ReLU → y
```

逐步写：

```text
out = ReLU(BN(Conv1(x)))
out = ReLU(BN(Conv2(out)))
out = BN(Conv3(out))       # last conv in F: BN only, no ReLU yet
y   = ReLU(out + x)        # nonlinearity after Add = post-activation
```

Identity 在**加之前**是干净的，加之后整段 $x+F(x)$ 还要过 ReLU。$F\to 0$ 时 $y\approx\operatorname{ReLU}(x)$，不是 $x$。负的特征会在每一个 block 出口被切掉——高速公路上每过一座城收一次费。

### 7.2 Pre-activation：旁路全程无 BN/ReLU

```text
x ──────────────────── identity path (no BN/ReLU) ────────┐
                                                          │
  F: BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3            │
     → BN → ReLU → Conv1x1 ───────────────────────────────+
                                                          │
                                                          y   # usually no ReLU after Add
```

```text
out = Conv1(ReLU(BN(x)))
out = Conv2(ReLU(BN(out)))
out = Conv3(ReLU(BN(out)))
y   = out + x
```

$F$ 在「先归一化、再激活」的空间里学修正；旁路 $S(x)=x$ 直通到下一 block。$F\to 0$ 时 $y\approx x$。深层时这条更干净的 identity mapping 往往更好训——这正是 2016 年那篇论文的动机。

| 维度 | Post-activation | Pre-activation |
|------|-----------------|----------------|
| BN/ReLU 相对 Conv | 多在 Conv 之后 | 多在 Conv 之前 |
| Add 之后 | 通常再 ReLU | 通常直接输出 |
| $F\to 0$ 时 | $y\approx\operatorname{ReLU}(x)$ | $y\approx x$ |
| Identity 路径 | 加前干净，加后仍被 ReLU | 整条旁路最干净 |
| 出处 | 原版 ResNet、本仓库 ResNet-50 | ResNet-v2 / pre-act ResNet |

「有 residual」不够，还要问：**Add 前后有没有 BN/ReLU 挡路。** 第 12 章 Transformer 里 Pre-LN / Post-LN 是同一类问题换了一套归一化：norm 在残差支路入口还是在加法之后，决定深层 stack 的 identity 是否干净。账记在这里，公式放到那里。

理论契约齐了：学 $\Delta$、shape 对齐、梯度有 $I$、尺度有 BN、顺序决定旁路干不干净。下面把它们装进一个能跑的 ResNet-50。

---

## 8. 一条具体的高速公路：ResNet-50

仓库实现：`resnet50/resnet_50.py`。输入默认 ImageNet $[N,3,224,224]$。读这一节时，把「50 是什么、每一张 feature map 有多大、哪些块是真 identity、浅层和深层的 $F$ 在改什么」当成同一条路，而不是四张独立的表。

### 8.1 「50」数的是带权层，不是 50 个 block

ResNet-50 的 50 = **weighted layers**（Conv + 最后的 fc），按原论文 / 通行约定：

| 不计进 50 | 计进 50 |
|-----------|---------|
| BN、ReLU、MaxPool、AdaptiveAvgPool | stem 的 $7\times 7$ Conv |
| 只做形状对齐的 projection $1\times 1$（通常不计） | 每个 Bottleneck 里 3 个 Conv |
| | 最后的 fc Linear |

四个 stage 的 Bottleneck 个数固定为 `[3, 4, 6, 3]`：

\[
1_{\text{stem}}
+3\times(3+4+6+3)
+1_{\text{fc}}
=1+48+1=50.
\]

| 部分 | 结构 | 计入 50 |
|------|------|---------|
| stem | $1\times 7\times 7$ conv | 1 |
| layer1 | 3 Bottleneck $\times$ 3 conv | 9 |
| layer2 | 4 Bottleneck $\times$ 3 conv | 12 |
| layer3 | 6 Bottleneck $\times$ 3 conv | 18 |
| layer4 | 3 Bottleneck $\times$ 3 conv | 9 |
| head | 1 $\times$ fc | 1 |
| 合计 | | **50** |

同系列只是在换「每块几层卷积」和「每 stage 几块」：

| 名字 | Block | 每 stage blocks | 层数 |
|------|-------|-----------------|------|
| ResNet-18 | Basic（2 个 $3\times 3$） | `[2,2,2,2]` | 18 |
| ResNet-34 | Basic | `[3,4,6,3]` | 34 |
| **ResNet-50** | **Bottleneck（3 conv）** | **`[3,4,6,3]`** | **50** |
| ResNet-101 | Bottleneck | `[3,4,23,3]` | 101 |
| ResNet-152 | Bottleneck | `[3,8,36,3]` | 152 |

50 不是 50 个 residual block——只有 $3+4+6+3=16$ 个 Bottleneck。也不是所有模块的总数。Stage 个数与 ResNet-34 相同；50 变深，主要是因为每块从 2 个 $3\times 3$ 换成了 3 个卷积，并且通道更宽。

### 8.2 为什么是 Bottleneck，而不是两层 $3\times 3$

较浅的 ResNet 用 standard block：两个 $3\times 3$。更深时改成 Bottleneck：

```text
1×1 reduce channels
  → 3×3 spatial processing   # 昂贵的空间卷积在窄通道上做
  → 1×1 expand channels
  → add shortcut
```

`expansion=4`：内部窄通道是 $C$，和 shortcut 相加的宽通道是 $4C$。$3\times 3$ 的计算量跟 channel 的平方有关，先压窄再卷积是在买深度：同样的预算下可以叠更多 identity 旁路。ResNeXt 后来在这个窄空间里再切成多组 parallel transformations（cardinality）；WideResNet 则少谈深度、多加 width。两条路都承认：residual 让「加深」不再那么吓人之后，还要在 width 与分组上花钱。第一遍记住 Bottleneck 的契约就够。

### 8.3 Shape 主线：降采样只发生在 stage 门口

完整的 residual 公式，连上 projection：

\[
y=\operatorname{ReLU}\bigl(F(x)+S(x)\bigr),
\qquad
S(x)=
\begin{cases}
x, & \text{H/W 与 channels 都相同},\\
\operatorname{BN}\bigl(\operatorname{Conv}_{1\times 1,\,s}(x)\bigr), & \text{否则}.
\end{cases}
\]

本仓库把 stride 放在 Bottleneck 的 **$3\times 3$** 上（与 torchvision 一致；2015 原论文把 stride 放在第一个 $1\times 1$，会丢掉一部分空间信息）。三条规律先放在路肩上，下面用数字把它们走实：

1. 降采样只发生在 stage 交接的第一个 block（`layer2.0 / 3.0 / 4.0`），靠 $3\times 3$ stride=2；stage 内后续 block 全部 stride=1。
2. 可加条件是 $\operatorname{shape}(F)=\operatorname{shape}(S)$。同 shape → 真 identity；否则 → $1\times 1$ projection（需要减半空间时带 stride=2）。
3. Bottleneck 通道套路：宽 → 窄（$1\times 1$）→ $3\times 3$ → 宽 $\times 4$，再和 shortcut 加。

```text
[N,    3, 224, 224]
  7×7 s2 ──────────────────────────► [N,   64, 112, 112]  # stem: spatial /2, raise C
  BN+ReLU, maxpool s2 ─────────────► [N,   64,  56,  56]  # /2 again; residual stages start
  layer1 ×3 (first block: proj, C↑)► [N,  256,  56,  56]  # channels only; grid still fine
  layer2 ×4 (only .0: 3×3 s2+proj) ► [N,  512,  28,  28]  # spatial /2, channels ×2
  layer3 ×6 (only .0: 3×3 s2+proj) ► [N, 1024,  14,  14]
  layer4 ×3 (only .0: 3×3 s2+proj) ► [N, 2048,   7,   7]
  GAP + fc ────────────────────────► [N, 1000]
```

Stem 的 $7\times 7$ 先用大感受野抓住低层边缘和纹理，并把图缩小到算得起的格子上。真正的 residual 游戏从 $56\times 56$ 开始。

**layer1** 三块，stride 全是 1。`layer1.0` 把 64 通道扩到 256，H/W 仍是 56，所以 shortcut 是 $1\times 1$ stride-1 projection——只换通道，不换格子。`layer1.1`、`layer1.2` 进出都是 `[N,256,56,56]`，第一次出现真 $x+F(x)$。三块做完，你仍在细网格上；这里的 $F$ 更像在已有的边缘/颜色图上补更干净的局部描述，而不是写出「这是狗」。细节还够用时，$F\approx 0$ 很有用：不要把 $56\times 56$ 的精细结构毁掉。

**layer2** 是理解「换挡 + 再 refine」的最小完整例子。只有首块 stride=2。

```text
layer2.0     in [N, 256, 56, 56]

  F:  1×1 s1 → [N, 128, 56, 56]      # bottleneck: cheap 3×3
      3×3 s2 → [N, 128, 28, 28]      # the only spatial /2 in this stage
      1×1 s1 → [N, 512, 28, 28]      # expansion=4

  S:  1×1 s2 + BN → [N, 512, 28, 28] # not true identity: H/W and C both change

  y = ReLU(F+S) → [N, 512, 28, 28]

layer2.1–2.3   true x+F(x), frozen shape [N, 512, 28, 28]
```

```text
                 ┌── S(x): 1×1 s2 + BN ──→ [N,512,28,28] ──┐
x [N,256,56,56] ─┤                                           + → ReLU → y
                 └── F(x): 1×1 → 3×3 s2 → 1×1 ──→ [N,512,28,28] ─┘
```

`layer3.0`、`layer4.0` 复制同一模板：通道再翻倍（512→1024→2048），空间再减半（28→14→7）。每个 stage 内部则是同分辨率上的多次 $x+F(x)$。中层的 $F$ 开始组合部件和重复结构；stage 门口那一下不是「又加了一个残差」，是换了一套更粗的坐标系，再在新地图上走几步。

**layer4** 已经是 $7\times 7$ 的粗网格、2048 维通道。增量更像在调语义向量场：拉开类间、压背景、为分类头对齐方向。经验上后面几块的 $F$ 不必很大——「保留已有语义 + 微调」常常够用。这是统计偏好，不是「深层只学物体、浅层只学边」的硬分区；通道是混杂的，概念跨多层。

把分辨率和语义放在同一条轴上：

```text
fine grid, appearance          coarse grid, class semantics
     │                                    │
  layer1 @56          layer2@28 / layer3@14         layer4@7
  Δ: edges / texture  Δ: parts + scale change       Δ: class-related refine
  identity often cheap   stage door = projection     often smaller F
```

深度在 ResNet 里同时是两件事：抽象级别（边 → 部件 → 物体），以及 residual 的修正级别（在哪张网格上、改多大的 $\Delta$）。可迁移性通常随深度递减：浅层换数据集仍好用，深层更贴原任务的类空间——这是「冻浅层、只训 head」的背景，不是另一套理论。

### 8.4 分类头看见谁，看不见谁

```text
layer4 [N, 2048, 7, 7]
    → AdaptiveAvgPool → [N, 2048, 1, 1]
    → flatten          → [N, 2048]
    → fc Linear(2048, 1000)
```

`fc` 只吃 layer4 经 GAP 后的 2048 维向量。它不能直接读 stem 的 `[N,64,112,112]`，也没有一条从 `conv1` 拉到分类头的 skip。早期信息可以**影响**最终向量——前向一路算过来，stage 内 residual 还让部分路径更短——但早期 feature 并没有**原样归档**给分类器。那是 DenseNet / U-Net / FPN 的契约。

| 说法 | 对吗 |
|------|------|
| fc 的输入 = stem conv1 输出 | 否，输入是 layer4 的 2048-d |
| 早期信息可以影响最终向量 | 是，沿计算图传播 |
| 早期 feature 原样保留给 fc | 否；跨 stage 有 projection、降采样、多次 $F$ |
| 分类头是多层 MLP | 标准是一层 Linear |

### 8.5 和手写代码的对应

| 概念 | `resnet50/resnet_50.py` |
|------|-------------------------|
| Bottleneck | 类 `Bottleneck`：`conv1/2/3` + `downsample` + `out += identity` |
| Post-activation | `out += identity` 之后 `self.relu(out)` |
| 真 identity | `downsample is None` 时 `identity = x` |
| Projection | `_make_layer` 里 `conv1x1 + BN` 赋给 `downsample` |
| 16 个 block | `resnet50()` → `[3,4,6,3]` |
| 起步靠近 $F\approx 0$ | `zero_init_residual` 把每个 `bn3.weight` 置零 |
| 分类头 | `self.fc = nn.Linear(2048, num_classes)` |

### 8.6 都叫 skip，契约可以完全不同

Residual 让「在同一尺度上加 $\Delta$」变得可训之后，人们用同一种直觉去做了别的融合。名字都叫 skip，问的问题不是一个。

| 结构 | 融合 | 在干什么 |
|------|------|----------|
| ResNet | 同尺度 addition | 学 $\Delta$，让 identity 廉价 |
| DenseNet | concatenation | 把旧 feature 原样留下，channels 增长 |
| U-Net / FPN skip | 跨尺度 concat 或 fuse | 把 encoder 的高分辨率细节送给 decoder |

```text
ResNet:   h + F(h)               same-grid correction
DenseNet: concat(old, new)       keep old channels; width grows
U-Net:    encoder detail ──► decoder    recover spatial detail
```

分类 ResNet 的深层**不会**把浅层细节原样交给 `fc`。你若要像素级边界，必须另拉一条跨尺度的线。不要因为都会画一道弧，就把优化旁路和细节回放当成同一设计。

走完这条高速公路，深度不再是 plain net 里那个一加深就训崩的东西。可以叠了。叠起来的每一块，信息还是按**预先规定的位置关系**流动：卷积看邻域，shortcut 看同一像素。若下一个问题是「这个 token 该读哪一个 token」，残差帮你把 block 堆高，却没有替你决定路由。

---

## 9. 残差仍然没给你的：按内容路由

ResNet 的连接在 architecture 里就定了：kernel 覆盖附近，$1\times 1$ 混合通道，shortcut 对齐同一位置。表示可以一层层 refine，但「谁读谁」不取决于这一次 forward 里向量的内容。第 12 章要做的事恰好相反：每次 forward 都根据当前内容，动态决定位置之间的连接强度。那叫 attention，也可以叫 content-dependent routing。

有趣的是，Transformer 并没有丢掉残差。它把同一句 $x\leftarrow x+F(x)$ 写了两遍：

```text
x ← x + Attn(Norm(x))
x ← x + MLP(Norm(x))
```

中心舞台变成一条固定宽度的 **residual stream**：Attn 和 MLP 都不重写整条流，只往里面加 $\Delta$。能把 attention 这种昂贵、难训的 $F$ 叠几十层，前置条件正是本章的 identity 旁路和 normalization。差别在于 $F$ 里第一次出现了「按内容选择读谁」。

因此下面这张表不是第二本小书，只回答一个问题：同样是残差增量，坐标轴和 $F$ 的含义换成了什么。

| | ResNet | ViT | Decoder-only LLM | DiT |
|--|--------|-----|------------------|-----|
| 残差对象 | 特征图 $h$ | 固定网格上的 patch token | token 残差流 | latent patch 流 |
| 加深主要改什么 | 分辨率 + 语义层级 | 同一坐标系上多轮视觉 refine | 同一句表示上多轮 refine | 网络内 refine $\times$ 去噪时间 |
| 浅层 $\Delta$ | 边/纹理，高分辨率 | 局部外观；学到的 Attn 仍常偏短程 | 词法、局部形式 | 局部 noisy latent |
| 中层 $\Delta$ | 部件 + stage 口换挡 | 部件/区域，Attn 变广 | 结构、指代；许多机制型电路 | 布局/物体（条件开始加深） |
| 深层 $\Delta$ | 类语义，粗网格 | 全局 / CLS | 下一 token 的决策精修 | 条件对齐 + $\epsilon$/v 预测 |
| 空间金字塔 | 强（56→7） | 弱（除非分层 ViT / Swin） | 无，序列长度固定 | latent 网格 + **噪声尺度 $t$** |
| 默认 identity | $F\to 0$；可 zero-init 末 BN | pre-norm residual | 同左 | 同左，常 **adaLN-Zero** 门控起于 0 |
| Head 读谁 | 最后 stage 的 GAP 向量 | 最后层 CLS 或 pool | 最后残差流（经 LN） | 最后层 → 噪声头 |

ViT 把图切成 patch，理论感受野一层就可以是全局的；**学出来的** attention 却仍然常常由局部长到全局，CLS（若使用）也是随深度累积全局摘要，浅层还「空」。没有 56→28 那种必然换挡，层级更软。小数据时这套归纳偏置往往弱于 ResNet——残差能让它堆起来，不能替它把卷积的局部性变回来。

LLM 把空间金字塔彻底拿掉。语义轴改成：局部形式 → 结构/机制 → 下一词的决策。粗分工值得单独记一笔，因为它就是「$F$ 在写什么」的语言版：Attn 更像路由和搬运，MLP 更像改写内容。Head 只读最终残差流，与 ResNet 的 `fc` 只读 layer4 同构。浅层往往更可迁移，输出相关的变化大量发生在深层附近——LoRA 常贴后部，和「冻浅训深」是同一类经验，不是定理。具体电路落在哪一层随规模和训练而变，表里写的是统计偏好。

DiT 多出一条轴，最容易和网络深度黏在一起。网络深度（block 1…$L$）仍是「对当前这张 noisy latent 做空间/语义 refine」，和 ViT 同类。扩散时间 $t:T\to 0$ 是外环：每一步网络预测噪声或速度，改的是 latent 本身。$t$ 大时偏好大结构，$t$ 小时偏好纹理高频。**网络浅 $\neq$ $t$ 大。** 条件则经 adaLN（有时 adaLN-Zero）写成 scale/shift/gate，去调制 Norm 后的激活，再沿残差加回 $x$。Zero-init 的门控让每个 block 起步接近 identity——和 `zero_init_residual`、和 $F\to 0$ 是同一默认，只是旋钮从 BN 的 $\gamma$ 换成了条件门。

```text
                    residual: x + F(x)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
     ResNet              ViT / LLM              DiT
   spatial gearshifts    fixed width,         fixed width
   edge→part→object      many writes          + condition (t)
                         on one stream        depth × noise scale
```

共同点仍然只有一句：好解被参数化成 identity 附近叠 $\Delta$。变的是 $\Delta$ 的语义轴——空间层级、语言机制、还是噪声尺度。残差没有提供的那一块是 $F$ 内部的 **content-dependent routing**。现在深度已经不是瓶颈，可以问下一章的母问题了：一个 token 怎样根据当前内容决定该读取谁。

---

## 10. 自测

1. 为什么更深的 plain network 理论上能模拟浅网，却可能连 train error 都更高？这件事和 overfitting 差在哪？  
2. 用「修改文章」解释 $h_{\ell+1}=h_\ell+F(h_\ell)$。$F=0$ 时 block 在做什么？  
3. 为什么说 residual 首先是再参数化，而不是「扩大了可表示函数类所以有效」？  
4. Channels 或 spatial size 改变时，为什么不能直接 $x+F(x)$？Projection 之后还是真 identity 吗？  
5. 写出一层的 $\partial h_{\ell+1}/\partial h_\ell$。把 $L$ 层写成 $\prod(I+J_F)$，说明它和 $\prod J_f$ 差在哪。  
6. 展开式 $h_L=h_\ell+\sum F_i(h_i)$ 怎样给出「直达」的梯度项？为什么这仍不等于一堆独立 ensemble？  
7. 在什么意义上 residual **不能**保证 gradient 永不消失？Post-activation 的 ReLU 和 projection shortcut 各破坏了公式里的哪一块？  
8. 为什么不断 addition 可能让 activation magnitude 增长？BatchNorm 的 $\gamma,\beta,\epsilon$ 各管什么？  
9. 为什么 inference 不能继续用当前 batch 的 mean/variance？Train 好、`eval()` 崩，第一嫌疑通常是什么？  
10. 序列模型为什么常常不用 BatchNorm、改用 LayerNorm？统计轴差在哪？  
11. Post-activation 与 pre-activation 在 Add 前后各差什么？$F\to 0$ 时输出分别接近什么？本仓库 ResNet-50 是哪一种？  
12. ResNet-50 的 50 怎么算？哪些层不计？它有多少个 Bottleneck、多少个真 identity shortcut？  
13. 写出 `layer2.0` 的 $F(x)$ 与 $S(x)$ 的 shape，说明二者为何能相加、为何 $S$ 不是 $x$。降采样发生在哪一层卷积？  
14. `fc` 的输入维度是多少？它能否直接读到 stem `conv1` 的 feature map？早期信息还能不能影响分类？  
15. ResNet skip、DenseNet concat、U-Net skip 各自的契约是什么？为什么说分类 ResNet 不会把浅层细节原样交给分类器？  
16. 用增量视角各用一句话概括浅 / 中 / 深层 residual 的学习偏好。为什么说这是统计偏好而不是硬分区？  
17. Stage 内的 $x+F(x)$ 和 stage 门口的 stride-2 + projection，增量性质差在哪？  
18. Transformer 的 residual stream 和 ResNet 的 $h+F(h)$ 同在什么地方、差在什么地方？Attn 子层和 MLP 子层大致各写哪类 $\Delta$？  
19. DiT 的网络深度和扩散时间 $t$ 为什么不能当成同一条「浅/深」轴？adaLN-Zero 与 `zero_init_residual` 共享什么默认？  
20. Residual 让你能把复杂 block 堆高。它仍然没有替你决定的那件事是什么？这句话怎样接到第 12 章？

---

## 11. 合上书再看一眼

残差把「加深」从一条只能重写的长路，改成了 identity 附近的许多小修正。Shape 对齐、$I+J_F$、BatchNorm、Add 前后的顺序，都是为了让这条旁路真的能走，而不是墙上多画一道弧。ResNet-50 只是把契约走成能跑的 50 层：16 个 Bottleneck，门口换挡，门内真 identity，头只读最后的 2048 维。

可以叠了，还不等于会选路。卷积的连接仍由位置预先规定。下一章把 $F$ 换成 attention：每个 token 根据当前内容决定该读取谁。残差流还在，normalization 还在；新出现的是 content-dependent routing。

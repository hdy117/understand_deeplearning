# 第 10 章：卷积网络（Convolutional Networks）

> 书：《Understanding Deep Learning》Ch.10  
> 上一章：第 9 章把 regularization 说成「训练数据没钉死的地方，学习系统必须另有偏好」；architecture 是注入偏好的一条路。  
> 本章：把「附近的东西关系更大、同一种模式可以出现在任意位置」直接写进层。  
> 后续：深度变难训 → 第 11 章 residual；内容决定远距离该读谁 → 第 12 章 Transformer；邻居关系本身是任意拓扑 → 第 13 章 GNN。  
> 阅读：先跟 data flow 和 tensor shape 走；标了推导的公式可以第二遍再展开。

---

## 0. Fully connected 把图像当成一袋数字

第 9 章临走时留了一句话：偏好可以进 loss、进 optimizer、进 data，也可以直接写进 architecture。图像几乎把该写什么偏好刻在输入结构上，fully connected layer 却一律看不见。

一张图至少有三件事同时成立。邻近 pixels 通常一起构成 edge、corner、texture，关系不是「任意两个像素平等」。同一种局部模式可以出现在画面任意位置：左上角的竖线和右下角的竖线是同一类东西。小 pattern 还能逐层组成 part，再组成 object。这三条不是审美，是统计结构。把训练图和测试图用同一套像素重排打乱，人就看不懂了；fully connected 层却几乎无感——它本来就没有「附近」这个概念，每个输入到每个 hidden unit 都有独立权重。

它付出的代价很具体。典型分类输入是 $224\times 224$ 的 RGB 图，也就是 $150{,}528$ 维。若第一层 hidden 还维持这个宽度，fully connected 的 weight 矩阵大约是

\[
150{,}528^{2}\approx 2.3\times 10^{10}
\]

大约二百二十七亿个参数。内存、计算、所需样本全爆炸。更糟的是浪费发生在结构上：平移几个像素会改写几乎每一个输入坐标，于是「树」的纹理必须在每个位置各学一遍。分类其实希望整图标签稳定，分割则希望 mask 跟着物体走；fully connected 对这两种几何都没有意见，它只看见一袋数字。

所以本章只追一个问题：

> **怎样把 locality 和 translation 写进一层，再靠深度把局部零件拼成物体？这套 inductive bias 换来了什么，又把世界里哪些事假装成了假的？**

调查链很短，后面每一节只补这一条链上的一个缺口：

```text
FC treats pixels as an unordered bag
        │
        ▼
a sliding local detector, same weights everywhere
        │
        ▼
2D kernel + tensor shapes
        │
        ▼
one filter must span all input channels;
many filters in parallel = output channels
        │
        ▼
depth composes parts; receptive field grows
        │
        ▼
padding / stride / dilation are geometric consequences
        │
        ▼
the bias helps on grids, and still cannot do
content-dependent long-range routing
or arbitrary topology
```

先不要背 kernel 公式。公式是把上面第一段实现写清楚；真正的设计决策是「一次只看窗口」和「同一套局部规则到处用」。

---

## 1. 先写一个会滑动的局部探测器

要把「附近更重要」写进层，最省事的办法不是让每个输出看完整张图，而是每次只看一小段。先把图像压成一条亮度序列，二维的麻烦稍后自然出现。

设想输入是

```text
low  low  low  low | high high high high
```

我们想找「从暗突然变亮」的位置。准备一个长度为 $3$ 的小探测器

\[
w=[-1,0,1],
\]

让它从左到右滑动，每个位置做一次 dot product：

```text
x:     x1   x2   x3   x4   x5   x6
        \   |   /
          z2 = w1 x1 + w2 x2 + w3 x3
               \   |   /
                 z3 = w1 x2 + w2 x3 + w3 x4   # same w
```

平坦区域左右差不多，响应接近 $0$；跨过暗到亮的边界，响应变大。把 $x=[0,0,0,1,1,1]$ 手算一遍：$[0,0,0]\to 0$，$[0,0,1]\to 1$，$[0,1,1]\to 1$，$[1,1,1]\to 0$。探测器没有记住「第几个位置」，它只记住一种局部对比；位置是滑出来的。

再把输入缩短成四个数，kernel 仍是 $[-1,0,1]$，只保留 kernel 完整落在内部的位置（valid，输出长度 $4-3+1=2$）：

\[
x=[0,0,1,1],\qquad w=[-1,0,1].
\]

第一个窗口 $[0,0,1]$：

\[
z_0=(-1)\cdot 0+0\cdot 0+1\cdot 1=1.
\]

第二个窗口 $[0,1,1]$：

\[
z_1=(-1)\cdot 0+0\cdot 1+1\cdot 1=1.
\]

所以

\[
z=[1,1].
\]

窗口从「全暗」滑到「右半全亮」，两次都踩在同一条上升沿上，两次都报 $1$。若把 $x$ 改成 $[1,1,0,0]$（亮变暗），同样两步得到 $[-1,-1]$：同一套三个权重，只是对比方向反了。这四个数已经够说明 locality（每个 $z$ 只用三个 $x$）和 sharing（两次用同一个 $w$）。

这里同时落下 CNN 的两个约束。**Local connectivity**：每个输出只依赖附近几个输入，远处权重直接是 $0$。**Weight sharing**：同一个 $w$ 在所有位置复用，不再给左上角和右下角各学一套竖线探测器。

工程库实际算的通常是

\[
z[i]=\sum_{u=0}^{K-1}w[u]\,x[i+u].
\]

数学书上的 convolution 会先把 kernel 翻转再滑；深度学习里一般不翻转，严格说是 cross-correlation。$w$ 是 learned parameter，训练可以自己学到需要的方向，于是社区仍然把它叫 convolution。第一遍只需记住：局部 window 与共享 weights 做 dot product，得到这个位置的 feature response。层里通常还会加 bias 和 activation；locality 和 sharing 发生在那个加权和上。

把同一件事说成 fully connected 的特例，参数账就清楚了。$D$ 维输入到 $D$ 个 hidden units 的密集层有 $D^{2}$ 个 weights、再加 $D$ 个 biases。kernel size 为 $3$、stride 为 $1$ 的卷积只有 $3$ 个 weights、一个 bias。对应的权重矩阵是带状的，而且对角线上反复出现同一组数：许多项被钉死为 $0$，剩下的被钉死为彼此相等。

用刚才那四个输入钉死数字。输入长度 $4$，要 $2$ 个 valid 输出（和上面 $z=[1,1]$ 一样）。Fully connected 从 $4$ 维到 $2$ 维：

\[
4\times 2=8\text{ weights}+2\text{ biases}=10\text{ 个参数}.
\]

同一件事用 $k=3$ 的 1D conv：三个共享 weights 加一个 bias，一共 **$4$ 个参数**，与序列再变长无关。若输入改成 $40$ 维、输出仍按 valid 得到 $38$ 个位置，FC 是 $40\times 38+38=1558$，卷积仍是 $3+1=4$。第 0 节那张 $224\times 224\times 3$ 图把同一笔账放大：第一层若仍维持 $150{,}528$ 个 hidden units，FC 约 $2.3\times 10^{10}$ 个 weights；一个 $3\times 3$、跨 $3$ 通道、再出 $64$ 个 channels 的卷积只有

\[
64\times(3\times 3\times 3)+64=1792
\]

个参数。少的不是「表达能力 magically 变弱」，是绝大多数「任意像素连任意 hidden」的边被钉死为 $0$ 或被强制相等。

```text
FC matrix (6 -> 6):     Conv k=3 (same size):

* * * * * *             a b c 0 0 0
* * * * * *             0 a b c 0 0
* * * * * *             0 0 a b c 0
* * * * * *             0 0 0 a b c
* * * * * *             0 0 0 0 a b   # edges depend on padding
* * * * * *             0 0 0 0 0 a
```

卷积并没有获得「fully connected 表达不了」的新函数类；它是在说：那些不稀疏、不共享的映射，我们根本不考虑。第 9 章把这种话叫 architecture 里的 inductive bias，也可以看成对函数族施加了无限大的 penalty。参数量因此不随序列变长而增长——$K$ 个 weights 仍然是 $K$ 个。计算量会增长：每多一个输出位置，就要再做一次同样的局部点积。

窗口滑到序列两端时，kernel 会探出输入范围。可以在外面补值再算，也可以只保留 kernel 完整落在内部的位置。这两种选择暂时只需知道它们存在：一个在注入边界假设，一个在让表示变短。窗口每次跳几格、内部要不要留空洞，同样先按「每次移一格、连续三个点」来想。几何旋钮是滑动探测器的后果，等深度把视野问题逼出来再拧。

一条线会滑了。像素却排成格子，而且常常叠着 R、G、B 三张皮。二维窗口该长什么样，一个探测器要不要同时看完三种颜色？

---

## 2. 升到 2D：kernel 与 tensor

把 1D 窗口换成正方形，加权和跟着变成一块邻域上的和。忽略 batch 时，图像写成

\[
X\in\mathbb R^{C_{\mathrm{in}}\times H_{\mathrm{in}}\times W_{\mathrm{in}}}.
\]

灰度图 $C_{\mathrm{in}}=1$；RGB 的 $C_{\mathrm{in}}=3$。kernel spatial size 取 $3\times 3$ 时，一个输出位置不再是三个数的点积，而是九个空间位置的加权和。对单通道、stride $1$、不补边，hidden unit 可以写成

\[
h_{ij}
=
a\!\left(
\beta+\sum_{m=1}^{3}\sum_{n=1}^{3}
\omega_{mn}\,x_{i+m-2,\,j+n-2}
\right).
\]

这只是把 1D 的「最近三个点」换成「最近 $3\times 3$ 块」，再加 bias 与 activation。kernel 在网格上既向右滑也向下滑，每个落点产生一个数。

RGB 立刻逼出下一件事：一个完整 filter 不能只看红色。人看见的边缘往往同时写在三个通道里，绿色里的竖线和红色里的竖线是同一条边的不同分量。于是一个 output feature 要综合整个局部立方体：

```text
one complete filter

R: 3x3 weights
G: 3x3 weights
B: 3x3 weights
        │
        └── sum all 27 products + bias
                         │
                         ▼
                 one output channel
```

单个完整 filter 的 shape 是 $(C_{\mathrm{in}},K_h,K_w)$。普通 Conv2d（`groups=1`）一层里有 $C_{\mathrm{out}}$ 个这样的 filter，整层 weight tensor 是

\[
W\in\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K_h\times K_w}.
\]

一个输出标量的 data flow 因此极短：取出对齐后的局部 patch，共 $C_{\mathrm{in}}\times K_h\times K_w$ 个数，与第 $o$ 个 filter 逐元素相乘并求和，加上该 output channel 的 bias，得到 $Y[o,i,j]$。stride $1$、无 dilation、不补边时，

\[
Y[o,i,j]
=
b_o
+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
W[o,c,u,v]\,X[c,i+u,j+v].
\]

同一 filter 滑过所有空间位置，铺成一张 feature map；有 $C_{\mathrm{out}}$ 个 filters，就有 $C_{\mathrm{out}}$ 张 feature maps。所谓「$3\times 3$ kernel」在多通道里从来不是 $9$ 个参数：每个完整 filter 有 $C_{\mathrm{in}}\times 3\times 3$ 个 weights。RGB 上那就是 $27$，再加一个 bias。

到这里 shape 已经能写对，但还没回答一个更刺的问题：为什么不满足于一个 filter？只滑一个探测器，信息会去哪？

---

## 3. 一个 filter 不够：channels 从哪来

单次卷积是在做局部加权平均，再被 ReLU 一类 activation 把负数剪掉。一种探测器只对一种对比敏感：$[-1,0,1]$ 喜欢从暗到亮，对从亮到暗、对水平线、对纹理都可能近乎沉默。只保留一个 output channel，等于强迫整张图只讲一种局部故事，后面的层再能说的话会非常少。

所以同一层并行放许多套 weights。每一套仍是「跨越全部 input channels 的完整 filter」，仍在所有位置共享；它们的差别只是在问不同的局部问题。一组输出构成一张 feature map，也叫一个 channel。第一层也许一张图学竖边，一张学横边，一张学某种颜色过渡——这些名字是事后解释，训练只是在学 $C_{\mathrm{out}}$ 组不同的局部点积。

下一层的输入已经不再是 RGB，而是 $C_{\mathrm{in}}$ 张 feature maps。新 filter 仍然必须穿过当时全部输入通道：它看的是「最近 $K\times K$ 个位置上、所有已经存在的 feature types」。输出 $16$ 个 channels 的意思是 $16$ 个完整 filters，不是「每个输入通道各自长出 $16$ 份互不相加的结果」。每个输出标量都会把局部立方体里的信息加总成一个数。

术语容易在「核」这个字上打架，所以数对象时要把单位说死：

| 所数对象 | 普通 Conv2d 里它是什么 |
|----------|------------------------|
| 完整 output filter | 一个 output channel 对应一套 $(C_{\mathrm{in}},K_h,K_w)$ |
| filter 内的二维 slice | 每个输入通道一张 $K_h\times K_w$ |
| 一个输出标量用到的 weights | $C_{\mathrm{in}}K_hK_w$ 个，外加该通道一个 bias |

最省事的说法：一层有 $C_{\mathrm{out}}$ 个完整 filters；每个 filter 内含 $C_{\mathrm{in}}$ 张二维 slices。

用一个具体 shape 把账钉住。假设 `groups=1` 的普通卷积：

```text
Input :  3 x 256 x 256
Output: 16 x 128 x 218
Kernel spatial size: 3 x 3
```

输出 $16$ 个 channels，因此有 $16$ 个完整 filters。单个 filter shape 是 $(3,3,3)$，整层

\[
W.\mathrm{shape}=(16,3,3,3).
\]

Weights $16\times 3\times 3\times 3=432$，每个 output channel 一个 bias，参数共 $448$。参数量不乘 $128\times 218$，因为空间位置共享同一套 $w$。计算量相反，必须乘上有多少个输出标量：

\[
16\times 128\times 218=446{,}464,
\]

每个标量 $27$ 次乘加，batch size 为 $1$ 时大约 $12{,}054{,}528$ 次 MAC。以后看见任何 Conv2d，都可以先拆成两句：参数量看 $C_{\mathrm{in}},C_{\mathrm{out}}$ 和 kernel size；compute 还要看 $H_{\mathrm{out}},W_{\mathrm{out}}$ 和 batch。

这个例子里 $128\times 218$ 很可疑。输入是方形，kernel 也是方形；若 height 与 width 使用完全相同的 stride、padding、dilation，输出也应是方形。常见的

```text
kernel=3, stride=2, padding=1, dilation=1
```

会得到 $3\times 256\times 256\to 16\times 128\times 128$。出现 $218$，说明还有信息没给：两个方向的 stride 或 padding 不同、width 事先被 crop、中间夹了 pooling、实际输入宽度不是 $256$，或 dilation 不对称。能唯一确定的是 filter 数、filter shape 和参数量；不能从「输入方形、输出非方形」反推出唯一的空间配置。

一层的视野仍然只有一个窗口那么大。猫不是 $3\times 3$ 的。深度究竟在拼什么？

---

## 4. 一层只看见窗口，深度才把零件拼起来

局部探测器的代价现在露出来了：单层无论并行多少 channels，每个输出位置看见的原始像素仍是那一块 $K\times K$。要识别物体，必须让更上层去组合已经检测出来的局部答案。

```text
Layer 1:  edge detector     sees a few pixels
Layer 2:  combines edges    sees a corner / texture
Layer 3:  combines parts    sees a larger fragment
Deeper:   evidence gathers  toward object-level
```

一个 hidden unit 的 **receptive field**，是原始输入里可能影响它的区域。三个 stride-$1$ 的 $3\times 3$ 叠在一起，单个位置的 theoretical receptive field 是 $7\times 7$，不是把三层窗口边长乘起来。原因是相邻 hidden units 的窗口大量重叠：

```text
Layer-1 neighbors, kernel size 3:

unit i-1: [x_{i-2}, x_{i-1}, x_i]
unit i  : [x_{i-1}, x_i, x_{i+1}]
unit i+1: [x_i, x_{i+1}, x_{i+2}]

A layer-2 unit that sums these three
sees the union [x_{i-2}, ..., x_{i+2}],
length 5, not 3 x 3 = 9.
```

没有下采样、dilation 为 $1$ 时，每加一层只把 receptive field 的边长增加 $k-1$。从单个像素 $r_0=1$ 数起，

\[
r_L=1+L(k-1).
\]

希望某个 unit 的 theoretical receptive field 盖住宽度 $n$ 的图，至少需要

\[
L\ge\left\lceil\frac{n-1}{k-1}\right\rceil.
\]

$3\times 3$、stride $1$ 对 $256\times 256$ 是 $L\ge 128$。这和 $\log_3(256)\approx 5$ 完全不是一回事。误把「每层看 $k$ 个」读成「每层把视野乘以 $k$」，就是把重叠的 union 当成了不相交的笛卡尔积。

把 $k=3$、stride $1$ 的边长逐层写出来（从单个像素 $r_0=1$ 起）：

| 层数 $L$ | $r_L=1+L(k-1)$ | 这一层新看见什么 |
|---|---|---|
| $0$ | $1$ | 一个 pixel |
| $1$ | $3$ | 一个 $3\times 3$ window |
| $2$ | $5$ | 两个重叠 window 的 union |
| $3$ | $7$ | 三层小核 $\equiv$ 一层 $7\times 7$ 的视野 |
| $5$ | $11$ | 仍远小于一张 $256$ 的图 |
| $10$ | $21$ | 大约一张小脸的局部 |
| $128$ | $257$ | 才刚盖住 $256$ 边长 |

1D 上同一公式：$k=3$ 时每层 $+2$。输入若只有 $[0,0,1,1]$ 那么短，两层 stride-$1$ 的 $k=3$ 已经让最中间那个 unit 看见全部四个位置（$r_2=5>4$）。图变大时，没有下采样就只能靠「一层 $+2$」慢慢爬——这才是后面要拧 stride 的原因。

深层因此不是把 $3\times 3$ 放大成一张全图矩阵；它是在已经共享的局部规则上，再学「这些局部答案怎样组成更大零件」。VGG 后来大量堆叠 $3\times 3$，图的就是这件事：三层 $3\times 3$ 的 receptive field 与一层 $7\times 7$ 相同，中间却多插入两次 nonlinear transformation，而每一对输入/输出通道的空间 weights 是 $3\times 9=27$，不是 $49$。小 kernel 的深度，是在用组合代替单层的大视野。

可是 $128$ 层只为了让 $3\times 3$ 看见 $256$ 的边，这个代价难看。有没有办法让窗口在边界上仍能落稳、让视野长得更快、让采样不必密到每个像素都算一遍？这些不是另起一章的新概念，而是滑动窗口自己带出来的几何问题。

---

## 5. Padding、stride、dilation：窗口怎么摆、怎么跳

窗口要完整落在图内，输出就会比输入短一截——这就是所谓 valid convolution。若希望边缘位置也产生输出，就必须在边界外假装还有值，这叫 **padding**。Zero padding 把外面当成 $0$；reflect 把边缘镜像出去；circular 把对边接上。它们都能让 shape 好看，但都在声明一种边界假设：图外面是黑的、是镜子，还是环面。边缘附近的响应会跟着变，于是「卷积对平移无条件友好」在有 padding 的有限图上本来就不是定理。

窗口也不必每次只走一格。**Stride** 是相邻两次落点的间距。Stride $1$ 每个位置都算；stride $2$ 大约隔一格采一次，空间尺寸通常近乎减半。计算下去了，细节和相位也下去了：物体平移一个像素，stride $2$ 采到的是另一套格子，输出不会简单跟着平移一格。把它当成免费加速，就是没看见采样定理。

若还想让单层看见更远、却不愿把 $K$ 加大到参数爆炸，可以把采样点隔开。**Dilation**（atrous，「带洞」）在 kernel 的权重之间插入零。有效覆盖边长

\[
K_{\mathrm{eff}}=d(K-1)+1.
\]

$K=3$、$d=2$ 时 $K_{\mathrm{eff}}=5$，仍然只有三个 learned weights，只是中间跳了一格。覆盖范围变大，局部连续性变差；纹理若刚好落在洞里，这一层可以视而不见。

把 padding、stride、dilation 写进同一条输出尺寸公式，只是在数「第一个合法落点之后，还能再跳几次」：

\[
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+p_t+p_b-d_h(K_h-1)-1}{s_h}
\right\rfloor+1,
\]

width 把 $H$ 换成 $W$，把上下 padding 换成左右。不需要硬背：$H_{\mathrm{in}}$ 加上两边补的格子，减去一个有效 kernel 还剩下多少可滑动距离，再按 stride 切段，最后把第一个位置加回来。第 3 节那个 $256\to 128$ 正是 $K=3$、$s=2$、$p=1$、$d=1$ 的结果；两个方向公式相同则输出仍是方形，$218$ 仍然只能来自公式里某个不对称的量。

这些旋钮会改 receptive field 的增长速度。记 $j_l$ 为第 $l$ 层相邻 units 在原图上的间隔，$r_l$ 为 receptive field 边长，则

\[
j_l=j_{l-1}s_l,
\qquad
r_l=r_{l-1}+(K_{\mathrm{eff},l}-1)j_{l-1}.
\]

没有下采样时 $j_l$ 恒为 $1$，于是回到上一节的线性增长。若每层 stride 都是 $s>1$，间隔指数变大；kernel 固定、dilation $1$ 的简化式是

\[
r_L
=
1+(k-1)\sum_{t=0}^{L-1}s^{t}
=
1+(k-1)\frac{s^{L}-1}{s-1}.
\]

这时覆盖全图所需层数确实变成 log-like，底数主要来自 stride $s$，不是 kernel size $k$。每层 $3\times 3$、stride $2$ 时 $r_L=2^{L+1}-1$，覆盖 $256$ 大约 $L\ge 8$。视野迅速变大，spatial resolution 也迅速变差。真实 CNN 很少每一层都 stride $2$，而是若干普通卷积中间夹少数 downsampling，所以应当逐层用递推算，而不是套一条口号。

还有一条更软的边界。Theoretical receptive field 只声明「存在一条影响路径」。权重、activation 和数据会让实际贡献集中在中心附近，这是 effective receptive field。覆盖全图不等于已经在用全图；路径存在，不等于模型给了每个像素同等的注意力。

几何旋钮把「看得更远」部分变成了「采样更稀」。空间分辨率被故意丢掉之后，网络用什么换 context？若输出还需要一张和输入一样密的图，丢掉的细节还回得来吗？

---

## 6. 空间变小换 context；变大并不把细节变回来

Downsampling 就是把上一节的 stride 说成一种设计：主动降低采样密度，换更大 context、更低 compute，以及一点点对微小平移不敏感。常见三条路。Strided convolution 一边学 feature 一边抽稀，权重可学，也可能 alias。Max pooling 在每个 window 里保留最大响应，强调「这种 pattern 有没有出现过」；输入平移一格，不少最大值仍在，于是带出局部 invariance。Average pooling 更像先平滑再采样。它们都按 channel 独立做，所以 $H,W$ 减半时 channel 数不变——变 channel 是另一件事，稍后才轮到。

典型的 backbone 还会在变稀的同时把 channel 加宽：

```text
[3, 256, 256]
    conv
[32, 256, 256]
    downsample
[64, 128, 128]
    downsample
[128, 64, 64]
```

空间格子变粗，单个 feature 对应的原图区域变大；更多 channel 用来表示更抽象的 feature types。这不是免费压缩。精确位置被丢掉了。分类也许正想丢掉「猫在左还是在右」；分割和检测不能把这件事忘干净。

若输出必须回到像素格子上，就要 upsampling。Nearest 或 bilinear interpolation 不学习，只是把已有值铺开。Max unpooling 记住当初 max pooling 的argmax，把值放回原位，其它位置为 $0$。Transposed convolution 把「stride $2$ 卷积」对应的线性算子转置：输入每个点按学习到的方式散布到更大的输出网格上。它是 transpose，不是数学逆。Checkerboard artifact 会在步长和 kernel 咬合不好时出现，正是因为散布规则是学出来的网格效应，不是在恢复被抽掉的频率。

Encoder 已经扔过的信息，不会因为 $H,W$ 变大就自动回家。这就是 U-Net 一类 architecture 要 skip connection 的原因：把 encoder 还没抽稀时的 high-resolution detail 直接递给 decoder，与已经变语义、变模糊的上采样特征融合。

```text
High-resolution encoder feature ─────────┐
                                         ▼
Low-resolution semantic feature → upsample → fuse → dense output
```

Skip 不是残差网络那条「默认 identity」的旁路，只是在说：尺寸和信息不是同一种东西。第 11 章的 residual 要解决的是另一件事。

空间尺寸会改了。有时我们其实只想改 channel 数：两支路要相加，必须先对齐 $C$；或者 $3\times 3$ 太贵，想先把通道压窄。kernel size 一直在回答「附近有多大」。若故意把邻域缩成一个点呢？

---

## 7. $1\times 1$：不看邻居，只改这个位置的混合

$K_h=K_w=1$ 时，滑动窗口里只剩「当前这个空间位置」，加权和退化成对 channels 的线性组合：

\[
z_o(i,j)
=
b_o+\sum_{c=1}^{C_{\mathrm{in}}}W_{o,c}\,x_c(i,j).
\]

它不扩大 spatial receptive field。完整 filter 仍然跨越全部输入通道，只是每张 slice 从 $3\times 3$ 变成了 $1\times 1$。同一套 $W$ 在所有位置共享，等价于在每个格子上跑同一个极小的 fully connected 层。

它有用，正因为第 3 节已经把「一个位置」变成了一个 feature 向量。$3\times 3$ 问附近有什么；$1\times 1$ 问这个位置已经有的那些 feature types 该怎样重新混合：升维、降维、在 segmentation head 里直接变成 class logits，或在昂贵空间卷积之前先压窄 channels。没有邻居可看，不是缺陷，是它选择只做 channel 几何、不做空间几何。

到这里，滑动、共享、通道、深度、采样，该出场的部件都在场了。可以回头问第 0 节那件被 fully connected 搞混的几何：猫往右挪，输出到底该跟着走，还是该原地不动？

---

## 8. 平移之后，输出该跟着走还是原地不动

任务把两个词分开。分类希望猫从画面左边走到右边，标签仍是「猫」——整体 prediction 对平移 **invariant**。分割希望 mask 跟猫一起走——输出对平移 **equivariant**。若变换为 $T$，

\[
f(Tx)=f(x)
\quad\text{是 invariance},
\qquad
f(Tx)=T f(x)
\quad\text{是同一类空间上的 equivariance}.
\]

Convolution 通过 weight sharing 得到的，首先是对平移的 equivariant bias：同一段纹理挪到别处，同一只 filter 仍会在别处亮起来。Feature map 跟着走，不是整图标签钉死不动。Invariance 往往是后来才加的：pooling 丢掉一点精确坐标、global aggregation 把整张 map 收成一个向量、训练数据里本来就有到处跑的猫。

不要背「CNN 天生平移不变」。Zero padding 让边缘的邻域和中心不一样；stride 与 pooling 改变采样相位，对 $1$ 像素平移甚至不 equivariant；物体移出画面，信息没了，什么 $T$ 都救不回来；把 feature map flatten 再接 fully connected head，空间格子被拆成与位置绑定的独立权重，equivariance 到此结束。更准确的一句是：

> Convolutional layer 倾向于 translation equivariance；最终任务、sampling 和 head 决定整体网络有多少 invariance，以及还剩多少空间结构。

这也解释了为什么不能把同一个「卷积堆到没空间」的配方套到所有视觉任务上。输出契约不同，该保留的几何就不同。

---

## 9. 同一个 backbone，三种输出契约

分类、检测、分割可以共用「局部探测器 + 深度组合」这条脊柱，但最后必须把空间信息处理到任务所要求的粒度。

| 任务 | 输出 | 空间信息还剩多少 |
|------|------|------------------|
| Classification | 一个类别分布 | 最终可以汇总全图 |
| Object detection | 类别 + bounding boxes | 必须留下物体在哪 |
| Semantic segmentation | 每个 pixel 的类别 | 要回到高分辨率格子 |

分类的 data flow 是把 convolutional features 收成一份全局描述，再接到 class logits。AlexNet 把这件事做成可训练系统：卷积层次、ReLU、pooling、大规模标注、GPU，以及 dropout 与 weight decay。关键不只是「层数变多」，而是这些部件第一次在 ImageNet 这种难度上一起工作。VGG 把配方收得更干净：反复堆小 $3\times 3$，空间逐渐变小、channel 逐渐变多，最后再接到三段 fully connected。更深带来更好的 ImageNet 数字，也把第 4 节的组合故事做成了经验事实。

检测不能把空间收得那么狠。它要在仍然按格子排好的 feature map 上同时问「这里像什么」和「框在哪」。分割更极端：encoder 可以先变语义、变稀，decoder 必须把格子铺回去；第 6 节的 upsample 与 skip 在这里不是装饰，是输出契约逼出来的。

Architecture 不是越大越好，而是 output contract 决定哪些 inductive bias 还在、哪些已经被 head 拆掉。VGG 式的 plain 堆叠还有一个当时已经看见的墙：深度继续加，训练误差也会变差。那不是普通 overfitting——overfitting 至少还能把训练集拟合得更好。第 11 章要处理的 degradation，就是从这里开始的。

在走下一章之前，先把这套 bias 自己的赢面和装瞎说清楚。否则很容易以为卷积是视觉的终点。

---

## 10. 这个 inductive bias 赢在哪，还缺什么

原书用 MNIST-1D 把第 1 节那个「卷积是 fully connected 的特例」做成实验。输入 $40$ 维，三层卷积加一个 fully connected 头，约 $2{,}050$ 个参数；同等层数、同等 hidden unit 数的密集网络约 $59{,}065$ 个参数。后者的函数类包含前者：把密集矩阵的大多数项设成 $0$、剩下的设成共享，就能精确复现卷积。两边都能把训练误差打到 $0$。测试误差却是卷积大约 $17\%$，密集网络大约 $40\%$。过参数通常并不自动更差（第 8 章），所以这不太像「参数少所以更好」。更像是：卷积强迫每个位置用同一套规则处理，而这份数据本来就是模板随机平移生成的。训练时搜索的是更小一簇事先合理的映射。

换一种说法，与第 9 章对齐：locality 加 weight sharing 是写进 architecture 的 regularizer。它在数据沉默的地方替你选解——选那些「这里的竖线检测器和那里的竖线检测器应该是同一个」的解。图像、音频、一维序列里，这条偏好经常对；它不是免费午餐。

有两件世界里的真事，卷积从一开始就拒绝建模。第一件：该读谁，有时取决于内容，而不是相对坐标。句子里的 `it` 该指向 `animal` 还是 `street`，不能预先规定「永远看左边第三个 token」；一张图里，两个相距很远的零件是否属于同一物体，也可能要等认出它们是什么之后才能决定。卷积的连接由 kernel、stride、dilation 在编译期定死，forward pass 不会因为内容改变接线。第 12 章的 attention 要做的，正是 **content-dependent routing**：每个位置根据当前内容决定读取谁。第二件：不是所有关系都长在规则网格上。分子键、社交关系、交通路网没有固定的 $3\times 3$ 邻域，node 编号也没有空间意义。第 13 章的 GNN 把「谁和谁相连」当成输入的一部分，用同一套规则在给定的 neighbors 上做 message passing。

即便仍在网格上，平移 equivariance 也不是所有视觉问题的真理。绝对位置有时有意义：照片上方更常是天，医学切片的解剖坐标不是随便滑的。Padding、边界、以及后来显式加入的 positional information，都是在承认这件事。卷积赢在「同一套局部规则到处用」；它没有宣称局部规则足以描述一切依赖，也没有宣称依赖永远与绝对坐标无关。

于是第 11 章的问题可以收成一句，而不必提前展开 residual 的公式：当 inductive bias 已经选对、层数还要继续加时，优化为什么反而更难？若多出来的层至少可以什么都不做，更深就不该把训练误差弄差。Plain 的深度没有把「什么都不做」变成容易走的默认路径。那是下一章的母问题。

---

## 11. 自测

1. Fully connected 层为什么既对 locality 失明，又必须在每个位置重学同一模式？参数爆炸来自哪一项乘法？  
2. 为什么说卷积层是 fully connected 层的特例，而不是一个全新的函数类？Weight sharing 把搜索空间怎样缩小？  
3. 机器学习里的 convolution 和数学 convolution 差在哪一步？为什么训练仍然走得通？  
4. 为什么 RGB 上一个完整 $3\times 3$ filter 的 shape 是 $3\times 3\times 3$，而不是 $3\times 3$？  
5. 输出 $64$ 个 channels 意味着多少个完整 filters？每个输出标量有没有把全部 input channels 加总？  
6. 为什么输入变大不增加卷积参数量，却增加 compute？  
7. 对 $3\times 256\times 256\to 16\times 128\times 218$，哪些量能确定，哪些不能？为什么 $218$ 可疑？  
8. 三层 stride-$1$ 的 $3\times 3$ 的 theoretical receptive field 为什么是 $7\times 7$，而不是把每层视野乘以 $3$？没有下采样时，盖住宽度 $n$ 需要怎样的层数？  
9. Padding、stride、dilation 各在改窗口的哪件事？为什么 padding 不只是为了 shape？  
10. 每层 stride $2$ 时，receptive field 的指数增长主要来自 $k$ 还是 $s$？Theoretical receptive field 盖住全图是否等于模型在用全图？  
11. Downsampling 用什么换什么？Upsampling 把 $H,W$ 变大之后，为什么仍可能缺细节？Transposed convolution 是逆运算吗？  
12. $1\times 1$ convolution 没有看邻居，它在混合什么？  
13. 为什么分割更需要 equivariance、分类更希望 invariance？举一个让卷积不再严格 translation-equivariant 的操作。  
14. 卷积预先规定了怎样的接线方式？哪些依赖要等到 Transformer 或 GNN 才能说清楚？  
15. VGG 把小 $3\times 3$ 叠深，赢在组合与非线性；它把下一章的什么困难也一起推到了台前？

---

## 12. 合上书再看一眼

Fully connected 层把图像当成一袋数字：没有「附近」，也没有「同一模式换个位置仍是它」。卷积把这两句话写进计算——每次只看一个局部窗口，同一套 weights 滑过所有位置。升到 2D 之后，一个完整 filter 必须穿过当时全部 input channels，一次滑动得到一个输出标量；并行许多个 filters，才有许多张 feature maps。参数量跟 $C_{\mathrm{in}},C_{\mathrm{out}}$、kernel 走，不跟照片的像素数走；计算量跟输出格子走。

一层仍然只看见窗口。深度把 edge 拼成 part、把 part 拼成 object，receptive field 随之变大；没有下采样时它几乎是线性涨的，重叠不容许你把视野当成 $\log_k n$。Padding 声明边界外是什么，stride 决定采样相位，dilation 用空洞换跨度——它们是滑动窗口的几何后果。空间变稀是在用位置换 context；把格子放大只恢复尺寸，细节往往要靠 skip 从 encoder 借回来。$1\times 1$ 则连邻居都不看，只把已经对齐的 channel 向量再混合一次。

这套 inductive bias 在网格、局部统计、平移可复用的世界里非常强，强到它可以看成对 fully connected 函数族的硬约束。它仍然不根据内容改接线，也不描述任意邻居关系；那两件事分别是 Transformer 与 GNN 的起点。即便偏选对了，plain 地继续加深还会把训练本身弄难。下一章要问的就是：怎样让新层默认先别破坏已有表示，只学习必要的修正。

# 第 14 章：无监督学习（Unsupervised Learning）

> 书：《Understanding Deep Learning》Ch.14  
> 前置：第 2–9 章用输入–标签对 \((x,y)\) 学习预测；第 5 章把训练写成最大化 \(p(y\mid x)\)；第 8 章说明训练集上的成绩可以骗人。  
> 本章：标签被拿走以后，模型究竟还能学习什么？一个生成模型怎样才算“好”？  
> 后续：Ch.15 GAN、Ch.16 Normalizing Flow、Ch.17 VAE、Ch.18 Diffusion。它们都是本章坐标系里的具体赌法。

---

## 0. 标签消失以后，模型还在学什么？

前面章节的问题很明确：

```text
输入 x  →  预测标签 y
```

训练数据是成对的 \(\{(x_i,y_i)\}\)。人已经替模型决定了“什么叫重要结构”：能把 \(y\) 预测准的那些因素，才值得学。

无监督学习只剩 observations：

```text
x₁, x₂, …, xₙ
```

没有人指出哪些变化重要、哪些只是噪声。一张人脸图里，姿态、身份、光照、背景、传感器噪声全部叠在同一个 pixel 向量里。模型必须自己决定：压缩什么、忽略什么、以及“再造一条像训练数据的新样本”究竟指什么。

因此本章只追问一个母问题：

> **在没有人工标签时，怎样从数据里发现可压缩、可解释或可生成的结构，并且判断学到的东西是不是真的覆盖了数据？**

整章可以读成一条调查链：拿走 \(y\) 之后，每一步只是回答链上的下一个缺口。把链放在这里，不是为了画目录，而是为了让你在下面每换一次话题时都能指回“我们为什么正在问这个”：

```text
只有 {x_i}，没有人工 y
        │
        ▼
还能学什么？压缩 / 聚类 / 表征 / 生成 / 给样本打概率
        │
        ▼
生成往往要一个内部坐标 z：简单噪声 → 复杂样本
        │
        ▼
“好”不是一张漂亮图，而是一组互相打架的性质
        │
        ▼
其中最容易被展示图掩盖的是：看起来真  vs  覆盖得全
        │
        ▼
每一把尺子（likelihood / IS / FID / precision-recall）
都只照亮其中一个缺口
        │
        ▼
GAN / Flow / VAE / Diffusion 在同一组轴上做不同交换
```

这条链的第一个缺口最容易被一个误会挡住，所以先拆开：

> **Unsupervised、generative、probabilistic 不是三个同义词，也不是严格套娃。**

它们回答的是三个不同问题：训练时有没有人工标签、能不能造出新样本、能不能给样本一个 \(p(x)\)。这三个问题不是三档递增的难度，而是三条可以独立打勾或打叉的轴；第四轴再问要不要发明内部坐标 \(z\)。后面四章之所以看起来像同一类东西，只是因为原书把镜头对准了“无人工标签的深度生成模型”；不是因为这三个词本来就是一件事。

---

## 1. 从「只有 \(x\)」出发：先分清四扇不同的门

上一节说“三个词是三根独立的轴”，这一节就把轴一根一根钉住：无标签本身只说明训练信号从哪来，不等于会生成；会生成不等于会写 \(p(x)\)；会写 \(p(x)\) 也不等于必须显式发明一个低维 \(z\)。四件事不是楼梯，是四扇可以分别开关的门。

### 1.1 发现结构：unsupervised model

广义上，只要训练数据没有人工 target labels，就属于 unsupervised learning。没有 \(y\) 之后，目标可以很不一样，而且**并不自动排成一条由弱到强的梯子**：clustering 要的是“这些点该待在一起”，dimensionality reduction 要的是更短的坐标，density estimation 要的是“空间中哪里稠”，generation 要的是再造一条新样本。同一套无标签数据，可以只做其中一件，也可以几件一起做。

最熟的例子是 k-means。它把每个 data point 映成一个离散的 cluster assignment：

\[
x \longmapsto z \in \{1,2,\ldots,K\}.
\]

这里的 \(z\) 已经是一个 latent variable：训练时没人把它当标签交上来，它是模型自己发明的“这个点属于哪一堆”。k-means 确实在学结构——哪几张脸更像一类——但它通常不会因此获得一个能画出新照片的机器。你最多能拿出某个簇的 centroid，那是平均脸，不是新样本。

> **没有标签，不等于会生成。Unsupervised 只声明训练信号从哪来，不声明模型具备哪种能力。**

另一条容易混进来的岔路是 self-supervised learning。它同样不需要人工 \(y\)，但会从数据里**构造**一个预测目标，例如把一句话遮掉一个 token 再预测它。训练形式看起来很像监督学习，监督却不是人标的。现代语境常把它单独列出；广义上它仍落在“无人工标签”这一侧。原书后面四章不走这条路，走的是生成。

把这条轴钉住后，自然想问：那“走生成”到底走的是什么？这就是下一扇门。

### 1.2 能制造类似数据：generative model

理解一堆数据的一种强硬方式是：你能不能再从里面抽一份出来？

Generative model 要的就是这件事。常见做法是先在一个简单空间里采样，再映射到数据空间：

\[
z \sim p(z),
\qquad
x = G_\theta(z).
\]

```text
simple latent z          e.g. Gaussian noise
        │
        ▼ generator Gθ
complex sample x         e.g. a face
```

这里的 \(z\) 先当占位符用：一个好采样的内部坐标。它为什么常常必要、它是不是压缩、两条映射方向是不是一回事，暂时按下不表。眼下只需接受一个常见写法：先抽简单的 \(z\)，再由 \(G_\theta\) 折成复杂的 \(x\)。

这一步比 clustering 强在哪里？生成器必须抓住**变化的因素**，而不只是决策边界或簇中心。只会把人脸和非人脸分开，画不出一张新脸；要把噪声变成脸，姿态、光照、身份这些因素总得以某种形式进到 \(G_\theta\) 里。

但“能 sample”仍然很弱。GAN 就是典型：它很会造图，却通常不告诉你这张图在模型眼里有多可能。

### 1.3 还能给样本打概率：probabilistic generative model

若要回答 sample 自己回答不了的问题，就需要下一扇门：模型不仅能抽出新的 \(x\)，还定义一个分布 \(p_\theta(x)\)。于是可以问：

- 这个样本在模型下有多大 probability / density？
- 一段没见过的 test data，模型觉得有多像自己的数据？
- 某个点是不是该被当成 outlier？

第 5 章里，有标签时我们最大化的是 \(p(y\mid x)\)。现在没有 \(y\)，最大似然的对象变成数据本身：

\[
\hat\theta
=
\arg\max_\theta
\prod_{i=1}^{N} p_\theta(x_i).
\]

连乘许多小于 1 的数会下溢，而且优化习惯写成最小化，于是训练目标变成 negative log-likelihood：

\[
\boxed{
\mathcal L(\theta)
=
-\sum_{i=1}^{N}\log p_\theta(x_i).
}
\]

这里有一个比“公式长这样”更重要的事实：**概率必须归一化**。你把更多 density 堆到某些点上，别处就得降下来。所以最大化训练点的 likelihood，并不是只在那些点插一根针；它同时也在惩罚“离数据很远的地方还剩太多概率”。后面用 test likelihood 检查 coverage，靠的就是这条约束。

到这里必须把两个分布分开，后面所有评价都在这两个对象之间拉锯：

| 符号 | 谁拥有它 | 我们能否直接写出 |
|------|----------|------------------|
| \(p_{data}(x)\) | 真实世界的数据规律 | 不能。只能拿到有限 samples \(D=\{x_i\}\) |
| \(p_\theta(x)\) | 参数 \(\theta\) 定义的模型 | 有时能精确算（Flow），有时只能 bound（VAE / Diffusion），有时根本不定义（典型 GAN） |

三条 data flow 也不要混：

```text
真实世界：  X ~ p_data                      → 得到 training sample x
模型训练：  dataset D                       → 调整 θ，使 pθ 接近 p_data
模型生成：  x ~ pθ(x)
            有 latent 时：z ~ p(z), x ~ pθ(x|z)
```

训练看见的是第二条；生成走的是第三条；我们真正关心的是第三条有没有模仿成功第一条。FID、precision/recall 后来做的事，就是在没有 \(p_{data}\) 解析式的情况下，比较“真实样本那一朵云”和“模型样本那一朵云”。

连续数据上还有一个记号陷阱。图像、语音的 \(p_\theta(x)\) 通常是 **density**，不是“这个精确像素点发生的概率”。Density 可以大于 1。例如 \([0,0.1]\) 上的均匀分布，density 处处等于 10；真正的 probability 必须对一个区域积分。比较 likelihood 时，默认大家用同一套数据表示、单位和 preprocessing。把 pixel 从 \([0,255]\) 改成 \([0,1]\)，连续 density 的数值会整体换一把尺子——不是模型突然变好了。

到这里，“无标签”和“能不能生成”“能不能打概率”已经分开，但它们都没有承诺模型内部一定要有一个未观测的 \(z\)。这个结构选择就是第四扇门。

### 1.4 内部要不要发明一个 \(z\)：latent variable model

前三扇门本身都不承诺 \(z\) 的语义：k-means 的簇编号只是离散标签，GAN 的噪声 \(z\) 只是采样入口，概率模型也可以完全不写 \(z\)。Latent variable 这一扇门单独问：模型里有没有一个训练时没被观测到的变量，并且这个变量被当作结构的一部分来使用。

Latent variable 的意思很窄：训练时没有被直接观测到的变量。它**可能**表示人脸姿态、光照、身份，或语音里的说话者与内容。注意“可能”——这是建模时的愿望，不是学完以后自动交付的因果因子。

映射可以有两个方向，而且它们不是一回事：

```text
x → z：encoding / clustering / representation
       看见一张脸，推断它的内部坐标

z → x：decoding / generation
       抽一个内部坐标，折成一张新脸
```

k-means 主要走上面那条；GAN 的 generator 主要走下面那条；VAE 两条都要，但生成新样本时仍然只走 \(z\sim p(z)\) 再 decode。把一张图 encode 再 decode，那是 reconstruction；从 prior 里随机抽 \(z\) 再 decode，才是 generation。Autoencoder 会重构，不等于它已经是一个好 generator——prior 抽到的 \(z\) 可能落在 decoder 从没见过的区域，输出会垮。

还有一个几乎人人会踩的坑：

> **Latent 的意思是“未观测”，不是“一定比 \(x\) 更低维”。**

同一个词在不同架构里负载完全不同：k-means 把 \(z\) 压成一个离散整数；VAE 通常把它设计成低维连续向量；Normalizing Flow 的 \(z\) 与 \(x\) 同维，本质是换一套可逆坐标；Diffusion 的 noisy state 甚至和图像同 shape。所以“latent 常常是压缩版 \(x\)”只是 clustering / VAE 那一类模型的直觉，不是定义；Flow 和 Diffusion 的 \(z\) 首先服务于概率计算或逐步去噪，不是为了给你一个 32 维的“人脸基因”。

最后，不是所有 probabilistic generator 都必须显式使用一个低维 \(z\)。Transformer decoder 就是重要反例：它用

\[
p(x)=\prod_t p(x_t\mid x_{<t})
\]

给整段序列算概率、也能逐 token 生成，训练信号来自数据自己的下一个 token，并没有一个单独的低维姿态向量。

### 1.5 四个词钉在四条轴上，不是套娃集合

把四个问题做成表格，套娃立刻破掉。每一行不是“更强的模型”，只是四列答案不同的组合：

| 模型 | 无人工标签？ | 能生成？ | 定义可讨论的 \(p(x)\)？ | 显式低维 latent？ |
|------|--------------|----------|-------------------------|-------------------|
| k-means | 是 | 基本不能 | 否 | 离散簇 |
| 典型 GAN | 常是 | 是 | 否（implicit generator） | 是 |
| Flow | 是 | 是 | 精确可算 | 同维坐标，未必压缩 |
| VAE | 是 | 是 | 常用 ELBO lower bound | 是 |
| Diffusion | 是 | 是 | 常用 bound | noisy \(x\)，基础形式无语义压缩 |
| Transformer decoder | self-supervised | 是 | 按 chain rule 可算 | 不必有低维 \(z\) |
| conditional GAN | 否，用了 class/text | 是 | 通常否 | 是 |

这张表的四个问题就是四条轴，轴与轴之间可以独立拨动：同样是 generative，conditional GAN 把训练信号轴拨到 supervised，Flow 把概率轴拨到 exact；同样是 latent variable model，Flow 的 \(z\) 是同维坐标，VAE 的 \(z\) 才追求压缩。不要把“用了 latent”自动等同于“低维”“语义”“能打分”中的任何一项。

“Generative”描述能力，“unsupervised”描述训练信号。Conditional image generator 用 class labels 或 text–image pairs 训练，它可以很 generative，却不是 unsupervised。本章后面默认讨论的是这两条轴的交集——无人工标签的生成模型——但请把交集当成原书的选题，而不是三个词的数学包含关系。

下一个缺口是：既然生成常常写成 \(z\mapsto x\)，为什么非要绕这个弯？直接在 pixel 空间里写一个分布，不行吗？

---

## 2. 为什么后面四章几乎都要一个 Latent Space？

上一节最后留下的问题还在：生成常常写成 \(z\mapsto x\)，为什么不直接在 pixel 空间里写一个 \(p(x)\)？

因为现实数据的 ambient space 极大，有意义的数据却只占其中薄薄一层。

一张 \(256\times 256\times 3\) 的图有约 \(2\times 10^5\) 个像素。如果每个像素独立乱填，得到的几乎全是雪花噪声。“自然人脸”不是这些像素的任意组合，而是嵌在高维空间里的一张弯曲薄片——常叫 data manifold：

```text
Huge pixel space
┌──────────────────────────────────┐
│  random noise                    │
│                                  │
│          ┌────────────┐          │
│          │ face sheet │          │
│          └────────────┘          │
│                                  │
└──────────────────────────────────┘
```

直接在 pixel 空间写一个简单分布，比如一个大 Gaussian，会把绝大部分概率浪费在雪花上。更顺的建模顺序是：

```text
z：简单、规则、容易采样（Gaussian、离散簇、噪声图）
        │  nonlinear generator
        ▼
x：落在那张薄片附近的复杂数据
```

思想是：

> **先在一个好采样的坐标系里动手，再学一张把该坐标系折进数据薄片的非线性地图。**

这张地图就是 generator / decoder。\(z\) 是地图上的坐标，不是数据里已经写好的标签。

前面 Transformer 那个反例同时提醒我们：直接在 \(x\) 空间里写一个复杂分布的路确实存在，所以这里说“几乎总要”而不是“必须”。问题只在于，后面四章想要的不是“能算概率”这一件事，还要能从好采样的坐标出发把新样本造出来；对图像这种高维对象，直接设计一个又简单、又好采样、又只落在人脸薄片上的 \(p(x)\)，比先设计一个简单 \(z\)、再学一张非线性 map 更难起步。

但要小心，“低维 \(z\) 的每一维对应一个真实语义因素”只是愿望，不是自动结果。即使 \(p_\theta(x)\) 学对了，\(z\) 的坐标系仍可以任意旋转、重参数化，而不改变观测分布；没有额外 supervision 或 inductive assumptions，disentanglement 往往不可唯一识别。下一节把“well-behaved latent space”和“disentangled latent space”分成两条轴，原因就在这里：前者说几何别太疯，后者说坐标轴最好正好对齐人类概念——后者苛刻得多。

有了“用 \(z\) 造 \(x\)”这条生产线，挑剔的问题马上出现：怎样才算造得好？只把 16 张最好看的图贴出来，够不够？

---

## 3. 一张好看的图远远不够：六个评价轴

上一节的问题正好替我们请来一位不买账的审稿人：你刚训完一个人脸生成器，贴出 16 张最好看的图，他不会停在“像不像照片”这一问上。原书的六条 desirable properties，其实就是他会连着问的六件事；它们彼此拆台，所以不存在单一冠军。问题越具体，越能看出为什么“漂亮图”只是及格线而不是满分线。

第一问是速度。从模型生成一个样本应足够快，并能吃到 GPU 的并行。GAN 通常一步 \(z\mapsto x\)；基础 Diffusion 要几十到上千步 denoising。如果应用是交互式编辑，“图更漂亮但要等 30 秒”可能直接被否。

第二问才是大家最熟悉的 **fidelity**：随便拿一张生成图，会不会一眼看出假？生成样本应与 real data 难以区分。Sample grid 主要回答这一问。

第三问立刻拆穿展示图。模型不应只会生成训练分布的一小部分。人脸包含年龄、肤色、姿态、表情等多种模式；如果只生成年轻正脸，即使每张都像照片，**coverage** 仍然很差：每张都真，但只会一种。GAN 里这种失败常叫 **mode collapse**——generator 找到一个 discriminator 很难骂的模板，就反复使用。你当然会把那一种模板里最好看的 16 张贴出来，所以展示图对 coverage 几乎没有抵抗力。

第四问转向内部坐标。Well-behaved latent space 要求：大多数 \(z\) 都映射到合理样本，而不是只有一小撮“好噪声”；小幅改变 \(z\)，输出 \(x\) 也平滑改变；相近的 \(z\) 对应语义相近的样本。检验手段之一是 interpolation：在两个合理 latent points 之间连线。如果中途穿过三眼怪物或雪花，轨迹已经离开 data manifold。

第五问更苛刻：每个 latent dimension 能不能只拧一个旋钮——\(z_1\) 管 pose，\(z_2\) 管 lighting，\(z_3\) 管 expression？这叫 disentanglement。没有 supervision 或额外假设时，它往往不可唯一识别：即使 \(p_\theta(x)\) 学对了，\(z\) 的坐标系仍可旋转而不改变观测分布。原书给四类模型在这一栏几乎都打了问号，不是疏忽。它和上一问不是一回事：几何别太疯，不等于坐标轴正好对齐人类概念。

第六问和“图漂不漂亮”几乎正交。若模型定义了 \(p_\theta(x)\)，希望能快速、准确地计算 test likelihood。Flow 通常可以精确算；VAE / Diffusion 的基本形式常优化或计算 lower bound；GAN 通常根本不定义可计算的 likelihood。这一条检查的是概率模型完不完整，不是人类观感。

六问并排，交换立刻出现。一步出图的模型，往往没有逐步精修带来的稳定性；坚持 exact likelihood 的模型，architecture 会被可逆性捆住，sample 常常没那么锐；把 \(z\) 留成语义坐标，又和“\(z\) 只是同维噪声”的设计冲突。原书 Figure 14.3 给四类模型打分，精确位置可以争，但结论很难争：

> **没有一种模型在所有轴上同时最优。后面四章是四组不同的交换，不是四次对同一奖杯的冲击。**

六条里最常被一张漂亮图一次性糊弄过去的，是 fidelity 和 coverage。下一节把它们抽成二维坐标。后面所有 metric，几乎都是在这两维上找不同的探针。

---

## 4. Fidelity 与 Coverage：最核心的二维坐标

审稿人的第二问和第三问看起来都像在说“模型不好”，诊断却完全相反。把它们做成二维，后面才知道每把尺子在看哪一维。

### 4.1 样本不真实：低 precision / 低 fidelity

```text
Real manifold:      ● ● ● ●
Generated samples:  ● × × ●
                       ↑
                    fake / implausible
```

生成样本落到 real-data manifold 外面：五官扭曲、纹理崩了、物体结构不合法。问法是：

> **随便拿一个 generated sample，它像不像真实数据？**

这是 precision 视角。Sample grid 主要看这个。

### 4.2 模式没覆盖：低 recall / 低 coverage

```text
Real modes:       cat  dog  bird  horse
Generated modes:  cat  dog
```

模型生成的内容可能都很真实，但 bird 和 horse 消失了。问法反过来：

> **随便拿一个 real sample / real mode，模型覆盖到它了吗？**

这是 recall 视角。只看 16 张漂亮图几乎测不到它。模型可以从一个很窄的 mode 里反复挑最好看的结果，把真实数据的大部分变化都漏掉。

### 4.3 四象限：mode collapse 只是其中一格

| Precision / Fidelity | Recall / Coverage | 生成器在做什么 | 典型诊断 |
|----------------------|-------------------|---------------|----------|
| 高 | 高 | 样本大多真实，也覆盖主要 modes | 理想目标 |
| 高 | 低 | 每张都很真，但只会少数模板 | mode collapse |
| 低 | 高 | 到处都生成，却有大量不真实样本 | support 太宽、质量差 |
| 低 | 低 | 既不真实，也没有覆盖数据 | 模型、目标或训练失败 |

Mode collapse 被单独拿出来喊，是因为它最能骗过眼睛：高 fidelity、低 coverage。反方向同样常见，只是更难看，展示时会被作者自己滤掉——support 铺得很开，图却很假。

所以评价生成模型的第一原则是：

> **“看起来真”与“覆盖得全”必须分开评估。随机展示 16 张图，几乎不能证明后者。**

下一节开始引入定量尺子。顺序不是“再背四个公式”，而是：上一把尺子照不到的那个缺口，逼出下一把尺子。

---

## 5. 第一把尺子：Test Likelihood

上一节只给了“概念上的两维”，还没有给数字。第一把尺子选最完整的入口：如果模型定义了一个归一化的 \(p_\theta(x)\)，那么最自然的问题就是——训练时没见过的真实样本，模型还认不认？这批样本记作

\[
D_{\mathrm{test}}=\{x_1,x_2,\ldots,x_N\}.
\]

其中，\(x_i\) 是第 \(i\) 个 test sample，\(\theta\) 是学到的参数，\(p_\theta(x_i)\) 是模型给这个点分配的 probability density（离散数据时就是 probability）。

如果暂时假设各个 test samples 是独立抽到的，那么模型赋给整批数据的 likelihood 是各样本 likelihood 的乘积：

\[
p_\theta(D_{\mathrm{test}})
=
\prod_{i=1}^{N}p_\theta(x_i).
\]

只要模型很不认可其中一部分真实样本，对应的 \(p_\theta(x_i)\) 很小，整个乘积就会被拉低。

实际计算不喜欢连乘：大量小数相乘容易 numerical underflow，乘法也不如加法好优化。取 logarithm 后，连乘变成连加，这就是通常报告的 test log-likelihood：

\[
\log p_\theta(D_{\mathrm{test}})
=
\sum_{i=1}^{N}\log p_\theta(x_i).
\]

log 只是把连乘变成连加，没有改变模型之间的优劣顺序。总和越大，说明模型整体越认可这批 test samples。训练时常见的 negative log-likelihood（NLL）只是再加一个负号：

\[
\operatorname{NLL}(D_{\mathrm{test}})
=
-\sum_{i=1}^{N}\log p_\theta(x_i).
\]

因此 log-likelihood 是越大越好，NLL 是越小越好；它们衡量的是同一件事。

这里的关键词是 test。若只看 train likelihood，模型可能记住训练集，在每个训练点附近堆出很高的 density，点与点之间却是一片空白。换成没见过的 \(D_{\mathrm{test}}\)，才是在检查模型有没有学到能够泛化的数据规律。这和第 8 章的教训完全相同：训练集上的高分不能证明模型理解了总体分布。

为什么 likelihood 多少能检查 coverage？写训练目标 NLL 时已经埋下伏笔：\(p_\theta\) 必须归一化。模型把概率过多地塞进“年轻人正脸”这个小角落，就没有足够的概率留给老年人、侧脸和其他真实模式；这些 test samples 的 log-likelihood 会下降。它因此能检查模型是否认得真实数据，但仍然没有回答“生成的图看起来是否自然”。

这里还藏着一个容易误解的地方：**高 density 不等于典型 sample**。以 \(d\) 维 standard Gaussian 为例：

\[
X\sim\mathcal N(0,I_d),
\]

这里 \(\mathcal N(0,I_d)\) 表示各坐标独立、方差为 1 的 Gaussian。因为

\[
\mathbb E\|X\|^2=d,
\]

其中 \(\mathbb E\) 是期望，\(\|X\|\) 是到原点的距离；且高维时 \(\|X\|^2\) 会集中在 \(d\) 附近，所以随机抽到的典型样本离原点约 \(\sqrt d\)。可是 point density 在原点最高，于是大多数样本落在半径约 \(\sqrt d\) 的薄壳上，而不是 density 最高的原点：

```text
point density 最高：        原点
随机抽样最常落到：          半径约 √d 的薄壳
```

图像模型中也会出现类似的错位。模型可能把背景统计和局部纹理学得很好，因此 test likelihood 很高，但生成图在人眼看来仍然模糊。反过来，GAN 可以生成很锐利的图片，却根本没有可计算的 \(p(x)\)。

因此，test likelihood 回答的是：“模型给没见过的真实数据分配了多少 density？”它不能单独回答“随机生成的图片在人眼看来是否自然”。这就是第一道缺口：likelihood 只适用于有概率语义的模型，而且概率评价与感知质量不是一回事。我们还需要一把不要求显式 \(p(x)\)、直接从生成图判断“像不像”的尺子。下一节的 IS 就是这个方向的早期尝试。

---

## 6. 没有 likelihood，先借一双分类器的眼睛：Inception Score

上一节留下两个现实问题：典型 GAN 没有可计算的 \(p(x)\)，而 likelihood 也不等于人眼观感。对图像来说，一个自然的替代方案是找一个已经训练好的 image classifier，请它充当感知探针。它不直接告诉我们真实分布是什么，只判断生成图是否像一个明确、合理的对象。

IS 只看 generated set，不看 real set。它要求生成结果同时满足两件事：

1. 对单张图，classifier 的 \(p(y\mid x)\) 要尖。比如图中确实是一只狗，而不是狗、沙发、汽车各有一点概率。
2. 对整批图，预测类别要分散。不能所有样本都被判成狗。

令
\[
p(y)=\frac1N\sum_{i=1}^N p(y\mid x_i),
\]

IS 用两者的 KL divergence 衡量“单张明确”与“全体多样”之间的差异：

\[
\boxed{
IS
=
\exp\Bigl(
\mathbb E_{x}
\bigl[
D_{KL}\bigl(p(y\mid x)\,\Vert\,p(y)\bigr)
\bigr]
\Bigr).
}
\]

直觉很简单。若每张图都被 classifier 认得很确定，但所有图都是同一类，那么 \(p(y\mid x)\) 很尖，\(p(y)\) 也集中在同一类，两者并不“不同”，KL 反而小。若每张图都很明确，同时整批样本均匀使用多个类别，KL 才会大。

但 IS 的补丁也带来了新的盲区。它没有看真实图，所以生成器即使整体偏离真实数据，只要能骗过 classifier，分数仍可能不错；它还依赖 classifier 的 label space，拿 ImageNet classifier 去评人脸或医学图像，本来就不太合适。更隐蔽的是，某个 class 只生成一个固定模板，也足以拿到不错的分数。IS 奖励的是类间多样，不奖励 class 内部的变化。

所以 IS 解决的是“生成集自己看起来是否明确且有类别变化”，并没有真正回答“它和真实数据像不像”。下一步必须把 real set 也放进比较中，这就引出 FID。

---

## 7. IS 还不看真实数据：FID 把两批样本放到同一张地图

上一节说 IS 的盲区是“不看 real set”，修复方向就明摆着：把 real images 和 generated images 都送进同一个预训练网络，在同一个 feature space 里比较两朵云。

不能直接比较 raw pixels。猫平移一个 pixel，像素 L2 距离可能很大，人却仍把它看成同一只猫。FID 借用 Inception network 的中间 feature，假定这些 feature 比像素更接近语义相似度。

高维 feature 的完整分布仍然很难估。FID 退一步，只保留每朵云的均值和协方差，把它们近似成
\[
\mathcal N(\mu_r,\Sigma_r),
\qquad
\mathcal N(\mu_g,\Sigma_g).
\]

然后计算两个 Gaussian 之间的 Fréchet distance：

\[
\boxed{
\|\mu_r-\mu_g\|^2
+
\operatorname{Tr}\left(
\Sigma_r+
\Sigma_g
-2\bigl(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\bigr)^{1/2}
\right).
}
\]

越小越好。均值项看两朵云的中心是否偏移，协方差项看云的形状和张开程度是否相近。因此 FID 比 IS 更像是在比较 \(p_{data}\) 和 \(p_\theta\)，但它比较的其实是 feature space 中两个 Gaussian 的摘要，不是原始分布本身。

这也解释了它的局限：真实 feature 云可能是多峰的，均值和协方差会漏掉死角；结果依赖 Inception feature extractor，网络忽略的细节不会进入分数；有限样本还会带来估计偏差。最关键的是，FID 只有一个数。中心偏了、云变瘪了、生成了许多不真实的点，都可能让 FID 变差，但它不会告诉你是哪一种失败。

IS 补的是“没有 real set”，FID 补上后又把 fidelity 和 coverage 压成了一个距离。下一节干脆把这两个问题拆开。

---

## 8. FID 只有一个数：把 fidelity 和 coverage 拆开

FID 已经同时看了 real set 和 generated set，却只交回一个数。实际调模型时，这个数不够用：我们得知道问题是“生成了很多假样本”，还是“只覆盖了真实数据的一小块”。Manifold precision / recall 就是为这个诊断设计的。

把 feature space 中真实样本占据的区域记作 data manifold，把生成样本占据的区域记作 model manifold。于是：

\[
\text{Precision}
=
\frac{\text{落在 data manifold 内的 generated samples}}
{\text{全部 generated samples}},
\]

\[
\text{Recall}
=
\frac{\text{落在 model manifold 内的 real samples}}
{\text{全部 real samples}}.
\]

Precision 问：“我生成的东西，大多数像不像真的？”它接近 fidelity。Recall 问：“真实数据的各种模式，我覆盖了多少？”它接近 coverage。

例如，100 个 generated samples 里有 80 个落在 real manifold 内，precision 是 \(0.8\)；100 个 real samples 里只有 40 个被 model manifold 覆盖，recall 是 \(0.4\)。这不是一句“FID 不够低”能说明的，而是一个很具体的处方：样本多数还算真实，但生成分布太窄，需要优先补 coverage，而不是继续只优化单张图的锐度。

当然，manifold 通常没有解析式。实践中会在 feature space 里用 k-nearest-neighbor 的邻域近似它：每个样本周围画一个球，半径取到第 \(k\) 个邻居的距离，把这些球的并看成该类样本的区域。生成点落进 real balls，算作 precision；真实点落进 generated balls，算作 recall。

这个近似也不能忘。\(k\) 太小，区域碎成孤岛，precision / recall 可能被低估；\(k\) 太大，球会吞掉空隙，把不真实的点也算进去。换 feature extractor 或换邻域规则，结果也会变化。Precision / recall 不是终于找到“真理”，只是比一个 FID 标量更适合定位失败类型。

---

## 9. 尺子不是排行榜，而是一轮排查

上一节的 precision / recall 解决了最后一个概念缺口，但真实工作中没人会把四把尺子全跑一遍然后报平均分。排查顺序取决于模型本身：先问它有没有概率语义，有就过 likelihood 这一关，没有就跳过；再回到样本本身，IS 检查生成集内部是否明确多样，但它不看 real set，只能当初筛；想看两朵云是否接近，用 FID；FID 分不清失败方向时，再用 precision / recall 定位。整体像下面这样走：

```text
有 pθ(x)？
   ├─ 是 → test likelihood：模型认得 unseen real samples 吗？
   └─ 否 → 跳过这关，不能拿不存在的 p(x) 硬算

所有模型都可以继续看：
   ├─ sample grid / nearest neighbors：图像是否真实，是否复读训练集？
   ├─ IS：生成集内部是否明确且有类别变化？
   ├─ FID：生成集与真实集的 feature 云整体接近吗？
   └─ precision / recall：失败主要是 fidelity 还是 coverage？
```

这张表不是另一套说法，只是把上面的排查顺序压缩成可复查的 cheat sheet：

| Metric | 它主要回答什么 | 它回答不了什么 |
|--------|----------------|----------------|
| Test likelihood | 模型对 unseen real data 的 density 是否合理 | 人眼是否喜欢；没有 \(p(x)\) 的模型无法直接用 |
| IS | 单张图是否明确、生成集是否跨多个类别 | 是否贴近 real set；class 内部是否多样 |
| FID | real / generated 两朵 feature 云整体有多近 | 失败究竟来自 fidelity 还是 coverage |
| Precision / Recall | 生成样本是否真实、真实模式是否被覆盖 | manifold 只能近似；结果依赖 feature 和邻域选择 |
| Human evaluation | 人的观感和偏好 | 成本、主观性、复现性 |

因此评价一个 generator，至少要把“看图”“看邻居”“看两朵云”“看覆盖方向”分开。若模型有概率语义，再补 likelihood 或它的 bound；若模型用于医学、编辑或下游分类，还要加 domain-specific task evaluation。没有一把 metric 能把一个未知高维分布压缩成无损的单个数字，每个分数都只是从某个角度投影出来的影子。

这套排查顺序也正好把后面四章接起来：GAN 可能绕过 likelihood，却在 sample quality 上很强；Flow 能给出 exact likelihood，却受可逆结构限制；VAE 用 ELBO 换取可训练的 latent 概率模型；Diffusion 用多步去噪换质量和稳定性。接下来比较模型时，先问它在哪一关下注，再问它在哪一关付出了代价。

---

## 10. 四类生成模型：同一组轴上的不同交换

上一节的排查表其实已经预告了这四章：没有一把尺子全能，所以模型只能先选“我在哪一关下注”。原书后面四章都从某种 latent 出发，用深度网络把它映到数据空间；差别不在“是不是生成模型”，而在它们愿意丢掉哪一条性质，去换哪一条性质。下面按“换来什么 / 丢掉什么”读，不要按“谁比谁强”读。

### GAN

```text
z → generator → x_fake
x_real / x_fake → discriminator
```

它把“像不像真的”交给一场对抗：generator 造假，discriminator 抓假。训练信号是“别被抓住”，不是 \(\log p(x)\)。

- 换来的：一步采样、图像可以很锐（fidelity 上限高）；
- 丢掉的：可计算 likelihood；
- 典型风险：mode collapse，也就是高 fidelity、低 coverage。

### Normalizing Flow

```text
x  ↔  z：可逆 mapping，Jacobian 可算
```

GAN 把“给样本打概率”整个丢掉，下一家恰好把这句话当成底线：Flow 坚持 \(p_\theta(x)\) 必须能精确写出来。办法是从简单密度出发，做一串可逆变换，用 change of variables 把密度跟着搬过来。

- 换来的：exact likelihood、exact latent mapping；
- 丢掉的：architecture 必须 invertible，表达能力被捆住，sample 往往没 GAN / Diffusion 那么锐。

### VAE

```text
x → approximate posterior q(z|x)
z → probabilistic decoder p(x|z)
```

Flow 的代价是 architecture 必须可逆。如果不想被可逆性捆住，就得换一个问题：能不能保留不可逆的 decoder，却仍然按概率训练？VAE 想要 latent variable model 的完整概率故事，但 \(p(x)=\int p(x\mid z)p(z)\,dz\) 通常积不出来，于是引入 encoder 去近似 posterior，优化 likelihood 的 lower bound（ELBO）。这是第 17 章的母问题。

- 换来的：概率故事清楚、latent space 通常较规则、训练稳定；
- 丢掉的：exact likelihood；样本常偏平滑（decoder 把不确定性吃进模糊里）。

### Diffusion

```text
x → gradually add noise → ε          （forward，固定不学）
gradually denoise ε → x              （reverse，才是模型）
```

VAE 里的 latent 仍是一个想抓语义的压缩向量。Diffusion 换了一条更谦逊的路：不要求 \(z\) 有语义，直接把“一步从噪声跳到图像”拆成许多小去噪步骤。第 18 章会说明：forward 加噪是设计好的，learned 的是 reverse。

- 换来的：训练稳定、sample quality 高，coverage 经验上往往也好；
- 丢掉的：多步 sampling 慢；基本形式优化的是 likelihood bound；这条 noise path 通常不是可拧的语义旋钮。

原书 Figure 14.3 的教学快照（赋值可以争，结构不要记反）：

| 模型 | 采样速度 | 样本质量 | Coverage | Latent 几何 | Disentangle | Likelihood |
|------|----------|----------|----------|-------------|-------------|------------|
| GAN | 快 | 高 | 弱 | 往往可用 | ？ | 无 |
| VAE | 快 | 常偏糊 | ？ | 较规则 | ？ | bound |
| Flow | 快 | 往往不够锐 | ？ | 可逆坐标 | ？ | 精确 |
| Diffusion | 基础形式慢 | 高 | 往往较好 | 噪声路径，非语义 | 弱 | bound |

这张表是当时的教学切片。后来有更快的 Diffusion sampler、更好 coverage 的 GAN、latent diffusion 之类的杂交；读它时抓住“交换”即可，不必把它当成永远的排行榜。

---

## 11. 自测

1. Unsupervised、generative、probabilistic generative 分别回答什么问题？为什么不是套娃？  
2. 为什么 k-means 是 unsupervised，却通常不算强 generative model？  
3. 为什么 Transformer decoder 能算 \(p(x)\)、能生成，却不必有一个低维语义 \(z\)？  
4. 训练、真实世界采样、模型生成三条 data flow 各从哪个分布里取点？  
5. 为什么最大化 likelihood 会间接惩罚远离数据的区域？  
6. Latent variable 一定比 \(x\) 维度更低吗？用 Flow 或 Diffusion 反驳。  
7. Fidelity 与 coverage 分别问哪句话？为什么 sample grid 几乎不能证明后者？  
8. 为什么 train likelihood 不能代替 test likelihood？这和第 8 章哪条教训是同一件事？  
9. 高维 Gaussian 的最高 density 点为什么往往不是典型 sample？这件事警告我们 likelihood 和“看起来像”可能分家。  
10. IS 里 \(p(y\mid x)\) 很尖、但全体都是同一类时，为什么分数仍然低？每个类只放一张模板时，为什么分数可以虚高？  
11. FID 的两项各自在比较什么？它为什么仍分不清 fidelity 和 coverage？  
12. Precision \(=0.8\)、recall \(=0.4\) 该怎样向模型开药，而不是只说“FID 还不够低”？  
13. GAN、Flow、VAE、Diffusion 各自用什么换什么？  
14. 比较连续 likelihood 时，为什么必须先对齐 preprocessing 和 measurement convention？

---

## 12. 合上书再看一眼

无监督学习只给你 \(\{x_i\}\)。模型必须自己决定哪些结构值得压缩、聚类或建模；没有人用 \(y\) 把重要性标好。

会生成，是对“我理解这个分布”的一种强硬检验：你得能再造一份。若还能写出 \(p_\theta(x)\)，检验可以升级为 likelihood。内部坐标 \(z\) 常常是为了让“从简单噪声走到复杂样本”变得可学——它是未观测变量，不一定是压缩，更不一定是可拧的语义旋钮。

一个好 generator 要同时面对采样速度、样本真实性、分布覆盖、latent 几何，以及（若需要）精确打分。这些轴互相拆台。IS、FID、precision/recall、test likelihood 每把尺子只照亮其中一个缺口；没有单一 metric，也没有单一模型，在所有维度上完胜。

后面四章可以当成四次不同的下注来读：GAN 丢掉 \(p(x)\) 换锐度和速度，Flow 用可逆性换精确 likelihood，VAE 用 ELBO 换可训的 latent 概率模型，Diffusion 用多步去噪换质量和稳定性。评价它们时，先问自己在看哪一条轴，再决定该信哪一把尺子。

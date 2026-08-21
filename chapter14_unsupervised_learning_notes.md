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

完整调查链如下。后面每一节都只多回答这一条链上的一个缺口：

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

有一个容易一上来就黏住的误会，先拆开：

> **Unsupervised、generative、probabilistic 不是三个同义词，也不是严格套娃。**

它们回答的是三个不同问题：训练时有没有人工标签、能不能造出新样本、能不能给样本一个 \(p(x)\)。后面四章之所以看起来像同一类东西，只是因为原书把镜头对准了“无人工标签的深度生成模型”；不是因为这三个词本来就是一件事。

---

## 1. 从「只有 \(x\)」出发：先分清三件不同的事

拿走 \(y\) 之后，模型不是突然只会做一件事。它可以继续发现结构，也可以学习生成，还可以给样本写一个 \(p(x)\)。后两档对后面四章更关键：能 sample 的模型不必会打概率（典型 GAN）；会写 \(p(x)\) 的模型，还要另外问能不能高效 sample。这是能力上的差别，不是三个词的包含关系。第四个问题更是另一条轴：要不要发明内部坐标 \(z\)。

### 1.1 发现结构：unsupervised model

广义上，只要训练数据没有人工 target labels，就属于 unsupervised learning。没有 \(y\) 之后，目标可以很不一样，而且**并不自动排成一条由弱到强的梯子**：clustering 要的是“这些点该待在一起”，dimensionality reduction 要的是更短的坐标，density estimation 要的是“空间中哪里稠”，generation 要的是再造一条新样本。同一套无标签数据，可以只做其中一件，也可以几件一起做。

原书后面四章都走生成这条路，所以下面只把其中两档——会不会造新样本、会不会给样本写 \(p(x)\)——按能力排成梯子。其余目标先放在“发现结构”这一层里，用 k-means 钉住。

最熟的例子是 k-means。它把每个 data point 映成一个离散的 cluster assignment：

\[
x \longmapsto z \in \{1,2,\ldots,K\}.
\]

这里的 \(z\) 已经是一个 latent variable：训练时没人把它当标签交上来，它是模型自己发明的“这个点属于哪一堆”。k-means 确实在学结构——哪几张脸更像一类——但它通常不会因此获得一个能画出新照片的机器。你最多能拿出某个簇的 centroid，那是平均脸，不是新样本。

所以：

> **没有标签，不等于会生成。Unsupervised 只声明训练信号从哪来，不声明模型具备哪种能力。**

另一条容易混进来的岔路是 self-supervised learning。它同样不需要人工 \(y\)，但会从数据里**构造**一个预测目标，例如把一句话遮掉一个 token 再预测它。训练形式看起来很像监督学习，监督却不是人标的。现代语境常把它单独列出；广义上它仍落在“无人工标签”这一侧。原书后面四章不走这条路，走的是生成。

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

这里的 \(z\) 先当占位符用：一个好采样的内部坐标。它为什么常常必要、它是不是压缩、两条映射方向是不是一回事，放到 1.4 和第 2 节再钉。眼下只需接受一个常见写法：先抽简单的 \(z\)，再由 \(G_\theta\) 折成复杂的 \(x\)。

这一步比 clustering 强在哪里？生成器必须抓住**变化的因素**，而不只是决策边界或簇中心。只会把人脸和非人脸分开，画不出一张新脸；要把噪声变成脸，姿态、光照、身份这些因素总得以某种形式进到 \(G_\theta\) 里。

但“能 sample”仍然很弱。GAN 就是典型：它很会造图，却通常不告诉你这张图在模型眼里有多可能。

### 1.3 还能给样本打概率：probabilistic generative model

再强一档：模型不仅能抽出新的 \(x\)，还定义一个分布 \(p_\theta(x)\)。于是可以问一些 sample 本身回答不了的问题：

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

### 1.4 内部要不要发明一个 \(z\)：latent variable model

上面三档说的是能力：发现结构、会生成、会打概率。每一档都可以带、也可以不带内部坐标。1.2 里的 \(z\) 只是生成时最常见的那一招，不是生成的定义。

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

| 模型 | \(z\) 是什么 | 是否压缩 |
|------|----------------|----------|
| k-means | 离散簇编号 | 是，压成一个整数 |
| VAE | 低维连续向量 | 通常是，这是设计目标 |
| Normalizing Flow | 与 \(x\) 同维的 Gaussian 坐标 | 通常不是压缩，是可逆换坐标系 |
| Diffusion | 与图像同 shape 的 noisy state | 基础形式并不做语义压缩 |

原书说“latent 常常是压缩版的 \(x\)”，那是对 clustering / VAE 那一类模型的直觉。Flow 和 Diffusion 也使用 unobserved variable，但它们的 \(z\) 首先是为了让概率计算或逐步去噪变得好做，不是为了给你一个 32 维的“人脸基因”。

最后，不是所有 probabilistic generator 都必须显式使用一个低维 \(z\)。Transformer decoder 就是重要反例：它用

\[
p(x)=\prod_t p(x_t\mid x_{<t})
\]

给整段序列算概率、也能逐 token 生成，训练信号来自数据自己的下一个 token，并没有一个单独的低维姿态向量。

### 1.5 四个词钉在四条轴上，不是套娃集合

把上一节的例子摊开，套娃立刻破掉：

| 模型 | 无人工标签？ | 能生成？ | 定义可讨论的 \(p(x)\)？ | 显式低维 latent？ |
|------|--------------|----------|-------------------------|-------------------|
| k-means | 是 | 基本不能 | 否 | 离散簇 |
| 典型 GAN | 常是 | 是 | 否（implicit generator） | 是 |
| Flow | 是 | 是 | 精确可算 | 同维坐标，未必压缩 |
| VAE | 是 | 是 | 常用 ELBO lower bound | 是 |
| Diffusion | 是 | 是 | 常用 bound | noisy \(x\)，基础形式无语义压缩 |
| Transformer decoder | self-supervised | 是 | 按 chain rule 可算 | 不必有低维 \(z\) |
| conditional GAN | 否，用了 class/text | 是 | 通常否 | 是 |

所以真正正交的是四条轴：

```text
训练信号轴：是否依赖人工 target labels？
  ├── unsupervised / self-supervised
  └── supervised / conditional supervision

能力轴：是否能够产生新 samples？
  ├── descriptive / clustering / representation
  └── generative

概率语义轴：是否定义可讨论的 pθ(x)？
  ├── implicit generator，例如典型 GAN
  └── probabilistic generator，例如 Flow / VAE / Diffusion

结构轴：是否引入 latent variable z？
  ├── explicit latent-variable model
  └── 不一定需要低维 latent，例如 autoregressive model
```

“Generative”描述能力，“unsupervised”描述训练信号。Conditional image generator 用 class labels 或 text–image pairs 训练，它可以很 generative，却不是 unsupervised。本章后面默认讨论的是这两条轴的交集——无人工标签的生成模型——但请把交集当成原书的选题，而不是三个词的数学包含关系。

下一个缺口是：既然生成常常写成 \(z\mapsto x\)，为什么非要绕这个弯？直接在 pixel 空间里写一个分布，不行吗？

---

## 2. 为什么几乎总要一个 Latent Space？

1.5 节末尾的问题还在：生成常常写成 \(z\mapsto x\)，为什么不直接在 pixel 空间里写一个 \(p(x)\)？

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

但“低维 \(z\) 的每一维对应一个真实语义因素”只是愿望。即使 \(p_\theta(x)\) 学对了，\(z\) 的坐标系仍可以任意旋转、重参数化，而不改变观测分布。没有额外 supervision 或 inductive assumptions，disentanglement 往往不可唯一识别。后面把“well-behaved latent space”和“disentangled latent space”分成两条轴，原因就在这里：前者说几何别太疯，后者说坐标轴最好正好对齐人类概念——后者苛刻得多。

有了“用 \(z\) 造 \(x\)”这条生产线，挑剔的问题马上出现：怎样才算造得好？只把 16 张最好看的图贴出来，够不够？

---

## 3. 一张好看的图远远不够：六个评价轴

有了“用 \(z\) 造 \(x\)”这条生产线，怎样才算造得好？假设你刚训完一个人脸生成器，贴出 16 张最好看的图。一个不买账的审稿人不会停在“像不像照片”这一问上。原书的六条 desirable properties，其实就是他会连着问的六件事；它们彼此拆台，所以不存在单一冠军。

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

若模型定义了 \(p_\theta(x)\)，最老实的问题是：没见过的真实数据，模型还认不认？

\[
\log p_\theta(D_{\mathrm{test}})
=
\sum_i \log p_\theta(x_i).
\]

第 8 章已经讲过：训练集上的成绩可以是记忆。Likelihood 同样如此。模型可以在每个训练点上插一根极细极高的针，点与点之间几乎不给 probability。Test likelihood 不看模型造出来的图，它问真实的未见过的点还认不认。归一化会同时施压 coverage：只给一个角落很高的密度，别处必须更低，一部分 test 点就会很难看。抽出来的图好不好看，这把尺子根本没检查。

高维 standard Gaussian 把「高 density \(\neq\) 典型样本」钉死。原点 \(x=0\) 的 point density 最高，但 \(\mathbb E\|X\|^2=d\)，典型样本贴在半径约 \(\sqrt{d}\) 的薄壳上。维度越高，壳越薄，原点越不典型。图像里背景统计和局部纹理会在大量维上累积：给对了 background statistics，test likelihood 可以很好，人眼仍觉得糊。反过来，GAN 可以很锐，却没有可比较的 \(\log p(x)\)。

GAN 整把尺子拿不起来。VAE / Diffusion 常常只能报 bound。连续 density 的数字还依赖 preprocessing 和 dequantization，跨论文必须先对齐测量约定。Likelihood 够不着的两件事——没有 \(p(x)\) 的模型、以及人看着好不好——逼出下一把尺子。

---

## 6. Likelihood 够不着感知：Inception Score

Inception Score（IS）专为图像设计，而且最自然的适用场景是 ImageNet 这类有 1000 类物体的生成。它的思想是：既然人眼评价贵、不稳定，就借用一个已经训好的 image classifier，当廉价的感知探针。

探针只看生成图，不问真实图。它希望同时满足两件事。

**单张图要明确。** Classifier 对生成图 \(x\) 的类别分布 \(p(y\mid x)\) 应该尖：这张图看起来就是一只明确的狗，而不是“可能是狗、也可能是沙发”。

**整组图要多样。** 把所有生成图的类别分布平均起来：

\[
p(y)=\frac1N\sum_{i=1}^N p(y\mid x_i)
\]

应该比较平：整个 generated set 不能全是狗。

IS 用这两条分布的 KL divergence 来同时奖励它们：

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

为什么这个式子恰好卡在“单张明确、全体多样”上？把 KL 展开：

\[
D_{KL}\bigl(p(y\mid x)\,\|\,p(y)\bigr)
=
\sum_y p(y\mid x)\log\frac{p(y\mid x)}{p(y)}.
\]

若单张图非常明确，\(p(y\mid x)\) 近似是类别 \(c\) 上的尖峰，则

\[
D_{KL}
\approx
\log\frac{1}{p(y=c)}.
\]

于是：

- 若全体都在狂造同一类，\(p(y=c)\approx 1\)，KL \(\approx 0\)，IS 低——多样失败；
- 若单张图本身就含糊，\(p(y\mid x)\) 接近平均后的 \(p(y)\)，KL 同样接近 0——明确失败；
- 若每张都自信，且各类被均匀用到，\(p(y=c)\approx 1/K\)，KL 大约是 \(\log K\)，IS 高。

这把尺子的盲区几乎都来自“它从不看真实图像”：

- 强依赖所用 classifier；换一个网络，数字可以跳；
- 最适合与 classifier 的 label space 相符的数据，拿去评人脸、医学图、分子图都会文不对题；
- 不直接比较 generated data 与 real data，模型可以整体偏移真实分布，只要类别探针仍被骗；
- **每个 class 只生成一个模板，也可能拿到高分**：类间多样已经满足，类内多样完全不奖励。

最后一条说明：IS 不是 coverage 的好尺子。它最多逼你覆盖 **class 这个粗糙的 mode 定义**，而且每个 class 一个样板就够了。要跟真实数据比“两朵云像不像”，需要下一把尺子。

---

## 7. IS 还不看真实数据：FID 去比较两朵云

Fréchet Inception Distance（FID）承认一件 IS 不承认的事：生成质量是 **generated set 相对 real set** 的性质，不是 generated set 自己对自己打分。

但直接在 raw pixels 里比较两朵云会失败。像素空间的距离和人类感知对不上：把一只猫平移一个 pixel，L2 可以很大，人却觉得还是同一只猫。所以 FID 先把 real / generated images 送进预训练 Inception network，取出深层 features——那些更靠近“这是什么物体”的激活——再在 feature space 里比较。

高维 feature 的完整分布仍然很难估。FID 退一步，只把两朵云近似成 Gaussian：

\[
\mathcal N(\mu_r,\Sigma_r),
\qquad
\mathcal N(\mu_g,\Sigma_g).
\]

然后算这两个 Gaussian 之间的 Fréchet distance（二维以上常被说成 2-Wasserstein：把一朵 Gaussian 云搬成另一朵要花多少功夫）：

\[
\boxed{
\|\mu_r-\mu_g\|^2
+
\operatorname{Tr}\left(
\Sigma_r+\Sigma_g
-2\bigl(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\bigr)^{1/2}
\right).
}
\]

越小越好。两项各管一件事：

```text
‖μr − μg‖²          两朵云的中心是否重合？
                     （平均语义是否一样）

Trace 项             两朵云的形状 / 张开程度是否一样？
                     （多样性、协方差结构是否一样）
```

所以 FID 同时对“生成图整体偏了”和“生成图多样性不够”敏感。这比 IS 更接近我们真正想比的 \(p_\theta\) vs \(p_{data}\)，只是比较发生在 classifier feature 里，并且只用了均值和协方差。

局限同样来自这些近似：

- 真实 feature 云往往不是 Gaussian，只匹配均值协方差会漏掉多峰、死角；
- 依赖 feature extractor；网络丢掉的信息——细纹理、精确空间布局——不进 metric；
- finite sample 下 FID 有估计偏差，生成 2048 张和 5 万张，数字不可直接比；
- **一个标量无法告诉你失败来自 fidelity 还是 coverage**。云的中心偏了、云瘪了，都会把 FID 变差。

IS 的缺口是“不看真实数据”。FID 补上了，却把上一节最在意的那两维又揉回一个数。于是需要一把故意把两维拆开的尺子。

---

## 8. FID 分不清两种失败：Manifold Precision / Recall

FID 已经把 generated set 和 real set 放在一起比了，却仍把第 4 节的两问揉成一个距离。Manifold precision / recall 的动机就是把它们拆开，不再求和成一个分数。

把真实样本所在的区域叫 data manifold，把生成样本所在的区域叫 model manifold。则：

\[
\text{Precision}
=
\frac{\text{落在 data manifold 内的 generated samples}}
{\text{全部 generated samples}}.
\]

\[
\text{Recall}
=
\frac{\text{落在 model manifold 内的 real samples}}
{\text{全部 real samples}}.
\]

Precision 高：生成出来的东西大多真实。  
Recall 高：真实数据的模式大多能被模型造出来。

一个数字例子。100 个生成样本里有 80 个落在 real manifold 上，precision \(=0.8\)。100 个真实样本里只有 60 个被 generated manifold 覆盖，recall \(=0.6\)。诊断立刻比 FID 清楚：

> 样本多数较真实，但 coverage 明显不足。

真正的麻烦是：我们没有解析的 manifold。实践中在 feature space 里用 k-nearest-neighbor hyperspheres 去近似——每个样本周围画一个球，半径等于到第 \(k\) 个邻居的距离，所有球的并当成“这块区域属于该类数据”：

```text
Real examples:     o  o    o
                   ╲  |    ╱     ← kNN balls
                    ╲ |  ╱
Generated point:      ×          inside → counts as realistic
```

这仍然依赖 representation 和邻域定义。\(k\) 太小，manifold 碎成孤岛，precision 会虚低；\(k\) 太大，球并在一起把空隙也算进去，假图会被算成真实。它把 fidelity / coverage 拆开了，但“拆开”本身建立在一个近似几何上。

---

## 9. 所以没有一个 Metric 够用

把四把尺子按“它补上了谁的缺口”排成一条链：

```text
需要给概率模型打分
        → Test likelihood
            缺口：GAN 没有 p(x)；高 density ≠ 好看
        → Inception Score（用分类器当感知探针）
            缺口：只看生成集，不看真实数据；不奖类内多样
        → FID（两朵 feature 云的距离）
            缺口：真实性和覆盖率揉成一个数
        → Manifold precision / recall
            缺口：manifold 是估的，仍依赖 feature
```

| Metric | 主要照亮什么 | 主要盲区 |
|--------|--------------|----------|
| Test likelihood | 概率模型对 unseen data 的 density，并经由归一化施压 coverage | 感知质量未必一致；部分模型不可精确算 |
| IS | 单图明确 + 类别多样 | 不与 real set 直接比较，忽略 class 内多样性 |
| FID | real / generated feature 分布有多近 | 不区分 fidelity / coverage，依赖 extractor 与 Gaussian 近似 |
| Precision / Recall | 把真实性与覆盖率拆开 | manifold 估计近似，依赖 representation 和 \(k\) |
| Human evaluation | 感知质量与偏好 | 昂贵、主观、难复现 |

推荐的不是另找一把“真正正确”的尺子，而是让几把尺子互相揭短：

```text
Sample grid                          → 看 fidelity，几乎不看 coverage
+ nearest-neighbor / memorization    → 是不是在复读训练集
+ FID 或 domain feature distance     → 两朵云整体像不像
+ precision / recall                 → 失败来自假图还是漏模式
+ likelihood 或 bound（若可用）      → 概率模型本身
+ task-specific evaluation           → 下游真正在乎的事
```

评价生成模型，和评价分类器不同。分类器往往有一个对的 label；生成器面对的是一整个未知分布 \(p_{data}\)，任何标量都只是这个分布的一个投影。

有了这组轴和这组尺子，就可以看后面四章各自押哪一边。

---

## 10. 四类生成模型：同一组轴上的不同交换

原书后面四章都从某种 latent 出发，用深度网络把它映到数据空间。差别不在“是不是生成模型”，而在它们愿意丢掉哪一条性质，去换哪一条性质。

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

它坚持 \(p_\theta(x)\) 必须能精确写出来。办法是从简单密度出发，做一串可逆变换，用 change of variables 把密度跟着搬过来。

- 换来的：exact likelihood、exact latent mapping；
- 丢掉的：architecture 必须 invertible，表达能力被捆住，sample 往往没 GAN / Diffusion 那么锐。

### VAE

```text
x → approximate posterior q(z|x)
z → probabilistic decoder p(x|z)
```

它想要 latent variable model 的完整概率故事，但 \(p(x)=\int p(x\mid z)p(z)\,dz\) 通常积不出来。于是引入 encoder 去近似 posterior，优化 likelihood 的 lower bound（ELBO）。这是第 17 章的母问题。

- 换来的：概率故事清楚、latent space 通常较规则、训练稳定；
- 丢掉的：exact likelihood；样本常偏平滑（decoder 把不确定性吃进模糊里）。

### Diffusion

```text
x → gradually add noise → ε          （forward，固定不学）
gradually denoise ε → x              （reverse，才是模型）
```

它把“一步从噪声跳到图像”拆成许多小去噪步骤。第 18 章会说明：forward 加噪是设计好的，learned 的是 reverse。

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

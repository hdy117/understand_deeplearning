# 第 12 章：Transformer（按内容动态路由信息）

> 书：《Understanding Deep Learning》Ch.12  
> 前置：第 10 章把连接写进相对位置；第 11 章用 residual 与 normalization 让复杂 block 可以反复堆叠。  
> 本章：序列里的每个位置，怎样根据**当前内容**决定该读取谁、读多少？  
> 阅读方式：先感到“接线必须随内容改”，再给三件事起名叫 query / key / value；先追一条 sample 的 data flow 和 shape，再看矩阵公式。

---

## 0. CNN 已经把“谁连谁”写死了

第 10 章的卷积网络把一件事直接写进 architecture：附近的东西关系更大，同一套局部规则可以在所有位置复用。每个 output 只看一个 window，window 由相对位置预先规定。Filter 会滑过猫，也会滑过车，但“这个位置有资格跟哪些位置说话”，在网络建好时就已经钉死，并不取决于窗口里此刻写了什么。

第 11 章又补了一块：residual 让新层默认接近 identity，normalization 稳住尺度，于是复杂 block 可以叠很多层而不立刻把已有表示毁掉。后面凡是能把同一套计算重复 12 次、96 次的故事，都先欠这两笔。

语言把这两笔还不够用的那一面推到眼前。考虑这句话：

```text
The animal did not cross the street because it was tired.
```

理解 `it` 时，相关位置可能离得很远，而且连谁由内容决定。这里 `it` 更该读取 `animal`，不是固定读取“前面第三个词”。把 `tired` 换成 `too wide`，同一格代词又该去读 `street`。CNN 的 window 做不到这种当场改接线：距离不一定近，关系也不由相对偏移唯一确定。

RNN 能处理变长序列，理论上也可以把信息一步步传到很远。可远距离信号必须穿过中间所有时间步，中间容易丢；更关键的是，它仍然没有一张按当前内容算出来的全图接线——每个时刻主要看见“上一步的 hidden”和“当前输入”，而不是对所有候选做一次匹配。Bahdanau 等人先在翻译模型里让输出词去 attend 输入词；Vaswani 等人再丢掉 recurrence，把交互全部改成 attention。本章走的就是这条路。

于是母问题只有一句：

> **怎样让序列中每个位置，在这一次 forward 里根据当前内容，动态决定该读取谁、读多少？**

这叫 attention。更直白的说法是 content-dependent routing：连接强度不是参数表里的固定条目，而是输入的函数。

```text
CNN:     connectivity written by relative position
RNN:     information walks step by step along time
here:    each token matches candidates by content, now
```

后面每一节只补这条链上的一个缺口：先把文字变成可以路由的 vectors；让每个位置提出需求、宣告自己、准备要发送的内容；把匹配变成一张权重表再混合 payload；解释为什么要缩放、为什么要多套 matching space、为什么还要 MLP 和 residual 才能叠层；最后在 softmax 前加上 mask，同一套路由就会变成 BERT、GPT 和翻译模型。合上书时，请带着这个问题去第 13 章：如果“谁连谁”本身就是数据，而不是每次由内容现算出来的呢？

---

## 1. 路由之前，先得有一排带位置的 vectors

母问题说的是“位置之间交换信息”，可模型并不能直接吞字符串。要路由，先得有被路由的对象。工程上这条路极短：

```text
Raw text
   → tokenizer          → subword tokens
   → vocabulary lookup  → integer token IDs
   → embedding table    → content vectors
   + position           → X^{(0)}
```

完整单词当 vocabulary，会在新词、形态变化和参数量上同时撞墙；一个字符一个 token，序列又会太长，后续每个位置都要和其他所有位置匹配。实践里用 subword：常见词单独占一项，罕见词拆成可组合的片段。Tokenizer 不是与模型无关的预处理。换一套 tokenizer，token IDs、序列长度和 embedding table 一起变，input protocol 就变了。

Embedding 是 learned lookup。每个 token ID 从一张表里取出一行 \(D\)-dimensional vector：

```text
token ID 42 → embedding table row 42 → D-dimensional vector
```

它不是词典释义。同一套 training objective 会同时推 embedding 和后面所有 projections；表里那一行只是“这个 ID 的可学习坐标”。工程里常用

\[
X\in\mathbb R^{B\times N\times D},
\]

其中 \(B\) 是 batch size，\(N\) 是 sequence length，\(D\) 是 embedding dimension。原书常把 token 放在列中，写作 \(D\times N\)；代码常写 \(N\times D\)。两者互为 transpose。第一遍选定一种 convention 就不要在同一条推导里来回切——本章其余公式一律按行是 token、也就是 \(N\times D\)。

只把 content embedding 摆好，还缺一件对语言致命的事。Self-attention 对输入顺序是 permutation equivariant 的：交换两个 tokens，输出会跟着交换，但模型无法分辨交换本身改变了什么。`dog bites man` 和 `man bites dog` 若只有内容向量、没有位置，对它来说是同一组可重排的向量。所以输入几乎总是

\[
X^{(0)}=X_{\mathrm{content}}+X_{\mathrm{position}}.
\]

Content 回答“我是谁”，position 回答“我在哪里”。Position 可以是 learned absolute embedding，可以是固定 sinusoidal encoding，也可以用相对位置或旋转等方式进到 attention 的 scores 里。第一遍不必收藏所有位置方案；只需记住：没有它，路由只知道一组内容，不知道句子的顺序语义。

现在每个位置手里都有一个向量。下一问立刻出现：这些向量凭什么决定读谁？

---

## 2. 每个位置必须当场做三件事

上一节末尾，每个位置手里已经有一个带位置的向量。CNN 会回答“读相对偏移落在 window 里的那些人”；把接线改成“所有人连所有人、权重固定”，仍然不够——`bank` 在“河岸”和“银行”两句里，该读的邻居完全不同。权重必须是**这一次输入**的函数，否则手里那排向量还是不知道读谁。

于是每个 token 至少要同时做三件事。第一件：提出自己现在缺什么。`it` 站在那句话里，需要的不是随便一个名词，而更像“可以被这个代词指代的、单数的实体”。第二件：让别人找得到自己。`animal` 必须把自己“实体、单数、名词”这类可匹配的特征亮出来，否则 `it` 的需求会落空。第三件：一旦被选中，真正送出去的不能只是那张用于匹配的标签，而应是这个位置要贡献的语义内容。

```text
each token
   ├── ask:      what information do I need
   ├── announce: how should others match me
   └── send:     what payload do I emit if selected

after matching
   → mix payloads by weight
   → contextual representation
```

把每个 token 看成一名资料员，三件事就更好记：query 是“我正在找什么”，key 是“别人该怎样对上我的号”，value 是“找到我以后我把哪份材料递出去”。名字来自信息检索——用一条 query 去对一批 keys 打分，再把对应的 values 取回来——但这里没有人工数据库字段。同一组 token vectors 经过三组不同的 learned projections，扮演三种角色：

\[
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
\]

常见 shape 是 \(Q,K\in\mathbb R^{B\times N\times d_k}\)，\(V\in\mathbb R^{B\times N\times d_v}\)。同一个 `it` 既发出需求，也宣告自己，也准备 payload；`animal` 同样三件事都做。Q/K/V 不是三堆不同的 tokens，而是同一排 tokens 的三种读法。

资料员只负责把角色分开，不要读成系统里真有三张表。下一节把这三件事变成一次可微的矩阵计算。

---

## 3. 匹配产生权重，权重去混合 values

三件事已经有名字，也有三组矩阵。对第 \(i\) 个位置，query \(q_i\) 要和每一个 key \(k_j\) 比一比像不像。标准做法不是再训练一张 \(N\times N\) 的查表，而是做 dot product：方向接近、幅度也大，分数就高。全体 query 对全体 key 一次算完：

\[
S=\frac{QK^{\top}}{\sqrt{d_k}}.
\]

对一个 batch 里的一条序列，中间那步是

\[
[N,d_k]\times[d_k,N]=[N,N].
\]

所以 \(S_{ij}\) 的意思极具体：第 \(i\) 个 query 与第 \(j\) 个 key 的匹配分数。这张表就是“谁想读谁”的全部候选接线。

分数还不是路由比例。Softmax 沿 key 这一维做，把第 \(i\) 行变成一组非负且和为 1 的权重：

\[
A=\operatorname{softmax}(S),
\qquad
\sum_{j=1}^{N}A_{ij}=1.
\]

然后——也只有在这一步——values 才进场：

\[
Z=AV.
\]

Shape 是 \([B,N,N]\times[B,N,d_v]\to[B,N,d_v]\)。第 \(i\) 个输出不是“复制那个最像的 token”，而是

\[
\begin{aligned}
s_{ij}&=\frac{q_i k_j^{\top}}{\sqrt{d_k}},\\
a_{ij}&=\operatorname{softmax}_{j}(s_{ij}),\\
z_i&=\sum_{j=1}^{N}a_{ij}v_j.
\end{aligned}
\]

整条 pipeline 可以画成：

```text
X
├── W_Q → Q ─┐
├── W_K → K ─┼→ QKᵀ / √d_k → mask → softmax → A
└── W_V → V ────────────────────────────────┐
                                             ▼
                                           Z = AV
```

最容易走错的一步是：Q 和 K 决定读谁、读多少；V 决定读到什么。没有再做一次 \(q_i\cdot v_j\) 或 \(k_j\cdot v_j\)。\(v_j\) 算好以后可以被所有 query 复用，区别只在各行的 \(a_{ij}\) 不同。

行 convention 下，角色和 shape 对得上：

| 对象 | shape | 这一步在干什么 |
|---|---:|---|
| \(q_i\) | \([d_k]\) | 位置 \(i\) 提出的需求 |
| \(k_j\) | \([d_k]\) | 位置 \(j\) 用来被匹配的标签 |
| \(a_{ij}\) | scalar | \(i\) 读取 \(j\) 的比例 |
| \(v_j\) | \([d_v]\) | 位置 \(j\) 真正携带的 payload |
| \(z_i\) | \([d_v]\) | 位置 \(i\) 读完全序列后的新表示 |

先把整条公式压成两个 token、\(D=2\)、先假装 \(W_Q=W_K=W_V=I\)（于是 \(Q=K=V=X\)），只看算术。两行是 `it` 和 `animal`：

\[
X=\begin{bmatrix}1&0\\0&1\end{bmatrix}
\quad
\begin{aligned}
&\text{row 0: }q_0=k_0=v_0=[1,0]\quad(\texttt{it})\\
&\text{row 1: }q_1=k_1=v_1=[0,1]\quad(\texttt{animal})
\end{aligned}
\]

\(d_k=2\)，\(\sqrt{d_k}=\sqrt{2}\approx 1.414\)。Score 矩阵：

\[
QK^{\top}
=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\qquad
S=\frac{QK^{\top}}{\sqrt{2}}
\approx
\begin{bmatrix}0.707&0\\0&0.707\end{bmatrix}.
\]

`it` 那一行 \(\operatorname{softmax}([0.707,\ 0])\approx[0.668,\ 0.332]\)。输出

\[
z_{\texttt{it}}
\approx
0.668[1,0]+0.332[0,1]
=[0.668,\ 0.332].
\]

还没学投影时，自己和自己最像，所以 `it` 仍以自己的 payload 为主，只混进约三分之一的 `animal`。训练要做的，就是让 \(W_Q,W_K\) 把 `it` 的 query 拧到更靠近 `animal` 的 key。

把同一套数改成「已经学歪了的匹配」。设投影后

\[
q_{\texttt{it}}=[1,1],
\quad
k_{\texttt{animal}}=[1,1],
\quad
k_{\texttt{it}}=[1,0],
\quad
v_{\texttt{it}}=[1,0],
\quad
v_{\texttt{animal}}=[0,2].
\]

未缩放点积：\(q\cdot k_{\texttt{it}}=1\)，\(q\cdot k_{\texttt{animal}}=2\)。除以 \(\sqrt{2}\) 后 \(s\approx[0.707,\ 1.414]\)，

\[
\operatorname{softmax}\approx[0.332,\ 0.668].
\]

于是

\[
z_{\texttt{it}}
\approx
0.332[1,0]+0.668[0,2]
=[0.332,\ 1.336].
\]

旋钮现在主要拧在 `animal` 的 value 上——这就是 content-dependent routing 的最小数字版：同样两个位置，分数表一变，读到的 payload 从「自己」变成「那个实体」。

把句子换成 `it was tired` 里的第三个词还是 `too wide`，只需再给 `street` 一个不同的 key。若 `tired` 让 \(q_{\texttt{it}}\) 靠近 \(k_{\texttt{animal}}\)，权重就像上一行的 \(0.668\)；若 `too wide` 把 query 拧向 \(k_{\texttt{street}}\)，同一格 `it` 会改读另一行 value。CNN 做不到这件事：相对偏移写死以后，权重表不会因第三个词翻面。

再用三个 tokens、一个 query 走一遍加权和。假设缩放和 mask 之后分数是 \(s_i=[2,1,0]\)，则

\[
a_i=\operatorname{softmax}([2,1,0])
\approx[0.665241,\ 0.244728,\ 0.090031].
\]

三个 value 已经各自算好：\(v_1=[1,0]\)，\(v_2=[0,2]\)，\(v_3=[3,1]\)。加权求和：

\[
\begin{aligned}
z_i
&=0.665241[1,0]+0.244728[0,2]+0.090031[3,1]\\
&=[0.935333,\ 0.579488].
\end{aligned}
\]

逐项看更清楚：最大的旋钮拧在 \(v_1\) 上，所以第一维主要来自它；第二维几乎全部来自 \(v_2\)；\(v_3\) 的权重最小，但仍把 \(0.27\) 和 \(0.09\) 加了进去。Attention 通常不是 argmax 挑出一个 token，而是把所有 values 按旋钮大小混合。

```text
Q/K produce the knobs
V is the content being mixed
output = weighted sum, not a copy of one token
```

回到那句 `it`。训练好的模型常常会把较大的 \(a_{ij}\) 分给 `animal`。那只说明这一次路由里，这个 query 觉得那个 key 更匹配；它不是“`it` 指代 `animal`”的因果证明，更不是完整解释。权重描述的是一次 soft mix，不是语言学鉴定书。

还有一个常被顺手说滑的词：cosine。只有先把 \(q\) 和 \(k\) 做成单位向量，\(q^{\top}k\) 才等于 cosine similarity。原书第 12 章的 dot-product self-attention 与 scaled dot-product self-attention 都不要求这一步。除以 \(\sqrt{d_k}\) 是在管 score 的尺度，不是把 inner product 改造成夹角。

公式里那个 \(\sqrt{d_k}\) 现在还没解释。分数表已经出现了，而且 \(d_k\) 一大，这张表上的数字会先把 softmax 顶死——为什么还要除一下？

---

## 4. 维度升高，分数会把 softmax 顶饱和

上一节故意把 \(\sqrt{d_k}\) 留着。若 query / key 各维的尺度差不多，dot product 是 \(d_k\) 项相加，\(d_k\) 变大时 \(|q^{\top}k|\) 往往跟着变大。Softmax 对大间隔极度敏感：一个位置接近 1，其余接近 0，输入再怎么微调也很难改输出。Gradient 变弱，训练变难。

除以 \(\sqrt{d_k}\) 就是把 score 拉回比较温和的范围，使 softmax 不会仅仅因为维度增加而过早变成 hard selection。它不管“权重和为 1”——那是 softmax 自己的归一化。两者分工不同：缩放管 magnitude，softmax 管变成一套 routing weights。

带上可选 mask \(M\)（谁有资格被读，后面单独成节），标准公式是

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^{\top}}{\sqrt{d_k}}+M
\right)V.
\]

到这里，一次路由已经能跑通。可 \(A\) 明明是一张 \(N\times N\) 的表。既然最后只是 \(Z=AV\)，为什么还要绕路用 \(QK^{\top}\) 现算这张表？直接学一张 \(A\) 不行吗？

---

## 5. 直接学一张 dense \(A\)，省不掉 \(N^{2}\)

上一节末尾那个想法要拆开，不能混成一句“不要 QK 就能变便宜”。

若每层固定一张随机矩阵 \(A_l\in\mathbb R^{N\times N}\)，每次 forward 都用同一组权重去混合 values，那只是随机 mixing，或者顶多是与内容无关的位置混合。它不知道这句里的 `it` 该读 `animal`。Decoder 里还必须强制 causal mask，否则固定矩阵会让当前位置读到未来，训练时偷看答案。

把 \(A_l\) 当成 parameter 训练，它可以学到一张**与输入无关**的 routing pattern。固定长度 \(N\) 上这是可行的，但代价立刻排成三排。Dense \(AV\) 仍是 \([N,N][N,d_v]\to[N,d_v]\)，复杂度仍是 \(O(N^{2}d_v)\)。参数量变成 \(O(N^{2})\)，和序列长度绑死，变长输入不好处理。Routing 不随当前 tokens 改变：`bank` 在所有句子里都被同一张表接线，这更像 learned global mixing，不是 self-attention 那种 content-dependent routing。

标准 attention 的两个主要二次项是：

\[
QK^{\top}:\ [N,d_k][d_k,N]\to[N,N],
\qquad
AV:\ [N,N][N,d_v]\to[N,d_v].
\]

| 部分 | 主要计算 | 中间 memory |
|------|----------|-------------|
| Q/K/V projections | \(O(ND^{2})\) | \(O(ND)\) |
| score \(QK^{\top}\) | \(O(N^{2}d_k)\) | \(O(N^{2})\) |
| weighted sum \(AV\) | \(O(N^{2}d_v)\) | \(O(Nd_v)\) |

绕过显式的 \(QK^{\top}\)，只要还做 dense \(AV\)，二次项就还在。FlashAttention 用 tiling 避免把完整 \(A\) 写进显存，能把 memory pressure 降下来，但 exact dense attention 的算术复杂度仍是 quadratic——它改善的是硬件上怎么算，不是把理论 compute 变成 linear。

真要 subquadratic，必须改变 \(A\) 的结构，而不是换一个初值。每个 query 若只读 \(k\ll N\) 个 keys（local window、block sparse、graph edges、少数 global tokens），复杂度大约 \(O(Nkd)\)，单层也就不再保证任意两 token 直接通信。若 \(A\approx UR^{\top}\) 且 \(r\ll N\)，则可先算 \(R^{\top}V\) 再左乘 \(U\)，变成约 \(O(Nrd)\)，前提是 dense attention 确实接近低秩。Kernel / linear attention 则把 \(\exp(q_i^{\top}k_j)\) 近似成 \(\phi(q_i)^{\top}\phi(k_j)\)，先聚合 keys/values 而不 materialize 每个 pair；这是在近似 kernel algebra，和“把 attention 矩阵随机初始化”不是一回事。

| 方法 | 随输入改接线？ | 主要复杂度 | 主要代价 |
|------|----------------|------------|----------|
| Standard QK attention | 是 | \(O(N^{2}d)\) | quadratic compute / memory |
| Fixed random \(A\) | 否 | \(O(N^{2}d)\) | 随机 routing |
| Trainable static dense \(A\) | 否 | \(O(N^{2}d)\) | \(O(N^{2})\) 参数，固定长度 |
| Local / sparse | 是，但受 mask 限制 | \(O(Nkd)\) | 远程连通性变差 |
| Low-rank | 可以 | \(O(Nrd)\) | 低秩假设可能伤表达 |
| Kernel / linear | 是，近似 | \(O(Nrd)\) | 近似误差与数值稳定 |
| FlashAttention | 是 | compute 仍约 \(O(N^{2}d)\) | 主要改善 memory 与利用率 |

结论只有一句：直接学 dense \(A\) 可以不写 \(QK^{\top}\)，但不能取消 dense \(AV\)，因此取消不了 \(O(N^{2})\)。本章其余部分仍用标准 scaled dot-product——它贵在那张按内容现算的表，而不是贵在 embedding lookup。

一套 Q/K 投影给出的，是**一个** matching space 里的相似度。指代、语法依赖、局部搭配、长距主题，未必挤得进同一把尺子。下一问就是：能不能并行学几套路由规则？

---

## 6. 一套 matching space 不够，所以切开成若干 heads

\(O(N^{2})\) 买到的是**一张**按内容现算的表，而这张表来自**一套** \(W_Q,W_K,W_V\)。`it` 找先行词、相邻词做局部搭配、两个动词抢同一个主语，需要的匹配关系可以很不一样。把 \(D\) 维拆成 \(H\) 个较低维的空间，让每套 projections 自己学一种打分方式，就是 multi-head attention：

```text
X
├── head 1: one routing space
├── head 2: another routing space
├── ...
└── head H
      ↓ concatenate
      ↓ output projection W_O
```

若总 model dimension 为 \(D\)，常取 \(d_h=D/H\)，使拼接后仍是 \(D\)。典型 shape flow：

```text
[B, N, D]
  → Q/K/V projections
  → [B, N, H, d_h]
  → transpose heads forward
  → [B, H, N, d_h]
  → scores [B, H, N, N]
  → head outputs [B, H, N, d_h]
  → concatenate [B, N, D]
  → W_O → [B, N, D]
```

每个 head 有自己的 learned projections，因而可以形成不同的 routing pattern；最后用 \(W_O\) 把 \(H\) 路结果混回同一套 representation。这不是把同一张 attention 复制 \(H\) 份。经验上 multi-head 几乎是让 self-attention 能用的必要条件；训练后有些 head 可以被剪掉而不致命，说明它们的价值也包括“多给几次初始化机会”，但第一遍先抓住那句：多套空间，不是多套复印。

到这里，tokens 已经能按内容交换信息。可是每个 token 内部的 features 还没有被好好加工，这样的层也还不能直接叠几十次。第 11 章留下的 residual 与 normalization，现在要进 block 里。

---

## 7. Token 之间混合完，还要在每个 token 内部改写

一次 attention 做的是 **token mixing**：位置 \(i\) 从序列里把信息读过来。若到此为止，每个位置只是别人 values 的线性组合（再加输出投影）。还缺对每个 token 自己那条 \(D\) 维特征的非线性加工——**feature / channel mixing**。Transformer block 把这两件事叠在 residual 里：

```text
X
 ├─ Attention: mix across tokens
 └─ MLP:       mix inside each token
residual keeps the old representation
LayerNorm keeps the scale usable
```

Attention 子层（Post-LN 写法，与原书一致）是

\[
H=X+\operatorname{Attention}(X),
\]

然后再 LayerNorm。Residual 的意思和第 11 章相同：主路保留旧表示，attention 学习 correction；若这一层不知该改什么，接近 0 的 residual 不会把已有表示砸烂。

MLP 对每个位置独立应用同一套全连接：

\[
\operatorname{MLP}(h)=W_2\,a(W_1h+b_1)+b_2.
\]

它不在 token 之间传话。跨位置交流已经由 attention 做完；MLP 只改写当前位置的 features，通常先把维度扩上去再压回来。工程里激活常用 GELU。原书把每个 token 写成一列，对每一列单独跑同一个全连接网络；换成我们的 \(N\times D\) convention，就是对每一行做同一件事。

LayerNorm 对每个 token 的 embedding 维做归一化，统计量来自这一条向量的 \(D\) 个分量，不依赖 batch 里其他样本，也不依赖序列里其他位置。变长序列和 autoregressive inference 都吃得下；这是它在这里比 BatchNorm 更常见的原因。

Pre-LN 与 Post-LN 只是 normalization 和子层的相对顺序：

```text
Post-LN: X → Attention → Add → LayerNorm
Pre-LN:  X → LayerNorm → Attention → Add
```

原书重点画的是 Post-LN。更深的网络后来更常见 Pre-LN，因为 identity 路径更干净、优化更稳。第一遍只要知道顺序会影响 optimization，不要把两种公式写进同一条推导。

一个真实模型是若干这样的 block 串起来。Encoder / decoder 的差别还不在 block 内部有没有 MLP，而在 attention 被允许看见谁。那张 \(N\times N\) 的表，默认是全连接；必须先禁止某些格子，否则 padding 会污染表示，生成模型会在训练时偷看答案。

---

## 8. 谁有资格被读：mask 把同一套路由切成不同模型

上一节末尾那张 \(N\times N\) 的表，默认人人可读人人。有些格子根本不该存在：padding 没有语义，生成时未来 token 是考题答案。禁止发生在 softmax 之前——不允许的位置把 score 设成极大的负数，softmax 之后对应 weight 接近 0，这条边等于断掉。Mask 定义的是信息通路，不只是“把 padding 删掉”这么窄。

Padding mask 最先出现，因为 batch 里句子长短不一，短句要补占位符。占位符没有语义，其他 tokens 不应读取它。Causal mask 则禁止看未来：生成第 \(t\) 个 token 时，答案还不该存在于输入里。

```text
Token 1 can see: 1
Token 2 can see: 1 2
Token 3 can see: 1 2 3
...

✓ × × ×
✓ ✓ × ×
✓ ✓ ✓ ×
✓ ✓ ✓ ✓
```

还可以只保留局部窗口或 graph 上的边。那是在用 mask 把“理论上的全连接”收成更稀的拓扑——第 13 章会把这种限制当成数据本身。这里先抓住：visibility pattern 不同，模型就不同。

**Encoder** 让每个 token 读取序列中所有有效位置（仍要挡住 padding）：

```text
all-to-all bidirectional self-attention
```

适合把一句话编成可复用的 contextual representations：分类、token labeling、抽取。BERT 就是这条路。输入两侧都可以看，于是 pre-training 可以做 Masked Language Modeling——随机挡住一些 token，用左右上下文把它找回来：

```text
original: The animal was tired
input:    The [MASK] was tired
target:   animal
```

填空本身不是目的。这个目标逼 encoder 学一套可迁移的双向表示；事后再加一个小 head，就可以做句分类、命名实体、span prediction。BERT 还用过“两句是否相邻”这种次要任务，收益有限。重点始终是：双向可见，才能用两边的证据恢复被挡住的词。

**Decoder** 用 causal self-attention：每个位置只能读自己和过去。它要的是 next-token 分布

\[
p(x_1,\ldots,x_N)
=
\prod_{t=1}^{N}
p(x_t\mid x_{<t}).
\]

GPT 类模型走这条路。训练时整句都在，但 causal mask 让位置 \(t\) 看不到 \(x_{t+1},x_{t+2},\ldots\)，于是所有位置可以并行算各自的 next-token loss：

```text
Input : x1 x2 x3 x4
Target: x2 x3 x4 x5
```

Inference 时未来 token 还不存在，必须一个一个接：

```text
context                  → predict token 1
context + token 1        → predict token 2
context + token 1 + 2    → predict token 3
```

Training 在 sequence 维上可并行；autoregressive inference 仍然 sequential。过去位置的 keys / values 可以写入 KV cache，避免每步把历史再算一遍，但新 query 仍要和历史里所有 keys 匹配。Cache 减的是重复计算，不是 \(O(N^{2})\) 这笔账本身。Decoder 也不是“只能做文本”的架构名，它首先是一种因果可见性；图像、音频同样可以按这种 mask 生成。

**Encoder–decoder** 把两种可见性接到同一条任务上。Encoder 先双向理解 source；decoder 对自己已经写出的 target 做 causal self-attention，再用 **cross-attention** 去读 encoder：query 来自 decoder，key / value 来自 encoder。翻译是最干净的例子：

```text
Source  → Encoder → memory K,V ─────────┐
                                         ▼
Past target → Decoder self-attn → cross-attn → next-token
```

Self-attention 的 Q/K/V 来自同一条序列；cross-attention 允许“法语句子里的当前词”去匹配“英语源句里的某个词”，接线仍然由内容决定，只是 query 和 key 不再同源。Training 常用真实的 previous target tokens（teacher forcing），inference 用模型自己已经生成的词，错误会向后累积。

三套模型不是三种神秘骨架。它们是同一套 scaled dot-product，加上三种“谁可以读谁”的规定。角色、缩放、多头、残差、可见性都齐了；还缺一次把 \([B,N,D]\) 钉在具体张量上的 forward。

---

## 9. 用 shape 把刚才的故事钉死

上一节末尾还缺一次把 \([B,N,D]\) 钉在具体张量上的 forward。下面这段 mini Transformer 不调用 `nn.TransformerEncoder`，只把 embedding、Q/K/V、scaled dot-product、causal mask、residual、LayerNorm 和 position-wise FFN 摊开。随机初始化没有语义质量，它的任务是让 computation graph 可见。需要 PyTorch，复制即可跑。

```python
import math
import torch
from torch import nn


class MiniTransformerBlock(nn.Module):
    """One decoder-style Transformer block with a causal mask."""

    def __init__(self, d_model=8, n_heads=2, d_ff=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One projection creates Q, K, and V, then we split the last dimension.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x: [B, N, D] = [batch, sequence, embedding]
        B, N, D = x.shape

        # [B,N,3D] -> [3,B,H,N,Dh]
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v: [B, H, N, Dh]

        # Every query compares with every key: [B,H,N,Dh] @ [B,H,Dh,N]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # scores: [B, H, N, N]

        # Decoder-style causal mask: position i cannot read positions j > i.
        future = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(future, float("-inf"))
        attention = scores.softmax(dim=-1)
        # Each row of attention sums to 1: [B,H,N,N]

        context = attention @ v
        # context: [B,H,N,Dh] -> [B,N,D] after joining heads
        context = context.transpose(1, 2).contiguous().reshape(B, N, D)
        x = self.norm1(x + self.out_proj(context))       # token mixing

        # FFN is applied independently to every token, not across N.
        x = self.norm2(x + self.ffn(x))                  # feature mixing
        return x, attention


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=10, max_seq_len=4,
                 d_model=8, n_heads=2, d_ff=16):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.block = MiniTransformerBlock(d_model, n_heads, d_ff)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        # token_ids: [B,N], for example [2,4]
        B, N = token_ids.shape
        positions = torch.arange(N, device=token_ids.device)
        x = self.token_embedding(token_ids)              # [B,N,D]
        x = x + self.position_embedding(positions)[None, :, :]  # [B,N,D]
        hidden, attention = self.block(x)
        logits = self.lm_head(hidden)                    # [B,N,vocab_size]
        return logits, attention


torch.manual_seed(0)
model = MiniTransformer()
token_ids = torch.tensor([
    [1, 5, 2, 8],    # sample 1
    [4, 3, 7, 0],    # sample 2; 0 can be a padding ID in a real dataset
])

logits, attention = model(token_ids)
print("token_ids:", token_ids.shape)  # [2, 4]
print("logits:", logits.shape)        # [2, 4, 10]
print("attention:", attention.shape)  # [2, 2, 4, 4]
print("last head, last query:", attention[0, 0, -1])
print("row sum:", attention[0, 0, -1].sum())  # tensor(1.)
```

对应的形状追踪：

```text
token_ids                 [B,N]       = [2,4]
   ↓ nn.Embedding
token embeddings          [B,N,D]     = [2,4,8]
   + position embeddings  [B,N,D]
   ↓ Linear(d_model, 3D)
Q, K, V                   [B,H,N,Dh]  = [2,2,4,4]
   ↓ Q @ Kᵀ / √Dh
scores                    [B,H,N,N]   = [2,2,4,4]
   ↓ causal mask + softmax(dim=-1)
attention weights         [B,H,N,N]   = [2,2,4,4]
   ↓ attention @ V
context                   [B,H,N,Dh]  = [2,2,4,4]
   ↓ transpose + concatenate heads
context                   [B,N,D]     = [2,4,8]
   ↓ output projection + residual + LayerNorm
   ↓ position-wise FFN (8 → 16 → 8)
hidden                    [B,N,D]     = [2,4,8]
   ↓ language-model head
logits                    [B,N,V]     = [2,4,10]
```

五个维度各管一个问题。\(B\)：一次并行几条样本，不同样本之间不做 attention。\(N\)：一条样本有多少 token，pair 矩阵是 \(N\times N\)。\(D\)：每个 token 用多少 features。\(H\)：同时学几套 matching space。\(d_h\)：每个 head 分到多少维，且 \(D=H\cdot d_h\)。

`attention[0,0,-1]` 是第 1 条样本、第 1 个 head、最后一个 query 的整行权重，shape 为 \([4]\)。Causal mask 下最后一行四个位置都可见；第 2 个位置只能看见前两个：

```text
attention[0, 0, 0] = [1, 0, 0, 0]
attention[0, 0, 1] = [*, *, 0, 0]
attention[0, 0, 2] = [*, *, *, 0]
attention[0, 0, 3] = [*, *, *, *]
```

`*` 是 softmax 后的非负权重，每一行和为 1。若改成 encoder，就不要这张下三角，只挡住 padding：

```python
# padding_mask: [B, N], True means "this token is padding"
padding_mask = token_ids.eq(0)
# scores: [B,H,N,N]
scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
```

Mask 的最后一个维度对应**被读取的 key positions**。看见 `[B,H,N,N]` 时，前一个 `N` 是谁在问（query），后一个 `N` 是谁被读（key）。

### 9.1 把一轮 attention 当成“每个位置问一遍全班”

如果第一次看这段代码，先暂时忘掉 `permute`、`transpose` 和矩阵乘法。只记住一个画面：LLM 一次处理一批句子；每句话有一排 token；每个 token 当前由一个向量表示。这里的五个维度分别回答五个不同的问题：

```text
[B, N, D]
 │  │  └── 每个 token 有多少个 feature
 │  └───── 一句话有多少个 token
 └──────── 一次并行处理多少句话
```

本例中：

```text
B = 2     同时处理两句话
N = 4     每句话有 4 个 token
D = 8     每个 token 用 8 个数字表示
H = 2     并行使用 2 套 attention 规则
Dh = 4    每个 head 看到 D/H = 8/2 = 4 个 feature
```

因此，进入 block 的 `x` 是 `[2,4,8]`。可以把它想成一摞表格：有 2 张表，每张表有 4 行，每行是一个 8 维 token 向量。**attention 不会把两句话混在一起**；第一个维度 `B` 只是让两张表并行计算。

#### 第一步：token ID 变成 token vectors

```python
token_ids                         # [B,N] = [2,4]
x = self.token_embedding(token_ids)       # [B,N,D] = [2,4,8]
x = x + self.position_embedding(positions)[None, :, :]
```

`token_ids` 里的数字只是词表编号，不是可以直接做语义运算的数。例如 `5` 不是“比 `2` 更有意义”。`nn.Embedding` 用编号查表，把每个 ID 换成一个 8 维向量。所以 `[2,4]` 中的每一个数字，都变成一行 `[8]`；四个 token 组成 `[4,8]`，两句话组成 `[2,4,8]`。

位置向量的 shape 是 `[N,D] = [4,8]`。加上 `[None,:,:]` 后变成 `[1,4,8]`，这个 `1` 可以广播到两个样本，于是它能和 `[2,4,8]` 相加。每个 token 最终同时携带两类信息：内容信息和位置信息。

#### 第二步：同一个 `x` 投影成 Q、K、V

```python
qkv = self.qkv(x)                  # [B,N,3D] = [2,4,24]
qkv = qkv.reshape(B, N, 3, H, Dh)  # [2,4,3,2,4]
q, k, v = qkv.permute(2, 0, 3, 1, 4)
                                   # each: [B,H,N,Dh] = [2,2,4,4]
```

`nn.Linear(8, 24)` 对最后一维做投影：每个 8 维 token 产生 24 个数字。这里的 24 不是新的语义维度，而是三份 `8` 拼在一起：

```text
24 = 3 × D = Q 的 8 维 + K 的 8 维 + V 的 8 维
```

接着把每份 8 维拆成两个 head、每个 head 4 维：

```text
一个 token 的 Q/K/V： [8]
拆成两个 head：       [H,Dh] = [2,4]
```

`reshape` 只是在重新解释数字的分组；它没有进行新的学习。`permute(2,0,3,1,4)` 则把维度顺序从 `[B,N,3,H,Dh]` 换成 `[3,B,H,N,Dh]`，这样第一个维度的三个切片就正好是 `q`、`k`、`v`。最后：

```text
q, k, v: [B,H,N,Dh] = [2,2,4,4]
```

注意：这里的 `H` 不是 batch，也不是 token 数。它表示有几套独立的 matching space；每个 head 都会独立算一张 attention 表。

#### 第三步：`q @ k.transpose(-2, -1)` 为什么得到 `[B,H,N,N]`

先忽略 `B` 和 `H`，只看一个 head、一个句子：

```text
Q       [N,Dh] = [4,4]
K       [N,Dh] = [4,4]
Kᵀ      [Dh,N] = [4,4]
Q @ Kᵀ [N,N]   = [4,4]
```

矩阵乘法的每一个格子都是一个点积：

```text
scores[i,j] = query[i] · key[j]
```

也就是：第 `i` 个位置提出的问题，和第 `j` 个位置的“可匹配标签”有多相似。这里有 4 个 query、4 个 key，所以得到 4×4 张表：

```text
                 被读取的 key position j
             0       1       2       3
发问的     ┌───────┬───────┬───────┬───────┐
query i=0  │ s00   │ s01   │ s02   │ s03   │
            ├───────┼───────┼───────┼───────┤
       i=1  │ s10   │ s11   │ s12   │ s13   │
            ├───────┼───────┼───────┼───────┤
       i=2  │ s20   │ s21   │ s22   │ s23   │
            ├───────┼───────┼───────┼───────┤
       i=3  │ s30   │ s31   │ s32   │ s33   │
            └───────┴───────┴───────┴───────┘
```

所以代码中的

```python
scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
```

并不是把两个 `[B,H,N,Dh]` 逐元素相乘。`@` 做的是最后两个维度的矩阵乘法，同时保留前面的 `B,H`：

```text
[B,H,N,Dh] @ [B,H,Dh,N] → [B,H,N,N]
[2,2,4,4] @ [2,2,4,4]   → [2,2,4,4]
```

这里第二个 `[2,2,4,4]` 虽然 shape 数字看起来没变，但含义已经变了：它是转置后的 `K`，最后两维从 `[N,Dh]` 变成了 `[Dh,N]`。`transpose(-2,-1)` 中的 `-2` 和 `-1` 表示“倒数第二维”和“最后一维”。

#### 第四步：causal attention 不是另一种乘法，而是删掉未来的边

这里应读作 **causal attention**（因果 attention），不是 casual attention。自回归 LLM 预测下一个 token 时，位置 `i` 不能读取未来位置 `j > i`。因此 causal mask 是一张固定的布尔表：

```text
future = torch.triu(torch.ones(4, 4), diagonal=1)

        key:  0      1      2      3
query 0      False  True   True   True
      1       False  False  True   True
      2       False  False  False  True
      3       False  False  False  False
```

`True` 的地方代表“这条边禁止存在”。代码在 softmax **之前**执行：

```python
scores = scores.masked_fill(future, float("-inf"))
attention = scores.softmax(dim=-1)
```

为什么必须先 mask？因为 softmax 会把一行分数变成概率。如果先 softmax，再把未来位置的概率改成 0，该行剩余权重的总和就小于 1；而且没有重新归一化。把分数设为 `-inf` 后，

```text
exp(-inf) = 0
```

于是 softmax 自动给被禁止的位置分配 0，并把剩余可见位置重新归一化为总和 1。`dim=-1` 正是在最后一个 `N` 上做 softmax，也就是沿着“被读取的 key positions”做；每个 query 对所有可见 key 的权重加起来等于 1。

例如某个 head 的某一行原始分数是：

```text
query position 2: [1.2, 0.4, 2.0, 5.0]
```

因果约束下位置 2 不能看位置 3，于是实际 softmax 的输入是：

```text
[1.2, 0.4, 2.0, -inf]
```

最后得到的 attention 权重形如：

```text
[较小权重, 较小权重, 最大权重, 0]
```

它仍然可以重点读取位置 2，也可以读取过去的位置 0、1，但绝不会从位置 3 偷看答案。mask 只改变哪些格子有资格参与 softmax，不改变 `QKᵀ` 的计算规则。

#### 第五步：`attention @ v` 是按权重混合 payload

现在 `attention` 已经不是“相似度”，而是一张路由比例表：

```text
attention: [B,H,N,N]
v:         [B,H,N,Dh]
```

仍然只看一个句子、一个 head：

```text
A [N,N]  @  V [N,Dh]  →  context [N,Dh]
```

第 `i` 行的计算是：

```text
context[i] = A[i,0] * v[0]
           + A[i,1] * v[1]
           + ...
           + A[i,N-1] * v[N-1]
```

这就是 attention 的核心：`Q` 和 `K` 决定旋钮，`V` 是被旋钮调节音量的内容。因果 mask 下，未来 value 的权重为 0，因此位置 `i` 的 `context[i]` 只由自己和过去的 values 混合而成。

在代码中：

```python
context = attention @ v                 # [B,H,N,Dh]
context = context.transpose(1, 2)       # [B,N,H,Dh]
context = context.contiguous().reshape(B, N, D)  # [B,N,H*Dh] = [B,N,D]
```

每个 head 先得到一份 `[N,Dh]` 的结果；`transpose(1,2)` 把 token 维和 head 维换到更适合拼接的顺序；`reshape` 把 `[H,Dh]` 拼回 `[D]`。本例中 `[2,4]` 个 head 特征重新合并成 `[8]`：

```text
[B,H,N,Dh] → [B,N,H,Dh] → [B,N,H·Dh] = [B,N,D]
[2,2,4,4] → [2,4,2,4] → [2,4,8]
```

最后 `out_proj`、residual 和 FFN 都不会改变 `[B,N,D]` 这个外形：

```text
[B,N,D] → attention output projection → [B,N,D]
         → residual + LayerNorm       → [B,N,D]
         → FFN: D → d_ff → D          → [B,N,D]
         → lm_head                    → [B,N,vocab_size]
```

这里的 `lm_head` 对每个位置独立把 8 维 hidden 映射成词表中 10 个 token 的 logits，所以本例最后是 `[2,4,10]`。它表示两句话、每句话四个位置、每个位置对十个候选 token 各有一个分数；训练时再用 causal shift 让位置 `i` 的 logits 对齐目标 token `i+1`。

把整轮计算压缩成一句可执行的记忆口诀：

```text
Q：我在找什么？
K：我适不适合被找到？
Q @ Kᵀ：每个位置对每个位置打分
mask：未来位置没有资格参加
softmax：把分数变成每行和为 1 的读取比例
A @ V：按比例混合各位置真正携带的内容
拼回 heads：把多套关系合成一个 D 维表示
```

真正训练语言模型还需要 dataset、target shift、cross-entropy、optimizer 和多层 block。这段代码只负责一件事：让你在张量上把第 2–8 节走通。Token 一直被写成 subword；下一问是：向量从图像来、或者 \(N\) 大到那张 \(N\times N\) 表放不下，同一套路由还成立吗？

---

## 10. Token 不必是词，序列也不必很短

刚才一直把 token 想成 subword。路由机制并不在乎向量从哪来：只要有 \(N\) 个 \(D\) 维向量，就可以做同一套 Q/K/V。代价也不在乎语义，只在乎 \(N\)。Encoder 里每个 query 对每个 key，pair 数随 \(N^{2}\) 走；decoder 大约一半格子被 mask 掉，仍是二次。这就是长序列首先撞上的墙。第 5 节已经说过出路：把 \(A\) 变稀疏、低秩、可核分解，或加少数 global tokens 当远距离驿站。少连一些边能省算力，也可能把真正需要的远距离关系剪断——和 CNN 扩大 receptive field 却不保证信息被用上，是同一类 trade-off。

图像把 \(N\) 推得更极端。每个 pixel 当 token，一张普通图就有几万个位置，dense attention 立刻不现实。Vision Transformer 的回答不是另发明一种 attention，而是改 token 的粒度：

```text
Image
 → non-overlapping patches
 → flatten / project each patch
 → patch tokens + position
 → Transformer encoder
```

Patch 越大，token 越少、attention 越便宜，细粒度空间信息越粗。二维邻接关系不再像卷积那样写进 kernel，而要靠 position 和数据去学。Swin 一类方法把 self-attention 限制在 window 内，并在相邻层平移 window，让信息跨窗走；再周期性合并 patch，重新引入类似 CNN 的 hierarchy。那是在图像上用 mask 做局部拓扑，不是新的母问题。

所以这一节没有新的字母。它只是把同一套 content-dependent routing 换了 token 来源，并再次提醒：\(O(N^{2})\) 来自那张 pair 表。

若连接事先已知、而且不是“几乎人人连人人”，还要不要每次用内容去现算一张稠密 \(A\)？这正是第 13 章接手的问题。

---

## 11. 自测

1. CNN 的连接由什么决定？那句 `it` / `animal` 的句子为什么让这种连接不够用？  
2. 为什么只有 content embedding、没有 position 时，模型分不清 `dog bites man` 和 `man bites dog`？  
3. Tokenizer 为什么不能当成与模型无关的预处理？  
4. 在给 Q/K/V 起名字之前，每个 token 必须当场做哪三件事？用资料员类比，三件事分别对应什么？  
5. 为什么说 Q/K/V 通常是同一组 tokens 的三种 projections，而不是三堆不同的词？  
6. Scores 的 shape 为什么是 \([N,N]\)？Softmax 沿哪一维做，每一行表示什么？  
7. 用第 3 节的三个 values 说明：attention 为什么是加权和，而不是挑出一个 token？  
8. 除以 \(\sqrt{d_k}\) 和 softmax 各自管什么？前者为什么不是 cosine？  
9. 直接学习一张 dense attention matrix，为什么仍消不掉 \(O(N^{2})\)？FlashAttention 改的是哪一笔、没改哪一笔？  
10. Multi-head 与“把同一张 attention 复制 \(H\) 份”差在哪里？  
11. Attention 与 position-wise MLP 分别混合什么？Residual 和 LayerNorm 在这里各还第 11 章的哪笔账？  
12. Padding mask 与 causal mask 分别禁止什么？为什么 mask 必须加在 softmax 之前？  
13. Encoder、decoder、cross-attention 的 visibility 各是什么？BERT 的填空和 GPT 的 next-token 怎样用可见性说清楚？  
14. 为什么 GPT 的 training 可以沿序列位置并行，inference 仍必须逐步生成？KV cache 消除了 quadratic 吗？  
15. 图像为什么常常切成 patches 再送进 Transformer？这和第 5 节的 \(N^{2}\) 是同一件事吗？

---

## 12. 合上书再看一眼

CNN 把“谁和谁说话”写进相对位置；语言要的是远距离、并且随这句话的内容改接线。于是每个位置当场提出需求、宣告自己、准备 payload，用 scaled dot-product 打分，softmax 成路由比例，再把 values 软混合成新的表示。缩放是怕维度把 softmax 顶死；multi-head 是怕一种相似度装不下几种关系；MLP 改写每个 token 内部的 features；residual 与 LayerNorm 让这个 block 可以像第 11 章那样叠下去。Mask 不改匹配的算术，它只规定哪些格子根本不该存在，于是同一套路由变成双向的 BERT、因果的 GPT、以及带 cross-attention 的翻译模型。

贵的不是三个字母，是那张按内容现算的 \(N\times N\) 表。直接学一张 dense \(A\) 绕不开这张表的乘法；想更便宜，就得让表变稀、变低秩，或换一套近似。

下一章把镜头转过 90 度。Transformer 默认站在一张几乎全连接的 token graph 上，用内容去加权每条边；图神经网络面对的是另一类数据：对象是 node，关系是已经给定的 edge，拓扑本身就是输入。通信被固定 connectivity 限制，再在这些边上做 message passing——有时也可以再加权，但“谁有资格跟谁说话”不再每次由内容从全图里现选。

| 模型 | 谁和谁交换信息 | 连接怎样决定 |
|------|----------------|--------------|
| CNN | grid 上局部邻居 | 固定相对位置 |
| Transformer | 通常所有 tokens | 由 content 动态加权 |
| GNN | graph 上的 neighbors | topology 先限制，边上再可加权 |

Content-dependent routing 与 topology-constrained aggregation，是同一件信息交换问题的两种约束。

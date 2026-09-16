# 第 13 章：图神经网络（Graph Neural Networks）

> 书：《Understanding Deep Learning》Ch.13  
> 上一章：Transformer 让 tokens 根据内容交换信息。  
> 本章：当“谁和谁相连”本身就是数据时，怎样让每个对象从邻居收集信息？  
> 下一章：架构篇到此结束；标签被拿走以后，模型还能学什么？  
> 阅读建议：先把手算和 tensor 流程走通，再回头看 permutation、normalization 和深度。

---

## 0. 序列会按内容路由；图把连接写成数据

第 12 章解决的是一件很具体的事：序列里每个 token 根据**当前内容**决定该读谁。连接可以几乎全开，再用 attention 加权；位置本身并不在数据里，要另补 position encoding。那套办法默认了一种形状——对象排成一条（或被 mask 的）序列。

有一类数据，连“排成序列”都是假的。分子里哪个碳叫 1 号、哪个叫 7 号，只是文件里的行号；路网里每个路口的邻居数不同，也不存在“上方三个 pixel”；社交网络里，好友关系本身就是要利用、有时还要预测的东西。这时 topology 不是预处理，而是数据。

前面的数据都有规则位置，图没有：

```text
Image : pixels on a regular 2D grid
Text  : tokens on a 1D sequence
Graph : nodes joined by arbitrary edges
```

图还带来三件 CNN 和 Transformer 不必同时面对的事：每张图的 node 数可以不同；每个 node 的 neighbor 数可以不同；有时训练数据不是许多张图，而是一张巨大的、只标了一部分的图。

于是本章只追一个母问题：

> **怎样让每个 node 用同一套规则，从自己的 neighbors 收集信息，并且不依赖任意的 node 编号？**

一句话先把层的形状钉住，后面所有公式都是它的特例：

\[
h'_v
=
U\bigl(
h_v,\
\operatorname{AGG}\{
M(h_u,h_v,e_{uv}):
u\in\mathcal N(v)
\}
\bigr).
\]

完整调查链如下。后面每一节只补这条链上的一个缺口：

```text
Transformer：序列上按内容路由
        │
        ▼
Graph：连接关系本身就是数据
        │
        ▼
同一套规则，每个 node 只从 neighbors 收信息
        │
        ▼
Message → Aggregate → Update
        │
        ▼
先手算一层（会执行）
        │
        ▼
写成 H、A：adjacency 只负责指定收谁
        │
        ▼
编号无语义 ⇒ permutation equivariance / invariance
        │
        ▼
degree 不同 ⇒ normalization
        │
        ▼
graph / node / edge 三种 output contract
        │
        ▼
许多张图，还是一张巨图；太大就要采样
        │
        ▼
层数变多：oversmoothing 与 oversquashing
```

这条链不要倒过来走。若还不知道一条 message 是什么 tensor，就先讨论 permutation equivariance，公式虽然正确，但没有可附着的具体计算。所以下面分两遍：第一遍只求会执行——当前 node 从哪些邻居收消息、每条 edge 上产生什么、多条 messages 沿哪个维度聚合、聚合结果怎样变成新的 representation；第二遍再问为什么必须对邻居顺序不变、为什么要 degree normalization、为什么不能靠把层数加到五十去“看见更远”。

---

## 1. 分子、路网、社交：对象不同，结构相同

先别背分类。把同一句话套进三个场景，图的最小零件自己会冒出来。

分子里，node 是 atom，edge 是 chemical bond。Atom 上可以挂 element、charge、mass；bond 上可以挂键型、键长。要预测的常常是整张图的性质：毒性、溶解度。一个碳原子独自的原子序数不够用，相邻原子和键的类型会改写它在这张分子里的角色。

路网里，node 是路口或路段，edge 是可通行连接。位置、车道、当前拥堵是 node 上的状态；距离、限速、转弯代价是 edge 上的状态。任务可能是旅行时间或路径。道路不是图像那种规则 grid：立交桥的 degree 可以很大，死胡同只有一个邻居。相对位置“上方 / 下方”在这里没有全局一致的含义。

社交网络里，node 是 user，edge 是好友、关注或互动。Profile 和历史行为挂在人身上，互动次数和时间挂在关系上。任务是推荐、风险、或者“这两人之间该不该有一条边”。这里关系几乎就是数据本身：没有 edge list，剩下的只是一堆互不相干的用户向量。

三张图差在对象，不差在结构：

```text
objects  →  nodes   (with node features)
relations →  edges   (with optional edge features)
task     →  graph / node / edge label
```

一个对象的 contextual representation，必须参考它连出去的那些对象。这就是后面 message passing 要落实的那句话。

形式上只需要

\[
G=(V,E),
\qquad
V=\{v_1,\ldots,v_N\},
\qquad
E\subseteq V\times V.
\]

可选地，每个 node 带 $x_v$，每条 edge 带 $e_{uv}$，整张图还可以带一个全局特征 $g$。第一遍不必把图的种类背完；每次落地先问五句：

```text
Node 是什么？
Edge 是什么？
方向有没有意义？
Node / edge 上有哪些 features？
要预测的是整张图、每个 node，还是某条边？
```

这五句会自动点名常见类型。好友关系往往是 undirected：边没有方向，adjacency 对称。引用网络是 directed：论文 A 引用 B，不等于 B 引用 A。同一对节点之间可以有公路、铁路、航线，那是 multigraph。知识图谱里人和公司不是一类对象，边也有“创立 / 位于”等不同类型，那是 heterogeneous。点云、网格、带坐标的路网则是 geometric：node 还住在空间里。原书后文主要走 undirected、带 node features 的情形；edge features 会在需要时再请回来。

还缺一块，也是图最不像图像和文本的地方：**node 编号没有语义**。把碳原子在文件里的顺序对调，分子还是那张分子；把用户 ID 重新哈希，社交图的拓扑不变。图像打乱 pixel 就毁了，句子打乱词序就毁了。图只毁在你改了边，不毁在你改了行号。后面 permutation 那一节会把这句话写成对网络的硬约束；现在只需先记住：任何只依赖“谁排在第 3 行”的算法，从一开始就在学错的东西。

于是设计目标已经清楚。CNN 对每个位置使用同一套局部规则，是因为图像的邻域形状到处一样。图的邻域形状到处不一样，但仍希望**同一套规则**——否则每个 node 都要单独学“邻居意味着什么”，既浪费参数，也无法迁移到 node 数不同的新图上。规则的作用范围也不能是“全体 node 任意配对”，否则拓扑这个数据就被扔掉了。每个 node 只从自己的 neighbors 收集信息：这叫 relational inductive bias。

---

## 2. 同一套规则：纸条、汇总、更新

把“从邻居收集信息”写成一次可执行的更新。对 node $v$，一层只做三步。

邻居 $u$ 向 $v$ 写一张纸条。纸条可以看发送者、接收者，以及这条边上的关系：

\[
m_{u\to v}=M(h_u,h_v,e_{uv}).
\]

$v$ 的信箱里现在有一个**集合**，不是排好第一、第二、第三的序列。把集合收成一条摘要，必须用不依赖列出顺序的运算：

\[
\bar m_v
=
\operatorname{AGG}
\{m_{u\to v}:u\in\mathcal N(v)\}.
\]

常用的 AGG 是 sum、mean、max，或者后面才登场的 attention-weighted sum。用旧状态和摘要更新自己：

\[
h'_v=U(h_v,\bar m_v).
\]

```text
Neighbor states + edge features
              │
              ▼
         Message function M
              │
              ▼
    permutation-invariant AGG
              │
              ▼
         Update function U
              │
              ▼
    new contextual node state
```

AGG 必须 permutation invariant，不是审美，是数据形状。Neighbors 是 set：没有“左邻固定是 slot 0”。若把邻居 list 打乱答案就变，模型就是在拟合文件里的偶然顺序。CNN 能给“上方 pixel”和“下方 pixel”不同的 kernel 权重，正因为 grid 上的相对位置处处同构；图没有这套全局一致的上下左右，只能先用对称的汇总，把“谁在第几个”扔掉。

这三步落到 tensor 上，并不神秘。先规定单个对象的 shape：

| 对象 | Shape | 含义 |
|------|-------|------|
| sender $h_u$ | $[D_{in}]$ | 发送者当前 representation |
| receiver $h_v$ | $[D_{in}]$ | 接收者当前 representation |
| edge $e_{uv}$ | $[D_e]$ | 关系类型、距离、方向等 |
| message $m_{u\to v}$ | $[D_m]$ | 一条 edge 上真正传递的 vector |
| aggregated $\bar m_v$ | $[D_m]$ | node $v$ 收到的全部 messages |
| updated $h'_v$ | $[D_{out}]$ | 下一层 node representation |

整张图若用 edge list 存，常见 data flow 是 gather–message–scatter：

```text
Node states H                 [N, D_in]
Edge indices                  [2, E]
Edge features                 [E, D_e]
      │
      ▼ gather sender / receiver rows
Per-edge states               [E, D_in]
      │
      ▼ message function M
Messages                      [E, D_m]
      │
      ▼ scatter-sum / mean by receiver
Aggregated messages           [N, D_m]
      │
      ▼ update function U
New node states H'            [N, D_out]
```

$N$ 是 node 数，$E$ 是 edge 数。`edge_indices[0, e]` 与 `edge_indices[1, e]` 指定第 $e$ 条边的 sender / receiver。Aggregation 把 edge 轴 $E$ 压回 node 轴 $N$，但保留 message 的 feature 轴 $D_m$。

从这个角度看，GNN 并不是“拿 adjacency matrix 做一次神秘乘法”，而是：

> **先为每条 edge 建立一条 message，再按接收 node 把 messages 收拢。**

Dense 的 $AH$ 只是这一流程在“message 等于发送者状态、AGG 等于求和”时的压缩写法。Message 一旦显式依赖 edge feature、依赖 $(u,v)$ 这对端点，或经过非线性 $M$，就不能再随手交换成一次普通矩阵乘法。

GCN 是这条配方的一个特例，不是 GNN 的别名。后面矩阵公式出现时，把它读成“线性、共享 $W$、sum 聚合”即可；换了 $M$ 或 AGG，层还在，只是不再等于那一次 $\widetilde AHW$。

---

## 3. 先把手算做完：三个 node 的一条线

最小的、已经不对称的图就够：

```text
 h = 1        h = 2        h = 4
  v1    ────    v2    ────    v3
degree 1       degree 2      degree 1
```

假设这一层把“自己 + 一跳 neighbors”加在一起——也就是先给每个 node 加一条 self-loop，再求和。

\[
h'_1=1+2=3,
\qquad
h'_2=1+2+4=7,
\qquad
h'_3=2+4=6.
\]

于是

\[
[1,2,4]\longrightarrow[3,7,6].
\]

中心 node 的 degree 更大，求和结果也更大。这还没有学任何参数，纯粹是记账：信箱里纸条多，总和就大。

若改成平均：

\[
h'_1=\frac{3}{2},\qquad h'_2=\frac{7}{3},\qquad h'_3=3.
\]

中心不再只因为人多而数值膨胀。但 $\{2\}$ 的均值和 $\{2,2\}$ 的均值都是 $2$，邻居**数量**从摘要里消失了。Sum 保住了 count，mean 稳住了 scale：这个交换在三个数字上已经发生，不必等正式的 degree matrix。

第一遍到这里只要求一件事：你能指出每个 $h'_v$ 是哪些输入加起来的。谁是 $\mathcal N(v)$、self-loop 算不算进去、AGG 是 sum 还是 mean，三句话必须能对上那三个输出。三个数可以手加；训练时要对所有 node 同时做这件事，所以下一节把它写成矩阵。编号合不合法、degree 会不会虚高，等公式能跑了再问。

把同一张图拆成真正的 Message → Aggregate → Update，仍然只用三个标量。无向边、暂不加 self-loop、message 就等于发送者状态（$M(h_u)=h_u$），AGG 用 mean，update 用「自己加摘要」：

\[
h'_v=h_v+\operatorname{mean}\{h_u:u\in\mathcal N(v)\}.
\]

```text
v1 -- v2 -- v3
h =  1     2      4
```

Messages：

```text
v1 → v2 : 1
v2 → v1 : 2
v2 → v3 : 2
v3 → v2 : 4
```

Mean aggregate（每个 receiver 只看入边）：

\[
\bar m_1=2,
\qquad
\bar m_2=\frac{1+4}{2}=2.5,
\qquad
\bar m_3=2.
\]

Update：

\[
h'_1=1+2=3,
\qquad
h'_2=2+2.5=4.5,
\qquad
h'_3=4+2=6.
\]

输出集合 $\{3,4.5,6\}$。中心不再只因为 degree 大而「加更多项」——mean 已经除掉信箱大小；它仍然和两端不同，因为纸条内容不同。

现在只改抽屉标签，把行号换成 $v3,v1,v2$（特征跟着人走）：

```text
new order:  v3, v1, v2
new H:      [4, 1, 2]
edges:      v3--v1 不存在；v1--v2、v2--v3 仍在
```

对每个 **人** 再走一遍同样的 $M$/mean/$U$，得到 $h'_{v3}=6$、$h'_{v1}=3$、$h'_{v2}=4.5$。写在新行序里是 $[6,3,4.5]$，**作为集合仍是 $\{3,4.5,6\}$**。Graph-level 若做 sum readout，$3+4.5+6=13.5$，与旧编号相同。Node-level 预测必须跟着人搬家：原来关于 v2 的 $4.5$ 现在在第 3 行，不能留在「第 2 行」那个抽屉里。这就是下一节 permutation equivariance / invariance 要写成 $P$ 的那句话，三个数已经发生过一遍。

---

## 4. Adjacency 不是神秘乘法，只是在点名

要训练，就得把图放进 tensor。工程里常用

\[
H\in\mathbb R^{N\times D},
\]

每一行是一个 node 的 $D$-dimensional embedding。原书常把 node 放在列里，写作 $D\times N$，公式里的左右乘要对调。两者互为 transpose。第一遍固定一种，中途不要换方向。本章按行是 node 写。

连接用 adjacency matrix

\[
A\in\{0,1\}^{N\times N}.
\]

本文约定：**行是 receiver，列是 sender**，

\[
A_{vu}=1\quad\Longleftrightarrow\quad u\to v.
\]

于是第 $v$ 行回答：“哪些 node 可以给 $v$ 发消息？”对 node features 做

\[
(AH)_v=\sum_u A_{vu}h_u
\]

就是把所有 sender 的行，按 $v$ 的信箱加起来。Undirected graph 两个方向同时存在，因此 $A=A^T$，行约定和列约定暂时等价。Directed graph 就不再等价。不同 library 可能对调 sender / receiver 的行列；真正不能错的是：**代码里 adjacency 的朝向必须和消息流一致**。大型图通常也不存 dense 的 $N\times N$，而用 edge list、COO、CSR 这类 sparse 形式——那正是第 2 节的 gather–scatter。

原始 $A$ 往往不含自己。若更新时希望保留自身，就加 self-loop：

\[
\widetilde A=A+I.
\]

每个 node 给自己也写一张纸条。上一节手算用的就是这个。

$A$ 的幂还有一层组合含义：$(A^2)_{uv}$ 与从 $u$ 到 $v$、长度为 2 的 **walks** 数量有关，更高次幂对应更长 walks。Walk 可以重复经过 node 和 edge，所以它不是 unique simple paths 的计数。非零只说明距离不超过这个长度。一层 GNN 看的是 1-hop；把层叠起来，receptive field 沿 walks 变大，和 CNN 里叠卷积核是同一类几何，只是网格换成了任意拓扑。

最朴素的一层 GCN 把“线性变换 + 按 $\widetilde A$ 求和 + 非线性”写成

\[
H'=\rho(\widetilde AHW).
\]

$W$ 在所有 node 上共享，$\widetilde A$ 决定谁与谁加在一起，$\rho$ 是 activation。也可以先聚合再变换；线性且所有边共享同一个 $W$ 时，

\[
\widetilde A(HW)=(\widetilde AH)W,
\]

两种写法是同一件事。关键不是矩阵先后，而是：

> **同一 $W$ 在所有 nodes 上共享，topology 决定谁与谁交换信息。**

把上一节那三个 node 写进这个公式。含 self-loop 的 adjacency 是

\[
\widetilde A
=
\begin{bmatrix}
1&1&0\\
1&1&1\\
0&1&1
\end{bmatrix},
\qquad
H
=
\begin{bmatrix}
1\\2\\4
\end{bmatrix}.
\]

希望每个 node 出两个 features，取共享

\[
W=\begin{bmatrix}2&-1\end{bmatrix}.
\]

先各自乘 $W$：

\[
HW
=
\begin{bmatrix}
2&-1\\
4&-2\\
8&-4
\end{bmatrix},
\qquad
[3,1][1,2]\to[3,2].
\]

再按 receiver 把 sender 的行加起来：

\[
\widetilde AHW
=
\begin{bmatrix}
6&-3\\
14&-7\\
12&-6
\end{bmatrix}.
\]

逐行正是手算：

```text
v1 收自己和 v2：          [2,-1] + [4,-2]              = [6,-3]
v2 收 v1、自己、v3：      [2,-1] + [4,-2] + [8,-4]     = [14,-7]
v3 收 v2 和自己：          [4,-2] + [8,-4]              = [12,-6]
```

若 $\rho=\operatorname{ReLU}$，负的通道被削掉，

\[
H'=\begin{bmatrix}6&0\\14&0\\12&0\end{bmatrix}.
\]

现在再读 $H'=\rho(\widetilde AHW)$，它不是抽象符号，而是三件已经见过的事：

```text
HW      : 所有 nodes 共享同一个 feature transformation
Ã (HW)  : topology 决定每个 receiver 汇总哪些 rows
ρ       : 对聚合后的 features 做 nonlinear update
```

简单 GCN 用 $\widetilde A=A+I$，等于给自己和邻居同一套 $W$。也可以拆开：

\[
h'_v
=
\rho
\bigl(
W_{\mathrm{self}}h_v
+
W_{\mathrm{nbr}}
\operatorname{AGG}_{u\in\mathcal N(v)}h_u
\bigr),
\]

让模型区分“我本来是谁”和“邻居刚告诉我什么”。这仍是同一条 message-passing 配方，只是 $U$ 更宽了一点。

手算留下两个还没付账的现象。第一，我们把三个 node 叫成了第 1、2、3 行——换行号，这套矩阵还讲不讲同一张图？第二，中心那一行数字就是更大——那是“v2 真的更重要”，还是 degree 造成的记账假象？下一节先处理编号，再处理尺度。

---

## 5. 编号只是抽屉标签

手算和矩阵都把三个 node 写在第 1、2、3 行。行号若只是抽屉标签，换一个标签，输出必须跟着人走，而不能跟着行号走。同一张图可以把 nodes 重新编号：

```text
old IDs:  v1  v2  v3
new IDs:  v3  v1  v2
```

拓扑没变，只是 $H$ 的行、以及 $A$ 的行和列，被同一个 permutation 一起重排。若 permutation matrix $P$ 负责这次重排（每一行、每一列恰好一个 $1$），则

\[
H\mapsto PH,
\qquad
A\mapsto PAP^T.
\]

图像打乱 pixel 会得到另一张图，句子打乱词序会得到另一句话；图打乱编号必须仍是同一张图。所以加在 $(H,A)$ 上的层不能“认出”行号。正确的要求按输出合同分成两种。

Node-level 的输出应当跟着 node 一起搬家。这叫 **permutation equivariance**：

\[
F(PH,PAP^T)=PF(H,A).
\]

改编号以后，原来关于 v2 的预测，必须出现在 v2 的新行上，而不是留在“第 2 行”这个抽屉里。上一节那种“共享 $W$、按 $A$ 的邻居求和”的层满足这条：它从不读取绝对行号，只读取 $A$ 指出的那些行。

Graph-level 的输出不应当变化。这叫 **permutation invariance**：

\[
r(PH,PAP^T)=r(H,A).
\]

分子毒性不取决于你在文件里先写哪个原子。实现上，先得到一组 node embeddings，再用对集合对称的 READOUT 收成一条图向量——sum / mean / max / attention pooling 都可以，但不能依赖“先平均前一半 node”。

```text
Node classification : IDs 改了，prediction 跟着对应 node 移动   (equivariant)
Graph classification: IDs 改了，整张图 prediction 不变         (invariant)
```

两件事容易混成一句“顺序无所谓”。顺序无所谓，不等于 adjacency 可以不动。只打乱 $H$ 的行、却不打乱 $A$，等于把别人的邻居表贴到你头上，图已经换了。Invariant 也不是 equivariant 的别名：一个跟着编号走，一个对编号无感。它们对应两类任务，不是两种口味。

这不是审美。编号是存储选择；模型若依赖它，学到的是文件系统，不是化学键。编号合法之后，手算里那个更俗的现象还在：中心 node 的数字就是更大。那是拓扑语义，还是 degree 造成的记账假象？

---

## 6. 邻居有多有少，数字不能跟着虚高

直接用 $AH$，高-degree node 会汇总更多项，magnitude 天然更大。三个 node 那条线上，$[3,7,6]$ 的中心项首先更大，不是因为 v2 的特征更“重要”，只因为它的信箱多一张纸条。若下游是 softmax 或对尺度敏感的分类器，模型可能先学会“信箱大的人比较重要”，而不是学会纸条上写了什么。

Mean normalization 用 degree matrix 把每一行除以入度。$\widetilde D$ 是对角阵，

\[
\widetilde D_{vv}=\sum_u\widetilde A_{vu},
\]

\[
H'=\rho(\widetilde D^{-1}\widetilde AHW).
\]

每个 node 对（含或不含自己的）neighbors 求平均。三个 node 的那条线上，这正是把 $[3,7,6]$ 换成 $[3/2,\ 7/3,\ 3]$ 的矩阵写法。Mean 让不同 degree 的 node 更可比，也丢掉 count：一个值为 2 的邻居，和两个值为 2 的邻居，摘要可以相同。

经典 GCN 常用对称形式，同时看 receiver 和 sender 的 degree：

\[
H'
=
\rho\bigl(
\widetilde D^{-1/2}
\widetilde A
\widetilde D^{-1/2}
HW
\bigr),
\qquad
\widetilde A=A+I.
\]

直观上，从超高-degree 的 sender 传来的消息被下调——它和太多人说话，单条边携带的“独特信息”更少；receiver 自己的 degree 也进入归一化，避免信箱大小单独主导 scale。不必把左右两个 $-1/2$ 背成仪式。先保留目的：

> **防止 degree 大小单纯主导 feature scale，让不同 degree 的 nodes 在数值上更可比。**

原书还写过一种略有差别的安置：先对 $A$ 做 $D^{-1/2}AD^{-1/2}$，再另外加 $I$。和把 self-loop 折进 $\widetilde A$ 再归一化，不是同一条代数式，意图相同。见到两种公式，先看 self-loop 进了哪一步，不要混着对矩阵。

AGG 还可以换成逐维 max：抓最强的局部证据，丢掉非最大值和数量。没有一种汇总永远更好。Sum 表达力强、保住 count，也把 degree 写进了 magnitude；mean 对 degree 更稳，可能分不清“一个 2”和“两个 2”；max 敏感于峰值，对重复计数几乎失明。后面的 attention 会把权重从“拓扑给的系数”改成“内容算出来的系数”，那是另一笔交易，等任务和连通性都说清再请它。

---

## 7. 你到底要预测谁？

一层 GNN 产出的是一组仍然按 node 排列的 embeddings：开头每一行只知道自己，结尾每一行带着邻域语境——和词向量进 Transformer 之前、之后的对比是同一类变化。这些 embeddings 还不是任务。任务决定最后那张 output contract，也决定你该用 equivariance 还是 invariance。

Graph-level 输入整张图，输出一个 label 或一组值：分子毒性、蛋白质性质、电路指标。需要

\[
h_G=\operatorname{READOUT}\{h_v:v\in V\},
\]

再接到线性层或 MLP。READOUT 必须 permutation invariant，否则换一种原子编号，毒性分数会跟着晃。原书的 mean pooling 是把所有 node 平均后再做一次仿射；sum pooling 则把图的大小留在 magnitude 里，有时那正是想要的。

Node-level 为每个 node 预测：论文领域、用户风险、路段拥堵、点云上“这一点属于机翼还是机身”。输出 shape 通常是 $[N,C]$，并且应当对编号 equivariant。Loss 在 node 上独立计算，但表示本身已经过邻居混合，所以这不是“每个点单独的 MLP”。

Edge-level / link prediction 问两个 node 之间该不该有边、是什么类型、交互有多强。最省事的打分是 dot product

\[
s_{uv}=h_u^\top h_v,
\]

过 sigmoid 得到“边存在”的概率；更宽的是

\[
s_{uv}=\operatorname{MLP}([h_u,h_v,e_{uv}]).
\]

Dot product 便宜，表达力有限：它几乎在说“表示相近的人更该连在一起”。关系若依赖方向、类型或第三条路上的语境，MLP 更老实。

三种任务不是三种 GNN，是同一条 message-passing 主干上三种读出方式。换 READOUT 或换打分函数，不必换“从邻居收纸条”这件事。

工程上立刻碰到另一件图像没有的麻烦：不同分子的 $N$ 不同，不能像固定分辨率的 batch 那样直接叠成规则 3D tensor。常用做法是把一个 batch 里的许多张图，拼成一张**互不相连**的大图：

```text
Graph A nodes  -- internal edges only
Graph B nodes  -- internal edges only
A 与 B 之间没有 edge
```

再用 batch assignment vector 记下每个 node 属于哪一张图，READOUT 只在各自内部做。这利用了 message passing 的局部性：没有边，就不会跨图传消息。看起来像“训练一张巨图”，语义仍是许多独立样本。Graph-level 任务只出现在这种 inductive 设定里——你必须有训练图和测试图，才能谈“学一条规则，应用到新分子”。

---

## 8. 许多张图，还是一张巨图

到目前为止，全书几乎都是 inductive：用带标签的训练集学映射，再应用到从未见过的新样本。分子毒性是典型 inductive——训练时见过一些分子，测试时拿来一张新分子。

图还经常把人逼到另一头。科学引文网、大型社交网往往是**一张**巨图：一部分 node 有标签，其余没有。训练和测试在同一张拓扑上，只是 label 的可见范围不同。这叫 transductive。原书也把它说成一种 semi-supervised：未标注 node 的 features 和连接，在训练时就已经在图里，模型可以借那些未标注结构来做决定；代价是新来一批未标注 node 时，常常需要重新跑。

```text
Inductive    : 许多张图 → 学可复用的局部规则 → 应用到新图 / 新 nodes
Transductive : 一张大图，部分 labels 已知 → 给同一张图里未知的 nodes 打标
```

不要把“看见 feature”和“看见 label”混成泄漏。Transductive 设定里，test node 的 feature 和边在训练时可以可见，它的 **target label 不能进 loss**。模型不是在抄答案，它是在一张部分上色的图上，根据邻居的颜色和结构去推断空白处。Node-level 和 edge-level 两种任务，inductive、transductive 都会出现；graph-level 只有 inductive。

Inductive 要求参数不能绑死在某一个训练 node 的 ID 上。为每个用户单独准备一套权重，换一张图就失效。这也是为什么本章从一开始就坚持 parameter sharing：同一 $M$、$W$ 用在所有 node 上，学的是“邻居意味着什么”，不是“3 号用户意味着什么”。

巨图还有另一件麻烦：整张图可能放不进 memory；一个 node 的多层 neighborhood 又会迅速膨胀。和 CNN 一样，每个输出 node 有 receptive field，在图上叫 **$k$-hop neighborhood**——第 $k$ 层看到的是所有长度不超过 $k$ 的 walks 能碰到的点。若图比较密、层数比较多，一个 batch 里几个 labeled nodes 的 receptive field 可能几乎等于整张图。这叫 graph expansion。

常见对策是采样，不是换掉 message passing。

Neighborhood sampling：从 batch 里的目标 node 往回走，每一层只随机保留固定数量的 neighbors。子图变小，而且每次抽到的邻居可以不同，带一点类似 dropout 的正则。Graph partitioning：先把大图切成内部边尽量密、跨切边尽量少的子图，把子图当作 batch；也可以随机把几块拼回来，并恢复它们之间原来的边。

Trade-off 很具体。Sampling 降低成本和 memory，但引入估计噪声，也可能刚好丢掉那条关键的边。Partition 保住块内结构，却切断块间路径——而有些任务的信号恰恰走在那些被切开的边上。没有一种切法是“更真的 GNN”；它们都是在近似那次本该扫过 $k$-hop 的全量聚合。

采样把邻居当成可以少抽的集合。还有另一种对待邻居的方式：边都留着，但按内容决定谁更重要。那就把第 12 章的 attention 请回图上。

---

## 9. 拓扑先划定谁能说话，内容再决定听谁更重

到目前为止，邻居之间的权重要么相等（sum / mean / max），要么由 degree 这类**拓扑量**决定。内容还没有资格说“这个邻居比那个更相关”。第 12 章的 attention 恰好会说这件事。把它搬到图上，只要加一张结构 mask。

对允许的边 $u\to v$ 算一个 score $s_{uv}$，只在 $u\in\mathcal N(v)$（常常含 self）里做 softmax：

\[
s_{uv}=a(h_u,h_v,e_{uv}),
\qquad
\alpha_{uv}
=
\operatorname{softmax}_{u\in\mathcal N(v)}(s_{uv}),
\]

\[
h'_v
=
\sum_{u\in\mathcal N(v)}\alpha_{uv}\,Wh_u.
\]

原始 GAT 把变换后的两端表示拼起来，再和一个可学向量做点积得到 $s_{uv}$；也可以换成 Transformer 那种 dot-product。差别不在分数的代数细节，而在 **softmax 的定义域**：不是全体 node，是 adjacency 允许的那一小撮。

```text
Transformer : 通常所有 token pairs 都可连接，再由内容加权
GAT         : 只有 graph edges（及 self-loop）允许连接，再由内容加权
```

Graph topology 像一张 structural mask；attention 只在允许的 neighbors 中分配路由比例。它并没有“自动忽略拓扑”——没有边的地方，权重就是 0。把 GAT 理解成“Transformer 加上邻接 mask”，比理解成“一种全新的卷积”更不容易走偏。和第 12 章一样，attention weight 描述的是该层、该次 forward 的路由比例，不是可靠的因果解释。

边到这里仍然可以只是“连不连”。许多任务里边自己带着需要传递的信息：键型、路长与方向、互动时间、关系类型。通用 message 从一开始就为它留了位置，$M(h_u,h_v,e_{uv})$ 不是装饰。还可以走得更远：构造 line graph / edge graph——原图的每条 edge 变成新图的一个 node；原图中两条边若共享端点，就在新图里连一条边。于是对“边”的更新，变成了对“点”的一次普通 message passing。节点特征和边特征都在时，可以在两种图之间来回：nodes 更新 nodes、nodes 更新 edges、edges 更新 nodes、edges 更新 edges。路段转移、键与键的相互作用、边序列，适合这种把 edge 当作一等公民的写法。

配方到这里已经够用：连通性来自数据，权重可以来自拓扑或来自内容。还剩一个很诱人的念头——一层看一跳，五十层是不是就看见了全世界？下一节把这个念头拆开。

---

## 10. Hop 数加长以后，两件事一起坏

一层 message passing 看一跳。两层看邻居的邻居。看起来只要把层数加上去，任何远处的 node 都能进 receptive field，图也会和深度 CNN 一样自动获得层次特征。实践里 GNN 曾长时间加不深：早期 GCN 和 GraphSAGE 常用两层，再往上堆，训练和测试一起变差。原因不是“深网络没有好答案”，而是图上的聚合会以两种特别的方式把信息弄坏。

第一件是 **oversmoothing**。反复把邻居平均（或做类似平均的线性混合）之后，不同 node 的 embeddings 越来越像：

```text
layer 0 : each node still distinct
layer 1 : mix 1-hop neighbors
layer 2 : mix 2-hop neighbors
  ...
layer L : nodes in a community collapse toward one average
```

随机游走的图像更刻薄：一个 node 对另一个的影响，大约正比于 $K$ 步 walk 走到对方的概率；步数变大，这个概率靠近 walk 的平稳分布，局部差别被洗掉。Node classification 靠的就是那些还没被洗掉的差别。Receptive field 变大了，可是大家长得一样，变大没有用。

第二件是 **oversquashing**。即便你不平均到融合，远处的邻居数也常随 hop 指数增长，而每个 node 仍只有一个固定 $D$-dimensional 向量当瓶颈：

```text
exponentially many distant nodes
              \ | /
         [ fixed D-vector ]
           too small a pipe
```

信息被过度压缩。理论 receptive field 覆盖了远方，不等于远方那条关键路径还能在 $D$ 维里占到一个可分辨的方向。这和 CNN 里“感受野盖住全图 ≠ 已经理解全局”是同一类警告，图上只是邻居爆炸得更狠。

深度还附带优化和计算问题：gradient 难走（有人称为 suspended animation）、$k$-hop 子图膨胀、sampling 噪声逐层累积。第 11 章的 residual 可以请进来，

\[
h'_v=h_v+F(h_v,\mathcal N(v)),
\]

让新层默认先别毁掉已有表示，深层 GNN 因此变得可训。Residual 不是过平滑的解毒剂：旁路能保住一份自身，聚合路径上的反复混合仍在发生。Normalization、rewiring、positional / structural encoding、hierarchical pooling、virtual nodes、attention、更谨慎的采样，都是在同一对矛盾上做交换——既想看远，又不想把所有人搅成一杯。

所以“GNN 越深信息越多、所以一定更好”这句话，在图上尤其不可靠。层数买到的是更长的 walks，不是自动更好的表示。

---

## 11. 三种 architecture，三种“谁能跟谁说话”

CNN、Transformer、GNN 现在可以收成同一张图上的三种连通性，而不是三门互不来往的课。

```text
CNN layer         : fixed grid neighbors + shared local kernel
Transformer layer : often fully connected tokens + content-dependent weights
GNN layer         : arbitrary topology neighbors + shared message function
```

三者都在做局部或成对的信息交换，差别是“谁有资格通信”，以及权重从哪里来。

| 模型 | Connectivity | Weight rule |
|------|----------------|-------------|
| CNN | grid locality | 固定相对位置的 kernel |
| Transformer | often all-to-all | 由内容算出的 attention |
| GNN | graph edges | 共享 message；可选地在边上 attention |

CNN 能给“上方”和“下方”不同权重，因为网格的相对位置处处同构。图没有这套上下，GNN 才必须用对邻居集合对称的 AGG。Transformer 的 attention 可以看成完全图上的 message passing；GNN 把完全图换成数据里那张稀疏图。GAT 是两者的交点：连通性来自图，权重来自内容。

从第 10 章到第 13 章，架构篇实际上在反复做同一件事：把数据里已经存在的结构，写成 parameter sharing 的约束。图像的 locality 变成卷积核；深度变成 residual 的默认恒等；序列的内容关联变成 attention；图的任意拓扑变成 message passing。约束换来的是样本效率，以及“换一个位置 / 换一个 node 不必重学”的归纳偏置。

也正是在这里，监督式架构能教的课程告一段落。后面的数据将常常不再配一张标签 $y$。没有 $y$ 之后，模型还该压缩什么、生成什么、怎样才算覆盖了数据分布——那是第 14 章的母问题。GNN 仍可以出现在生成模型里，但那已经是另一条链：先问无标签时“好”是什么，再问某一种 generator 怎样实现它。

---

## 12. 自测

1. 给一个现实任务，分别定义 nodes、edges、node features、edge features 和 target。  
2. 用“传纸条”解释 message、aggregate、update；并指出哪一步必须对邻居顺序不变。  
3. 在 `v1—v2—v3`、特征 $[1,2,4]$、含 self-loop 的设定下，写出 sum 与 mean 的一层输出。它们各保留 / 丢失了什么？  
4. 为什么说 GNN 的核心是 gather–message–scatter，而不是“对 adjacency 做一次神秘乘法”？何种特例下两者等价？  
5. 为什么要加 self-loop？不加时，node 怎样才能在更新后仍看见自己的旧状态？  
6. $(A^2)_{uv}$ 计数的是什么？为什么它不是 unique simple paths？  
7. Node-level equivariance 与 graph-level invariance 有什么区别？只打乱 $H$ 的行、不打乱 $A$，为什么已经换了一张图？  
8. Degree normalization 要防止的是哪一种 scale 假象？Mean 与对称归一化各自还看了谁的 degree？  
9. Graph classification 为什么需要 permutation-invariant 的 READOUT？不同大小的图怎样做成一个 batch，且不会跨图传消息？  
10. Inductive 与 transductive 差在哪里？Transductive 里看见 test node 的 feature，为什么不等于 label 泄漏？  
11. Graph attention 与 Transformer attention 的 connectivity 有何不同？Attention weight 能当成因果解释吗？  
12. Oversmoothing 与 oversquashing 分别把信息弄坏在哪一步？为什么 receptive field 变大不等于远程信息被有效利用？Residual 能自动消除它们吗？

---

## 13. 合上书再看一眼

Transformer 在序列上按内容路由。图把“谁和谁相连”写成数据本身：node 数可变，degree 可变，编号只是抽屉标签，有时全世界只有一张巨图。GNN 的回答极短——每个 node 用同一套规则，从 neighbors 收纸条：先写 message，再用不依赖顺序的 AGG 汇总，再用 $U$ 更新自己。线性、共享、求和时，这三步压缩成 $\widetilde AHW$；一般情形仍是边上的 gather–scatter，不是一次神秘的矩阵咒语。GCN 只是其中一个特例。

因为编号无语义，node 输出必须跟着 permutation 搬家，图输出必须对 permutation 无感。因为 degree 不同，直接求和会让信箱大的 node 数值虚高，于是有 mean 和对称归一化——稳住尺度，同时可能丢掉计数。任务不改这条主干，只改读出：整张图要 invariant READOUT，每个 node 要 equivariant 的逐点头，边则在一对表示上打分。许多张小图可以拼成互不相连的大图来 batch；一张巨图则往往是 transductive 的，看见结构不等于看见标签，太大了就要采样，采样就会漏。

一层看一跳。把层加厚，walks 变长，两件事一起坏：反复混合把局部差别洗成社区平均，指数增长的远邻又被塞进固定的 $D$ 维瓶颈。Residual 让深层可训，并不自动修好这两件事。

CNN、Transformer、GNN 都是在规定“谁能和谁说话”。网格、全连接再按内容加权、数据里的任意边，是三种连通性，不是三个物种。架构篇到此把“有标签、且数据自带结构”能写进网络的事说完了。下一章标签被拿走，只剩 $\{x_i\}$。模型必须自己决定什么值得压缩、什么叫生成了一条像数据的新样本——监督式的 input–label 合同，从那里开始不够用了。

# 序列模型演化史：从 RNN 到 Hybrid DeltaNet

> 一条主线：**历史信息怎么存？怎么读？怎么忘？**
> 每个模型都是这三个问题的不同答案。
>
> 📖 阅读提示：本文公式均为 LaTeX，建议使用支持 KaTeX 的查看器（VS Code / Typora / Obsidian）以获得最佳渲染效果。

```
1990s          2014        2017          2020           2021         2024          2025
 RNN    →    LSTM    →  Transformer → Linear Attn →  DeltaNet → Gated DeltaNet → Hybrid
 压缩历史      学会遗忘      全表检索      记账本累加      纠错式写入    装上遗忘门      分工合作
 O(N)串行     O(N)串行     O(N²)并行      O(N)并行        O(N)并行      O(N)并行        粗活+细活
```

---

这条时间线先不要当成模型名的背诵表。它其实一直在追问同一件事：当第 $t$ 个 token 到来时，模型怎样让过去继续影响现在，同时又不让计算和记忆成本失控？下面的四个词是读完整条演化史的坐标系。

## 0. 先把四个词钉在同一条状态链上

假设输入是一串按顺序到来的 token：

```text
sequence:  我  →  吃  →  苹果
             t=1   t=2    t=3
             │     │      │
state:      h₀ →  h₁  →  h₂  →  h₃
```

- **序列（sequence）**：输入不是一袋无序 token，而是 $x_1,x_2,\ldots,x_T$ 这样的有序对象。
- **时间（time）**：这里主要指更新索引 $t$。它不一定是现实世界的秒数，而是“第几个输入到达、计算推进到哪一步”。
- **状态（state）**：处理到第 $t$ 步后，模型对当前上下文形成的内部表示，记作 $h_t$；它通常是一串数字，不是人类可直接读的句子。
- **记忆（memory）**：过去的信息已经改变了状态，而这个状态继续参与当前计算，所以历史对现在仍有影响。

最小的状态转移式只有一句：

$$h_t=f(h_{t-1},x_t)$$

如果还要产生预测，则再从状态读出：

$$y_t=g(h_t)$$

这四个词不是四个并列模块，而是同一条因果链上的四个观察角度：**序列给出顺序，时间标记更新，状态承载当前表示，记忆描述历史如何通过状态留下影响。**

### 一个小例子：当前输入相同，历史仍然不同

读到“苹果”时，当前输入都可以写成 $x_t=\text{“苹果”}$，但前文不同会带来不同的旧状态：

$$h_t^{(吃)}=f(h_{t-1}^{(我吃)},\text{“苹果”}),\qquad
h_t^{(用)}=f(h_{t-1}^{(我用)},\text{“苹果”})$$

于是同一个词可能被理解成水果，也可能被理解成品牌或设备。状态不是历史的逐字副本，而是面向任务的**压缩表示**：它只保留当前计算可能用得上的影响。这正是 RNN 的力量，也是它后来遇到长程依赖瓶颈的根源。

后面每一代模型都可以用三个问题检查：

1. **怎么存？** 历史被压成向量、账本矩阵，还是直接保留成 token 表？
2. **怎么读？** 当前输入只能读一个状态，还是可以对历史位置按需检索？
3. **怎么忘？** 信息被动覆写、由门控选择性衰减，还是根本不压缩而交给 mask 和上下文边界管理？

## 全局速览表

| 模型 | 状态形式 | 长程依赖 | 训练并行 | 推理状态大小 | 核心创新 |
|---|---|---|---|---|---|
| RNN | 向量 h | ✗ 梯度消失 | ✗ | O(d) | 循环压缩历史 |
| LSTM | 向量对 (h, c) + 门控 | ⚠️ 缓解 | ✗ | O(d) | 遗忘/输入/输出门 |
| Transformer | 无状态（KV cache） | ✅ 精确 | ✅✅✅ | O(N·d) | attention 全表路由 |
| Linear Attention | 矩阵 S | ⚠️ 有损累加 | ✅ | O(d²) 恒定 | kernel trick 去 softmax |
| DeltaNet | 矩阵 S + 纠错写入 | ⚠️ 更准的有损 | ✅（chunked） | O(d²) 恒定 | delta rule = 在线 SGD |
| Gated DeltaNet | 矩阵 S + 衰减门 γ | ✅ 可控遗忘 | ✅（chunked） | O(d²) 恒定 | 数据驱动的遗忘 |
| Hybrid | 上述混合 | ✅ 混合互补 | ✅ | O(d²)+少量O(N·d) | 分层分工 |

表中的“状态”要和上面的状态链对应起来：RNN 的 $h_t$ 是唯一记忆体；LSTM 把长期存储拆成 $c_t$，再用 $h_t$ 做工作区和输出；线性记忆模型把状态升级成一个可查询的矩阵 $S_t$；Transformer 则不再要求一个固定大小的状态代表全部历史，而是保留一张可检索的 token 表。

---

# 1. RNN（Recurrent Neural Network）

## 思想

**把全部历史压缩进一个固定向量**。每来一个新 token，就把旧状态和输入“搅拌”成新状态。也就是说，RNN 把上面的抽象状态链直接实现成：

```text
旧状态 hₜ₋₁ + 当前输入 xₜ  →  状态更新函数 f  →  新状态 hₜ  →  预测 yₜ
```

三个问题的答案：
- **怎么存？** 一个向量 h，容量固定
- **怎么读？** 输出层直接读 h
- **怎么忘？** 没得选——旧信息被反复覆写，永远"遗忘"（这是 bug 也是 feature）

## 📐 公式

状态更新（单步）：

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

梯度消失的数学根源——反向传播时误差沿时间连乘：

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=0}^{k-1} \tanh'(a_{t-i}) \cdot W_h$$

由于 $|\tanh'(\cdot)| \le 1$，连乘项模长**指数衰减**（除非 $W_h$ 特征值全部大于 1，那又会梯度爆炸）——两头都是死路。

### 把“记忆”看成状态的传递

用“我吃苹果”走一遍，RNN 并不是在第三步突然读取一份独立的历史文件：

```text
t=1: h₁ = f(h₀, “我”)       当前状态带上“我”的影响
t=2: h₂ = f(h₁, “吃”)       “我”不再作为原 token 出现，而是通过 h₁ 继续参与
t=3: h₃ = f(h₂, “苹果”)     “苹果”的解释同时受当前输入和 h₂ 影响
```

所以“记忆”不是额外外挂的缓存，而是**过去已经改变了状态**。如果两段历史不同，即使当前 $x_t$ 相同，旧状态也不同，得到的 $h_t$ 通常不同。这一点解释了 RNN 为什么能处理顺序，也解释了为什么状态容量和梯度传播距离会成为硬约束。

## 比喻

**传话游戏** 📣：第一个人看到句子，把意思压缩成一句话传给下一个人。传到最后，开头的内容早被稀释得没影了——这就是梯度消失的直观形态：模型"想不起"很久之前的事。

## PyTorch 手搓

```python
import torch

batch, T, d_in, d_hidden = 2, 16, 32, 64

W_x = torch.randn(d_in, d_hidden) * 0.1   # 输入→状态
W_h = torch.randn(d_hidden, d_hidden) * 0.1 # 状态→状态（循环的"自转"）
b   = torch.zeros(d_hidden)

h = torch.zeros(batch, d_hidden)          # 唯一的记忆体，初始化全零
for t in range(T):
    x_t = x[:, t]                          # 本步输入 [B, d_in]
    h = torch.tanh(x_t @ W_x + h @ W_h + b)  # 搅拌：新输入 + 旧记忆
    # h 现在"同时装着"第 0~t 个 token 的压缩信息
```

> ⚠️ 教学版：真实模型权重应是 `nn.Parameter` 并被 `nn.Linear` 管理。

## 复杂度小结

- 时间 O(N·d²)，但 **N 步必须串行**——GPU 再强也只能排队
- 瓶颈：tanh 导数 ≤ 1，连乘 → 梯度消失；$W_h$ 特征值 > 1 时 → 梯度爆炸

### 一个容易混淆的形状问题

输入若是：

$$X\in\mathbb{R}^{B\times T\times d_{\text{in}}}$$

RNN 会沿着时间维 $T$ 逐步更新，并把每个时间步的状态堆叠成：

$$H=[h_1,h_2,\ldots,h_T]\in\mathbb{R}^{B\times T\times d_{\text{hidden}}}$$

最后一个状态 $h_T$ 只是整段序列的一个压缩摘要，不等于整个输出序列 $H$。多层 RNN 会把上一层的 $H$ 作为下一层的输入；只有在 `hidden_size == d_model` 时，最后一层输出才会恰好具有 `[B, T, d_model]` 的形状。保留时间维，意味着模型保留了“每一步处理到哪里”的状态轨迹，而不是只留下句末一个答案。

---

# 2. LSTM（Long Short-Term Memory）

## 思想

RNN 的病根：**所有信息挤在一个 h 里，读写不分家**。LSTM 的解法是分家：状态不再只有一个向量，而是由长期存储 $c_t$ 和工作状态 $h_t$ 共同组成。这样，“要长期保留什么”和“当前要对外暴露什么”可以是两件事。

- **细胞状态 c**：一条"高速公路"，信息可以原样通过多步（解决梯度消失）
- **隐藏状态 h**：当前时间步的工作区和对外接口，读取 $c_t$ 后再输出
- **三个门控**：信息进出 c 都要过安检，安检力度**由数据自己学**

## 📐 公式

四个门与状态更新（$\sigma$ 为 sigmoid，$\odot$ 为逐元素乘）：

$$f_t = \sigma(W_f [h_{t-1}, x_t]) \qquad \text{(遗忘门：旧记忆留多少)}$$

$$i_t = \sigma(W_i [h_{t-1}, x_t]) \qquad \text{(输入门：新信息写多少)}$$

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t]) \qquad \text{(候选内容：想写的新东西)}$$

$$o_t = \sigma(W_o [h_{t-1}, x_t]) \qquad \text{(输出门：露出多少给外界)}$$

核心更新——**加法路径导数恒为 1**，梯度沿 c 高速无损回流几百步：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$h_t = o_t \odot \tanh(c_t)$$

## 比喻

**公司财务 + 三道审批门** 🏢：
- c 是对公账户，账本可以长期不动（高速公路）
- 遗忘门：旧预算保留比例，每季度审批一次
- 输入门：新开支入账比例
- 输出门：对外披露多少

对比 RNN 的传话游戏：LSTM 允许"把原话誊进笔记本原样保管"，而不是全靠脑子记。

## PyTorch 手搓

```python
def lstm_step(x_t, h_prev, c_prev, W, b):
    # W: [d_in + d_hidden, 4 * d_hidden]，把四个门的权重打包成一个矩阵
    gates = torch.cat([x_t, h_prev], dim=-1) @ W + b      # [B, 4H]
    i, f, g, o = gates.chunk(4, dim=-1)                    # 一刀切四份

    i = torch.sigmoid(i)   # 输入门 ∈ (0,1)：新内容写入比例
    f = torch.sigmoid(f)   # 遗忘门 ∈ (0,1)：旧记忆保留比例（学出 ≈1 = 长期记忆）
    g = torch.tanh(g)      # 候选内容：未经门控的原始写入值
    o = torch.sigmoid(o)   # 输出门：对外暴露比例

    c = f * c_prev + i * g    # 主记忆更新：高速公路上的加法（梯度友好的关键！）
    h = o * torch.tanh(c)       # 对外输出
    return h, c

h = torch.zeros(batch, d_hidden)
c = torch.zeros(batch, d_hidden)   # LSTM 比 RNN 多一个"账本"
for t in range(T):
    h, c = lstm_step(x[:, t], h, c, W_lstm, b_lstm)
```

## 复杂度小结

- 时间 O(N·d²)，依然**串行**，但细胞状态的高速公路让有效记忆长度从 ~10 步涨到 ~100+ 步
- 遗忘门 f 学出 ≈ 1 的行 = 长期记忆；≈ 0 的行 = 短期缓冲。门控把"遗忘"从 bug 变成了 **feature**——这一思想后面还会借尸还魂（Gated DeltaNet）

从 RNN 到 LSTM，变化不是“多了几个 sigmoid”这么简单，而是状态职责发生了拆分：RNN 只能把历史不断揉进 $h_t$，LSTM 则允许信息沿 $c_t$ 的加法路径稳定流动，再由 $h_t$ 决定这一刻要让什么信息参与下一层或输出。

---

# 3. Transformer（Self-Attention）

## 思想

**别压缩了，全员到齐，直接查表**。RNN/LSTM 把历史压成向量是"有损压缩"；Transformer 反其道而行——每个 token 的表示全程保留，用 attention 在需要时**按需检索**。三个问题的答案被彻底改写：
- **怎么存？** 不存！所有 token 的 K/V 摆在那里（KV cache 是推理期的缓存优化，不是架构必需）
- **怎么读？** Q 对全体 K 打相似度，softmax 变"采购权重"，对 V 加权求和
- **怎么忘？** 靠 causal mask 封死未来（decoder），或全靠数据自己学（encoder）

这里“时间”换了一种实现方式。Transformer 不靠 $h_{t-1}\to h_t$ 的递推来制造顺序，而是把位置编码和 causal mask 放进并行计算：每个位置仍然只能看到允许看到的历史位置。推理时的 KV cache 不是一个压缩后的单向量状态，而是已生成 token 的 key/value 表，因此它精确，却会随上下文长度 $N$ 增长。

## 📐 公式

Scaled Dot-Product Attention（Q、K、V 由输入经三个投影矩阵得到：$Q = XW_Q$，$K = XW_K$，$V = XW_V$）：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

缩放因子的由来：score 是 $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$，$d_k$ 项独立求和使方差 $\propto d_k$；除以 $\sqrt{d_k}$ 把 logit 拉回 O(1)，防止 softmax 饱和。

Causal 版本（decoder 的自回归约束，$M$ 为 mask，未来位置取 $-\infty$）：

$$\mathrm{Attention}_{causal} = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

## 比喻

**图书馆检索系统** 📚：
- K/V 是全馆书架上的书（key 是书名索引，value 是书的内容）
- Q 是你的检索词
- QKᵀ 是检索词和每本书的匹配打分；softmax 把打分变成"借阅比例"
- 输出 = 按借阅比例混合出来的"知识套餐"

RNN 是"背了一脑子压缩笔记的人"，Transformer 是"带着检索词进图书馆的人"——信息从不丢失，只在需要时被调取。

## PyTorch 手搓（MHA + causal mask）

```python
import math

batch, T, d_model, n_head = 2, 16, 64, 8
d_k = d_model // n_head

w_q = torch.randn(d_model, d_model) * 0.05   # 教学用 randn；真模型用 nn.Linear
w_k = torch.randn(d_model, d_model) * 0.05
w_v = torch.randn(d_model, d_model) * 0.05

q = (x @ w_q).view(batch, T, n_head, d_k).transpose(1, 2)  # [B, H, T, d_k]
k = (x @ w_k).view(batch, T, n_head, d_k).transpose(1, 2)
v = (x @ w_v).view(batch, T, n_head, d_k).transpose(1, 2)

scores = q @ k.transpose(-1, -2) / math.sqrt(d_k)          # [B, H, T, T]

# causal mask：第 i 行封死 j > i 的"未来"，自回归生成的基石
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))           # -inf → softmax 后 = 0

attn = torch.softmax(scores, dim=-1)                       # 每行 = 概率分布（采购单）
out = attn @ v                                             # 按单取货
```

## 复杂度小结

- 时间 O(N²·d)：每个 token 和所有 token 交互——全局性的定价
- 训练**完全并行**（矩阵乘伺候），GPU 越多越好——这是它掀翻 LSTM 的根本原因
- 推理靠 KV cache 把逐 token 生成从 O(N³) 拉回 O(N²)，但 cache 本身 O(N·d) 随上下文线性膨胀 → 催生后面所有"恒定状态"流派

---

# 4. Linear Attention

## 思想

Transformer 的 N² 来自 softmax：它阻止了 QKᵀ 与 V 的结合律重排。Linear Attention 用 **kernel trick** 换掉 softmax，让结合律成立：

$$\mathrm{sim}(q, k) = \phi(q)^\top \phi(k)$$

$$\mathrm{Attn}(Q, K, V) = \frac{\phi(Q)\big(\phi(K)^\top V\big)}{\phi(Q)\big(\phi(K)^\top \mathbf{1}\big)}$$

关键一步是结合律：先算 $\phi(K)^\top V$（累加一个 d×d 矩阵），再让 Q 去查它。矩阵一旦累加完成，每个新 token 只花 O(d²)——**序列复杂度从 O(N²) 降到 O(N)**。

累加式定义的状态：

$$S_t = \sum_{i=1}^{t} \phi(k_i)\, v_i^\top$$

## 比喻

**追加式记账本** 📒：每来一笔交易 (k_i, v_i)，就抄一行 $\phi(k_i) v_i^\top$ 进总账 S。查账时用 φ(q) 去点乘总账——O(d²)，不管账本积了多厚。

致命弱点：**串味**。两个不垂直的 key 会互相污染——"苹果→红"和"苹果→绿"记在一起，查出来是混合色。这个病灶正是 DeltaNet 的切入点。换句话说，Linear Attention 又把“状态”带回来了，只是它不再是 RNN 的向量，而是一个把 key 映射到 value 的矩阵账本。

## PyTorch 手搓

```python
d_feat, d_v = 64, 64

def feature_map(x):
    # φ: R^d → R^{d_feat}，论文常用 elu(x)+1（保证非负，像未归一化的概率）
    return torch.nn.functional.elu(x) + 1

S = torch.zeros(d_v, d_feat)                    # 累加器：所有历史的"总账"，恒定大小
denom_cache = torch.zeros(d_feat)               # 归一化分母的累加器
for t in range(T):
    k_t = feature_map(k_seq[t])                 # [d_feat]
    q_t = feature_map(q_seq[t])
    v_t = v_seq[t]                              # [d_v]

    S += torch.outer(v_t, k_t)                  # 追加式写入：v kᵀ 外积累加
    denom_cache += k_t

    o_t = S @ q_t / (denom_cache @ q_t)         # 查账：S·q 再归一化
```

对比 Transformer 那一节的 `attn @ v`：`attn` 是 N×N 的，而这里 S 是 d_feat×d_v 的——**把 N² 换成了 d²**。

## 复杂度小结

- 时间 O(N·d²)，恒定状态 O(d²)——上下文拉到 100 万 token，状态还是 64KB 📦
- 代价：写入是"无脑追加"，key 不垂直就串味；且永不遗忘，账本会饱和
- 训练可并行（chunked 分块累加，块间递推）

---

# 5. DeltaNet

## 思想

治 Linear Attention 的串味病：**写入前先查旧账，只写差额（delta）**。$\beta_t \in (0,1)$ 是数据自己学的"学习率"。因此，状态不只是“见过哪些内容”的总和，还会根据当前目标修正自己：它更像一个持续在线更新的联想记忆。

## 📐 公式

Delta rule 更新（Widrow-Hoff / LMS 规则）：

$$S_t = S_{t-1} + \beta_t \, (v_t - S_{t-1} k_t)\, k_t^\top$$

等价展开形式——"先沿 k_t 方向擦除旧关联，再写入新值"：

$$S_t = S_{t-1}\left(I - \beta_t k_t k_t^\top\right) + \beta_t\, v_t k_t^\top$$

**最美的等价视角（在线 SGD）**：它恰好是对损失函数走一步梯度下降——

$$L(S) = \frac{1}{2} \left\| S k_t - v_t \right\|^2, \qquad \nabla_S L = \left(S k_t - v_t\right) k_t^\top$$

$$S_t = S_{t-1} - \beta_t \, \nabla_S L$$

> **DeltaNet = 在线 SGD 训练一个联想记忆矩阵。状态不是被"设计"出来的，是被"学"出来的。**

## 比喻

**会改错的笔记本** ✏️（对比 Linear Attention 的追加式记账本）：
- Linear Attention：见到"苹果→绿"，直接抄一行——两本旧账串味
- DeltaNet：先翻旧账看到"苹果→红"，算出差额"绿−红"，**只写差额**——检索出来是干净的"绿"

记忆容量因此质变：同一页纸可以反复改写，而不是越写越花。

## PyTorch 手搓

```python
d_k = d_v = 64

S = torch.zeros(d_v, d_k)                       # 矩阵记忆体，RNN 向量状态的"升级款"
for t in range(T):
    k_t = torch.nn.functional.normalize(k_seq[t], dim=-1)  # key 归一化：内积=余弦相似度
    v_t, q_t = v_seq[t], q_seq[t]
    beta_t = torch.sigmoid(beta_raw[t])          # 学习率：数据决定这步写多狠

    pred = S @ k_t                               # 读：记忆对 k_t 的"当前答案"
    err  = v_t - pred                            # 惊讶度：正确值 − 旧答案

    S = S + beta_t * torch.outer(err, k_t)       # 纠错式写入：只写差额，不写生肉！
    o_t = S @ q_t                                # 读取输出
```

和 Linear Attention 手搓版逐行对比，只差写入那一行——从 `S += outer(v, k)` 变成 `S += β·outer(v − S·k, k)`。这一行，就是整篇论文。

## 复杂度小结

- 时间 O(N·d²)，状态 O(d²) 恒定——享受 Linear Attention 的全部好处
- 纯递推可并行化训练：chunk 内利用 $\prod_t (I - \beta_t k_t k_t^\top) = I - KAK^\top$（WY 表示）压成矩阵乘，块间递推
- 残留病灶：记忆永不遗忘，长期饱和 → 下一位选手登场

---

# 6. Gated DeltaNet

## 思想

给 DeltaNet 的记忆装上**遗忘闸门**：状态的每个通道独立衰减，γ 由当前输入自己生成——模型学会"这条信息值得记多久"。这是 LSTM 遗忘门思想在矩阵记忆上的借尸还魂：门控把遗忘从 bug 变成 feature（第二回了！）。它还顺带治好了一个隐蔽问题：**有效长程衰减**——梯度沿时间轴有了 γ 的指数收缩，深层训练稳定性大增。

到这里，“记忆怎么忘”终于不再只有两个极端：RNN 是被动覆写，Linear/DeltaNet 接近永不遗忘，Gated DeltaNet 则让矩阵记忆的不同通道拥有不同保质期。

## 📐 公式

$$\gamma_t = \sigma(W_\gamma x_t) \in (0,1)^{d_v} \qquad \text{(逐通道衰减系数)}$$

$$S_t = \gamma_t \odot \left[ S_{t-1}\left(I - \beta_t k_t k_t^\top\right) \right] + \beta_t\, v_t k_t^\top$$

- $\gamma \approx 1$ 的行：长期档案，原样保留
- $\gamma \approx 0$ 的行：便签纸，快速过期

## 比喻

**给笔记本的每一页贴保质期标签** 🏷️：
- 有的页贴"永久"，有的贴"本周有效"
- 翻到过期页时内容自动淡化（γ 衰减），不会和后来的记录打架
- 贴什么标签，由写笔记当下的内容自己决定

## PyTorch 手搓

```python
S = torch.zeros(d_v, d_k)
for t in range(T):
    k_t = torch.nn.functional.normalize(k_seq[t], dim=-1)
    v_t, q_t = v_seq[t], q_seq[t]
    beta_t = torch.sigmoid(beta_raw[t])
    gamma_t = torch.sigmoid(gamma_raw[t])        # 新增：逐通道衰减系数 ∈ (0,1)^{d_v}

    pred = S @ k_t
    # 先沿 k_t 方向做 DeltaNet 的"擦旧写新"，再整体乘以衰减门
    S = gamma_t[:, None] * (S - beta_t * torch.outer(pred, k_t)) \
        + beta_t * torch.outer(v_t, k_t)

    o_t = S @ q_t
```

对比 DeltaNet 版只多了 `gamma_t[:, None] *` 一个因子——又是一个"一行之差，一篇论文"的故事。

## 复杂度小结

- 与 DeltaNet 同级：O(N·d²) 时间、O(d²) 状态、chunked 并行训练
- 遗忘能力 = 混合长短记忆：γ 小的通道管近期模式，γ 大的通道管长程语义
- 凭此拿到工业界入场券：**Qwen3-Next、Kimi Linear 这类 2025 旗舰混合模型的主力层**

---

# 7. Hybrid DeltaNet（混合架构）

## 思想

没有免费午餐，只有**分工**：纯 DeltaNet 处理不了"精确检索"（复制一段电话号码、精确引用前文），纯 attention 又太贵。现代解法是**按层分工**：
- **大量 Gated DeltaNet 层**：干粗活——长程信息流的压缩、搬运、持续记忆，恒定状态随便上下文多长
- **少数 full attention 层**：干细活——精确召回、in-context learning、罕见模式检索

类比 Transformer 自己的内部 already 有分工（FFN 管记忆，attention 管路由），hybrid 是把分工**提升到层间**：记忆型层 + 检索型层交替堆叠。经验配比：attention 层占 1/4 ~ 1/8，长上下文成本砍掉大半，能力基本不掉。

## 📐 公式

整体 FLOPs（L_delta 个线性层 + L_attn 个注意力层）：

$$\mathrm{FLOPs} \approx L_{delta} \cdot O(N d^2) + L_{attn} \cdot O(N^2 d)$$

推理期状态总量：

$$\mathrm{State} \approx L_{delta} \cdot O(d^2) + L_{attn} \cdot O(N d)$$

attention 层被限制在少数（如 2/16 层），长上下文场景整体逼近 O(N)。

## 比喻

**工厂流水线** 🏭：
- DeltaNet 车间：粗加工、仓储物流（量大、便宜、允许近似）
- Attention 车间：精密质检、定制件（量少、昂贵、必须分毫不差）

整厂产能由粗加工车间撑起来，质检车间只在关键节点抽查。

## PyTorch 手搓（层调度示意）

```python
class HybridBlock(nn.Module):
    """一层 = 二选一：Gated DeltaNet 层（默认）或 Attention 层（按需插入）"""
    def __init__(self, d_model, n_head, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            # 检索型层：昂贵但精确，负责精确召回
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        else:
            # 记忆型层：恒定状态，负责长程信息流（内部即第 6 节逻辑）
            self.delta = GatedDeltaNetLayer(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        if self.use_attention:
            a, _ = self.attn(x, x, x, need_weights=False)   # 自注意力，Q=K=V=x
        else:
            a = self.delta(x)                                # gated delta rule 更新矩阵记忆
        x = x + a                       # residual
        return x + self.ffn(x)          # residual + FFN

# 层调度表：7 个线性层 + 1 个注意力层，周期重复
use_attn_schedule = [False]*7 + [True]
model = nn.Sequential(*[
    HybridBlock(d_model, n_head, use_attention=use_attn_schedule[i % 8])
    for i in range(16)               # 16 层：14 个 DeltaNet 层 + 2 个 attention 层
])

out = model(x)  # [B, T, d_model]
```

## 复杂度小结

- 整体训练 FLOPs：linear 层 O(N·d²) 占大头，attention 层 O(N²·d) 只占 1/8 → 长序列场景近线性
- 推理状态：每 linear 层 O(d²) 恒定 + 每 attention 层 O(N·d) 的 KV cache——后者被限制在少数层
- 能力定位：linear 层管"大概记得"，attention 层管"一字不差"——互补，而非替代

---

## 终章：一张图收束三代恩怨

```
            历史怎么存？          读取方式            遗忘机制           复杂度
────────────────────────────────────────────────────────────────────────────
RNN         压缩成向量 h         只能读 h            被迫遗忘          O(N) 串行
LSTM        向量 + 门控账本      读 tanh(c)          遗忘门（学）      O(N) 串行
Transformer 不存，摆全表(KV)     Q 对全表打分        只能 mask 未来     O(N²) 并行
LinearAttn  矩阵 S = Σφ(k)vᵀ     φ(q)ᵀS             无（饱和）        O(N) 并行
DeltaNet    矩阵 S + 纠错写入     qᵀS                无（饱和）        O(N) 并行
GatedDelta  矩阵 S + γ 衰减       qᵀS                γ 门（逐通道学）  O(N) 并行
Hybrid      linear 层 + 少数KV  分工：近似 + 精确    γ 门 + mask      ≈O(N) 并行
```

三代演化，一脉相承：

1. **RNN → LSTM**：遗忘从 bug 变 feature（门控思想诞生）
2. **LSTM → Transformer**：检索取代压缩，并行性掀桌（attention 思想诞生）
3. **Transformer → Linear/Delta/Gated**：N² 太贵，状态压缩回来，但带着门控和纠错两把新武器
4. **→ Hybrid**：承认没有银弹，让每一层只干自己最擅长的事

> 历史不会重演，但会押韵。LSTM 的遗忘门在 DeltaNet 里重生，attention 的全局性在 hybrid 里降级为"少数派特权"——序列模型的演化，是一部**旧思想不断借新躯壳还魂**的历史。

## 迁移检查：换一个序列，你还能预测吗？

1. **如果把“我吃苹果”改成“苹果吃我”，哪一项首先改变？**
   输入顺序改变，所以每一步接收到的 $x_t$ 不同，状态轨迹 $h_1,h_2,h_3$ 也不同；模型不需要额外保存一个“顺序标签”，顺序已经写进状态更新过程。
2. **如果两个样本当前 token 都是“苹果”，为什么输出仍可能不同？**
   因为它们的旧状态不同：$f(h_{t-1}^{(吃)},x_t)$ 和 $f(h_{t-1}^{(用)},x_t)$ 不是同一个计算。当前输入相同，不代表当前上下文相同。
3. **什么时候应优先使用 attention，而不是继续扩大压缩状态？**
   当任务需要一字不差地找回很久以前的局部片段时，固定大小的向量或矩阵状态可能会丢失细节；attention 用更大的 KV 表换取精确检索。Hybrid 的思路正是让少数 attention 层负责这类“细活”，让线性记忆层负责长程信息流。

把答案压成一句可迁移的判断规则：**先看任务需要的是“带着历史继续计算”，还是“回到历史某个位置精确取回”；前者适合状态记忆，后者需要可检索的记忆。**

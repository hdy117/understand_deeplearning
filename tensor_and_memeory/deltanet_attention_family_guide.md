# 从 RNN 到 Hybrid DeltaNet：序列记忆到底写在哪里

这篇教程只追问一件事：**新 token 到来时，模型手里的旧记忆长什么样，落笔时写进去的是什么。**

先把整章的答案放在这里。所有序列层都在做同一对动作：用当前问题去读记忆，再按当前 token 改记忆。差别只有两处。第一处是记忆的形状：RNN / LSTM 是一个固定长度的向量，Transformer 是按 token 保留的档案，linear attention 这一支是一张大小固定的矩阵。第二处是矩阵上的笔法：linear attention 把新的 value 追加进去；DeltaNet 先读出这把 key 的旧答案，只把「新 value 减旧答案」写回去；Gated DeltaNet 在纠错之前先把整张旧矩阵乘上一个衰减；Hybrid DeltaNet 让这张矩阵走长序列，另外留少数几层完整 attention 去翻原始档案。

所以贯穿全章的不变量可以写成：

$$
\text{state}_t=\operatorname{update}(\text{state}_{t-1},x_t),
\qquad
\text{output}_t=\operatorname{read}(\text{state}_t,q_t).
$$

$x_t$ 是第 $t$ 个输入，$q_t$ 是当前的 query。state 的形状决定你能记住什么；update 的笔法决定旧事实会不会被涂脏。读完以后，给你一个机制的名字，你应该能预测三件事：state 会不会随着序列长度 $N$ 变大，同一把 key 第二次出现时旧 value 是叠加还是被修正，以及一条很少重复的长字符串能不能被逐字抄出来。

下面不用七套互不相干的比喻。我们始终盯住两个读数：苹果的颜色，和旁边那个不该被碰掉的电话号码。每换一种账本，都问同一句：绿色写进去之后，这两个数各变成多少。

~~~mermaid
flowchart TD
    A[Vector summary<br/>RNN] -->|shared coordinates smear old facts| B[Gated vector<br/>LSTM]
    B -->|still no token address| C[Token archive<br/>Transformer]
    C -->|lookup grows as N squared| D[Fixed matrix, append write<br/>Linear attention]
    D -->|same key accumulates ink| E[Write the error<br/>DeltaNet]
    E -->|untouched keys never fade| F[Decay, then correct<br/>Gated DeltaNet]
    F -->|fixed rank cannot copy a rare string| G[Matrix layers plus a few archive layers<br/>Hybrid DeltaNet]
~~~

图里每一条箭头都是上一本账本答不上的那个问题。复杂度可以先记成三档，细表留到最后一节对账：Transformer 的查找随 $N^2$ 涨；RNN / LSTM 的 state 是 $O(d)$，不随 $N$ 涨；矩阵记忆的 state 是 $O(d_v d_k)$，扫描 $N$ 步的计算是 $O(N d_v d_k)$。$d$ 是 hidden 宽度，$d_k$、$d_v$ 是 key 和 value 的宽度。

> 代码约定：每个示例都可以单独复制运行。注释只写在真正改 state 的那几行。示例追求把更新规则算对，不是生产环境的 fused kernel。

## 1. 两个事实，三种写法

对话按时间到来，模型一次只能看一个 token。我们把要记住的内容收成三个数，方便手算：

| 顺序 | 事实 | 地址 key | 要写入的 value |
|---|---|---|---|
| 1 | 苹果是红色 | $k_{\text{apple}}=[1,0]$ | $1$ |
| 2 | 电话是 5 | $k_{\text{phone}}=[0,1]$ | $5$ |
| 3 | 苹果改成绿色 | 仍是 $k_{\text{apple}}=[1,0]$ | $3$ |

这两把 key 互相垂直，而且长度都是 1。垂直的意思是：沿苹果方向落笔，电话方向的坐标乘出来是 0。后面会看到，这个简化把「笔法」和「钥匙彼此漏墨」拆开了。真实模型里的 key 是学出来的，很少这么干净。

此后每一节的目标读数都是：苹果读出来是 $3$，电话读出来仍是 $5$。做不到的机制，我们看它具体歪在哪个数上。

## 2. RNN：整段历史挤在一个向量里

RNN 的 state 是一个 hidden 向量 $h_t$。每来一个 token，用旧向量和当前输入算出一个新向量，下一步只带走它。桌面上始终只有这一页。

$$
h_t=\tanh(W_x x_t+W_h h_{t-1}+b_h).
$$

$x_t$ 的长度是 $d_{\text{in}}$，$h_t$ 的长度是 $d_h$。$W_x$ 是 $d_h\times d_{\text{in}}$，$W_h$ 是 $d_h\times d_h$。读的时候没有第二本账可翻，输出层直接看 $h_t$。

用同一组数看它怎么涂脏。把 $h$ 的两维临时理解成 $[苹果, 电话]$，当前是 $[1, 5]$。玩具更新先把整页打一个折扣，再把绿色加进第一维。这里去掉了 $\tanh$，只保留「整个向量被同一次乘法带动」这件事：

```python
import torch

h = torch.tensor([1.0, 5.0])  # [apple, phone]
h = 0.5 * h + torch.tensor([3.0, 0.0])  # 整页一起缩放，绿色只加在第一维
assert h.tolist() == [3.5, 2.5]
```

苹果变成 $3.5$，既不是原来的红 $1$，也不是想要的绿 $3$。电话从 $5$ 掉到 $2.5$，因为折扣乘在整个向量上，没有哪一维可以躲开。真实的 $\tanh(W_x x+W_h h)$ 也是同一种结构：$W_h$ 把所有坐标混在一起，新 token 没有自己的抽屉。

还有第二条压力，它直接通向 LSTM。误差从 $h_t$ 传回 $h_{t-k}$ 时，中间要连乘 $k$ 次 $W_h$。$W_h$ 的最大奇异值小于 $1$ 时，连乘会按指数变小，很早的 token 就推不动参数了。这就是 vanishing gradient。state 只有一页，梯度还走不回那一页的开头。

API 形状可以顺便核对：序列再长，留下的 state 仍是一个向量。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
batch_size, seq_len, input_size, hidden_size = 2, 5, 4, 8
x = torch.randn(batch_size, seq_len, input_size)
y, h_last = nn.RNN(input_size, hidden_size, batch_first=True)(x)
assert y.shape == (batch_size, seq_len, hidden_size)
assert h_last.shape == (1, batch_size, hidden_size)  # N 没有进入 state
```

这一节的账：苹果 $3.5$，电话 $2.5$。下一节不增加纸的张数，只给每一维配上自己的闸门。

## 3. LSTM：同一页纸，每一维可以单独开关

LSTM 仍保存固定长度的向量，只是把 state 拆成对外的 $h_t$ 和长期的 cell $c_t$。三个 gate 都是当前输入和上一刻 $h$ 算出来的，数值在 $(0,1)$ 里，按坐标相乘：

$$
\begin{aligned}
f_t&=\sigma(W_f[x_t,h_{t-1}]+b_f),&
i_t&=\sigma(W_i[x_t,h_{t-1}]+b_i),\\
\tilde c_t&=\tanh(W_c[x_t,h_{t-1}]+b_c),&
c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
o_t&=\sigma(W_o[x_t,h_{t-1}]+b_o),&
h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

$f_t$ 决定旧 cell 留下多少，$i_t$ 决定新候选 $\tilde c_t$ 写入多少，$o_t$ 决定这一拍对外露出多少。关键结构在 cell 那一行：旧状态是**加**上去的。$f_t\approx 1$、$i_t\approx 0$ 的坐标上，旧数值可以几乎原样走过很多步，梯度也沿着这个接近 $1$ 的乘数走，不必每步都再乘一次 $W_h$。

玩具里我们直接指定闸门，假设训练已经把两件事实放到了两个坐标上。电话那一维 $f=1,i=0$，苹果那一维 $f=0,i=1$，候选值是绿色 $3$：

```python
import torch

c = torch.tensor([1.0, 5.0])  # [apple, phone]
f = torch.tensor([0.0, 1.0])
i = torch.tensor([1.0, 0.0])
g = torch.tensor([3.0, 0.0])  # 候选的绿色
c = f * c + i * g
assert c.tolist() == [3.0, 5.0]
```

苹果是 $3$，电话仍是 $5$。这是这种结构**能够表达**的最好情况，也是它和后面 DeltaNet 的读数第一次重合。两者重合的原因不同。LSTM 靠的是「两件事实碰巧住在不同坐标，闸门知道保护哪一维」。真实网络里，$W_x$ 和 $W_h$ 在进闸门之前就把坐标混过了，没有任何东西保证「电话」永远占着第二维。闸门能保护一个坐标，变不出一个按内容寻址的抽屉。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
batch_size, seq_len, input_size, hidden_size = 2, 5, 4, 8
x = torch.randn(batch_size, seq_len, input_size)
y, (h_last, c_last) = nn.LSTM(input_size, hidden_size, batch_first=True)(x)
assert y.shape == (batch_size, seq_len, hidden_size)
assert h_last.shape == c_last.shape == (1, batch_size, hidden_size)
```

cell 和 hidden 一样，长度都不随 $N$ 增长。要回答「第 17 个 token 到底是什么」，那张原始纸条得还在。下一节停止摘要，把每个 token 留档。

## 4. Transformer：问题来了，再去档案室里称一称

Transformer 的 state 就是历史 token 自己。每个 token 变成三份投影：query $q$、key $k$、value $v$。当前问题拿自己的 $q$，去和每一张历史 key 做点积，softmax 把点积变成一套加起来等于 $1$ 的权重，再按权重把 value 混在一起。

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V.
$$

$Q,K$ 是 $N\times d_k$，$V$ 是 $N\times d_v$。$M$ 是 mask。causal 模式下第 $t$ 个 query 只能看见 $i\le t$ 的 token。$N$ 个 query 对 $N$ 个 key，匹配次数是 $N^2$ 量级。

用三个 token 的分数看「档案还在」是什么意思。value 仍是 $[1, 5, 3]$，分别是红、电话、绿。问苹果时，分数集中在第三个 token；问电话时，分数集中在第二个 token。softmax 之后权重接近 one-hot，两个问题各取各的纸条：

```python
import torch

values = torch.tensor([1.0, 5.0, 3.0])  # red, phone, green，三张还在的纸条

def retrieve(scores: torch.Tensor) -> torch.Tensor:
    weight = torch.softmax(scores, dim=0)
    return torch.dot(weight, values)

apple = retrieve(torch.tensor([0.0, 0.0, 5.0]))
phone = retrieve(torch.tensor([0.0, 5.0, 0.0]))
assert apple > 2.9  # 几乎就是绿色 3
assert phone > 4.9  # 几乎就是电话 5
```

苹果约是 $3$，电话约是 $5$。两个事实都还在，因为 state 是那三张纸条，query 只是在称该拿哪一张。权重向量的长度等于已经看见的 token 数，所以序列一长，每个新问题都要再称一遍全部历史。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
batch_size, seq_len, d_model, num_heads = 2, 5, 8, 2
x = torch.randn(batch_size, seq_len, d_model)
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
y, weights = nn.MultiheadAttention(d_model, num_heads, batch_first=True)(
    x, x, x, attn_mask=causal_mask, need_weights=True
)
assert y.shape == (batch_size, seq_len, d_model)
assert weights.shape == (batch_size, seq_len, seq_len)  # 查找表的边长是 N
```

下一节要处理的就是这张边长为 $N$ 的表。如果权重能写成 key 和 query 的乘积，求和的顺序就可以换：先把历史折进一张固定矩阵，查询时只做一次矩阵乘向量。

## 5. Linear attention：把档案折进一张矩阵，笔法是追加

先看一个 query 想要的和。第 $t$ 步的输出，是过去每个 value 乘上「这把 key 和当前 query 有多像」：

$$
o_t=\sum_{i\le t} v_i\,(k_i^{\top} q_t).
$$

括号可以换位置。标量 value 时，$v_i(k_i^{\top} q)=(v_i k_i)^{\top} q$，于是

$$
o_t=\Big(\sum_{i\le t} v_i k_i\Big)^{\top} q_t.
$$

value 若是向量，乘积 $v_i k_i^{\top}$ 是一张 $d_v\times d_k$ 的外积，累加和是矩阵 $S_t$，读取是 $o_t=S_t q_t$。无论 $N$ 长到哪里，$S_t$ 的形状都不变。这就是 linear attention 把 $N^2$ 查找换成线性扫描的代数原因：历史被提前求和了。

教学上先丢掉 softmax 的分母。完整的 linear attention 通常还维护一个归一化向量 $z_t=\sum_{i\le t}\phi(k_i)$，输出要除以 $z_t^{\top} q_t$。分母让它更像 softmax 的「权重加总为 1」。本教程盯的是写入规则，所以先让每条记录以 $1$ 的资格入账。

用第 1 节的三笔追加写一遍。value 是标量时，矩阵退化成一行，追加就是 $S\leftarrow S+v k$：

```text
S starts at [0, 0]
+ 1 * [1, 0]  ->  [1, 0]     apple = red
+ 5 * [0, 1]  ->  [1, 5]     phone is stored
+ 3 * [1, 0]  ->  [4, 5]     green is added on top of red
```

```python
import torch

k_apple = torch.tensor([1.0, 0.0])
k_phone = torch.tensor([0.0, 1.0])
S = torch.zeros(2)
S = S + 1.0 * k_apple
S = S + 5.0 * k_phone
S = S + 3.0 * k_apple  # 追加绿色，旧的红色还在这一列里
assert torch.allclose(S, torch.tensor([4.0, 5.0]))
assert torch.dot(S, k_apple).item() == 4.0  # 1+3，红色没有被撤走
assert torch.dot(S, k_phone).item() == 5.0  # 垂直的电话没被这一笔碰到
```

电话是 $5$，苹果是 $4$。$4$ 就是 $1+3$：同一把 key 上的墨水只会累加，账本没有「把红色撤掉」这个动作。垂直的另一把 key 是安全的，所以干扰首先发生在**你想更新的那条关联**上，不一定先发生在旁边那条上。

钥匙一旦不垂直，旁边也会漏。假设问电话时 query 在苹果方向上有 $0.2$ 的分量，第三次追加的绿色会给电话读数再加 $3\times 0.2=0.6$。矩阵是一张共享的线性映射，靠近的 key 共用列空间。下一节先处理最干净的那一种失败：同一把 key，新 value 应该替换旧 value。

通用的追加扫描是下面这个形状。$S$ 是 $d_v\times d_k$，每个 token 先写入再读取，因此当前 token 看得见自己，看不见未来。

```python
import torch

def linear_attention_scan(k_seq: torch.Tensor, v_seq: torch.Tensor, q_seq: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(v_seq.size(-1), k_seq.size(-1))
    outputs = []
    for k_t, v_t, q_t in zip(k_seq, v_seq, q_seq):
        state = state + torch.outer(v_t, k_t)  # 追加，不撤旧答案
        outputs.append(state @ q_t)
    return torch.stack(outputs, dim=0)

torch.manual_seed(0)
seq_len, d_k, d_v = 6, 4, 5
outputs = linear_attention_scan(torch.randn(seq_len, d_k), torch.randn(seq_len, d_v), torch.randn(seq_len, d_k))
assert outputs.shape == (seq_len, d_v)
```

## 6. DeltaNet：落笔前先读旧答案，只写差额

第 5 节的矩阵在第三次写入之前，苹果方向上已经躺着 $1$。想要的是 $3$。追加把差额和旧答案一起留下了，所以读数是 $4$。

Delta rule 把这一笔改成三步。$S_{t-1}$ 是 $d_v\times d_k$，$k_t$ 是 $d_k$，$v_t$ 是 $d_v$，$\beta_t\in[0,1]$：

$$
\hat v_t = S_{t-1} k_t,
\qquad
e_t = v_t-\hat v_t,
\qquad
S_t = S_{t-1}+\beta_t\, e_t k_t^{\top}.
$$

人话：用这把 key 问一遍旧账，得到预测 $\hat v_t$；新 value 减去预测，是这一拍的惊讶 $e_t$；只把惊讶沿 $k_t$ 加回去。$\beta_t$ 是这一拍愿意改多少。我们的数字里 $\beta=1$，旧预测是 $1$，惊讶是 $3-1=2$，苹果方向加上 $2$，变成 $3$。电话方向和 $k_{\text{apple}}$ 点积为 $0$，加上去的那一列碰不到它。

```text
S = [1, 5]
predict apple = 1
error = 3 - 1 = 2
S = [1, 5] + 2 * [1, 0] = [3, 5]
```

同一笔更新还有第二种读法，两式相等：

$$
S_t
= S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^{\top}
= S_{t-1}(I-\beta_t k_t k_t^{\top})+\beta_t v_t k_t^{\top}.
$$

右边是「先沿 $k_t$ 擦掉旧成分，再写入 $\beta_t v_t$」。当 $\|k_t\|=1$ 且 $\beta_t=1$ 时，擦完以后这个方向的读数是 $0$，写完是 $v_t$，也就是整条关联被替换。$\beta_t$ 在 $0$ 和 $1$ 之间时，替换是按比例的。仍要求 $\|k_t\|=1$：

$$
v_t^{\text{new}}=(1-\beta_t)\,\hat v_t+\beta_t v_t,
\qquad
S_t=S_{t-1}-\hat v_t k_t^{\top}+v_t^{\text{new}} k_t^{\top}.
$$

$\beta=1$ 时新答案就是 $v_t$，$\beta=0$ 时新答案就是旧预测，矩阵不变。$\beta=0.5$ 时，苹果从 $1$ 走到 $1+0.5\times 2=2$，正好在红和绿的中点。

它也是一次梯度下降。把矩阵当成一个小线性模型，损失是当前这一对 key / value 的平方误差

$$
L_t(S)=\frac12\|S k_t-v_t\|_2^2,
\qquad
\nabla_S L_t=(S k_t-v_t)k_t^{\top}.
$$

沿负梯度走一步，步长 $\beta_t$，得到的就是上面的更新。DeltaNet 因此是在序列上在线训练一张 associative memory：Widrow-Hoff 的 delta rule，也就是 LMS。2021 年的 fast weight 工作已经把 linear attention 看成这种快权重；2024 年的 DeltaNet 保留这条更新，并给出沿序列长度并行训练的算法。

「擦干净」依赖单位 key。擦完再读，剩下的是 $\hat v\,(1-\|k\|^2)$。$\|k\|=1$ 时剩下 $0$；否则旧答案只会缩一截。所以实现里把 $k$ 做 L2 normalize：$0\le\beta\le 1$ 时，转移矩阵 $I-\beta kk^{\top}$ 在 $k$ 方向上的特征值是 $1-\beta\in[0,1]$，这一维不会越写越放大。垂直于 $k$ 的方向特征值是 $1$，那些关联原样留下。

```python
import torch
import torch.nn.functional as F

k_apple = torch.tensor([1.0, 0.0])
k_phone = torch.tensor([0.0, 1.0])

def delta_write(state: torch.Tensor, value: float, key: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    prediction = torch.dot(state, key)
    error = value - prediction
    corrected = state + beta * error * key
    # 第二种写法：擦掉旧成分，再写入 beta * value。两式必须相同。
    eye = torch.eye(key.numel())
    value_vec = torch.tensor([value])
    key_col = key.view(-1, 1)
    state_matrix = state.view(1, -1)
    other = state_matrix @ (eye - beta * torch.outer(key, key)) + beta * torch.outer(value_vec, key)
    assert torch.allclose(corrected, other.view(-1))
    return corrected

S = torch.zeros(2)
S = delta_write(S, 1.0, k_apple)
S = delta_write(S, 5.0, k_phone)
S = delta_write(S, 3.0, k_apple, beta=1.0)
assert torch.allclose(S, torch.tensor([3.0, 5.0]))

S_half = delta_write(torch.tensor([1.0, 5.0]), 3.0, k_apple, beta=0.5)
assert torch.allclose(S_half, torch.tensor([2.0, 5.0]))  # 红与绿的中点，电话不动

def deltanet_scan(k_seq: torch.Tensor, v_seq: torch.Tensor, q_seq: torch.Tensor, beta_raw: torch.Tensor) -> torch.Tensor:
    state = torch.zeros(v_seq.size(-1), k_seq.size(-1))
    outputs = []
    for k_t, v_t, q_t, beta_raw_t in zip(k_seq, v_seq, q_seq, beta_raw):
        k_t = F.normalize(k_t, dim=-1)  # 让 beta=1 时这一维可以被擦干净
        beta_t = torch.sigmoid(beta_raw_t)
        prediction = state @ k_t
        error = v_t - prediction
        state = state + beta_t * torch.outer(error, k_t)
        outputs.append(state @ q_t)
    return torch.stack(outputs, dim=0)

torch.manual_seed(0)
seq_len, d_k, d_v = 6, 4, 5
outputs = deltanet_scan(
    torch.randn(seq_len, d_k), torch.randn(seq_len, d_v), torch.randn(seq_len, d_k), torch.randn(seq_len)
)
assert outputs.shape == (seq_len, d_v)
```

这一节的账：苹果 $3$，电话 $5$。和 LSTM 玩具的读数一样，但地址是 key，不是「希望事实自己站到某个坐标上」。

两件事情矩阵仍然做不到。第一，某把 key 如果再也不出现，delta 就没有机会去改它，那一列会永远留着。第二，$S$ 是一个秩最多 $\min(d_v,d_k)$ 的线性映射。它能精确记住的、彼此独立的关联，数量级就是 $d_k$，不会随着 $N$ 增加。逐字抄一条很长的稀有字符串，需要的是 token 的原始位置，不是一条低秩的 key 到 value 映射。下一节先给「再也不被提到的旧墨水」一条退场的路。

## 7. Gated DeltaNet：先衰减整张旧账，再做同一次纠错

DeltaNet 只会在某把 key **再次出现**时修正它。一个早已写错、后来再没被问到的事实，没有误差项去碰它。Gated DeltaNet 在纠错之前加一道数据相关的衰减。同样的衰减，GLA 和 Mamba2 乘在「追加 $vk^{\top}$」前面；这里乘在 delta 写入前面。作者论文的标题就是这个结构：用 delta rule 改进 Mamba2。$\alpha_t=1$ 时，它应当原样退回上一节。

每个 token 按这个顺序走：

$$
\tilde S_{t-1}=\alpha_t S_{t-1},
\qquad
\hat v_t=\tilde S_{t-1} k_t,
\qquad
S_t=\tilde S_{t-1}+\beta_t(v_t-\hat v_t)k_t^{\top}.
$$

合成一式，就是

$$
S_t=\alpha_t S_{t-1}(I-\beta_t k_t k_t^{\top})+\beta_t v_t k_t^{\top},
\qquad 0<\alpha_t\le 1.
$$

$\alpha_t$ 在教学公式里是标量。实现里它经常按 head、甚至按 value 的 channel 各有一个数；通道版就是左乘 $\operatorname{Diag}(\alpha_t)$。Kimi Linear 里的 Kimi Delta Attention 把这道门拆得更细。门怎么参数化可以换，更新顺序先固定成：先衰减，再对衰减后的矩阵做 delta。

用数字看「被碰到的 key」和「没被碰到的 key」命运不同。旧账是 $[1, 5]$，$\alpha=0.9$，然后用 $\beta=1$ 把苹果写成 $3$：

```text
decay:     [1, 5] -> [0.9, 4.5]
predict:   apple reads 0.9
error:     3 - 0.9 = 2.1
write:     [0.9, 4.5] + 2.1 * [1, 0] = [3.0, 4.5]
```

苹果仍是 $3$。因为 $\beta=1$ 且 key 是单位向量，这个方向被擦干净再写上 $v$，衰减剩不剩得下来都不影响最终读数。电话没有被这把 key 碰到，只吃到衰减，从 $5$ 变成 $4.5$。若接下来两步都不再写入、$\alpha$ 保持 $0.5$，电话会继续变成 $4.5\times 0.5\times 0.5$。DeltaNet 没有这个乘数，电话会停在 $5$，连同任何过期的列一起停到序列结束。

```python
import torch
import torch.nn.functional as F

def gated_delta_write(state: torch.Tensor, value: float, key: torch.Tensor, beta: float, alpha: float) -> torch.Tensor:
    decayed = alpha * state  # 先让整张旧账褪色
    prediction = torch.dot(decayed, key)  # 预测必须读褪色之后的账
    error = value - prediction
    return decayed + beta * error * key

S = gated_delta_write(torch.tensor([1.0, 5.0]), 3.0, torch.tensor([1.0, 0.0]), beta=1.0, alpha=0.9)
assert torch.allclose(S, torch.tensor([3.0, 4.5]))

plain = gated_delta_write(torch.tensor([1.0, 5.0]), 3.0, torch.tensor([1.0, 0.0]), beta=1.0, alpha=1.0)
assert torch.allclose(plain, torch.tensor([3.0, 5.0]))  # alpha=1 就是 DeltaNet

def gated_deltanet_scan(k_seq, v_seq, q_seq, beta_raw, alpha_raw):
    state = torch.zeros(v_seq.size(-1), k_seq.size(-1))
    outputs = []
    for k_t, v_t, q_t, beta_raw_t, alpha_raw_t in zip(k_seq, v_seq, q_seq, beta_raw, alpha_raw):
        k_t = F.normalize(k_t, dim=-1)
        beta_t = torch.sigmoid(beta_raw_t)
        alpha_t = torch.sigmoid(alpha_raw_t).unsqueeze(-1)  # 每个 value channel 一个衰减
        state = alpha_t * state
        prediction = state @ k_t
        error = v_t - prediction
        state = state + beta_t * torch.outer(error, k_t)
        outputs.append(state @ q_t)
    return torch.stack(outputs, dim=0)

torch.manual_seed(0)
seq_len, d_k, d_v = 6, 4, 5
outputs = gated_deltanet_scan(
    torch.randn(seq_len, d_k),
    torch.randn(seq_len, d_v),
    torch.randn(seq_len, d_k),
    torch.randn(seq_len),
    torch.randn(seq_len, d_v),
)
assert outputs.shape == (seq_len, d_v)
```

$\alpha$ 长年靠近 $1$，旧关联留得久，过期的列也退得慢。$\alpha$ 靠近 $0$，矩阵灵活，远距离的事实留不住。这和 LSTM 的 forget gate 是同一类旋钮，乘在不同形状的 state 上。

输出端有时还有一道门，乘在 $o_t=S_t q_t$ 之后。那道门决定这一拍向残差流交多少，不改矩阵里已经写下的内容。别把它和 $\alpha$ 算成同一道门：$\alpha$ 改记忆，$\beta$ 改写入强度，输出门改露出多少。

推理时上面的循环就是控制流，一步接一步，像 RNN。训练时纯循环喂不饱 GPU。DeltaNet 把一个 chunk 内部的逐步纠错收成下三角结构上的矩阵乘，用的表示叫 WY representation；chunk 之间仍然把上一块的 $S$ 传给下一块。计算顺序变了，上一节和这一节的更新规则没有变。并行那一段的推导放在论文和作者博客的 Part II，这里不把公式再分叉一次。

到这里，苹果可以被改成 $3$，电话会按 $\alpha$ 慢慢变旧，矩阵大小始终是 $d_v\times d_k$。逐字抄写仍缺一张按位置检索的原始档案。

## 8. Hybrid DeltaNet：长序列走矩阵，精确检索留给少数层

固定秩的矩阵擅长的是「反复出现的关联」：苹果的颜色、某种语言模式、一条可以被 key 再次点名的事实。它不擅长的是只出现一次、必须按原位置取回的字符串。电话号码若是一串很少重复的 token，下一步生成依赖的是档案里那一个具体位置，而不是 $d_k$ 个线性槽位里的一个投影。

Hybrid 的办法是层的排班，不是又一条新的更新公式。连续几层只跑 Gated DeltaNet，state 固定，扫描随 $N$ 线性增长；每隔若干层插入一层完整 attention，在那里重新看到 token 级的 key / value 表。Qwen3-Next 采用 Gated DeltaNet 和 Gated Attention 的 hybrid，层数比例大约是 $3:1$。Kimi Linear 用更细粒度门控的 KDA，再和 Multi-Head Latent Attention 交替。比例是成本旋钮：attention 层越多，精确检索越多，随 $N^2$ 增长的那部分账单也越多。

~~~mermaid
flowchart TD
    X[Token stream] --> D1[Gated DeltaNet layer]
    D1 --> D2[Gated DeltaNet layer]
    D2 --> D3[Gated DeltaNet layer]
    D3 --> A[Attention layer<br/>exact token lookup]
    A --> D4[Gated DeltaNet layer]
~~~

真实 block 旁边还有 FFN 或 MoE，负责加工当前表示。那条路不保存我们这里讨论的历史账本，所以图里先不画。

下面的代码把排班写成真的循环：每 $4$ 层的最后一层走 causal attention，其余层走 delta 扫描。教学块为了短，delta 路径里用了固定的 $\alpha=0.9$，并且让 $q,k,v$ 都从同一个线性层长出来。生产实现里这三项是分开的投影，key 前面通常还有 short convolution。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def delta_layer(x: torch.Tensor, proj: nn.Linear, alpha: float = 0.9) -> torch.Tensor:
    qkv = proj(x)
    q, k, v = qkv.chunk(3, dim=-1)
    k = F.normalize(k, dim=-1)
    batch_size, seq_len, d_model = x.shape
    state = x.new_zeros(batch_size, d_model, d_model)
    outputs = []
    for t in range(seq_len):
        k_t, v_t, q_t = k[:, t, :], v[:, t, :], q[:, t, :]
        state = alpha * state
        prediction = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
        error = v_t - prediction
        state = state + torch.bmm(error.unsqueeze(-1), k_t.unsqueeze(1))
        outputs.append(torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1))
    return x + torch.stack(outputs, dim=1)  # 加回残差流

def attention_layer(x: torch.Tensor, attention: nn.MultiheadAttention) -> torch.Tensor:
    seq_len = x.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
    exact, _ = attention(x, x, x, attn_mask=causal_mask, need_weights=False)
    return x + exact

def hybrid_stack(x: torch.Tensor, proj: nn.Linear, attention: nn.MultiheadAttention, period: int = 4) -> torch.Tensor:
    for layer_index in range(period):
        if layer_index == period - 1:
            x = attention_layer(x, attention)  # 这一层打开原始 token 档案
        else:
            x = delta_layer(x, proj)  # 其余层只维护固定矩阵
    return x

torch.manual_seed(0)
batch_size, seq_len, d_model, num_heads = 2, 6, 8, 2
x = torch.randn(batch_size, seq_len, d_model)
proj = nn.Linear(d_model, 3 * d_model)
attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
y = hybrid_stack(x, proj, attention, period=4)
assert y.shape == x.shape
```

容易和排班搞混的一种写法，是在**每一层内部**把矩阵输出和 attention 输出按权重加起来。那也是一种融合，但每一层都付了 $N^2$ 的账，不再是 $3:1$ 的层数比例。Qwen3-Next 那类 hybrid 省钱的原因，是大多数层根本不建 attention 表。

| 你要的能力 | 交给谁 | 原因 |
|---|---|---|
| 长上下文里反复出现的关联 | Gated DeltaNet | state 不随 $N$ 增长，同一把 key 可以被修正，过期列会衰减 |
| 逐字抄写、按位置找回稀有 token | attention 层 | 原始 token 还在，query 能点名到那一个位置 |
| 训练时让 GPU 做大矩阵乘 | chunk 内的 WY 形式，加上 attention kernel | chunk 内并行，chunk 之间仍传递矩阵 state |
| 控制 $N^2$ 的比例 | 层的排班 | 精确检索只出现在少数层 |

## 9. 用同一张账把七种机制再读一遍

七种机制，三个问题。state 里有什么，read 怎么取，update 写下去的是什么。

| 机制 | state | read | update | 苹果 / 电话这个例子里 |
|---|---|---|---|---|
| RNN | 一个向量，不随 $N$ 涨 | 直接读向量 | 整页和当前输入混成新的一页 | 玩具里变成 $3.5$ / $2.5$，两维都被带动 |
| LSTM | cell 加 hidden，仍是向量 | 读 gated hidden | 按坐标遗忘、写入、露出 | 若两事实已在不同坐标，可以是 $3$ / $5$ |
| Transformer | 全部 token 的 K/V，随 $N$ 涨 | softmax 在位置上称权重 | 新 token 追加进档案，旧纸条还在 | 分数够尖时约是 $3$ / $5$ |
| linear attention | $d_v\times d_k$ 矩阵 | $S q$ | 追加 $v k^{\top}$ | $4$ / $5$，红色还叠在绿色下面 |
| DeltaNet | 同一张矩阵 | 先用 $k$ 读旧答案，再用 $q$ 读输出 | 只写 $\beta(v-\hat v)k^{\top}$ | $\beta=1$ 时是 $3$ / $5$ |
| Gated DeltaNet | 带衰减的同一张矩阵 | 衰减之后再读旧答案 | 先乘 $\alpha$，再做 delta | $\alpha=0.9,\beta=1$ 时是 $3$ / $4.5$ |
| Hybrid DeltaNet | 矩阵层加少数 token 表 | 两种读法按层出现 | 矩阵层纠错并衰减，attention 层保留原档 | 关联走矩阵，逐字抄写走 attention 层 |

四个容易混的边界，用这张表就能定下来：

- 线性复杂度说的是随 $N$ 的增长阶。每步还要做 $d_v\times d_k$ 的矩阵运算。$N$ 很大时这笔账划算；$N$ 不大而 $d$ 很大时，常数、kernel 和访存会让它并不更快。
- DeltaNet 的 state 和 linear attention 一样大。变的是笔法：追加得到苹果 $4$，写误差得到苹果 $3$。
- $\alpha$ 乘在 state 上。输出门乘在读完之后的向量上。两道门前后不同。
- Hybrid 的精确检索发生在那些真正建了 token 表的层。矩阵层的秩仍然封顶。

再做四道转移练习。答案就在这一节，先自己预测再对照。

1. 把 RNN 的 hidden 加宽到很大，它会变成 Transformer 吗？不会。加宽只是把那一页纸写得更密，state 里仍然没有「第 17 个 token」这个地址。电话和苹果还是挤在同一次 $W_h$ 乘法里。
2. 第 6 节里若跳过预测、直接追加，公式会变成什么？$e_t$ 换成 $v_t$，更新回到 $S\leftarrow S+\beta v k^{\top}$，也就是 linear attention。先读旧答案，是 DeltaNet 相对追加唯一多出来的那一步。
3. $\alpha$ 连续很多步等于 $0.99$，和连续等于 $0.5$，矩阵里没被再次点名的列会怎样？$0.99$ 每步只掉 $1\%$，远距离的关联还在，过期噪声也在。$0.5$ 两步就剩下四分之一，最近的纠错很灵活，早先的电话号码留不住。被 $\beta=1$ 再次点名的单位 key 例外：那一列会被写成当前 $v$，不由残留的 $\alpha$ 决定。
4. 把 $d_k$ 加大，能不能取消 attention 层？可以推迟这个需求，不能从结构上取消。$d_k$ 再大也是一个常数，精确关联的容量封在这个常数上；attention 的档案随 $N$ 增长。稀有字符串要的是位置，不是更多的线性槽位。$d_k$ 足够覆盖你关心的关联时，attention 层可以更少，这正是 $3:1$ 一类比例在调的东西。

把全章收成一句可以带走的判断：

> **先看 state 的形状，再看 read 有没有地址，最后看 update 写的是整页、原值，还是误差。** 向量是一页摘要，token 表是档案室，矩阵是一张固定的关联账。DeltaNet 在这张账上写误差，Gated DeltaNet 写误差之前先衰减，Hybrid 把逐字抄写留给少数档案层。

## 10. 延伸阅读

公式想再对一遍，按这个顺序就够了。Part I 是 delta rule 本身，Part II 是 chunk 并行，Gated Delta Networks 是把衰减接到 delta 上。

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- [DeltaNet Explained, Part I](https://sustcsonglin.github.io/blog/2024/deltanet-1/)（Part II、Part III 在同一博客）
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Qwen3-Next 架构介绍](https://qwen.ai/blog?id=qwen3-next)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)

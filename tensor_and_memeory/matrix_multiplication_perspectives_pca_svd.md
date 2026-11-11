# 矩阵乘法的几副眼镜：从 $A^T$ 到 Gram、PCA 与 SVD

> 一份面向教学的讲义：先用直觉和小例子理解“谁在和谁交流”，再补上 shape、公式和算法。

## 怎么读这份讲义

这份讲义反复使用同一条主线：

```text
矩阵乘法的多种视角
        ↓
transpose：把一个内积搬到另一边的 adjoint
        ↓
Gram matrix：两两 dot product 的总账本
        ↓
quadratic form：把方向放进矩阵，得到一个“碗”
        ↓
SVD：把任意线性 map 拆成旋转 → 拉伸 → 旋转
        ↓
PCA：对中心化数据做 SVD，找方差最大的方向
```

符号先认熟：$A\in\mathbb R^{m\times n}$ 有 $m$ 行、$n$ 列；$a_{:j}$ 表示第 $j$ 列，$a_{i:}$ 表示第 $i$ 行；$x\in\mathbb R^n$ 是输入方向，$y\in\mathbb R^m$ 是输出方向；$A^T$ 是 transpose，$\langle u,v\rangle=u^Tv$ 是 Euclidean inner product。

## 0. 同一个小例子，先不急着背维度

取

$$
A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\qquad
B=\begin{bmatrix}5&6\\7&8\end{bmatrix}.
$$

矩阵乘法给出

$$
C=AB=\begin{bmatrix}
1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\
3\cdot5+4\cdot7 & 3\cdot6+4\cdot8
\end{bmatrix}
=\begin{bmatrix}19&22\\43&50\end{bmatrix}.
$$

这四个数可以从四扇门进入：

1. **dot product 视角**：$C_{ij}$ 是 $A$ 的第 $i$ 行和 $B$ 的第 $j$ 列的 dot product。
2. **row 视角**：$C$ 的第 $i$ 行，是 $B$ 的各行按 $A$ 第 $i$ 行的系数做 linear combination。
3. **column 视角**：$C$ 的第 $j$ 列，是 $A$ 的各列按 $B$ 第 $j$ 列的系数做 linear combination。
4. **outer product 视角**：$AB$ 是若干个 rank-1 外积相加。
5. **composition 视角**：先做 $B$，再做 $A$，所以 $ABx=A(Bx)$。

同一扇门只是在回答不同问题：某个格子怎么算？整行在做什么？整列在做什么？一个 map 如何拼接？矩阵乘法不是五种不同的运算，而是一件事的五种记账方式。

## 1. 矩阵乘法的六副眼镜

### 1.1 dot product：一个格子是一场小会谈

$$
C_{ij}=a_{i:}b_{:j}=\sum_{k=1}^{n}A_{ik}B_{kj}.
$$

$k$ 是“共同接口”：$A$ 的第 $k$ 个输入坐标，必须和 $B$ 的第 $k$ 个输出坐标对接。于是每个 $C_{ij}$ 都是在问：**$A$ 的第 $i$ 行和 $B$ 的第 $j$ 列有多同频？**

### 1.2 row 视角：$A$ 的一行是一组配方

设 $A$ 的第 $i$ 行是 $(\alpha_1,\alpha_2,\ldots,\alpha_n)$，而 $B$ 的行分别是 $b_{1:},\ldots,b_{n:}$，那么

$$
C_{i:}=\alpha_1b_{1:}+\alpha_2b_{2:}+\cdots+\alpha_nb_{n:}.
$$

在小例子里，$A$ 的第一行是 $(1,2)$，所以

$$
C_{1:}=1\begin{bmatrix}5&6\end{bmatrix}+2\begin{bmatrix}7&8\end{bmatrix}
=\begin{bmatrix}19&22\end{bmatrix}.
$$

这正是 `linear layer` 的直觉：一行 weight 像一个 neuron 的配方，按比例拿来几条已有的“原料行”。

### 1.3 column 视角：$B$ 的一列是一组混合权重

设 $B$ 的第 $j$ 列是 $(\beta_1,\ldots,\beta_n)^T$，那么

$$
C_{:j}=\beta_1a_{:1}+\beta_2a_{:2}+\cdots+\beta_na_{:n}.
$$

小例子里，$B$ 的第一列是 $(5,7)^T$，所以

$$
C_{:1}=5\begin{bmatrix}1\\3\end{bmatrix}+7\begin{bmatrix}2\\4\end{bmatrix}
=\begin{bmatrix}19\\43\end{bmatrix}.
$$

这一扇眼镜特别适合解释“矩阵的列是输入 basis 经过 map 后的去向”：如果 $A$ 是一个 map，$A$ 的第 $j$ 列就是输入 basis $e_j$ 被送到哪里。

### 1.4 outer product：一层一层铺地毯

把共同索引 $k$ 拆开：

$$
AB=\sum_{k=1}^{n} A_{:k}B_{k:}.
$$

这里 $A_{:k}$ 是一个列向量，$B_{k:}$ 是一个行向量，二者的 outer product 是一个矩阵。每一块都是 rank-1 的“地毯”，所有地毯叠起来才是 $AB$。

### 1.5 composition：先做谁，顺序不能乱

$$
(AB)x=A(Bx).
$$

最右边的 $B$ 先作用于 $x$，再由 $A$ 接手。像先穿袜子再穿鞋，倒放时顺序自然反过来：

$$
(AB)^T=B^TA^T.
$$

### 1.6 geometry：网格、方向和长度怎么被改写

矩阵 $A$ 不只是一个数字表，它是一个 map：$x\mapsto Ax$。它可以旋转、拉伸、压扁、翻转，甚至把一个平面压到一条线。SVD 会把这些动作拆开，让“方向”和“拉伸倍数”分别露出来。

## 2. $A^T$ 的深刻意义：不是“把表格翻一下”

### 2.1 第一层：行列互换

$$
(A^T)_{ij}=A_{ji}.
$$

这是定义，但不是最有用的解释。真正有用的是：**transpose 把“看列”切换成“看行”**。如果 $A$ 的列代表 feature，$A^T$ 的行就把同一批 feature 摆到行上；如果 $A$ 的行代表 sample，$A^T$ 的列就重新暴露这些 sample。

### 2.2 第二层：transpose 是 inner product 下的 adjoint

对任意合法的 $A,x,y$，都有

$$
\boxed{\langle Ax,y\rangle=\langle x,A^Ty\rangle}.
$$

用矩阵写就是

$$
(Ax)^Ty=x^T(A^Ty).
$$

读法是：**$A$ 把 $x$ 送到输出空间后再和 $y$ 做相似度；等价地，先把 $y$ 用 $A^T$ 拉回输入空间，再和原来的 $x$ 做相似度。**

手算例子：

$$
A=\begin{bmatrix}1&2\\0&1\end{bmatrix},\quad
x=\begin{bmatrix}2\\-1\end{bmatrix},\quad
y=\begin{bmatrix}1\\3\end{bmatrix}.
$$

左边：$Ax=(0,-1)^T$，所以 $(Ax)^Ty=-3$。右边：$A^Ty=(1,5)^T$，所以 $x^T(A^Ty)=2-5=-3$。数字相同不是巧合，而是 transpose 的定义性约束。

### 2.2.1 把“伴随”翻译成人话：先送出去，还是先拉回来

“内积结构”其实只是说：这个空间有一把比较向量的尺子。这里的尺子是普通 dot product：

$$
\langle u,v\rangle=u^Tv.
$$

现在有两条完全等价的路线：

```text
路线 A：x -- A --> Ax -- 与 y 做 inner product --> 一个 scalar
路线 B：y -- A^T --> A^T y -- 与 x 做 inner product --> 同一个 scalar
```

所以，“$A^T$ 是 $A$ 的 adjoint”不是说它神秘地复制了 $A$，而是说：**当我们把 $A$ 从 inner product 的左边搬到右边时，能保证所有比较结果不变的那个 map，就叫 $A$ 的 adjoint。** 在普通 Euclidean 坐标下，这个 map 恰好写成 transpose。

把定义展开，就能看见 transpose 为什么必然出现：

$$
\begin{aligned}
\langle Ax,y\rangle
&=\sum_i\left(\sum_j A_{ij}x_j\right)y_i\\
&=\sum_jx_j\left(\sum_iA_{ij}y_i\right)\\
&=\sum_jx_j(A^Ty)_j
=\langle x,A^Ty\rangle.
\end{aligned}
$$

括号里的那一坨正是 $(A^Ty)_j$：原来按行收集的系数，被重新按列收集回来。

一个很实用的类比是“责任回传”：$A$ 把 input coordinates 混合成 output coordinates；如果 output 方向 $y$ 对结果打了一个分数，$A^Ty$ 就把这份分数沿着原来的连接分配回 input coordinates。neural network 的 backward pass 里，linear layer 的 gradient 之所以反复出现 transpose，就是这个 adjoint 规则在工作。

这里的符号表是：$A\in\mathbb R^{m\times n}$，$x\in\mathbb R^n$，$y\in\mathbb R^m$，因此 $Ax\in\mathbb R^m$、$A^Ty\in\mathbb R^n$，两边最后都是 scalar。

最后再连回 $A^TA$：

$$
\|Ax\|^2=\langle Ax,Ax\rangle=\langle x,A^TAx\rangle.
$$

这句话的意思是：$A^TA$ 在 input space 里记录了“经过 $A$ 以后，哪些方向被拉得更长”。SVD 和 PCA 后面都会回来使用这本账。

### 2.3 第三层：为什么 least squares 总会出现 $A^T$

想让 $Ax$ 尽量接近 $b$，定义误差

$$
L(x)=\|Ax-b\|_2^2.
$$

把误差对 $x$ 的梯度设为零：

$$
\nabla_xL=2A^T(Ax-b)=0
\quad\Longrightarrow\quad
\boxed{A^TAx=A^Tb}.
$$

$A^T$ 在这里不是装饰，而是在把“输出空间里的残差”拉回“输入参数空间”，让每个参数坐标都收到自己的责任报告。

### 2.4 一个常见误区：transpose 不是 inverse

$A^T$ 只是交换行列；只有在 $A$ 是 orthogonal matrix 时，才有 $A^{-1}=A^T$。一般矩阵可能不可逆，或者是长方形矩阵，根本没有普通意义上的 inverse。

## 3. $A^TA$：Gram matrix 是一张“全员相亲表”

### 3.1 手算一张最小的 Gram table

令

$$
A=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}
=\left[\,a_{:1}\;a_{:2}\,\right].
$$

两列分别是 $a_{:1}=(1,1,0)^T$ 与 $a_{:2}=(0,1,1)^T$。于是

$$
A^TA=
\begin{bmatrix}
a_{:1}^Ta_{:1}&a_{:1}^Ta_{:2}\\
a_{:2}^Ta_{:1}&a_{:2}^Ta_{:2}
\end{bmatrix}
=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

表格含义：对角线 $2,2$ 是两列的 squared norm；非对角线 $1$ 是两列的 dot product。若再除以两列 norm 的乘积，就得到 cosine similarity。

### 3.2 为什么它必然 symmetric、positive semidefinite

对称性：

$$
(A^TA)^T=A^TA.
$$

半正定性：对任意 $z$，

$$
z^T(A^TA)z=(Az)^T(Az)=\|Az\|_2^2\ge0.
$$

这条等式是一个强力桥梁：它把一个看起来抽象的矩阵性质，翻译成“某个向量的长度平方”。

### 3.3 $A^TA$ 和 $AA^T$ 各自在比较谁

- $A^TA$：比较 **columns with columns**，常对应 feature-feature similarity；shape 是 $n\times n$。
- $AA^T$：比较 **rows with rows**，常对应 sample-sample similarity；shape 是 $m\times m$。

二者都来自同一个 transpose + multiplication，但问题不同：一个问“哪些 feature 一起出现”，另一个问“哪些 sample 的轨迹相似”。

## 4. PCA：在所有方向中，找方差最大的那一根轴

### 4.1 先把数据摆正

令 $X\in\mathbb R^{N\times D}$：每行是一个 sample，每列是一个 feature。先对每一列减去均值，得到 centered data $X_c$。这一步把云团的中心搬到原点，否则第一主成分可能只是“均值很大”的方向。

协方差矩阵是

$$
\boxed{C=\frac1{N-1}X_c^TX_c}.
$$

所以 covariance matrix 就是 centered data 的 Gram matrix 再除以 $N-1$：它的对角线是每个 feature 的 variance，非对角线是 feature 之间的 covariance。

### 4.2 用“投影后有多散”定义主成分

取单位方向 $w$，把每个 sample 投影到它上面：$X_cw$。投影后的总平方长度是

$$
\|X_cw\|_2^2=w^TX_c^TX_cw.
$$

因此第一主成分是

$$
\max_{\|w\|_2=1}w^TCw.
$$

用 Lagrange multiplier 推导：

$$
\mathcal L(w,\lambda)=w^TCw-\lambda(w^Tw-1),
\qquad
Cw=\lambda w.
$$

所以 **PCA 的方向是 covariance matrix 的 eigenvector，方向上的 variance 是对应 eigenvalue**。

### 4.3 三个点的完整手算

数据点为 $(1,1),(2,2),(3,3)$。均值是 $(2,2)$，中心化后

$$
X_c=\begin{bmatrix}-1&-1\\0&0\\1&1\end{bmatrix},\qquad
C=\frac12X_c^TX_c=\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

矩阵 $C$ 的 eigenvectors 可以选成

$$
w_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},
\quad \lambda_1=2;
\qquad
w_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},
\quad \lambda_2=0.
$$

云团沿 $w_1$ 方向有变化，沿 $w_2$ 方向没有变化，所以 PCA 把二维数据压成一维仍不会丢掉信息。

## 5. SVD：任何矩阵都能拆成“旋转、拉伸、旋转”

### 5.1 分解式与每个符号的工作

对 $A\in\mathbb R^{m\times n}$，

$$
\boxed{A=U\Sigma V^T}.
$$

| 符号 | 含义 | 典型 shape |
|---|---|---|
| $V$ | 输入空间的 orthonormal directions，right singular vectors | $n\times n$ |
| $\Sigma$ | 沿这些方向的 stretching strengths | $m\times n$ |
| $U$ | 输出空间的 orthonormal directions，left singular vectors | $m\times m$ |
| $\sigma_i$ | $\Sigma$ 的非负对角元素，singular values | scalar |

对一个输入 $x$，最右边先做 $V^Tx$，再由 $\Sigma$ 分别拉伸，最后由 $U$ 旋转到输出空间：

$$
x\xrightarrow{V^T}\text{换坐标}\xrightarrow{\Sigma}\text{各轴缩放}\xrightarrow{U}\text{换回输出坐标}.
$$

### 5.2 SVD 与 outer product 的桥

把分解按列拆开：

$$
A=\sum_{i=1}^r\sigma_i u_i v_i^T.
$$

每一项 $u_iv_i^T$ 都是 rank-1 外积；$\sigma_i$ 是这块地毯的音量。按 $\sigma_i$ 从大到小排序，所以保留前 $k$ 项得到最佳 rank-$k$ approximation：

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^T.
$$

### 5.3 SVD 为什么会从 $A^TA$ 长出来

右奇异向量满足

$$
A^TAv_i=\sigma_i^2v_i.
$$

也就是说：$v_i$ 是 Gram matrix $A^TA$ 的 eigenvector，eigenvalue 是 $\sigma_i^2$。然后

$$
u_i=\frac{Av_i}{\sigma_i}\quad(\sigma_i>0).
$$

这就是一条可计算的路线：先看 $A^TA$ 的“方向账本”，再恢复输出方向。

## 6. PCA 与 SVD：同一条主线的两个名字

对 centered data 做 SVD：

$$
X_c=U\Sigma V^T.
$$

代回 covariance：

$$
C=\frac1{N-1}X_c^TX_c
=\frac1{N-1}V\Sigma^2V^T.
$$

因此：

- PCA directions = columns of $V$（centered data 的 right singular vectors）；
- PCA variances = $\sigma_i^2/(N-1)$；
- explained variance ratio = $\sigma_i^2/\sum_j\sigma_j^2$。

这不是两个巧合的算法，而是同一件事的两种入口：PCA 从“找最大投影方差”进入，SVD 从“分解线性 map”进入，最后在 $X_c^TX_c$ 这里汇合。

## 7. PyTorch 最小可运行例子

```python
import torch

# 每行是一个 sample，每列是一个 feature。
X = torch.tensor([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
])

# 1) center：每一列减去自己的均值。
Xc = X - X.mean(dim=0, keepdim=True)

# 2) Gram / covariance：Xc.T @ Xc 再除以 N - 1。
C = Xc.T @ Xc / (Xc.shape[0] - 1)

# 3) covariance 的 eigen-decomposition（对称矩阵用 eigh）。
eigvals, eigvecs = torch.linalg.eigh(C)
order = torch.argsort(eigvals, descending=True)
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# 4) centered data 的 SVD。
U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

print("Xc =\n", Xc)
print("C =\n", C)
print("PCA variances =", eigvals)
print("PCA directions =\n", eigvecs)
print("S^2 / (N-1) =", S.square() / (X.shape[0] - 1))
print("Vh.T =\n", Vh.T)
```

`eigvecs` 与 `Vh.T` 可能差一个符号：如果一个 eigenvector 是 $v$，那么 $-v$ 也是同一个方向。比较方向时看夹角或绝对值，不要把正负号当成算法错误。

## 8. 常见混淆与练习

### 8.1 先自己回答

1. $A^TA$ 的每一格比较的是 rows 还是 columns？
2. 为什么 covariance 前必须先 center？
3. 为什么 $A^TA$ 的 eigenvalue 是 $\sigma_i^2$，而不是 $\sigma_i$？
4. 对于 $A\in\mathbb R^{m\times n}$，$AA^T$ 与 $A^TA$ 分别更像哪两种“相似度账本”？

### 8.2 答案桥

1. $A^TA$ 的第 $(i,j)$ 格是第 $i$ 列与第 $j$ 列的 dot product。
2. 不 center，第一方向会混入均值位置；PCA 要找的是“围绕均值的变化方向”。
3. $A^TAv_i=A^T(\sigma_i u_i)=\sigma_i A^Tu_i=\sigma_i^2v_i$，因为 $A^Tu_i=\sigma_i v_i$。
4. $A^TA$ 比较 columns，$AA^T$ 比较 rows。

## 9. 一句话收拢

**transpose 是 inner product 的 adjoint；$A^TA$ 把 columns 的两两关系记成 Gram matrix；PCA 在这个 Gram/covariance 账本里找最大方差的 eigen-directions；SVD 则把同一条结构拆成按重要性排序的 rank-1 外积，或者旋转 → 拉伸 → 旋转。**

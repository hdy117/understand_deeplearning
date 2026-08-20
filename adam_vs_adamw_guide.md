# Adam 与 AdamW 学习指南

> 衔接：《Understanding Deep Learning》第 6 章（Adam）与第 9 章（L2 / weight decay）  
> 可运行示例：`adam_vs_adamw_demo.py`  
> 第 6 章总笔记：`chapter6_fitting_models_notes.md`

---

## 1. 一句话总览

| | 一句话 |
|--|--------|
| **Adam** | 动量找方向 + 二阶矩按维归一化，每个参数自适应步长 |
| **Adam 的坑** | 若把 L2 塞进梯度，衰减会被 \(1/\sqrt{v}\) 扭曲/冲掉 |
| **AdamW** | 数据梯度仍走 Adam；**weight decay 与自适应更新解耦** |

```text
Adam 思想:   别用同一个 α 硬闯各向异性损失曲面
Adam 问题:   L2 耦合进 g 后，正则语义乱了（另有泛化/冷启动等次要问题）
AdamW 动机:  恢复「按比例缩权重」，且不经过 √v
```

---

## 2. Adam 的思想

### 2.1 要解决什么问题？

固定学习率的困境（书图 6.9）：

```text
损失曲面：竖直很陡，水平很缓

  α 按陡方向设  →  水平方向爬得极慢
  α 按缓方向设  →  竖直方向过冲、抖
```

### 2.2 三块积木

| 积木 | 做什么 | 解决什么 |
|------|--------|----------|
| **一阶矩 \(m\)** | 梯度的指数滑动平均（动量） | 平滑噪声、减少山谷振荡 |
| **二阶矩 \(v\)** | 梯度平方的滑动平均 | 估计「这个参数方向有多陡」 |
| **归一化更新** | \(\Delta \propto m / \sqrt{v}\) | 陡的方向自动小步，缓的方向自动大步 |

```text
g（当前梯度，吵）
        │
        ├─► 平均 → m（方向更稳）
        │
        └─► g² 平均 → v（各维陡度）
                    │
                    ▼
            步长 ≈ α · m / √v
            （大致保留方向，尺度被归一）
```

### 2.3 为什么深度学习爱用它？

1. **各向异性曲率**：不同参数/层梯度量级差很大  
2. 对初始 lr 相对宽容，前期收敛快  
3. 深层网络中能**平衡各层更新**（UDL 6.4）

---

## 3. Adam 有什么问题？

### 3.1 核心问题：L2 与自适应步长耦合（AdamW 的直接动机）

对 **SGD**，L2 与 weight decay 几乎等价：

\[
\phi \leftarrow \phi - \alpha(\nabla L + \lambda\phi)
= (1-\alpha\lambda)\,\phi - \alpha\nabla L
\]

对 **Adam**，若仍把 \(\lambda\phi\) 加进梯度 \(g\)，再做 \(m/\sqrt{v}\)：

\[
\Delta\phi_i^{\text{decay}}
\approx
-\,\alpha\cdot\frac{\lambda\,\phi_i}{\sqrt{v_i}}
=
-\,\underbrace{\Big(\frac{\alpha\lambda}{\sqrt{v_i}}\Big)}_{\text{有效衰减系数}}
\cdot \phi_i
\]

| \(\sqrt{v}\) | 有效衰减 | 典型情况 |
|--------------|----------|----------|
| **大** | **弱**（衰减慢） | 历史梯度猛的参数 |
| **小** | **强**（衰减猛） | 历史梯度弱的参数 |

**注意**：主因不是「参数数值 \(|\phi|\) 大」，而是 **有效衰减 \(\propto 1/\sqrt{v}\)**。  
\(|\phi|\) 大只会间接影响（\(\lambda\phi\) 抬高 \(|g|\) → 抬高 \(v\)）。

```text
错误直觉:  参数大 ──► 衰减进梯度 ──► 衰减慢

正确因果:  λφ 混进 g ──► 被 1/√v 缩放
                         ├── √v 大：学习慢，衰减也慢（正则被打折）
                         └── √v 小：衰减相对过猛
```

### 3.2 其他问题（次要）

| 问题 | 说明 |
|------|------|
| **泛化** | 默认超参下有时不如仔细调的 SGD+momentum（Wilson et al., 2017）；认真搜超参后 Adam 可与 SGD 相当（Choi et al., 2019） |
| **冷启动** | 初期 \(m,v\) 样本少、估计噪 → warmup / RAdam |
| **理论** | 原凸收敛证明有反例 → AMSGrad 等；实践中原版仍常用 |

---

## 4. 为什么需要 AdamW？

**W = Weight decay（解耦的权重衰减）**（Loshchilov & Hutter, 2019）。

书中定位（第 6 章 Notes + 第 9 章）：

> 对 Adam，每个参数有效学习率不同，故 **L2 与 weight decay 不是一回事**。  
> AdamW 正确实现 weight decay 后，带正则时性能明显更好。

```text
Adam + L2（耦合）:

  g = ∇L + λφ  ──►  m,v  ──►  φ -= α m/√v
       ↑
   正则被「自适应」扭曲

AdamW（解耦）:

  g = ∇L  ──►  m,v  ──►  φ -= α m/√v
                          φ -= α λ φ   ← 单独衰减，不进 v
```

| | Adam + L2 | AdamW |
|--|-----------|--------|
| 正则进 \(g\)？ | 是 | 否 |
| 衰减被 \(\sqrt{v}\) 缩放？ | 是 | 否 |
| 各参数衰减强度 | 跟梯度历史耦合 | 近似按固定比例 |
| 调 \(\lambda\) | 难 | 更像「真·权重衰减」 |

现代训练（尤其 Transformer）几乎总要 weight decay → **默认常用 AdamW，不是 Adam 废了，而是带正则时用 AdamW 更干净。**

---

## 5. 完整公式

约定：\(t=1,2,\ldots\)，\(g_t=\nabla_\phi L(\phi_{t-1})\)（默认**只含任务损失**）。  
\(m_0=v_0=0\)。平方、开方、除法均为**逐元素**。

常见默认：\(\beta_1=0.9,\;\beta_2=0.999,\;\epsilon=10^{-8}\)。

### 5.1 标准 Adam（无正则）

\[
\begin{aligned}
g_t &= \nabla_\phi L(\phi_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{\odot 2} \\
\hat m_t &= \frac{m_t}{1-\beta_1^{t}} \\
\hat v_t &= \frac{v_t}{1-\beta_2^{t}} \\
\phi_t &= \phi_{t-1} - \alpha\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
\]

### 5.2 Adam + L2（耦合写法，常被误称为 weight decay）

\[
\begin{aligned}
g_t &= \nabla_\phi L(\phi_{t-1}) + \lambda\,\phi_{t-1} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{\odot 2} \\
\hat m_t &= \frac{m_t}{1-\beta_1^{t}},\quad
\hat v_t = \frac{v_t}{1-\beta_2^{t}} \\
\phi_t &= \phi_{t-1} - \alpha\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
\]

\(\lambda\phi\) 进了 \(g\)，因此也进了 \(m,v\)，衰减被 \(1/\sqrt{\hat v}\) 缩放。

> 损失里写 \(\frac{\lambda}{2}\|\phi\|^2\) 时梯度为 \(\lambda\phi\)；写 \(\lambda\|\phi\|^2\) 时为 \(2\lambda\phi\)。下文统一用「梯度多一项 \(\lambda\phi\)」。

### 5.3 AdamW（解耦 weight decay）

\[
\begin{aligned}
g_t &= \nabla_\phi L(\phi_{t-1}) \qquad\text{（不含衰减）} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{\odot 2} \\
\hat m_t &= \frac{m_t}{1-\beta_1^{t}},\quad
\hat v_t = \frac{v_t}{1-\beta_2^{t}} \\
\phi_t &= \phi_{t-1}
- \alpha\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+ \lambda\,\phi_{t-1}
\right)
\end{aligned}
\]

等价形式：

\[
\phi_t
=
(1-\alpha\lambda)\,\phi_{t-1}
-
\alpha\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

### 5.4 并排对照

| 步骤 | Adam | Adam+L2 | AdamW |
|------|------|---------|-------|
| 梯度 | \(g=\nabla L\) | \(g=\nabla L+\lambda\phi\) | \(g=\nabla L\) |
| \(m,v\) | 由 \(g\) | 由含 \(\lambda\phi\) 的 \(g\) | 只由 \(\nabla L\) |
| bias fix | \(\hat m,\hat v\) | 同左 | 同左 |
| 更新 | \(\phi-\alpha\frac{\hat m}{\sqrt{\hat v}+\epsilon}\) | 同左 | 再额外 \(-\alpha\lambda\phi\) |

### 5.5 伪代码

**Adam**

```text
m ← 0; v ← 0; t ← 0
while not converged:
    t ← t + 1
    g ← ∇L(φ)
    m ← β1 m + (1-β1) g
    v ← β2 v + (1-β2) (g ⊙ g)
    m̂ ← m / (1 - β1^t)
    v̂ ← v / (1 - β2^t)
    φ ← φ - α · m̂ / (√v̂ + ε)
```

**AdamW**

```text
m ← 0; v ← 0; t ← 0
while not converged:
    t ← t + 1
    g ← ∇L(φ)                         # 不含 weight decay
    m ← β1 m + (1-β1) g
    v ← β2 v + (1-β2) (g ⊙ g)
    m̂ ← m / (1 - β1^t)
    v̂ ← v / (1 - β2^t)
    φ ← φ - α · ( m̂/(√v̂+ε) + λ·φ )
```

### 5.6 与 UDL 书中符号对应

| 书中 | 本文 / 论文常用 |
|------|-----------------|
| \(\beta\)（一阶） | \(\beta_1\) |
| \(\gamma\)（二阶） | \(\beta_2\) |
| 式 (6.15)–(6.17) | 标准 Adam |
| AdamW | Notes + §9.1 weight decay |

---

## 6. 数值示例（可手算）

公共设定：

\[
\phi=\begin{bmatrix}1\\1\end{bmatrix},
\quad
g_{\text{data}}=\begin{bmatrix}10.0\\0.1\end{bmatrix},
\quad
\alpha=0.05,\;
\beta_1=0.9,\;
\beta_2=0.999,\;
\lambda=0.2
\]

两参数**起点相同**，数据梯度差 **100 倍**：

- \(\phi_0\)：大梯度（类似尺度大的参数）  
- \(\phi_1\)：小梯度（类似尺度小的参数）

### 6.1 第 1 步：bias correction 后的关键事实

\(m_0=v_0=0\) 时：

\[
\hat m_1 = g,\quad
\frac{\hat m_1}{\sqrt{\hat v_1}}=\mathrm{sign}(g)
\quad\Rightarrow\quad
\text{adam\_step}\approx\alpha\cdot\mathrm{sign}(g)
\]

**Adam 会把不同大小的梯度压成差不多的步长。**

### 6.2 Adam（无正则）

\[
g=\begin{bmatrix}10\\0.1\end{bmatrix}
\Rightarrow
\text{step}=\begin{bmatrix}0.05\\0.05\end{bmatrix}
\Rightarrow
\phi\leftarrow\begin{bmatrix}0.95\\0.95\end{bmatrix}
\]

### 6.3 Adam + L2（耦合）

\[
g=\nabla L+\lambda\phi
=\begin{bmatrix}10.2\\0.3\end{bmatrix}
\]

| 参数 | 数据梯度 | \(\lambda\phi\) | 占比 |
|------|----------|-----------------|------|
| 大梯度维 | 10.0 | 0.2 | **2%**，几乎被淹没 |
| 小梯度维 | 0.1 | 0.2 | **67%**，正则主导 \(g\) |

再归一化后：

\[
\text{step}\approx\begin{bmatrix}0.05\\0.05\end{bmatrix}
\Rightarrow
\phi\leftarrow\begin{bmatrix}0.95\\0.95\end{bmatrix}
\]

与「无正则 Adam」几乎一样 → **正则意图被自适应冲掉**。

### 6.4 AdamW（解耦）

\[
\begin{aligned}
\text{adam\_step}&=\begin{bmatrix}0.05\\0.05\end{bmatrix} \\
\text{decay}&=\alpha\lambda\phi=\begin{bmatrix}0.01\\0.01\end{bmatrix} \\
\phi&\leftarrow\begin{bmatrix}0.94\\0.94\end{bmatrix}
\end{aligned}
\]

### 6.5 第 1 步对照表

| | 用的 \(g\) | adam_step | 额外 decay | 更新后 \(\phi\) |
|--|-----------|-----------|------------|----------------|
| **Adam** | \([10,\,0.1]\) | \([0.05,\,0.05]\) | 无 | \([0.95,\,0.95]\) |
| **Adam+L2** | \([10.2,\,0.3]\) | \([0.05,\,0.05]\) | 无（混在 g 里） | \([0.95,\,0.95]\) |
| **AdamW** | \([10,\,0.1]\) | \([0.05,\,0.05]\) | \([0.01,\,0.01]\) | \([0.94,\,0.94]\) |

### 6.6 场景：数据梯度为 0（只看正则）

\(\alpha=0.1,\;\lambda=0.1,\;g_{\text{data}}=[0,0],\;\phi=[1,1]\)：

```text
Adam+L2:  g=λφ → step≈α·sign(φ)
          1.00 → 0.90 → 0.80 → 0.70 → …   （近似等差往 0）

AdamW:    g=0 → adam_step≈0，只剩 φ←(1-αλ)φ
          1.00 → 0.99 → 0.9801 → 0.9703 → …  （等比衰减）
```

- **Adam+L2**：近似每步减固定量，不是真正的 weight decay  
- **AdamW**：指数式按比例缩小

### 6.7 流程图

```text
              g_data = [10, 0.1]
              φ = [1, 1]

Adam:
  g ──► m,v ──► step≈[0.05,0.05] ──► φ=[0.95,0.95]

Adam+L2:
  g = [10,0.1] + λ[1,1] = [10.2, 0.3]
        │  小梯度维：正则占 67%
        │  大梯度维：正则占 2%
        ▼
  归一化后 step 仍≈[0.05,0.05]  → 正则被冲掉/扭曲

AdamW:
  g = [10, 0.1] ──► adam_step≈[0.05,0.05]
  φ ──────────────► decay = αλ·φ = [0.01,0.01]  （不进 m,v）
                              │
                              ▼
                    φ ← [0.94, 0.94]
```

### 6.8 运行示例脚本

```bash
python adam_vs_adamw_demo.py
```

会打印：第 1 步手算核对、有数据梯度的多步对比、纯正则对比。

---

## 7. 算法选型速查

| 算法 | 思想 | 主要问题 | 何时用 |
|------|------|----------|--------|
| **SGD+Mom** | 噪声 + 动量 | 要认真调 lr/schedule | 最终泛化敏感、算力允许细调 |
| **Adam** | 动量 + 按维归一化 | 与 L2 耦合；默认超参下泛化有时差 | 快速试模型、异质梯度尺度 |
| **AdamW** | Adam + 解耦 weight decay | 仍要调 \(\alpha,\lambda\) 等 | **带权重衰减的主流默认** |

---

## 8. 一页公式盒

\[
\boxed{
\begin{aligned}
\textbf{Adam:}
&\quad
\phi \leftarrow \phi - \alpha\frac{\hat m(\nabla L)}{\sqrt{\hat v(\nabla L)}+\epsilon}
\\[6pt]
\textbf{Adam+L2:}
&\quad
\hat m,\hat v \text{ 用 }\nabla L+\lambda\phi
\quad\Rightarrow\quad
\text{衰减被 }1/\sqrt{v}\text{ 扭曲}
\\[6pt]
\textbf{AdamW:}
&\quad
\phi \leftarrow (1-\alpha\lambda)\phi - \alpha\frac{\hat m(\nabla L)}{\sqrt{\hat v(\nabla L)}+\epsilon}
\end{aligned}
}
\]

---

## 9. 自测

1. 为什么第 1 步里 Adam 对 \(g=[10,0.1]\) 两维 step 都是 \(\alpha\)？  
2. Adam+L2 时，大梯度维和小梯度维上 \(\lambda\phi\) 的占比分别如何？为何最终 step 仍几乎相同？  
3. \(g_{\text{data}}=0\) 时，Adam+L2 与 AdamW 的 \(\phi\) 轨迹有何本质差别？  
4. 「参数大所以衰减慢」这句话哪里不准确？应改成什么？  
5. 实现时：`torch.optim.Adam(weight_decay=…)` 与 `AdamW(weight_decay=…)` 语义差在哪？

---

## 10. 相关文件

| 文件 | 内容 |
|------|------|
| `chapter6_fitting_models_notes.md` | 第 6 章拟合模型总笔记 |
| `adam_vs_adamw_demo.py` | 本指南的可运行数值示例 |
| `gradient_descent.py` | 手工 GD；不同参数用不同有效 lr 的实验 |

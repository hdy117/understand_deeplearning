# 第 9 章：正则化（Regularization）

> 书：《Understanding Deep Learning》Ch.9  
> 前置：第 8 章说明过参数区里许多函数都能把训练损失压到零；test 成绩取决于点与点之间选了哪一条。  
> 本章：训练数据没钉死的地方，偏好从哪里注入？  
> 后续：第 10 章把偏好写进层的结构（局部性与平移）。

---

## 0. 训练损失沉默以后，凭什么选？

第 8 章临走时的局面：插值阈值之后，\(L_D(\phi)=0\) 的 \(\phi\) 常常不止一个。同一份训练点，可以连成平滑曲线，也可以连成剧烈振荡。训练损失已经不能当裁判。

母问题：

> **在数据没有充分约束的地方，学习系统凭什么选择其中一个函数？**

原书把 **regularization** 的狭义说成「给损失加一项，偏好某些 \(\phi\)」；机器学习里这个词更宽：任何缩小 generalization gap 的策略都算。本章按这条从窄到宽走：先写进 objective，再承认 optimizer 自己也在选，最后是一串看起来像工程口诀、其实都在往同一处注入偏好的方法。

调查链：

```text
许多零损失解，train 不再区分它们
        │
        ▼
显式：L_data + λ g(φ)，几何上把最低点挪走
        │
        ▼
MAP：g(φ) 就是 −log prior；Gaussian → L2
        │
        ▼
普通 SGD 上 L2 = 每步乘法收缩；Adam 上必须把衰减解耦（AdamW）
        │
        ▼
隐式：有限步长和 mini-batch 噪声也在选解
        │
        ▼
时间、平均、遮挡、噪声、别人的权重、假数据：偏好进不同管道
        │
        ▼
强度一律在 validation 上选 → 下一章把偏好焊进卷积
```

Regularization 通常用一点 statistical bias 换更少 Variance。过强就 underfit：train 和 validation 一起变差，第一反应不该是再加大 \(\lambda\)。

---

## 1. 显式：把不喜欢的 \(\phi\) 写进标量

无正则时

\[
\hat\phi=\arg\min_\phi\sum_{i=1}^{I}\ell_i(x_i,y_i).
\]

加上一个随「越不喜欢的参数」变大的标量 \(g(\phi)\)：

\[
\hat\phi
=
\arg\min_\phi
\left(
\sum_{i=1}^{I}\ell_i(x_i,y_i)
+
\lambda\,g(\phi)
\right).
\]

\(\lambda>0\) 管两堆东西的相对重量。原书用 Gabor 损失的碗做图：正则项在参数空间中心最低，加进去之后局部最小变少，全局最小的位置也挪了。Regularization 不是训完再修饰，它从第一步梯度就改轨迹。

---

## 2. 概率脸：Gaussian prior 就是 L2

第 5 章的最大似然只问「谁最让已观察的 \(y\) 像会发生」：

\[
\hat\phi_{\mathrm{ML}}
=
\arg\max_\phi
\prod_i p(y_i\mid x_i,\phi).
\]

MAP 多问一句：看数据之前，哪些 \(\phi\) 更合理。后验 \(\propto p(D\mid\phi)p(\phi)\)，取负对数：

\[
-\log p(\phi\mid D)
=
-\sum_i\log p(y_i\mid x_i,\phi)
-\log p(\phi)+C.
\]

于是 **penalty \(\leftrightarrow -\log p(\phi)\)**。\(\lambda\) 的数值还依赖损失是求和还是平均，不能脱离 convention 横比。

零均值各向同性 Gaussian

\[
p(W)\propto\exp\bigl(-\|W\|_2^2/(2\sigma_p^2)\bigr)
\]

给出 \(-\log p(W)=\|W\|_2^2/(2\sigma_p^2)+C\)。Prior 越尖（\(\sigma_p\) 越小），越确信权重该靠近零，正则越强。MAP 仍然只留一个最可能的点；真正的 Bayes 会把整个后验留着，预测时积分——那是后面 heuristics 里的另一档。

最常用的显式项：

\[
L_{\mathrm{reg}}(W)=L_D(W)+\lambda\|W\|_2^2.
\]

梯度 \(\nabla L_D+2\lambda W\)。普通 GD：

\[
W\leftarrow(1-2\alpha\lambda)W-\alpha\nabla L_D.
\]

每一步先按比例缩一点，再按数据梯度走。有人把 penalty 写成 \(\frac{\lambda}{2}\|W\|^2\)，梯度就少那个 2；先对 convention，别让系数偷走半杯咖啡。Bias 通常不衰减：它主要做平移，不直接连向大批输入方向。这是设计习惯，不是定律。

小权重**倾向于**让输入扰动少放大输出（线性模型里 \(\Delta f=W\Delta x\)；深层的 Lipschitz 粗略被各层谱范数乘积上界）。ReLU 有层间缩放对称，欧氏范数也不等于函数复杂度，所以「L2 ⇒ 平滑 ⇒ 泛化」不能当定理。\(\lambda\) 太大，Bias 升，欠拟合。

L0 数非零个数，最稀、最难优化；L1 推稀疏；Elastic net 两者折中。本章主线仍是 L2，因为它和后面的 weight decay 是同一张脸的两面。

---

## 3. Adam 上，L2 不再等于 weight decay

普通 SGD 里，把 \(\lambda\|W\|^2\) 塞进梯度，和每步做

\[
W\leftarrow(1-\alpha\lambda_{wd})W-\alpha\nabla L_D
\]

是同一件事（系数对齐后 \(\lambda_{wd}=2\lambda\)）。Adam 会按坐标用历史二阶量缩放梯度。L2 混进梯度再被这套尺子乘一遍：

\[
W\leftarrow W-\alpha P_t(\nabla L_D+2\lambda W).
\]

衰减项也按「这一维最近有多陡」被扭曲。**AdamW** 把衰减解耦：

\[
W\leftarrow W-\alpha P_t\nabla L_D-\alpha\lambda_{wd}W.
\]

数据梯度继续自适应；收缩按固定比例，不经过 \(P_t\)。项目里的对照：`adam_vs_adamw_guide.md`、`adam_vs_adamw_demo.py`。第 6 章把 Adam 当下山的尺子；本章补一句：尺子也会改你对「小权重」的定义。

---

## 4. 隐式：optimizer 不是中立运输车

objective 一字不改，初始化、步长、batch、算法仍可能停在不同的零损失解上。这叫 **implicit regularization**：不是代码里多了一行永久 penalty，而是离散更新和随机抽样限制了可达轨迹。

连续梯度流 \(\dot\phi=-\nabla L\)。有限步长 GD 的低阶近似相当于在

\[
\widetilde L_{\mathrm{GD}}
\approx
L+\frac{\alpha}{4}\|\nabla L\|^2
\]

上做流（原书 9.8）。更大的 \(\alpha\) 额外惩罚大梯度区。这是局部展开，不是任意学习率下的恒等；太大仍然振荡或飞走。「大学习率泛化更好」是有条件的经验，和第 20 章 batch/学习率比是亲戚。

SGD 再多一项 batch 之间的意见分歧（原书 9.9）：

\[
\widetilde L_{\mathrm{SGD}}
\approx
L+\frac{\alpha}{4}\|\nabla L\|^2
+\frac{\alpha}{4B}\sum_b\|\nabla L_b-\nabla L\|^2.
\]

它偏好「每个 batch 都觉得这里不错」，而不是「平均损失碰巧相同、各 batch 却在打架」。小 batch 有时比全 batch 更能泛化，这是一条解释，不是唯一解释。

第 8 章说零损失解不唯一；本节说下山路径本身就是选择机制。

---

## 5. 偏好还可以从别的管道进

原书把其余方法叫 heuristics。它们并不杂乱：全部是在数据沉默处加约束，只是入口不同。

**Early stopping。** 网络往往先学大尺度重复规律，再拟合细碎噪声。Validation 最好的时刻通常在训练损失还没到底之前。停早，权重大不到、optimizer 也走不远，**effective capacity** 下降，representational capacity 没变。正确协议：用 validation 决定何时停，test 不参与这个决定。

**Ensemble。** 独立训练若干模型再平均（分类可平均概率或投票）。直觉来自第 8 章的 Variance：\(\bar f\) 比单次 \(f_D\) 抖得少。多样性来自不同初始化、不同数据子集（bagging）、不同架构。收益被模型间误差相关 \(\rho\) 卡住：相关接近 1，再加成员也买不了多少。贵，但是最干净的降 Variance 药之一。

**Dropout。** 训练时每个 hidden unit 以概率 \(1-q\) 被置零。Inverted dropout 把留下的激活除以 \(q\)，测试时整网全开、不必再缩放。它强迫表示不能绑死在某一个 unit 上，近似于指数多个子网络的廉价 ensemble。\(q\) 太小，就变成欠拟合。

**噪声。** 输入噪声 ≈ 在训练点周围多钉一些约束（小扰动不该改标签）。权重噪声推模型去更宽的盆地。Label smoothing 把 one-hot 换成正确类 \(1-\varepsilon\)、其余 \(\varepsilon/(K-1)\)，阻止 softmax 把概率顶到 0/1，梯度在「已经对了」之后仍在。假设错了会伤：给猫图加「狗」的平滑，或把医学标签随便平滑，都是在教错误的不变性。

**Bayes。** MAP 取后验众数；完整预测是 \(\int p(y\mid x,\phi)p(\phi\mid D)\,d\phi\)。Dropout 在测试时也开着，有时被当成这种积分的粗糙 MC。深层网的真后验几乎算不出，这一档更多是世界观：不要只信一个 \(\phi\)。

**Transfer / multi-task / self-supervised。** 从别的任务或「预测自己挖掉的部分」带回已经付过钱的先验。预训练权重是一种极强的 \(g(\phi)\)，只是不写成 \(\lambda\|W\|^2\)。

**Augmentation。** 把「哪些变化不该改标签」变成额外训练点：图像的翻转裁剪、语音的变速。这是对数据流形的硬假设。翻转「b」和「d」会毁字符识别；医学影像乱旋转可能把左右病灶对调。Augmentation 不是免费数据，是把 inductive bias 写进抽样过程。

一张地图：

```text
Data          : 更多点、augmentation、transfer
Architecture  : 留给第 10–13 章
Loss          : L2 / L1、label smoothing
Optimizer     : 步长、batch、Adam vs AdamW
Training time : early stopping
Prediction    : ensemble、Bayes 积分
```

---

## 6. 强度在 validation 上选

\(\lambda\)、dropout 率、何时停止、augmentation 的猛度，全是第 8 章的超参数。协议不变：train 拟合，validation 比较，test 只估一次。Train 和 validation 都差，先查优化和容量（Bias / 没训到位），不要把正则当万能旋钮。Train 很好、validation 很差，才是 Variance 主导的典型药方：数据、增强、L2、early stop、ensemble。

第 8 章的三项分解在这里重新到账：正则几乎总是抬一点 Bias、压一点 Variance。没有免费的第三项。

---

## 7. 自测

1. 为什么训练损失为零以后仍需要 regularization？  
2. 狭义（加进 objective）和广义（任何改善泛化的策略）差在哪？  
3. 从零均值 Gaussian prior 写出 L2 penalty。  
4. 普通 GD 下 \(L_D+\lambda\|W\|^2\) 的更新为什么能写成乘法收缩？Bias 通常为什么不衰减？  
5. Adam + L2 和 AdamW 的 \(P_t\) 分别乘在哪一段上？  
6. 有限步长 GD 的修正项 \(\frac{\alpha}{4}\|\nabla L\|^2\) 惩罚的是什么？它为什么不是全局恒等？  
7. SGD 多出来的 batch 分歧项在偏好哪类最低点？  
8. Early stopping 改的是 representational capacity 还是 effective capacity？该看哪份数据决定何时停？  
9. Ensemble 的收益怎样被 \(\rho\) 限制？Dropout 和它是什么关系？  
10. Inverted dropout 的缩放发生在训练还是测试？  
11. 举一个 augmentation 破坏标签不变性的反例。  
12. MAP 和 posterior predictive 的根本差别是什么？  
13. Train、validation 都很差时，为什么不该第一反应加大 \(\lambda\)？  
14. 第 8 章的 inductive bias 在本章分别从哪些管道进系统？下一章准备走哪一条？

---

## 8. 合上书再看一眼

训练损失只约束训练点。过参数区里零损失解成片，数据在点与点之间沉默。Regularization 就是在沉默处加入偏好：可以写成 \(L+\lambda g(\phi)\)，可以是 Gaussian prior 那张脸，可以是每步把权重乘小一点，也可以是 Adam 解耦后的衰减。Optimizer 的步长和 batch 噪声同样在选解。Early stopping、dropout、增强、ensemble 看起来像口诀，入口不同，干的是同一件事。

它们几乎总是用一点 Bias 换更少 Variance；强度只在 validation 上拧。下一章不再往损失里加 \(\lambda\)，而把「附近更重要、同样的图案可以出现在任意位置」直接焊进一层的权重共享里。

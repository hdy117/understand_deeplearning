---
template: doc
title: ControlNet 与 LoRA：给 Diffusion 模型加“手”和“风格”
subtitle: 从空间控制、低秩适配到两者组合的完整教学
file: controlnet-lora-guide
lang: zh
---

# ControlNet 与 LoRA：给 Diffusion 模型加“手”和“风格”

> 这份教程围绕一个问题展开：**我想让生成结果按指定构图走，同时保持某种角色、画风或领域特征，应该把控制信号放在哪里？**
>
> 答案可以先压缩成一句话：**ControlNet 负责“画面里的东西放在哪里”，LoRA 负责“模型倾向于怎样画它”。**

## 0. 先看最终答案：两种能力，两个插口

假设我们有一张房屋的 edge map（边缘图）和一句 prompt：

- edge map：告诉模型墙、门、窗的大致位置；
- prompt：告诉模型 “sunlit wooden cabin beside a lake”；
- style LoRA：告诉模型 “watercolor illustration”。

我们希望得到的不是一张任意的木屋，而是：

1. **空间结构**大体遵守 edge map；
2. **语义内容**遵守 prompt；
3. **笔触和色彩**带有 LoRA 学到的 watercolor 风格。

把这个过程画成一条管线：

~~~flow LR
[Text prompt] -> [Text encoder]
[Control image\nedge / depth / pose] -> [ControlNet]
[Noisy latent z_t] -> [Frozen UNet]
[Text encoder] -> [Frozen UNet]
[ControlNet] -> [Multi-scale residuals]
[Multi-scale residuals] -> [Frozen UNet]
[LoRA adapters\nDelta W = B A] --> [Frozen UNet]
[Frozen UNet] -> [Denoised latent]
[Denoised latent] -> [VAE decoder]
[VAE decoder] -> [Generated image]
~~~

这张图的读法是：

- ControlNet 读取一份额外的 image condition，产生多尺度 residual；
- residual 被送回同一个 diffusion UNet 的中间层；
- LoRA 则挂在 UNet 或 text encoder 的线性层上，改变它们的权重映射；
- 两者最终在同一条 denoising path 上相遇。

最容易混淆的地方也在这里：**ControlNet 是 activation path（激活路径）的旁路，LoRA 是 weight path（权重路径）的增量。** 一个给网络“额外的输入”，一个改网络“怎样处理输入”。

---

## 1. 先把任务拆成三个旋钮

生成一张图时，至少有三个互相独立的旋钮：

| 旋钮 | 它控制什么 | 常见实现 |
|---|---|---|
| prompt condition | 画什么、有哪些语义 | text encoder + cross-attention |
| spatial condition | 物体在哪里、姿态怎样、轮廓怎样 | ControlNet、T2I-Adapter |
| model adaptation | 画风、角色、产品外观、领域分布 | LoRA、full fine-tuning、DreamBooth |

把它们写成一个抽象函数：

$$
\hat{\epsilon}
=
\epsilon_{\theta+\Delta_{\mathrm{LoRA}}}
\left(
z_t,\ t,\ e_{\mathrm{text}};\ r_{\mathrm{ControlNet}}(c,z_t,t)
\right).
$$

这句话的白话是：UNet 预测噪声时，使用已经加上 LoRA 增量的参数；同时，ControlNet 根据 control image $c$ 计算 residual，再把 residual 注入 UNet 的若干层。

符号表：

| 符号 | 含义 | 典型 shape |
|---|---|---|
| $z_t$ | 时间步 $t$ 的 noisy latent | SD 1.5 常见为 [B, 4, H/8, W/8] |
| $t$ | diffusion timestep | 标量或 [B] |
| $e_{\mathrm{text}}$ | text encoder 输出 | SD 1.5 常见为 [B, 77, 768] |
| $c$ | control image 或 control map | 常见为 [B, 3, H, W]，具体通道由模型决定 |
| $r_{\mathrm{ControlNet}}$ | ControlNet 产生的多尺度 residual | 每层 shape 随 UNet 分辨率变化 |
| $\Delta_{\mathrm{LoRA}}$ | LoRA 对原权重的低秩增量 | 与对应权重同 shape |
| $\hat{\epsilon}$ | UNet 的噪声预测 | 与 $z_t$ 同 shape |

这里的 $H,W$ 是像素空间的高和宽，$B$ 是 batch size；它们不是“某个神秘的模型内部尺寸”。UNet 只是先把图像压到 latent space，再在多个分辨率上处理。

### 1.1 一个可验收的统一例子

我们固定一个例子，后文不再随意换题：

~~~text
输入：
  512×512 的房屋照片
  它的 Canny edge map
  prompt = "a small wooden cabin beside a lake, morning light"
  watercolor LoRA

期待：
  edge map 决定门窗和屋顶的大致位置
  prompt 决定“木屋、湖、晨光”
  LoRA 决定水彩纸张、笔触和颜色倾向
~~~

如果输出完全不遵守门窗位置，先查 ControlNet；如果结构正确但不像水彩，先查 LoRA；如果结构和风格都对但语义错，先查 prompt、base model 和 text encoder。

---

## 2. Diffusion 主干：ControlNet 和 LoRA 要插在哪里

### 2.1 从 noisy latent 到 denoised latent

以 latent diffusion 为例，训练时先把真实图像 $x_0$ 编码成 latent $z_0$，再加入噪声：

$$
z_t=\alpha_t z_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

模型学习预测噪声：

$$
\mathcal{L}_{\mathrm{diffusion}}
=
\mathbb{E}_{z_0,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(z_t,t,e_{\mathrm{text}})
\right\|_2^2
\right].
$$

读法：

- 先拿干净 latent $z_0$；
- 按 timestep $t$ 加入已知噪声 $\epsilon$；
- 模型看到 noisy latent、时间和文本；
- 模型要猜出刚才加进去的噪声。

符号表：

| 符号 | 含义 | shape / 单位 |
|---|---|---|
| $x_0$ | 原始图像 | [B, 3, H, W]，像素值通常归一化 |
| $z_0$ | VAE 编码后的 latent | [B, C_z, H/f, W/f] |
| $\alpha_t,\sigma_t$ | scheduler 在时间步 $t$ 的系数 | 标量或可 broadcast 的 [B,1,1,1] |
| $\epsilon$ | 标准 Gaussian noise | 与 $z_0$ 同 shape |
| $\epsilon_\theta$ | denoising network，通常是 UNet | 与 $z_t$ 同 shape |

推理时没有 $z_0$，只有随机 latent $z_T$。scheduler 让模型反复执行“预测噪声 → 去掉一部分噪声”，最后得到 $z_0$ 的估计，再由 VAE decoder 还原图片。

### 2.2 ControlNet 的插口

ControlNet 不直接替换原 UNet。它通常保留一条 locked（冻结）主干，再复制一条 trainable（可训练）分支去读取 control image。分支产生的输出经过 zero convolution（零初始化卷积）后，作为 residual 加回主 UNet。

可以把第 $l$ 个层级写成：

$$
h_{l+1}
=
F_l(h_l,e_{\mathrm{text}})
+
Z_l\left(G_l(h_l,c,t,e_{\mathrm{text}})\right).
$$

- $F_l$：原始 UNet 的 frozen block；
- $G_l$：ControlNet 的 trainable copy 或对应的 control branch；
- $Z_l$：zero convolution；
- $c$：edge、depth、pose、segmentation 等条件；
- 第二项是“额外建议”，不是把原模型整段推倒重来。

为什么初始化为零？

$$
Z_l(\cdot)=0
\quad\Longrightarrow\quad
h_{l+1}=F_l(h_l,e_{\mathrm{text}})
$$

训练刚开始时，ControlNet 的分支对主干输出没有影响，于是模型从一个已经能工作的 base diffusion model 出发，逐渐学会使用新的条件。这里的“零”是输出初始化为零；训练后 $Z_l$ 会学出有意义的卷积核。

### 2.3 LoRA 的插口

对一个线性层：

$$
y = W x,
\qquad
W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
$$

LoRA 不直接更新 $W$，而是使用：

$$
W' = W + \lambda\Delta W,
\qquad
\Delta W=\frac{\alpha}{r}BA.
$$

其中：

$$
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r},
\qquad
r\ll \min(d_{\mathrm{in}},d_{\mathrm{out}}).
$$

读法：

- $W$ 仍然冻结；
- $A$ 先把输入压到 $r$ 维；
- $B$ 再把 $r$ 维映射回输出；
- 只训练 $A,B$；
- $\alpha/r$ 控制增量的标准化尺度。

参数量从 $d_{\mathrm{out}}d_{\mathrm{in}}$ 变成 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$。

例如 $d_{\mathrm{in}}=d_{\mathrm{out}}=1024,r=16$：

- full fine-tuning：$1{,}048{,}576$ 个权重；
- LoRA：$16(1024+1024)=32{,}768$ 个权重；
- LoRA 大约只训练原矩阵的 $3.125\%$。

这里的“低秩”不是说模型只能学到一个小主题，而是说**每个被选中的线性层的权重改变量被限制在低维子空间里**。低秩是参数化约束，不是“能力等级”。

---

## 3. ControlNet：它到底在控制什么

### 3.1 ControlNet 的一句话公式

$$
\boxed{
\text{main UNet feature}
\leftarrow
\text{main UNet feature}
+
\text{control residual}
}
$$

这条式子回答的是“条件从哪里进入网络”。ControlNet 的重点不是生成另一张图，而是用 control image 影响 denoising 过程中的中间 feature。

### 3.2 同一道题走一遍：房屋的 edge map

我们把 512×512 的房屋照片转换为 edge map：

1. **第一步：准备 condition**

   ~~~text
   original image: [1, 3, 512, 512]
   edge map:       [1, 3, 512, 512]
   ~~~

   edge map 主要表达轮廓和边界，不表达“木头的质感”或“晨光的色温”。

2. **第二步：进入 ControlNet**

   ControlNet 用自己的 encoder 处理 edge map，并在不同分辨率产生 residual：

   ~~~text
   control image
       ├─ residual at high resolution
       ├─ residual at middle resolution
       └─ residual at low resolution
   ~~~

   高分辨率 residual 更容易影响边缘和局部位置；低分辨率 residual 更容易影响整体布局。具体层数与 channel 数取决于 base model。

3. **第三步：加回 UNet**

   在 denoising 的每一个 timestep，UNet 都会收到这些 residual：

   ~~~text
   noisy latent + text embedding + ControlNet residuals
                                ↓
                        noise prediction
   ~~~

4. **第四步：修改控制强度**

   如果设置 controlnet_conditioning_scale=0，近似得到没有 ControlNet 的结果；增大它，模型更愿意服从 edge map，但过大可能让画面僵硬、纹理异常或牺牲 prompt 语义。

### 3.3 ControlNet 的训练目标

训练样本通常是：

$$
(x_0,\ c,\ y)
$$

- $x_0$：目标图像；
- $c$：从目标图像生成或标注得到的 control image；
- $y$：文本 caption。

训练过程：

1. 把 $x_0$ 编码成 $z_0$；
2. 随机采样 timestep $t$ 和 noise $\epsilon$；
3. 得到 $z_t$；
4. frozen base model 提供原有能力；
5. trainable ControlNet 读取 $c$，产生 residual；
6. 用预测噪声和真实噪声计算 MSE loss。

$$
\mathcal{L}_{\mathrm{ControlNet}}
=
\mathbb{E}
\left[
\left\|
\epsilon-
\epsilon_{\theta,\phi}(z_t,t,e_y,c)
\right\|_2^2
\right],
$$

其中 $\theta$ 是 base UNet 参数，$\phi$ 是 ControlNet 参数；常见设置是冻结 $\theta$，只更新 $\phi$。

### 3.4 ControlNet 能控制什么，不能控制什么

| control type | 擅长 | 不擅长 |
|---|---|---|
| Canny / edge | 轮廓、边界、几何布局 | 材质、颜色、物体身份 |
| depth | 前后关系、空间结构 | 精确边缘、文字 |
| pose | 人体关键点和姿态 | 手指细节、服装纹理 |
| segmentation | 区域类别和布局 | 类别内部的具体外观 |
| scribble | 粗略构图 | 严格几何测量 |
| lineart | 线稿结构 | 真实光照和材质 |

一个实用判断：

> 如果你的输入能画成一张“空间地图”，ControlNet 往往合适；如果你的输入主要是“我希望模型学会一种外观”，LoRA 往往合适。

### 3.5 ControlNet 的“控制”不是硬约束

ControlNet 产生的是神经网络 residual，不是 CAD solver，也不是像素级 copy-paste。它可以强烈倾向于遵守输入结构，但最终结果仍由 base model、prompt、scheduler、denoising steps、controlnet_conditioning_scale 和随机 seed 共同决定。

所以 edge map 的一扇窗可能变成另一种窗；pose 的手指可能不精确；depth map 的尺度也可能被模型重新解释。把它看成“可学习的软约束”更准确。

---

## 4. LoRA：它到底在改变什么

### 4.1 LoRA 的一句话公式

$$
\boxed{
y=(W+\lambda BA)x
=
Wx+\lambda B(Ax)
}
$$

同一个线性层现在有两条路径：

- 原路径：$Wx$；
- LoRA 路径：先压缩 $Ax$，再还原 $B(Ax)$。

### 4.2 同一道题走一遍：把普通木屋变成水彩木屋

还是同一个 prompt 和同一个 edge map。现在我们加载一个在 watercolor 数据上训练的 LoRA：

1. **第一步：加载 base model**

   base model 提供“木屋、湖、晨光”的一般生成能力。

2. **第二步：挂入 LoRA**

   LoRA 可能被加载到 UNet 的 attention projection，也可能同时加载到 text encoder。不同 checkpoint 的训练目标不同，不能只凭文件名猜它挂在哪里。

   ~~~text
   original Linear:
       y = W x

   with LoRA:
       y = W x + scale * B(A x)
   ~~~

3. **第三步：设置 adapter weight**

   ~~~text
   adapter weight = 0.0  → 接近 base model
   adapter weight = 0.5  → 风格较轻
   adapter weight = 1.0  → 接近训练时强度
   adapter weight > 1.0  → 更强，但更容易过拟合或失真
   ~~~

4. **第四步：和 ControlNet 同时 denoise**

   LoRA 改变 UNet 里面的映射，ControlNet 给 UNet 额外的中间 feature。它们是两条不同的干预路径，所以可以组合。

### 4.3 LoRA 的初始化为什么常把 B 设为零

常见初始化是：

- $A$ 用随机初始化；
- $B$ 初始化为零。

于是训练开始时：

$$
BA=0
\quad\Longrightarrow\quad
W'=W.
$$

这保证 adapter 刚挂上去时，模型行为接近 base model。随着梯度更新 $B$，LoRA 分支逐渐获得影响力。

注意一个细节：如果 $B=0$，第一步对 $A$ 的梯度可能是零，但对 $B$ 的梯度通常不是零；之后 $B$ 变得非零，$A$ 也开始学习。不能把“某个分支初始输出为零”误解成“整个 LoRA 永远学不动”。

### 4.4 LoRA 通常挂在哪些模块

Stable Diffusion 系列里常见的目标包括：

- UNet cross-attention 的 to_q、to_k、to_v、to_out；
- UNet 的其他 linear / projection 层；
- text encoder 的 attention 或 MLP 层；
- 新架构中对应的 transformer block。

没有一个全模型通用的 target_modules 列表。原因很简单：SD 1.5、SDXL、SD3、Flux 的模块命名和 denoiser 结构不同。训练前要打印 named_modules，确认目标层确实存在；推理时要使用和 LoRA checkpoint 同一 base family。

### 4.5 LoRA 适合学习什么

| 目标 | LoRA 的适配度 | 原因 |
|---|---:|---|
| 一种画风或材质 | 高 | 通常可由较小的权重增量表达 |
| 一个角色或产品外观 | 高，但要配合 caption | 需要学习身份特征与触发词 |
| 一个新概念 | 中到高 | 取决于数据数量和概念复杂度 |
| 精确姿态和布局 | 低 | 它没有显式的空间输入 |
| 强几何约束 | 低 | 更适合 ControlNet / T2I-Adapter |
| 完全改变模型能力 | 低 | 可能需要 full fine-tuning 或新架构 |

LoRA 能影响构图，但它没有一张明确的边缘图、深度图或 pose map 可供逐像素参考。因此“用 LoRA 训练一个固定姿势”可能在训练集附近有效，却不等于拥有可控的 pose interface。

---

## 5. ControlNet 和 LoRA 的关键差异

| 维度 | ControlNet | LoRA |
|---|---|---|
| 主要问题 | 如何遵守外部空间条件 | 如何低成本改变模型行为 |
| 干预位置 | activation / feature path | weight path |
| 输入 | edge、depth、pose、segmentation 等 | 通常是 prompt + 训练样本 |
| 训练参数 | 额外 control branch | 每层两个低秩矩阵 |
| base model | 通常需要同一 diffusion family | 必须与 checkpoint family 匹配 |
| 输出文件 | ControlNet 权重 | LoRA adapter 权重 |
| 推理旋钮 | controlnet_conditioning_scale | adapter_weights 或 lora_scale |
| 是否天然控制几何 | 是，软控制 | 否，间接影响 |
| 是否适合风格 | 可以学，但不是主要设计点 | 是 |
| 能否组合 | 可以 | 可以 |

用一句工程化的比喻：

- ControlNet 像给画家一张“施工图”；
- LoRA 像给画家换一套“笔、颜料和审美习惯”。

施工图不能自动决定笔触；换画笔也不能保证门窗落在指定坐标。两者搭配，才是“结构可控 + 风格可换”。

---

## 6. Tensor shape：从输入到 UNet 的一遍追踪

下面以 SD 1.5 风格、512×512 输入为例。具体 channel 数会随 model family 改变，因此重点放在维度角色，不要死背某个实现的层数。

### 6.1 输入和 latent

~~~text
image        [B, 3, 512, 512]
control      [B, 3, 512, 512]      # Canny 常复制为 3 channels
z_0          [B, 4, 64, 64]        # VAE downsample factor 8
z_t          [B, 4, 64, 64]
~~~

其中：

- 3 是 RGB channel；
- 4 是 SD 1.5 常见的 latent channel；
- 64 = 512 / 8；
- B 可以是 1，也可以是多个样本。

### 6.2 Text condition

CLIP text encoder 常输出：

~~~text
prompt ids        [B, 77]
text embeddings   [B, 77, 768]
~~~

77 是该模型常见的最大 token length；并非所有 diffusion model 都是 77。text embedding 通过 cross-attention 影响 UNet feature。

### 6.3 UNet 和 ControlNet 的多尺度 feature

UNet 会在不同空间分辨率上工作：

~~~text
high resolution feature   [B, C1, 64, 64]
middle resolution feature [B, C2, 32, 32]
low resolution feature    [B, C3, 16, 16]
~~~

ControlNet 不需要只产生一个固定大小的 output。它可以产生一组 residual：

~~~text
down_block_residuals = [
    r_0, r_1, ...,
]
mid_block_residual = r_mid
~~~

每个 residual 要和它被加到的 UNet feature shape 对齐。这里正是最常见的 shape bug 来源：

- 控制图的空间尺寸没有对齐；
- 选错了 SD 1.5 / SDXL 的 ControlNet；
- 直接把 RGB tensor 当成已经预处理好的 latent；
- batch 维和 list-of-images 的维度配错。

### 6.4 LoRA 的 shape

对于一个 projection：

~~~text
x:        [..., d_in]
W:        [d_out, d_in]
A:        [r, d_in]
B:        [d_out, r]
Wx:       [..., d_out]
B(Ax):    [..., d_out]
~~~

LoRA 的中间 rank r 很小，所以 adapter 文件通常远小于完整 UNet。文件小不代表它脱离 base model 可以独立生成：LoRA 只保存增量，必须附着在兼容的 base model 上。

---

## 7. 一个最小的 PyTorch LoRA 实现

这段代码不依赖 Diffusers，只展示 LoRA 的数学结构。它把 base linear layer 冻结，只让 A 和 B 接受梯度。

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A minimal LoRA-wrapped linear layer."""

    def __init__(self, d_in, d_out, rank=4, alpha=1.0, bias=True):
        super().__init__()

        # Frozen base weight.
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.weight.requires_grad_(False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)

        # Trainable low-rank factors:
        # A: [rank, d_in], B: [d_out, rank]
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

        self.scale = alpha / rank

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        low_rank = F.linear(F.linear(x, self.A), self.B)
        return base + self.scale * low_rank


torch.manual_seed(0)
layer = LoRALinear(d_in=8, d_out=6, rank=2, alpha=4.0)

x = torch.randn(3, 8)
y = layer(x)

print("x:", x.shape)              # [3, 8]
print("y:", y.shape)              # [3, 6]
print("trainable parameters:",
      sum(p.numel() for p in layer.parameters() if p.requires_grad))
print("expected:",
      2 * (8 + 6))                # rank * (d_in + d_out)
~~~

这段实现的关键是：

$$
\texttt{F.linear(F.linear(x,A),B)}
\quad\Longleftrightarrow\quad
B(Ax).
$$

如果把 B 初始化为零，low_rank 在一开始就是零；如果误把 weight.requires_grad 保持为 True，你就不再是在做纯 LoRA 训练，而是在做“full fine-tuning + LoRA”。

### 7.1 LoRA 参数量的账单

~~~python
def count_parameters(module, trainable_only=False):
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


d_in, d_out, rank = 1024, 1024, 16
full = d_in * d_out
lora = rank * (d_in + d_out)

print("full:", full)
print("lora:", lora)
print("ratio:", lora / full)
~~~

对这个例子：

$$
\frac{r(d_{\mathrm{in}}+d_{\mathrm{out}})}
{d_{\mathrm{in}}d_{\mathrm{out}}}
=
\frac{16(1024+1024)}{1024^2}
\approx 0.03125.
$$

这个比例只说明参数量，不保证每个任务都能达到 full fine-tuning 的效果。rank 太小可能欠拟合，rank 太大则增加显存、文件大小和过拟合风险。

---

## 8. 一个最小的 ControlNet residual 组件

真实 ControlNet 会复制或复用更深的 UNet encoder，并在多个 block 产生 residual。下面的 toy module 只展示最关键的连接方式：control image 经过 encoder，经过 zero convolution，再加到主 feature。

~~~python
import torch
import torch.nn as nn


class TinyControlBranch(nn.Module):
    def __init__(self, control_channels=3, hidden_channels=32):
        super().__init__()

        self.control_encoder = nn.Sequential(
            nn.Conv2d(control_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # Zero convolution: output starts at exactly zero.
        self.zero_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, hidden, control):
        # hidden:  [B, hidden_channels, H, W]
        # control: [B, 3, H, W]
        control_feature = self.control_encoder(control)
        residual = self.zero_conv(control_feature)
        return hidden + residual, residual


torch.manual_seed(0)
branch = TinyControlBranch()
hidden = torch.randn(2, 32, 64, 64)
control = torch.randn(2, 3, 64, 64)

out, residual = branch(hidden, control)

print("residual absolute max at init:", residual.abs().max().item())
print("out - hidden absolute max:", (out - hidden).abs().max().item())
~~~

初始化时两个 max 都应接近 0。训练以后，zero_conv 会不再是零，control feature 才会逐渐影响主干。

这个 toy module 省略了真实实现中的：

- 多尺度 down blocks；
- timestep embedding；
- text cross-attention；
- frozen UNet copy；
- mixed precision；
- classifier-free guidance；
- scheduler。

它的价值是把一条连接单独拎出来看：**ControlNet 的输出是 residual，而不是另一张最终图片。**

---

## 9. Diffusers 推理：ControlNet + LoRA 组合

下面的示例使用 Stable Diffusion 1.5 风格的 Canny ControlNet。ControlNet model ID 和 base model 必须属于兼容 family；LoRA 则通过环境变量指向一个与 base model 匹配的本地目录或 Hub repo。

### 9.1 安装依赖

~~~bash
pip install -U torch diffusers transformers accelerate safetensors pillow opencv-python
~~~

如果你只想先跑通 API，可以把分辨率和 steps 调小；真实生成仍然受显存、模型大小和 scheduler 影响。

### 9.2 完整推理脚本

~~~python
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"

# Optional:
#   set LORA_PATH=path/to/a_compatible_lora
# It may also be a Hub repo id.
LORA_PATH = os.environ.get("LORA_PATH")
LORA_WEIGHT_NAME = os.environ.get("LORA_WEIGHT_NAME")
INPUT_IMAGE = Path("cabin.png")
OUTPUT_IMAGE = Path("cabin_controlnet_lora.png")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=dtype,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=dtype,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

if device == "cuda":
    # Lower peak VRAM than moving the entire pipeline at once.
    pipe.enable_model_cpu_offload()
else:
    pipe.to(device)


# 1. Load and resize the source image.
source = Image.open(INPUT_IMAGE).convert("RGB")
source = source.resize((512, 512))

# 2. Make a Canny control image.
rgb = np.asarray(source)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# ControlNet checkpoints commonly expect a 3-channel image.
control_rgb = np.stack([edges, edges, edges], axis=-1)
control_image = Image.fromarray(control_rgb)


# 3. Optionally attach a LoRA adapter.
if LORA_PATH:
    load_kwargs = {"adapter_name": "style"}
    if LORA_WEIGHT_NAME:
        load_kwargs["weight_name"] = LORA_WEIGHT_NAME

    pipe.load_lora_weights(LORA_PATH, **load_kwargs)
    pipe.set_adapters(["style"], adapter_weights=[0.75])


prompt = (
    "a small wooden cabin beside a lake, morning light, "
    "watercolor illustration, paper texture, detailed brush strokes"
)
negative_prompt = (
    "low quality, blurry, distorted architecture, "
    "extra windows, unreadable text"
)

generator = torch.Generator(device=device).manual_seed(1234)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=control_image,
    controlnet_conditioning_scale=0.85,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
)

result.images[0].save(OUTPUT_IMAGE)
print(f"saved to {OUTPUT_IMAGE}")
~~~

这段代码的控制关系如下：

| 代码参数 | 它控制什么 | 调大后的典型现象 |
|---|---|---|
| controlnet_conditioning_scale | 遵守 control image 的强度 | 结构更贴近 edge，但可能更僵硬 |
| adapter_weights | LoRA 风格/概念增量的强度 | 风格更明显，也可能过饱和或过拟合 |
| guidance_scale | prompt 相对于无条件分支的引导强度 | 文本语义更强，但过大可能失真 |
| num_inference_steps | denoising 的步数 | 通常更慢，质量不一定单调增加 |
| generator.manual_seed | 随机噪声的 seed | 固定后便于比较旋钮变化 |

**不要把这三个 scale 混成一个旋钮。**

- guidance_scale 主要在 classifier-free guidance 内工作；
- controlnet_conditioning_scale 主要缩放 ControlNet residual；
- LoRA adapter weight 主要缩放 $\Delta W$。

不同 pipeline 版本的参数名和默认值可能变化，运行前应对照当前 Diffusers API；文末给出了官方文档链接。

### 9.3 对比实验：一次只改一个变量

为了知道是谁造成了变化，固定：

~~~text
prompt
negative_prompt
control image
seed
scheduler
num_inference_steps
~~~

然后跑一个小网格：

| ControlNet scale | LoRA scale | 用途 |
|---:|---:|---|
| 0.0 | 0.0 | base model 基线 |
| 0.8 | 0.0 | 只看空间控制 |
| 0.0 | 0.75 | 只看 LoRA |
| 0.8 | 0.75 | 看组合结果 |
| 1.2 | 0.75 | 看强控制副作用 |
| 0.8 | 1.2 | 看强风格副作用 |

如果四张图同时换了 seed，就很难判断变化来自哪个模块。实验最怕“旋钮全拧了，然后凭感觉写论文”。

---

## 10. LoRA 的加载、混合与合并

### 10.1 加载一个 LoRA

Diffusers 风格的核心 API 是：

~~~python
pipe.load_lora_weights(
    "path/or/repo",
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="style",
)
pipe.set_adapters(
    ["style"],
    adapter_weights=[0.75],
)
~~~

weight_name 是否需要填写，取决于 checkpoint 仓库中实际的文件名。不要假设所有 LoRA 都叫 pytorch_lora_weights.safetensors。

### 10.2 混合多个 LoRA

如果 base model 和所有 adapter 兼容，可以加载多个 adapter：

~~~python
pipe.load_lora_weights(
    "path/to/style_lora",
    weight_name="style.safetensors",
    adapter_name="style",
)
pipe.load_lora_weights(
    "path/to/character_lora",
    weight_name="character.safetensors",
    adapter_name="character",
)

pipe.set_adapters(
    ["style", "character"],
    adapter_weights=[0.6, 0.8],
)
~~~

概念上，这是：

$$
W'
=
W
+
\lambda_1\Delta W_{\mathrm{style}}
+
\lambda_2\Delta W_{\mathrm{character}}.
$$

但多个 LoRA 不一定线性、互不干扰。它们可能：

- 修改同一批层；
- 争夺同一组 attention pattern；
- 使用不同的 trigger word；
- 来自不同 base model；
- 对颜色、身份或构图产生冲突。

所以 adapter weight 需要用小网格验证，不能把数值相加后当成“能力百分比”。

### 10.3 fuse 和 unload

如果推理阶段不再需要动态切换 LoRA，可以把增量 fuse 到 base weight：

$$
W_{\mathrm{fused}}
=
W+\lambda BA.
$$

这样可减少运行时 adapter 管理开销，但会失去快速开关的便利。典型流程：

~~~python
pipe.fuse_lora(
    adapter_names=["style"],
    lora_scale=0.75,
)
pipe.unload_lora_weights()
~~~

fuse 前应保存原始 pipeline 或 adapter 文件；调试阶段通常保留未 fuse 版本更方便。

---

## 11. 训练 ControlNet 和 LoRA：两条训练路线

### 11.1 训练 ControlNet

训练数据至少需要：

~~~text
target image x_0
control image c
caption y
~~~

一个 batch 的概念流程：

~~~python
# Pseudocode: exact argument names depend on the Diffusers training script.

with torch.no_grad():
    z_0 = vae.encode(target_image).latent_dist.sample()
    noise = torch.randn_like(z_0)
    z_t = scheduler.add_noise(z_0, noise, timestep)
    text_emb = text_encoder(input_ids)[0]

# Only ControlNet parameters are trainable.
down_residuals, mid_residual = controlnet(
    z_t,
    timestep,
    encoder_hidden_states=text_emb,
    controlnet_cond=control_image,
    return_dict=False,
)

noise_pred = unet(
    z_t,
    timestep,
    encoder_hidden_states=text_emb,
    down_block_additional_residuals=down_residuals,
    mid_block_additional_residual=mid_residual,
).sample

loss = torch.nn.functional.mse_loss(noise_pred, noise)
loss.backward()
optimizer.step()
optimizer.zero_grad()
~~~

训练时需要确认：

- base UNet、VAE、text encoder 是否真的冻结；
- control image 是否与 target image 对齐；
- control 预处理是否和推理一致；
- checkpoint 是否保存了正确的 config 和 base model family；
- validation 是否固定 seed 并保存可视化结果。

### 11.2 训练 LoRA

LoRA 的训练数据可以是：

~~~text
target image x_0
caption y
~~~

LoRA 不必有显式 control image。训练时：

1. 冻结 base model；
2. 把 LoRA 注入目标层；
3. 只把 LoRA 参数交给 optimizer；
4. 按原 diffusion objective 训练；
5. 保存 adapter 权重和训练配置。

伪代码：

~~~python
for parameter in base_model.parameters():
    parameter.requires_grad_(False)

trainable_parameters = [
    parameter
    for parameter in base_model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=1e-4,
)

for batch in dataloader:
    with torch.no_grad():
        z_0 = vae.encode(batch["image"]).latent_dist.sample()
        noise = torch.randn_like(z_0)
        z_t = scheduler.add_noise(
            z_0, noise, batch["timestep"]
        )
        text_emb = text_encoder(batch["input_ids"])[0]

    noise_pred = unet(
        z_t,
        batch["timestep"],
        encoder_hidden_states=text_emb,
    ).sample

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
~~~

真正的实现还需要 mixed precision、gradient accumulation、gradient checkpointing、data augmentation、caption / trigger word 设计、checkpoint resume、validation prompt，以及训练和推理分辨率一致性。

### 11.3 两者一起训练是否一定更好

不一定。先问任务是否真的需要“新空间条件”。

| 需求 | 推荐起点 |
|---|---|
| 给现成图片指定姿态或构图，换成某种风格 | 现成 ControlNet + 风格 LoRA |
| 有新型 control signal，现成 ControlNet 不认识 | 训练 custom ControlNet |
| 只想学一种角色、产品或画风 | 先训练 LoRA |
| 既有新 control signal，又有新领域外观 | 分阶段训练，先验证单模块，再组合 |
| 需要精确复现像素或几何 | 考虑 deterministic compositor、inpainting 或专门的几何/渲染方案 |

组合训练会把“空间条件问题”和“外观适配问题”绑在一起，调参更难。教学和工程上都建议先跑：

1. base model；
2. base + ControlNet；
3. base + LoRA；
4. base + ControlNet + LoRA。

每一步都要保存结果，才能知道哪一层引入了问题。

---

## 12. ControlNet + LoRA 的常见坑

### 12.1 把 ControlNet 当成 edge-to-image 网络

ControlNet 不是传统意义上“输入 edge，直接输出 RGB 图”的 generator。它参与每一步 denoising，最终图像仍由整个 diffusion pipeline 生成。只给 edge、不写清 prompt，模型可能知道形状，却不知道材质、光线和语义。

### 12.2 把 LoRA 当成空间控制器

LoRA 可以学习训练数据里常见的构图倾向，但它没有明确的外部坐标输入。要传入当前样本的 pose、depth 或 edge，应使用相应的 ControlNet 或其他 spatial conditioning 方法。

### 12.3 base model family 不匹配

以下组合很危险：

~~~text
SDXL base + SD 1.5 LoRA
SD 1.5 ControlNet + SDXL base
Flux LoRA + Stable Diffusion UNet
~~~

即便代码没有立刻报 shape error，语义也可能完全不对。优先检查 model family、text encoder 类型、denoiser 类型、resolution / VAE、LoRA 的训练说明和 ControlNet 的 model card。

### 12.4 trigger word 没有写进 prompt

有些 LoRA 在训练 caption 中使用了专门的 trigger token。如果不写它，adapter 可能弱；如果写错拼写，模型可能只表现出很轻的变化。trigger word 不是 LoRA 机制本身的必需品，而是训练数据和 checkpoint 约定的一部分。

### 12.5 control image 和 target image 不对应

ControlNet 训练要求 control image 与 target image 处在同一个样本坐标系中：

~~~text
错误：把另一张照片的 edge map 配给当前 target
错误：resize 后裁剪区域不一致
错误：pose map 左右翻转但 target 没翻转
~~~

训练数据一旦错位，模型学到的是“控制信号和目标之间没有可靠关系”。

### 12.6 控制太强，结果像“贴图”

症状：

- 轮廓非常像 control image；
- 纹理破碎；
- prompt 语义被压掉；
- 画面显得僵硬。

处理顺序：

1. 降低 controlnet_conditioning_scale；
2. 检查 control image 是否过密、过噪；
3. 减少不必要的边缘；
4. 固定 seed 比较不同 scale；
5. 再决定是否修改 prompt。

### 12.7 LoRA 太强，风格盖住内容

症状：

- 所有图都长得像训练集；
- 颜色过饱和；
- 角色身份泄漏到无关物体；
- 结构和 prompt 逐渐失去作用。

先降低 adapter weight，再检查训练 caption、数据多样性和 rank。不要第一反应就把 CFG 拉到很高；CFG 是另一条路径。

### 12.8 把参数数量当成最终效果

LoRA 的 rank、ControlNet 的参数量、训练 step 数都不是质量保证。决定效果的还有训练样本是否清洁、caption 是否准确、base model 是否适合、condition 是否可靠、validation 是否覆盖目标使用场景，以及推理时的 scale 是否匹配训练分布。

---

## 13. 排错表：先看症状，再看旋钮

| 症状 | 更可能的原因 | 第一处检查 |
|---|---|---|
| 完全不遵守 edge / pose | ControlNet 没加载、scale 太低、condition 预处理错 | model ID、image、controlnet_conditioning_scale |
| 结构对但风格消失 | LoRA 未加载、路径/文件名错、scale 为 0 | load_lora_weights 返回、adapter 名称 |
| 风格明显但结构跑掉 | ControlNet scale 太低或 LoRA 太强 | 先固定 LoRA，扫描 ControlNet scale |
| 图像发灰、发黑、色彩异常 | dtype、VAE、预处理或 checkpoint 不匹配 | base family、dtype、image range |
| shape mismatch | SD 1.5 / SDXL 混用，或 control batch 错 | 打印所有输入和 residual shape |
| prompt 不听话 | CFG 太低、LoRA 过强、prompt 太长或含冲突词 | 先跑 base baseline |
| 多个 LoRA 互相打架 | 修改同一层、权重过高、训练域冲突 | 每次只开一个 adapter |
| 训练 loss 降但生成无变化 | 没有更新 LoRA/ControlNet 参数，或验证路径没加载新 checkpoint | 检查 requires_grad、optimizer 参数和保存路径 |
| 训练后结构出现伪影 | control-target 对齐错误、zero conv/学习率设置不当 | 可视化训练样本三元组 |

可以在代码里加入几条廉价的检查：

~~~python
def inspect_tensor(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"min={tensor.min().item():.4f}, "
        f"max={tensor.max().item():.4f}"
    )

inspect_tensor("control", torch.from_numpy(control_rgb))
~~~

检查 tensor 不应该只看 shape。dtype、device、数值范围也会决定后面是否出错。

---

## 14. 一个更清晰的实验计划

如果你正在学习，建议按下面的顺序做，而不是一上来同时加载五个 adapter：

### 实验 A：base baseline

~~~text
输入：prompt
输出：base model image
目的：知道模型原本会画什么
~~~

### 实验 B：只加 ControlNet

~~~text
输入：prompt + edge map
输出：结构受控图
目的：确认 condition 真正生效
~~~

### 实验 C：只加 LoRA

~~~text
输入：prompt + LoRA
输出：风格/角色适配图
目的：确认 adapter 真正生效
~~~

### 实验 D：组合

~~~text
输入：prompt + edge map + LoRA
输出：结构和风格同时变化
目的：确认两条干预路径兼容
~~~

### 实验 E：扫描旋钮

固定 seed，改变：

~~~text
ControlNet scale: 0.0, 0.4, 0.8, 1.2
LoRA scale:        0.0, 0.4, 0.8, 1.2
~~~

把结果排成 4×4 网格。这个网格比“我感觉 0.8 差不多”更能帮助你建立 scale 的直觉。

---

## 15. 选择指南：我到底该用哪个

~~~flow TB
[What do you need to change?] -> {Need an external spatial map?}
{Need an external spatial map?} -> [Use ControlNet]: Yes
{Need an external spatial map?} -> {Need a reusable style or concept adapter?}: No
{Need a reusable style or concept adapter?} -> [Use LoRA]: Yes
{Need a reusable style or concept adapter?} -> [Start with prompt and base model]: No
[Use ControlNet] -> {Also need a style or subject change?}
{Also need a style or subject change?} -> [ControlNet + LoRA]: Yes
{Also need a style or subject change?} -> [ControlNet only]: No
[Use LoRA] -> {Need exact per-sample geometry?}
{Need exact per-sample geometry?} -> [Add ControlNet]: Yes
{Need exact per-sample geometry?} -> [LoRA only]: No
~~~

这张图的重点是决策顺序：

1. 先问有没有**每个样本都不同的空间输入**；
2. 再问有没有需要复用的风格、角色或产品概念；
3. 两个答案都为“有”时，组合 ControlNet 和 LoRA。

---

## 16. 最后把两者放进同一个 mental model

把整个系统想象成一位画家：

- base diffusion model：画家的基础能力；
- prompt：客户用语言描述“画什么”；
- ControlNet：客户递来一张施工图，告诉“东西摆在哪里”；
- LoRA：客户给画家一套特定画笔和颜料，告诉“用什么风格画”；
- scheduler：规定每一轮擦掉多少草稿、补上多少细节；
- seed：决定最初那团随机草稿长什么样。

对应到数学：

$$
\begin{aligned}
\text{LoRA:}\quad
&W' = W+\frac{\alpha}{r}BA,\\[4pt]
\text{ControlNet:}\quad
&h_{l+1}=F_l(h_l)+Z_l(G_l(h_l,c)),\\[4pt]
\text{Diffusion:}\quad
&\hat{\epsilon}
=
\epsilon_{W'}(z_t,t,e_{\mathrm{text}};\{r_l\}).
\end{aligned}
$$

三句话读懂：

1. LoRA 改的是 linear map；
2. ControlNet 加的是 multi-scale feature residual；
3. diffusion UNet 把它们都用在每个 denoising timestep 上。

如果只记一条工程经验：

> **先分别证明“结构控制有效”和“adapter 有效”，再组合；先固定 seed 和 prompt，再一次只改一个 scale。**

---

## 17. 参考资料与版本边界

这份教程的机制依据：

- [ControlNet paper: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- [ControlNet official implementation](https://github.com/lllyasviel/ControlNet)
- [Diffusers ControlNet pipeline API](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)
- [LoRA paper: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Diffusers LoRA loading API](https://huggingface.co/docs/diffusers/api/loaders/lora)
- [Diffusers LoRA training guide](https://huggingface.co/docs/diffusers/training/lora)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)

版本边界：

- 不同 diffusers 版本的 pipeline 参数名、默认 scale、LoRA checkpoint 命名可能变化；
- SD 1.5、SDXL、SD3、Flux 的 denoiser、text encoder 和 LoRA 结构并不相同；
- 代码中的 Canny ControlNet ID 是示例，实际使用前要阅读对应 model card 的许可证、预处理方式和兼容 base model；
- 本教程解释机制和实验方法，不替代具体 checkpoint 的使用说明。


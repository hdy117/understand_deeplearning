# Pixel U-Net Flow Matching 设计

- 日期：2026-11-18
- 状态：已实施并通过结构与 40 epoch 实验验证
- 范围：新增完全绕过 VAE 的 64×64 像素空间 Conv2D U-Net Flow Matching 实验

## 背景

此前 latent 生成路径都先由 structural VAE 编码，再把生成 latent 交给 VAE decoder。真实 latent 的重建质量很好，但生成 latent 解码后仍有结构伪影。一个主要怀疑是 structural VAE 是 reconstruction-oriented autoencoder，生成 latent 可能偏离 decoder 熟悉的有效 manifold。

本设计新增一条完全独立的 Pixel U-Net Flow 路径。它不加载 VAE，不计算 latent 统计，也不经过 encode/decode；模型直接接收 `[B,3,64,64]` 图像并预测像素空间 velocity。

## 目标

1. 新增直接的 64×64 像素空间 Conv2D U-Net Flow Matching。
2. 完全不依赖 VAE checkpoint、latent 标准化或 decoder。
3. 使用卷积 Residual Block、down/up sampling、skip connection 和 FiLM 条件。
4. 用与现有数据相同的 CIFAR-10 64×64 图像进行训练。
5. 输出独立的 checkpoint、图片、日志和对比材料。
6. 判断绕开 VAE 后，局部结构伪影是否改善，以及像素空间训练成本是否可接受。

## 非目标

- 不删除或修改现有 Flow、DDPM、VAE 和 U-Net latent 实验。
- 不在第一版加入 DDPM、guidance、reflow、optimal transport 或 adversarial loss。
- 不引入额外数据集或额外图像分辨率。
- 不保证像素空间模型一定优于 latent 模型；实验目的就是验证该假设。
- 不把 Pixel Flow loss 与 latent Flow 或 DDPM epsilon-MSE 直接比较。

## 数据与坐标

数据集现有图像为 `[0,1]` 的三通道图像。训练前映射到对称范围：

$$
\mathbf{x}_1
=
2\mathbf{x}_{\mathrm{image}}-1
\in[-1,1]^3.
$$

Flow source 为标准高斯：

$$
\mathbf{x}_0\sim\mathcal N(0,I).
$$

线性插值路径：

$$
\mathbf{x}_t
=
(1-t)\mathbf{x}_0+t\mathbf{x}_1,
\qquad
\mathbf{u}^*
=
\mathbf{x}_1-\mathbf{x}_0.
$$

训练目标：

$$
\mathcal{L}
=
\left\|
v_\theta(\mathbf{x}_t,t,y)
-
(\mathbf{x}_1-\mathbf{x}_0)
\right\|_2^2.
$$

采样结束后裁剪到 `[-1,1]`，再映射回 `[0,1]` 保存。

## 模型结构

公共接口：

```text
images:    [B,3,64,64]
time:      [B,1]
labels:    [B,10]
velocity:  [B,3,64,64]
```

网络先通过 FiLM 条件 ResBlock，然后进行三级 down/up sampling：

```text
64×64,  32 channels
32×32,  64 channels
16×16, 128 channels
 8×8,  256 channels
```

默认每个 level 使用 2 个 ResBlock。基础 ResBlock 为：

```text
GroupNorm → FiLM → SiLU → Conv3×3
          → GroupNorm → FiLM → SiLU → Conv3×3
          + residual
```

downsample 使用 stride-2 的 3×3 convolution；upsample 使用 nearest interpolation 后接 3×3 convolution。decoder 在每层 concatenate encoder skip。输出 head 是 `Conv2d(32,3,3,padding=1)`。

时间继续使用 sinusoidal embedding + MLP；类别 one-hot 通过 `Linear(10, condition_dim)`。二者拼接后通过 condition MLP，为每个 ResBlock 生成 FiLM scale/shift。

## 默认配置

```text
image_channels = 3
image_size = 64
base_channels = 32
channel_multipliers = (1,2,4,8)
blocks_per_level = 2
condition_dim = 256
category_num = 10
total_steps = 100
sampling_seed = 42
dropout = 0.0
```

模型要求 `image_size` 能被：

$$
2^{(\text{levels}-1)}=8
$$

整除。当前 64×64 满足。

## 训练

默认：

```text
epochs = 80
batch_size = 128
lr = 2e-4
weight_decay = 1e-4
optimizer = AdamW
scheduler = one-shot cosine
precision = CUDA BF16
grad_clip = 1.0
```

训练每个 batch：

1. 从 DataLoader 读取 `[B,3,64,64]`；
2. 把图像从 `[0,1]` 映射到 `[-1,1]`；
3. 采样 `x0 ~ N(0,I)`；
4. 采样每个样本独立的 `t ~ U[0,1]`；
5. 构造 `x_t` 和 velocity target；
6. U-Net 预测 velocity；
7. 计算 MSE、反向传播和 optimizer step。

不计算 VAE latent statistics，不加载 VAE checkpoint。

## 文件边界

计划新增：

- `pixel_flow_unet.py`
  - `PixelUNetFlowMatching`；
  - 直接处理 `[B,3,64,64]`；
  - 复用当前 `FiLMResBlock` 的设计或提取公共实现。
- `train_pixel_flow_unet.py`
  - 独立训练、resume、checkpoint 和采样。
- `config.py`
  - `PixelFlowModelConfig`；
  - `PixelFlowTrainConfig`。
- `tests/test_pixel_flow_unet.py`
  - shape、skip、梯度、BF16、输入校验。

输出目录：

```text
outputs_pixel_flow_unet/pixel_flow_epoch.pt
outputs_pixel_flow_unet/pixel_flow_best.pt
outputs_pixel_flow_unet/pixel_flow_epoch.png
outputs_pixel_flow_unet/pixel_flow_best.png
```

## 错误处理

- 输入必须是 `[B,3,64,64]`。
- 输入图像必须在 `[0,1]`，模型内部负责映射到 `[-1,1]`。
- time 必须是 `[B,1]` 浮点张量，范围为 `[0,1]`。
- label 必须是合法 one-hot。
- downsample/upsample 后的空间形状与 skip 不一致时立即报错。
- checkpoint 空间尺寸或通道不匹配时拒绝 resume。
- 采样出现 NaN/Inf 时立即失败。

## 测试

1. 默认 `[2,3,64,64]` 输入产生同形状 velocity。
2. 四个 level 的 down/up 输出尺寸正确。
3. skip concatenate 的通道数正确。
4. 前向和反向梯度有限。
5. CUDA BF16 smoke test 正常。
6. 错误图像尺寸、time 和 label 被拒绝。
7. checkpoint state dict 可以严格加载。
8. 使用固定 seed 完成一次 10-class、100-step Euler 采样。

## 评估

最终总览图统一包含：

1. 真实图像；
2. VAE-16 structural 重建；
3. Transformer-16 Flow；
4. U-Net-16 Flow；
5. U-Net-16 DDPM；
6. Pixel U-Net Flow。

Pixel Flow 评估重点：

- 多类别视觉结构；
- 同类多样本；
- 像素梯度强度与高频伪影；
- 生成图像的 channel mean/std；
- 训练和采样耗时；
- 是否仍出现 VAE decoder 式的涂抹或流动色块。

因为没有 VAE latent，现有的 latent distribution 指标不适用于该实验。若环境中没有可靠的 FID/Inception 依赖，不以 FID 作为第一版硬门槛，先用视觉、像素统计和训练稳定性判断。

## 风险

- 像素空间维度为 `3×64×64=12,288`，远高于 VAE latent 的 `4×16×16=1,024`，训练明显更难。
- 标准 CFM MSE 直接在像素上训练，可能产生更平均、更软的结果。
- 去掉 VAE 可以消除 decoder manifold mismatch，但不会自动解决 Flow 的 coupling 和 conditional mean 问题。
- 64×64 输入需要更深 U-Net；如果显存或速度不可接受，应调整 batch size，而不是缩小 skip 结构。

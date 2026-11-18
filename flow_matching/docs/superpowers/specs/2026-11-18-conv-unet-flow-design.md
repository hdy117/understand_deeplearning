# Conv2D U-Net Flow Matching 设计

- 日期：2026-11-18
- 状态：待评审
- 范围：新增 16×16 latent 的 Conv2D U-Net Flow 模型、训练入口、checkpoint 和测试

## 背景

现有 16×16 structural VAE 已经能提供明显好于 8×8 VAE 的重建：

```text
8×8 VAE： MSE 0.008471, PSNR 20.721, SSIM 0.635
16×16 VAE：MSE 0.003104, PSNR 25.081, SSIM 0.870
```

对应的 latent 是 `[B,4,16,16]`，flatten 后得到 256 个 Flow token。当前 Transformer Flow 已经能把 generated latent 的均值、方差和相邻空间相关性训练到接近真实值，但最终图像仍有明显结构伪影。主要问题是纯 token Transformer 缺少图像需要的局部卷积、多尺度感受野和 U-Net skip connection。

本设计新增一条独立的 Conv2D U-Net Flow 路径，以现有 `outputs_latent16_structural/vae_best.pt` 为基准，不重新训练 VAE，也不覆盖现有 Transformer Flow。

## 目标

1. 新增一个接收 `[B,256,4]` latent token、输出同形状 velocity 的 Conv2D U-Net。
2. U-Net 内部保持二维 `[B,4,16,16]` 空间结构，不经过 flatten 后再 attention。
3. 使用卷积 ResBlock、down/up sampling 和 skip connection 学习局部及多尺度结构。
4. 时间与类别通过每个 ResBlock 的 FiLM 注入。
5. 新增独立训练入口，默认使用 16×16 structural VAE，不修改现有 Transformer 训练行为。
6. 产物写入独立目录，旧 checkpoint 和对照实验保持完整。

## 非目标

- 不重新训练或修改 VAE loss。
- 不删除现有 Transformer Flow。
- 不在第一版加入 Transformer bottleneck、cross-attention 或 classifier-free guidance。
- 不修改 CIFAR-10 数据、类别定义或图像分辨率。
- 不保证 U-Net 自动消除所有生成伪影；目标是提供更适合二维 latent 的归纳偏置，并用实验验证。

## 模型设计

### 公共接口

`ConvUNetFlowMatching.forward()` 与现有 Flow 模型保持相同接口：

```text
xt:          [B, 256, 4]
time:        [B, 1]
label_batch: [B, 10]
velocity:    [B, 256, 4]
```

模型内部先执行：

```text
[B, 256, 4]
    → transpose
[B, 4, 16, 16]
```

完成 U-Net 前向后，再转回 `[B,256,4]`。这样训练器和 VAE 不需要理解 U-Net 内部布局。

### 条件编码

时间沿用连续 sinusoidal embedding + MLP；类别 one-hot 通过 `Linear(10, cond_dim)` 映射。两个向量拼接后经过 condition MLP，得到每个样本一份全局条件向量：

$$
\mathbf{c}(t,y)
=
\operatorname{MLP}
\left(
[\mathbf{u}(t);\mathbf{e}(y)]
\right).
$$

每个 ResBlock 使用 FiLM：

$$
\operatorname{FiLM}(\mathbf{h};\mathbf{c})
=
(1+\boldsymbol{\gamma}(\mathbf{c}))
\operatorname{GroupNorm}(\mathbf{h})
+\boldsymbol{\beta}(\mathbf{c}).
$$

默认 `cond_dim=256`，每个 block 单独学习自己的 scale 和 shift。

### U-Net 结构

默认 encoder 通道为 `(64,128,256)`，对应空间尺寸：

```text
Level 0: 64 channels,  16×16
Level 1: 128 channels,  8×8
Level 2: 256 channels,  4×4 bottleneck
```

数据流如下：

```text
latent [B,4,16,16]
    → input Conv2d 4→64
    → ResBlock ×2 @16×16
    → Downsample
    → ResBlock ×2 @8×8
    → Downsample
    → Bottleneck ResBlock ×2 @4×4
    → Upsample + concat skip @8×8
    → ResBlock ×2 @8×8
    → Upsample + concat skip @16×16
    → ResBlock ×2 @16×16
    → output Conv2d 64→4
    → [B,4,16,16] → [B,256,4]
```

基础 ResBlock 为：

```text
GroupNorm → SiLU → Conv3×3
          → GroupNorm → SiLU → Conv3×3
          + residual
```

FiLM 条件在每次 GroupNorm 后注入。downsample 使用 stride-2 的 3×3 convolution；upsample 使用 nearest interpolation 后接 3×3 convolution。skip connection 通过 concatenate 加入 decoder。

第一版不加入 attention。目标是先验证卷积 U-Net 的空间归纳偏置是否比 256-token Transformer 更适合当前 latent。

## 配置

新增独立 `UNetFlowConfig`：

```text
latent_channels = 4
latent_size = 16
num_tokens = 256
base_channels = 64
channel_multipliers = (1,2,4)
blocks_per_level = 2
condition_dim = 256
category_num = 10
total_steps = 100
```

U-Net 要求 `latent_size` 能被 `2^(len(channel_multipliers)-1)` 整除。当前 16×16 对三个 level 满足要求。

## 训练入口

新增 `flow_matching_unet.py` 和 `train_flow_matching_unet.py`。

默认训练设置：

```text
VAE checkpoint: outputs_latent16_structural/vae_best.pt
output dir:     outputs_flow_unet_latent16/
batch size:     256
epochs:         80
lr:             2e-4
optimizer:      AdamW
scheduler:      one-shot CosineAnnealingLR
precision:      CUDA BF16 autocast
grad clip:      1.0
```

训练沿用 VAE posterior mean、channel 标准化、线性 Flow 插值和 MSE velocity target。checkpoint 保存：

```text
model_type: conv2d_unet_flow
unet_config
vae_config
latent_mean
latent_std
model
optimizer
scheduler
next_epoch
best_loss
```

输出文件：

```text
outputs_flow_unet_latent16/unet_flow_epoch.pt
outputs_flow_unet_latent16/unet_flow_best.pt
outputs_flow_unet_latent16/unet_flow_epoch.png
outputs_flow_unet_latent16/unet_flow_best.png
```

## 错误处理

- 输入 token 数必须等于 `latent_size²`。
- latent channel 必须与配置一致。
- `time` 必须是 `[B,1]` 浮点张量，范围为 `[0,1]`。
- `label_batch` 必须是合法 one-hot。
- VAE 的 `vae_config` 必须产生配置声明的 latent 尺寸。
- checkpoint 的模型类型不匹配时拒绝加载。
- U-Net 空间下采样与上采样后的形状不一致时立即报错。

## 测试

1. token → image latent → token 的形状往返测试。
2. 三个 level 的 down/up 输出尺寸正确。
3. skip connection 拼接后的通道数正确。
4. `[B,256,4]` 输入产生同形状 velocity。
5. 前向和反向梯度有限，BF16 smoke test 正常。
6. 非法 token 数、非法 time 和非法 label 被拒绝。
7. 用现有 16×16 VAE 完成一次 10-class 的 100-step Euler 采样。

## 验收

以现有 16×16 Transformer Flow 为基线：

- generated latent channel mean error 不高于当前约 `0.031`；
- std ratio 继续接近 1；
- generated round-trip RMSE 接近真实 latent；
- 在相同 seed 和类别条件下，生成图的结构伪影减少；
- 训练过程稳定，不出现 NaN、checkpoint 不兼容或采样形状错误。

最终对比图需要同时包含真实图像、VAE 重建、16×16 Transformer Flow 和 16×16 U-Net Flow。

## 风险

- Conv U-Net 参数和显存占用可能高于当前 Transformer，需要根据实际 GPU 显存调整 batch size。
- 16×16 latent 分辨率较短，U-Net 只能下采样两次；bottleneck 为 4×4，不能继续下采样。
- FiLM 条件可能仍不足以表示复杂的 time-class 交互；第一版先用简单稳定的实现。
- U-Net 改善的是模型归纳偏置，不保证单独解决所有 Flow manifold 误差。如果仍失败，下一步再评估 minibatch OT、source whitening 或辅助 endpoint loss。

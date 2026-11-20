# Pixel U-Net DDPM 实验设计

- 日期：2026-11-18
- 状态：待评审
- 范围：在 64×64 像素空间复用 Pixel U-Net，新增独立的 ε-prediction DDPM 实验

## 背景

Pixel U-Net Flow 已经证明：完全绕开 VAE 可以避免 latent decoder manifold，但直接学习 12,288 维像素空间 velocity 仍然困难。40 epoch 实验能匹配真实图像的低层 mean/std/梯度，却没有获得稳定的对象结构。

为了区分 Flow Matching 目标与像素空间生成难度，本设计新增 Pixel U-Net DDPM。它使用相同的 64×64 U-Net、数据、类别条件和训练预算，只把 velocity field 改为 epsilon noise predictor，并使用 ancestral diffusion sampling。

## 目标

1. 新增完全无 VAE 的 Pixel U-Net DDPM。
2. 复用 Pixel Flow 的 64×64 Conv U-Net backbone 和 617 万参数结构。
3. 使用 1000-step linear DDPM 和 ε-prediction。
4. 与 Pixel U-Net Flow 使用相同数据、batch、优化器和 40 epoch 预算。
5. 比较两者的多类别生成图、多样本、低层像素统计和采样成本。
6. 不覆盖现有 Pixel Flow、Transformer latent Flow 或 VAE 产物。

## 非目标

- 不修改 Pixel U-Net backbone。
- 不重新引入 VAE。
- 不在第一版实现 DDIM、score-based SDE、v-prediction、min-SNR 或 guidance。
- 不比较 DDPM epsilon-MSE 与 Flow velocity-MSE。
- 不保证 DDPM 一定优于 Pixel Flow。

## 数据与坐标

训练图片先映射到：

$$
\mathbf{x}_0=2\mathbf{x}_{\mathrm{image}}-1\in[-1,1].
$$

DDPM 使用：

```text
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
schedule = linear
prediction = epsilon
```

前向扩散：

$$
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

训练目标：

$$
\mathcal{L}
=
\left\|
\epsilon_\theta(\mathbf{x}_t,t,y)-\epsilon
\right\|_2^2.
$$

离散 timestep 进入 Pixel U-Net 前映射到 `[0,1]`：

$$
t_{\mathrm{input}}=\frac{t}{T-1}.
$$

## 模型

复用现有 `PixelUNetFlowMatching` 网络主体：

```text
input/output: [B,3,64,64]
channels:     32 / 64 / 128 / 256
downsample:   64 → 32 → 16 → 8
blocks:       2 per level
condition:    time/class concat + MLP + FiLM
parameters:   6,171,011
```

新增包装层 `PixelDDPMUNet`：

```text
predict_noise(x_t, timestep, label) -> epsilon
q_sample(x_0, timestep, noise) -> x_t
sample(labels, generator) -> x_0
```

Pixel Flow backbone 的输入接口是连续时间 `[0,1]`，因此 DDPM wrapper 负责把离散 timestep 归一化后传入。

## 反向采样

从 `x_999 ~ N(0,I)` 开始：

$$
\boldsymbol{\mu}_t
=
\frac{
\mathbf{x}_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(\mathbf{x}_t,t,y)
}{
\sqrt{\alpha_t}
},
$$

$$
\mathbf{x}_{t-1}
=
\boldsymbol{\mu}_t
+
\sqrt{\beta_t}\mathbf{z}.
$$

`t=0` 不加噪声。最终裁剪到 `[-1,1]`，再映射到 `[0,1]` 保存。

采样使用独立 `torch.Generator`，固定 seed 可复现。

## 训练

与 Pixel Flow 的 40 epoch 实验对齐：

```text
epochs: 40
batch_size: 256
optimizer: AdamW
lr: 2e-4
weight_decay: 1e-4
precision: CUDA BF16
grad_clip: 1.0
```

由于 1000-step 采样成本高：

- 每个 epoch 记录 training epsilon-MSE；
- 每 5 epoch 保存一次预览图；
- best checkpoint 改善时立即保存；
- 最后 epoch 保存最终图。

## 文件边界

计划新增：

- `pixel_ddpm.py`
  - schedule buffers；
  - `PixelDDPMUNet`；
  - `q_sample()`；
  - ancestral reverse sampling。
- `train_pixel_ddpm.py`
  - 独立训练、resume、checkpoint 和采样。
- `config.py`
  - 复用 `PixelFlowModelConfig` 和 `DDPMConfig`；
  - 新增 `PixelDDPMTrainConfig`。
- `tests/test_pixel_ddpm.py`
  - schedule、q_sample、shape、梯度、固定 seed、BF16。

输出：

```text
outputs_pixel_ddpm_unet/pixel_ddpm_epoch.pt
outputs_pixel_ddpm_unet/pixel_ddpm_best.pt
outputs_pixel_ddpm_unet/pixel_ddpm_epoch.png
outputs_pixel_ddpm_unet/pixel_ddpm_best.png
outputs_pixel_ddpm_unet/pixel_ddpm_samples64.png
```

## 错误处理

- 输入必须是 `[B,3,64,64]`。
- timestep 必须是 `[B]` 整数 tensor，范围 `[0,999]`。
- label 必须是合法 one-hot。
- Pixel backbone spatial/channel 配置不匹配时拒绝运行。
- checkpoint model type 不匹配时拒绝 resume。
- 采样出现 NaN/Inf 时立即失败。

## 测试

1. schedule 长度和 `alpha_bar` 单调性正确。
2. `q_sample()` 闭式公式正确。
3. Pixel DDPM noise prediction 输出 `[B,3,64,64]`。
4. 前向和反向梯度有限。
5. CUDA BF16 forward 正常。
6. 固定 seed 的 reverse sample 可复现。
7. 非法 timestep、图像 shape 和 label 被拒绝。

## 评估

在总览图中增加 Pixel DDPM 行，与：

1. 真实图像；
2. VAE-16 reconstruction；
3. Transformer-16 Flow；
4. U-Net-16 Flow；
5. U-Net-16 DDPM；
6. Pixel U-Net Flow；
7. Pixel U-Net DDPM

对比。

指标：

- 多类别视觉结构；
- 64 张同类多样本；
- 图像 mean/std；
- 梯度强度；
- 训练和采样耗时；
- 是否改善 Pixel Flow 的语义幻觉。

## 风险

- 64×64 DDPM 1000-step 采样比 Pixel Flow 100-step 慢约一个数量级。
- 40 epoch 对 pixel DDPM 可能明显不足。
- 标准 CFM/DDPM MSE 都可能学习条件均值，像素空间仍可能产生“照片质感但语义错误”的结果。
- 本轮目标是判断方向和成本，不以 FID 作为硬门槛。

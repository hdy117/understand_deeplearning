# Conv U-Net DDPM 实验设计

- 日期：2026-11-18
- 状态：待评审
- 范围：在现有 16×16 structural VAE 上新增独立的 ε-prediction DDPM 训练与采样实验

## 背景

当前 16×16 structural VAE 重建已经明显优于 8×8 VAE。使用同一个 Conv2D U-Net backbone 的 Flow Matching 也比 256-token Transformer 有更低的训练 loss 和更接近真实的 latent 分布，但最终图像仍有明显结构伪影。

为了判断问题是否来自 Flow Matching 的训练目标、noise-data coupling 或确定性 ODE 采样，本设计新增独立的 DDPM 实验。DDPM 复用相同的 VAE、U-Net backbone、数据、seed、batch size 和训练 budget，只将 velocity field 改为噪声预测器，并将 deterministic Euler 换成 1000-step ancestral sampling。

## 目标

1. 新增标准 ε-prediction DDPM，固定 1000 个线性 diffusion timesteps。
2. 复用现有 Conv U-Net backbone，保持约 595 万参数量。
3. 使用现有 `outputs_latent16_structural/vae_best.pt` 和相同 channel 标准化。
4. 新增独立训练入口、checkpoint 类型、采样和 latent 分析。
5. 与 Transformer-16、Conv U-Net Flow-16 做同 VAE、同测试集的生成效果与成本对比。
6. 不覆盖任何已有 Flow 或 VAE 产物。

## 非目标

- 不修改 16×16 structural VAE。
- 不训练独立的 VAE 或改变 latent 标准化方式。
- 不在第一版实现 DDIM、score-based SDE、classifier-free guidance 或 v-prediction。
- 不与 DDPM 的另一套更大 backbone 比较；第一版必须和 U-Net Flow 使用相同网络。
- 不要求 DDPM 一定生成更清晰的图像；实验目标是提供公平对照。

## DDPM 定义

默认设置：

```text
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02
schedule = linear
prediction_type = epsilon
sampling_seed = 42
```

根据 `beta_t` 定义：

$$
\alpha_t=1-\beta_t,
\qquad
\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s.
$$

### 前向扩散

给定标准化后的 VAE latent $\mathbf{x}_0$ 和高斯噪声 $\epsilon$：

$$
\mathbf{x}_t
=
\sqrt{\bar{\alpha}_t}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

网络输入 `[B,256,4]` token、类别 one-hot 和离散 timestep，输出同形状噪声预测。离散 timestep 在进入现有 U-Net 前使用：

$$
t_{\mathrm{input}}
=
\frac{t}{T-1}\in[0,1].
$$

### 训练目标

$$
\mathcal{L}_{\mathrm{DDPM}}
=
\mathbb{E}_{t,\mathbf{x}_0,\epsilon}
\left[
\left\|
\epsilon_\theta(\mathbf{x}_t,t,y)-\epsilon
\right\|_2^2
\right].
$$

### 反向采样

从 $\mathbf{x}_{999}\sim\mathcal N(0,I)$ 开始，对 $t=999,\ldots,0$：

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
\sqrt{\beta_t}\mathbf{z},
\qquad
\mathbf{z}\sim\mathcal N(0,I),
$$

当 $t=0$ 时不添加最后的随机噪声。采样使用独立 `torch.Generator`，确保给定 seed 可复现。

## Backbone 复用

现有 `ConvUNetFlowMatching` 已包含输入 Conv、FiLM ResBlock、down/up sampling、skip connection、time embedding 和 class embedding。DDPM 通过组合方式复用它：

```text
DDPMUNet
  └── backbone: ConvUNetFlowMatching
```

DDPM 的 forward 将 `t/(T-1)` 作为 backbone 的连续时间输入，backbone 输出的四通道 tensor 被视为预测噪声。Flow 类中的 `get_xt_and_xt_dot()` 不属于 DDPM 路径。

为了避免语义混淆，DDPM 的公共接口单独定义：

```text
predict_noise(x_t, timestep, label_batch) -> epsilon
q_sample(x_0, timestep, noise) -> x_t
sample(labels, generator) -> x_0
```

## 配置与训练

新增：

```text
DDPMConfig:
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    sampling_seed = 42

DDPMTrainConfig:
    epochs = 80
    batch_size = 256
    lr = 2e-4
    weight_decay = 1e-4
    amp = True
    output_dir = outputs_ddpm_unet_latent16/
    vae_checkpoint = outputs_latent16_structural/vae_best.pt
```

训练与 U-Net Flow 保持相同：

- CIFAR-10 50,000 张训练图片；
- VAE posterior mean；
- 每个 channel 使用训练集统计量标准化；
- AdamW；
- one-shot cosine scheduler；
- BF16 autocast；
- gradient clipping 1.0；
- checkpoint 每 epoch 保存；
- best checkpoint 在训练后半段按平均 loss 选择。

DDPM timestep 均匀采样，不加入重要性采样。

## 文件边界

计划新增：

- `ddpm.py`
  - schedule buffers；
  - `q_sample()`；
  - noise prediction；
  - ancestral reverse sampling。
- `train_ddpm.py`
  - 独立训练、resume、checkpoint 和采样。
- `config.py`
  - `DDPMConfig`、`DDPMTrainConfig`。
- `tests/test_ddpm.py`
  - schedule、前向扩散、反向采样、shape、梯度和 BF16 测试。
- `analyze_latent_distribution.py`
  - 支持 `model_type="ddpm_conv2d_unet"`，使用 DDPM sampler 而非 Euler。

输出：

```text
outputs_ddpm_unet_latent16/ddpm_epoch.pt
outputs_ddpm_unet_latent16/ddpm_best.pt
outputs_ddpm_unet_latent16/ddpm_epoch.png
outputs_ddpm_unet_latent16/ddpm_best.png
outputs_ddpm_unet_latent16/latent_distribution.json
outputs_ddpm_unet_latent16/latent_distribution.md
```

## 错误处理

- timestep 必须是整数 tensor，形状 `[B]`，范围 `[0,T-1]`。
- latent token 必须是 `[B,256,4]`。
- 类别标签必须是合法 one-hot。
- schedule 张量长度必须为 `T`，且满足 `ᾱ_t` 单调递减。
- checkpoint model type 不匹配时拒绝加载。
- 采样过程中检测 NaN/Inf 并立即失败。
- VAE latent size/channel 与 DDPM backbone 不一致时拒绝训练。

## 测试与验收

### 正确性

1. `beta`、`alpha`、`alpha_bar` 形状和数值范围正确。
2. `alpha_bar` 严格单调递减。
3. `q_sample()` 输出形状正确，固定 `x0/epsilon/t` 时结果确定。
4. reverse sample 使用固定 seed 时可复现。
5. U-Net noise prediction 前向和反向梯度有限。
6. BF16 forward 正常。
7. 非法 timestep、token shape 和 label 被拒绝。

### 模型对比

DDPM 与 U-Net Flow 使用相同 VAE、U-Net、优化器、batch 和训练 epoch。比较：

- generated latent mean error；
- channel std ratio；
- outside real 3σ；
- generated round-trip RMSE；
- 每类多样本生成图和总览对比；
- 100 张图的采样耗时；
- 是否出现模式坍缩、网格伪影或高对比错误纹理。

DDPM 的 epsilon-MSE 与 Flow 的 velocity-MSE 不直接比较。

## 风险

- 1000-step ancestral sampling 比 Flow 的 100-step Euler 慢约一个数量级。
- 共享 U-Net 不一定是 DDPM 最优架构，但这个选择保证与 U-Net Flow 的可解释公平对比。
- DDPM 的随机反向路径会带来样本间较大波动，必须看多样本和 latent 指标。
- linear schedule 是标准 baseline；如果训练不稳定，后续单独实验 cosine schedule，而不是在第一版混入多个变量。

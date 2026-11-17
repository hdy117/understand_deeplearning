# VAE 16×16 Latent 升级设计

- 日期：2026-11-17
- 状态：待评审
- 范围：`flow_matching/` 下的 VAE、Flow Matching、训练入口、分析工具和 checkpoint

## 背景

当前 VAE 把 `64×64×3` 图像压缩成 `4×8×8` latent，只有 256 个 latent 数值。Flow Matching 随后在这个 8×8 网格上生成 64 个 token，VAE decoder 再把低分辨率 latent 还原成像素。现有实验已经表明，Flow generated latent 的相邻空间相关性和通道尺度可以接近真实 VAE latent，但最终图像仍然很软。这说明继续给 Flow 增加条件能力，收益会受 VAE 的空间瓶颈限制。

本设计的目标不是提高 latent channel 数，而是降低空间压缩倍率。默认路径从 8 倍下采样改为 4 倍下采样，得到 `[B,4,16,16]` latent，对应 256 个 Flow token。

## 目标

1. VAE 能从 `64×64` 图像产生 `[B,4,16,16]` latent，并正确重建 `64×64` 图像。
2. VAE encoder 和 decoder 的下采样/上采样阶段由同一个 `downsample_factor` 推导，不再依赖固定三次 stride-2。
3. Flow Matching 的 token 数量和位置编码从 VAE latent grid 推导，避免在代码中散落硬编码的 64。
4. 先完成 VAE-only A/B，确认 `16×16` latent 的重建质量优于当前 `8×8`，再训练 Flow。
5. 旧 VAE、Flow checkpoint 和实验图片保持可追溯，不覆盖唯一副本。

## 非目标

- 不把图像分辨率从 `64×64` 提升到更高像素分辨率。
- 不在这一阶段增加 latent channel、修改 CIFAR-10 数据或分类标签格式。
- 不引入 U-Net Flow backbone、cross-attention、classifier-free guidance 或新的 ODE solver。
- 不保证 `16×16` latent 自动得到清晰的生成图；该实验只先解除 VAE 的空间瓶颈。

## 方案

### VAE 结构

保留四个 hidden stage 和四个 latent channel，把 encoder 的 stage stride 显式定义为：

```text
downsample_factor=8: [1, 2, 2, 2]
downsample_factor=4: [1, 2, 2, 1]
```

编码空间因此分别是：

```text
64 -> 64 -> 32 -> 16 -> 16  (factor=4)
64 -> 64 -> 32 -> 16 -> 8   (factor=8)
```

decoder 使用 encoder stage 的逆序构造。某个 encoder stage 的 stride 为 2 时，对应 decoder stage 增加一次 nearest upsample；stride 为 1 时保持空间大小。这样 latent 尺寸完全由 `VAEConfig.downsample_factor` 决定，并且 `VAE.latent_size` 与实际卷积输出保持一致。

### Flow token 接口

`FlowMatchingModelConfig` 增加 `num_tokens`，默认值为 `16×16=256`。训练器在构造 Flow 前验证：

```text
num_tokens == (img_size / vae.downsample_factor)^2
```

`SinusoidalPositionEmbedding` 使用 `num_tokens`，不再写死 64。`VAE.encode_tokens()` 和 `decode_tokens()` 继续使用统一接口：

```text
[B, num_tokens, 4] <-> [B, 4, latent_size, latent_size]
```

分析脚本、采样器和 checkpoint metadata 都从 VAE 的 `latent_size` 与 Flow 配置的 `num_tokens` 读取尺寸。

### 实验隔离

VAE-only 实验写入独立输出目录，例如：

```text
outputs_latent16/vae_best.pt
outputs_latent16/best_reconstruction.png
```

当前 8×8 产物继续保留在原目录和 `legacy_*` 归档目录。只有 VAE-only A/B 达标后，Flow 才使用新 VAE checkpoint 启动训练。

## 备选方案

### 提高 latent channels

把 `latent_channels` 从 4 增加到 8 或 16，同时保持 8×8 网格。这个方案不会增加 Flow token 数，计算成本最低，但每个 token 仍对应原图 8×8 区域，空间细节上限没有改变。只适合作为次要实验。

### 2 倍下采样

把 latent 改成 `[B,4,32,32]`，即 1024 个 token。空间质量潜力更高，但 self-attention 相对当前 64 token 增加约 256 倍，训练和显存成本过高。当前阶段不采用。

### 4 倍下采样

把 latent 改成 `[B,4,16,16]`，即 256 个 token。attention 成本相对当前约 16 倍，FFN 和 VAE 卷积成本约 4 倍；在 16 GB GPU 上仍可训练。这是当前收益和成本最平衡的方案。

## 流程

```text
1. VAE stride/refactor + shape tests
             ↓
2. VAE-only 8×8 vs 16×16 重建 A/B
             ↓
3. 16×16 达标后更新 Flow token/position metadata
             ↓
4. 用新 VAE latent 训练 Flow Matching
             ↓
5. 比较真实图像 / VAE 重建 / Flow 生成
```

## 错误处理与兼容性

- `downsample_factor` 必须是 2 的非负整数次幂，且不能超过 VAE stage 能提供的下采样次数。
- VAE forward 发现实际 latent 空间尺寸与 `latent_size` 不一致时立即报错。
- Flow 构造时发现 `num_tokens` 与 VAE grid 不一致时立即报错。
- checkpoint 写入 `vae_config`、`image_size`、`latent_size`、`num_tokens` 和 Flow 配置。
- 新 checkpoint 不尝试加载旧 VAE 或旧 Flow 权重；旧产物只用于评估和对照。

## 测试与验收

### 结构测试

1. `downsample_factor=8` 产生 `[B,4,8,8]`，token 数 64。
2. `downsample_factor=4` 产生 `[B,4,16,16]`，token 数 256。
3. encoder/decoder 对两种因子都完成 `[B,3,64,64] -> latent -> [B,3,64,64]` round-trip。
4. Flow 在 256 token 上完成前向、反向和一次 Euler 采样。
5. `decode_tokens(encode_tokens(image))` 与直接 `decode(encode(image))` 一致。

### VAE A/B

使用相同数据、优化器、epoch 和随机种子，分别训练当前 8×8 与提议的 16×16 VAE。至少比较：

- validation reconstruction MSE；
- 同批次原图与重建图；
- latent round-trip RMSE；
- 训练耗时和峰值显存。

### Flow 验收

VAE A/B 通过后，再比较真实 latent 与 Flow generated latent 的 mean error、std ratio、相邻空间相关性和 round-trip RMSE。最终人工检查每个类别的生成图，重点确认轮廓细节是否比 8×8 版本更多，而不是只观察颜色块。

## 风险

- 16×16 latent 会把 Flow attention 成本提高约 16 倍，80 epoch 训练可能明显变慢。
- 更少下采样会提高 VAE decoder 的局部表达能力，但像素级 MSE 仍可能产生软边缘；若 A/B 仍模糊，下一阶段才考虑 perceptual/L1 损失。
- 当前仓库已经包含未提交的 AdaLN、标准化和训练脚本改动；实施时必须保留这些修改，并以它们为 8×8 基线。

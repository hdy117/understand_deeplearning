# ResNet-50 学习与推理项目

这个目录同时保留两条互补路径：

1. `resnet_50.py`：从零实现 ResNet-50，用于理解 Bottleneck、projection shortcut、stage shape 和 residual data flow；
2. `pretrained_inference.py`：调用 `torchvision` 官方 ImageNet-1K 预训练权重，执行真实图片推理并生成可视化。

## 目录结构

```text
resnet50/
├── __init__.py
├── resnet_50.py                  # 手写 canonical ResNet-50，带原理/shape 注释
├── pretrained_inference.py       # 官方权重加载、预处理、Top-K、可视化和 CLI
├── requirements.txt              # 当前已验证的依赖版本
├── data/
│   ├── dog.jpg                   # PyTorch Hub 官方示例图片
│   └── README.md                 # 数据来源
├── outputs/
│   ├── dog_predictions.png       # 已真实生成的推理可视化
│   └── README.md
└── tests/
    ├── test_resnet_50.py
    └── test_pretrained_inference.py
```

## 1. 手写 ResNet-50

Canonical architecture 使用 `[3, 4, 6, 3]` 个 Bottleneck blocks：

```text
Input [N,3,224,224]
  -> 7x7 Conv + MaxPool
[N,64,56,56]
  -> layer1 x3
[N,256,56,56]
  -> layer2 x4
[N,512,28,28]
  -> layer3 x6
[N,1024,14,14]
  -> layer4 x3
[N,2048,7,7]
  -> AdaptiveAvgPool + FC
[N,1000]
```

每个 Bottleneck：

```text
Residual branch:
1x1 reduce -> 3x3 spatial processing -> 1x1 expand

Shortcut branch:
identity；shape 不同则使用 1x1 projection

Output:
ReLU(identity + residual)
```

### Bottleneck 内 H/W 什么时候变化？

不是所有 Bottleneck 都只改变 channels。

- Stage 内的普通 block：`conv1/conv2/conv3` 都保持 H/W；
- `layer2、layer3、layer4` 的第一个 block：`conv2` 使用 stride 2，H/W 减半；
- `conv1` 和 `conv3` 都是 1x1，主要改变 channels；
- 降采样 block 的 shortcut 同时使用 stride-2 1x1 projection，使两条分支具有相同 shape 后才能相加。

当前代码真实 hooks 输出：

| Block / operation | Output shape | 发生了什么？ |
|-------------------|--------------|---------------|
| `layer1.0.conv1` | `[1,64,56,56]` | H/W 不变，保持窄 channels |
| `layer1.0.conv2` | `[1,64,56,56]` | stride 1，H/W 不变 |
| `layer1.0.conv3` | `[1,256,56,56]` | H/W 不变，channels 扩展 4 倍 |
| `layer1.0.shortcut` | `[1,256,56,56]` | 只投影 channels |
| `layer2.0.conv1` | `[1,128,56,56]` | 降低 channels，H/W 不变 |
| `layer2.0.conv2` | `[1,128,28,28]` | **stride 2，H/W 减半** |
| `layer2.0.conv3` | `[1,512,28,28]` | H/W 不变，channels 扩展 4 倍 |
| `layer2.0.shortcut` | `[1,512,28,28]` | shortcut 同步降 H/W、升 channels |
| `layer2.1.conv1/2/3` | `28×28 → 28×28` | stage 内后续 block 不再降采样 |
| `layer3.0.conv2` | `28×28 → 14×14` | stage 3 首块降采样 |
| `layer4.0.conv2` | `14×14 → 7×7` | stage 4 首块降采样 |

### H/W 减半后，Shortcut 怎样与 Residual 相加？

Residual block 更完整的公式不是永远写成 \(F(x)+x\)，而是：

\[
y=\operatorname{ReLU}(F(x)+S(x)),
\]

其中 shortcut function \(S\) 有两种情况：

\[
S(x)=
\begin{cases}
x, & \text{输入输出 shape 相同},\\
\operatorname{BN}(\operatorname{Conv}_{1\times1,s}(x)),
& \text{H/W 或 channels 改变}.
\end{cases}
\]

以 `layer2.0` 为例：

```text
Input x: [N,256,56,56]

Residual branch F(x):
  conv1 1×1, s=1 → [N,128,56,56]
  conv2 3×3, s=2 → [N,128,28,28]
  conv3 1×1, s=1 → [N,512,28,28]

Shortcut S(x):
  conv 1×1, s=2  → [N,512,28,28]
  BatchNorm       → [N,512,28,28]

Add:
  F(x) + S(x)     → [N,512,28,28]
```

所以 transition block 中代码变量虽然名为 `identity`，它实际上会被 projection
shortcut 覆盖；这时不是严格的 identity function。只有输入输出 shape 相同的
普通 block 才真正执行 \(S(x)=x\)。

| Bottleneck 位置 | Block input/output shape 是否相同？ | Shortcut |
|-----------------|--------------------------------------|----------|
| `layer1.0` | 不同：channels `64→256`，H/W 不变 | 1×1 stride-1 projection |
| `layer1.1～1.2` | 完全相同 | 原样 identity |
| `layer2.0` | 不同：`256×56×56 → 512×28×28` | 1×1 stride-2 projection |
| `layer2.1～2.3` | 完全相同：`512×28×28` | 原样 identity |
| `layer3.0` | 不同：`512×28×28 → 1024×14×14` | 1×1 stride-2 projection |
| `layer3.1～3.5` | 完全相同：`1024×14×14` | 原样 identity |
| `layer4.0` | 不同：`1024×14×14 → 2048×7×7` | 1×1 stride-2 projection |
| `layer4.1～4.2` | 完全相同：`2048×7×7` | 原样 identity |

因此最准确的总结是：

```text
同一个 stage 内：H/W 保持不变；
进入 layer2、layer3、layer4 时：首个 Bottleneck 把 H/W 减半；
每个 stage 的首块负责建立新的 spatial/channel shape，后续 blocks 复用它。
```

运行随机参数 smoke test：

```bash
.venv/Scripts/python.exe resnet50/resnet_50.py
```

预期输出：

```text
input:  (1, 3, 224, 224)
output: (1, 1000)
parameters: 25,557,032
```

## 2. Torchvision 官方预训练 ResNet-50

官方 API：

```python
from torchvision.models import ResNet50_Weights, resnet50

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
```

当前环境中：

```text
DEFAULT             = IMAGENET1K_V2
ImageNet classes    = 1000
Parameters          = 25,557,032
Official acc@1      = 80.858
Official acc@5      = 95.434
Checkpoint file     ≈ 97.79 MB
```

这些指标来自当前安装的 `torchvision 0.28.0` weights metadata。

### 为什么必须使用 `weights.transforms()`？

预训练模型只对与训练 recipe 匹配的输入有明确含义。当前 V2 transforms 为：

```text
Resize shorter side to 232
CenterCrop 224x224
Convert to tensor
Normalize mean = [0.485, 0.456, 0.406]
Normalize std  = [0.229, 0.224, 0.225]
```

如果颜色顺序、scale 或 normalization 错误，即使网络和 weights 都正确，预测也可能明显变差。

## 3. 运行真实推理与可视化

使用内置示例图：

```bash
.venv/Scripts/python.exe -m resnet50.pretrained_inference
```

完整命令：

```bash
.venv/Scripts/python.exe -m resnet50.pretrained_inference \
  --image resnet50/data/dog.jpg \
  --output resnet50/outputs/dog_predictions.png \
  --top-k 5 \
  --device cpu
```

第一次运行会下载官方 checkpoint 到 PyTorch cache；之后自动复用。

当前真实运行结果：

```text
1. Samoyed          43.66%
2. white wolf        2.11%
3. Pomeranian        1.32%
4. Great Pyrenees    0.80%
5. Eskimo dog        0.59%
```

可视化：

`outputs/dog_predictions.png`

左侧是输入图片，右侧是 Top-K probability bar chart。

## 4. 换成自己的图片

```bash
.venv/Scripts/python.exe -m resnet50.pretrained_inference \
  --image D:/path/to/your_image.jpg \
  --output resnet50/outputs/my_prediction.png \
  --top-k 5 \
  --device auto
```

`--device auto` 会优先使用 CUDA，CUDA 不可用时自动选择 CPU。

注意：该模型只认识 ImageNet-1K 的 1000 个类别。输入不属于这些类别时，softmax 仍会被迫选择最相近类别；高概率也不等于模型真正理解图片。

## 5. 推理 Data Flow

```text
PIL image
  -> official resize/crop/normalize
Tensor [3,224,224]
  -> add batch dimension
Tensor [1,3,224,224]
  -> ResNet-50
Logits [1,1000]
  -> Softmax
Probabilities [1000]
  -> Top-K + ImageNet metadata
Labels and probabilities
  -> Matplotlib Agg backend
PNG visualization
```

## 6. 测试

```bash
.venv/Scripts/python.exe -m pytest resnet50/tests -q
```

测试不会下载官方 checkpoint：推理单元测试通过 dependency injection 使用固定 logits model。真实官方权重下载和推理作为独立 integration/ad-hoc verification 执行。

## 7. 安装依赖

当前项目已有本地 `.venv`。重新安装可运行：

```bash
uv pip install --python .venv/Scripts/python.exe -r resnet50/requirements.txt
```

官方模型文档：

https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

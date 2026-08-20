"""从零实现 canonical ResNet-50，不依赖 torchvision 的模型封装。

以 ImageNet 常见的 224x224 RGB 输入为例，shape 主线是：

    [N, 3, 224, 224] -> stem -> [N, 64, 56, 56]
    -> layer1 [N, 256, 56, 56]
    -> layer2 [N, 512, 28, 28]
    -> layer3 [N, 1024, 14, 14]
    -> layer4 [N, 2048, 7, 7]
    -> global average pooling -> [N, num_classes]

ResNet 让每个 block 学 residual update F(x)，输出 x + F(x)。若新增 block
暂时没有有用修正，F(x) 接近 0 就能保留 identity；这比让多层 nonlinear
transformations 从头重建 identity 更容易优化。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

__all__ = ["Bottleneck", "ResNet", "resnet50"]

NormLayer = Callable[[int], nn.Module]


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """创建只混合 channels 的 1x1 projection convolution。

    它不扩大 spatial receptive field；用于 Bottleneck 的 channel 压缩/扩展，
    也用于 shortcut shape 不匹配时投影 identity。BatchNorm 已提供 affine
    bias，因此 convolution 设置 ``bias=False``。
    """

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """创建负责 spatial pattern extraction 的 3x3 convolution。

    ``padding=dilation`` 使 stride 1 时保持 H/W；stride 2 时则在提取 feature
    的同时把空间分辨率约减半。
    """

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Bottleneck(nn.Module):
    """ResNet-50 的 ``1x1 -> 3x3 -> 1x1`` residual block。

    标准配置 ``channels=64`` 时，channel flow 为：

        in_channels -> 64 -> 64 -> 64 * expansion(4) = 256

    昂贵的 3x3 convolution 在较窄空间运行，因此称为 Bottleneck。
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: NormLayer = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        # 标准 ResNet-50 中 groups=1、base_width=64，所以 width == channels。
        # 这两个参数也保留了扩展为 ResNeXt grouped convolution 的接口。
        width = int(channels * (base_width / 64.0)) * groups

        # 1x1 reduce：把输入投影到较窄的 bottleneck channel space。
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        # 3x3 process：真正处理邻域结构；stage 切换时 stride=2 放在这里。
        self.conv2 = conv3x3(
            width,
            width,
            stride=stride,
            groups=groups,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        # 1x1 expand：扩展到 channels * 4，准备与 identity 相加。
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        # identity 保存 block 输入；仅 shape 不匹配时才走 projection shortcut。
        identity = x

        # Residual branch：reduce -> spatial processing -> expand。
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # stride 或 channel expansion 使两条分支 shape 不同时，1x1 projection
        # 会同时调整 identity 的 H/W 和 channels，满足 element-wise add 接口。
        if self.downsample is not None:
            identity = self.downsample(x)

        # 核心计算：旧表示 + learned correction。
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet backbone 加 global-pooling classification head。

    ResNet-50 的 ``stage_depths=[3,4,6,3]``。每个 stage 内保持 shape；进入
    layer2-4 的首个 block 才通过 stride 2 降采样并增加 channels。
    """

    def __init__(
        self,
        block: type[Bottleneck],
        stage_depths: Sequence[int],
        *,
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        zero_init_residual: bool = False,
        norm_layer: NormLayer = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        if len(stage_depths) != 4:
            raise ValueError("stage_depths must contain exactly four integers")
        if any(depth <= 0 for depth in stage_depths):
            raise ValueError("every stage depth must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self._norm_layer = norm_layer
        # 记录“当前 tensor 的 channel 数”。每构造完一个 stage 都会更新，
        # 下一 stage 才能建立正确的输入 shape contract。
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        # Stem：7x7 convolution 与 max pool 连续两次降采样，快速把原图转换成
        # 低层 feature grid。in_channels 可配置，因此也支持 4-channel 输入。
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个 stages 的 Bottleneck 输出 channels 依次为 256/512/1024/2048。
        # layer1 保持 56x56；layer2-4 的首块 stride 2，使 H/W 逐级减半。
        self.layer1 = self._make_layer(block, 64, stage_depths[0])
        self.layer2 = self._make_layer(block, 128, stage_depths[1], stride=2)
        self.layer3 = self._make_layer(block, 256, stage_depths[2], stride=2)
        self.layer4 = self._make_layer(block, 512, stage_depths[3], stride=2)

        # 每个 channel 的 HxW feature map 汇总成一个数；Adaptive 形式允许
        # 网络接收不同分辨率，而不要求分类 head 固定读取 7x7。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights(zero_init_residual)

    def _make_layer(
        self,
        block: type[Bottleneck],
        channels: int,
        blocks: int,
        *,
        stride: int = 1,
    ) -> nn.Sequential:
        """构造一个 residual stage，并维护 identity/residual 的 shape 接口。

        第一个 block 可能修改 H/W 或 channels，所以可能需要 projection shortcut；
        后续 blocks 的输入输出 shape 相同，可以使用纯 identity shortcut。
        """

        norm_layer = self._norm_layer
        out_channels = channels * block.expansion
        downsample: nn.Module | None = None

        # 只要 spatial size 或 channel 数任一不同，就不能直接 element-wise add。
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels, stride),
                norm_layer(out_channels),
            )

        # Stage 首块承担入口处的降采样/channel expansion。
        layers: list[nn.Module] = [
            block(
                self.inplanes,
                channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                norm_layer=norm_layer,
            )
        ]
        # 首块之后，stage 内所有 tensor 都固定为 out_channels。
        self.inplanes = out_channels

        # 后续 blocks 不再改变 shape，shortcut 直接复用原输入。
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        """初始化参数，并可令每个 residual branch 初始接近零。

        Kaiming initialization 配合 ReLU 控制 signal scale。若启用
        ``zero_init_residual``，最后一个 BatchNorm 的 scale 为 0，使初始
        F(x) 近似 0，整个 Bottleneck 因而更接近 identity mapping。
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck) and module.bn3.weight is not None:
                    nn.init.zeros_(module.bn3.weight)

    def forward(self, x: Tensor) -> Tensor:
        """执行 stem -> four stages -> global pooling -> logits。"""

        # Stem: [N,C,H,W] -> [N,64,约 H/4,约 W/4]。
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Feature hierarchy：空间分辨率逐级降低，semantic channels 逐级增加。
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [N,2048,H',W'] -> [N,2048,1,1] -> [N,2048] -> logits。
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet50(
    *,
    num_classes: int = 1000,
    in_channels: int = 3,
    zero_init_residual: bool = False,
) -> ResNet:
    """构造 canonical ResNet-50。

    Args:
        num_classes: 分类 head 的输出类别数。
        in_channels: 输入 channel 数；ImageNet RGB 默认为 3。
        zero_init_residual: 是否令每个 residual branch 初始接近 0。
    """

    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        zero_init_residual=zero_init_residual,
    )


if __name__ == "__main__":
    # 独立运行时只做随机参数模型的结构 smoke test。官方预训练权重、ImageNet
    # preprocessing 与可视化由 pretrained_inference.py 负责。
    model = resnet50()
    model.eval()
    example = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(example)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"input:  {tuple(example.shape)}")
    print(f"output: {tuple(output.shape)}")
    print(f"parameters: {parameter_count:,}")

"""手写 ResNet-50 的行为与 architecture contract 测试。"""

from __future__ import annotations

import torch

from resnet50 import Bottleneck, resnet50


def test_resnet50_forward_returns_class_logits() -> None:
    """Adaptive pooling + FC 应把任意支持的空间尺寸转换为 class logits。"""

    model = resnet50(num_classes=10)
    model.eval()

    inputs = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (1, 10)


def test_resnet50_uses_canonical_bottleneck_stage_depths() -> None:
    """ResNet-50 的“50”来自 canonical Bottleneck [3,4,6,3] 结构。"""

    model = resnet50()

    # 每个 Bottleneck 有三层 convolution，最后输出 channels 扩展为 4 倍。
    assert Bottleneck.expansion == 4
    assert [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)] == [
        3,
        4,
        6,
        3,
    ]
    assert all(
        isinstance(block, Bottleneck)
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4)
        for block in layer
    )

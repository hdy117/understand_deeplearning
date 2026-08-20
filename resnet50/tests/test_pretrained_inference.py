"""官方预训练模型推理 primitive 的无网络单元测试。"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn

from resnet50.pretrained_inference import (
    ModelBundle,
    Prediction,
    predict_image,
    run_inference,
    visualize_predictions,
)


class FixedLogitsModel(nn.Module):
    """返回固定 logits，隔离官方权重下载并精确测试 Top-K 逻辑。"""

    def forward(self, _inputs: Tensor) -> Tensor:
        return torch.tensor([[0.0, 2.0, 1.0]])


def test_predict_image_returns_ranked_labels_and_probabilities() -> None:
    """Logits 应经过 softmax，并按概率映射到对应类别标签。"""

    image = Image.new("RGB", (16, 16), color="white")

    predictions = predict_image(
        image=image,
        model=FixedLogitsModel(),
        preprocess=lambda _image: torch.zeros(3, 8, 8),
        categories=("class-a", "class-b", "class-c"),
        device=torch.device("cpu"),
        top_k=2,
    )

    assert [prediction.label for prediction in predictions] == ["class-b", "class-c"]
    assert predictions[0].probability > predictions[1].probability
    assert predictions[0].class_index == 1


def test_visualize_predictions_writes_a_png(tmp_path: Path) -> None:
    """可视化应生成可打开的 PNG，而不依赖交互式 GUI。"""

    image = Image.new("RGB", (32, 24), color=(40, 120, 200))
    predictions = [
        Prediction(class_index=1, label="tabby cat", probability=0.72),
        Prediction(class_index=2, label="tiger cat", probability=0.18),
    ]
    output_path = tmp_path / "nested" / "prediction.png"

    result_path = visualize_predictions(
        image=image,
        predictions=predictions,
        output_path=output_path,
        title="Test prediction",
    )

    assert result_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as rendered:
        assert rendered.format == "PNG"


def test_run_inference_combines_prediction_and_visualization(tmp_path: Path) -> None:
    """完整 pipeline 应返回结构化结果，并把可视化写到指定路径。"""

    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (24, 24), color="white").save(image_path)
    output_path = tmp_path / "result.png"
    bundle = ModelBundle(
        model=FixedLogitsModel(),
        preprocess=lambda _image: torch.zeros(3, 8, 8),
        categories=("class-a", "class-b", "class-c"),
        device=torch.device("cpu"),
        weights_name="test-weights",
    )

    result = run_inference(
        image_path=image_path,
        output_path=output_path,
        top_k=2,
        bundle=bundle,
    )

    assert [prediction.label for prediction in result.predictions] == [
        "class-b",
        "class-c",
    ]
    assert result.visualization_path == output_path
    assert result.weights_name == "test-weights"
    assert output_path.is_file()

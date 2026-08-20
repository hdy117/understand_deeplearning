"""Torchvision 官方 ResNet-50 的推理与可视化工具。

本模块把推理过程拆成可观察的步骤：

1. PIL 图片经过官方 weights 绑定的预处理；
2. 增加 batch 维度后输入网络；
3. logits 经过 softmax 变成概率；
4. 取 Top-K，并把 ImageNet class index 映射成人类可读标签。

后续的命令行入口会复用这些小函数，避免把模型加载、推理、绘图揉成
一大段难以测试的脚本。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class Prediction:
    """一条分类结果。

    Attributes:
        class_index: 模型输出向量中的 ImageNet 类别下标。
        label: torchvision weights metadata 提供的人类可读类别名。
        probability: softmax 后的概率，范围为 [0, 1]。
    """

    class_index: int
    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """一次加载后可重复使用的官方模型资源。

    ``torchvision`` 的 weights enum 不只包含参数下载地址，还绑定了训练时对应的
    preprocessing recipe 和 1000 个 ImageNet categories。把这些对象放在同一个
    bundle 中，可避免误用“正确模型 + 错误归一化”这种隐蔽 bug。
    """

    model: nn.Module
    preprocess: Callable[[Image.Image], Tensor]
    categories: Sequence[str]
    device: torch.device
    weights_name: str


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """完整推理 pipeline 的输出。"""

    predictions: tuple[Prediction, ...]
    visualization_path: Path
    device: str
    weights_name: str


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """解析推理设备；默认优先 CUDA，否则回退 CPU。"""

    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def load_official_resnet50(
    device: str | torch.device | None = None,
) -> ModelBundle:
    """加载 torchvision 官方 ImageNet-1K V2 预训练 ResNet-50。

    ``ResNet50_Weights.DEFAULT`` 当前指向 ``IMAGENET1K_V2``。第一次运行时，
    torchvision 会把约 98 MB 的 checkpoint 下载到 PyTorch 本地 cache；后续运行
    直接复用。``weights.transforms()`` 是与这份权重配套的 resize/crop/normalize
    流程，不应手写一套看似相同但参数不同的预处理。
    """

    # 延迟导入 torchvision，让只测试纯 Prediction/visualization 的代码启动更快。
    from torchvision.models import ResNet50_Weights
    from torchvision.models import resnet50 as torchvision_resnet50

    target_device = resolve_device(device)
    weights = ResNet50_Weights.DEFAULT

    # weights=... 同时确定网络结构与 checkpoint；不是先建随机网络再“猜着加载”。
    model = torchvision_resnet50(weights=weights).to(target_device)
    model.eval()

    return ModelBundle(
        model=model,
        preprocess=weights.transforms(),
        categories=tuple(weights.meta["categories"]),
        device=target_device,
        weights_name=weights.name,
    )


def predict_image(
    *,
    image: Image.Image,
    model: nn.Module,
    preprocess: Callable[[Image.Image], Tensor],
    categories: Sequence[str],
    device: torch.device,
    top_k: int = 5,
) -> list[Prediction]:
    """对一张图片执行分类，并返回按概率降序排列的 Top-K。

    依赖通过参数注入，而不是在函数内部偷偷下载权重。这样既方便单元测试，
    也允许调用方复用已经加载好的模型，批量推理时不会重复初始化网络。

    Shape flow:
        PIL image
          -> preprocess
        [C, H, W]
          -> unsqueeze batch dimension
        [1, C, H, W]
          -> model
        [1, num_classes]
          -> softmax + topk
        list[Prediction]
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # 官方 transform 会完成 resize、center crop、ToTensor 和 ImageNet normalize。
    # convert("RGB") 保证灰度图或 RGBA 图也满足 ResNet stem 的 3-channel 接口。
    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    # eval() 让 BatchNorm 使用训练期间保存的 running statistics，
    # 同时关闭 Dropout（标准 ResNet-50 本身没有 Dropout，但这是稳妥的推理协议）。
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits[0], dim=0)

    if probabilities.numel() != len(categories):
        raise ValueError(
            "category count must match the number of model output classes: "
            f"{len(categories)} != {probabilities.numel()}"
        )

    # 当 top_k 大于类别总数时自动截断，避免 torch.topk 越界。
    k = min(top_k, probabilities.numel())
    top_probabilities, top_indices = torch.topk(probabilities, k=k)

    return [
        Prediction(
            class_index=int(class_index),
            label=categories[int(class_index)],
            probability=float(probability),
        )
        for probability, class_index in zip(top_probabilities, top_indices, strict=True)
    ]


def visualize_predictions(
    *,
    image: Image.Image,
    predictions: Sequence[Prediction],
    output_path: str | Path,
    title: str = "Torchvision ResNet-50 prediction",
) -> Path:
    """把输入图片和 Top-K 分类概率保存为 PNG。

    使用 non-interactive ``Agg`` backend，因此在无桌面的服务器、CI 或当前
    CLI 环境中也能运行。横向 bar chart 比纯文本更容易观察模型是“一枝独秀”
    还是在几个相近类别之间犹豫。
    """

    if not predictions:
        raise ValueError("predictions must not be empty")

    # 延迟导入 matplotlib，避免只使用 predict_image 的调用方承担绘图库启动开销。
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # barh 从下往上绘制；反转列表后，最高概率会显示在图的最上方。
    labels = [prediction.label for prediction in reversed(predictions)]
    probabilities = [
        prediction.probability * 100.0 for prediction in reversed(predictions)
    ]

    figure, (image_axes, bar_axes) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": (1.2, 1.0)},
    )
    figure.suptitle(title, fontsize=14, fontweight="bold")

    image_axes.imshow(image.convert("RGB"))
    image_axes.set_title("Input image")
    image_axes.axis("off")

    bars = bar_axes.barh(labels, probabilities, color="#4472C4")
    bar_axes.set_xlim(0.0, 100.0)
    bar_axes.set_xlabel("Probability (%)")
    bar_axes.set_title("Top predictions")
    bar_axes.grid(axis="x", alpha=0.25)
    bar_axes.bar_label(bars, fmt="%.2f%%", padding=3)

    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def run_inference(
    *,
    image_path: str | Path,
    output_path: str | Path,
    top_k: int = 5,
    device: str | torch.device | None = None,
    bundle: ModelBundle | None = None,
) -> InferenceResult:
    """读取一张图片，运行官方 ResNet-50，并保存预测可视化。

    ``bundle`` 参数主要用于两类场景：

    - 单元测试注入一个小模型，避免下载 checkpoint；
    - 批量推理复用同一官方模型，避免每张图片都重新加载权重。
    """

    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"input image does not exist: {source}")

    resources = bundle if bundle is not None else load_official_resnet50(device)

    # copy() 让图片数据脱离 with block 内部的文件句柄；之后绘图仍可安全使用。
    with Image.open(source) as opened:
        image = opened.convert("RGB").copy()

    predictions = predict_image(
        image=image,
        model=resources.model,
        preprocess=resources.preprocess,
        categories=resources.categories,
        device=resources.device,
        top_k=top_k,
    )
    rendered_path = visualize_predictions(
        image=image,
        predictions=predictions,
        output_path=output_path,
        title=f"ResNet-50 · {resources.weights_name} · {resources.device}",
    )

    return InferenceResult(
        predictions=tuple(predictions),
        visualization_path=rendered_path,
        device=str(resources.device),
        weights_name=resources.weights_name,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run official torchvision ResNet-50 inference with visualization."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=package_dir / "data" / "dog.jpg",
        help="input image path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_dir / "outputs" / "dog_predictions.png",
        help="output PNG visualization path",
    )
    parser.add_argument("--top-k", type=int, default=5, help="number of classes to show")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    return parser


def main() -> None:
    """Command-line entry point."""

    args = _build_argument_parser().parse_args()
    result = run_inference(
        image_path=args.image,
        output_path=args.output,
        top_k=args.top_k,
        device=args.device,
    )

    print(f"weights: {result.weights_name}")
    print(f"device:  {result.device}")
    for rank, prediction in enumerate(result.predictions, start=1):
        print(
            f"{rank:>2}. {prediction.label:<30} "
            f"{prediction.probability * 100:6.2f}% "
            f"(class {prediction.class_index})"
        )
    print(f"visualization: {result.visualization_path.resolve()}")


if __name__ == "__main__":
    main()

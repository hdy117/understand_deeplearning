"""Compare per-class VAE latents with Flow Matching samples.

原理
----
VAE encoder 把真实图片映射到 latent space：

    z_real = VAE.encode(image)

Flow Matching 则从高斯噪声开始，通过数值积分得到生成 latent：

    z_0 ~ N(0, I) -> z_1_generated

因此，不能只看生成图片是否能被 decoder 解码。VAE decoder 对很多
latent 都能输出一张图片；更有用的检查是比较 ``z_1_generated`` 和
真实图片编码得到的 ``z_real`` 是否具有相近的分布，并检查：

    generated latent -> VAE decoder -> VAE encoder

是否还能回到原来的 latent。后一个过程叫 round-trip check。真实 latent
的 round-trip 误差是 VAE 自身的参考值；如果生成 latent 的误差明显更大，
说明 Flow 输出偏离了 VAE 熟悉的 latent manifold。

本脚本按类别分别统计。每个类别的统计都在 ``[sample, token, channel]``
上展开为 ``[sample * token, channel]``，所以 mean/std 描述的是该类别的
latent channel 分布，而不是像素分布。

指标解释
--------
``mean_abs_error``
    每个 channel 的 |mean_generated - mean_real|。
``std_ratio``
    ``std_generated / std_real``，接近 1 表示离散程度相近。
``generated_outside_real_3sigma``
    生成 latent 落在真实类别 ``mean ± 3 * std`` 之外的比例。
``*_roundtrip_rmse``
    ``encode(decode(z))`` 与原始 ``z`` 的 RMSE。生成值若远高于真实值，
    通常表示生成 latent 不在 VAE 的有效区域内。

Example
-------
python analyze_latent_distribution.py \
    --checkpoint outputs/flow_matching_epoch.pt \
    --max-real 512 \
    --generated 512 \
    --output outputs/latent_distribution.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torchvision.io import read_image

from config import DEVICE, FMConfig, FlowMatchingModelConfig
from flow_matching import FlowMatching
from vae import VAE, load_vae_from_ckpt


@dataclass
class LatentStats:
    """展开 token 维度后，每个 latent channel 的摘要统计。

    一个 latent 的形状是 ``[N, 64, 4]``。这里不保存每个 token 的完整
    分布，而是把前两维展开，得到每个 channel 的总体统计，便于类别间
    快速比较。
    """

    mean: list[float]
    std: list[float]
    q10: list[float]
    median: list[float]
    q90: list[float]


@dataclass
class ClassReport:
    """一个类别的真实 latent 与生成 latent 的分布比较结果。"""

    class_name: str
    class_index: int
    real_count: int
    generated_count: int
    real: LatentStats
    generated: LatentStats
    mean_abs_error: list[float]
    std_abs_error: list[float]
    std_ratio: list[float]
    generated_outside_real_3sigma: float
    real_roundtrip_rmse: float
    generated_roundtrip_rmse: float
    token_channel_mean_abs_error: list[list[float]]
    token_channel_std_ratio: list[list[float]]
    token_mean_abs_error: list[float]
    token_std_ratio: list[float]
    worst_mean_error_token: int
    worst_mean_error_value: float
    lowest_std_ratio_token: int
    lowest_std_ratio_value: float


FIELD_DESCRIPTIONS = {
    "class_name": "类别名称；按训练集目录名字典序排列。",
    "class_index": "one-hot 条件中的类别下标。",
    "real_count": "用于统计的真实图片数量。",
    "generated_count": "Flow 为该类别生成的 latent 数量。",
    "real": "真实图片经过 VAE encoder 得到的 latent 总体统计。",
    "generated": "Flow 采样得到的 latent 总体统计。",
    "mean_abs_error": "每个 channel 的 |生成均值 - 真实均值|。越小越接近。",
    "std_abs_error": "每个 channel 的生成标准差与真实标准差的绝对差。",
    "std_ratio": "生成标准差 / 真实标准差；接近 1 表示方差接近。",
    "generated_outside_real_3sigma": "生成值落在真实均值 ± 3 倍真实标准差之外的比例。",
    "real_roundtrip_rmse": "真实 latent 经过 decode→encode 后的 RMSE，作为 VAE 自身误差基线。",
    "generated_roundtrip_rmse": "生成 latent 经过 decode→encode 后的 RMSE；明显高于 real-roundtrip 表示偏离 VAE latent manifold。",
    "token_channel_mean_abs_error": "[64][4]；每个空间 token、每个 channel 的均值绝对差。用于发现空间位置误差。",
    "token_channel_std_ratio": "[64][4]；每个空间 token、每个 channel 的生成/真实标准差比例。",
    "token_mean_abs_error": "[64]；每个 token 对四个 channel 的 mean_abs_error 平均值。",
    "token_std_ratio": "[64]；每个 token 对四个 channel 的 std_ratio 平均值。",
    "worst_mean_error_token": "mean_abs_error 最大的 token 下标，token 顺序与 VAE flatten 顺序一致。",
    "worst_mean_error_value": "最差 token 的平均均值误差。",
    "lowest_std_ratio_token": "std_ratio 最低的 token 下标。",
    "lowest_std_ratio_value": "最低 token 的平均标准差比例。",
}


class LatentDistributionAnalyzer:
    """执行只读的、按类别条件化的 latent 分布诊断。

    训练模型不在这里修改；本类只负责三件事：

    1. 从真实图片获得 VAE latent；
    2. 用对应的 one-hot label 采样 Flow latent；
    3. 计算分布差异与 VAE round-trip 误差。
    """

    def __init__(
        self,
        *,
        cfg: FMConfig,
        flow_checkpoint: Path,
        vae_checkpoint: Path,
        device: str | torch.device = DEVICE,
        batch_size: int = 64,
        sampling_steps: int | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.seed = seed
        self.model_cfg: FlowMatchingModelConfig = cfg.fm_model_cfg
        self.sampling_steps = sampling_steps or self.model_cfg.total_steps

        self.vae: VAE = load_vae_from_ckpt(str(vae_checkpoint))
        self.vae.to(self.device).eval()

        self.flow = FlowMatching(
            model_config=self.model_cfg,
            device=self.device,
        ).eval()
        checkpoint = torch.load(
            flow_checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        self.flow.load_state_dict(checkpoint["model"])

        # 真实 latent 的类别边界来自目录名；类别 index 必须和训练时
        # DataHolder 使用的 sorted(category_name) 顺序一致。
        self.categories = self._find_categories(Path(cfg.data_cfg.train_dir))
        if len(self.categories) != self.model_cfg.category_num:
            raise ValueError(
                f"found {len(self.categories)} categories, expected "
                f"{self.model_cfg.category_num}"
            )

    @staticmethod
    def _find_categories(data_dir: Path) -> list[tuple[str, list[Path]]]:
        """按训练时的字典序找到类别和图片路径。

        ``sorted`` 很重要：Flow 的 one-hot 第 i 位必须继续对应同一个
        类别，否则比较的就是错类别的真实 latent。
        """
        categories: list[tuple[str, list[Path]]] = []
        for category_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            paths = sorted(p for p in category_dir.iterdir() if p.is_file())
            if paths:
                categories.append((category_dir.name, paths))
        return categories

    @staticmethod
    def _stats(latents: torch.Tensor) -> LatentStats:
        """计算每个 channel 的分位数统计。

        ``latents`` 是 ``[N, tokens, channels]``；展开 token 维度后，
        每一列就是一个 latent channel 的观测集合。
        """
        values = latents.reshape(-1, latents.shape[-1]).float()
        quantiles = torch.quantile(
            values,
            torch.tensor([0.10, 0.50, 0.90], device=values.device),
            dim=0,
        )
        return LatentStats(
            mean=values.mean(dim=0).cpu().tolist(),
            std=values.std(dim=0, unbiased=False).cpu().tolist(),
            q10=quantiles[0].cpu().tolist(),
            median=quantiles[1].cpu().tolist(),
            q90=quantiles[2].cpu().tolist(),
        )

    def _encode_paths(self, paths: Iterable[Path]) -> torch.Tensor:
        """读取一批真实图片，并取 VAE posterior mean 作为 latent。

        这里使用 ``encode_tokens``，与训练 Flow 时的代码保持一致；不使用
        VAE posterior sampling，避免比较结果额外混入 VAE 随机噪声。
        """
        images = torch.stack(
            [read_image(str(path)).float().div(255.0) for path in paths]
        ).to(self.device)
        with torch.inference_mode():
            return self.vae.encode_tokens(images)

    def _encode_paths_batched(self, paths: list[Path]) -> torch.Tensor:
        """分批编码，避免一次性把一个类别的全部图片放进显存。"""
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            chunks.append(self._encode_paths(paths[start : start + self.batch_size]))
        return torch.cat(chunks, dim=0)

    def _sample_class(self, class_index: int, count: int) -> torch.Tensor:
        """采样一个类别的 Flow latent。

        初始状态是 ``z_0 ~ N(0, I)``。随后用显式 Euler 积分近似：

            z_{k+1} = z_k + dt * v_theta(z_k, t_k, class)

        返回值的形状是 ``[count, tokens, latent_channels]``，和
        ``VAE.encode_tokens`` 的输出完全一致。
        """
        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + class_index
        )
        result = []
        # 同一个类别的所有样本共享 one-hot 条件；噪声不同，所以得到的
        # 是该类别的多个生成样本，而不是重复同一张 latent。
        labels = torch.zeros(
            self.batch_size,
            self.model_cfg.category_num,
            device=self.device,
        )
        labels[:, class_index] = 1.0

        with torch.inference_mode():
            for start in range(0, count, self.batch_size):
                current = min(self.batch_size, count - start)
                latent = torch.randn(
                    current,
                    self.vae.latent_size * self.vae.latent_size,
                    self.model_cfg.latent_channels,
                    device=self.device,
                    generator=generator,
                )
                current_labels = labels[:current]
                for step in range(self.sampling_steps):
                    # 模型训练时的时间范围是 [0, 1)，所以最后一次使用
                    # t=(steps-1)/steps，并把状态推进到近似 t=1。
                    time = torch.full(
                        (current,),
                        step / self.sampling_steps,
                        device=self.device,
                    )
                    latent = latent + (1.0 / self.sampling_steps) * self.flow(
                        latent,
                        time,
                        current_labels,
                    )
                result.append(latent)
        return torch.cat(result, dim=0)

    def _roundtrip_rmse(self, latents: torch.Tensor) -> float:
        """计算 ``z -> decode(z) -> encode(image)`` 的 RMSE。

        真实 latent 也要经过同样的流程，因为 VAE 本身不是完美可逆的。
        只有把生成误差和真实误差比较，才能判断问题是否来自 Flow。
        """
        values = []
        with torch.inference_mode():
            for start in range(0, len(latents), self.batch_size):
                chunk = latents[start : start + self.batch_size]
                values.append(
                    (
                        self.vae.encode_tokens(self.vae.decode_tokens(chunk)) - chunk
                    ).square()
                )
        return torch.cat(values).mean().sqrt().item()

    def analyze(self, max_real: int, generated: int) -> list[ClassReport]:
        """逐类别完成真实 latent、生成 latent 和差异指标的计算。"""
        reports = []
        for class_index, (class_name, paths) in enumerate(self.categories):
            selected_paths = paths[:max_real]
            real = self._encode_paths_batched(selected_paths)
            sampled = self._sample_class(class_index, generated)

            real_stats = self._stats(real)
            generated_stats = self._stats(sampled)
            real_mean = torch.tensor(real_stats.mean, device=self.device)
            real_std = torch.tensor(real_stats.std, device=self.device)
            generated_mean = torch.tensor(
                generated_stats.mean, device=self.device
            )
            generated_std = torch.tensor(generated_stats.std, device=self.device)
            eps = 1e-6

            # 使用真实类别的 mean/std 定义参考坐标系；这里的 3 sigma
            # 是一个简单、可解释的异常比例指标，不等价于完整概率密度。
            z_score = (sampled - real_mean) / (real_std + eps)
            # 保留 token 维度重新统计。全局统计会把 64 个空间位置混在一起，
            # 可能掩盖“总体分布正常但空间布局错误”的问题。
            real_token_mean = real.mean(dim=0)
            real_token_std = real.std(dim=0, unbiased=False)
            generated_token_mean = sampled.mean(dim=0)
            generated_token_std = sampled.std(dim=0, unbiased=False)
            token_mean_error = (generated_token_mean - real_token_mean).abs()
            token_std_ratio = generated_token_std / (real_token_std + eps)
            token_mean_error_summary = token_mean_error.mean(dim=1)
            token_std_ratio_summary = token_std_ratio.mean(dim=1)
            worst_mean_token = int(token_mean_error_summary.argmax().item())
            lowest_std_token = int(token_std_ratio_summary.argmin().item())
            reports.append(
                ClassReport(
                    class_name=class_name,
                    class_index=class_index,
                    real_count=len(selected_paths),
                    generated_count=generated,
                    real=real_stats,
                    generated=generated_stats,
                    mean_abs_error=(generated_mean - real_mean)
                    .abs()
                    .cpu()
                    .tolist(),
                    std_abs_error=(generated_std - real_std).abs().cpu().tolist(),
                    std_ratio=(generated_std / (real_std + eps)).cpu().tolist(),
                    generated_outside_real_3sigma=(z_score.abs() > 3)
                    .float()
                    .mean()
                    .item(),
                    real_roundtrip_rmse=self._roundtrip_rmse(real),
                    generated_roundtrip_rmse=self._roundtrip_rmse(sampled),
                    token_channel_mean_abs_error=token_mean_error.cpu().tolist(),
                    token_channel_std_ratio=token_std_ratio.cpu().tolist(),
                    token_mean_abs_error=token_mean_error_summary.cpu().tolist(),
                    token_std_ratio=token_std_ratio_summary.cpu().tolist(),
                    worst_mean_error_token=worst_mean_token,
                    worst_mean_error_value=token_mean_error_summary[
                        worst_mean_token
                    ].item(),
                    lowest_std_ratio_token=lowest_std_token,
                    lowest_std_ratio_value=token_std_ratio_summary[
                        lowest_std_token
                    ].item(),
                )
            )
        return reports


def print_reports(reports: list[ClassReport]) -> None:
    """打印适合快速比较的类别级摘要。

    JSON 文件保留四个 channel 的完整数值；终端表格只打印四个 channel
    的平均差异，避免把屏幕变成一张难以阅读的数字墙。
    """
    print(
        "class       mean|Δ|  std|Δ|  std-ratio  outside-3σ  "
        "real-rt  generated-rt"
    )
    for report in reports:
        mean_error = sum(report.mean_abs_error) / len(report.mean_abs_error)
        std_error = sum(report.std_abs_error) / len(report.std_abs_error)
        ratio = sum(report.std_ratio) / len(report.std_ratio)
        print(
            f"{report.class_name:<10} {mean_error:7.4f}  {std_error:7.4f}  "
            f"{ratio:9.4f}  {report.generated_outside_real_3sigma:10.4f}  "
            f"{report.real_roundtrip_rmse:7.4f}  "
            f"{report.generated_roundtrip_rmse:12.4f}"
        )


def write_markdown_report(
    path: Path,
    reports: list[ClassReport],
    *,
    checkpoint: Path,
    sampling_steps: int,
) -> None:
    """写一份带字段解释的人工可读报告。

    JSON 适合后续程序处理；Markdown 适合人工检查，尤其是逐 token 的
    空间差异。token index 使用 VAE ``flatten(start_dim=2).transpose(1, 2)``
    的顺序，因此 index 0 到 63 对应 8x8 latent grid 的行优先位置。
    """
    lines = [
        "# Flow Matching latent distribution report",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- sampling steps: `{sampling_steps}`",
        "",
        "## 字段说明",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{name}` | {description} |"
        for name, description in FIELD_DESCRIPTIONS.items()
    )
    lines.extend(
        [
            "",
            "## 类别总体结果",
            "",
            "| class | mean|Δ| | std-ratio | outside-3σ | real-rt | generated-rt | worst token | lowest std token |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for report in reports:
        mean_error = sum(report.mean_abs_error) / len(report.mean_abs_error)
        std_ratio = sum(report.std_ratio) / len(report.std_ratio)
        lines.append(
            f"| {report.class_name} | {mean_error:.4f} | {std_ratio:.4f} | "
            f"{report.generated_outside_real_3sigma:.4f} | "
            f"{report.real_roundtrip_rmse:.4f} | "
            f"{report.generated_roundtrip_rmse:.4f} | "
            f"{report.worst_mean_error_token} ({report.worst_mean_error_value:.4f}) | "
            f"{report.lowest_std_ratio_token} ({report.lowest_std_ratio_value:.4f}) |"
        )

    lines.extend(["", "## 逐 token 空间结果", ""])
    for report in reports:
        lines.extend(
            [
                f"### {report.class_name}",
                "",
                "`token` 按 8x8 latent grid 行优先编号；`mean|Δ|` 和 `std-ratio` "
                "是该 token 四个 channel 的平均值。",
                "",
                "| token | row | col | mean|Δ| | std-ratio |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for token, (mean_error, std_ratio) in enumerate(
            zip(report.token_mean_abs_error, report.token_std_ratio)
        ):
            lines.append(
                f"| {token} | {token // 8} | {token % 8} | "
                f"{mean_error:.4f} | {std_ratio:.4f} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/flow_matching_epoch.pt"),
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path("outputs/vae_best.pt"),
    )
    parser.add_argument("--max-real", type=int, default=512)
    parser.add_argument("--generated", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/latent_distribution.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/latent_distribution.md"),
        help="带字段解释和逐 token 表格的 Markdown 报告",
    )
    return parser.parse_args()


def main() -> None:
    """解析参数、运行诊断，并把完整结果保存为 JSON。"""
    args = parse_args()
    if args.max_real <= 0 or args.generated <= 0 or args.batch_size <= 0:
        raise ValueError("--max-real, --generated, and --batch-size must be positive")
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")

    cfg = FMConfig()
    analyzer = LatentDistributionAnalyzer(
        cfg=cfg,
        flow_checkpoint=args.checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        sampling_steps=args.steps,
        seed=args.seed,
    )
    reports = analyzer.analyze(args.max_real, args.generated)
    print_reports(reports)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "_description": "按类别比较 VAE latent 与 Flow Matching latent。",
                "_field_descriptions": FIELD_DESCRIPTIONS,
                "reports": [asdict(report) for report in reports],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {args.output}")
    write_markdown_report(
        args.report,
        reports,
        checkpoint=args.checkpoint,
        sampling_steps=analyzer.sampling_steps,
    )
    print(f"saved: {args.report}")


if __name__ == "__main__":
    main()

"""Compare two VAE checkpoints on the same fixed test reconstruction set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from config import DataConfig
from dataset import ImgDataset
from utils import get_logger
from vae import LPIPSLoss, VAE, load_vae_from_ckpt, structural_similarity


logger = get_logger()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare old and new VAE reconstructions."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    lpips_metric: LPIPSLoss,
) -> dict[str, float]:
    """Calculate deterministic posterior-mean reconstruction metrics."""
    model.eval()
    sample_count = 0
    mse_sum = 0.0
    mae_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        mean, _ = model.encode(images)
        reconstructions = model.decode(mean)
        errors = (reconstructions.float() - images.float()).square()
        per_sample_mse = errors.flatten(start_dim=1).mean(dim=1)
        per_sample_mae = (
            reconstructions.float() - images.float()
        ).abs().flatten(start_dim=1).mean(dim=1)
        per_sample_psnr = -10.0 * torch.log10(
            per_sample_mse.clamp_min(1e-12)
        )

        batch_count = images.shape[0]
        sample_count += batch_count
        mse_sum += per_sample_mse.sum().item()
        mae_sum += per_sample_mae.sum().item()
        psnr_sum += per_sample_psnr.sum().item()
        ssim_sum += (
            structural_similarity(reconstructions, images).item() * batch_count
        )
        lpips_sum += lpips_metric(reconstructions, images).item() * batch_count

    if sample_count == 0:
        raise RuntimeError("test loader produced no samples")

    return {
        "mse": mse_sum / sample_count,
        "mae": mae_sum / sample_count,
        "psnr": psnr_sum / sample_count,
        "ssim": ssim_sum / sample_count,
        "lpips": lpips_sum / sample_count,
    }


@torch.inference_mode()
def save_comparison_grid(
    baseline: VAE,
    candidate: VAE,
    dataset: ImgDataset,
    output_path: Path,
    device: torch.device,
) -> None:
    """Save one fixed sample per class: original, baseline, and candidate."""
    class_indices = [
        int(label.argmax().item())
        for label in dataset.data_holder.label_tensor
    ]
    category_count = len(set(class_indices))
    selected: list[int] = []
    seen_classes: set[int] = set()
    for index, class_index in enumerate(class_indices):
        if class_index not in seen_classes:
            selected.append(index)
            seen_classes.add(class_index)
        if len(selected) == category_count:
            break
    if len(selected) != category_count:
        raise RuntimeError("could not find one sample for every class")

    images = torch.stack(
        [dataset.data_holder.img_tensor[index] for index in selected]
    ).to(device)
    baseline_mean, _ = baseline.encode(images)
    candidate_mean, _ = candidate.encode(images)
    baseline_images = baseline.decode(baseline_mean).float().cpu()
    candidate_images = candidate.decode(candidate_mean).float().cpu()
    grid = make_grid(
        torch.cat((images.cpu(), baseline_images, candidate_images), dim=0),
        nrow=category_count,
        padding=2,
    )
    save_image(grid, output_path)


def format_delta(
    baseline: float,
    candidate: float,
    *,
    lower_is_better: bool,
) -> str:
    delta = (candidate - baseline) / baseline * 100.0
    direction = "下降" if delta < 0 else "增加"
    better = delta < 0 if lower_is_better else delta > 0
    verdict = "改善" if better else "变差"
    return f"{delta:+.2f}%（{direction}，{verdict}）"


def write_report(
    output_path: Path,
    *,
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    baseline_params: int,
    candidate_params: int,
    baseline_channels: int,
    candidate_channels: int,
    baseline_latent_size: int,
    candidate_latent_size: int,
) -> None:
    lines = [
        "# VAE 重建对比",
        "",
        "两个模型使用相同的 32×32 测试集、posterior mean 重建和评估代码。"
        "MSE、MAE、LPIPS 越低越好，PSNR、SSIM 越高越好。",
        "",
        "| 配置 | latent channels | latent shape | parameters |",
        "|---|---:|---|---:|",
        f"| baseline | {baseline_channels} | `{baseline_channels}×{baseline_latent_size}×{baseline_latent_size}` | {baseline_params:,} |",
        f"| candidate | {candidate_channels} | `{candidate_channels}×{candidate_latent_size}×{candidate_latent_size}` | {candidate_params:,} |",
        "",
        "| metric | baseline | candidate | change |",
        "|---|---:|---:|---|",
        (
            f"| MSE | {baseline_metrics['mse']:.6f} | "
            f"{candidate_metrics['mse']:.6f} | "
            f"{format_delta(baseline_metrics['mse'], candidate_metrics['mse'], lower_is_better=True)} |"
        ),
        (
            f"| MAE | {baseline_metrics['mae']:.6f} | "
            f"{candidate_metrics['mae']:.6f} | "
            f"{format_delta(baseline_metrics['mae'], candidate_metrics['mae'], lower_is_better=True)} |"
        ),
        (
            f"| PSNR | {baseline_metrics['psnr']:.4f} | "
            f"{candidate_metrics['psnr']:.4f} | "
            f"{format_delta(baseline_metrics['psnr'], candidate_metrics['psnr'], lower_is_better=False)} |"
        ),
        (
            f"| SSIM | {baseline_metrics['ssim']:.6f} | "
            f"{candidate_metrics['ssim']:.6f} | "
            f"{format_delta(baseline_metrics['ssim'], candidate_metrics['ssim'], lower_is_better=False)} |"
        ),
        (
            f"| LPIPS | {baseline_metrics['lpips']:.6f} | "
            f"{candidate_metrics['lpips']:.6f} | "
            f"{format_delta(baseline_metrics['lpips'], candidate_metrics['lpips'], lower_is_better=True)} |"
        ),
        "",
        "![VAE reconstruction comparison](reconstruction_comparison.png)",
        "",
        "图片从上到下依次是三行：原始测试图、baseline 重建、candidate 重建；"
        "每一列对应一个类别，使用完全相同的输入与 posterior mean。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_config = DataConfig()
    dataset = ImgDataset(data_config.test_dir, data_config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    baseline = load_vae_from_ckpt(str(args.baseline)).to(device).eval()
    candidate = load_vae_from_ckpt(str(args.candidate)).to(device).eval()
    lpips_metric = LPIPSLoss(device=device).eval()
    if baseline.image_size != candidate.image_size:
        raise ValueError(
            "checkpoint image sizes differ: "
            f"{baseline.image_size} and {candidate.image_size}"
        )
    baseline_metrics = evaluate(baseline, loader, device, lpips_metric)
    candidate_metrics = evaluate(candidate, loader, device, lpips_metric)
    baseline_params = parameter_count(baseline)
    candidate_params = parameter_count(candidate)
    save_comparison_grid(
        baseline,
        candidate,
        dataset,
        args.output_dir / "reconstruction_comparison.png",
        device,
    )

    payload = {
        "baseline": {
            "checkpoint": str(args.baseline),
            "latent_channels": baseline.latent_channels,
            "latent_size": baseline.latent_size,
            "parameters": baseline_params,
            "metrics": baseline_metrics,
        },
        "candidate": {
            "checkpoint": str(args.candidate),
            "latent_channels": candidate.latent_channels,
            "latent_size": candidate.latent_size,
            "parameters": candidate_params,
            "metrics": candidate_metrics,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "README.md",
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_params=baseline_params,
        candidate_params=candidate_params,
        baseline_channels=baseline.latent_channels,
        candidate_channels=candidate.latent_channels,
        baseline_latent_size=baseline.latent_size,
        candidate_latent_size=candidate.latent_size,
    )
    logger.info("saved comparison to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

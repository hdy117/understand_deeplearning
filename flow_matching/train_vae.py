"""Train the VAE that supplies latent tokens to flow matching."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from config import FMConfig
from utils import get_logger, set_seed
from vae import VAE, vae_loss


logger = get_logger()


class ImageFolderWithFloatImages(ImageFolder):
    """Keep ImageFolder labels while returning NCHW floats in [0, 1]."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = self.loader(path)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(image.height, image.width, len(image.getbands()))
        pixels = pixels.permute(2, 0, 1).float().div_(255.0)
        if self.transform is not None:
            pixels = self.transform(pixels)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return pixels, label


def build_dataset(data_dir: str) -> ImageFolderWithFloatImages:
    """Load RGB pixels in [0, 1] without changing the class-directory layout."""
    return ImageFolderWithFloatImages(data_dir)


def train_one_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: FMConfig,
) -> dict[str, float]:
    """Run one training epoch and return loss components."""
    model.train()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg.vae_train_cfg.amp and device.type == "cuda",
        ):
            reconstruction, mean, logvar, _ = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(
                images,
                reconstruction,
                mean,
                logvar,
                kl_weight=cfg.vae_train_cfg.kl_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"] += loss.detach().item()
        totals["reconstruction"] += reconstruction_loss.detach().item()
        totals["kl"] += kl_loss.detach().item()
        batches += 1

    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    cfg: FMConfig,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate reconstruction and save a small image comparison."""
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    batches = 0
    sample_images: torch.Tensor | None = None
    sample_reconstruction: torch.Tensor | None = None

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg.vae_train_cfg.amp and device.type == "cuda",
        ):
            reconstruction, mean, logvar, _ = model(images, sample_latent=False)
            loss, reconstruction_loss, kl_loss = vae_loss(
                images,
                reconstruction,
                mean,
                logvar,
                kl_weight=cfg.vae_train_cfg.kl_weight,
            )
        if sample_images is None:
            sample_images = images.detach().cpu()
            sample_reconstruction = reconstruction.float().detach().cpu()
        totals["loss"] += loss.detach().item()
        totals["reconstruction"] += reconstruction_loss.detach().item()
        totals["kl"] += kl_loss.detach().item()
        batches += 1

    if batches == 0 or sample_images is None or sample_reconstruction is None:
        raise RuntimeError("validation loader produced no batches")
    metrics = {name: value / batches for name, value in totals.items()}
    return metrics, sample_images, sample_reconstruction


@torch.inference_mode()
def save_sample_comparison(
    images: torch.Tensor,
    reconstruction: torch.Tensor,
    path: Path,
) -> None:
    """Save top-row originals and bottom-row reconstructions."""
    count = min(images.shape[0], reconstruction.shape[0])
    if count == 0:
        return
    grid = make_grid(
        torch.cat([images[:count], reconstruction[:count]], dim=0),
        nrow=max(1, min(count, 8)),
        padding=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)


def save_checkpoint(
    path: Path,
    model: VAE,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_validation_loss: float,
    cfg: FMConfig,
) -> None:
    """Save enough state to resume training or extract latent tokens later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    vae_config = asdict(cfg.vae_cfg)
    torch.save(
        {
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vae_config": vae_config,
            "image_size": cfg.train_cfg.img_size,
            "data_config": {
                "data_dir": str(cfg.data_cfg.data_dir),
                "train_dir": str(cfg.data_cfg.train_dir),
                "test_dir": str(cfg.data_cfg.test_dir),
            },
        },
        temporary_path,
    )
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the image-to-token VAE using config.py settings."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="optional checkpoint from which to resume",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = FMConfig()

    set_seed(cfg.vae_train_cfg.seed)
    device = torch.device(cfg.vae_train_cfg.device)
    output_dir = Path(cfg.vae_train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(cfg.data_cfg.train_dir)
    test_dataset = build_dataset(cfg.data_cfg.test_dir)
    common_loader_args = {
        "batch_size": cfg.vae_train_cfg.batch_size,
        "num_workers": cfg.vae_train_cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.vae_train_cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args,
    )

    model = VAE(config=cfg.vae_cfg, image_size=cfg.train_cfg.img_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.vae_train_cfg.lr,
        weight_decay=cfg.vae_train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.vae_train_cfg.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.vae_train_cfg.amp and device.type == "cuda",
    )

    start_epoch = 0
    best_validation_loss = math.inf
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(f"device:{device}")
    logger.info(f"image_size:{cfg.train_cfg.img_size}")
    logger.info(f"latent_channels:{cfg.vae_cfg.latent_channels}")
    logger.info(f"latent_size:{model.latent_size}")
    logger.info(f"parameters:{parameter_count}")
    logger.info(f"train:{len(train_dataset)}, test:{len(test_dataset)}")

    for epoch in range(start_epoch, cfg.vae_train_cfg.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
        )
        validation_metrics, sample_images, sample_reconstruction = evaluate(
            model, test_loader, device, cfg
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        logger.info(f"epoch:{epoch + 1}")
        logger.info(f"learning_rate:{learning_rate:.2e}")
        logger.info(f"train loss:{train_metrics['loss']:.6f}")
        logger.info(f"train reconstruction:{train_metrics['reconstruction']:.6f}")
        logger.info(f"train kl:{train_metrics['kl']:.4f}")
        logger.info(f"validation loss:{validation_metrics['loss']:.6f}")
        logger.info(
            f"validation reconstruction:{validation_metrics['reconstruction']:.6f}"
        )
        logger.info(f"validation kl:{validation_metrics['kl']:.4f}")

        save_sample_comparison(
            sample_images,
            sample_reconstruction,
            output_dir / "latest_reconstruction.png",
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                cfg=cfg,
            )
            save_sample_comparison(
                sample_images,
                sample_reconstruction,
                output_dir / "best_reconstruction.png",
            )
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            cfg=cfg,
        )

    logger.info(f"training complete, best checkpoint:{output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

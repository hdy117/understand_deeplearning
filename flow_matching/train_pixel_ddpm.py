"""Train a direct 32x32 pixel-space Conv2D U-Net DDPM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from config import (
    DDPMConfig,
    DataConfig,
    PixelDDPMTrainConfig,
    PixelFlowModelConfig,
    SAMPLES_PER_CLASS,
)
from dataset import ImgDataset
from pixel_ddpm import PixelDDPMUNet
from utils import Timer, get_logger, set_seed


logger = get_logger()


@dataclass
class TrainMeta:
    best_loss: float = 1e12
    next_epoch: int = 0


class TrainPixelDDPM:
    def __init__(
        self,
        *,
        pixel_config: PixelFlowModelConfig | None = None,
        ddpm_config: DDPMConfig | None = None,
        train_config: PixelDDPMTrainConfig | None = None,
        data_config: DataConfig | None = None,
    ) -> None:
        self.pixel_config = pixel_config or PixelFlowModelConfig()
        self.ddpm_config = ddpm_config or DDPMConfig()
        self.train_config = train_config or PixelDDPMTrainConfig()
        self.data_config = data_config or DataConfig()
        if self.data_config.image_size != self.pixel_config.image_size:
            raise ValueError(
                f"DataConfig.image_size={self.data_config.image_size} does not "
                "match PixelFlowModelConfig.image_size="
                f"{self.pixel_config.image_size}"
            )
        set_seed(self.train_config.seed)

        self.output_dir = Path(self.train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "pixel_ddpm_epoch.pt"
        self.best_checkpoint_path = self.output_dir / "pixel_ddpm_best.pt"
        self.sample_path = self.output_dir / "pixel_ddpm_epoch.png"
        self.best_sample_path = self.output_dir / "pixel_ddpm_best.png"

        train_dataset = ImgDataset(
            data_dir=self.data_config.train_dir,
            data_config=self.data_config,
        )
        loader_args = {
            "batch_size": self.train_config.batch_size,
            "num_workers": self.train_config.num_workers,
            "pin_memory": self.train_config.device != "cpu",
            "persistent_workers": self.train_config.num_workers > 0,
        }
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_args,
        )
        self.model = PixelDDPMUNet(
            pixel_config=self.pixel_config,
            ddpm_config=self.ddpm_config,
            device=self.train_config.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_config.epochs,
            eta_min=self.train_config.lr * 0.05,
        )
        self.train_meta = TrainMeta()
        self.timer = Timer()
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(
            f"model: PixelDDPMUNet, parameters: {parameter_count}, "
            f"image: {self.pixel_config.image_channels}x"
            f"{self.pixel_config.image_size}x{self.pixel_config.image_size}, "
            f"timesteps: {self.ddpm_config.timesteps}"
        )

    def train(self) -> None:
        device_type = torch.device(self.train_config.device).type
        autocast_enabled = self.train_config.amp and device_type == "cuda"
        for epoch in range(self.train_meta.next_epoch, self.train_config.epochs):
            logger.info("================================")
            logger.info(
                f"current epoch:{epoch}, total:{self.train_config.epochs}"
            )
            self.timer.start()
            self.model.train()
            loss_sum = 0.0

            for images, labels in self.train_loader:
                images = images.to(self.train_config.device, non_blocking=True)
                labels = labels.to(self.train_config.device, non_blocking=True)
                batch_size = images.shape[0]
                x0 = self.model.backbone.images_to_flow(images)
                noise = torch.randn_like(x0)
                timesteps = torch.randint(
                    0,
                    self.ddpm_config.timesteps,
                    (batch_size,),
                    dtype=torch.long,
                    device=x0.device,
                )
                xt = self.model.q_sample(x0, timesteps, noise)

                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    predicted_noise = self.model.predict_noise(
                        xt,
                        timesteps,
                        labels,
                    )
                    loss = F.mse_loss(predicted_noise.float(), noise.float())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()
                loss_sum += loss.item()

            mean_loss = loss_sum / max(len(self.train_loader), 1)
            logger.info(f"epoch:{epoch}, mean loss:{mean_loss:.6f}")

            # Checkpoints describe the next epoch, including the best model.
            self.scheduler.step()
            self.train_meta.next_epoch = epoch + 1
            if (
                epoch >= self.train_config.epochs * 0.5
                and mean_loss < self.train_meta.best_loss
            ):
                self.train_meta.best_loss = mean_loss
                self.save_checkpoint(self.best_checkpoint_path)
                self.sample(self.best_sample_path)
                logger.info(f"best loss:{self.train_meta.best_loss:.6f}")

            self.save_checkpoint(self.checkpoint_path)
            if (
                epoch % self.train_config.preview_interval == 0
                or epoch + 1 == self.train_config.epochs
            ):
                self.sample(self.sample_path)
            logger.info(f"epoch:{epoch} time:{self.timer.stop()} s")

    @torch.inference_mode()
    def sample(self, output_path: Path) -> None:
        self.model.eval()
        labels = torch.eye(
            self.pixel_config.category_num,
            device=self.train_config.device,
        ).repeat_interleave(SAMPLES_PER_CLASS, dim=0)
        generator = torch.Generator(device=self.train_config.device).manual_seed(
            self.ddpm_config.sampling_seed
        )
        flow_images = self.model.sample(labels, generator=generator)
        images = self.model.backbone.flow_to_images(flow_images)
        torchvision.utils.save_image(
            images,
            output_path,
            nrow=SAMPLES_PER_CLASS,
        )
        logger.info(f"save {output_path}")

    def save_checkpoint(self, output_path: Path) -> None:
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_type": "pixel_ddpm_conv2d_unet",
                "next_epoch": self.train_meta.next_epoch,
                "best_loss": self.train_meta.best_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "pixel_model_config": asdict(self.pixel_config),
                "ddpm_config": asdict(self.ddpm_config),
                "pixel_ddpm_train_config": asdict(self.train_config),
            },
            temporary_path,
        )
        temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a direct pixel-space Conv2D U-Net DDPM."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pixel_config = PixelFlowModelConfig()
    ddpm_config = DDPMConfig()
    train_config = PixelDDPMTrainConfig()
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        train_config.epochs = args.epochs
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        if args.lr <= 0:
            raise ValueError("--lr must be positive")
        train_config.lr = args.lr
    if args.timesteps is not None:
        if args.timesteps <= 1:
            raise ValueError("--timesteps must be greater than one")
        ddpm_config.timesteps = args.timesteps
    if args.output_dir is not None:
        train_config.output_dir = str(args.output_dir)

    trainer = TrainPixelDDPM(
        pixel_config=pixel_config,
        ddpm_config=ddpm_config,
        train_config=train_config,
    )
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False)
        if checkpoint.get("model_type") != "pixel_ddpm_conv2d_unet":
            raise ValueError(f"{args.resume} is not a Pixel DDPM checkpoint")
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.scheduler.T_max = train_config.epochs
        trainer.train_meta.next_epoch = int(checkpoint["next_epoch"])
        trainer.train_meta.best_loss = float(checkpoint["best_loss"])
        logger.info(
            f"resumed from {args.resume} at epoch {trainer.train_meta.next_epoch}"
        )
    trainer.train()


if __name__ == "__main__":
    main()

from numpy import fabs

from utils import get_logger, set_seed, Timer
from config import (
    FlowMatchingTrainConfig,
    FMConfig,
    DataConfig,
    FlowMatchingModelConfig,
    SAMPLES_PER_CLASS,
)
from dataset import ImgDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flow_matching import FlowMatching
from vae import VAE, load_vae_from_ckpt
from pathlib import Path
import torchvision
from dataclasses import asdict, dataclass
import argparse, os

logger = get_logger()


@dataclass
class TrainFMMeta:
    best_loss: float = 1e12
    next_epoch: int = 0


class TrainFlowMatching(nn.Module):
    def __init__(self, fm_cfg: FMConfig = None):
        super().__init__()

        # check flow matching cofnig
        self.fm_cfg: FMConfig = fm_cfg

        if self.fm_cfg is None:
            err = "Flow matching config is not provided."
            logger.error(err)
            raise ValueError(err)

        self.data_cfg: DataConfig = self.fm_cfg.data_cfg
        self.model_cfg: FlowMatchingModelConfig = self.fm_cfg.fm_model_cfg
        self.train_cfg: FlowMatchingTrainConfig = self.fm_cfg.train_cfg
        if self.data_cfg.image_size != self.train_cfg.img_size:
            raise ValueError(
                f"DataConfig.image_size={self.data_cfg.image_size} does not match "
                f"FlowMatchingTrainConfig.img_size={self.train_cfg.img_size}"
            )

        # set random seed
        set_seed(self.fm_cfg.train_cfg.seed)

        # dataset and dataloader
        train_dataset = ImgDataset(
            data_dir=self.data_cfg.train_dir, data_config=self.data_cfg
        )
        test_dataset = ImgDataset(
            data_dir=self.data_cfg.test_dir, data_config=self.data_cfg
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.train_cfg.num_workers > 0,
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.train_cfg.num_workers > 0,
        )

        # model
        self.flow_net: FlowMatching = FlowMatching(
            model_config=self.model_cfg, device=self.train_cfg.device
        )
        self.output_dir = Path(self.train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flow_net_path = self.output_dir / "flow_matching_epoch.pt"
        self.flow_net_eval_img = self.output_dir / "flow_matching_epoch.png"
        self.flow_net_path_best = self.output_dir / "flow_matching_best.pt"
        self.flow_net_eval_img_best = self.output_dir / "flow_matching_best.png"
        self.train_meta: TrainFMMeta = TrainFMMeta()

        # load vae model for encoder and decoder
        self.vae: VAE = None
        self.vae_ckpt_path = Path(self.train_cfg.vae_checkpoint)
        self.vae = load_vae_from_ckpt(self.vae_ckpt_path)
        if self.vae is None:
            err = f"Failed to load vae model from {self.vae_ckpt_path}"
            logger.error(err)
            raise ValueError(err)

        # vae to cuda and to eval mode
        self.vae = self.vae.to(self.train_cfg.device)
        self.vae.eval()
        if self.vae.image_size != self.train_cfg.img_size:
            raise ValueError(
                f"{self.vae_ckpt_path} expects {self.vae.image_size}x"
                f"{self.vae.image_size} images, but training is configured for "
                f"{self.train_cfg.img_size}x{self.train_cfg.img_size}; retrain "
                "the VAE with the current data"
            )
        if self.vae.latent_channels != self.model_cfg.latent_channels:
            raise ValueError(
                f"VAE latent_channels={self.vae.latent_channels} does not match "
                "FlowMatchingModelConfig.latent_channels="
                f"{self.model_cfg.latent_channels}"
            )
        expected_tokens = self.vae.latent_size * self.vae.latent_size
        if self.model_cfg.num_tokens != expected_tokens:
            raise ValueError(
                f"FlowMatchingModelConfig.num_tokens={self.model_cfg.num_tokens} "
                "does not match VAE latent grid "
                f"{self.vae.latent_size}x{self.vae.latent_size}="
                f"{expected_tokens} tokens"
            )

        # The VAE was trained with a small KL weight, so its latent channels do
        # not share the standard-normal scale assumed by the flow source.  Work
        # in a whitened latent coordinate system and undo it before decoding.
        latent_mean, latent_std = self._estimate_latent_stats()
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)
        logger.info(
            "latent_mean: "
            f"{[round(value, 6) for value in latent_mean.flatten().tolist()]}"
        )
        logger.info(
            "latent_std: "
            f"{[round(value, 6) for value in latent_std.flatten().tolist()]}"
        )

        # loss function
        self.loss_fn = F.mse_loss

        # optimzer
        self.optimizer = torch.optim.AdamW(
            self.flow_net.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=1e-3,
        )

        # One long cosine schedule avoids the repeated high-LR jolts caused by
        # warm restarts every ten epochs.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.train_cfg.epoches,
            eta_min=self.train_cfg.lr * 0.05,
        )

        # timer
        self.timer = Timer()

    def _estimate_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-channel mean/std of deterministic VAE training latents."""
        channel_count = self.model_cfg.latent_channels
        latent_sum = torch.zeros(
            channel_count,
            dtype=torch.float64,
            device=self.train_cfg.device,
        )
        latent_square_sum = torch.zeros_like(latent_sum)
        sample_count = 0
        timer = Timer()
        timer.start()

        with torch.inference_mode():
            for img_batch, _ in self.train_dataloader:
                img_batch = img_batch.to(self.train_cfg.device, non_blocking=True)
                latent_batch = self.vae.encode_tokens(img_batch).reshape(
                    -1, channel_count
                )
                latent_batch = latent_batch.to(torch.float64)
                latent_sum += latent_batch.sum(dim=0)
                latent_square_sum += latent_batch.square().sum(dim=0)
                sample_count += latent_batch.shape[0]

        if sample_count == 0:
            raise RuntimeError("cannot estimate latent statistics from an empty loader")

        mean = latent_sum / sample_count
        variance = torch.clamp(
            latent_square_sum / sample_count - mean.square(),
            min=1e-8,
        )
        std = variance.sqrt()
        elapsed = timer.stop()
        logger.info(
            f"estimated VAE latent statistics from {sample_count} token values "
            f"in {elapsed} s"
        )
        return (
            mean.float().reshape(1, 1, channel_count),
            std.float().reshape(1, 1, channel_count),
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Map VAE latents into the standard-normal-scale flow coordinates."""
        return (latents - self.latent_mean) / self.latent_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Map flow coordinates back to the VAE decoder's latent scale."""
        return latents * self.latent_std + self.latent_mean

    def train_model(self):
        # train
        for epoch in range(self.train_meta.next_epoch, self.train_cfg.epoches):
            logger.info(f"=================")
            logger.info(f"current epoch:{epoch}, total:{self.train_cfg.epoches}")

            accu_loss = 0.0
            self.timer.start()
            self.flow_net.train()

            for index, (img_batch, label_batch) in enumerate(self.train_dataloader):
                # load batches from dataloader and send to device
                img_batch: torch.Tensor = img_batch.to(self.train_cfg.device)
                label_batch: torch.Tensor = label_batch.to(self.train_cfg.device)

                # shape
                B = img_batch.shape[0]

                # Encode each image into the VAE's latent token grid.
                with torch.no_grad():
                    x1 = self.normalize_latents(self.vae.encode_tokens(img_batch))

                # Sample continuous flow time and build the interpolation.
                x0 = torch.randn_like(x1, device=x1.device)
                T = torch.rand((B, 1), device=x1.device)
                xt, target_velocity = self.flow_net.get_xt_and_xt_dot(
                    x0=x0,
                    x1=x1,
                    T=T,
                )

                # Keep velocity prediction in BF16 and the loss in float32.
                device_type = torch.device(self.train_cfg.device).type
                autocast_enabled = (
                    self.train_cfg.amp
                    and device_type == "cuda"
                )
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    pred_velocity = self.flow_net.forward(xt, T, label_batch)
                    loss = self.loss_fn(
                        pred_velocity.float(),
                        target_velocity.float(),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.flow_net.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

                # save train meta
                accu_loss = accu_loss + loss.item()

            mean_loss = accu_loss / max(len(self.train_dataloader), 1)
            logger.info(f"epoch:{epoch}, mean loss:{mean_loss:.6f}")

            # Advance the schedule and resume position before saving any
            # checkpoint, including the best checkpoint.
            self.scheduler.step()
            self.train_meta.next_epoch = epoch + 1

            # Save the best model under the same post-epoch state as the
            # regular checkpoint so resuming never repeats a completed epoch.
            if (epoch >= self.train_cfg.epoches * 0.5) and (
                mean_loss < self.train_meta.best_loss
            ):
                self.train_meta.best_loss = mean_loss
                self.save_checkpoint(self.flow_net_path_best)
                self.sample(
                    flow_net=self.flow_net,
                    img_path=self.flow_net_eval_img_best,
                )
                logger.info(f"best loss:{self.train_meta.best_loss}")

            # save ckpt and sample
            self.save_checkpoint(self.flow_net_path)
            self.sample(flow_net=self.flow_net, img_path=self.flow_net_eval_img)

            elsaped = self.timer.stop()
            logger.info(f"""epoch:{epoch} time:{elsaped} s""")

    @torch.inference_mode()
    def sample(self, flow_net: FlowMatching, img_path: str = None):
        img_path = Path(img_path) if img_path is not None else self.flow_net_eval_img

        # eval only
        flow_net.eval()

        # sampling
        one_step = 1.0 / self.model_cfg.total_steps
        batch_size = self.model_cfg.category_num * SAMPLES_PER_CLASS
        token_count = self.vae.latent_size * self.vae.latent_size
        label_batch = torch.eye(
            self.model_cfg.category_num,
            device=self.train_cfg.device,
        ).repeat_interleave(SAMPLES_PER_CLASS, dim=0)
        sampling_generator = torch.Generator(device=self.train_cfg.device).manual_seed(
            self.model_cfg.sampling_seed
        )
        latent = torch.randn(
            size=[batch_size, token_count, self.model_cfg.latent_channels],
            device=self.train_cfg.device,
            generator=sampling_generator,
        )

        # Integrate the learned velocity field from t=0 to t=1.
        for step in range(self.model_cfg.total_steps):
            T = torch.full(
                size=[batch_size, 1],
                fill_value=step * one_step,
                device=self.train_cfg.device,
            )
            latent = latent + one_step * flow_net.forward(
                latent,
                T,
                label_batch,
            )

        img = self.vae.decode_tokens(self.denormalize_latents(latent))
        torchvision.utils.save_image(img, img_path, nrow=SAMPLES_PER_CLASS)
        logger.info(f"save {img_path}")

    def save_checkpoint(self, out_path: str | Path = None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = out_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "next_epoch": self.train_meta.next_epoch,
                "best_loss": self.train_meta.best_loss,
                "model": self.flow_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "fm_model_config": asdict(self.model_cfg),
                "latent_mean": self.latent_mean.cpu(),
                "latent_std": self.latent_std.cpu(),
            },
            temporary_path,
        )
        temporary_path.replace(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        nargs="?",
        const="best",
        default=None,
        help=(
            "resume from a checkpoint; omit the value to use "
            "the configured output directory's flow_matching_best.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override FlowMatchingTrainConfig.epoches for this run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override the checkpoint and sample output directory",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="override FlowMatchingModelConfig.num_tokens for this run",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=None,
        help="override FlowMatchingModelConfig.latent_channels for this run",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="override the frozen VAE checkpoint used for encoding and decoding",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override FlowMatchingTrainConfig.batch_size for this run",
    )

    args = parser.parse_args()

    fm_cfg = FMConfig()
    if args.epochs is not None:
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        fm_cfg.train_cfg.epoches = args.epochs
    if args.output_dir is not None:
        fm_cfg.train_cfg.output_dir = str(args.output_dir)
    if args.vae_checkpoint is not None:
        fm_cfg.train_cfg.vae_checkpoint = str(args.vae_checkpoint)
    if args.num_tokens is not None:
        if args.num_tokens <= 0:
            parser.error("--num-tokens must be positive")
        fm_cfg.fm_model_cfg.num_tokens = args.num_tokens
    if args.latent_channels is not None:
        if args.latent_channels <= 0:
            parser.error("--latent-channels must be positive")
        fm_cfg.fm_model_cfg.latent_channels = args.latent_channels
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        fm_cfg.train_cfg.batch_size = args.batch_size
    train_flow_matching: TrainFlowMatching = TrainFlowMatching(fm_cfg=fm_cfg)

    if args.resume is not None:
        ckpt_path = (
            os.path.join(
                fm_cfg.train_cfg.output_dir,
                "flow_matching_best.pt",
            )
            if args.resume == "best"
            else args.resume
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        train_flow_matching.flow_net.load_state_dict(ckpt["model"])
        train_flow_matching.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            train_flow_matching.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Older checkpoints did not save scheduler state. Fast-forward a
            # fresh one-shot cosine schedule to the resumed epoch.
            for _ in range(int(ckpt["next_epoch"])):
                train_flow_matching.scheduler.step()
        if "latent_mean" in ckpt and "latent_std" in ckpt:
            train_flow_matching.latent_mean.copy_(
                ckpt["latent_mean"].to(train_flow_matching.latent_mean.device)
            )
            train_flow_matching.latent_std.copy_(
                ckpt["latent_std"].to(train_flow_matching.latent_std.device)
            )
        train_flow_matching.train_meta.next_epoch = ckpt["next_epoch"]
        train_flow_matching.train_meta.best_loss = ckpt["best_loss"]

    # train
    train_flow_matching.train()
    train_flow_matching.train_model()


if __name__ == "__main__":
    main()

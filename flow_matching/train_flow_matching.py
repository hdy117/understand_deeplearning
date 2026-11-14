from utils import get_logger, set_seed, Timer
from config import (
    FlowMatchingTrainConfig,
    FMConfig,
    DataConfig,
    FlowMatchingModelConfig,
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
from dataclasses import dataclass

logger = get_logger()


@dataclass
class TrainDitMeta:
    best_loss: float = 1e12
    next_epoch: int = 0


class TrainDit(nn.Module):
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
        self.output_dir = Path(self.fm_cfg.vae_train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flow_net_path = self.output_dir / "flow_matching.pt"
        self.flow_net_eval_img = self.output_dir / "flow_matching_reconstruction.png"
        self.train_meta: TrainDitMeta = TrainDitMeta()

        # load vae model for encoder and decoder
        self.vae: VAE = None
        self.vae_ckpt_path = self.output_dir / "vae_best.pt"
        self.vae = load_vae_from_ckpt(self.vae_ckpt_path)
        if self.vae is None:
            err = f"Failed to load vae model from {self.vae_ckpt_path}"
            logger.error(err)
            raise ValueError(err)

        # vae to cuda and to eval mode
        self.vae = self.vae.to(self.train_cfg.device)
        self.vae.eval()

        # loss function
        self.loss_fn = F.mse_loss

        # optimzer
        self.optimizer = torch.optim.AdamW(
            self.flow_net.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=1e-3,
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=self.train_cfg.lr * 0.1,
        )

        # Additional artifact paths retained for future evaluation workflows.
        self.dit_best_ckpt = self.output_dir / "dit_best_ckpt.pth"
        self.dit_last_ckpt = self.output_dir / "dit_last_ckpt.pth"
        self.dit_best_reconstruction = self.output_dir / "dit_best_reconstruction.png"
        self.dit_last_reconstruction = self.output_dir / "dit_last_reconstruction.png"

        # timer
        self.timer = Timer()

    def train(self):
        self.flow_net.train()
        # train
        for epoch in range(self.train_cfg.epoches):
            logger.info(f"=================")
            logger.info(f"current epoch:{epoch}, total:{self.train_cfg.epoches}")
            accu_loss = 0.0
            for index, (img_batch, label_batch) in enumerate(self.train_dataloader):
                # load batches from dataloader and send to device
                img_batch: torch.Tensor = img_batch.to(self.train_cfg.device)
                label_batch: torch.Tensor = label_batch.to(self.train_cfg.device)

                # shape
                B = img_batch.shape[0]

                # encode image to latent tokens, [batch, 64, 4]
                with torch.no_grad():
                    x1 = self.vae.encode_tokens(img_batch)
                x0 = torch.randn_like(x1, device=x1.device)

                # Sample continuous flow time and build the interpolation.
                T = torch.rand(B, device=x1.device)
                xt, target_velocity = self.flow_net.get_xt_and_xt_dot(
                    x0=x0,
                    x1=x1,
                    T=T,
                )

                # model predict velocity
                pred_velocity = self.flow_net.forward(xt, T, label_batch)

                # loss
                loss = self.loss_fn(pred_velocity, target_velocity)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                # save train meta
                accu_loss = accu_loss + loss.item()
                if (epoch >= self.train_cfg.epoches * 0.8) and (
                    loss.item() < self.train_meta.best_loss
                ):
                    self.train_meta.best_loss = loss.item()
                    self.save_checkpoint()
                    logger.info(f"best loss:{self.train_meta.best_loss}")

            mean_loss = accu_loss / max(len(self.train_dataloader), 1)
            logger.info(f"epoch:{epoch + 1} mean loss:{mean_loss:.6f}")
            # scheduler for lr
            self.scheduler.step()
            self.train_meta.next_epoch = epoch + 1

    @torch.inference_mode()
    def eval(self):
        ckpt = torch.load(
            self.flow_net_path,
            map_location=self.train_cfg.device,
            weights_only=True,
        )
        flow_net = FlowMatching(
            model_config=self.model_cfg,
            device=self.train_cfg.device,
        )
        flow_net.load_state_dict(ckpt["model"])

        flow_net.eval()
        one_step = 1.0 / self.model_cfg.total_steps
        batch_size = self.model_cfg.category_num
        token_count = self.vae.latent_size * self.vae.latent_size
        label_batch = torch.eye(
            batch_size,
            device=self.train_cfg.device,
        )
        latent = torch.randn(
            size=[batch_size, token_count, self.model_cfg.latent_channels],
            device=self.train_cfg.device,
        )

        # Integrate the learned velocity field from t=0 to t=1.
        for step in range(self.model_cfg.total_steps):
            T = torch.full(
                size=[batch_size],
                fill_value=step * one_step,
                device=self.train_cfg.device,
            )
            latent = latent + one_step * flow_net.forward(
                latent,
                T,
                label_batch,
            )

        img = self.vae.decode_tokens(latent)
        torchvision.utils.save_image(img, self.flow_net_eval_img, nrow=5)
        logger.info(f"save {self.flow_net_eval_img}")

    def save_checkpoint(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.flow_net_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "next_epoch": self.train_meta.next_epoch,
                "best_loss": self.train_meta.best_loss,
                "model": self.flow_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            temporary_path,
        )
        temporary_path.replace(self.flow_net_path)


if __name__ == "__main__":
    fm_cfg = FMConfig()
    train_dit = TrainDit(fm_cfg=fm_cfg)
    train_dit.train()
    train_dit.save_checkpoint()
    train_dit.eval()

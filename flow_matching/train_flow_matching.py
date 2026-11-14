from utils import get_logger, set_seed, Timer
from config import (
    DiTTrainConfig,
    FMConfig,
    DataConfig,
    DiTModelConfig,
    DEVICE,
    IMG_SIZE,
)
from dataset import ImgDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dit import FlowMatching
from vae import VAE, load_vae_from_ckpt
from pathlib import Path
import os
import torchvision
from dataclasses import dataclass

logger = get_logger()


@dataclass
class TrainDitMeta:
    best_loss: float = 1e12
    cur_epoch: int = 0


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
        self.model_cfg: DiTModelConfig = self.fm_cfg.model_cfg
        self.train_cfg: DiTTrainConfig = self.fm_cfg.train_cfg

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
            persistent_workers=self.train_cfg.num_workers / 2,
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.train_cfg.batch_size,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.train_cfg.num_workers / 2,
        )

        # model
        self.flow_net: FlowMatching = FlowMatching(
            model_config=self.model_cfg, device=self.train_cfg.device
        )
        self.flow_net_path = (
            os.path.join(self.fm_cfg.vae_train_cfg.output_dir, "flow_matching.pt"),
        )
        self.flow_net_eval_img = os.path.join(
            os.path.join(
                self.fm_cfg.vae_train_cfg.output_dir, "flow_matching_reconstruction.png"
            ),
        )
        self.train_meta: TrainDitMeta = TrainDitMeta()

        # load vae model for encoder and decoder
        self.vae: VAE = None
        self.vae_ckpt_path = os.path.join(
            fm_cfg.vae_train_cfg.output_dir, "vae_best.pt"
        )
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
            optizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=self.train_cfg.lr * 0.1,
        )

        # dir and checkpoint
        self.output_dir = Path(__file__).parent / "outputs"
        self.dit_best_ckpt = self.output_dir / "dit_best_ckpt.pth"
        self.dit_last_ckpt = self.output_dir / "dit_last_ckpt.pth"
        self.dit_best_reconstruction = self.output_dir / "dit_best_reconstruction.png"
        self.dit_last_reconstruction = self.output_dir / "dit_last_reconstruction.png"

        # timer
        self.timer = Timer()

    def train(self):
        # one step size of sampler
        one_step = 1.0 / self.model_cfg.total_steps

        # train
        for epoch in range(self.train_cfg.epoches):
            for index, (img_batch, label_batch) in enumerate(self.train_dataloader):
                # load batches from dataloader and send to device
                img_batch: torch.Tensor = img_batch.to(self.train_cfg.device)
                label_batch: torch.Tensor = label_batch.to(self.train_cfg.device)

                # shape
                B = img_batch.shape[0]

                # encode image to latent tokens, [batch, 64, 4]
                x1 = self.vae.encode_tokens(img_batch)
                x0 = torch.randn_like(x1, device=x1.device)

                # target velocity
                target_velocity = x1 - x0

                # xt, interplotation by step T
                T = torch.rand(B, device=x1.device) * one_step
                xt = (1.0 - T[:, None]) * x0 + T[:, None] * x1

                # model predict velocity
                pred_velocity = self.flow_net.forward(xt, T)

                # loss
                loss = self.loss_fn(xt, pred_velocity).mean(dim=-1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # save train meta
                if loss.item() < self.train_meta.best_loss:
                    self.train_meta.best_loss = loss.item()
                    self.save_checkpoint()

            # scheduler for lr
            self.scheduler.step()

    def eval(self):
        ckpt = torch.load(self.flow_net_path)
        flow_net: FlowMatching = ckpt["model"]

        flow_net.eval()
        flow_net = flow_net.to(device=self.train_cfg.device)

        with torch.no_grad():
            # one step size of sampler
            one_step = torch.tensor(
                [1.0 / self.model_cfg.total_steps], device=self.train_cfg.device
            )

            # batch size
            B = 10

            # noise
            x0 = torch.randn(size=[B, 64, 4], device=self.train_cfg.device)

            # sample
            for t in range(self.model_cfg.total_steps):
                T = torch.ones(size=[B], device=self.train_cfg.device)
                pred_velocity = flow_net.forward(x0, T)
                x0 = x0 + one_step * pred_velocity

            # decoder
            img = self.vae.decode_tokens(x0)

            # save
            torchvision.utils.save_image(img, self.flow_net_eval_img)
            logger.info(f"save {self.flow_net_eval_img}")

    def save_checkpoint(self):
        torch.save(
            {
                "cur_epoch": self.train_meta.cur_epoch,
                "best_loss": self.train_meta.best_loss,
                "model": self.flow_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.flow_net_path,
        )

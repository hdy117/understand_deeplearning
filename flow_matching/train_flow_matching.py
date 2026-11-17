from numpy import fabs

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
        self.flow_net_path = self.output_dir / "flow_matching_epoch.pt"
        self.flow_net_eval_img = self.output_dir / "flow_matching_epoch.png"
        self.flow_net_path_best = self.output_dir / "flow_matching_best.pt"
        self.flow_net_eval_img_best = self.output_dir / "flow_matching_best.png"
        self.train_meta: TrainFMMeta = TrainFMMeta()

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
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=self.optimizer,
        #     T_max=self.train_cfg.epoches,
        #     eta_min=self.train_cfg.lr * 0.1,
        # )

        # timer
        self.timer = Timer()

    def train_model(self):
        # train
        for epoch in range(self.train_cfg.epoches):
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

                # encode image to latent tokens, [batch, 64, 4]
                with torch.no_grad():
                    x1 = self.vae.encode_tokens(img_batch)

                # Sample continuous flow time and build the interpolation.
                x0 = torch.randn_like(x1, device=x1.device)
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
                torch.nn.utils.clip_grad_norm_(
                    self.flow_net.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

                # save train meta
                accu_loss = accu_loss + loss.item()

            # save the best
            mean_loss = accu_loss / max(len(self.train_dataloader), 1)
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
            logger.info(f"epoch:{epoch}, mean loss:{mean_loss:.6f}")

            # scheduler for lr
            self.scheduler.step()
            self.train_meta.next_epoch = epoch + 1

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
        batch_size = self.model_cfg.category_num
        token_count = self.vae.latent_size * self.vae.latent_size
        label_batch = torch.eye(
            batch_size,
            device=self.train_cfg.device,
        )
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
        torchvision.utils.save_image(img, img_path, nrow=5)
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
            },
            temporary_path,
        )
        temporary_path.replace(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume from ckpt")

    args = parser.parse_args()

    fm_cfg = FMConfig()
    train_flow_matching: TrainFlowMatching = TrainFlowMatching(fm_cfg=fm_cfg)

    if args.resume:
        ckpt_path = os.path.join(
            fm_cfg.vae_train_cfg.output_dir, "flow_matching_best.pt"
        )
        ckpt = torch.load(ckpt_path)
        train_flow_matching.flow_net.load_state_dict(ckpt["model"])
        train_flow_matching.optimizer.load_state_dict(ckpt["optimizer"])
        train_flow_matching.train_meta.next_epoch = ckpt["next_epoch"]
        train_flow_matching.train_meta.best_loss = ckpt["best_loss"]

    # train
    train_flow_matching.train()
    train_flow_matching.train_model()


if __name__ == "__main__":
    main()

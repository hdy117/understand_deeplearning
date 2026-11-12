from utils import get_logger, set_seed
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

logger = get_logger()


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

        # loss function
        self.loss_fn = F.mse_loss

        # optimzer
        self.optimizer = torch.optim.AdamW(
            self.flow_net.parameters(),
            lr=self.train_cfg.lr,
        )

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optizer=self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=self.train_cfg.lr * 0.1,
        )

    def train(self):
        pass

    def eval(self):
        pass

    def save_checkpoint(self):
        pass

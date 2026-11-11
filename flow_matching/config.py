"""训练配置：dataclass 的典型用法 —— 只声明字段，行为由装饰器生成。"""

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from plistlib import InvalidFileException
from cv2 import log
import torch
import torch.nn as nn
from pathlib import Path
import os
from traitlets import default
import loguru

from utils import get_logger

logger = get_logger()

logger.info(f"py file path:{__file__}")


@dataclass
class TrainConfig:
    epoches: int = 1000
    batch_size: int = 64
    lr: float = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts | None = None
    optimizer: torch.optim.AdamW | None = None

    def setup_optimizer(self, optimizer: torch.optim.AdamW = None):
        if optimizer is None:
            err_info: str = "error, torch optimizer is none."
            logger.error(err_info)
            raise ValueError(err_info)
        else:
            self.optimizer = optimizer
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=1, eta_min=1e-4
            )


@dataclass
class FMDataConfig:
    data_dir: str = field(default=Path(__file__).parent / "data", init=True)
    train_dir: str | None = field(default="train", init=False)
    test_dir: str | None = field(default="test", init=False)

    def __post_init__(self) -> None:
        """calculate train_dir and test_dir based on data_dir at post init stage

        Args:
            data_dir (_type_): _description_

        Raises:
            InvalidFileException: _description_
            InvalidFileException: _description_
            InvalidFileException: _description_
            InvalidFileException: _description_
        """
        self.train_dir = os.path.join(self.data_dir, self.train_dir)
        self.test_dir = os.path.join(self.data_dir, self.test_dir)

        # path exists check
        if not os.path.exists(self.data_dir):
            err_info: str = f"path does not exist. {self.data_dir}"
            logger.error(err_info)
            raise InvalidFileException(err_info)
        if not os.path.exists(self.train_dir):
            err_info: str = f"path does not exist. {self.train_dir}"
            logger.error(err_info)
            raise InvalidFileException(err_info)
        if not os.path.exists(self.test_dir):
            err_info: str = f"path does not exist. {self.test_dir}"
            logger.error(err_info)
            raise InvalidFileException(err_info)

        # logger.info info
        logger.info(f"self.data_dir:{self.data_dir}")
        logger.info(f"self.train_dir:{self.train_dir}")
        logger.info(f"self.test_dir:{self.test_dir}")


@dataclass
class FMConfig:
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    data_cfg: FMDataConfig = field(default_factory=FMDataConfig)


if __name__ == "__main__":
    fm_cfg: FMConfig = FMConfig()
    logger.info(fm_cfg)

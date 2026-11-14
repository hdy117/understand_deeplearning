"""训练配置：dataclass 的典型用法 —— 只声明字段，行为由装饰器生成。"""

from dataclasses import dataclass, field
from typing import ClassVar
from plistlib import InvalidFileException
import torch
from pathlib import Path
import os

from utils import get_logger

logger = get_logger()

logger.info(f"py file path:{__file__}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE: int = 64


@dataclass
class DiTTrainConfig:
    epoches: int = 30
    batch_size: int = 64
    lr: float = 2e-4
    device = DEVICE
    img_size: int = IMG_SIZE
    num_workers: int = 16
    seed: int = 42


@dataclass
class DataConfig:
    data_dir: str = field(default=Path(__file__).parent / "data", init=True)
    train_dir: str | None = field(default="train", init=False)
    test_dir: str | None = field(default="test", init=False)
    category_num: ClassVar[int] = 10  # num of categories

    def __post_init__(self) -> None:
        """calculate train_dir and test_dir based on data_dir at post init stage

        Args:
            data_dir (_type_): train or test data dir

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
class DiTModelConfig:
    d_model: int = 64
    latent_channels: int = 4  # VAE token feature dimension
    n_layer: int = 6
    n_head: int = 8
    total_steps: int = 24
    category_num: int = DataConfig.category_num


@dataclass
class VAEConfig:
    """VAE architecture; the latent layout is part of the model interface."""

    in_channels: int = 3
    latent_channels: int = 4
    hidden_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    downsample_factor: int = 8


@dataclass
class VAETrainConfig:
    """The training script owns runtime-only settings."""

    epochs: int = 30
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = DEVICE
    num_workers: int = 16
    seed: int = 42
    amp: bool = True
    kl_weight: float = 1e-4
    output_dir: str = field(
        default_factory=lambda: str(Path(__file__).parent / "outputs")
    )
    sample_count: int = 16


@dataclass
class FMConfig:
    train_cfg: DiTTrainConfig = field(default_factory=DiTTrainConfig)
    fm_model_cfg: DiTModelConfig = field(default_factory=DiTModelConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    vae_cfg: VAEConfig = field(default_factory=VAEConfig)
    vae_train_cfg: VAETrainConfig = field(default_factory=VAETrainConfig)


if __name__ == "__main__":
    fm_cfg: FMConfig = FMConfig()
    logger.info(fm_cfg)

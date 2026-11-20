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
IMG_SIZE: int = 32
VAE_DOWNSAMPLE_FACTOR: int = 4
TRAIN_BATCH_SIZE: int = 512
TRAIN_EPOCHS: int = 200
RUN_TAG: str = f"bs{TRAIN_BATCH_SIZE}_e{TRAIN_EPOCHS}"
if (
    IMG_SIZE <= 0
    or VAE_DOWNSAMPLE_FACTOR <= 0
    or IMG_SIZE % VAE_DOWNSAMPLE_FACTOR != 0
):
    raise ValueError("IMG_SIZE must be divisible by VAE_DOWNSAMPLE_FACTOR")
LATENT_SIZE: int = IMG_SIZE // VAE_DOWNSAMPLE_FACTOR
LATENT_CHANNELS: int = 32
LATENT_TOKEN_COUNT: int = LATENT_SIZE * LATENT_SIZE
SAMPLES_PER_CLASS: int = 10
if SAMPLES_PER_CLASS <= 0:
    raise ValueError("SAMPLES_PER_CLASS must be positive")


@dataclass
class FlowMatchingTrainConfig:
    epoches: int = TRAIN_EPOCHS
    batch_size: int = TRAIN_BATCH_SIZE
    lr: float = 1e-3
    device = DEVICE
    amp: bool = True
    img_size: int = IMG_SIZE
    num_workers: int = 16
    seed: int = 42
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).parent
            / f"outputs_flow_transformer_latent{LATENT_CHANNELS}_ds4_{RUN_TAG}"
        )
    )
    vae_checkpoint: str = field(
        default_factory=lambda: str(
            Path(__file__).parent
            / f"outputs_vae_latent{LATENT_CHANNELS}_ds4_ssim01_lpips01_{RUN_TAG}"
            / "vae_best.pt"
        )
    )


@dataclass
class DataConfig:
    data_dir: str = field(default=Path(__file__).parent / "data", init=True)
    image_size: int = IMG_SIZE
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
        if self.image_size <= 0:
            raise ValueError("DataConfig.image_size must be positive")
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
class FlowMatchingModelConfig:
    d_model: int = 128
    latent_channels: int = LATENT_CHANNELS  # VAE token feature dimension
    num_tokens: int = LATENT_TOKEN_COUNT
    n_layer: int = 6
    n_head: int = 8
    conditioning: str = "ada_ln_zero"
    total_steps: int = 100
    sampling_seed: int = 42
    category_num: int = DataConfig.category_num


@dataclass
class DDPMConfig:
    """Discrete-time DDPM schedule shared by pixel-space diffusion models."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    sampling_seed: int = 42


@dataclass
class PixelFlowModelConfig:
    """Conv2D U-Net velocity model operating directly on 32x32 images."""

    image_channels: int = 3
    image_size: int = IMG_SIZE
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    blocks_per_level: int = 2
    condition_dim: int = 256
    category_num: int = DataConfig.category_num
    total_steps: int = 100
    sampling_seed: int = 42
    dropout: float = 0.0


@dataclass
class PixelFlowTrainConfig:
    epochs: int = TRAIN_EPOCHS
    batch_size: int = TRAIN_BATCH_SIZE
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = DEVICE
    amp: bool = True
    num_workers: int = 16
    seed: int = 42
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).parent / f"outputs_pixel_flow_unet_{RUN_TAG}"
        )
    )


@dataclass
class PixelDDPMTrainConfig:
    epochs: int = TRAIN_EPOCHS
    batch_size: int = TRAIN_BATCH_SIZE
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = DEVICE
    amp: bool = True
    num_workers: int = 16
    seed: int = 42
    preview_interval: int = 5
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).parent / f"outputs_pixel_ddpm_unet_{RUN_TAG}"
        )
    )


@dataclass
class VAEConfig:
    """VAE architecture; the latent layout is part of the model interface."""

    in_channels: int = 3
    # An integer uses a final 1x1 posterior projection. ``None`` keeps every
    # channel produced by the final encoder stage instead.
    latent_channels: int | None = LATENT_CHANNELS
    hidden_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    downsample_factor: int = VAE_DOWNSAMPLE_FACTOR


@dataclass
class VAETrainConfig:
    """The training script owns runtime-only settings."""

    epochs: int = TRAIN_EPOCHS
    batch_size: int = TRAIN_BATCH_SIZE
    lr: float = 2e-4
    weight_decay: float = 1e-4
    device: str = DEVICE
    num_workers: int = 16
    seed: int = 42
    amp: bool = True
    kl_weight: float = 1e-4
    l1_weight: float = 0.0
    ssim_weight: float = 0.1
    lpips_weight: float = 0.1
    output_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).parent
            / f"outputs_vae_latent{LATENT_CHANNELS}_ds4_ssim01_lpips01_{RUN_TAG}"
        )
    )
    sample_count: int = 16


@dataclass
class FMConfig:
    train_cfg: FlowMatchingTrainConfig = field(default_factory=FlowMatchingTrainConfig)
    fm_model_cfg: FlowMatchingModelConfig = field(
        default_factory=FlowMatchingModelConfig
    )
    data_cfg: DataConfig = field(default_factory=DataConfig)
    vae_cfg: VAEConfig = field(default_factory=VAEConfig)
    vae_train_cfg: VAETrainConfig = field(default_factory=VAETrainConfig)


if __name__ == "__main__":
    fm_cfg: FMConfig = FMConfig()
    logger.info(fm_cfg)

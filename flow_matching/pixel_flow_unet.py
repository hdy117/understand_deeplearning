"""Pixel-space Conv2D U-Net velocity field."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, PixelFlowModelConfig
from flow_matching import TimestepEmbedding
from pixel_unet_blocks import FiLMResBlock, group_norm


class PixelUNetFlowMatching(nn.Module):
    """Predict pixel-space velocity directly, without a VAE."""

    def __init__(
        self,
        model_config: PixelFlowModelConfig | None = None,
        device=DEVICE,
    ) -> None:
        super().__init__()
        config = model_config or PixelFlowModelConfig()
        self.config = config
        self.image_channels = config.image_channels
        self.image_size = config.image_size
        self.total_steps = config.total_steps
        self.category_num = config.category_num

        if self.image_channels <= 0 or self.image_size <= 0:
            raise ValueError("image_channels and image_size must be positive")
        if config.base_channels <= 0 or not config.channel_multipliers:
            raise ValueError("base_channels and channel_multipliers must be positive")
        if any(multiplier <= 0 for multiplier in config.channel_multipliers):
            raise ValueError("channel_multipliers must contain positive integers")
        if config.blocks_per_level <= 0 or config.condition_dim <= 0:
            raise ValueError("blocks_per_level and condition_dim must be positive")
        downsample_count = len(config.channel_multipliers) - 1
        required_divisor = 2**downsample_count
        if self.image_size % required_divisor != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"{required_divisor} for {len(config.channel_multipliers)} levels"
            )

        channels = [
            config.base_channels * multiplier
            for multiplier in config.channel_multipliers
        ]
        self.input_conv = nn.Conv2d(
            self.image_channels,
            channels[0],
            kernel_size=3,
            padding=1,
        )
        self.time_embed = TimestepEmbedding(config.condition_dim)
        self.label_embed = nn.Linear(self.category_num, config.condition_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2 * config.condition_dim, config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        current_channels = channels[0]
        for level, level_channels in enumerate(channels):
            level_blocks = []
            for _ in range(config.blocks_per_level):
                level_blocks.append(
                    FiLMResBlock(
                        current_channels,
                        level_channels,
                        config.condition_dim,
                        dropout=config.dropout,
                    )
                )
                current_channels = level_channels
            self.encoder_blocks.append(nn.ModuleList(level_blocks))
            if level < len(channels) - 1:
                self.downsamplers.append(
                    nn.Conv2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.up_convs.append(
                nn.Conv2d(
                    current_channels,
                    skip_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            decoder_level = []
            block_in_channels = 2 * skip_channels
            for _ in range(config.blocks_per_level):
                decoder_level.append(
                    FiLMResBlock(
                        block_in_channels,
                        skip_channels,
                        config.condition_dim,
                        dropout=config.dropout,
                    )
                )
                block_in_channels = skip_channels
            self.decoder_blocks.append(nn.ModuleList(decoder_level))
            current_channels = skip_channels

        self.output_norm = group_norm(current_channels)
        self.output_conv = nn.Conv2d(
            current_channels,
            self.image_channels,
            kernel_size=3,
            padding=1,
        )
        self.to(device)

    @staticmethod
    def images_to_flow(images: torch.Tensor) -> torch.Tensor:
        """Map image values in [0,1] to Flow coordinates in [-1,1]."""
        if images.ndim != 4:
            raise ValueError(f"images must be rank 4 NCHW, got {images.shape}")
        if not images.is_floating_point():
            raise ValueError("images must be floating point")
        if torch.any(images < 0.0) or torch.any(images > 1.0):
            raise ValueError("images must contain values in [0,1]")
        return images * 2.0 - 1.0

    @classmethod
    def flow_to_images(cls, flow_images: torch.Tensor) -> torch.Tensor:
        """Map Flow coordinates back to [0,1] images."""
        return (flow_images.clamp(-1.0, 1.0) + 1.0) * 0.5

    def _validate_flow_inputs(
        self,
        xt: torch.Tensor,
        time: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> None:
        expected = (self.image_channels, self.image_size, self.image_size)
        if xt.ndim != 4 or tuple(xt.shape[1:]) != expected:
            raise ValueError(
                f"xt must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(xt.shape)}"
            )
        if not xt.is_floating_point():
            raise ValueError("xt must be floating point")
        if (
            time.ndim != 2
            or time.shape != (xt.shape[0], 1)
            or time.device != xt.device
            or not time.is_floating_point()
            or not torch.all((time >= 0.0) & (time <= 1.0))
        ):
            raise ValueError(
                f"time must be floating [B,1] on {xt.device} in [0,1], "
                f"got {tuple(time.shape)} on {time.device}"
            )
        if (
            label_batch.ndim != 2
            or label_batch.shape != (xt.shape[0], self.category_num)
            or label_batch.device != xt.device
            or not label_batch.is_floating_point()
        ):
            raise ValueError(
                f"label_batch must be floating [{xt.shape[0]},{self.category_num}] "
                f"on {xt.device}, got {tuple(label_batch.shape)} on {label_batch.device}"
            )
        is_binary = torch.logical_or(label_batch == 0.0, label_batch == 1.0).all()
        is_single_class = torch.eq(label_batch.sum(dim=1), 1.0).all()
        if not is_binary or not is_single_class:
            raise ValueError("label_batch rows must each contain exactly one class")

    def forward(
        self,
        xt: torch.Tensor,
        T: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_flow_inputs(xt, T, label_batch)
        condition = self.condition_mlp(
            torch.cat(
                (self.time_embed(T[:, 0]), self.label_embed(label_batch)),
                dim=-1,
            )
        )

        hidden = self.input_conv(xt)
        skips: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                hidden = block(hidden, condition)
            skips.append(hidden)
            if level < len(self.downsamplers):
                hidden = self.downsamplers[level](hidden)

        for level, blocks in enumerate(self.decoder_blocks):
            hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
            hidden = self.up_convs[level](hidden)
            skip = skips[-2 - level]
            if hidden.shape != skip.shape:
                raise RuntimeError(
                    f"decoder shape {tuple(hidden.shape)} does not match "
                    f"skip shape {tuple(skip.shape)}"
                )
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, condition)

        return self.output_conv(F.silu(self.output_norm(hidden)))

    def get_xt_and_xt_dot(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}"
            )
        time = T.reshape(-1, *([1] * (x0.ndim - 1)))
        return (1.0 - time) * x0 + time * x1, x1 - x0


if __name__ == "__main__":
    model = PixelUNetFlowMatching(PixelFlowModelConfig())
    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))

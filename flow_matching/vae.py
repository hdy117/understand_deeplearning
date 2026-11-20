"""A convolutional VAE whose latent grid can feed a token Transformer.

Image tensors use ``float32`` values in ``[0, 1]`` and NCHW layout.  The
default image size is 32x32, the spatial compression is 8x, and the default
latent width is 32 channels.  Setting ``latent_channels=None`` retains every
channel from the final encoder stage by splitting its 256 outputs directly
into Gaussian mean and log-variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMG_SIZE, VAEConfig
import os
from utils import get_logger

logger = get_logger()


class LPIPSLoss(nn.Module):
    """Differentiable LPIPS using a frozen AlexNet feature extractor."""

    def __init__(
        self,
        *,
        backbone: str = "alex",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        try:
            from lpips import LPIPS
        except ImportError as error:
            raise ImportError(
                "LPIPS training requires the optional 'lpips' package: "
                "python -m pip install lpips"
            ) from error

        self.metric = LPIPS(net=backbone, verbose=False)
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)
        self.to(device)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} and "
                f"{target.shape}"
            )
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError(
                "LPIPS inputs must be RGB NCHW tensors, got "
                f"{tuple(prediction.shape)}"
            )
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("LPIPS inputs must use a floating-point dtype")
        distance = self.metric(
            prediction.float() * 2.0 - 1.0,
            target.float() * 2.0 - 1.0,
        )
        return distance.mean()


@dataclass(frozen=True)
class VAELossResult:
    """Total loss and separately observable reconstruction components."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    ssim: torch.Tensor
    lpips: torch.Tensor
    kl: torch.Tensor


class VAE(nn.Module):
    """Encode 32x32 images into a Gaussian latent and decode them back."""

    def __init__(
        self,
        config: VAEConfig | None = None,
        image_size: int = IMG_SIZE,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else VAEConfig()
        self.image_size = image_size

        if image_size <= 0 or image_size % self.config.downsample_factor != 0:
            raise ValueError(
                "image_size must be positive and divisible by "
                f"downsample_factor={self.config.downsample_factor}"
            )
        channels = self.config.hidden_channels
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("hidden_channels must contain four positive integers")
        requested_latent_channels = self.config.latent_channels
        if requested_latent_channels is None:
            if channels[-1] % 2 != 0:
                raise ValueError(
                    "the final encoder channel count must be even because it "
                    "is split into mean and log-variance"
                )
            self.latent_channels = channels[-1] // 2
        else:
            if requested_latent_channels <= 0:
                raise ValueError("latent_channels must be positive when provided")
            self.latent_channels = requested_latent_channels

        downsample_factor = self.config.downsample_factor
        maximum_factor = 2 ** (len(channels) - 1)
        if (
            downsample_factor <= 0
            or downsample_factor & (downsample_factor - 1)
            or downsample_factor > maximum_factor
        ):
            raise ValueError(
                "downsample_factor must be a positive power of two no greater "
                f"than {maximum_factor}, got {downsample_factor}"
            )

        self.latent_size = image_size // self.config.downsample_factor
        downsample_steps = int(math.log2(downsample_factor))
        self.encoder_strides = (
            (1,)
            + (2,) * downsample_steps
            + (1,) * (len(channels) - downsample_steps - 1)
        )

        encoder_layers: list[nn.Module] = []
        current_channels = self.config.in_channels
        for out_channels, stride in zip(channels, self.encoder_strides):
            encoder_layers.append(
                self._block(current_channels, out_channels, stride=stride)
            )
            current_channels = out_channels
        if requested_latent_channels is not None:
            encoder_layers.append(
                nn.Conv2d(
                    current_channels,
                    requested_latent_channels * 2,
                    kernel_size=1,
                )
            )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = [
            nn.Conv2d(
                self.latent_channels,
                channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        ]
        for stage_index in range(len(channels) - 1, 0, -1):
            decoder_layers.append(
                self._block(
                    channels[stage_index],
                    channels[stage_index - 1],
                    stride=1,
                    upsample=self.encoder_strides[stage_index] == 2,
                )
            )
        decoder_layers.append(
            nn.Conv2d(
                channels[0],
                self.config.in_channels,
                kernel_size=3,
                padding=1,
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def _block(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        upsample: bool = False,
    ) -> nn.Module:
        """Build a small residual-free convolutional block."""
        layers: list[nn.Module] = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.extend(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels), num_channels=out_channels
                ),
                nn.SiLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels), num_channels=out_channels
                ),
                nn.SiLU(),
            ]
        )
        return nn.Sequential(*layers)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and log variance for a batch of images."""
        self._validate_images(images)
        parameters = self.encoder(images)
        actual_size = parameters.shape[-2:]
        expected_size = (self.latent_size, self.latent_size)
        if actual_size != expected_size:
            raise RuntimeError(
                "encoder spatial output does not match latent_size: "
                f"got {actual_size}, expected {expected_size}"
            )
        mean, logvar = parameters.chunk(2, dim=1)
        return mean, logvar.clamp(min=-30.0, max=20.0)

    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the Gaussian posterior with the reparameterization trick."""
        if mean.shape != logvar.shape:
            raise ValueError("mean and logvar must have the same shape")
        standard_deviation = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(mean) * standard_deviation

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent batch and return pixel values in ``[0, 1]``."""
        if latents.ndim != 4:
            raise ValueError(f"latents must be rank 4, got {latents.shape}")
        if latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"latents must have {self.latent_channels} channels, "
                f"got {latents.shape[1]}"
            )
        if latents.shape[-2:] != (self.latent_size, self.latent_size):
            raise ValueError(
                f"latents must be {self.latent_size}x{self.latent_size}, "
                f"got {tuple(latents.shape[-2:])}"
            )
        if latents.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"latents must be floating point, got {latents.dtype}")
        return torch.sigmoid(self.decoder(latents))

    def forward(
        self,
        images: torch.Tensor,
        *,
        sample_latent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct images and return reconstruction, mean, logvar, latent."""
        mean, logvar = self.encode(images)
        latent = self.reparameterize(mean, logvar) if sample_latent else mean
        reconstruction = self.decode(latent)
        return reconstruction, mean, logvar, latent

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent tokens as ``[B, S, C]``."""
        mean, _ = self.encode(images)
        return mean.flatten(start_dim=2).transpose(1, 2)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode ``[B, S, C]`` flow-matching tokens into images."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.latent_channels:
            raise ValueError(
                f"tokens must be [B, positions, {self.latent_channels}], "
                f"got {tuple(tokens.shape)}"
            )
        expected_positions = self.latent_size * self.latent_size
        if tokens.shape[1] != expected_positions:
            raise ValueError(
                f"tokens must contain {expected_positions} positions, "
                f"got {tokens.shape[1]}"
            )
        latents = tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        return self.decode(latents)

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be rank 4 NCHW, got {images.shape}")
        expected_shape = (self.config.in_channels, self.image_size, self.image_size)
        if tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                f"images must have shape [B, {expected_shape}], got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise ValueError(f"images must be floating point, got {images.dtype}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "VAE":
        """Restore a VAE and its image-size contract from a training checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        model = cls(
            config=VAEConfig(**checkpoint["vae_config"]),
            image_size=checkpoint["image_size"],
        )
        model.load_state_dict(checkpoint["model"])
        return model


def vae_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float,
    l1_weight: float = 0.0,
    ssim_weight: float = 0.0,
    lpips_weight: float = 0.0,
    lpips_distance: torch.Tensor | None = None,
) -> VAELossResult:
    """Return total and component losses for one reconstruction batch."""
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    if l1_weight < 0 or ssim_weight < 0 or lpips_weight < 0:
        raise ValueError("reconstruction loss weights must be non-negative")
    if lpips_weight > 0 and lpips_distance is None:
        raise ValueError("lpips_distance is required when lpips_weight is positive")
    if lpips_distance is not None and lpips_distance.ndim != 0:
        raise ValueError("lpips_distance must be a scalar tensor")

    mse_loss = F.mse_loss(reconstruction, target)
    reconstruction_loss = mse_loss
    if l1_weight:
        reconstruction_loss = reconstruction_loss + l1_weight * F.l1_loss(
            reconstruction,
            target,
        )

    ssim_loss = reconstruction.new_zeros(())
    if ssim_weight > 0:
        ssim_loss = (
            1.0 - structural_similarity(reconstruction, target)
        )
        reconstruction_loss = reconstruction_loss + ssim_weight * ssim_loss

    lpips_loss = reconstruction.new_zeros(())
    if lpips_weight > 0:
        assert lpips_distance is not None
        lpips_loss = lpips_distance.to(
            device=reconstruction.device,
            dtype=reconstruction.dtype,
        )
        reconstruction_loss = reconstruction_loss + lpips_weight * lpips_loss

    kl_per_batch = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).sum(
        dim=(1, 2, 3)
    )
    kl_loss = kl_per_batch.mean()
    total_loss = reconstruction_loss + kl_weight * kl_loss
    return VAELossResult(
        total=total_loss,
        reconstruction=reconstruction_loss,
        mse=mse_loss,
        ssim=ssim_loss,
        lpips=lpips_loss,
        kl=kl_loss,
    )


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return a differentiable RGB structural-similarity score."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} and "
            f"{target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError(f"images must be rank 4 NCHW, got {prediction.shape}")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Validation targets remain float32 while AMP reconstruction can be
    # float16. SSIM statistics are more stable when both sides use one dtype.
    prediction = prediction.float()
    target = target.float()
    channels = prediction.shape[1]
    coordinates = (
        torch.arange(window_size, device=prediction.device, dtype=torch.float32)
        - window_size // 2
    )
    gaussian = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    kernel = (
        gaussian[:, None] * gaussian[None, :]
    ).expand(channels, 1, window_size, window_size)
    kernel = kernel.to(dtype=prediction.dtype)
    padding = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2

    prediction_mean = F.conv2d(
        prediction,
        kernel,
        padding=padding,
        groups=channels,
    )
    target_mean = F.conv2d(
        target,
        kernel,
        padding=padding,
        groups=channels,
    )
    prediction_variance = torch.clamp(
        F.conv2d(
            prediction.square(),
            kernel,
            padding=padding,
            groups=channels,
        )
        - prediction_mean.square(),
        min=0.0,
    )
    target_variance = torch.clamp(
        F.conv2d(
            target.square(),
            kernel,
            padding=padding,
            groups=channels,
        )
        - target_mean.square(),
        min=0.0,
    )
    covariance = (
        F.conv2d(
            prediction * target,
            kernel,
            padding=padding,
            groups=channels,
        )
        - prediction_mean * target_mean
    )

    numerator = (2.0 * prediction_mean * target_mean + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        prediction_mean.square() + target_mean.square() + c1
    ) * (prediction_variance + target_variance + c2)
    return (numerator / denominator).mean()


def load_vae_from_ckpt(ckpt_path: str = None) -> VAE:
    if not os.path.exists(ckpt_path):
        err = f"{ckpt_path} does not exist"
        raise ValueError(err)
    ckpt = torch.load(ckpt_path, weights_only=False)
    if "vae_config" in ckpt and "image_size" in ckpt:
        vae = VAE(
            config=VAEConfig(**ckpt["vae_config"]),
            image_size=int(ckpt["image_size"]),
        )
    else:
        # Legacy checkpoints did not store their architecture metadata.
        vae = VAE()
    vae.load_state_dict(ckpt["model"])
    return vae

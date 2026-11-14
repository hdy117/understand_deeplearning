from utils import get_logger

logger = get_logger()

import math
import torch
import torch.nn as nn
from config import DiTModelConfig, DEVICE, IMG_SIZE


class Img2Tokens(nn.Module):

    def __init__(self, img_size: int = IMG_SIZE, kernel_size: int = 8):
        super().__init__()
        # image to tokens
        self.img_2_tokens = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=kernel_size,
                padding=0,
            )
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """image to [batch, n_token, kernel_size*kernel_size]

        Args:
            img (torch.Tensor): [batch, channel, h, w]

        Returns:
            torch.Tensor: [batch, n_token, kernel_size*kernel_size]
        """

        # check, [batch, 3, h, w]
        if img.ndim != 4:
            err = f"img dim must be 4, but got {img.ndim}."
            logger.error(err)
            raise ValueError(err)

        img_conv: torch.Tensor = self.img_2_tokens(
            img
        )  # [batch, kernel_size*kernel_size, h/8, w/8]
        img = img_conv.flatten(
            start_dim=2, end_dim=-1
        )  # [batch, kernel_size*kernel_size, n_token]
        img_tokens = img.permute(0, 2, 1)  # [batch, n_token, kernel_size*kernel_size]
        return img_tokens


class TimestepEmbedding(nn.Module):
    """Encode continuous flow-matching time into a model-width vector."""

    def __init__(self, embedding_dim: int, max_period: float = 10_000.0):
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be a positive even number")

        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        # A registered buffer follows model.to(device) and is not a checkpoint parameter.
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time[:, None] * self.frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.mlp(embedding)


class FlowMatching(nn.Module):
    def __init__(
        self,
        model_config: DiTModelConfig | None = None,
        device=DEVICE,
    ):
        super().__init__()
        model_config = model_config if model_config is not None else DiTModelConfig()
        self.d_model: int = model_config.d_model  # model dim
        self.latent_channels: int = model_config.latent_channels
        self.category_num: int = model_config.category_num
        self.n_layer: int = model_config.n_layer  # layer of model
        self.total_steps: int = (
            model_config.total_steps
        )  # total steps for flow matching
        self.n_head: int = model_config.n_head  # head for transformers

        logger.info(
            f"self.d_model:{self.d_model}, self.n_layer:{self.n_layer}, self.total_steps: {self.total_steps}, self.n_head: {self.n_head}"
        )

        # check
        if (
            self.d_model <= 0
            or self.latent_channels <= 0
            or self.category_num <= 0
            or self.n_layer <= 0
            or self.n_head <= 0
            or self.total_steps <= 0
        ):
            raise RuntimeError(
                "d_model, latent_channels, category_num, n_layer, n_head, "
                "and total_steps must all be positive"
            )

        if self.d_model % self.n_head != 0:
            raise RuntimeError(
                f"d_model={self.d_model} must be divisible by n_head={self.n_head}"
            )

        # calculate
        self.one_step: float = 1.0 / self.total_steps
        logger.info(f"self.one_step:{self.one_step}")

        # VAE tokens are narrow [B, S, 4], so project them to Transformer width.
        self.input_proj = nn.Linear(self.latent_channels, self.d_model)
        self.time_embed = TimestepEmbedding(self.d_model)
        self.label_embed = nn.Linear(self.category_num, self.d_model)

        # flow matching with transformer
        self.dit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_head,
                dim_feedforward=4 * self.d_model,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=self.n_layer,
        )

        # The public flow interface stays in VAE latent space.
        self.output_proj = nn.Linear(self.d_model, self.latent_channels)
        self.to(device)

    def forward(
        self,
        xt: torch.Tensor,
        T: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Predict class-conditioned latent velocity at continuous time ``T``.

        Args:
            xt (torch.Tensor): latent tokens [batch, n_token, latent_channels]
            T (torch.Tensor): continuous time [batch] or [batch, 1]
            label_batch (torch.Tensor): one-hot class labels [batch, category_num]

        Returns:
            torch.Tensor: [batch, n_token, latent_channels]
        """
        # check x shape, make sure it has rank 3
        if xt.ndim != 3:
            err: str = f"x should have rank 3, but got {xt.ndim}."
            logger.error(err)
            raise ValueError(err)

        if xt.shape[-1] != self.latent_channels:
            err = (
                f"x last dimension must be {self.latent_channels}, "
                f"but got {xt.shape[-1]}."
            )
            logger.error(err)
            raise ValueError(err)

        # T is continuous time, not a discrete class index.
        if T.ndim != 1 and T.ndim != 2:
            err: str = f"T be dim [batch] or [batch, 1], but got {T.shape}."
            logger.error(err)
            raise ValueError(err)

        if T.ndim == 2 and T.shape[1] != 1:
            err = f"T with rank 2 must have shape [batch, 1], but got {T.shape}."
            logger.error(err)
            raise ValueError(err)

        time = T.reshape(-1)
        if time.shape[0] != xt.shape[0]:
            err = (
                f"T batch size must be {xt.shape[0]}, "
                f"but got {time.shape[0]}."
            )
            logger.error(err)
            raise ValueError(err)
        if time.device != xt.device:
            err = (
                f"xt and T must be on the same device, "
                f"but got {xt.device} and {time.device}."
            )
            logger.error(err)
            raise ValueError(err)
        if not time.is_floating_point():
            err = f"T must be floating point, but got {time.dtype}."
            logger.error(err)
            raise ValueError(err)
        if time.min().item() < 0.0 or time.max().item() > 1.0:
            err = f"T should be within [0, 1], but got {time}."
            logger.error(err)
            raise ValueError(err)

        if label_batch.ndim != 2 or label_batch.shape != (
            xt.shape[0],
            self.category_num,
        ):
            err = (
                f"label_batch must be [{xt.shape[0]}, {self.category_num}], "
                f"but got {tuple(label_batch.shape)}."
            )
            logger.error(err)
            raise ValueError(err)
        if label_batch.device != xt.device:
            err = (
                f"xt and label_batch must be on the same device, "
                f"but got {xt.device} and {label_batch.device}."
            )
            logger.error(err)
            raise ValueError(err)
        if not label_batch.is_floating_point():
            err = f"label_batch must be floating point, but got {label_batch.dtype}."
            logger.error(err)
            raise ValueError(err)
        is_binary = torch.logical_or(label_batch == 0.0, label_batch == 1.0).all()
        is_single_class = torch.eq(label_batch.sum(dim=1), 1.0).all()
        if not is_binary or not is_single_class:
            err = "label_batch rows must each contain exactly one one-hot class."
            logger.error(err)
            raise ValueError(err)

        hidden = self.input_proj(xt)
        condition = self.time_embed(time) + self.label_embed(label_batch)
        hidden = hidden + condition[:, None, :]
        pred_velocity = self.output_proj(self.dit(hidden))

        return pred_velocity

    def get_xt_and_xt_dot(
        self, x0: torch.Tensor, x1: torch.Tensor, T: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """scheduler for flow matching, return x_t and target velocity

        Args:
            x0 (torch.Tensor): noise
            x1 (torch.Tensor): target
            T (torch.Tensor): continuous time [batch] in [0, 1]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: x_t, x_t_dot
        """
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have the same shape, got {x0.shape} and {x1.shape}"
            )
        t = T.reshape(-1, *([1] * (x0.ndim - 1)))
        x_t = (1.0 - t) * x0 + t * x1  # calcualte x_t by t and x0, x1
        x_t_dot = x1 - x0  # calculate x_t dot

        return (x_t, x_t_dot)


if __name__ == "__main__":
    model_cfg = DiTModelConfig()
    device = DEVICE
    flow_matching_net = FlowMatching(model_config=model_cfg, device=device)
    x1 = torch.randn(size=[64, 32, model_cfg.latent_channels], device=device)
    x0 = torch.randn_like(x1, device=x1.device)
    class_ids = torch.arange(x1.shape[0], device=device) % model_cfg.category_num
    label_batch = torch.eye(model_cfg.category_num, device=device)[class_ids]

    # get x_t
    for i in range(model_cfg.total_steps):
        time = torch.full(
            (x1.shape[0],),
            i / model_cfg.total_steps,
            device=device,
        )
        xt, xt_dot = flow_matching_net.get_xt_and_xt_dot(
            x0=x0,
            x1=x1,
            T=time,
        )
        pred_velocity = flow_matching_net.forward(xt, time, label_batch)
        logger.info(f"xt.shape:{xt.shape}, xt_dot.shape:{xt_dot.shape}")

from utils import get_logger

logger = get_logger()

import math
import torch
import torch.nn as nn
from config import FlowMatchingModelConfig, DEVICE, IMG_SIZE


class Img2Tokens(nn.Module):

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        kernel_size: int = 8,
    ):
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
        """image to [batch, n_token, kernel_size*kernel_size], while n_token=(img_size/kernel_size)^2

        Args:
            img (torch.Tensor): [batch, channel, h, w]

        Returns:
            torch.Tensor: [batch, n_token, kernel_size*kernel_size], while n_token=(img_size/kernel_size)^2
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


class SinusoidalPositionEmbedding(nn.Module):
    """Standard fixed sinusoidal encoding for a sequence of token positions."""

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        *,
        max_period: float = 10_000.0,
    ):
        super().__init__()
        if num_positions <= 0:
            raise ValueError("num_positions must be positive")
        if embedding_dim <= 0 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be a positive even number")
        if max_period <= 0:
            raise ValueError("max_period must be positive")

        positions = torch.arange(num_positions, dtype=torch.float32)[:, None]
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(max_period) / embedding_dim)
        )
        embedding = torch.empty(num_positions, embedding_dim)
        embedding[:, 0::2] = torch.sin(positions * div_term)
        embedding[:, 1::2] = torch.cos(positions * div_term)

        # Standard sinusoidal encodings are constants, not model weights.
        self.register_buffer("embedding", embedding[None, :, :], persistent=False)

    def forward(self) -> torch.Tensor:
        return self.embedding


class AdaLNZeroTransformerBlock(nn.Module):
    """Transformer block with per-sample scale, shift, and residual gates.

    The final modulation projection starts at zero.  Every block is therefore
    an identity residual block at initialization, then learns how strongly the
    global time-and-class condition should modulate attention and the MLP.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(
            d_model,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(
            d_model,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )

        # AdaLN-Zero: start every block as an identity operation.
        nn.init.zeros_(self.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.ada_ln_modulation[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one globally conditioned attention/MLP block.

        Args:
            hidden: Token features with shape ``[batch, n_token, d_model]``.
            condition: One global vector per batch item, shape
                ``[batch, d_model]``.
        """
        if hidden.ndim != 3:
            raise ValueError(f"hidden must have rank 3, got {hidden.shape}")
        if condition.shape != (hidden.shape[0], hidden.shape[2]):
            raise ValueError(
                "condition must have shape "
                f"[{hidden.shape[0]}, {hidden.shape[2]}], got {condition.shape}"
            )

        modulations = self.ada_ln_modulation(condition)[:, None, :]
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulations.chunk(6, dim=-1)

        normalized = self.norm1(hidden)
        normalized = normalized * (1.0 + scale_attention) + shift_attention
        attention_output, _ = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = hidden + gate_attention * attention_output

        normalized = self.norm2(hidden)
        normalized = normalized * (1.0 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(normalized)
        return hidden + gate_mlp * mlp_output


class FlowMatching(nn.Module):
    def __init__(
        self,
        model_config: FlowMatchingModelConfig | None = None,
        device=DEVICE,
    ):
        super().__init__()
        model_config = (
            model_config if model_config is not None else FlowMatchingModelConfig()
        )
        self.d_model: int = model_config.d_model  # model dim
        self.latent_channels: int = model_config.latent_channels
        self.num_tokens: int = model_config.num_tokens
        self.category_num: int = model_config.category_num
        self.n_layer: int = model_config.n_layer  # layer of model
        self.total_steps: int = (
            model_config.total_steps
        )  # total steps for flow matching
        self.n_head: int = model_config.n_head  # head for transformers
        self.conditioning: str = model_config.conditioning

        logger.info(
            f"self.d_model:{self.d_model}, self.n_layer:{self.n_layer}, "
            f"self.total_steps:{self.total_steps}, self.n_head:{self.n_head}, "
            f"self.conditioning:{self.conditioning}, "
            f"self.num_tokens:{self.num_tokens}"
        )

        # check
        if (
            self.d_model <= 0
            or self.latent_channels <= 0
            or self.num_tokens <= 0
            or self.category_num <= 0
            or self.n_layer <= 0
            or self.n_head <= 0
            or self.total_steps <= 0
        ):
            raise RuntimeError(
                "d_model, latent_channels, num_tokens, category_num, n_layer, "
                "n_head, and total_steps must all be positive"
            )

        if self.d_model % self.n_head != 0:
            raise RuntimeError(
                f"d_model={self.d_model} must be divisible by n_head={self.n_head}"
            )
        if self.conditioning != "ada_ln_zero":
            raise RuntimeError(
                "this FlowMatching implementation expects "
                "conditioning='ada_ln_zero'"
            )

        # calculate
        self.one_step: float = 1.0 / self.total_steps
        logger.info(f"self.one_step:{self.one_step}")

        # Project the VAE token width to the Transformer hidden dimension.
        self.input_proj = nn.Linear(self.latent_channels, self.d_model)
        # The fixed sinusoidal coordinates make attention aware of token order.
        self.position_embed = SinusoidalPositionEmbedding(
            num_positions=self.num_tokens,
            embedding_dim=self.d_model,
        )
        self.time_embed = TimestepEmbedding(self.d_model)
        self.label_embed = nn.Linear(self.category_num, self.d_model)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Every block receives the same global time/class embedding but learns
        # its own scale, shift, and residual gates through AdaLN-Zero.
        self.flow_matching = nn.ModuleList(
            [
                AdaLNZeroTransformerBlock(
                    d_model=self.d_model,
                    nhead=self.n_head,
                    dim_feedforward=4 * self.d_model,
                )
                for _ in range(self.n_layer)
            ]
        )

        # The public flow interface stays in VAE latent space.
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.latent_channels),
        )
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
            T (torch.Tensor): continuous time [batch, 1]
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

        # T must be a floating-point time column on xt's device.
        if (
            T.ndim != 2
            or T.shape != (xt.shape[0], 1)
            or T.device != xt.device
            or not T.is_floating_point()
            or not torch.all((T >= 0.0) & (T <= 1.0))
        ):
            err = (
                f"T must be floating with shape [{xt.shape[0]}, 1], "
                f"values in [0, 1], and device {xt.device}; got shape "
                f"{tuple(T.shape)}, dtype {T.dtype}, and device {T.device}."
            )
            logger.error(err)
            raise ValueError(err)
        time = T[:, 0]

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

        expected_tokens = self.position_embed.embedding.shape[1]
        if xt.shape[1] != expected_tokens:
            err = (
                f"xt must contain {expected_tokens} latent tokens, "
                f"but got {xt.shape[1]}."
            )
            logger.error(err)
            raise ValueError(err)

        hidden = self.input_proj(xt) + self.position_embed()
        condition = self.condition_mlp(
            torch.cat(
                (self.time_embed(time), self.label_embed(label_batch)),
                dim=-1,
            )
        )
        for block in self.flow_matching:
            hidden = block(hidden, condition)
        pred_velocity = self.output_proj(hidden)

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
    model_cfg = FlowMatchingModelConfig()
    device = DEVICE
    flow_matching_net = FlowMatching(model_config=model_cfg, device=device)
    x1 = torch.randn(
        size=[64, model_cfg.num_tokens, model_cfg.latent_channels],
        device=device,
    )
    x0 = torch.randn_like(x1, device=x1.device)
    class_ids = torch.arange(x1.shape[0], device=device) % model_cfg.category_num
    label_batch = torch.eye(model_cfg.category_num, device=device)[class_ids]

    # get x_t
    for i in range(model_cfg.total_steps):
        time = torch.full(
            (x1.shape[0], 1),
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

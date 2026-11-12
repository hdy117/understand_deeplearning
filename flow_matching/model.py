from turtle import forward

from utils import get_logger

logger = get_logger()

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, DEVICE, IMG_SIZE


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


class FMNet(nn.Module):
    def __init__(self, model_config: ModelConfig = ModelConfig(), device=DEVICE):
        super().__init__()
        self.d_model: int = model_config.d_model  # model dim
        self.n_layer: int = model_config.n_layer  # layer of model
        self.total_steps: int = (
            model_config.total_steps
        )  # total steps for flow matching
        self.n_head: int = model_config.n_head  # head for transformers
        self.category_num: int = model_config.category_num  # total category num
        self.device = device

        logger.info(
            f"self.d_model:{self.d_model}, self.n_layer:{self.n_layer}, self.total_steps: {self.total_steps}, self.n_head: {self.n_head}"
        )

        # check
        if self.d_model <= 0 or self.n_layer <= 0 or self.total_steps <= 0:
            raise RuntimeError(
                f"self.d_model:{self.d_model}, {self.n_layer}, {self.total_steps}"
            )

        # check again
        if self.d_model % self.n_head != 0:
            raise RuntimeError(
                f"self.d_model and self.n_head mismatch, yushu must can be 0"
            )

        # calculate
        self.one_step: float = 1.0 / self.total_steps
        logger.info(f"self.one_step:{self.one_step}")

        # flow mathcing with transformer
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

        # generate step t one hot for embedding
        self.step_t_one_hot: torch.Tensor = torch.eye(
            self.total_steps, device=self.device
        )

        # step T embedding to feed into DiT
        self.step_2_embed = nn.Sequential(
            nn.Linear(in_features=self.total_steps, out_features=self.d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """velocity prediction with repect to x and step T

        Args:
            x (torch.Tensor): [batch, n_token, d_model]
            T (int): step T

        Returns:
            torch.Tensor: [batch, n_token, d_model]
        """
        # check x shape, make sure it has rank 3
        if x.ndim != 3:
            err: str = f"x should have rank 3, but got {x.ndim}."
            logger.error(err)
            raise ValueError(err)

        # check T shape, make sure it is an integer
        if not isinstance(T, int):
            err: str = f"T should be an integer, but got {T}."
            logger.error(err)
            raise ValueError(err)

        # check if I within [0, self.total_steps)
        if T < 0 or T >= self.total_steps:
            err: str = f"T should be within [0, self.total_steps), but got {T}."
            logger.error(err)
            raise ValueError(err)

        # from step t one hot to step t embedding
        t_embed: torch.Tensor = self.step_2_embed(self.step_t_one_hot[T]).view(
            1, 1, self.d_model
        )

        # add step t embedding as condition
        x_hat: torch.Tensor = x + t_embed

        # forward
        pred_velocity: torch.Tensor = self.dit(x_hat)

        return pred_velocity

    def get_xt_and_xt_dot(
        self, x0: torch.Tensor, x1: torch.Tensor, T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """scheduler for flow matching, return x_t and target velocity

        Args:
            x0 (torch.Tensor): noise
            x1 (torch.Tensor): target
            T (int): step T

        Returns:
            tuple[torch.Tensor, torch.Tensor]: x_t, x_t_dot
        """
        t = T * self.one_step  # ratio by step T and one step
        x_t = (1.0 - t) * x0 + t * x1  # calcualte x_t by t and x0, x1
        x_t_dot = x1 - x0  # calculate x_t dot

        return (x_t, x_t_dot)


if __name__ == "__main__":
    model_cfg = ModelConfig()
    flow_matching_net = FMNet(model_config=model_cfg)
    flow_matching_net = flow_matching_net.to(device="cuda")
    x1 = torch.randn(size=[64, 32, model_cfg.d_model])
    x1 = x1.to(device="cuda")
    x0 = torch.randn_like(x1, device=x1.device)

    # get x_t
    for i in range(model_cfg.total_steps):
        xt, xt_dot = flow_matching_net.get_xt_and_xt_dot(
            x0=x0,
            x1=x1,
            T=i,
        )
        pred_velocity = flow_matching_net.forward(xt, i)
        logger.info(f"xt.shape:{xt.shape}, xt_dot.shape:{xt_dot.shape}")

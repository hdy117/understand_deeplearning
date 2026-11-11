from turtle import forward

from utils import get_logger

logger = get_logger()

import torch
import torch.nn as nn


class FMNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass

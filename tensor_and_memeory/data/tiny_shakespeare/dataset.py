"""PyTorch Dataset shared by the RNN, LSTM, and Transformer demos."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharLanguageModelDataset(Dataset):
    """Return overlapping character windows for next-token prediction.

    Each item is a pair ``(x, y)`` with shape ``[seq_len]``. ``y`` is ``x``
    shifted one character to the right, so all three model families can use
    the same causal language-model objective.
    """

    def __init__(self, split: str = "train", seq_len: int = 128, root: Path | None = None):
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")

        dataset_root = root or Path(__file__).resolve().parent
        payload = torch.load(dataset_root / f"{split}.pt", weights_only=True)
        self.tokens = payload["tokens"]
        self.seq_len = seq_len
        if self.tokens.numel() <= seq_len:
            raise ValueError("split is shorter than seq_len + 1")

    def __len__(self) -> int:
        return self.tokens.numel() - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # CrossEntropyLoss expects integer class indices in torch.long.
        window = self.tokens[index : index + self.seq_len + 1].to(torch.long)
        return window[:-1], window[1:]

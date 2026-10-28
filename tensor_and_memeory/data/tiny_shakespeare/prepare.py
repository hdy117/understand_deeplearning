"""Prepare the public Tiny Shakespeare text for sequence-model demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def prepare_dataset(root: Path, train_fraction: float = 0.9, valid_fraction: float = 0.05) -> None:
    """Encode the text by character and save independent token streams."""
    input_path = root / "input.txt"
    text = input_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Dataset is empty: {input_path}")

    if not 0 < train_fraction < 1 or not 0 < valid_fraction < 1:
        raise ValueError("train_fraction and valid_fraction must be between 0 and 1")
    if train_fraction + valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be less than 1")

    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    # The vocabulary has only 65 characters, so uint8 is enough on disk.
    ids = torch.tensor([stoi[char] for char in text], dtype=torch.uint8)

    n_chars = len(ids)
    train_end = int(n_chars * train_fraction)
    valid_end = train_end + int(n_chars * valid_fraction)
    splits = {
        "train": ids[:train_end],
        "valid": ids[train_end:valid_end],
        "test": ids[valid_end:],
    }
    for name, tokens in splits.items():
        torch.save({"tokens": tokens}, root / f"{name}.pt")

    metadata = {
        "source_url": SOURCE_URL,
        "format": "character-level next-token prediction",
        "storage_dtype": "torch.uint8; dataset returns torch.long windows",
        "num_characters": n_chars,
        "vocab_size": len(chars),
        "split_sizes": {name: int(tokens.numel()) for name, tokens in splits.items()},
        "train_fraction": train_fraction,
        "valid_fraction": valid_fraction,
        "test_fraction": 1 - train_fraction - valid_fraction,
        "itos": chars,
        "stoi": stoi,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"saved {n_chars:,} characters, vocab_size={len(chars)}")
    for name, tokens in splits.items():
        print(f"{name:>5}: {tokens.numel():,} tokens -> {name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory containing input.txt and receiving prepared files",
    )
    args = parser.parse_args()
    prepare_dataset(args.root)


if __name__ == "__main__":
    main()

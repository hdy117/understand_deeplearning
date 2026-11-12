#!/usr/bin/env python3
"""Resize every train/test image to 64x64 and overwrite it in place."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from PIL import Image


TARGET_SIZE = (64, 64)
SPLITS = ("train", "test")
IMAGE_SUFFIXES = {".png"}

logger = logging.getLogger("resize_images_to_64")


def collect_images(root: Path) -> tuple[Path, ...]:
    """Collect split images and reject unexpected files before modifying data."""
    image_paths: list[Path] = []
    unsupported: list[Path] = []

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")

        for path in sorted(split_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                unsupported.append(path)
                continue
            image_paths.append(path)

    if unsupported:
        examples = ", ".join(str(path) for path in unsupported[:5])
        raise ValueError(
            f"found {len(unsupported)} unsupported files; "
            f"first files: {examples}"
        )
    if not image_paths:
        raise FileNotFoundError(f"no PNG images found under {root}")

    return tuple(image_paths)


def resize_one(path: Path) -> bool:
    """Return False when already resized; otherwise atomically overwrite the PNG."""
    with Image.open(path) as source:
        source.load()
        if source.size == TARGET_SIZE and source.mode == "RGB":
            return False

        resized = source.convert("RGB").resize(
            TARGET_SIZE,
            resample=Image.Resampling.LANCZOS,
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        resized.save(temporary_path, format="PNG")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return True


def verify_images(paths: Sequence[Path]) -> None:
    """Check every image is now a readable 64x64 RGB PNG."""
    for path in paths:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != TARGET_SIZE:
                raise ValueError(
                    f"{path} has size {image.size}; expected {TARGET_SIZE}"
                )
            if image.mode != "RGB":
                raise ValueError(
                    f"{path} has mode {image.mode}; expected RGB"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resize all PNG images under train/ and test/ to 64x64 in place."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="data directory containing train/ and test/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="number of parallel image workers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be greater than zero")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root = args.root.expanduser().resolve()
    try:
        paths = collect_images(root)
        logger.info("found %d images under %s", len(paths), root)
        logger.info("target size: %dx%d", *TARGET_SIZE)

        resized_count = 0
        skipped_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(resize_one, path): path for path in paths}
            for completed, future in enumerate(as_completed(futures), start=1):
                path = futures[future]
                if future.result():
                    resized_count += 1
                else:
                    skipped_count += 1
                if completed % 5_000 == 0 or completed == len(paths):
                    logger.info("processed %d/%d images", completed, len(paths))

        verify_images(paths)
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error("%s", error)
        return 1

    logger.info(
        "complete: resized=%d, already 64x64=%d, verified=%d",
        resized_count,
        skipped_count,
        len(paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

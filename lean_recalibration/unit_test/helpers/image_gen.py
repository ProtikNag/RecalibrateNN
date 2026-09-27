"""
Deterministic synthetic image generator used by the test suite.

Generates small RGB images with PIL so the test suite never depends on any external
dataset or network access. Each "concept"/class/random folder gets a distinct color
palette + shape signature (seeded by folder name) so that:

  * images are visually distinguishable (useful if a human inspects test/data/),
  * generation is fully deterministic (same inputs -> byte-identical images across
    Windows/Linux runs, since PIL drawing is pure Python/C without OS-specific AA),
  * it is trivially fast (5 tiny images per folder, per the user's requirement).

All paths are handled with ``os.path`` / ``pathlib`` only -- no hard-coded separators --
so folder creation and file naming are OS-agnostic.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

DEFAULT_IMAGE_SIZE = (64, 64)  # small + fast; scripts under test resize as needed anyway
DEFAULT_IMAGES_PER_FOLDER = 5


def _seed_from_name(name: str) -> int:
    """Stable, OS/platform-independent integer seed derived from a string."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _color_for(seed: int, offset: int) -> tuple:
    r = (seed * 37 + offset * 97) % 200 + 30
    g = (seed * 59 + offset * 131) % 200 + 30
    b = (seed * 83 + offset * 173) % 200 + 30
    return (r % 256, g % 256, b % 256)


def generate_images_in_folder(
    folder: str | Path,
    count: int = DEFAULT_IMAGES_PER_FOLDER,
    size=DEFAULT_IMAGE_SIZE,
    prefix: str = "img",
    seed_name: str | None = None,
) -> list:
    """Create `count` deterministic .jpg images inside `folder` (created if missing).

    Returns the list of absolute file paths written. Skips generation for any file that
    already exists so repeated test runs are idempotent (re-running the generator does not
    recreate/rewrite images unless the folder is cleared first).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seed_name = seed_name if seed_name is not None else str(folder)
    seed = _seed_from_name(seed_name)

    written = []
    for i in range(count):
        file_path = folder / f"{prefix}_{i:02d}.jpg"
        if not file_path.exists():
            img = Image.new("RGB", size, color=_color_for(seed, i))
            draw = ImageDraw.Draw(img)
            # Draw a couple of simple deterministic shapes so images differ slightly and
            # are not just flat single-color blocks (helps the CAV/SVM training step see
            # some non-trivial per-image structure).
            w, h = size
            draw.rectangle(
                [w * 0.15, h * 0.15, w * 0.85, h * 0.85],
                outline=_color_for(seed, i + 1),
                width=2,
            )
            draw.ellipse(
                [w * 0.3, h * 0.3, w * 0.7, h * 0.7],
                fill=_color_for(seed, i + 2),
            )
            img.save(file_path, format="JPEG", quality=90)
        written.append(str(file_path.resolve()))
    return written


def generate_dataset_tree(
    root: str | Path,
    class_names: Iterable[str],
    images_per_folder: int = DEFAULT_IMAGES_PER_FOLDER,
    size=DEFAULT_IMAGE_SIZE,
    splits: Iterable[str] = ("train", "valid"),
) -> dict:
    """Create `<root>/<split>/<class>/` folders (default: train + valid) with images.

    Returns {split: {class_name: [file_paths]}}.
    """
    root = Path(root)
    result: dict = {}
    for split in splits:
        result[split] = {}
        for class_name in class_names:
            folder = root / split / class_name
            result[split][class_name] = generate_images_in_folder(
                folder,
                count=images_per_folder,
                size=size,
                prefix=class_name,
                seed_name=f"{split}/{class_name}",
            )
    return result


def generate_concept_folder(
    root: str | Path,
    concept_name: str,
    images_per_folder: int = DEFAULT_IMAGES_PER_FOLDER,
    size=DEFAULT_IMAGE_SIZE,
) -> list:
    """Create `<root>/<concept_name>/` with `images_per_folder` synthetic images."""
    folder = Path(root) / concept_name
    return generate_images_in_folder(
        folder, count=images_per_folder, size=size, prefix=concept_name, seed_name=f"concept/{concept_name}"
    )


def generate_random_folder(
    root: str | Path,
    folder_name: str = "random",
    images_per_folder: int = DEFAULT_IMAGES_PER_FOLDER,
    size=DEFAULT_IMAGE_SIZE,
) -> list:
    """Create `<root>/<folder_name>/` with `images_per_folder` synthetic 'random/negative' images."""
    folder = Path(root) / folder_name
    return generate_images_in_folder(
        folder, count=images_per_folder, size=size, prefix="random", seed_name=f"random/{folder_name}"
    )

from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


# Change this to the directory containing files like 1_coat.jpg and 2_face.png.
IMAGE_FOLDER = Path(".")
CATEGORIES = ("coat", "face", "background", "legs", "random")
IMAGES_PER_CATEGORY = 2
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGE_SIZE = (224, 224)


def main():
    files = [
        path for path in IMAGE_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    random.shuffle(files)

    chosen = {}
    for category in CATEGORIES:
        matches = [
            path for path in files
            if category in path.stem.lower().split("_")
        ]
        if len(matches) < IMAGES_PER_CATEGORY:
            raise ValueError(
                f"Need {IMAGES_PER_CATEGORY} images for '{category}', found {len(matches)}."
            )
        chosen[category] = matches[:IMAGES_PER_CATEGORY]

    for category in CATEGORIES:
        fig, axes = plt.subplots(1, IMAGES_PER_CATEGORY, figsize=(6, 3))
        for col, path in enumerate(chosen[category]):
            with Image.open(path) as image:
                resized = image.convert("RGB").resize(
                    IMAGE_SIZE, Image.Resampling.LANCZOS
                )
                axes[col].imshow(resized)
            #axes[col].set_title(path.name)
            axes[col].axis("off")

        fig.tight_layout()
        output_file = IMAGE_FOLDER / f"{category}.png"
        fig.savefig(output_file, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Mosaic saved to {output_file}")


if __name__ == "__main__":
    main()
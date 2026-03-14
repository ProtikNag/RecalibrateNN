#!/usr/bin/env python3
"""
Animal Detector + Segmenter: deer, horse, zebra
=================================================
Iterates over every image in a directory and, for the top detection:
  1. Writes a bounding-box .txt file (bbox/)
  2. Writes the animal cut out from its background (segmented/)

Both output files share the same stem as the source image.

Output layout
-------------
  <output_dir>/
    bbox/           – {image_stem}.txt   (animal, score, x1/y1/x2/y2, w/h)
    segmented/      – {image_stem}.png   (animal on white or transparent bg)

Usage
-----
  # In case you want to run the code with the images presemt in current directory
  python segment_animals.py path/to/images/
  
  python segment_animals.py /home/balanced_dataset/train/deer/ --output_dir /home/segmented_data/output_segmented_deer --threshold 0.70
  python segment_animals.py path/to/images/ --no_transparent
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    SamModel,
    SamProcessor,
)

# ---------------------------------------------------------------------------
# Target animals + text queries fed to OWLv2
# ---------------------------------------------------------------------------
ANIMAL_QUERIES: dict[str, list[str]] = {
    "deer":  ["a deer", "a wild deer", "a fawn"],
    "horse": ["a horse", "a brown horse", "a white horse", "a black horse"],
    "zebra": ["a zebra", "a zebra with black and white stripes"],
}

_FLAT_QUERIES: list[str] = [q for qs in ANIMAL_QUERIES.values() for q in qs]
_QUERY_TO_ANIMAL: dict[str, str] = {
    q: animal for animal, qs in ANIMAL_QUERIES.items() for q in qs
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: str):
    print("Loading OWLv2 (zero-shot detector) …")
    owl_proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = (
        Owlv2ForObjectDetection
        .from_pretrained("google/owlv2-base-patch16-ensemble")
        .to(device)
        .eval()
    )

    print("Loading SAM (Segment Anything Model) …")
    sam_proc  = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = (
        SamModel
        .from_pretrained("facebook/sam-vit-base")
        .to(device)
        .eval()
    )

    return owl_proc, owl_model, sam_proc, sam_model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_animals(
    image: Image.Image,
    owl_proc: Owlv2Processor,
    owl_model: Owlv2ForObjectDetection,
    device: str,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return (boxes, scores, labels) sorted by descending score."""
    inputs = owl_proc(
        text=[_FLAT_QUERIES],
        images=image,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = owl_proc.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    return results["boxes"], results["scores"], results["labels"]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_box(
    image: Image.Image,
    box: torch.Tensor,
    sam_proc: SamProcessor,
    sam_model: SamModel,
    device: str,
) -> np.ndarray:
    """Run SAM on a single bounding box; return a boolean mask (H, W)."""
    inputs = sam_proc(
        images=image,
        input_boxes=[[box.tolist()]],   # shape: [1 image][1 box]
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_proc.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # masks[0] → (1 box, 3 candidates, H, W); index 0 = highest IoU
    return masks[0][0][0].numpy().astype(bool)


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------

def apply_mask(
    image: Image.Image,
    mask: np.ndarray,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    transparent_bg: bool = False,
) -> Image.Image:
    """Cut the animal out of *image*; background → solid colour or transparent."""
    mask = mask.astype(bool)

    if transparent_bg:
        rgba = np.array(image.convert("RGBA"))
        rgba[..., 3] = np.where(mask, 255, 0)
        return Image.fromarray(rgba, mode="RGBA")

    rgb = np.array(image.convert("RGB"))
    out = np.full_like(rgb, bg_color, dtype=np.uint8)
    out[mask] = rgb[mask]
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    bbox_dir: Path,
    seg_dir: Path,
    owl_proc: Owlv2Processor,
    owl_model: Owlv2ForObjectDetection,
    sam_proc: SamProcessor,
    sam_model: SamModel,
    device: str,
    threshold: float,
    transparent_bg: bool,
    bg_color: tuple[int, int, int],
) -> bool:
    """Detect + segment the top animal; save bbox txt and masked image."""
    print(f"\n{'─' * 60}")
    print(f"Image : {image_path.name}")

    image = Image.open(image_path).convert("RGB")

    # ── Detection ────────────────────────────────────────────────────────────
    boxes, scores, labels = detect_animals(
        image, owl_proc, owl_model, device, threshold
    )

    if len(boxes) == 0:
        print(f"  No target animals detected (threshold={threshold}). Skipping.")
        return False

    top_score  = scores[0].item()
    top_animal = _QUERY_TO_ANIMAL.get(labels[0], "animal")
    top_box    = boxes[0]
    x1, y1, x2, y2 = top_box.tolist()

    print(f"  Top: {top_animal!r}  score={top_score:.3f}  "
          f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # ── Bounding-box txt ─────────────────────────────────────────────────────
    original_area = image.width * image.height
    box_area = (x2 - x1) * (y2 - y1)
    box_ratio = box_area / original_area
    
    bbox_path = bbox_dir / f"{image_path.stem}.txt"
    bbox_path.write_text(
        f"animal: {top_animal}\n"
        f"score:  {top_score:.4f}\n"
        f"x1:     {x1:.2f}\n"
        f"y1:     {y1:.2f}\n"
        f"x2:     {x2:.2f}\n"
        f"y2:     {y2:.2f}\n"
        f"width:  {x2 - x1:.2f}\n"
        f"height: {y2 - y1:.2f}\n"
        f"original_area: {original_area:.2f}\n"
        f"box_area: {box_area:.2f}\n"
        f"box_ratio: {box_ratio:.4f}\n"
    )
    print(f"  Saved (bbox)     : {bbox_path}")

    # ── Segmentation ─────────────────────────────────────────────────────────
    mask = segment_box(image, top_box, sam_proc, sam_model, device)
    masked_img = apply_mask(image, mask, bg_color=bg_color, transparent_bg=transparent_bg)

    seg_path = seg_dir / f"{image_path.stem}.png"
    masked_img.save(seg_path)
    print(f"  Saved (segmented): {seg_path}")

    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_dir: str,
    output_dir: str = "output",
    threshold: float = 0.10,
    transparent_bg: bool = False,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_dir   = Path(input_dir)
    bbox_dir = Path(output_dir) / "bbox"
    seg_dir  = Path(output_dir) / "segmented"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Device        : {device}")
    print(f"Input dir     : {in_dir}  ({len(image_files)} image(s))")
    print(f"BBox output   : {bbox_dir}")
    print(f"Seg output    : {seg_dir}")

    if not image_files:
        print(f"No images found. Supported extensions: {IMAGE_EXTENSIONS}")
        return

    for p in image_files:
        print(f"  {p.name}")

    owl_proc, owl_model, sam_proc, sam_model = load_models(device)

    ok = skipped = 0
    for image_path in image_files:
        try:
            if process_image(
                image_path, bbox_dir, seg_dir,
                owl_proc, owl_model, sam_proc, sam_model,
                device, threshold, transparent_bg, bg_color,
            ):
                ok += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  ERROR – {image_path.name}: {exc}")
            skipped += 1

    print(f"\n{'═' * 60}")
    print(f"Done.  Written: {ok}   Skipped/failed: {skipped}")
    print(f"  bbox/      → {bbox_dir}")
    print(f"  segmented/ → {seg_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Detect deer, horse, and zebra in every image inside a directory. "
            "Saves a bounding-box .txt (bbox/) and a masked image (segmented/) "
            "for each file, both named after the source image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing input images.")
    parser.add_argument(
        "--output_dir", default="output",
        help="Root output directory (bbox/ and segmented/ created inside it).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Minimum detection confidence (0–1).",
    )
    parser.add_argument(
        "--no_transparent", action="store_false",
        help="Use transparent background in segmented images (RGBA PNG).",
    )
    parser.add_argument(
        "--bg_color", nargs=3, type=int, default=[255, 255, 255],
        metavar=("R", "G", "B"),
        help="Background fill colour when --transparent is not set.",
    )
    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        transparent_bg=args.no_transparent,
        bg_color=tuple(args.bg_color),
    )

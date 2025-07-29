from PIL import Image
import sys
import os

def image_to_pdf(image_path, output_pdf=None):
    if not output_pdf:
        output_pdf = os.path.splitext(image_path)[0] + ".pdf"
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(output_pdf, "PDF", resolution=100.0)
    print(f"Saved PDF to {output_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python svg_image.py <image_path> [output_pdf]")
    else:
        image_path = sys.argv[1]
        output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
        image_to_pdf(image_path, output_pdf)

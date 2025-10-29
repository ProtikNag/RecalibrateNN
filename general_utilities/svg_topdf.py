import cairosvg
import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python svg_image.py <image_path> [output_pdf]")
    else:
        image_path = sys.argv[1]
        output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
        cairosvg.svg2pdf(url=image_path, write_to=output_pdf)

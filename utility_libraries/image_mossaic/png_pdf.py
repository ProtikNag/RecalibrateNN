"""Convert PNG images to PDF without recompressing the image data.

Install the dependency with: pip install img2pdf
"""

import argparse
from pathlib import Path

import img2pdf


def png_to_pdf(inputs: list[Path], output: Path) -> None:
	for image in inputs:
		if image.suffix.lower() != ".png" or not image.is_file():
			raise ValueError(f"Not a PNG file: {image}")

	output.write_bytes(img2pdf.convert([str(image) for image in inputs]))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert one or more PNG files to a lossless PDF."
	)
	parser.add_argument("images", nargs="+", type=Path, help="Input PNG file(s)")
	parser.add_argument("-o", "--output", required=True, type=Path, help="Output PDF")
	args = parser.parse_args()

	if args.output.suffix.lower() != ".pdf":
		parser.error("output filename must end in .pdf")

	try:
		png_to_pdf(args.images, args.output)
	except (ValueError, OSError, img2pdf.AlphaChannelError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()

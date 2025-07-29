
import cairosvg
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

image1 = '/home/srikanth/study1/RecalibrateNN/xai_images/gradcam/vgg16/before/0/gradcam_13.svg'
image2 = '/home/srikanth/study1/RecalibrateNN/xai_images/gradcam/vgg16/before/0/gradcam_14.svg'

def svg_to_png(svg_path, output_png_path, width, height):
    # Convert SVG to PNG with specified width and height
    cairosvg.svg2png(url=svg_path, write_to=output_png_path, output_width=width, output_height=height)

def create_pdf_from_images(image_paths, output_pdf_path, page_width, page_height):
    c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))
    
    current_x = 0
    for img_path in image_paths:
        c.drawImage(img_path, current_x, 0, width=page_width // 2, height=page_height)
        current_x += page_width // 2
    
    c.save()

def main(svg1_path, svg2_path, output_pdf_path, target_width, target_height):
    png1_path = "temp1.png"
    png2_path = "temp2.png"
    
    # Convert and resize SVGs to PNGs
    svg_to_png(svg1_path, png1_path, target_width // 2, target_height)
    svg_to_png(svg2_path, png2_path, target_width // 2, target_height)
    
    # Create PDF concatenating the two images side by side
    create_pdf_from_images([png1_path, png2_path], output_pdf_path, target_width, target_height)

    print(f"PDF saved to {output_pdf_path}")

if __name__ == "__main__":
    # Example usage: resize both images to 800x600 total (each image 400x600)
    svg_file1 = image1
    svg_file2 = image2
    output_pdf = "output.pdf"
    total_width = 800
    total_height = 600
    
    main(svg_file1, svg_file2, output_pdf, total_width, total_height)

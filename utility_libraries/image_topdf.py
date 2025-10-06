import os
from PIL import Image
import io
import base64

def convert_images_to_svg(input_dir):

    svg_dir = os.path.join(input_dir, "svg")
    os.makedirs(svg_dir, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            img_path = os.path.join(input_dir, filename)
            try:
                with Image.open(img_path) as img:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    width, height = img.size
                    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <image href="data:image/png;base64,{img_str}" width="{width}" height="{height}"/>
</svg>'''
                    svg_filename = os.path.splitext(filename)[0] + ".svg"
                    svg_path = os.path.join(svg_dir, svg_filename)
                    with open(svg_path, "w", encoding="utf-8") as f:
                        f.write(svg_content)
                    print(f"Converted {filename} -> svg/{svg_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def convert_images_to_pdf(input_dir):
    pdf_dir = os.path.join(input_dir, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            img_path = os.path.join(input_dir, filename)
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
                    pdf_path = os.path.join(pdf_dir, pdf_filename)
                    img.save(pdf_path, "PDF")
                    print(f"Converted {filename} -> pdf/{pdf_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")




if __name__ == "__main__":
    input_dir = input("Enter the directory containing images: ").strip()
    if os.path.isdir(input_dir):
        convert_images_to_pdf(input_dir)
        convert_images_to_svg(input_dir)
    else:
        print("Invalid directory.")




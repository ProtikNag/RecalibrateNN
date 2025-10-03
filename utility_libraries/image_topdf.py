import os
from PIL import Image

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
    else:
        print("Invalid directory.")

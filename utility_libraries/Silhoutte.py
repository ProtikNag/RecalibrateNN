from rembg import remove
from PIL import Image, ImageOps


def create_silhouette(input_path: str, output_path: str) -> None:
    """
    Remove background from an image and create a silhouette.
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the silhouette image
    """
    # 1. Remove the background
    input_image = Image.open(input_path).convert("RGBA")
    subject_only = remove(input_image)

    # 2. Extract Alpha Channel (mask) and invert
    alpha = subject_only.split()[-1]
    # Turn white to black, black to white (silhouette)
    silhouette = ImageOps.invert(alpha.convert('L'))

    # 3. Save as silhouette
    silhouette.save(output_path)
    print(f"Silhouette saved as {output_path}")


# Example usage
if __name__ == "__main__":
    create_silhouette('gradcam_8.png', 'zebra_silhouette.png')

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

def create_morphed_image(segmentation_path, bbox_path, background_folder, output_folder, reduce=False, ratio_size = 0.9):
    """
    Overlay a segmented image onto a random background image within the bounding box.
    
    Args:
        segmentation_path: Path to the zebra segmentation image
        bbox_path: Path to the bounding box coordinates file
        background_folder: Path to the background images folder
        output_folder: Path to save the morphed image
    """
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load the segmentation image
    segmentation_img = Image.open(segmentation_path).convert('RGBA')
    if segmentation_img is None:
        print(f"Error loading segmentation image: {segmentation_path}")
        return
    
    original_width, original_height = segmentation_img.size
    
    # Read bounding box coordinates
    x, y, w, h = parse_bbox_file(bbox_path)
    bbox_area = w*h
    original_area = original_width * original_height
    ratio_size = bbox_area/original_area
    # Get random background image
    background_images = os.listdir(background_folder)
    if not background_images:
        print(f"No background images found in {background_folder}")
        return
    
    random_bg = random.choice(background_images)
    background_path = os.path.join(background_folder, random_bg)

    background_img = Image.open(background_path).convert('RGBA')

    
    if background_img is None:
        print(f"Error loading background image: {background_path}")
        return
    
    # Resize background to match original zebra image size
    background_img = background_img.resize((original_width, original_height), Image.Resampling.LANCZOS)
    # Crop the zebra image using bounding box coordinates
    x, y, w, h = int(x), int(y), int(w), int(h)
    zebra_cropped = segmentation_img.crop((x, y, x + w, y + h))
    if(reduce == False):
        zebra_cropped = zebra_cropped.resize((w, h), Image.Resampling.LANCZOS)
    else:
        zebra_cropped = zebra_cropped.resize((int(w*ratio_size), int(h*ratio_size)), Image.Resampling.LANCZOS)
    # Create a copy of background for pasting
    result_img = background_img.copy()
    # Paste the cropped zebra into the background at bbox location with alpha compositing
    result_img.alpha_composite(zebra_cropped, (x, y))
    morphed_image = result_img
    # Save the morphed image
    output_filename = os.path.basename(segmentation_path)
    output_path = os.path.join(output_folder, output_filename)
    morphed_image.convert('RGB').save(output_path)
    print(f"Morphed image saved to: {output_path}")
    


# Set paths
def process_all_zebras(segmentation_base_folder, bbox_base_folder, background_folder, output_folder):
    """
    Process all zebra segmentation images from subdirectories.

    Args:
        segmentation_base_folder: Path to the base segmentation folder
        bbox_base_folder: Path to the base bounding box folder
        background_folder: Path to the background images folder
        output_folder: Path to save morphed images
    """
    folder_path = segmentation_base_folder
    for image_file in os.listdir(folder_path):
        print(f"Processing {image_file} in {folder_path}...")
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            segmentation_path = os.path.join(folder_path, image_file)
            bbox_filename = os.path.splitext(image_file)[0] + '.txt'
            bbox_path = os.path.join(bbox_base_folder, bbox_filename)
            
            if os.path.exists(bbox_path):
                create_morphed_image(segmentation_path, bbox_path, background_folder, output_folder, reduce = False, ratio_size = 0.9)
            else:
                print(f"Bbox file not found for {image_file}")
        

def parse_bbox_file(bbox_path):
    """
    Parse bounding box file and extract x, y, width, height.
    
    Args:
        bbox_path: Path to the bounding box file
        
    Returns:
        Tuple of (x, y, w, h)
    """
    with open(bbox_path, 'r') as f:
        lines = f.readlines()
    
    bbox_dict = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            if(key.strip() in ['x1', 'y1', 'width', 'height']):
                bbox_dict[key.strip()] = float(value.strip())
    
    x = bbox_dict['x1']
    y = bbox_dict['y1']
    w = bbox_dict['width']
    h = bbox_dict['height']
    
    return x, y, w, h

# Execute

#animal_base_folder = input("Enter the path to the zebra segmentation folder: ")
#animal_segmentation = os.path.join(animal_base_folder, "segmented")
#animal_bbox = os.path.join(animal_base_folder, "bbox")
#background_folder = input("Enter the path to the background folder: ")
#output_folder = input("Enter the path to the output folder for morphed images: ")
#process_all_zebras(animal_segmentation, animal_bbox, background_folder, output_folder)


parser = argparse.ArgumentParser(description="Create morphed zebra images by overlaying segmentations onto backgrounds.")
parser.add_argument("--segmentation", required=True, help="Path to the zebra segmentation folder")
parser.add_argument("--background", required=True, help="Path to the background folder")
parser.add_argument("--output", required=True, help="Path to the output folder for morphed images")

args = parser.parse_args()

animal_segmentation = os.path.join(args.segmentation, "segmented")
animal_bbox = os.path.join(args.segmentation, "bbox")
process_all_zebras(animal_segmentation, animal_bbox, args.background, args.output)



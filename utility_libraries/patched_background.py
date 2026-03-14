from PIL import Image
import numpy as np
import os
import argparse
import random

def superimpose_with_transparency(image_path, concept_image_path, bbox, scale = 0.3):
    """
    Superimpose 30% of bounding box with a concept image and make background transparent.
    
    Args:
        image_path: Path to main image
        concept_image_path: Path to concept image to superimpose
        bbox: Bounding box as (x, y, width, height)
        output_path: Path to save result
    """
    # Open images
    main_img = Image.open(image_path).convert("RGBA")
    concept_img = Image.open(concept_image_path).convert("RGBA")
    
    # Calculate 10% of bounding box
    x, y, w, h = bbox
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize concept image to 10% of bbox size
    concept_img_resized = concept_img.resize((new_w, new_h))
    
    
    # Calculate position (center of bbox)
    x_pos = int(x + (w - new_w) / 2)
    y_pos = int(y + (h - new_h) / 2)
    
    # Create result by pasting concept image onto main image
    result = main_img.copy()
    
    # Paste concept image at bbox position
    result.paste(concept_img_resized, (x_pos, y_pos), concept_img_resized)
    return result, concept_img

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

"""
python patched_background.py --original-image-dir /home/balanced_dataset_subset/train/zebra/ --bb-dir /home/output_zebra_train/bbox/
--concept-image-dir /mnt/sdc/concepts/deer/coat/ --output-dir /home/patch_inplace/train/zebra_with_deer_coat/
"""

# Usage
if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Superimpose concept images on main images with bounding boxes.')
    parser.add_argument('--original-image-dir', type=str, default="C:\\Users\\srikant1\\Downloads\\Prediction\\seg\\train",
                        help='Path to original image directory')
    parser.add_argument('--bb-dir', type=str, default="C:\\Users\\srikant1\\Downloads\\Prediction\\bb",
                        help='Path to bounding box directory')

    parser.add_argument('--concept-image-dir', type=str, default="C:\\Users\\srikant1\\Downloads\\Prediction\\concept",
                        help='Path to concept image directory')
    parser.add_argument('--output-dir', type=str, default="C:\\Users\\srikant1\\Downloads\\Prediction\\output",
                        help='Path to output directory')
    args = parser.parse_args()
    original_image_directory = args.original_image_dir
    concept_image_directory = args.concept_image_dir
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)
    bb_directory = args.bb_dir
    
    bb_items = []
    for root, _, files in os.walk(bb_directory):
        for file_name in files:
            if file_name.lower().endswith(".txt"):
                bb_items.append(file_name)
    concept_images = []
    for root, _, files in os.walk(concept_image_directory):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                concept_images.append(os.path.join(root, file_name))
    count = 0
    for bb_file in sorted(bb_items):
        #extract file name from the file 
        # Try to find the corresponding image file for the bbox file
        base_name = bb_file.replace('.txt', '')
        found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            file_name = base_name + ext
            image_path = os.path.join(original_image_directory, file_name)
            if os.path.isfile(image_path):
                found = True
                break
        if not found:
            print(f"No image found for bbox file {bb_file} with image path {image_path}")
            continue
        x, y, w, h = parse_bbox_file(os.path.join(bb_directory, bb_file))
        image_path = os.path.join(original_image_directory, file_name)  
        concept_image_path = random.choice(concept_images)
        output_path = os.path.join(output_directory, file_name)
        #print(f"Processing {image_path}")
        #print(f"with bbox ({x}, {y}, {w}, {h})")
        #print(f"and concept image {concept_image_path}")
        try:
            result_image, concept_img = superimpose_with_transparency(image_path, concept_image_path, (x, y, w, h), scale = 0.3)
            print(f"Saved patched image to {output_path}")
            # Convert RGBA to RGB before saving as JPEG
            if result_image.mode == 'RGBA':
                result_image = result_image.convert('RGB')
            result_image.save(output_path)
            count += 1
        except Exception as e:
            continue

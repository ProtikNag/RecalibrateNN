import os
from PIL import Image
# Get directory paths from user
before_dir = input("Enter the 'before' directory path: ")
after_dir = input("Enter the 'after' directory path: ")
merged_dir = input("Enter the 'merged' directory path: ")
# Verify directories exist
if not os.path.exists(before_dir) or not os.path.exists(after_dir):
    raise Exception("One or both directories do not exist!")
# Create merged directory if it doesn't exist
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)
# Get list of files from both directories
before_files = sorted([f for f in os.listdir(before_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
after_files = sorted([f for f in os.listdir(after_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

# Process each pair of images
for before_file, after_file in zip(before_files, after_files):
    # Open images
    before_img = Image.open(os.path.join(before_dir, before_file))
    after_img = Image.open(os.path.join(after_dir, after_file))
    
    # Ensure both images are the same size
    width = max(before_img.width, after_img.width)
    height = max(before_img.height, after_img.height)
    
    # Resize if necessary
    before_img = before_img.resize((width, height))
    after_img = after_img.resize((width, height))
    
    # Create new image by placing before and after side by side
    merged_img = Image.new('RGB', (width * 2, height))
    merged_img.paste(before_img, (0, 0))
    merged_img.paste(after_img, (width, 0))
    
    # Save merged image
    output_filename = f'merged_{os.path.splitext(before_file)[0]}.jpg'
    merged_img.save(os.path.join(merged_dir, output_filename))

print("Merging complete!")

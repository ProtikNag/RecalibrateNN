import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
xai_images = xai_images
xai_method = 'integrated_gradient'  # <-- Make XAI method a variable
model_name = 'vgg16'
class_idx = 0
before_dir = fr'xai_images/{xai_method}/{model_name}/before/{class_idx}/{xai_method}_0.png'
after_dir = fr'xai_images/{xai_method}/{model_name}/after/{class_idx}/{xai_method}_0.png'
 
#image_files = sorted([f for f in os.listdir(before_dir) if f.endswith('.png')])
image_files =  before_dir
img_size = (224, 224)
 
fig, axes = plt.subplots(len(image_files), 1, figsize=(8.27, 11.69))
 
fig.suptitle(f'{xai_method} - {model_name} - Class {class_idx}', fontsize=16, y=0.865)
fig.text(0.25, 0.835, 'Before', ha='center', va='center', fontsize=12, fontweight='bold')
fig.text(0.75, 0.835, 'After', ha='center', va='center', fontsize=12, fontweight='bold')
 
for idx, img_name in enumerate(image_files):
    before_img = Image.open(os.path.join(before_dir, img_name)).resize(img_size)
    after_img = Image.open(os.path.join(after_dir, img_name)).resize(img_size)
 
    axes[idx, 0].imshow(np.array(before_img))
    axes[idx, 0].axis('off')
 
    axes[idx, 1].imshow(np.array(after_img))
    axes[idx, 1].axis('off')
 
plt.tight_layout(rect=[0, 0.07, 1, 0.89])
plt.savefig(f'{xai_method}_{model_name}_class{class_idx}_comparison_table.png', format='png', dpi=300, bbox_inches='tight')

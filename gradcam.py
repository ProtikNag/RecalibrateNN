import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
from model import DeepCNN

# Configuration
CLASS_NAMES = ["deer", "horse", "zebra"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 224
MODEL_PATH = "./model_weights/imbalanced_model.pth"
TARGET_LAYER = "conv_block4.0"  # Use a conv layer, not a linear one
CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification_1"

# Load model
model = DeepCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Hook containers
activations = {}
gradients = {}

def save_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

# Register hooks
target_layer_module = dict(model.named_modules())[TARGET_LAYER]
target_layer_module.register_forward_hook(save_activation(TARGET_LAYER))
target_layer_module.register_full_backward_hook(save_gradient(TARGET_LAYER))  # Fixed deprecated hook

# Load one image from each class
def load_images():
    images, labels, raw_images = [], [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name, "train")
        img_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not img_files:
            raise FileNotFoundError(f"No valid image found in {folder}")
        img_path = os.path.join(folder, img_files[7])
        image = Image.open(img_path).convert('RGB')
        raw_images.append(image.copy())
        images.append(transform(image))
        labels.append(idx)
    return torch.stack(images), labels, raw_images

# Grad-CAM computation
def compute_gradcam(input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, class_idx]
    loss.backward()

    act = activations[TARGET_LAYER]        # [B, C, H, W]
    grad = gradients[TARGET_LAYER]         # [B, C, H, W]

    pooled_grad = torch.mean(grad[0], dim=[1, 2])  # [C]
    for i in range(act.shape[1]):
        act[0, i, :, :] *= pooled_grad[i]

    heatmap = act[0].mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# Overlay heatmap on image
def show_gradcam_on_image(img: Image.Image, heatmap: np.ndarray, alpha=0.5):
    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img_np, 1 - alpha, 0)
    return superimposed_img

# Main execution
def main():
    inputs, labels, raw_images = load_images()
    inputs = inputs.to(DEVICE)

    for i in range(len(labels)):
        input_img = inputs[i].unsqueeze(0)
        class_idx = labels[i]
        heatmap = compute_gradcam(input_img, class_idx)
        result_img = show_gradcam_on_image(raw_images[i], heatmap)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_images[i])
        plt.title(f"Original - {CLASS_NAMES[class_idx]}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(result_img[..., ::-1])  # Convert BGR to RGB
        plt.title(f"Grad-CAM - {CLASS_NAMES[class_idx]}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

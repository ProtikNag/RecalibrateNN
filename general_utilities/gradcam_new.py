import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from torchvision import transforms
from copy import deepcopy

from utils import get_model_layers
from custom_dataloader import SingleClassDataLoader

# Configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 224
TARGET_LAYER = "conv_block4.0"
CLASS_NAMES = ["deer", "horse", "zebra"]
TARGET_IDX_LIST = [0, 1, 2]

# Load transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Activation and gradient hooks
activation = {}
gradient = {}


def save_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradient[name] = grad_output[0]

    return hook


def register_hooks(model, layer_name):
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(save_activation(layer_name))
    layer.register_full_backward_hook(save_gradient(layer_name))


def compute_gradcam(input_tensor, model, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, class_idx]
    loss.backward()

    act = activation[TARGET_LAYER]
    grad = gradient[TARGET_LAYER]
    pooled_grad = torch.mean(grad[0], dim=[1, 2])
    weighted_act = (pooled_grad.view(-1, 1, 1) * act[0]).sum(dim=0)

    heatmap = weighted_act.cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap


def show_gradcam_on_image(img: Image.Image, heatmap: np.ndarray, alpha=0.5):
    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, alpha, img_np, 1 - alpha, 0)
    return superimposed_img


def select_top_delta_tcav_examples(tcav_before, tcav_after, image_list, k=5):
    delta_scores = np.array(tcav_after) - np.array(tcav_before)
    top_indices = np.argsort(-delta_scores)[:k]
    return [image_list[i] for i in top_indices], [tcav_before[i] for i in top_indices], [tcav_after[i] for i in top_indices]


def visualize_gradcams(model_before, model_after, selected_images, selected_labels):
    fig, axs = plt.subplots(len(selected_images), 3, figsize=(12, 4 * len(selected_images)))

    for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        for model, col, tag in zip([model_before, model_after], [1, 2], ['Before', 'After']):
            register_hooks(model, TARGET_LAYER)
            heatmap = compute_gradcam(img_tensor, model, label)
            result_img = show_gradcam_on_image(img, heatmap)
            axs[i, col].imshow(result_img[..., ::-1])
            axs[i, col].axis("off")
            axs[i, col].set_title(f"Grad-CAM ({tag}) - {CLASS_NAMES[label]}")

        axs[i, 0].imshow(img)
        axs[i, 0].axis("off")
        axs[i, 0].set_title(f"Original - {CLASS_NAMES[label]}")

    plt.tight_layout()
    plt.savefig("./results/gradcam_comparison.pdf")
    plt.show()


# === Main Flow ===
def run_gradcam_tcav_visualization(model_before, model_after, concept_loader_list, random_loader, class_dataloaders):
    from main import compute_cav, compute_tcav_score  # <- Adjust import

    selected_images = []
    selected_labels = []
    tcav_before_all = []
    tcav_after_all = []

    for i, (concept_loader, class_loader) in enumerate(zip(concept_loader_list, class_dataloaders)):
        cav = compute_cav(model_before, concept_loader, random_loader, TARGET_LAYER)
        score_before = compute_tcav_score(model_before, TARGET_LAYER, cav, class_loader, TARGET_IDX_LIST[i])
        score_after = compute_tcav_score(model_after, TARGET_LAYER, cav, class_loader, TARGET_IDX_LIST[i])

        dataset = class_loader.dataset
        selected_images.append(dataset[0][0])  # PIL image from dataset
        selected_labels.append(TARGET_IDX_LIST[i])
        tcav_before_all.append(score_before)
        tcav_after_all.append(score_after)

    top_imgs, scores_b, scores_a = select_top_delta_tcav_examples(tcav_before_all, tcav_after_all, selected_images)
    visualize_gradcams(model_before, model_after, top_imgs, selected_labels)


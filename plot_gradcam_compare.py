import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from config import DEVICE, CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST
from custom_dataloader import SingleClassDataLoader
from gradcam_utils import GradCAM


def get_target_layer(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in model.")


def replace_relu_with_out_of_place(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_with_out_of_place(module)


def denormalize(tensor):
    denorm = transforms.Normalize(mean=[-1]*3, std=[2]*3)
    return denorm(tensor).clamp(0, 1)


def load_images(target_class):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    val_dir = os.path.join(CLASSIFICATION_DATA_BASE_PATH, target_class, "valid")
    dataset = SingleClassDataLoader(val_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    return loader


def compute_sensitivity(model, loader, class_idx):
    model.eval()
    image_tensors = []
    sensitivity_scores = []

    for imgs in loader:
        imgs = imgs.to(DEVICE)
        imgs.requires_grad_()
        outputs = model(imgs)
        grads = torch.autograd.grad(outputs[:, class_idx].sum(), imgs, retain_graph=True)[0]
        sens = grads.abs().mean(dim=[1, 2, 3])
        sensitivity_scores.extend(sens.cpu().tolist())
        image_tensors.extend(imgs.cpu())
    return image_tensors, sensitivity_scores


def plot_gradcam_comparison(model_before, model_after, gradcam_before, gradcam_after,
                            image_tensors, sensitivity_scores, class_idx, target_class):
    top_indices = sorted(range(len(sensitivity_scores)), key=lambda i: -sensitivity_scores[i])[:5]
    top_images = [image_tensors[i] for i in top_indices]
    top_scores = [sensitivity_scores[i] for i in top_indices]

    for idx, (img_tensor, score) in enumerate(zip(top_images, top_scores)):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        cam_before = gradcam_before.generate(img_tensor, class_idx=torch.tensor([class_idx], device=DEVICE))
        cam_after = gradcam_after.generate(img_tensor, class_idx=torch.tensor([class_idx], device=DEVICE))

        raw_img = denormalize(img_tensor.squeeze().detach().cpu()).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(raw_img)
        axes[1].imshow(cam_before, cmap='jet', alpha=0.5)
        axes[1].set_title("Grad-CAM (Before)")
        axes[1].axis('off')

        axes[2].imshow(raw_img)
        axes[2].imshow(cam_after, cmap='jet', alpha=0.5)
        axes[2].set_title("Grad-CAM (After)")
        axes[2].axis('off')

        plt.suptitle(f"Top-{idx+1} Sensitive {target_class.capitalize()} Image")
        plt.tight_layout()
        plt.show()


def main(model_name: str, model_before_path: str, model_after_path: str,
         target_class: str, target_layer_name: str):

    # === Load models ===
    model_before = torch.load(model_before_path, map_location=DEVICE)
    model_after = torch.load(model_after_path, map_location=DEVICE)

    replace_relu_with_out_of_place(model_before)
    replace_relu_with_out_of_place(model_after)

    model_before.to(DEVICE).eval()
    model_after.to(DEVICE).eval()

    # === Target class index ===
    if target_class not in TARGET_CLASS_LIST:
        raise ValueError(f"Class '{target_class}' not found in TARGET_CLASS_LIST.")
    class_idx = TARGET_CLASS_LIST.index(target_class)

    # === Get target layer ===
    layer_before = get_target_layer(model_before, target_layer_name)
    layer_after = get_target_layer(model_after, target_layer_name)
    gradcam_before = GradCAM(model_before, layer_before)
    gradcam_after = GradCAM(model_after, layer_after)

    # === Load images ===
    loader = load_images(target_class)

    # === Compute sensitivity ===
    image_tensors, sensitivity_scores = compute_sensitivity(model_before, loader, class_idx)

    # === Plot comparison ===
    plot_gradcam_comparison(model_before, model_after,
                            gradcam_before, gradcam_after,
                            image_tensors, sensitivity_scores,
                            class_idx, target_class)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Model name (e.g., resnet50 or vgg)")
    parser.add_argument('--model_before_path', type=str, required=True, help="Path to model before retraining")
    parser.add_argument('--model_after_path', type=str, required=True, help="Path to model after retraining")
    parser.add_argument('--target_class', type=str, required=True, help="Target class name (e.g., zebra)")
    parser.add_argument('--target_layer', type=str, required=True, help="Target layer name (e.g., features.28)")

    args = parser.parse_args()
    main(args.model_name, args.model_before_path, args.model_after_path,
         args.target_class, args.target_layer)
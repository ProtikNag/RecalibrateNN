import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from utils import get_base_model_image_size

"""
  https://captum.ai/tutorials/Image_and_Text_Classification_LIME
  https://captum.ai/tutorials/TorchVision_Interpret
  https://captum.ai/api/
  https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
  https://github.com/jacobgil/pytorch-grad-cam
"""

def find_last_conv_layer_pytorch(model):
    """
    Finds and returns the last nn.Conv2d layer in a given PyTorch model.
    Iterates through all named modules of the provided model and keeps track of the last encountered nn.Conv2d layer.
    Prints the name and the module of the last Conv2d layer found.
    Args:
        model (torch.nn.Module): The PyTorch model to search for Conv2d layers.
    Returns:
        tuple or None: A tuple (name, module) of the last nn.Conv2d layer found, or None if no Conv2d layer exists in the model.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    if last_conv:
        print(f"Last Conv2d layer: {last_conv[0]} - {last_conv[1]}")
        return last_conv
    else:
        print("No Conv2d layer found.")
        return None

def get_image_array(MODEL_NAME, image_list):
    """
    Converts a list of image file paths into a tensor suitable for model input.
    Args:
        MODEL_NAME (str): Name of the model, used to determine the required input image size.
        image_list (list of str): List of file paths to the images to be processed.
    Returns:
        torch.Tensor: A tensor containing the transformed images, moved to the appropriate device (CPU or CUDA).
    Notes:
        - Images are resized to the input size required by the specified model.
        - Images are normalized using ImageNet mean and standard deviation.
        - The function automatically selects CUDA if available, otherwise uses CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = get_base_model_image_size(MODEL_NAME)
    VALID_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    image_array = []
    for fname in image_list:
        img =  Image.open(fname).convert('RGB') 
        transformed_img = VALID_TRANSFORM(img)
        image_array.append(transformed_img)
    image_array = np.array(image_array)
    image_array = torch.tensor(image_array).to(device)
    return (image_array)



def predictdata(image_tensors , model):
    """
    Predicts class labels and probabilities for a batch of image tensors using a given model.
    Args:
        image_tensors (torch.Tensor): A batch of image tensors with shape (batch_size, ...).
        model (torch.nn.Module): The trained PyTorch model to use for prediction.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - all_preds: Tensor of predicted class indices for each image in the batch (shape: [batch_size]).
            - all_probs: Tensor of predicted class probabilities for each image in the batch (shape: [batch_size, num_classes]).
    Notes:
        - The function automatically uses GPU if available, otherwise falls back to CPU.
        - The model is set to evaluation mode and gradients are not computed during prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_probs = []
    # Stack the list of tensors into a single batch tensor
    model.eval()  # Make sure model is in eval mode
    with torch.no_grad():
        # If image_array is already a batch, process all at once:
        outputs = model(image_tensors)  # shape: (batch_size, num_classes)
        probs = nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)  # get predicted class indices for batch
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    all_preds = torch.tensor(np.array(all_preds), dtype=torch.long, device=device)
    all_probs = torch.tensor(np.array(all_probs), dtype=torch.float, device=device)
    return all_preds ,all_probs

 
def xai_integrated_gradients(model_name, model, num_classes,images,n_steps=200, save_dir = './' ):
    """
    xai_integrated_gradients(model_name, model, num_classes, images, n_steps=200, save_dir='./')
    Generates and visualizes Integrated Gradients attributions for a set of images and a given model.
    Args:
        model_name (str): Name or identifier of the model architecture.
        model (torch.nn.Module): The PyTorch model to be explained.
        num_classes (int): Number of classes to process (assumes one image per class in `images`).
        images (list): List of image data, one per class, to be explained.
        n_steps (int, optional): Number of steps for the Integrated Gradients approximation. Default is 200.
        save_dir (str, optional): Directory to save the resulting attribution visualizations. Default is './'.
    Returns:
        None
    Notes:
        - The function assumes that `get_image_array`, `predictdata`, `IntegratedGradients`, `viz`, `np`, `torch`, `os`, and `LinearSegmentedColormap` are available in the scope.
        - The function saves heatmap overlays of attributions for each image in the specified directory.
        - The function currently assumes a hardcoded mapping of class indices to class names (e.g., 0 = deer, 1 = horse, 2 = zebra).
    """
    #hard coded the classes for now 0 = deer , 1 = Horse, 2 = Zebra
    
    for class_idx in range(num_classes):
        save_dir_new = save_dir + f'/{class_idx}'
        input_tensors = get_image_array(model_name, images[class_idx])
        all_preds_tensors, all_probs_tensors = predictdata(input_tensors,model)
        print(all_preds_tensors, all_probs_tensors)
        # Baseline: black image (all zeros)
        baseline = torch.zeros_like(input_tensors)
        integrated_gradients = IntegratedGradients(model)
        # `.attribute` supports batches: returns attributions per image :contentReference[oaicite:2]{index=2}
        attributions = integrated_gradients.attribute(input_tensors,
                        baselines=baseline,
                        target=all_preds_tensors,
                        n_steps=200,
                        internal_batch_size=10
                       )   # shape [10, 3, 224, 224]
                       
        default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)               
        for i in range(len(attributions)):
            #Convert the CHW to HWC 
            attr = attributions[i].cpu().detach().numpy().transpose(1, 2, 0)
            # Plot heatmap overlay
            vis_result = viz.visualize_image_attr(
                attr,
                np.array(input_tensors[i].cpu()),
                method='heat_map',
                cmap=default_cmap,
                sign='positive',
                show_colorbar=True,
                outlier_perc=1,
                title=f"Integrated Gradients - Class {all_preds_tensors[i].item()}"
            )
            fig, _ = vis_result
            if save_dir_new:
                filename = f"integrated_gradients_sample_{i}_class_{all_preds_tensors[i]}.png"
                filepath = os.path.join(save_dir_new,filename)
                fig.savefig(filepath)
    return 

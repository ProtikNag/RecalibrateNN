# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on```python
# Copyright [2025] [Srikanth KS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
/*
 * Copyright (c) 2025 Srikanth K S. All rights reserved.
 * Licensed under the APACHE2 License.
 * Author : Srikanth K S
 * Version 1.0
 */
"""
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from utils import get_base_model_image_size
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

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
        try:
          img =  Image.open(fname).convert('RGB') 
          transformed_img = VALID_TRANSFORM(img)
          image_array.append(transformed_img)
        except Exception as e:
          pass
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

 
def xai_integrated_gradients(model_name, model, num_classes,images,n_steps=200, save_dir = './' ,title_prefix = ""):
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
    for i in range(num_classes):
        os.makedirs(save_dir +f'/{i}', exist_ok=True)
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
                        internal_batch_size=10)   # shape [10, 3, 224, 224]
        default_cmap = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'),(0.25, '#000000'), 
                                                                         (1, '#000000')], N=256)               
        for i in range(len(attributions)):
            #Convert the CHW to HWC 
            attr = attributions[i].cpu().detach().numpy().transpose(1, 2, 0)
            orig_img = np.array(input_tensors[i].cpu()).transpose(1, 2, 0)
            # Plot heatmap overlay
            # Plot blended heatmap overlay
            vis_result = viz.visualize_image_attr(
                attr,
                orig_img,
                method='heat_map', # <-- 'blended_heat_map',  # <-- changed here
                cmap=default_cmap,
                sign='positive',
                show_colorbar=True,
                outlier_perc=1,
                use_pyplot  = False,
                #title=f"Integrated Gradients - Class {all_preds_tensors[i].item()} - {title_prefix}"
            )
            fig, _ = vis_result
            if save_dir_new:
                filename = f"integrated_gradients_{i}_class_{all_preds_tensors[i]}.png"
                filepath = os.path.join(save_dir_new,filename)
                fig.savefig(filepath, format='png')
                
                filename = f"integrated_gradients_{i}_class_{all_preds_tensors[i]}.pdf"
                filepath = os.path.join(save_dir_new,filename)
                fig.savefig(filepath, format='pdf')
    return 

 


def show_cam_on_image(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Convert PIL image to OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + img_cv
    superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img)

"""_summary_
GradCAM class for generating Class Activation Maps (CAM) using the Grad-CAM technique.
    Returns:
        _type_: _description_
"""
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)
        print(f"Registered hooks for layer: {self.target_layer}")

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            # Get the predicted class if no target class is specified
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass to compute gradients of the target class with respect to the target layer's output
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activations
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # ReLU on the weighted activations
        heatmap = torch.sum(self.activations, dim=1).squeeze()
        heatmap = nn.functional.relu(heatmap)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()
    
def xai_gradcam_explainer(MODEL_NAME, model, images, num_classes,save_dir, title_prefix =""):
    """Function to explain the model predictions using GradCAM.  """
    target_layer = find_last_conv_layer_pytorch(model)[1]  # Get the last convolutional layer
    show_fig = False
    grad_cam = GradCAM(model, target_layer)

    for i in range(num_classes):
        print(save_dir +f'/{i}')
        os.makedirs(save_dir +f'/{i}', exist_ok=True)
    for i in range(num_classes):
        image_array = get_image_array(MODEL_NAME, images[i])
        all_preds ,all_probs = predictdata(image_array , model)
        for img_idx in range(len(image_array)):
            try:
                original_image = Image.open(images[i][img_idx])
                image = image_array[img_idx].unsqueeze(0)
                heatmap = grad_cam(image, target_class= i)
                image_np = image.squeeze(0).detach().cpu().numpy()
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) 
                cam_image = show_cam_on_image(original_image, heatmap)
                if(show_fig == True):
                    plt.figure()
                    plt.imshow(cam_image)
                #cam_image.save(os.path.join(save_dir, str(i), f'gradcam_{img_idx}.png'))
                fig, ax = plt.subplots()
                ax.imshow(cam_image)
                ax.axis('off')
                #ax.set_title(f"GradCAM - Class{i} - {title_prefix}", fontsize=12)
                fig.savefig(os.path.join(save_dir, str(i), f'gradcam_{img_idx}.png'), format='png', bbox_inches='tight')
                fig.savefig(os.path.join(save_dir, str(i), f'gradcam_{img_idx}.pdf'), format='pdf', bbox_inches='tight')
                plt.close(fig)
                print(os.path.join(save_dir, str(i), f'gradcam_{img_idx}.png'))
            except Exception as e:
                print(f"Error processing image {images[i][img_idx]}: {e}")
                continue
    print(f"GradCAM results saved in {save_dir}")
    return
    
def permute_callback(images_np, model):
    # Convert to torch tensor and permute to (N, C, H, W)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()
    device = next(model.parameters()).device
    images_tensor = images_tensor.to(device)
    with torch.no_grad():
        outputs = model(images_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs
def lime_explainer(model, image_tensor,save_fig_path,  org_image_array=None):
    show_fig = False
    explainer = LimeImageExplainer()    
    for i in range(len(image_tensor)):
        explanation = explainer.explain_instance(
            image_tensor[i].cpu().numpy().transpose(1, 2, 0), 
            lambda x: permute_callback(x, model), 
            top_labels=1, 
            hide_color=0, 
            num_samples=1000,
        )
        temp = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=20, hide_rest=True)
        image, mask = temp
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        #Load the original image if provided
        if org_image_array is not None:
            org_image = org_image_array[i]
            org_image = Image.open(org_image)
            org_image = np.array(org_image)
        axes[0].imshow(org_image if org_image is not None else image_tensor[i].cpu().numpy().transpose(1, 2, 0))
        #axes[0].set_title('Original image')
        axes[0].axis('off')
        axes[1].imshow(mark_boundaries(image.astype(np.uint8), mask))
        #axes[1].set_title('LIME Explanation')
        axes[1].axis('off')
        # Save the figure
        plt.tight_layout()

        fig_path = os.path.join(save_fig_path, f'lime_explanation_{i}.png')
        plt.savefig(fig_path, bbox_inches='tight',format='png')

        fig_path = os.path.join(save_fig_path, f'lime_explanation_{i}.pdf')
        plt.savefig(fig_path, bbox_inches='tight',format='pdf')
        plt.close(fig)

        # Optionally show the figure
        if show_fig:
            plt.figure(fig.number)
            plt.show()

def xai_lime_explainer(MODEL_NAME, model, images, num_classes, save_dir ):
    """
    Function to explain the model predictions using LIME.
    """
    for i in range(num_classes):
        os.makedirs(save_dir +f'/{i}', exist_ok=True)
    
    for i in range(num_classes):
        image_array = get_image_array(MODEL_NAME, images[i])
        all_preds ,all_probs = predictdata(image_array , model)
        save_dir_dest = os.path.join(save_dir, str(i))
        print(f"{save_dir_dest}.")
        lime_explainer(model, image_array, save_dir_dest, images[i])

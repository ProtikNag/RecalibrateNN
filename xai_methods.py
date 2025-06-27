import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torchvision
from torchvision import models
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import get_base_model_image_size

"""
  https://captum.ai/tutorials/Image_and_Text_Classification_LIME
  https://captum.ai/tutorials/TorchVision_Interpret
  https://captum.ai/api/
  https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
  https://github.com/jacobgil/pytorch-grad-cam
"""

def find_last_conv_layer_pytorch(model):
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
    #hard coded the classes for now 0 = deer , 1 = Horse, 2 = Zebra
    for class_idx in range(num_classes):
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
        
            if save_dir:
                filename = f"integrated_gradients_sample_{i}_class_{all_preds_tensors[i]}.png"
                filepath = os.path.join(save_dir, filename)
                fig.savefig(filepath)
    return 




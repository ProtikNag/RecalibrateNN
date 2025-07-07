import torch
"""
This script performs explainable AI (XAI) analysis using Integrated Gradients on a multiclass image classification model (e.g., VGG16) before and after modification. It loads two versions of a PyTorch model, identifies the last convolutional layer, and applies Integrated Gradients to a set of sample images from three classes (deer, horse, zebra). The results are saved to specified directories.
Main functionalities:
- Loads original and modified PyTorch models from specified paths.
- Identifies the last convolutional layer in each model.
- Applies Integrated Gradients XAI method to visualize model explanations for each class.
- Saves the generated explanations to disk for both model versions.
Arguments:
    --org_model_path (str): Path to the original (unmodified) model checkpoint.
    --modified_model_path (str): Path to the modified model checkpoint.
    --model_name (str): Name of the model architecture (e.g., 'vgg16').
    --save_dir (str): Directory to save the XAI results.
Global Variables:
    IMAGES_CLASS_0, IMAGES_CLASS_1, IMAGES_CLASS_2 (list): Lists of image file paths for each class.
    IMAGES (list): List containing all class image lists.
    activation (dict): Stores activations from registered hooks.
    output_shape (dict): Stores output shapes from registered hooks.
Functions:
    get_activation(layer_name): Returns a hook function to capture activations and output shapes for a given layer.
Usage:
    Run the script with the required arguments to generate and save Integrated Gradients explanations for both the original and modified models.
"""
from PIL import Image
import argparse
import pandas as pd
from torchvision import transforms
from utils import get_base_model_image_size
import os
from xai_methods import (xai_integrated_gradients, find_last_conv_layer_pytorch, xai_gradcam_explainer, xai_lime_explainer)
from dataset import get_image_dataset , num_classes

XAI_Integrated_gradients = True
XAI_GradCAM              = True
XAI_Lime                 = True

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None

activation = {}
output_shape = {}
df = pd.DataFrame()

IMAGES = get_image_dataset('vgg16')

def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
    return hook

def get_model(model_path, modified_model_path=None):
    model = torch.load(model_path)
    if(modified_model_path is not None):
        model.load_state_dict(torch.load(modified_model_path, weights_only=True))
    model.eval()
    return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obtain the original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")
    args = parser.parse_args()
    args = parser.parse_args([
        "--org_model_path", "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth",
        "--modified_model_path", "/mnt/data/results/vgg16/loss_vgg16_features.12_0.5.pth",
        "--model_name", "vgg16",
        "--save_dir", "./xai_images/integrated_gradient"
    ])
    BASE_MODEL_PATH = args.org_model_path.strip()
    MODIFIED_MODEL_PATH = args.modified_model_path.strip()
    MODEL_NAME = args.model_name.strip().lower()
    IMAGES = get_image_dataset(MODEL_NAME)

    ################################################################################################################
    ############# Integrated Gradients ##########################################################
    if(XAI_Integrated_gradients == True):
        save_dir = args.save_dir.strip()
        model = get_model(BASE_MODEL_PATH)
        save_dir_before = os.path.join(save_dir, MODEL_NAME+'/before')
        xai_integrated_gradients(MODEL_NAME, model, num_classes, IMAGES, n_steps=200, save_dir = save_dir_before)
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        save_dir_after = os.path.join(save_dir, MODEL_NAME+'/after')
        xai_integrated_gradients(MODEL_NAME, model_modified, num_classes, IMAGES, n_steps=200, save_dir = save_dir_after)
    ################################################################################################################
    ############# GRAD CAM Implementation ##########################################################
    if(XAI_GradCAM == True):
        model = get_model(BASE_MODEL_PATH)
        save_dir = os.path.join('./xai_images/gradcam', MODEL_NAME,  'before')
        #def xai_gradcam_explainer(MODEL_NAME, model, images, num_classes,save_dir):
        xai_gradcam_explainer(MODEL_NAME, model,IMAGES, num_classes, save_dir)
        save_dir = os.path.join('./xai_images/gradcam', MODEL_NAME,  'after')
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        xai_gradcam_explainer(MODEL_NAME, model_modified,IMAGES, num_classes, save_dir)
    ################################################################################################################
    ############# Lime Implementation ##########################################################
    if(XAI_Lime == True):
        model = get_model(BASE_MODEL_PATH)
        save_dir = os.path.join('./xai_images/lime', MODEL_NAME,  'before')
        xai_lime_explainer(MODEL_NAME, model, IMAGES, num_classes, save_dir)
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        save_dir = os.path.join('./xai_images/lime', MODEL_NAME,  'after')
        xai_lime_explainer(MODEL_NAME, model_modified, IMAGES, num_classes, save_dir)


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
from xai_methods import (xai_integrated_gradients, find_last_conv_layer_pytorch)

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None

activation = {}
output_shape = {}
df = pd.DataFrame()

IMAGES_CLASS_0 = [
    '/home/multiclass_classification/deer/train/b23f5bb88b.jpg',
    '/home/multiclass_classification/deer/train/38cdafa0ff.jpg',
    '/home/multiclass_classification/deer/train/3a1776bf9d.jpg',
    '/home/multiclass_classification/deer/train/image_35.jpg',
    '/home/multiclass_classification/deer/train/image_21.jpg',
    '/home/multiclass_classification/deer/train/image_79.jpg',
    '/home/multiclass_classification/deer/train/4171515070.jpg',
    '/home/multiclass_classification/deer/train/image_65.jpg',
    '/home/multiclass_classification/deer/train/image_32.jpg',
    '/home/multiclass_classification/deer/train/4d88a12299.jpg'
]

IMAGES_CLASS_1 = [
    '/home/multiclass_classification/horse/train/05_003.png',
    '/home/multiclass_classification/horse/train/07_060.png',
    '/home/multiclass_classification/horse/train/03_038.png',
    '/home/multiclass_classification/horse/train/horse01-7.png',
    '/home/multiclass_classification/horse/train/horse01-3.png',
    '/home/multiclass_classification/horse/train/horse03-0.png',
    '/home/multiclass_classification/horse/train/07_034.png',
    '/home/multiclass_classification/horse/train/horse02-7.png',
    '/home/multiclass_classification/horse/train/07_070.png',
    '/home/multiclass_classification/horse/train/horse31-4.png'
]

IMAGES_CLASS_2 = [
    '/home/multiclass_classification/zebra/train/n02391049_541.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_10158.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_2177.jpg',
    '/home/multiclass_classification/zebra/train/n02391049_7434.jpg',
    '/home/multiclass_classification/zebra/train/008.jpg',
    '/home/multiclass_classification/zebra/train/image_8.jpeg',
    '/home/multiclass_classification/zebra/train/n02391049_1743.jpg',
    '/home/multiclass_classification/zebra/train/image_116.jpeg',
    '/home/multiclass_classification/zebra/train/n02391049_9136.jpg',
    '/home/multiclass_classification/zebra/train/image_45.jpeg'
]

IMAGES = [IMAGES_CLASS_0, IMAGES_CLASS_1, IMAGES_CLASS_2]

def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
    return hook

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
        "--save_dir", "./integrated_gradient"
    ])
    BASE_MODEL_PATH = args.org_model_path.strip()
    MODIFIED_MODEL_PATH = args.modified_model_path.strip()
    MODEL_NAME = args.model_name.strip().lower()
    save_dir = args.save_dir.strip()
    model = torch.load(BASE_MODEL_PATH)
    return_value = find_last_conv_layer_pytorch(model)
    last_layer = return_value[0]
    print(last_layer)
    model.eval()
    num_classes = 3
    save_dir_before = os.path.join(save_dir, MODEL_NAME+'/before')
    for i in range(num_classes):
      os.makedirs(save_dir_before +f'/{i}', exist_ok=True)
    xai_integrated_gradients(MODEL_NAME, model, num_classes, IMAGES, n_steps=200, save_dir = save_dir_before)
    model_modified = torch.load(BASE_MODEL_PATH)
    model_modified.load_state_dict(torch.load(MODIFIED_MODEL_PATH, weights_only=True))
    return_value = find_last_conv_layer_pytorch(model_modified)
    last_layer = return_value[0]
    model_modified.eval()
    save_dir_after = os.path.join(save_dir, MODEL_NAME+'/after')
    for i in range(num_classes):
      os.makedirs(save_dir_after +f'/{i}', exist_ok=True)
    xai_integrated_gradients(MODEL_NAME, model_modified, num_classes, IMAGES, n_steps=200, save_dir = save_dir_after)

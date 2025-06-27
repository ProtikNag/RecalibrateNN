from torch import torch
from PIL import Image
import argparse
import os
import pandas as pd
import numpy as np
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from utils import get_base_model_image_size
from xai_methods import (xai_integrated_gradients,find_last_conv_layer_pytorch)
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
        # This print has been added for you to visualize if the size is too large then the time taken fror convergence will be large
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
        #logging.info(f"Verify the output shape : Layername = {layer_name} ,Input.shape : {input[0].shape},  output.shape : {output.shape}")
    return hook



if(__name__ == '__main__'):
    ############## Parser #################################
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Obtain the original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    args = parser.parse_args()
    args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth", 
    "--modified_model_path" , "/mnt/data/results/vgg16/loss_vgg16_features.12_0.5.pth", "--model_name", "vgg16"])
    BASE_MODEL_PATH = args.org_model_path.strip()
    MODIFIED_MODEL_PATH = args.modified_model_path.strip()
    MODEL_NAME = args.model_name.strip().lower()   
    model = torch.load(BASE_MODEL_PATH)
    return_value = find_last_conv_layer_pytorch(model)
    last_layer = return_value[0] 
    print(last_layer)
    model.eval()
    num_classes = 3
    xai_integrated_gradients(MODEL_NAME, model, num_classes, IMAGES, n_steps=200, save_dir = './integrated_gradient/before' )
    model_modified = torch.load(BASE_MODEL_PATH)
    model_modified.load_state_dict(torch.load(MODIFIED_MODEL_PATH, weights_only=True))
    return_value = find_last_conv_layer_pytorch(model_modified)
    last_layer = return_value[0]
    model_modified.eval()
    xai_integrated_gradients(MODEL_NAME, model_modified, num_classes,IMAGES, n_steps=200, save_dir = './integrated_gradient/after' )

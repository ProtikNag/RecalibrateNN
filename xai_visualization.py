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
"""
This script performs explainable AI (XAI) analysis using Integrated Gradients on a multiclass image classification 
model (e.g., VGG16) before and after modification. It loads two versions of a PyTorch model, identifies the last convolutional layer, 
and applies Integrated Gradients to a set of sample images from three classes (deer, horse, zebra). The results are saved to specified directories.
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from dotenv import load_dotenv
from ConfigSingleton import ConfigSingleton
from xai_methods import (xai_integrated_gradients, find_last_conv_layer_pytorch, xai_gradcam_explainer, xai_lime_explainer)

XAI_Integrated_gradients = True
XAI_GradCAM              = True
XAI_Lime                 = True

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
RECALIBRATED_MODEL = None

activation = {}
output_shape = {}
df = pd.DataFrame()

#MODEL_NAME = 'vgg16'
#RECALIBRATED_MODEL = 'loss_vgg16_features.12_0.5.pth'
#MODEL_NAME = 'inception_v3'
#RECALIBRATED_MODEL = 'loss_inception_v3_Mixed_6b.branch7x7dbl_4.conv_0.6.pth'
#MODEL_NAME  = "resnet50"
#RECALIBRATED_MODEL = 'loss_resnet50_layer3.5.conv1_0.6.pth'
#MODEL_NAME  = "mobilenet_v3_large"
#RECALIBRATED_MODEL = 'loss_mobilenet_v3_large_features.5_0.6.pth'

######################################


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
        #print(model)
    model.eval()
    return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obtain the original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--modified_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--config_file", type=str, default=None, help="Configuration file")
    parser.add_argument("--save_dir", type=str, default=None, help="Specify a save directory to save the results")
    args = parser.parse_args()
    """
    args = parser.parse_args([
        "--org_model_path", f"/home/srikanth/trained_models/pytorch/{MODEL_NAME}/{MODEL_NAME}.pth",
        "--modified_model_path", f"./reslts/loss_vgg16_features.5_0.5.pth",
        "--model_name", f"vgg16",
        "--config_file", "./config_cub_3classes.yaml"
        "--save_dir", "./xai_images/integrated_gradient"
    ])
    """
    
    """    

    args = parser.parse_args([
        "--org_model_path", f"/home/srikanth/trained_models/pytorch/caltech/vgg16/vgg16.pth",
        "--modified_model_path", f"/mnt/data/results/{MODEL_NAME}/{RECALIBRATED_MODEL}",
        "--model_name", f"{MODEL_NAME}",
        "--save_dir", "./xai_images/integrated_gradient"
    ])
    """

    
    ''
    
    BASE_MODEL_PATH = args.org_model_path.strip()
    #Replace the model base path from the path override
    MODIFIED_MODEL_PATH = args.modified_model_path.strip()
    MODEL_NAME = args.model_name.strip().lower()
    config_file = args.config_file
    config = ConfigSingleton(config_file)
    
    XAI_Integrated_gradients = config.INTEGRATED_GRADIENT
    XAI_GradCAM              = config.GRADCAM
    XAI_Lime                 = config.LIME
    IMAGES                   = config.XAI_IMAGE_PATH
    num_classes              = config.XAI_NUMCLASSES
    save_dir                 = args.save_dir.strip()
    #print(XAI_Integrated_gradients,XAI_GradCAM,XAI_Lime, IMAGES)
    ################################################################################################################
    ############# Integrated Gradients ##########################################################
    if(XAI_Integrated_gradients == True):
            
        print(BASE_MODEL_PATH)
        try:
          model = get_model(BASE_MODEL_PATH)
        except Exception as e:
          print(e)
          exit()
        save_dir_before = os.path.join(save_dir,MODEL_NAME, 'integrated_gradient','before')
        xai_integrated_gradients(MODEL_NAME, model, num_classes, IMAGES, n_steps=200, save_dir = save_dir_before, title_prefix = "before")
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        save_dir_after = os.path.join(save_dir,MODEL_NAME, 'integrated_gradient','after')
        xai_integrated_gradients(MODEL_NAME, model_modified, num_classes, IMAGES, n_steps=200, save_dir = save_dir_after, title_prefix = "after")
    ################################################################################################################
    ############# GRAD CAM Implementation ##########################################################
    if(XAI_GradCAM == True):
        model = get_model(BASE_MODEL_PATH)
        save_dir_before = os.path.join(save_dir,MODEL_NAME, 'gradcam', 'before')
        xai_gradcam_explainer(MODEL_NAME, model,IMAGES, num_classes, save_dir_before, title_prefix ="before")
        save_dir_after = os.path.join(save_dir,MODEL_NAME,'gradcam',  'after')
        print(save_dir_before, save_dir_after)
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        xai_gradcam_explainer(MODEL_NAME, model_modified,IMAGES, num_classes, save_dir_after, title_prefix ="after")
    ################################################################################################################
    ############# Lime Implementation ##########################################################
    if(XAI_Lime == True):
        model = get_model(BASE_MODEL_PATH)
        save_dir_before = os.path.join(save_dir,MODEL_NAME, 'lime', 'before')
        xai_lime_explainer(MODEL_NAME, model, IMAGES, num_classes, save_dir_before)
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        save_dir_after = os.path.join(save_dir,MODEL_NAME, 'lime', 'after')
        xai_lime_explainer(MODEL_NAME, model_modified, IMAGES, num_classes, save_dir_after)

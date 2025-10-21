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
from dotenv import load_dotenv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from ConfigSingleton import ConfigSingleton
from xai_methods import (xai_integrated_gradients, find_last_conv_layer_pytorch, xai_gradcam_explainer, xai_lime_explainer)

XAI_Integrated_gradients = True
XAI_GradCAM              = True
XAI_Lime                 = False

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

    parser.add_argument(
        "--override_image_path", 
        action='store_true', 
        default=False, 
        help="Flag to enable image path override. If set, --image_path must be provided."
    )    
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
    #IMAGES                   = config.XAI_IMAGE_PATH
    num_classes              = config.XAI_NUMCLASSES
    save_dir                 = args.save_dir.strip()
    #print(XAI_Integrated_gradients,XAI_GradCAM,XAI_Lime, IMAGES)
    # Get image paths from config and create IMAGES list
    base_image_path = config.CLASSIFICATION_DATA_BASE_PATH
    print(args.override_image_path )
    if(args.override_image_path == True):
      base_image_path = input("? --override_image_path flag is set. Please enter the image path: ")        
    print(base_image_path)
    base_image_path = os.path.join(base_image_path, "train")
    print(base_image_path)
    image_dirs = [os.path.join(base_image_path, d) for d in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, d))]
    image_dirs = sorted(image_dirs)
    IMAGES = [[] for _ in range(0,len(image_dirs))]  # Create a list of lists for each class
    for i in range(0,len(image_dirs)):
        count  = 0
        # Iterate through all subdirectories
        for root, dirs, files in os.walk(image_dirs[i]):
            for file in files:
                
                # Check for common image extensions
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    IMAGES[i].append(os.path.join(root, file))
                    count = count + 1
                    if(count >=200) :
                        break
    # Sort the paths for consistent ordering
    #for i in range(0,len(IMAGES)):
	  #  print(f"Image files in {IMAGES[i]}")
    if not IMAGES:
        print(f"No images found in {image_dirs}")
        exit()
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

    ############# Lime Implementation ##########################################################
    if(XAI_Lime == True):
        model = get_model(BASE_MODEL_PATH)
        save_dir_before = os.path.join(save_dir,MODEL_NAME, 'lime', 'before')
        xai_lime_explainer(MODEL_NAME, model, IMAGES, num_classes, save_dir_before)
        model_modified = get_model(BASE_MODEL_PATH, MODIFIED_MODEL_PATH)
        save_dir_after = os.path.join(save_dir,MODEL_NAME, 'lime', 'after')
        xai_lime_explainer(MODEL_NAME, model_modified, IMAGES, num_classes, save_dir_after)

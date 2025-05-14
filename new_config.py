import os.path
import random
from re import L
import numpy as np
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_MODEL = "resnet50"

CLASSIFICATION_DATA_BASE_PATH = "/home/multiclass_classification"
TARGET_CLASS_LIST = ["deer","horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]

CONCEPT_FOLDER = [
    "/home/concept/deer/coat",             # for deer - Coat
    "/home/concept/horse/horse_skin",      # for horse
    "/home/concept/zebra/stripes"          # for zebra
]
RANDOM_FOLDER = "/home/concept/random/"

RESULTS_PATH = './results/' + BASE_MODEL + '/'

##Hyper parameters
LEARNING_RATE = 1e-3
EPOCHS = 1
BATCH_SIZE = 64
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0, 1.01, 0.10)]




# Dynamically determine the number of classes
def get_num_classes(base_path):
    return len([
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ])

def getbasemodel_imagesize(base_model):
  model_root = "/home/srikanth/trained_models/pytorch"
  model_path = os.path.join(model_root, BASE_MODEL + "/" + BASE_MODEL + ".pth")
  if(base_model == 'inception_v3'):
    image_size = 299
  else:
    image_size = 224
  return model_path, image_size

def getmodel_layers(model):
  layers = []
  # List only convolutional layers
  for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"Conv Layer Name: {name}, Layer Type: {type(module)}")
        layers.append(name)
    if isinstance(module, nn.MaxPool2d):
        print(f"MaxPool2d Layer Name: {name}, Layer Type: {type(module)}")
        layers.append(name)
  #Return the last 3 layers mostl they can be critical or the dimentionality will be reduced.         
  return (layers[-4:])  

def override_parameters(base_model):
    # Load the model
    MODEL_PATH, IMAGE_SIZE = getbasemodel_imagesize(base_model)
    MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
    LAYER_NAMES = getmodel_layers(MODEL)
    TRANSFORMS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),])
    MODEL.to(DEVICE)
    return MODEL_PATH, MODEL , LAYER_NAMES, TRANSFORMS

# Do not edit these parameters either they are automatically calulated or not used  
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#remove any trailing white spaces ad leading white spaces
BASE_MODEL = BASE_MODEL.strip().lower()
INITIAL_MODEL_PATH = "./model_weights/imbalanced_model.pth"

MODEL_PATH,IMAGE_SIZE = getbasemodel_imagesize(BASE_MODEL) 


#load model 
MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
TRANSFORMS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),])
MODEL.to(DEVICE)
#Get all bottleneck layers
LAYER_NAMES = getmodel_layers(MODEL)
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)


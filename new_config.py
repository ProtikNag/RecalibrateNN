import os.path
import random
import numpy as np
import torch
from torchvision import models
from torchvision import transforms

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSIFICATION_DATA_BASE_PATH = "/home/multiclass_classification"
TARGET_CLASS_LIST = ["deer","horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]
BASE_MODEL = "resnet50"

CONCEPT_FOLDER = [
    "/home/concept/deer/coat",             # for deer - Coat
    "/home/concept/horse/horse_skin",      # for horse
    "/home/concept/zebra/stripes"          # for zebra
]

RANDOM_FOLDER = "/home/concept/random"
RESULTS_PATH = "./results/"+ BASE_MODEL + "_retrained_model.pth"

MODEL_PATH = '/home/srikanth/trained_models/pytorch/resnet50/resnet50.pth'
#LAYER_NAME1 = 'features.26'
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
IMAGE_SIZE = 224


LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3

MODEL = torch.load(MODEL_PATH)
TRANSFORMS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),])
MODEL.to(DEVICE)

# LAYER_NAMES = ["conv_block3.0", "conv_block4.0", "fc.0"]


LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0, 1.01, 0.5)]



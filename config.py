import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import os

# Dynamically determine the number of classes
def get_num_classes(base_path):
    return len([
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ])

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
IMAGE_SIZE = 224
#Verify that the learnig rate of the model is similar to the learning rate used here 
LEARNING_RATE = 1e-2
EPOCHS = 10 
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONCEPT_FOLDER = "/home/srikanth/data_folder/dataset/data/concept/stripes_fake"
CONCEPT_FOLDER_FAKE = "/home/srikanth/data_folder/dataset/data/concept/stripes_fake"
RANDOM_FOLDER = "/home/srikanth/data_folder/dataset/data/concept/random"
MULTICLASS_CLASSIFICATION_BASE = "/home/srikanth/data_folder/dataset/data/binary_classification/"
RESULTS_PATH = "./results"
TARGET_CLASS_NAME = "zebra"

NUM_CLASSES = get_num_classes(MULTICLASS_CLASSIFICATION_BASE)
MODEL_PATH = "/home/srikanth/trained_models/pytorch/googlenet/googlenet_model.pth"
MODEL_WEIGHTS = "/home/srikanth/trained_models/pytorch/googlenet/googlenet_weights.pth"
# Load the model
# Step 1: Load the default GoogLeNet model
#MODEL = models.googlenet(pretrained=False)  # Set pretrained=True if you want to load the default pretrained weights
#MODEL.load_state_dict(torch.load(MODEL_WEIGHTS))
MODEL = torch.load(MODEL_PATH)
LAYER_NAME = "inception4a"
TRANSFORMS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),])

# These are tunable parameters 
#Lower the better 
LAMBDA_ALIGN = 0.75
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

RESULTS_FILE_NAME = f"retrained_L{LAMBDA_ALIGN}_{TARGET_CLASS_NAME}_Layer{LAYER_NAME}_weights.pth"

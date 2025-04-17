import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

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
NUM_CLASSES = 3
MODEL_PATH = "/home/srikanth/trained_models/pytorch/googlenet/googlenet_model.pth"
MODEL_WEIGHTS = "/home/srikanth/trained_models/pytorch/googlenet/googlenet_weights.pth"
# Load the model
# Step 1: Load the default GoogLeNet model
#MODEL = models.googlenet(pretrained=False)  # Set pretrained=True if you want to load the default pretrained weights
#MODEL.load_state_dict(torch.load(MODEL_WEIGHTS))
MODEL = torch.load(MODEL_PATH)
LAYER_NAME = "inception4a"
TRANSFORMS = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),])
CONCEPT_FOLDER = "/home/srikanth/data_folder/dataset/data/concept/stripes_fake"
CONCEPT_FOLDER_FAKE = "/home/srikanth/data_folder/dataset/data/concept/stripes_fake"
RANDOM_FOLDER = "/home/srikanth/data_folder/dataset/data/concept/random"
BINARY_CLASSIFICATION_BASE = "/home/srikanth/data_folder/dataset/data/binary_classification/"
RESULTS_PATH = "./results"
TARGET_CLASS_NAME = "zebra"
# These are tunable parameters 
#Lower the better 
LAMBDA_ALIGN = 0.75
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

RESULTS_FILE_NAME = f"retrained_L{LAMBDA_ALIGN}_{TARGET_CLASS_NAME}_Layer{LAYER_NAME}_weights.pth"

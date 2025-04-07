import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LEARNING_RATE = 1e-3
EPOCHS = 30
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
MODEL = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
MODEL.fc = nn.Linear(MODEL.fc.in_features, NUM_CLASSES).to(DEVICE)
LAYER_NAME = "inception4a"
#provide the concept of stripes only in case you want to generate the concept of tiger, then a new model will be generated for the same
#Currently this model can only handle 1 concept at a time
CONCEPT_FOLDER = "./data/concept/stripes_fake"
# store concepts that are veriy similar to the target concept
# for example, if the target concept is tiger, then the similar concepts that are not tigers but looks like tiger are leapord 
RANDOM_FOLDER = "./data/concept/random"
# currently contains 2 classes we need to change it in case if we have multiple classes 
BINARY_CLASSIFICATION_BASE = "./data/binary_classification/"
RESULTS_PATH = "./results/retrained_model.pth"
# Note that the folder name of the concept should exactly match the name here
ZEBRA_CLASS_NAME = "zebra"

# These are tunable parameters 
LAMBDA_ALIGN = 0.75
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

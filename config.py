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
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2
MODEL = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
MODEL.fc = nn.Linear(MODEL.fc.in_features, NUM_CLASSES).to(DEVICE)
LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
BINARY_CLASSIFICATION_BASE = "./data/binary_classification/"
RESULTS_PATH = "./results/retrained_model.pth"
ZEBRA_CLASS_NAME = "zebra"
LAMBDA_ALIGN = 0.75
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

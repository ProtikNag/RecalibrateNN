import random
import numpy as np
import torch
from model import DeepCNN
from train import training_biased_model

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CONCEPT_FOLDER = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification/"
RESULTS_PATH = "./results/retrained_model.pth"

LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2
MODEL = training_biased_model(
    DeepCNN(num_classes=NUM_CLASSES).to(DEVICE),
    CLASSIFICATION_DATA_BASE_PATH
)
LAYER_NAME = "conv_block4"
TARGET_CLASS_NAME = "zebra"
LAMBDA_ALIGN = 0.5
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

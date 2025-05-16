import os.path
import random
import numpy as np
import torch
from model import DeepCNN

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification/"
TARGET_CLASS_LIST = ["deer", "horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]

CONCEPT_FOLDER = [
    "./data/concept/deer/coat",             # for deer - Coat
    "./data/concept/horse/horse_skin",      # for horse
    "./data/concept/zebra/stripes"          # for zebra
]
RANDOM_FOLDER = "./data/concept/random"
RESULTS_PATH = "./results/retrained_model.pth"
INITIAL_MODEL_PATH = "./model_weights/imbalanced_model.pth"

LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
MODEL = DeepCNN(num_classes=NUM_CLASSES)
MODEL.load_state_dict(torch.load(INITIAL_MODEL_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE)
# LAYER_NAMES = ["conv_block3.0", "conv_block4.0", "fc.0"]
LAYER_NAMES = ["conv_block4.0"]
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0, 1.01, 0.5)]



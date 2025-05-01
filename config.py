import random
import numpy as np
import torch
from model import DeepCNN

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGET_CLASS_1_FOLDER = "./data/multi_class_classification/tiger/train"
TARGET_CLASS_2_FOLDER = "./data/multi_class_classification/zebra/train"
CONCEPT_FOLDER_1 = "./data/concept/fur_texture"
CONCEPT_FOLDER_2 = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification/"
RESULTS_PATH = "./results/retrained_model.pth"
MODEL_CHECKPOINT_PATH = "./model_weights/imbalanced_model.pth"

LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 2
MODEL = DeepCNN(num_classes=NUM_CLASSES)
MODEL.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE)
# LAYER_NAMES = ["conv_block3.0", "conv_block4.0", "fc.0"]
LAYER_NAMES = ["conv_block4.0"]
TARGET_CLASS_1 = "tiger"
TARGET_CLASS_2 = "zebra"
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0, 1.01, 0.5)]



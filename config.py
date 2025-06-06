import os.path
import random
import numpy as np
import torch
from utils import get_num_classes

# Do not edit these parameters either they are automatically calculated or not used
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification"
TARGET_CLASS_LIST = ["deer", "horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]


CONCEPT_FOLDER_LIST = [
    "./data/concept/deer/coat",             # for deer - Coat
    "./data/concept/horse/horse_skin",      # for horse
    "./data/concept/zebra/stripes"          # for zebra
]
RANDOM_FOLDER = "./data/concept/random/"
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
LINEAR_CLASSIFIER_TYPE = 'SGDClassifier'
#LINEAR_CLASSIFIER_TYPE = 'LogisticRegression'

##Hyper parameters
LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 64
#LAMBDA_ALIGNS = [round(i.item(), 2) for i in np.arange(0.5, 1.01, 3.10)]
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0, 1.01, 0.10)]
#LAMBDA_ALIGNS = [0.5]

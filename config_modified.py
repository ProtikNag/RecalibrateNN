import os.path
import random
import numpy as np
import torch
from utils import get_num_classes

from config import *

CLASSIFICATION_DATA_BASE_PATH = "/home/multiclass_classification"
TARGET_CLASS_LIST = ["deer", "horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]
CONCEPT_FOLDER_LIST = [
    "/home/concept/deer/coat",             # for deer - Coat
    "/home/concept/horse/horse_skin",      # for horse
    "/home/concept/zebra/stripes"          # for zebra
]

RANDOM_FOLDER = "/home/concept/random/"
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
LINEAR_CLASSIFIER_TYPE = 'SGDClassifier'
#LINEAR_CLASSIFIER_TYPE = 'LogisticRegression'

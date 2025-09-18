import os.path
import random
import numpy as np
import torch
from utils import get_num_classes

from config import *

CLASSIFICATION_DATA_BASE_PATH = "/mnt/sdd/caltech_training/recalib/"
TARGET_CLASS_LIST = ["001.Black_footed_Albatross", "044.Frigatebird", "100.Brown_Pelican"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]
CONCEPT_FOLDER_LIST = [
    "/mnt/sdd/caltech/concepts/beak/Black_Footed_Albatross",             # for 001.Black_footed_Albatross
    "/mnt/sdd/caltech/concepts/beak/Frigatebird",      # for 044.Frigatebird
    "/mnt/sdd/caltech/concepts/beak/Brown_Pelican"          # for 100.Brown_Pelican
]

ENABLE_DEBUGPRINT = 1
RANDOM_FOLDER = "/home/concept/random/"
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
LINEAR_CLASSIFIER_TYPE = 'SGDClassifier'
#LINEAR_CLASSIFIER_TYPE = 'LogisticRegression'
EPOCHS = 25
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0.4, 0.8, 0.10)]
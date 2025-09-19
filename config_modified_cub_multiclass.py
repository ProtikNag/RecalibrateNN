import os.path
import random
import numpy as np
import torch
from utils import get_num_classes

from config import *

CLASSIFICATION_DATA_BASE_PATH = "/mnt/sdd/caltech_training/recalib_multiclass/"
TARGET_CLASS_LIST = ["002.Laysan_Albatross",
                      "003.Sooty_Albatross",
                      "004.Groove_billed_Ani",
                      "005.Crested_Auklet",
                      "006.Least_Auklet",
                      "007.Parakeet_Auklet",
                      "008.Rhinoceros_Auklet",
                      "009.Brewer_Blackbird",
                      "010.Red_winged_Blackbird",
                      "011.Rusty_Blackbird", ]

TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]
CONCEPT_FOLDER_LIST = [
    "/mnt/sdd/caltech/concepts/beak/Laysan_Albatross",
    "/mnt/sdd/caltech/concepts/beak/Sooty_Albatross",
    "/mnt/sdd/caltech/concepts/beak/Groove_billed_Ani",
    "/mnt/sdd/caltech/concepts/beak/Crested_Auklet",
    "/mnt/sdd/caltech/concepts/beak/Least_Auklet",
    "/mnt/sdd/caltech/concepts/beak/Parakeet_Auklet",
    "/mnt/sdd/caltech/concepts/beak/Rhinoceros_Auklet",
    "/mnt/sdd/caltech/concepts/beak/Brewer_Blackbird",
    "/mnt/sdd/caltech/concepts/beak/Red_winged_Blackbird",
    "/mnt/sdd/caltech/concepts/beak/Rusty_Blackbird",
]

ENABLE_DEBUGPRINT = 1
RANDOM_FOLDER = "/home/concept/random/"
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
LINEAR_CLASSIFIER_TYPE = 'SGDClassifier'
#LINEAR_CLASSIFIER_TYPE = 'LogisticRegression'
EPOCHS = 25
LAMBDA_ALIGNS = [round(i.item(),2) for i in np.arange(0.4, 0.8, 0.10)]
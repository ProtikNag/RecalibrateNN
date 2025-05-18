import os.path
import random
import numpy as np
import torch
from model import DeepCNN
from utils import get_num_classes, get_model_layers, get_base_model_image_size, get_model_weight_path

# Do not edit these parameters either they are automatically calculated or not used
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_MODEL = "imbalanced_model_1"
BASE_MODEL_PATH = get_model_weight_path(BASE_MODEL)
CLASSIFICATION_DATA_BASE_PATH = "./data/multi_class_classification_1"
TARGET_CLASS_LIST = ["deer", "horse", "zebra"]
TARGET_FOLDER_LIST = [os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train") for class_name in TARGET_CLASS_LIST]
CONCEPT_FOLDER_LIST = [
    "./data/concept/deer/coat",  # for deer - Coat
    "./data/concept/horse/horse_skin",  # for horse
    "./data/concept/zebra/stripes"  # for zebra
]
RANDOM_FOLDER = "./data/concept/random/"
RESULTS_PATH = './results/' + BASE_MODEL + '/'

##Hyper parameters
LEARNING_RATE = 1e-3
EPOCHS = 15
BATCH_SIZE = 64
LAMBDA_ALIGNS = [round(i.item(), 2) for i in np.arange(0.5, 1.01, 3.10)]

# remove any trailing white spaces ad leading white spaces
BASE_MODEL = BASE_MODEL.strip().lower()
IMAGE_SIZE = get_base_model_image_size(BASE_MODEL)
NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)

# Load the model
MODEL = DeepCNN(num_classes=NUM_CLASSES)
MODEL.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=True))
MODEL.to(DEVICE)
LAYER_NAMES = get_model_layers(MODEL)

if __name__ == "__main__":
    # Print and test the configuration parameters
    print("Device Avilable: ", DEVICE)
    print("Base model: ", BASE_MODEL, " Base model path: ", BASE_MODEL_PATH)
    print("Classification_data_base_path: ", CLASSIFICATION_DATA_BASE_PATH)
    print("Target folder list: ", TARGET_FOLDER_LIST)
    print("Concept folder list: ", CONCEPT_FOLDER_LIST)
    print("Random folder: ", RANDOM_FOLDER)
    print("Image size: ", IMAGE_SIZE)
    print("Number of classes: ", NUM_CLASSES)
    print("Last 4 layer names: ", LAYER_NAMES)
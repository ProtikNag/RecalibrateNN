import os
import copy
import pandas as pd
import os.path
import numpy as np
import argparse


from logger import Logger_Singleton
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from datetime import datetime

from config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
    DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
    CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
)
from dataset import get_layer_list, get_image_dataset, get_lambda_val, get_model_path

if(os.environ.get('PLATFORM') == "Srikanth"):
  print("overriding config paths to point to directory structure of srikanth. Note Protik will not have this parameter set ") 
  from config_modified import (
      LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
      DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
      CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
  )
  print("Taking all the required path from Srikanths folder" )

if(os.environ.get('PLATFORM') == "CUB"):
  print("overriding config paths to point to directory structure of srikanth. Note Protik will not have this parameter set with CALTECH Data ") 
  from config_modified_cub import (
      LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
      DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
      CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
  )
  print("Taking all the required path from CALTECH Dataset folder" )

if(os.environ.get('PLATFORM') == "IMAGENET"):
  print("overriding config paths to point to directory structure of srikanth. Note Protik will not have this parameter set with IMAGENET Data ") 
  from config_modified_inet import (
      LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
      DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
      CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
  )
  print("Taking all the required path from IMAGENET Dataset folder" )

from utils import (
    get_base_model_image_size, get_model_layers, predict_from_loader,load_model, load_model_statedict,get_class_folder_dicts,
    load_train_valid_dataset, load_train_dataset_concept_random     
)

from tcav_utils import (util_compute_cav, util_compute_sensitivity_score, util_compute_tcav_score_from_sensitivity)


MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None

activation = {}
output_shape = {}
df = pd.DataFrame()

def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        # This print has been added for you to visualize if the size is too large then the time taken fror convergence will be large
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
        #logging.info(f"Verify the output shape : Layername = {layer_name} ,Input.shape : {input[0].shape},  output.shape : {output.shape}")
    return hook




if __name__ == "__main__":
    ############## Parser #################################
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Obtainthe original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--recal_model_basepath", type=str, default=None, help="Specify a location of the recalibrated model base path")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--before_after", action='store_true', help="Default parameter for before after comparison if its true then before after comparison will be done")
    args = parser.parse_args()
    # Take the parameters passed by the program instead of user as this is running in debug mode. 
    if(os.getenv('DEBUG')):
        #args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth",  "--recal_model_basepath", "/mnt/data/results/" , "--model_name", "vgg16"])
        args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/resnet50/resnet50.pth", "--recal_model_basepath", "/mnt/data/results/" , "--model_name", "resnet50"])
        #args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/mobilenet_v3_small/mobilenet_v3_small.pth", "--recal_model_basepath", "/mnt/data/results/" ,  "--model_name", "mobilenet_v3_small"])
        #args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/mobilenet_v3_large/mobilenet_v3_large.pth",  "--recal_model_basepath", "/mnt/data/results/" , "--model_name", "mobilenet_v3_large"])
        #args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/inception_v3/inception_v3.pth",  "--recal_model_basepath", "/mnt/data/results/" , "--model_name", "inception_v3"])

    before_after = args.before_after  
    BASE_MODEL_PATH = args.org_model_path
    MODEL_NAME = args.model_name
    lambda_val_list = get_lambda_val(MODEL_NAME)
    if(args.recal_model_basepath):
        recal_model_basepath = args.recal_model_basepath
    else:
        recal_model_basepath = None
    
    
    if not BASE_MODEL_PATH or not MODEL_NAME:
        raise ValueError("Please provide valid paths for org_model_path, and model_name")
    print(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ############## Logging #################################
    log_filename = f"results/{MODEL_NAME}/{formatted_datetime}_sensitivity_compute.log"
    logger = Logger_Singleton(log_filename)   
    logger.info(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    logger.info(f"Using device: {DEVICE}")
    dataframe_filename = f"./results/{MODEL_NAME}/sensitivity_audit_trail_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    # Set the device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ############## Model BEFORE #################################   
    #Load the model
    model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
    model_trained.to(device)
    full_filelist = []
    full_class_idx = []
    if(before_after == True):
      
      layers = get_layer_list(MODEL_NAME)
    else:
      MODEL = load_model(MODEL_NAME, BASE_MODEL_PATH)
      layers = get_model_layers(MODEL)
      del (MODEL)
    ############## Load data #################################
    dataset_loader, val_loader,TRAIN_TRANSFORM , VALID_TRANSFORM, class_names = load_train_valid_dataset(MODEL_NAME,CLASSIFICATION_DATA_BASE_PATH,BATCH_SIZE)
    TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]
    class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train"), \
                                                              transform=VALID_TRANSFORM), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
    print(dataset_loader)
    filelist = dataset_loader.dataset.getfilelist()
    for i,j in zip(filelist[0], filelist[1]):
        full_filelist.append(i)
        full_class_idx.append(j)
    df["Full filepath"] = full_filelist
    df["Full Class Index"] = full_class_idx
    print(df["Full filepath"], df["Full Class Index"])
    
    concept_loader_list, random_loader = load_train_dataset_concept_random(MODEL_NAME, CONCEPT_FOLDER_LIST, RANDOM_FOLDER, BATCH_SIZE)
    stored_cav_vector = {}
    for lambda_val in lambda_val_list:
        for layer_name in layers:
          try:
              ############## BEFORE #################################
              model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
              hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
              model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
              print("Computing the cav vectors can take a while stand by")
              logger.info("Computing the cav vectors can take a while stand by")
              cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation) for concept_loader in concept_loader_list]
              stored_cav_vector[layer_name] = cav_vectors 
              logger.info("Computing the sensitivity score can take a while stand by")
              independent_sensitivityscore = [util_compute_sensitivity_score(model_trained, layer_name, cav, class_loader, idx, activation) \
                                           for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
              logger.info(f"Sensitivity score for each image is {independent_sensitivityscore}")
              independent_sensitivityscore = [cpudata.cpu().numpy() for cpudata in independent_sensitivityscore]
              tcav_before = util_compute_tcav_score_from_sensitivity(independent_sensitivityscore)
              independent_sensitivityscore = np.concatenate(independent_sensitivityscore)
              logger.info(f"tcav_before is {tcav_before}")
              sensitivityscore_Before = f"sensitivityscore_before_{layer_name}_{lambda_val}"
              df[sensitivityscore_Before ] = independent_sensitivityscore
              hook_handle.remove()
              activation.clear()  # Clear activations to free memory
              torch.cuda.empty_cache()
              print(before_after)
              if(before_after == True):
                print("Evaluating the output of model after")
                ############## AFTER ###################################
                ############## Model AFTER #################################   
                #Load the model
                if(recal_model_basepath != None):
                    modified_model_path = get_model_path(MODEL_NAME, layer_name, lambda_val,recal_model_basepath)
                else:
                    modified_model_path = get_model_path(MODEL_NAME, layer_name, lambda_val)
                model_trained = load_model_statedict(model_trained, modified_model_path)
                model_trained.to(device)
                model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                independent_sensitivityscore = [util_compute_sensitivity_score(model_trained, layer_name, cav, class_loader, idx, activation) \
                                      for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
                logger.info(f"Sensitivity score for each image After is {independent_sensitivityscore}")
                independent_sensitivityscore = [cpudata.cpu().numpy() for cpudata in independent_sensitivityscore]
                tcav_after = util_compute_tcav_score_from_sensitivity(independent_sensitivityscore)
                independent_sensitivityscore = np.concatenate(independent_sensitivityscore)
                logger.info(f"tcav_after is {tcav_after}")
                sensitivityscore_After = f"sensitivityscore_After_{layer_name}_{lambda_val}"
                df[sensitivityscore_After ] = independent_sensitivityscore
                hook_handle.remove()
                activation.clear()  # Clear activations to free memory
                torch.cuda.empty_cache()
                df.to_csv(dataframe_filename, index = False)
              else:
                # We need to go just once for before only computation and n times for before after based on lambda hence break
                df.to_csv(dataframe_filename, index = False) 
                continue
          except Exception as e:
            print(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}")
            logger.info(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}")
          try:
            del model_trained
          except Exception as e:
            print(f"Model trained variable not yet defined  ")
          if(before_after == False):
            break
    
    
    

#load_model_statedict load_model_statedict(model_name, model_path):

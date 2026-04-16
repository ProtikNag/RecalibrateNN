import os
import copy
import pandas as pd
import os.path
import numpy as np
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from logger import Logger_Singleton
from dotenv import load_dotenv
from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
import random

from ConfigSingleton import ConfigSingleton
from utils import (
    get_base_model_image_size, get_model_layers, predict_from_loader,load_model, load_model_statedict,get_class_folder_dicts,
    load_train_valid_dataset, load_train_dataset_concept_random, get_model_weight_path 
)

from tcav_utils import (util_compute_cav, util_compute_sensitivity_score, util_compute_tcav_score_from_sensitivity)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


RANDOM_STATE = 132
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def worker_init_fn(worker_id):
    worker_seed = RANDOM_STATE + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# Set global seed
set_seed(RANDOM_STATE)

def get_model_path(MODEL_NAME, layer_name, lambda_val, recalibrated_model_base_path='./results'):
    base_path = None
    if(recalibrated_model_base_path != None):
        base_path = recalibrated_model_base_path +MODEL_NAME.strip() + '/loss_' + MODEL_NAME.strip() + '_' + layer_name.strip() + '_' + str(lambda_val) + '.pth' 
        print(base_path)
        try:
            with open(base_path, 'r') as f:
                pass
        except Exception as e:
            print("File not found in the given path ")
            raise FileNotFoundError(f"The file '{base_path}' was not found.")
    else:
        base_path = recalibrated_model_base_path +MODEL_NAME.strip() + '/loss_' + MODEL_NAME.strip() + '_' + layer_name.strip() + '_' + str(lambda_val) + '.pth'
        
        try:
            with open(base_path, 'r') as f:
                pass
        except Exception as e:
            print("File not found in the given path ")
            raise FileNotFoundError(f"The file '{base_path}' was not found.")
    print("Base path where the model is located ",base_path)
    return (base_path)
        
    

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
OVERRIDE_RECALIB = None


activation = {}
output_shape = {}
df = pd.DataFrame()

def build_results_df(base_df, sensitivity_columns):
    if not sensitivity_columns:
        return base_df.copy()
    extra_df = pd.DataFrame(sensitivity_columns, index=base_df.index)
    return pd.concat([base_df, extra_df], axis=1).copy()

def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        # This print has been added for you to visualize if the size is too large then the time taken fror convergence will be large
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
        #logging.info(f"Verify the output shape : Layername = {layer_name} ,Input.shape : {input[0].shape},  output.shape : {output.shape}")
    return hook


def get_layernames_override(MODEL_NAME, config):
    if(MODEL_NAME == 'vgg16'):
        return (config.VGG_RECALIB)
    if(MODEL_NAME == 'resnet50'):
        return (config.RESNET50_RECALIB)
    if(MODEL_NAME == 'inception_v3'):
        return (config.INCEPTION_V3_RECALIB)
    if(MODEL_NAME == 'mobilenet_v3_small'):
        return (config.MOBILENET_V3_SMALL_RECALIB)
    if(MODEL_NAME == 'mobilenet_v3_large'):
        return (config.MOBILENET_V3_LARGE_RECALIB)





if __name__ == "__main__":
    ############## Parser #################################
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Obtainthe original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--recal_model_basepath", type=str, default=None, help="Specify a location of the recalibrated model base path")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--before_after", action='store_true', help="Default parameter for before after comparison if its true then before after comparison will be done")
    parser.add_argument("--config", type=str, default=None, help="specify the yaml file ")
    parser.add_argument("--store_results", type=str, default=None, help="specify the location where teh results should be stored ")
    parser.add_argument("--load_validation_dataset", action='store_true', help="If passed the validation dataset will be loaded instead (default: False)")

    
    args = parser.parse_args()
    config_file = args.config
    save_dir = args.store_results
    load_validationdataset = args.load_validation_dataset
    print(config_file)
    if config_file is not None:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    else:
        raise FileNotFoundError(f"Config file parameter not provided in the command line")
    config = ConfigSingleton(config_file)
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    if(DEVICE =='cuda'):
      torch.cuda.manual_seed(RANDOM_STATE)
      torch.cuda.manual_seed_all(RANDOM_STATE)  # For multi-GPU setups
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    if(DEVICE =='cuda'):
      # Ensure deterministic behavior (may impact performance)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    CLASSIFICATION_DATA_BASE_PATH = config.CLASSIFICATION_DATA_BASE_PATH
    TARGET_CLASS_LIST = config.TARGET_CLASS_LIST
    RANDOM_FOLDER = config.RANDOM_FOLDER
    CONCEPT_FOLDER_LIST = config.CONCEPT_FOLDER_LIST
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    NUM_CLASSES = config.NUM_CLASSES
    LAMBDA_ALIGNS = config.LAMBDA_ALIGNS
    LINEAR_CLASSIFIER_TYPE = config.LINEAR_CLASSIFIER_TYPE
    
    print("Config file loaded successfully.")
    # Get the training dataset and the validation dataset folders
    
    before_after = args.before_after  
    BASE_MODEL_PATH = args.org_model_path
    MODEL_NAME = args.model_name
    lambda_val_list = LAMBDA_ALIGNS
    if(args.recal_model_basepath):
        recal_model_basepath = args.recal_model_basepath
    else:
        recal_model_basepath = None
    
    
    if not BASE_MODEL_PATH or not MODEL_NAME:
        raise ValueError("Please provide valid paths for org_model_path, and model_name")
    print(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ############## Logging #################################
    
    save_folder = f"{save_dir}/{MODEL_NAME}"
    os.makedirs(save_folder, exist_ok=True)
    log_filename = os.path.join(save_folder , f"{formatted_datetime}_sensitivity_compute.log")

    dataframe_filename = os.path.join(save_folder , f"sensitivity_audit_trail_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    
    logger = Logger_Singleton(log_filename)   
    logger.info(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    logger.info(f"Using device: {DEVICE}")
    print(f"Results are stored in {log_filename}, {dataframe_filename}")
    # Set the device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ############## Model BEFORE #################################   
    #Load the model
    print(MODEL_NAME, BASE_MODEL_PATH)
    model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
    model_trained.to(device)
    full_filelist = []
    full_class_idx = []
    if(before_after == True):
        if(config.OVERRIDE_RECALIB == True):
            layers  =   get_layernames_override(MODEL_NAME, config)
            print(layers)
        else:
            layers = get_model_layers(model_trained)
            print(layers)
    else:
        MODEL = load_model(MODEL_NAME, BASE_MODEL_PATH)
        layers = get_model_layers(MODEL)
        del (MODEL)
    ############## Load data #################################
    dataset_loader, val_loader,TRAIN_TRANSFORM , VALID_TRANSFORM, class_names = load_train_valid_dataset(MODEL_NAME,CLASSIFICATION_DATA_BASE_PATH,BATCH_SIZE, random_state = RANDOM_STATE)
    TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]
    if(load_validationdataset):
          #Load validation dataset instead
          #Update the data loader with val_loader since we are overriding using valication dataset
          #This is just used for the filelist 
          dataset_loader =  val_loader
          class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH,"valid",class_name), \
                                                              transform=VALID_TRANSFORM), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
    else:
          class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH,"train",class_name), \
                                                              transform=VALID_TRANSFORM), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
    print(dataset_loader)
    filelist = dataset_loader.dataset.getfilelist()
    for i,j in zip(filelist[0], filelist[1]):
        full_filelist.append(i)
        full_class_idx.append(j)
    #df["Full filepath"] = full_filelist
    #df["Full Class Index"] = full_class_idx
    #print(df["Full filepath"], df["Full Class Index"])  
    base_df = pd.DataFrame({
        "Full filepath": full_filelist,
        "Full Class Index": full_class_idx,
    })
    sensitivity_columns = {}
    print(base_df["Full filepath"], base_df["Full Class Index"])  
    concept_loader_list, random_loader = load_train_dataset_concept_random(MODEL_NAME, CONCEPT_FOLDER_LIST, RANDOM_FOLDER, BATCH_SIZE)
    stored_cav_vector = {}
    for layer_name in layers:
        try:
            ############## BEFORE Do it once #################################
            model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
            hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
            model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
            print("Computing the cav vectors can take a while stand by")
            logger.info("Computing the cav vectors can take a while stand by")
            #cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation, LINEAR_CLASSIFIER_TYPE) for concept_loader in concept_loader_list]
            cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation, LINEAR_CLASSIFIER_TYPE, random_state=RANDOM_STATE + i) for i, concept_loader in enumerate(concept_loader_list)]
            stored_cav_vector[layer_name] = cav_vectors 
            logger.info("Computing the sensitivity score can take a while stand by")
            independent_sensitivityscore = [util_compute_sensitivity_score(model_trained, layer_name, cav, class_loader, idx, activation) \
                                                    for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
            logger.info(f"Sensitivity score for each image is {independent_sensitivityscore}")
            independent_sensitivityscore = [cpudata.cpu().numpy() for cpudata in independent_sensitivityscore]
            tcav_before = util_compute_tcav_score_from_sensitivity(independent_sensitivityscore)
            independent_sensitivityscore = np.concatenate(independent_sensitivityscore)
            logger.info(f"tcav_before is {tcav_before}")
            sensitivityscore_Before = f"sensitivityscore_before_{layer_name}"
            #df[sensitivityscore_Before ] = independent_sensitivityscore
            sensitivity_columns[sensitivityscore_Before] = independent_sensitivityscore
            
            hook_handle.remove()
            activation.clear()  # Clear activations to free memory
            torch.cuda.empty_cache()
            #df.to_csv(dataframe_filename, index = False)
            build_results_df(base_df, sensitivity_columns).to_csv(dataframe_filename, index=False)
            try:
                del model_trained
            except Exception as e:
                print(f"Model trained variable not yet defined  ")
            if(before_after == True):
                for lambda_val in lambda_val_list:
                    try:
                        ###########AFTER######################
                        if(recal_model_basepath != None):
                            modified_model_path = get_model_path(MODEL_NAME, layer_name, lambda_val,recal_model_basepath)
                            model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
                            model_trained = load_model_statedict(model_trained, modified_model_path)
                            hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                            model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                            model_trained.to(device)
                            print("Computing the cav vectors can take a while stand by")
                            logger.info("Computing the cav vectors can take a while stand by")
                            try:
                                print(concept_loader_list , random_loader) 
                                #cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation,LINEAR_CLASSIFIER_TYPE) for concept_loader in concept_loader_list]
                                cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation, LINEAR_CLASSIFIER_TYPE, random_state=RANDOM_STATE + i) for i, concept_loader in enumerate(concept_loader_list)]
                                stored_cav_vector[layer_name] = cav_vectors 
                                logger.info("Computing the sensitivity score can take a while stand by")
                                independent_sensitivityscore = [util_compute_sensitivity_score(model_trained, layer_name, cav, class_loader, idx, activation) \
                                                                    for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
                                logger.info(f"Sensitivity score for each image is {independent_sensitivityscore}")
                                independent_sensitivityscore = [cpudata.cpu().numpy() for cpudata in independent_sensitivityscore]

                                tcav_after = util_compute_tcav_score_from_sensitivity(independent_sensitivityscore)
                                independent_sensitivityscore = np.concatenate(independent_sensitivityscore)
                                logger.info(f"tcav_after is {tcav_after}")
                                sensitivityscore_After = f"sensitivityscore_After_{layer_name}_{lambda_val}"
                                #df[sensitivityscore_After ] = independent_sensitivityscore
                                sensitivity_columns[sensitivityscore_After] = independent_sensitivityscore

                                hook_handle.remove()
                                activation.clear()  # Clear activations to free memory
                                torch.cuda.empty_cache()
                                #df.to_csv(dataframe_filename, index = False)
                                build_results_df(base_df, sensitivity_columns).to_csv(dataframe_filename, index=False)  
                            except Exception as e:
                                print(f"Exception obtained while computing util_compute_cav {e}")
                                continue
                    except Exception as e:
                        #df.to_csv(dataframe_filename, index = False)
                        build_results_df(base_df, sensitivity_columns).to_csv(dataframe_filename, index=False)  
                        print(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}")
                        logger.info(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}")
                        continue
                try:
                    del model_trained
                except Exception as e:
                    print(f"Model trained variable not yet defined  ")
                    continue
        except Exception as e:
            print(f"Obtained exception while processing Layer{layer_name} , Exception details {e}")
            logger.info(f"Obtained exception while processing Layer{layer_name} , Exception details {e}")
            continue
        

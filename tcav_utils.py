import copy
import os.path
from logger import Logger_Singleton
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DeepCNN

from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from datetime import datetime
import argparse
from config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
    DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
    CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
)

if(os.environ.get('PLATFORM') == "Srikanth"):
  print("overriding config paths to point to directory structure of srikanth. Note Protik will not have this parameter set ") 
  from config_modified import (
      LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
      DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
      CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
  )
  print("Taking all the required path from Srikanths folder" )


from utils import (
    get_num_classes, get_class_folder_dicts, train_cav, evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence, get_model_weight_path, get_base_model_image_size, get_model_layers, predict_from_loader 
)


def util_compute_cav(model, loader_positive, loader_random, layer_name, activation, orthogonal=False, dump_cav=True):
    logging = Logger_Singleton()
    pos_acts, rnd_acts = [], []
    model.eval()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for imgs in loader_positive:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            pos_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
        for imgs in loader_random:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            rnd_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
    pos_acts = np.vstack(pos_acts)
    rnd_acts = np.vstack(rnd_acts)
    
    cav = train_cav(pos_acts, rnd_acts, orthogonal, LINEAR_CLASSIFIER_TYPE)
    if(dump_cav == True):
      cav_filename = '.'+os.sep+'cav'+os.sep+f"{layer_name}_cav.pkl"
      os.makedirs('.'+os.sep+'cav', exist_ok=True)
      with open(cav_filename, 'wb') as f:
         pickle.dump(cav, f)
    logging.info(f"CAV for layer {layer_name} trained")
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)


def util_compute_tcav_score(model, layer_name, cav_vector, dataset_loader, target_idx, activation):
    logging = Logger_Singleton()
    logging.info(f"Computing TCAV score for layer: {layer_name}, target index: {target_idx}")
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, target_idx]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()  # check the documentation for the default values
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = F.normalize(grad_flat, p=2, dim=1)
            S = (grad_norm * cav_vector).sum(dim=1)
            scores.append(S > 0)  # additional logging for sensitivity analysis
    scores = torch.cat(scores)
    logging.info(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    print(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    return scores.float().mean().item()
    
def util_compute_sensitivity_score(model, layer_name, cav_vector, dataset_loader, target_idx, activation ):
    logging = Logger_Singleton()
    logging.info(f"Computing TCAV score for layer: {layer_name}, target index: {target_idx}")
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, target_idx]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()  # check the documentation for the default values
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = F.normalize(grad_flat, p=2, dim=1)
            S = (grad_norm * cav_vector).sum(dim=1)
            #append the real sensitivity score
            scores.append(S)  # additional logging for sensitivity analysis
    scores = torch.cat(scores)
    logging.info(f"Sensitivity score : {layer_name}, target index: {target_idx}, Scores: {scores}")
    print(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}, Scores: {scores}")
    return scores
    
    
def util_compute_tcav_score_from_sensitivity(scores):
    logging = Logger_Singleton()
    tcav_score = scores.float().mean().item()
    logging.info(f"TCAV score {tcav_score}")
    return
    

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

from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from datetime import datetime
import argparse

from utils import (
    get_num_classes, get_class_folder_dicts, train_cav, evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence, get_model_weight_path, get_base_model_image_size, get_model_layers, predict_from_loader 
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def uils_getlayers(model_name):
    LAYER_NAMES = None
    if(model_name == 'vgg16'):
        LAYER_NAMES = ['features.2','features.5','features.7','features.12','features.14','features.17','features.19','features.24']
    if(model_name == 'inception_v3'):
        LAYER_NAMES = ['Mixed_5b.branch5x5_2.conv','Mixed_5b.branch3x3dbl_1.conv','Mixed_5b.branch3x3dbl_2.conv','Mixed_5b.branch3x3dbl_3.conv','Mixed_5b.branch_pool.conv','Mixed_5c.branch1x1.conv', 
'Mixed_5c.branch5x5_1.conv','Mixed_5c.branch5x5_2.conv','Mixed_5c.branch3x3dbl_1.conv','Mixed_5c.branch3x3dbl_2.conv','Mixed_5c.branch3x3dbl_3.conv','Mixed_5c.branch_pool.conv',
'Mixed_5d.branch1x1.conv','Mixed_5d.branch5x5_1.conv','Mixed_5d.branch5x5_2.conv','Mixed_5d.branch3x3dbl_1.conv','Mixed_5d.branch3x3dbl_2.conv','Mixed_5d.branch3x3dbl_3.conv',
'Mixed_5d.branch_pool.conv','Mixed_6a.branch3x3.conv','Mixed_6a.branch3x3dbl_1.conv','Mixed_6a.branch3x3dbl_2.conv','Mixed_6a.branch3x3dbl_3.conv','Mixed_6b.branch1x1.conv',
'Mixed_6b.branch7x7_1.conv','Mixed_6b.branch7x7_2.conv','Mixed_6b.branch7x7_3.conv','Mixed_6b.branch7x7dbl_1.conv','Mixed_6b.branch7x7dbl_2.conv','Mixed_6b.branch7x7dbl_3.conv',
'Mixed_6b.branch7x7dbl_4.conv','Mixed_6b.branch7x7dbl_5.conv','Mixed_6b.branch_pool.conv','Mixed_6c.branch1x1.conv','Mixed_6c.branch7x7_1.conv','Mixed_6c.branch7x7_2.conv',
'Mixed_6c.branch7x7_3.conv','Mixed_6c.branch7x7dbl_1.conv','Mixed_6c.branch7x7dbl_2.conv','Mixed_6c.branch7x7dbl_3.conv','Mixed_6c.branch7x7dbl_4.conv','Mixed_6c.branch7x7dbl_5.conv',
'Mixed_6c.branch_pool.conv','Mixed_6d.branch1x1.conv','Mixed_6d.branch7x7_1.conv','Mixed_6d.branch7x7_2.conv','Mixed_6d.branch7x7_3.conv','Mixed_6d.branch7x7dbl_1.conv',
'Mixed_6d.branch7x7dbl_2.conv','Mixed_6d.branch7x7dbl_3.conv','Mixed_6d.branch7x7dbl_4.conv','Mixed_6d.branch7x7dbl_5.conv','Mixed_6d.branch_pool.conv','Mixed_6e.branch1x1.conv', 
'Mixed_6e.branch7x7_1.conv','Mixed_6e.branch7x7_2.conv','Mixed_6e.branch7x7_3.conv','Mixed_6e.branch7x7dbl_1.conv','Mixed_6e.branch7x7dbl_2.conv','Mixed_6e.branch7x7dbl_3.conv', 
'Mixed_6e.branch7x7dbl_4.conv','Mixed_6e.branch7x7dbl_5.conv','Mixed_6e.branch_pool.conv','Mixed_7a.branch3x3_1.conv','Mixed_7a.branch3x3_2.conv','Mixed_7a.branch7x7x3_1.conv', 
'Mixed_7a.branch7x7x3_2.conv','Mixed_7a.branch7x7x3_3.conv','Mixed_7a.branch7x7x3_4.conv','Mixed_7b.branch1x1.conv','Mixed_7b.branch3x3_1.conv','Mixed_7b.branch3x3_2a.conv', 
'Mixed_7b.branch3x3_2b.conv','Mixed_7b.branch3x3dbl_1.conv','Mixed_7b.branch3x3dbl_2.conv']
    if(model_name == 'resnet50'):
        LAYER_NAMES = ['layer1.0.conv2','layer1.0.conv3','layer1.0.downsample.0','layer1.1.conv1','layer1.1.conv2','layer1.1.conv3','layer1.2.conv1', 
        'layer1.2.conv2','layer1.2.conv3','layer2.0.conv1','layer2.0.con2','layer2.0.conv3','layer2.0.downsample.0','layer2.1.conv1', 
        'layer2.1.conv2','layer2.1.conv3','layer2.2.conv1','layer2.2.conv2','layer2.2.conv3','layer2.3.conv1', 
        'layer2.3.conv2','layer2.3.conv3','layer3.0.conv1','layer3.0.conv2','layer3.0.conv3','layer3.0.downsample.0', 
        'layer3.1.conv1','layer3.1.conv2','layer3.1.conv3','layer3.2.conv1','layer3.2.conv2','layer3.2.conv3','layer3.3.conv1', 
        'layer3.3.conv2','layer3.3.conv3','layer3.4.conv1','layer3.4.conv2','layer3.4.conv3','layer3.5.conv1','layer3.5.conv2', 
        'layer3.5.conv3','layer4.0.conv1','layer4.0.conv2','layer4.0.conv3','layer4.0.downsample.0','layer4.1.conv1','layer4.1.conv2',
        'layer4.1.conv3','layer4.2.conv1','layer4.2.conv2']
    if(model_name == 'mobilenet_v3_small'):
        LAYER_NAMES = ['features.5','features.7','features.10','features.12','features.14','features.17','features.19','features.21']
    if(model_name == 'mobilenet_v3_large'):
        LAYER_NAMES = ['features.5','features.7','features.10','features.12','features.14']
    return LAYER_NAMES
    


def util_compute_cav(model, loader_positive, loader_random, layer_name, activation, LINEAR_CLASSIFIER_TYPE, orthogonal=False, dump_cav=True):
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
    tcav_score = []
    for score in scores:
      tcav_score.append(score.mean().item())
    logging.info(f"TCAV score {tcav_score}")
    return tcav_score
    

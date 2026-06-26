# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on```python
# Copyright [2025] [Srikanth KS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""
/*
 * Copyright (c) 2025 Srikanth K S. All rights reserved.
 * Licensed under the APACHE2 License.
 * Author : Srikanth K S
 * Version 1.0
 */
"""
"""
Known bug
/mnt/data/python_venv/lib/python3.12/site-packages/torch/autograd/graph.py:825: UserWarning: adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at ../aten/src/ATen/Context.cpp:91.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass

"""

import copy
import os.path
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from logger import Logger_Singleton
from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from datetime import datetime
import argparse
from dotenv import load_dotenv
from ConfigSingleton import ConfigSingleton
import random
import numpy as np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils import (
    get_num_classes, get_class_folder_dicts, train_cav, evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence, get_model_weight_path, get_base_model_image_size, get_model_layers, predict_from_loader
)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if(DEVICE =='cuda'):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


torch.use_deterministic_algorithms(True, warn_only=True)


MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
RANDOM_STATE = 132
activation = {}
output_shape = {}
printed_layers = set()

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

def get_activation(layer_name, activation_store=None):
    def hook(model, input, output):
        if activation_store is not None:
            activation_store[layer_name] = output
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        # Only print once per layer to avoid duplicate outputs
        if layer_name not in printed_layers:
            print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
            printed_layers.add(layer_name)
        #LOGGING.info(f"Verify the output shape : Layername = {layer_name} ,Input.shape : {input[0].shape},  output.shape : {output.shape}")
    return hook


def compute_cav(model, loader_positive, loader_random, layer_name, orthogonal=False):
    pos_acts, rnd_acts = [], []
    model.eval()
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
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)


def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, target_idx):
    LOGGING.info(f"Computing TCAV score for layer: {layer_name}, target index: {target_idx}")
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, target_idx]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                LOGGING.warning(f"Layer {layer_name} is unused for target index {target_idx} in the current batch; assigning zero TCAV contribution.")
                scores.append(torch.zeros(imgs.size(0), dtype=torch.bool, device=DEVICE))
                continue
            grad = grad.detach()
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = F.normalize(grad_flat, p=2, dim=1)
            S = (grad_norm * cav_vector).sum(dim=1)
            scores.append(S > 0)  # additional LOGGING for sensitivity analysis
    scores = torch.cat(scores)
    LOGGING.info(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    print(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    return scores.float().mean().item()

def _Initialize_random_seed(random_state):
    # Set random seeds for reproducibility at the beginning of main function
    torch.manual_seed(random_state)
    if(DEVICE == 'cuda'):
      torch.cuda.manual_seed(random_state)
      torch.cuda.manual_seed_all(random_state)  # For multi-GPU setups
    np.random.seed(random_state)
    random.seed(random_state)
    
    if(DEVICE == 'cuda'):
      # Ensure deterministic behavior (may impact performance)
      torch.use_deterministic_algorithms(True, warn_only=True)

      torch.backends.cudnn.benchmark = False
    LOGGING.info("Main function started.")


def _compute_accuracy_metric(model, validation_loader, target_idx_list):
    """
    Compute accuracy and average confidence using validation loader.
    Stores results in the logger and returns them to the caller.
    
    Args:
        model: The trained model to evaluate
        validation_loader: DataLoader for validation data
        target_idx_list: List of target class indices
        
    Returns:
        tuple: (accuracy, precision, recall, f1_score, average_confidence)
    """
    try:
        # Compute accuracy metrics
        accuracy, precision, recall, f1_score = evaluate_accuracy(model, validation_loader)
        
        # Compute average confidence
        average_confidence = compute_avg_confidence(model, validation_loader, target_idx_list)
        return accuracy, precision, recall, f1_score, average_confidence
        
    except Exception as e:
        LOGGING.error(f"Error during metrics computation: {e}")
        print(f"Error during metrics computation: {e}")
        raise
    

def _process_cav(model, layer_name, concept_loader_list, random_loader, class_dataloaders, TARGET_IDX_LIST):
    LOGGING.info(f"Processing base model: {BASE_MODEL}, layer: {layer_name}")
    cav_results = {}
    hook_handles = []
    try:        
        for layer in layer_name:
            LOGGING.info(f"Computing CAV and TCAV scores for layer: {layer}")
            print(f"Computing CAV and TCAV scores for layer: {layer}")
            try:
                # Remove any previous hooks before registering new one
                for handle in hook_handles:
                    handle.remove()
                hook_handles.clear()
                
                # Register hook and store handle
                hook_handle = model.get_submodule(layer).register_forward_hook(get_activation(layer))
                hook_handles.append(hook_handle)
                
                # Compute CAV vectors
                cav_vectors = [compute_cav(model, concept_loader, random_loader, layer) 
                              for concept_loader in concept_loader_list]
                
                # Compute TCAV scores
                tcav_scores = [compute_tcav_score(model, layer, cav, class_loader, idx)
                              for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
                
                cav_results[layer] = {
                    'cav_vectors': cav_vectors,
                    'tcav_scores': tcav_scores
                }
                LOGGING.info(f"Completed CAV and TCAV computation for layer: {layer}, TCAV scores: {tcav_scores}")
                print(f"Completed CAV and TCAV computation for layer: {layer}, TCAV scores: {tcav_scores}")
            finally:
                # Remove hooks after each layer computation
                try:
                    for handle in hook_handles:
                        handle.remove()
                    hook_handles.clear()
                except Exception as e:
                    LOGGING.error(f"Error during deregistering hooks for layer {layer}: {e}")
                    print(f"Error during deregistering hooks for layer {layer}: {e}")
                        
    except Exception as e:
        LOGGING.error(f"Error during processing base model: {e}")
        print(f"Error during processing base model: {e}")
    finally:
        # Ensure all hooks are removed before returning
        for handle in hook_handles:
            try:
                handle.remove()
            except Exception as e:
                LOGGING.error(f"Error during final deregistering hooks: {e}")
                print(f"Error during final deregistering hooks: {e}")
                
    return cav_results
RECALIBRATE_TARGET_CLASS = [1,0,0]
def recalibrate_model(model, layer_names, cav_dict, validation_loader, training_loader, target_class, LAMBDA_ALIGNS, random_state=132):
    """
    Recalibrate model weights for a specific target class across multiple layers simultaneously.
    
    Args:
        model: PyTorch model to recalibrate
        layer_names: List of layer names to recalibrate
        cav_dict: Dictionary mapping layer_name -> CAV vector for the target class
        validation_loader: DataLoader for validation
        training_loader: DataLoader for training
        target_class: Target class index to recalibrate
        LAMBDA_ALIGNS: List of alignment weights to try
        random_state: Random seed
    """
    try:
        for LAMBDA_ALIGN in LAMBDA_ALIGNS:
            # Reset random seeds before each training iteration for consistency
            torch.manual_seed(random_state + hash(str(LAMBDA_ALIGN)) % 1000)
            torch.cuda.manual_seed(random_state + hash(str(LAMBDA_ALIGN)) % 1000)
            np.random.seed(random_state + hash(str(LAMBDA_ALIGN)) % 1000)
            random.seed(random_state + hash(str(LAMBDA_ALIGN)) % 1000)
            LAMBDA_CLS = round(1.0 - LAMBDA_ALIGN, 2)
            LOGGING.info(f"Recalibrating class {target_class} with Lambda Align: {LAMBDA_ALIGN}, Lambda Classification: {LAMBDA_CLS}")
            print(f"Recalibrating class {target_class} with Lambda Align: {LAMBDA_ALIGN}, Lambda Classification: {LAMBDA_CLS}")
            
            model_trained = copy.deepcopy(model).to(DEVICE)
            
            # Register forward hooks for all layers
            activation_dict = {}
            try:
                for layer_name in layer_names:
                    model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name, activation_dict))
            except Exception as e:
                LOGGING.error(f"Exception while registering forward hooks: {e}")
                print(f"Exception while registering forward hooks: {e}")
                continue
            
            try:
                model_trained.train()
            except Exception as e:
                LOGGING.error(f"Exception while calling model train: {e}")
                continue
            
            # Freeze all parameters except those in target layers
            try:
                for name, param in model_trained.named_parameters():
                    param.requires_grad = any(layer in name for layer in layer_names)
                model_trained.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None)
            except Exception as e:
                LOGGING.error(f"Exception while freezing parameters: {e}")
                continue
            
            # Initialize optimizer for trainable parameters
            torch.manual_seed(random_state + hash(str(LAMBDA_ALIGN)) % 1000)
            try:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=LEARNING_RATE)
            except Exception as e:
                LOGGING.error(f"Exception while creating optimizer: {e}")
                continue
            
            loss_history = {"total": [], "cls": [], "align": [], "correct": [], "incorrect": []}
            
            try:
                for epoch in range(EPOCHS):
                    total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
                    correct_loss_epoch = incorrect_loss_epoch = 0.0
                    batch_count = 0
                    
                    for imgs, labels in training_loader:
                        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        
                        # Forward pass
                        # Classification loss
                        if BASE_MODEL == 'inception_v3':
                            outputs, outputs_aux = model_trained(imgs)
                            cls_loss = 0.6 * nn.CrossEntropyLoss()(outputs, labels) + 0.4 * nn.CrossEntropyLoss()(outputs_aux, labels)
                        else:
                            outputs = model_trained(imgs)
                            cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                        
                        # Alignment loss across all layers
                        align_loss = 0.0
                        target_mask = (labels == target_class)
                        non_target_mask = (labels != target_class)
                        if target_mask.any():
                            # For target class: encourage alignment with CAV
                            for layer_name in layer_names:
                                if layer_name in activation_dict:
                                    f_l = activation_dict[layer_name].view(imgs.size(0), -1)
                                    cav = cav_dict[layer_name].unsqueeze(0)
                                    cosine_sim = F.cosine_similarity(f_l[target_mask], cav, dim=1)
                                    align_loss += (1 - torch.mean(cosine_sim))  # Minimize alignment penalty
                                    correct_loss_epoch += (1 - torch.mean(cosine_sim)).item()
                        
                        if non_target_mask.any():
                            pass
                            if(0):
                                # For non-target classes: penalize alignment with target CAV
                                for layer_name in layer_names:
                                    if layer_name in activation_dict:
                                        f_l = activation_dict[layer_name].view(imgs.size(0), -1)
                                        cav = cav_dict[layer_name].unsqueeze(0)
                                        cosine_sim = F.cosine_similarity(f_l[non_target_mask], cav, dim=1)
                                        align_loss += torch.mean(cosine_sim)  # Maximize to penalize false classification
                                        incorrect_loss_epoch += torch.mean(cosine_sim).item()
                        
                        # Combined loss
                        loss = LAMBDA_ALIGN * align_loss + LAMBDA_CLS * cls_loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_trained.parameters(), max_norm=7)
                        optimizer.step()
                        
                        total_loss_epoch += loss.item()
                        cls_loss_epoch += cls_loss.item()
                        align_loss_epoch += align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss
                        batch_count += 1
                    
                    # Record average loss per epoch
                    if batch_count > 0:
                        loss_history["total"].append(total_loss_epoch / batch_count)
                        loss_history["cls"].append(cls_loss_epoch / batch_count)
                        loss_history["align"].append(align_loss_epoch / batch_count)
                        loss_history["correct"].append(correct_loss_epoch / batch_count)
                        loss_history["incorrect"].append(incorrect_loss_epoch / batch_count)
                        
                        LOGGING.info(f"Epoch {epoch + 1}/{EPOCHS} - Total Loss: {loss_history['total'][-1]:.4f}, "
                                   f"Align Loss: {loss_history['align'][-1]:.4f}")
                        print(f"Epoch {epoch + 1}/{EPOCHS} - Total Loss: {loss_history['total'][-1]:.4f}, "
                              f"Align Loss: {loss_history['align'][-1]:.4f}")
                
                # Validation metrics
                acc_after, precision_after, recall_after, f1_after = evaluate_accuracy(model_trained, validation_loader)
                
                LOGGING.info(f"Accuracy After: {acc_after:.4f}, Precision: {precision_after:.4f}, "
                           f"Recall: {recall_after:.4f}, F1: {f1_after:.4f}")
                print(f"Accuracy After: {acc_after:.4f}, Precision: {precision_after:.4f}, "
                      f"Recall: {recall_after:.4f}, F1: {f1_after:.4f}")
                
                # Save results
                stats = {
                    "Target Class": target_class,
                    "Layers": str(layer_names),
                    "Lambda Alignment": LAMBDA_ALIGN,
                    "Lambda Classification": LAMBDA_CLS,
                    "Accuracy": round(acc_after, 3),
                    "Precision": round(precision_after, 3),
                    "Recall": round(recall_after, 3),
                    "F1 Score": round(f1_after, 3),
                }
                
                # Save loss plots
                layers_str = "_".join(layer_names)
                classificationloss_filename = os.path.join(RESULTS_PATH, f"loss_{BASE_MODEL}_class{target_class}_{layers_str}_{LAMBDA_ALIGN}.pdf")
                alignmentloss_filename = os.path.join(RESULTS_PATH, f"alignment_loss_{BASE_MODEL}_class{target_class}_{layers_str}_{LAMBDA_ALIGN}.pdf")
                total_loss_file = os.path.join(RESULTS_PATH, f"total_loss_{BASE_MODEL}_class{target_class}_{layers_str}_{LAMBDA_ALIGN}.pdf")
                
                plot_loss_figure(loss_history["total"], loss_history["align"], loss_history["cls"], EPOCHS,
                               classificationloss_filename, alignmentloss_filename, total_loss_file)
                
                # Save statistics
                statistic_filename = os.path.join(RESULTS_PATH, f"statistics_{BASE_MODEL}_class{target_class}.csv")
                save_statistics(stats, statistic_filename)
                
                # Save model
                modelsave_filename = os.path.join(RESULTS_PATH, f"model_{BASE_MODEL}_class{target_class}_{layers_str}_{LAMBDA_ALIGN}.pth")
                torch.save(model_trained.state_dict(), modelsave_filename)
                
                LOGGING.info(f"Training completed for class {target_class}, Lambda Align: {LAMBDA_ALIGN}")
                
            except Exception as e:
                LOGGING.error(f"Error during training with Lambda Align {LAMBDA_ALIGN}: {e}")
                print(f"Error during training with Lambda Align {LAMBDA_ALIGN}: {e}")
    
    except Exception as e:
        LOGGING.error(f"Error during model recalibration: {e}")
        print(f"Error during model recalibration: {e}")


def main(random_state=132):
    #Initialize random seed for reproducibility
    _Initialize_random_seed(random_state)
    try:
        LOGGING.info(f"Computing the CAV vectors and TCAV scores for the base model {BASE_MODEL} before training")
        cav_results_before = _process_cav(MODEL, LAYER_NAMES, concept_loader_list, random_loader, class_dataloaders,  TARGET_IDX_LIST)
        # Print and log TCAV scores for each layer
        for layer in LAYER_NAMES:
            if layer in cav_results_before:
                tcav_before = cav_results_before[layer]['tcav_scores']
                print(f"TCAV Score before training for layer {layer}: {tcav_before}")
                LOGGING.info(f"TCAV Score before training for layer {layer}: {tcav_before}")
        #Compute and log accuracy and average confidence for the base model before training
        acc_before, precision_before, recall_before, f1_before, avg_conf_before = _compute_accuracy_metric(MODEL, validation_loader, TARGET_IDX_LIST)
        LOGGING.info(f"Accuracy Before: {acc_before:.4f}")
        LOGGING.info(f"Precision Before: {precision_before:.4f}")
        LOGGING.info(f"Recall Before: {recall_before:.4f}")
        LOGGING.info(f"F1 Score Before: {f1_before:.4f}")
        LOGGING.info(f"Average Confidence Before: {avg_conf_before}")
        print(f"Accuracy Before: {acc_before:.4f}")
        print(f"Precision Before: {precision_before:.4f}")
        print(f"Recall Before: {recall_before:.4f}")
        print(f"F1 Score Before: {f1_before:.4f}")
        print(f"Average Confidence Before: {avg_conf_before}")
        
    except Exception as e:
        LOGGING.error(f"Error during initial CAV and TCAV computation: {e}")
        print(f"Error during initial CAV and TCAV computation: {e}")
    # Run recalibration only for classes flagged with 1 in RECALIBRATE_TARGET_CLASS
    
    for class_idx, target_class in enumerate(TARGET_IDX_LIST):
        #In case the target index is out of bounds, default to 0 (no recalibration) exaple
        # if the target_indx = [0..10] and the RECALIBRATE_TARGET_CLASS = [1,0,0] then the target_class = 3 onwards will be out of bounds and will default to 0
        #easier version is flag = RECALIBRATE_TARGET_CLASS[class_idx] 
        flag = RECALIBRATE_TARGET_CLASS[class_idx] if 0 <= class_idx < len(RECALIBRATE_TARGET_CLASS) else 0
        if flag != 1:
            LOGGING.info(f"Skipping recalibration for class index {class_idx} (flag={flag})")
            print(f"Skipping recalibration for class index {class_idx} (flag={flag})")
            continue
        target_class = TARGET_IDX_LIST[class_idx]
        cav_dict = {
            layer: cav_results_before[layer]['cav_vectors'][class_idx]
            for layer in LAYER_NAMES
            if layer in cav_results_before
        }
        LOGGING.info(f"Starting recalibration for class index {class_idx} (target_class={target_class})")
        print(f"Starting recalibration for class index {class_idx} (target_class={target_class})")
        recalibrate_model(MODEL, LAYER_NAMES, cav_dict, validation_loader, dataset_loader, target_class, LAMBDA_ALIGNS)

if __name__ == "__main__":
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Override model name and model path")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--model_path", type=str, default=None, help="Specify a model path to override the default path")
    parser.add_argument("--config_file", type=str, default=None, help="Specify a config file to override the default path")
    parser.add_argument("--store_results", type=str, default=None, help="Specify a locaiton to store the results")
    
    args = parser.parse_args()
    config_file = args.config_file
    print(config_file)
    if config_file is not None:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    else:
        raise FileNotFoundError(f"Config file parameter not provided in the command line")
    config = ConfigSingleton(config_file)
    SEED = config.SEED
    np.random.seed(SEED)
    torch.manual_seed(SEED)
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
    if(os.getenv('DEBUG')):
        #args = parser.parse_args(["--org_model_path" , "/home/srikanth/trained_models/pytorch/vgg16/vgg16.pth","--model_name", "vgg16"])
        args = parser.parse_args(["--model_path" , "/home/srikanth/trained_models/pytorch","--model_name", "mobilenet_v3_small"])
    # Check if both parameters are provided
    if not args.model_name or not args.model_path:
        print("Error: Both --model_name and --model_path must be provided.")
    else:
        BASE_MODEL = args.model_name.strip().lower()
        BASE_MODEL_PATH = args.model_path.strip()
    # Override the model name if provided
    if args.model_name:
        BASE_MODEL = args.model_name.strip().lower()
        BASE_MODEL = BASE_MODEL.strip().lower()
        MODEL_PATH = get_model_weight_path(BASE_MODEL, BASE_MODEL_PATH)
        # Configure LOGGING
        RESULTS_BASE_PATH = args.store_results
        if(RESULTS_BASE_PATH == None):
            RESULTS_BASE_PATH = './results' 
        RESULTS_PATH = RESULTS_BASE_PATH +'/' + BASE_MODEL + '/'
        os.makedirs(RESULTS_PATH, exist_ok=True)
        log_filename = f"{RESULTS_BASE_PATH}/{BASE_MODEL}/audit_trail_{BASE_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        LOGGING = Logger_Singleton(log_filename)
        LOGGING.info("Script started.")
        IMAGE_SIZE = get_base_model_image_size(BASE_MODEL)
        # load model
        print(MODEL_PATH, DEVICE)
        if(DEVICE == 'cpu'):
            MODEL = torch.load(MODEL_PATH, map_location=DEVICE, weights_only = False)
        else:
            MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
        MODEL.to(DEVICE)        
        # Get all bottleneck layers
        LAYER_NAMES = get_model_layers(MODEL)
        LOGGING.info(f"Layer names present in this model are {LAYER_NAMES}")
        LAYER_NAMES = get_model_layers(MODEL)[2:]
        LOGGING.info(f"Layer names trained now in this model are {LAYER_NAMES}")
        print(f"Layer names trained now in this model are {LAYER_NAMES}")
        if(config.OVERRIDE_RECALIB):
            #
            if(BASE_MODEL == 'vgg16'):
                #Get the layers
                LAYER_NAMES = config.VGG_RECALIB
            if(BASE_MODEL == 'resnet50'):
                LAYER_NAMES = config.RESNET50_RECALIB
            if(BASE_MODEL == 'inception_v3'):
                LAYER_NAMES = config.INCEPTION_V3_RECALIB
            if(BASE_MODEL == 'mobilenet_v3_small'):
                LAYER_NAMES = config.MOBILENET_V3_SMALL_RECALIB
            if(BASE_MODEL == 'mobilenet_v3_large'):
                LAYER_NAMES = config.MOBILENET_V3_LARGE_RECALIB
            LOGGING.info(f"Layer names Override the following layers {LAYER_NAMES} were considered in model {BASE_MODEL}")
        NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
        
    # Transformations
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    VALID_TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    LOGGING.info(f"Model overridden with: {args.model_name}")
    LOGGING.info(f"Model path: {BASE_MODEL_PATH}")
    LOGGING.info(f"Hyperparameters - Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Device: {DEVICE}")
    LOGGING.info(f"Target Classes: {TARGET_CLASS_LIST}, Lambda Aligns: {LAMBDA_ALIGNS}")
    try:
        print("Calling methods get_class_folder_dicts")
        print("Classification base path ", CLASSIFICATION_DATA_BASE_PATH)
        train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
        print("Train folders , valid folders and class name ", train_folders, valid_folders, class_names)
        TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]
        print("Loading train datasets stand by")
        train_dataset = MultiClassImageDataset(train_folders, transform=TRAIN_TRANSFORM)
        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE)    
        dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
	                            shuffle=True, generator=generator,
	                            worker_init_fn=worker_init_fn)
        print("Loading val datasets stand by")
        val_dataset = MultiClassImageDataset(valid_folders, transform=VALID_TRANSFORM)
        validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH, "train/" + class_name  ),
                                                              transform=VALID_TRANSFORM), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
        print("Loading concept datasets stand by")
        concept_loader_list = [DataLoader(SingleClassDataLoader(path, transform=VALID_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True) for path in CONCEPT_FOLDER_LIST]
        print("Loading random datasets stand by")
        random_loader = DataLoader(SingleClassDataLoader(RANDOM_FOLDER, transform=VALID_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True)
        LOGGING.info("Data preparation completed successfully.")
    except Exception as e:
        LOGGING.error(f"Error during data preparation: {e}")
    main()
    LOGGING.info("Script execution finished.")



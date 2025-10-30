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
import os
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import csv
from sklearn.linear_model import SGDClassifier, LogisticRegression
import torch.nn as nn
from logger import Logger_Singleton
from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from torchvision.models import resnet50, vgg16,inception_v3, mobilenet_v3_large, mobilenet_v3_small
from torch.utils.data import DataLoader
from torchvision import transforms
import random


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 132


# Dynamically determine the number of classes
def get_num_classes(base_path):
    return len([
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ])


# Get base model image size
def get_base_model_image_size(base_model_path):
    if 'inception' in base_model_path:
        image_size = 299
    else:
        image_size = 224

    return image_size


def get_model_weight_path(base_model, model_root_path=r'./model_weights'):
    base_model_path = os.path.join(model_root_path, base_model + "/" + base_model + ".pth")
    print(model_root_path, base_model_path)
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model weights not found at {base_model_path}")
    return base_model_path


def get_model_layers(model):
    layer_types = (nn.Conv2d, nn.MaxPool2d)
    layers = []
    param_names = [name for name, _ in model.named_parameters()]
    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            if any(pname.startswith(name) for pname in param_names):
                layers.append(name)
            else:
                layers.append(name)
    return layers


# Auto parse class folders
def get_class_folder_dicts(base_dir):
    print(base_dir)

    train_base_path = os.path.join(base_dir, 'train')
    valid_base_path = os.path.join(base_dir, 'valid')

    if not os.path.isdir(train_base_path):
        raise FileNotFoundError(f"Train directory not found at: {train_base_path}")
    if not os.path.isdir(valid_base_path):
        raise FileNotFoundError(f"Validation directory not found at: {valid_base_path}")

    # Get a list of class names from the train directory
    # Assumes both train and valid folders have the same set of classes
    classes = sorted([d for d in os.listdir(train_base_path) if os.path.isdir(os.path.join(train_base_path, d))])

    train_class_folders = {}
    valid_class_folders = {}

    for idx, class_name in enumerate(classes):
        # Build the full path for each class folder
        train_class_folder = os.path.join(train_base_path, class_name)
        valid_class_folder = os.path.join(valid_base_path, class_name)

        # Store the path-to-index mapping in the dictionaries
        if os.path.exists(train_class_folder):
            train_class_folders[train_class_folder] = idx
        if os.path.exists(valid_class_folder):
            valid_class_folders[valid_class_folder] = idx
    print(train_class_folders, valid_class_folders, classes)
    return train_class_folders, valid_class_folders, classes
    

def get_orthogonal_vector(cav_vector):
    random_vector = np.random.randn(*cav_vector.shape)
    projection = np.dot(random_vector, cav_vector) * cav_vector
    orthogonal_vector = random_vector - projection
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    # dot_product = np.dot(orthogonal_vector, cav_vector)

    return orthogonal_vector


def train_cav(concept_activations, random_activations, orthogonal=False, classifier_type='LinearSVC', random_state=RANDOM_STATE):
    np.random.seed(random_state)
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    if classifier_type == 'LinearSVC':
        clf = LinearSVC(max_iter=1500, random_state = random_state)
    elif classifier_type == 'SGDClassifier':
        clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # hinge = SVM-like
    elif classifier_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state = random_state)

    clf.fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector

    if orthogonal:
        cav_vector = get_orthogonal_vector(cav_vector)

    return cav_vector


def evaluate_accuracy(model, loader):
    """
    Evaluates the model on a given DataLoader `loader` and returns:
    - Overall accuracy, precision, recall, F1 score

    If the evaluation fails due to empty predictions or labels, returns 0s and prints diagnostics.
    """
    import os
    os.makedirs("./results", exist_ok=True)

    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_labels = []
    batch_count = 0
    with torch.no_grad():
        batch_count = batch_count + 1
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            # Debug model output
            print(f"[Debug] Model output stats — min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")

            if outputs.shape[0] == 0:
                print("[Warning] Model returned empty output.")
                logging.info(f"Processing image batch: {batch_count}, the labels were {labels}")
                continue
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

    if len(all_preds) == 0 or len(all_labels) == 0:
        print("[Error] No predictions or labels collected. Returning default scores.")
        return 0.0, 0.0, 0.0, 0.0

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Optional: Class-wise statistics
    class_names = loader.dataset.class_names if hasattr(loader.dataset, "class_names") else [str(i) for i in sorted(set(all_labels))]

    class_correct_counts = {name: 0 for name in class_names}
    class_total_counts = {name: 0 for name in class_names}

    for true, pred in zip(all_labels, all_preds):
        try:
            class_name = class_names[true]
            class_total_counts[class_name] += 1
            if true == pred:
                class_correct_counts[class_name] += 1
        except IndexError:
            print(f"[Error] Label index {true} out of class_names range.")

    for class_name in class_names:
        correct = class_correct_counts[class_name]
        total = class_total_counts[class_name]
        print(f"Class: {class_name} — Correct: {correct} / {total}")

    return acc, precision, recall, f1


def compute_avg_confidence(model, loader, target_idx_list):
    """
    Computes the average confidence score for predictions that match each target class index in `target_idx_list`.

    Returns a list of average confidence values (in order of target_idx_list).
    """
    model.eval()
    class_confidences = {idx: [] for idx in target_idx_list}

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            if outputs.ndim != 2:
                raise ValueError(f"[Error] Unexpected output shape: {outputs.shape}")
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            num_classes = probs.shape[1]
            for idx in target_idx_list:
                if idx >= num_classes:
                    print(f"[Warning] Target index {idx} exceeds number of model output classes ({num_classes}). Skipping.")
                    continue
                confs = probs[preds == idx, idx]
                class_confidences[idx].extend(confs.tolist())

    avg_confidences = {}
    for idx in target_idx_list:
        confs = class_confidences.get(idx, [])
        if len(confs) == 0:
            print(f"[Warning] No predictions made for class {idx}. Confidence will be 0.")
            avg_confidences[idx] = 0.0
        else:
            avg_confidences[idx] = float(np.mean(confs))

    return [avg_confidences.get(idx, 0.0) for idx in target_idx_list]



def predict_from_loader(val_loader, model, class_names):
    model.eval()
    results = []
    all_labels = []
    all_preds = []
    class_confidences = {cls: [] for cls in class_names}
    class_counts = {cls: 0 for cls in class_names}
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            confidences, preds = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
            for i in range(images.size(0)):
                true_class = class_names[labels[i].item()]
                pred_class = class_names[preds[i].item()]
                all_preds.append(pred_class)
                all_labels.append(true_class)
                class_confidences[pred_class].append(confidences[i].item())
                class_counts[true_class] += 1
                results.append({
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "Prediction Confidence": confidences[i].item()
                })
    acc = accuracy_score(all_labels, all_preds)
    avg_confidences = {cls: (sum(vals)/len(vals) if vals else 0.0) for cls, vals in class_confidences.items()}
    print(f"DEBUG acc = {acc} , avg_confidences = {avg_confidences}, class_counts: {class_counts}")
    return results, avg_confidences, class_counts, acc


def plot_loss_figure(total_loss_history, align_loss_history, cls_loss_history, epochs,
                     classification_loss="./results/classification_loss.pdf",
                     alignment_loss="./results/alignment_loss.pdf", total_loss="./results/total_loss.pdf"):
    # plot total loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), total_loss_history, marker='o', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Total Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    total_loss_plot_path = total_loss
    plt.savefig(total_loss_plot_path)
    plt.close()

    # plot classification loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), cls_loss_history, marker='x', label='Classification Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Classification Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    cls_loss_plot_path = classification_loss
    plt.savefig(cls_loss_plot_path)
    plt.close()

    # plot alignment loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), align_loss_history, marker='s', label='Alignment Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Alignment Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    align_loss_plot_path = alignment_loss
    plt.savefig(align_loss_plot_path)
    plt.close()


def save_statistics(stat, filename="results/statistics.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stat.keys())

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()
        writer.writerow(stat)
        
        
def load_model(model_name, model_path):
    logger = Logger_Singleton()
    logger.info(f"Model name: {model_name}, Model path {model_path}")
    """
    Load the model state dictionary from the specified path.
    """
    print(model_path)
    model = torch.load(model_path)
    model.eval()
    return model

def load_model_statedict(model, model_path):
    logger = Logger_Singleton()
    logger.info(f"Model Symmary: {model}, Model path {model_path}")
    """
    Load the model state dictionary from the specified path.
    """
    print(model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

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
def load_train_valid_dataset(MODEL_NAME, CLASSIFICATION_DATA_BASE_PATH,BATCH_SIZE, random_state=RANDOM_STATE):
    generator = torch.Generator()
    generator.manual_seed(random_state)    
    # Set global seed
    set_seed(random_state)
    logger = Logger_Singleton()
    # Transformations
    IMAGE_SIZE = get_base_model_image_size(MODEL_NAME)
    TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    VALID_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
    logger.info(f"Training folders: {train_folders} and the validation folders: {valid_folders}")
    train_dataset = MultiClassImageDataset(train_folders, transform=TRAIN_TRANSFORM)
    dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                 shuffle=True,
                                 generator=generator,
                                 worker_init_fn=worker_init_fn)
    val_dataset = MultiClassImageDataset(valid_folders, transform=VALID_TRANSFORM)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return  dataset_loader, val_loader,TRAIN_TRANSFORM,VALID_TRANSFORM, class_names

def load_train_dataset_concept_random(MODEL_NAME, concept_folder_list, random_folder_list, BATCH_SIZE, random_state = RANDOM_STATE):
    generator = torch.Generator()
    generator.manual_seed(random_state)    
    # Set global seed
    set_seed(random_state)
    logger = Logger_Singleton()
    IMAGE_SIZE = get_base_model_image_size(MODEL_NAME)
    TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    VALID_TRANSFORM = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    logger.info(f"Loading concept datasets stand by from the folders {concept_folder_list}")
    print("Loading concept datasets stand by")
    concept_loader = [DataLoader(SingleClassDataLoader(path, transform=TRAIN_TRANSFORM ), 
                                 batch_size = BATCH_SIZE, shuffle = True,
                                 generator = generator,worker_init_fn = worker_init_fn
                                 ) for path in concept_folder_list]
    logger.info(f"Loading random datasets stand by from the folder {random_folder_list}")
    print("Loading random datasets stand by")
    random_loader = DataLoader(SingleClassDataLoader(random_folder_list, transform=TRAIN_TRANSFORM), 
                               batch_size=BATCH_SIZE, shuffle=True,
                               generator = generator,worker_init_fn = worker_init_fn
                               ) 
    return concept_loader, random_loader

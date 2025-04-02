import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from torchvision import transforms

# Configuration
LEARNING_RATE = 1e-1
EPOCHS = 10
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
DATASET_FOLDER = "./data/zebra_fake/train/zebra"
VALIDATION_FOLDER = "./data/zebra_fake/valid/zebra"
RESULTS_PATH = "./results/retrained_model.pth"
K = 340  # ImageNet class index for zebra

# Weight balancing
LAMBDA_ALIGN = 0.9
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Custom Dataset Class
class ConceptDataset(Dataset):
    """A dataset class to load images from a specified folder."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create datasets and loaders
concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
train_dataset = ConceptDataset(DATASET_FOLDER, transform=transform)
val_dataset = ConceptDataset(VALIDATION_FOLDER, transform=transform)


concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate_accuracy_in_eval_mode(model, data_loader, zebra_idx):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs in data_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == zebra_idx).sum().item()
            total += imgs.size(0)
    return (correct / total) * 100 if total > 0 else 0

zebra_idx = K
acc_before = evaluate_accuracy_in_eval_mode(model, validation_loader, zebra_idx)
print(f"Accuracy on eval mode: {acc_before:.2f}%")

def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()

def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()

def evaluate_accuracy_in_train_mode(model, data_loader, zebra_idx):
    model.train()
    model.apply(set_batchnorm_eval)
    model.apply(set_dropout_eval)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs in data_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == zebra_idx).sum().item()
            total += imgs.size(0)
    return (correct / total) * 100 if total > 0 else 0

zebra_idx = K
acc_before = evaluate_accuracy_in_train_mode(model, validation_loader, zebra_idx)
print(f"Accuracy on train mode: {acc_before:.2f}%")
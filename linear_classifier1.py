import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
BATCH_SIZE = 64

# Dataset Class
class ConceptDataset(Dataset):
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

# Model Preparation
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
model.eval()

# Activation hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Datasets
stripes_fake_dataset = ConceptDataset("./data/concept/stripes_fake", transform=transform)
stripes_dataset = ConceptDataset("./data/concept/striped", transform=transform)
random_dataset = ConceptDataset("./data/concept/random", transform=transform)

# Dataloaders
stripes_fake_loader = DataLoader(stripes_fake_dataset, batch_size=BATCH_SIZE, shuffle=True)
stripes_loader = DataLoader(stripes_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Extract Features
def extract_features(loader, label, model, layer_name):
    features, labels = [], []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            acts = activation[layer_name].view(imgs.size(0), -1).cpu().numpy()
            features.append(acts)
            labels.extend([label] * imgs.size(0))
    return np.vstack(features), np.array(labels)

# Data for Classifiers
X_stripes_fake, y_stripes_fake = extract_features(stripes_fake_loader, 1, model, LAYER_NAME)
X_stripes, y_stripes = extract_features(stripes_loader, 1, model, LAYER_NAME)
X_random, y_random = extract_features(random_loader, 0, model, LAYER_NAME)

# Dataset 1: stripes_fake vs random
X_fake_vs_random = np.vstack((X_stripes_fake, X_random))
y_fake_vs_random = np.hstack((y_stripes_fake, y_random))

# Dataset 2: stripes vs random
X_stripes_vs_random = np.vstack((X_stripes, X_random))
y_stripes_vs_random = np.hstack((y_stripes, y_random))

# Train Linear Classifiers
clf_fake_vs_random = LinearSVC()
clf_fake_vs_random.fit(X_fake_vs_random, y_fake_vs_random)

clf_stripes_vs_random = LinearSVC()
clf_stripes_vs_random.fit(X_stripes_vs_random, y_stripes_vs_random)

# Predictions
y_pred_fake_vs_random = clf_fake_vs_random.predict(X_fake_vs_random)
y_pred_stripes_vs_random = clf_stripes_vs_random.predict(X_stripes_vs_random)

# Evaluation Metrics
def evaluate(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"**{title}**")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    return acc, precision, recall, f1

# Evaluate both classifiers
metrics_fake_vs_random = evaluate(y_fake_vs_random, y_pred_fake_vs_random, "Stripes_Fake vs Random")
metrics_stripes_vs_random = evaluate(y_stripes_vs_random, y_pred_stripes_vs_random, "Stripes vs Random")

# Plot Comparison
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values_fake_vs_random = metrics_fake_vs_random
values_stripes_vs_random = metrics_stripes_vs_random

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values_fake_vs_random, width, label='Stripes_Fake vs Random')
plt.bar(x + width/2, values_stripes_vs_random, width, label='Stripes vs Random')

plt.xticks(x, labels)
plt.ylabel('Score')
plt.title('Performance Comparison of Linear Classifiers')
plt.legend()
plt.grid(True)
plt.show()

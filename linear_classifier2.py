import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from PIL import Image
import matplotlib.pyplot as plt

from config import (LAYER_NAME,
                    IMAGE_SIZE, 
                    BATCH_SIZE, 
                    DEVICE, 
                    MODEL,
                    CONCEPT_FOLDER, 
                    RANDOM_FOLDER,
                    CONCEPT_FOLDER_FAKE)

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, folder_path, label, transform=None):
        self.folder_path = folder_path
        self.label = label
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.view(-1).numpy(), self.label  # Flatten image to 1D vector

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load Datasets
stripes_fake_dataset = ImageDataset(CONCEPT_FOLDER_FAKE, label=1, transform=transform)
stripes_dataset = ImageDataset(CONCEPT_FOLDER, label=1, transform=transform)
random_dataset = ImageDataset(RANDOM_FOLDER, label=0, transform=transform)

# Combine Datasets
def prepare_data(stripes_dataset, random_dataset):
    X, y = [], []
    for img, label in stripes_dataset:
        X.append(img)
        y.append(label)
    for img, label in random_dataset:
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

# Data for Classifiers
X_fake_vs_random, y_fake_vs_random = prepare_data(stripes_fake_dataset, random_dataset)
X_stripes_vs_random, y_stripes_vs_random = prepare_data(stripes_dataset, random_dataset)

# Train Classifiers
clf_fake_vs_random = LinearSVC(max_iter=10000)
clf_fake_vs_random.fit(X_fake_vs_random, y_fake_vs_random)

clf_stripes_vs_random = LinearSVC(max_iter=10000)
clf_stripes_vs_random.fit(X_stripes_vs_random, y_stripes_vs_random)

# Predictions
y_pred_fake_vs_random = clf_fake_vs_random.predict(X_fake_vs_random)
y_pred_stripes_vs_random = clf_stripes_vs_random.predict(X_stripes_vs_random)

# Evaluation
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
plt.title('Performance Comparison of Linear Classifiers on Original Images')
plt.legend()
plt.grid(True)
plt.show()

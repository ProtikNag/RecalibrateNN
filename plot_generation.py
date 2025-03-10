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
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
CONCEPT_FOLDERS = {
    "dotted": "./data/concept/dotted",
    "striped": "./data/concept/striped",
    "zigzagged": "./data/concept/zigzagged"
}
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
RESULTS_PATH = "./results/retrained_model.pth"
K = 340  # ImageNet class index for zebra

# Weight balancing
LAMBDA_ALIGN = 1.0
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# Function to train CAV
def train_cav(concept_activations, random_activations):
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    clf = LinearSVC().fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)
    return cav_vector

# Load Pre-trained GoogLeNet
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
model.train()

# Freeze all layers except the target layer
for name, param in model.named_parameters():
    param.requires_grad = (LAYER_NAME in name)

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Register hook to capture activations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Function to collect activations
def collect_activations(loader):
    activations = []
    model.eval()
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            acts = activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy()
            activations.append(acts)
    return np.vstack(activations)

# Compute TCAV score
def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, k):
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, k]
            grad = torch.autograd.grad(h_k.sum(), f_l)[0]
            grad_flat = grad.view(grad.size(0), -1)
            S = (grad_flat * cav_vector).sum(dim=1)
            scores.append(S > 0)
    scores = torch.cat(scores)
    return scores.float().mean().item()

# Compute TCAV scores for multiple concepts
dataset = ConceptDataset(DATASET_FOLDER, transform=transform)
dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)

tcav_scores_before = {}
cav_vectors = {}

for concept, folder in CONCEPT_FOLDERS.items():
    concept_dataset = ConceptDataset(folder, transform=transform)
    concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Collect activations
    concept_activations = collect_activations(concept_loader)
    random_activations = collect_activations(random_loader)

    # Compute CAV
    cav_vector = train_cav(concept_activations, random_activations)
    cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)
    cav_vectors[concept] = cav_vector

    # Compute TCAV score before recalibration
    tcav_score = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, K)
    tcav_scores_before[concept] = tcav_score

# Model Recalibration
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        labels = torch.full((imgs.size(0),), K, device=DEVICE, dtype=torch.long)

        outputs = model(imgs)
        f_l = activation[LAYER_NAME]

        classification_loss = nn.CrossEntropyLoss()(outputs, labels)

        h_k = outputs[:, K]
        grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0]
        grad_flat = grad.view(grad.size(0), -1)

        # Compute alignment loss for multiple CAVs
        alignment_losses = []
        for cav_vector in cav_vectors.values():
            alignment_losses.append(-torch.mean(torch.relu(grad_flat @ cav_vector)))

        alignment_loss = sum(alignment_losses) / len(alignment_losses)

        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

# Compute TCAV scores after recalibration
tcav_scores_after = {}
for concept, cav_vector in cav_vectors.items():
    tcav_score = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, K)
    tcav_scores_after[concept] = tcav_score

# Plot TCAV Scores
fig, ax = plt.subplots(figsize=(8, 6))
concepts = list(CONCEPT_FOLDERS.keys())
before_scores = [tcav_scores_before[c] for c in concepts]
after_scores = [tcav_scores_after[c] for c in concepts]

x = np.arange(len(concepts))
width = 0.35

ax.bar(x - width/2, before_scores, width, label='Before Recalibration', color='royalblue')
ax.bar(x + width/2, after_scores, width, label='After Recalibration', color='firebrick')

ax.set_xlabel('Concepts')
ax.set_ylabel('TCAV Score')
ax.set_title(f'TCAV Scores for {LAYER_NAME} (Zebra Class)')
ax.set_xticks(x)
ax.set_xticklabels(concepts)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

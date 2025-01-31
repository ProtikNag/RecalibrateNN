# retrain_nn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.calculate_cav import train_cav, get_layer_activations
from torch.utils.data import DataLoader
from src.concept_dataset import ConceptDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Configuration
LEARNING_RATE = 0.1
EPOCHS = 20
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
CONCEPT_FOLDER = "./data/concept/striped"
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
MODEL_NAME = "googlenet"
LAYER_NAME = "inception4a"
RESULTS_PATH = "./results/retrained_model.pth"

# Load pre-trained model
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
model.train()

# Freeze all layers except the target layer
for name, param in model.named_parameters():
    if LAYER_NAME not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
dataset = ConceptDataset(DATASET_FOLDER, transform=transform)

concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Collect activations for concept and random examples
concept_activations = []
random_activations = []

model.eval()
with torch.no_grad():
    for imgs in concept_loader:
        imgs = imgs.to(DEVICE)
        activations = get_layer_activations(model, LAYER_NAME, imgs)
        concept_activations.append(activations.cpu().numpy())

    for imgs in random_loader:
        imgs = imgs.to(DEVICE)
        activations = get_layer_activations(model, LAYER_NAME, imgs)
        random_activations.append(activations.cpu().numpy())

concept_activations = np.vstack(concept_activations)
random_activations = np.vstack(random_activations)

# Train CAV
cav_vector = train_cav(concept_activations, random_activations)
cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

# Define optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Training loop with latent space alignment loss
loss_history = []

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)

        # Forward pass
        outputs = model(imgs)

        # Compute activations and cosine similarity with CAV
        activations = get_layer_activations(model, LAYER_NAME, imgs)
        activations = activations.view(imgs.size(0), -1)
        cav_norm = cav_vector / cav_vector.norm()  # Normalize CAV
        activations_norm = activations / activations.norm(dim=1, keepdim=True)
        cosine_similarity = torch.sum(activations_norm * cav_norm, dim=1)

        # Latent space alignment loss (maximize alignment)
        alignment_loss = -torch.mean(cosine_similarity)

        # Total loss (use only alignment loss)
        loss = alignment_loss
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save the recalibrated model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")

# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(EPOCHS), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.show()
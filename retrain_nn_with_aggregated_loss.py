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
EPOCHS = 30
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/striped"
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
RESULTS_PATH = "./results/retrained_model.pth"
VALIDATION_FOLDER = "./data/dataset/valid_zebra"
K = 340  # ImageNet class index for zebra

# Weight balancing
LAMBDA_ALIGN = 0.5
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


def train_cav(concept_activations, random_activations):
    """
    Train a logistic regression model to separate concept and random activations.
    Returns a normalized CAV vector.
    """
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))


    clf = LinearSVC().fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector

    return cav_vector

# 3. Load Pre-trained GoogLeNet
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

# Create datasets and loaders
concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
dataset = ConceptDataset(DATASET_FOLDER, transform=transform)

concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Register hook to capture activations
activation = {}
def get_activation(name):
    """Hook function to capture layer activations."""
    def hook(model, input, output):
        activation[name] = output
    return hook

model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Collect activations for CAV computation
concept_activations, random_activations = [], []
model.eval()
with torch.no_grad():
    for imgs in concept_loader:
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        acts = activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy()
        concept_activations.append(acts)
    for imgs in random_loader:
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        acts = activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy()
        random_activations.append(acts)

concept_activations = np.vstack(concept_activations)
random_activations = np.vstack(random_activations)

# Compute CAV
cav_vector = train_cav(concept_activations, random_activations)
cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

# 4. TCAV Score Computation
def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, k):
    """Compute TCAV score using the directional derivative."""
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

# Compute original TCAV score
original_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, K)
print(f"Original TCAV Score: {original_tcav:.4f}")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            images.append(Image.open(img_path).convert('RGB'))
    return images

def evaluate_zebra_accuracy(model, images, zebra_idx):
    model.eval()
    correct = 0
    total = 0
    for img in images:
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(x)
            predicted_class = outputs.argmax(dim=1).item()
        if predicted_class == zebra_idx:
            correct += 1
        total += 1
    return (correct / total) * 100 if total > 0 else 0

zebra_images = load_images_from_folder(VALIDATION_FOLDER)
zebra_idx = K
acc_before = evaluate_zebra_accuracy(model, zebra_images, zebra_idx)
print(f"Accuracy on zebra class before recalibration: {acc_before:.2f}%")

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Training Loop
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        labels = torch.full((imgs.size(0),), K, device=DEVICE, dtype=torch.long)

        # Forward pass
        outputs = model(imgs)
        f_l = activation[LAYER_NAME]

        # Classification loss
        classification_loss = nn.CrossEntropyLoss()(outputs, labels)


        f_l_flat = f_l.view(f_l.size(0), -1)  # [batch_size, num_features]
        norm_f_l = torch.norm(f_l_flat, dim=1, keepdim=True)  # [batch_size, 1]
        cos_sim = (f_l_flat @ cav_vector) / (norm_f_l + 1e-8)  # [batch_size]
        alignment_loss = -torch.mean(torch.abs(cos_sim))

        # Total loss
        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save the retrained model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")
acc_after = evaluate_zebra_accuracy(model, zebra_images, zebra_idx)
print(f"Accuracy on zebra class after recalibration: {acc_after:.2f}%")

# Compute retrained TCAV score
retrained_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, K)
print(f"Retrained TCAV Score: {retrained_tcav:.4f}")

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(EPOCHS), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.show()
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
LEARNING_RATE = 1e-2
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
DATASET_FOLDER = "./data/zebra_fake/train/zebra"
VALIDATION_FOLDER = "./data/zebra_fake/valid/zebra"
RESULTS_PATH = "./results/retrained_model.pth"
K = 340  # ImageNet class index for zebra

# Weight balancing
LAMBDA_ALIGN = 0.8
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
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))


    clf = LinearSVC().fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector

    return cav_vector


def cosine_similarity_loss(grad, cav_vector):
    grad_norm = grad / grad.norm(dim=1, keepdim=True)
    cav_norm = cav_vector / cav_vector.norm()  # Ensure CAV is unit norm
    cosine_similarity = torch.sum(grad_norm * cav_norm, dim=1)
    return torch.mean(1 - torch.abs(cosine_similarity))


model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
model.train()

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


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

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

cav_vector = train_cav(concept_activations, random_activations)
cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, k):
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, k]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()
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

def evaluate_accuracy(model, data_loader, zebra_idx):
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
acc_before = evaluate_accuracy(model, validation_loader, zebra_idx)
print(f"Accuracy on zebra class before recalibration: {acc_before:.2f}%")

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()

def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()

# Training Loop
loss_history = []
for epoch in range(EPOCHS):
    model.train()

    # freeze model layers except the target layer
    for name, param in model.named_parameters():
        param.requires_grad = (LAYER_NAME in name)

    model.apply(set_batchnorm_eval) # freeze batchnorm layers
    model.apply(set_dropout_eval)   # freeze dropout layers

    total_loss = 0.0

    for idx, imgs in enumerate(dataset_loader):
        imgs = imgs.to(DEVICE)
        labels = torch.full((imgs.size(0),), K, device=DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(imgs)
        predicted_labels = outputs.argmax(dim=1)
        f_l = activation[LAYER_NAME]

        classification_loss = nn.CrossEntropyLoss()(outputs, labels)

        h_k = outputs[:, K]
        grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0]
        grad_flat = grad.view(grad.size(0), -1)

        alignment_loss = cosine_similarity_loss(grad_flat, cav_vector)

        print("Alignment loss: ", alignment_loss)
        print("Classification loss: ", classification_loss)

        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
        optimizer.step()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.7f}")

# Save the retrained model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")
acc_after = evaluate_accuracy(model, validation_loader, zebra_idx)
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
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =====================
# Configuration
# =====================
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/stripes_fake"
RANDOM_FOLDER = "./data/concept/random"
BINARY_CLASSIFICATION_BASE = "./data/binary_classification/"
RESULTS_PATH = "./results/retrained_model.pth"
ZEBRA_CLASS_NAME = "zebra"

LAMBDA_ALIGN = 0.75
LAMBDA_CLS = 1.0 - LAMBDA_ALIGN

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================
# Dataset Definitions
# =====================
class MultiClassImageDataset(Dataset):
    def __init__(self, class_folders, transform=None):
        self.samples = []
        self.transform = transform
        for folder_path, label in class_folders.items():
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ConceptDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_files = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# =====================
# Utilities
# =====================
def get_class_folder_dicts(base_dir):
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    train_class_folders, valid_class_folders = {}, {}
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(base_dir, class_name)
        train_folder = os.path.join(class_path, 'train')
        valid_folder = os.path.join(class_path, 'valid')
        if os.path.exists(train_folder):
            train_class_folders[train_folder] = idx
        if os.path.exists(valid_folder):
            valid_class_folders[valid_folder] = idx
    return train_class_folders, valid_class_folders, classes


def train_cav(concept_activations, random_activations):
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))
    clf = LinearSVC().fit(X, y)
    cav_vector = clf.coef_.squeeze()
    return cav_vector / np.linalg.norm(cav_vector)


def cosine_similarity_loss(grad, cav_vector):
    grad_norm = grad / grad.norm(dim=1, keepdim=True)
    cav_norm = cav_vector / cav_vector.norm()
    cosine_similarity = torch.sum(grad_norm * cav_norm, dim=1)
    return torch.mean(1 - torch.abs(cosine_similarity))


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def compute_tcav_score(model, layer_name, cav_vector, data_loader, class_idx):
    model.eval()
    scores = []
    for imgs, _ in data_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            activations = activation[layer_name]
            h_k = outputs[:, class_idx]
            grad = torch.autograd.grad(h_k.sum(), activations, retain_graph=True)[0].detach()
            grad_flat = grad.view(grad.size(0), -1)
            score = (grad_flat * cav_vector).sum(dim=1)
            scores.append(score > 0)
    return torch.cat(scores).float().mean().item()


def set_batchnorm_eval(module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()


def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()


# =====================
# Data Preparation
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_folders, valid_folders, class_names = get_class_folder_dicts(BINARY_CLASSIFICATION_BASE)
NUM_CLASSES = len(class_names)
ZEBRA_IDX = class_names.index(ZEBRA_CLASS_NAME)

train_dataset = MultiClassImageDataset(train_folders, transform=transform)
val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)

dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# Model Setup
# =====================
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(DEVICE)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES).to(DEVICE)
model.train()

activation = {}
model.get_submodule(LAYER_NAME).register_forward_hook(lambda m, i, o: activation.update({LAYER_NAME: o}))

# =====================
# CAV Computation
# =====================
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

cav_vector = train_cav(np.vstack(concept_activations), np.vstack(random_activations))
cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

original_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, ZEBRA_IDX)
print(f"Original TCAV Score (Zebra): {original_tcav:.4f}")

acc_before = evaluate_accuracy(model, validation_loader)
print(f"Accuracy before recalibration: {acc_before:.2f}%")

# =====================
# Model Fine-tuning
# =====================
model.train()
model.apply(set_batchnorm_eval)
model.apply(set_dropout_eval)

for name, param in model.named_parameters():
    param.requires_grad = (LAYER_NAME in name)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

loss_history = []
for epoch in range(EPOCHS):
    total_loss = 0.0
    for imgs, labels in dataset_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        classification_loss = nn.CrossEntropyLoss()(outputs, labels)

        f_l = activation[LAYER_NAME]
        h_k = outputs[:, ZEBRA_IDX]
        grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0]
        grad_flat = grad.view(grad.size(0), -1)

        alignment_loss = cosine_similarity_loss(grad_flat, cav_vector)
        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.7f}")

# =====================
# Save and Evaluate
# =====================
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")

acc_after = evaluate_accuracy(model, validation_loader)
print(f"Accuracy after recalibration: {acc_after:.2f}%")

retrained_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, ZEBRA_IDX)
print(f"Retrained TCAV Score (Zebra): {retrained_tcav:.4f}")

# =====================
# Visualization
# =====================
plt.figure(figsize=(8, 6))
plt.plot(range(EPOCHS), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()
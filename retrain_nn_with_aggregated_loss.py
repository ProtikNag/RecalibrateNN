import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from torchvision import transforms

# Configuration
LEARNING_RATE = 0.001
EPOCHS = 15
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER_NAME = "inception4a"
CONCEPT_FOLDER = "./data/concept/striped"
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
RESULTS_PATH = "./results/retrained_model.pth"
K = 340  # ImageNet class index for zebra
LAMBDA_ALIGN = 1.0

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

# 2. CAV Training Function
# def train_cav(concept_activations, random_activations):
#     """Train a linear SVM to compute the Concept Activation Vector (CAV)."""
#     X = np.vstack([concept_activations, random_activations])
#     y = np.hstack([np.ones(concept_activations.shape[0]), np.zeros(random_activations.shape[0])])
#     clf = LinearSVC()
#     clf.fit(X, y)
#     cav = clf.coef_[0]
#     return cav

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

# Define the remaining model after the target layer
class RemainingModel(nn.Module):
    """Submodel containing layers after inception4a."""
    def __init__(self, original_model):
        super(RemainingModel, self).__init__()
        self.inception4b = original_model.inception4b
        self.inception4c = original_model.inception4c
        self.inception4d = original_model.inception4d
        self.inception4e = original_model.inception4e
        self.maxpool4 = original_model.maxpool4
        self.inception5a = original_model.inception5a
        self.inception5b = original_model.inception5b
        self.avgpool = original_model.avgpool
        self.dropout = original_model.dropout
        self.fc = original_model.fc

    def forward(self, x):
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

remaining_model = RemainingModel(model).to(DEVICE)

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

        # # Alignment loss using directional derivative
        # h_k = outputs[:, K]
        # grad = torch.autograd.grad(h_k.sum(), f_l, create_graph=True)[0]
        # grad_flat = grad.view(grad.size(0), -1)
        # S = (grad_flat * cav_vector).sum(dim=1)
        # alignment_loss = torch.mean(torch.relu(-S))  # Encourage S > 0

        f_l_flat = f_l.view(f_l.size(0), -1)  # [batch_size, num_features]
        norm_f_l = torch.norm(f_l_flat, dim=1, keepdim=True)  # [batch_size, 1]
        cos_sim = (f_l_flat @ cav_vector) / (norm_f_l + 1e-8)  # [batch_size]
        alignment_loss = -torch.mean(torch.abs(cos_sim))

        # Total loss
        loss = classification_loss + LAMBDA_ALIGN * alignment_loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Save the retrained model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")

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
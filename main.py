import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    DEVICE,
    MODEL,
    LAYER_NAME,
    CONCEPT_FOLDER,
    RANDOM_FOLDER,
    CLASSIFICATION_DATA_BASE_PATH,
    RESULTS_PATH,
    TARGET_CLASS_NAME,
    LAMBDA_ALIGN,
    LAMBDA_CLS,
)
from custom_dataloader import ConceptDataset, MultiClassImageDataset
from utils import get_class_folder_dicts, train_cav, cosine_similarity_loss, evaluate_accuracy, plot_loss_figure

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
NUM_CLASSES = len(class_names)
TARGET_IDX = class_names.index(TARGET_CLASS_NAME)

train_dataset = MultiClassImageDataset(train_folders, transform=transform)
val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = MODEL
model.train()

# Hook activations
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Compute CAV
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
    for imgs, _ in dataset_loader:
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


original_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, validation_loader, TARGET_IDX)
print(f"Original TCAV Score (Zebra): {original_tcav:.4f}")
acc_before = evaluate_accuracy(model, validation_loader)
print(f"Accuracy before recalibration: {acc_before:.2f}%")


def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()


def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()


model.train()
for name, param in model.named_parameters():
    param.requires_grad = (LAYER_NAME in name)
model.apply(set_batchnorm_eval)
model.apply(set_dropout_eval)

params_to_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(filter(lambda p: p.requires_grad, params_to_train), lr=LEARNING_RATE)

# Recalibrate using only Zebra images
label = [train_folders[path] for path in train_folders.keys() if TARGET_CLASS_NAME in path][0] # target images label while original training
target_class_datapath = {
    os.path.join(CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_NAME, 'train'): label
}
target_class_dataset = MultiClassImageDataset(target_class_datapath, transform=transform)
target_class_loader = DataLoader(target_class_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
total_loss_history = []
cls_loss_history = []
align_loss_history = []

for epoch in range(EPOCHS):
    total_loss_epoch = 0.0
    cls_loss_epoch = 0.0
    align_loss_epoch = 0.0

    for imgs, labels in target_class_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)

        classification_loss = nn.CrossEntropyLoss()(outputs, labels)

        # alignment loss via cosine similarity
        f_l = activation[LAYER_NAME]
        h_k = outputs[:, TARGET_IDX]
        grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0]
        grad_flat = grad.view(grad.size(0), -1)
        # S = torch.matmul(grad_flat, cav_vector)
        # alignment_loss = -S.mean()
        alignment_loss = cosine_similarity_loss(grad_flat, cav_vector)

        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss

        # accumulate loss
        total_loss_epoch += loss.item()
        cls_loss_epoch += classification_loss.item()
        align_loss_epoch += alignment_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
        optimizer.step()

    n_batches = len(target_class_loader)
    total_loss_history.append(total_loss_epoch / n_batches)
    cls_loss_history.append(cls_loss_epoch / n_batches)
    align_loss_history.append(align_loss_epoch / n_batches)

    print(f"Epoch {epoch + 1}/{EPOCHS}, "
          f"Total Loss: {total_loss_history[-1]:.7f}, "
          f"Cls Loss: {cls_loss_history[-1]:.7f}, "
          f"Align Loss: {align_loss_history[-1]:.7f}")

# Save model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")

acc_after = evaluate_accuracy(model, validation_loader)
print(f"Accuracy after recalibration: {acc_after:.2f}%")

retrained_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, validation_loader, TARGET_IDX)
print(f"Retrained TCAV Score (Zebra): {retrained_tcav:.4f}")

plot_loss_figure(total_loss_history, cls_loss_history, align_loss_history, EPOCHS)

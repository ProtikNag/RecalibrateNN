import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, DEVICE, MODEL, LAYER_NAME,
    CONCEPT_FOLDER_1, CONCEPT_FOLDER_2, RANDOM_FOLDER,
    CLASSIFICATION_DATA_BASE_PATH, RESULTS_PATH,
    TARGET_CLASS_1, TARGET_CLASS_2,
    LAMBDA_ALIGN, LAMBDA_CLS,
)
from custom_dataloader import ConceptDataset, MultiClassImageDataset
from utils import (
    get_class_folder_dicts, train_cav,
    evaluate_accuracy, plot_loss_figure, save_statistics
)

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
TARGET_IDX_1 = class_names.index(TARGET_CLASS_1)
TARGET_IDX_2 = class_names.index(TARGET_CLASS_2)

train_dataset = MultiClassImageDataset(train_folders, transform=transform)
val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

concept_loader_1 = DataLoader(ConceptDataset(CONCEPT_FOLDER_1, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
concept_loader_2 = DataLoader(ConceptDataset(CONCEPT_FOLDER_2, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(ConceptDataset(RANDOM_FOLDER, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

# Model and hook setup
model = MODEL
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Compute two CAV vectors

def compute_cav(loader_positive, loader_random):
    pos_acts, rnd_acts = [], []
    model.eval()
    with torch.no_grad():
        for imgs in loader_positive:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            pos_acts.append(activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy())
        for imgs in loader_random:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            rnd_acts.append(activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy())
    pos_acts = np.vstack(pos_acts)
    rnd_acts = np.vstack(rnd_acts)
    cav = train_cav(pos_acts, rnd_acts)
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)

cav_vector_1 = compute_cav(concept_loader_1, random_loader)
cav_vector_2 = compute_cav(concept_loader_2, random_loader)

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

# Compute original TCAV scores
original_tcav_1 = compute_tcav_score(model, LAYER_NAME, cav_vector_1, validation_loader, TARGET_IDX_1)
original_tcav_2 = compute_tcav_score(model, LAYER_NAME, cav_vector_2, validation_loader, TARGET_IDX_2)
print(f"Original TCAV Score ({TARGET_CLASS_1}): {original_tcav_1:.4f}")
print(f"Original TCAV Score ({TARGET_CLASS_2}): {original_tcav_2:.4f}")
acc_before, precision_before, recall_before, f1_before = evaluate_accuracy(model, validation_loader)
print(f"Accuracy before recalibration: {acc_before:.2f}%")

# Model training prep
model.train()
for name, param in model.named_parameters():
    param.requires_grad = (LAYER_NAME in name)
model.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# Training loop
loss_history = {"total": [], "cls": [], "align": []}

for epoch in range(EPOCHS):
    total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
    for imgs, labels in dataset_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        cls_loss = nn.CrossEntropyLoss()(outputs, labels)

        f_l = activation[LAYER_NAME].view(imgs.size(0), -1)
        cav_1 = cav_vector_1 / cav_vector_1.norm()
        cav_2 = cav_vector_2 / cav_vector_2.norm()

        mask_1, mask_2 = (labels == TARGET_IDX_1), (labels == TARGET_IDX_2)

        def alignment(acts, cav):
            acts_norm = acts / acts.norm(dim=1, keepdim=True)
            return -torch.mean(torch.sum(acts_norm * cav, dim=1))

        align_loss_1 = alignment(f_l[mask_1], cav_1) if mask_1.any() else torch.tensor(0.0, device=DEVICE)
        align_loss_2 = alignment(f_l[mask_2], cav_2) if mask_2.any() else torch.tensor(0.0, device=DEVICE)
        align_loss = 0.0 * align_loss_1 + align_loss_2

        loss = LAMBDA_ALIGN * align_loss + LAMBDA_CLS * cls_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
        optimizer.step()

        total_loss_epoch += loss.item()
        cls_loss_epoch += cls_loss.item()
        align_loss_epoch += align_loss.item()

    n_batches = len(dataset_loader)
    loss_history["total"].append(total_loss_epoch / n_batches)
    loss_history["cls"].append(cls_loss_epoch / n_batches)
    loss_history["align"].append(align_loss_epoch / n_batches)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Total: {loss_history['total'][-1]:.6f}, "
          f"Cls: {loss_history['cls'][-1]:.6f}, Align: {loss_history['align'][-1]:.6f}")

# Save model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
print(f"Model saved at {RESULTS_PATH}")

# Post-training evaluation
acc_after, precision_after, recall_after, f1_after = evaluate_accuracy(model, validation_loader)
print(f"Accuracy after recalibration: {acc_after:.2f}%")

retrained_tcav_1 = compute_tcav_score(model, LAYER_NAME, cav_vector_1, validation_loader, TARGET_IDX_1)
retrained_tcav_2 = compute_tcav_score(model, LAYER_NAME, cav_vector_2, validation_loader, TARGET_IDX_2)
print(f"Retrained TCAV Score ({TARGET_CLASS_1}): {retrained_tcav_1:.4f}")
print(f"Retrained TCAV Score ({TARGET_CLASS_2}): {retrained_tcav_2:.4f}")

# Save stats
stats = {
    "Lambda Classification": LAMBDA_CLS,
    "Lambda Alignment": LAMBDA_ALIGN,
    "TCAV Before (Class 1)": original_tcav_1,
    "TCAV After (Class 1)": retrained_tcav_1,
    "TCAV Before (Class 2)": original_tcav_2,
    "TCAV After (Class 2)": retrained_tcav_2,
    "Accuracy Before": acc_before,
    "Accuracy After": acc_after,
    "Precision Before": precision_before,
    "Precision After": precision_after,
    "Recall Before": recall_before,
    "Recall After": recall_after,
    "F1 Before": f1_before,
    "F1 After": f1_after,
}

plot_loss_figure(loss_history["total"], loss_history["align"], loss_history["cls"], EPOCHS)
save_statistics(stats)
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, DEVICE, LAYER_NAMES,
    CONCEPT_FOLDER_1, CONCEPT_FOLDER_2, RANDOM_FOLDER, MODEL,
    CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_1, TARGET_CLASS_2,
    LAMBDA_ALIGNS, TARGET_CLASS_1_FOLDER, TARGET_CLASS_2_FOLDER,
)
from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from utils import (
    get_class_folder_dicts, train_cav, get_orthogonal_vector,
    evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence
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

target_class_1_dataset = DataLoader(SingleClassDataLoader(TARGET_CLASS_1_FOLDER, transform=transform), batch_size=BATCH_SIZE)
target_class_2_dataset = DataLoader(SingleClassDataLoader(TARGET_CLASS_2_FOLDER, transform=transform), batch_size=BATCH_SIZE)

concept_loader_1 = DataLoader(
    SingleClassDataLoader(CONCEPT_FOLDER_1, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)
concept_loader_2 = DataLoader(
    SingleClassDataLoader(CONCEPT_FOLDER_2, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)
random_loader = DataLoader(
    SingleClassDataLoader(RANDOM_FOLDER, transform=transform),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model and hook setup
activation = {}


def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output

    return hook


def compute_cav(model, loader_positive, loader_random, layer_name, orthogonal=False):
    pos_acts, rnd_acts = [], []
    model.eval()
    with torch.no_grad():
        for imgs in loader_positive:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            pos_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
        for imgs in loader_random:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            rnd_acts.append(activation[layer_name].view(imgs.size(0), -1).cpu().numpy())
    pos_acts = np.vstack(pos_acts)
    rnd_acts = np.vstack(rnd_acts)

    cav = train_cav(pos_acts, rnd_acts, orthogonal)
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)


def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, target_idx):
    model.eval()
    scores = []

    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, target_idx]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()
            grad_flat = grad.view(grad.size(0), -1)
            S = (grad_flat * cav_vector).sum(dim=1)
            scores.append(S > 0)
    scores = torch.cat(scores)
    return scores.float().mean().item()


def main():
    for layer_name in LAYER_NAMES:
        print(f"Computing cav for {layer_name}")
        for LAMBDA_ALIGN in LAMBDA_ALIGNS:
            print(f"LAMBDA_ALIGN = {LAMBDA_ALIGN}")
            LAMBDA_CLS = round((1.0 - LAMBDA_ALIGN), 2)

            model_trained = copy.deepcopy(MODEL).to(DEVICE)
            model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))

            # Compute two CAV vectors
            cav_vector_1 = compute_cav(model_trained, concept_loader_1, random_loader, layer_name, orthogonal=False)
            cav_vector_2 = compute_cav(model_trained, concept_loader_2, random_loader, layer_name, orthogonal=False)
            cav_vector_2_orthogonal = compute_cav(model_trained, concept_loader_2, random_loader, layer_name, orthogonal=True)

            # Compute original TCAV scores
            original_tcav_1 = compute_tcav_score(model_trained, layer_name, cav_vector_1, target_class_1_dataset, TARGET_IDX_1)
            original_tcav_2 = compute_tcav_score(model_trained, layer_name, cav_vector_2, target_class_2_dataset, TARGET_IDX_2)
            acc_before, precision_before, recall_before, f1_before = evaluate_accuracy(model_trained, validation_loader)
            avg_conf_1_before, avg_conf_2_before = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_1, TARGET_IDX_2)

            # Model training prep
            model_trained.train()
            for name, param in model_trained.named_parameters():
                param.requires_grad = (layer_name in name)
            model_trained.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=LEARNING_RATE)

            # Training loop
            loss_history = {"total": [], "cls": [], "align": []}

            print("Started training")

            for epoch in range(EPOCHS):
                total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
                for imgs, labels in dataset_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                    optimizer.zero_grad()

                    outputs = model_trained(imgs)
                    cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                    f_l = activation[layer_name].view(imgs.size(0), -1)

                    mask_1, mask_2 = (labels == TARGET_IDX_1), (labels == TARGET_IDX_2)

                    def alignment(acts, cav):
                        cosine_similarity = F.cosine_similarity(acts, cav.unsqueeze(0), dim=1)
                        return -torch.mean(cosine_similarity)

                    align_loss_1 = alignment(f_l[mask_1], cav_vector_1) if mask_1.any() else torch.tensor(
                        0.0,
                        device=DEVICE
                    )
                    align_loss_2 = alignment(f_l[mask_2], cav_vector_2) if mask_2.any() else torch.tensor(
                        0.0,
                        device=DEVICE
                    )
                    align_loss = align_loss_1 + align_loss_2

                    loss = LAMBDA_ALIGN * align_loss + LAMBDA_CLS * cls_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_trained.parameters(), max_norm=7)
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


            model_trained.eval()
            # Post-training evaluation
            acc_after, precision_after, recall_after, f1_after = evaluate_accuracy(model_trained, validation_loader)
            retrained_tcav_1 = compute_tcav_score(model_trained, layer_name, cav_vector_1, target_class_1_dataset, TARGET_IDX_1)
            retrained_tcav_2 = compute_tcav_score(model_trained, layer_name, cav_vector_2, target_class_2_dataset, TARGET_IDX_2)
            avg_conf_1_after, avg_conf_2_after = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_1, TARGET_IDX_2)

            # Save stats
            stats = {
                "Layer Name": layer_name,
                "Lambda Classification": LAMBDA_CLS,
                "Lambda Alignment": LAMBDA_ALIGN,
                "TCAV Before (Class 1)": round(original_tcav_1,3),
                "TCAV After (Class 1)": round(retrained_tcav_1, 3),
                "TCAV Before (Class 2)": round(original_tcav_2, 3),
                "TCAV After (Class 2)": round(retrained_tcav_2, 3),
                "Accuracy Before": round(acc_before, 3),
                "Accuracy After": round(acc_after, 3),
                "Precision Before": round(precision_before, 3),
                "Precision After": round(precision_after, 3),
                "Recall Before": round(recall_before, 3),
                "Recall After": round(recall_after, 3),
                "F1 Before": round(f1_before, 3),
                "F1 After": round(f1_after, 3),
                "Avg Conf 1 Before": round(avg_conf_1_before, 3),
                "Avg Conf 1 After": round(avg_conf_1_after, 3),
                "Avg Conf 2 Before": round(avg_conf_2_before, 3),
                "Avg Conf 2 After": round(avg_conf_2_after, 3),
            }

            plot_loss_figure(loss_history["total"], loss_history["align"], loss_history["cls"], EPOCHS)
            save_statistics(stats)


if __name__ == "__main__":
    main()

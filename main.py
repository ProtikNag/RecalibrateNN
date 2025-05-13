import logging
import copy
import os.path


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from new_config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, DEVICE, LAYER_NAMES,
    RANDOM_FOLDER, MODEL, CONCEPT_FOLDER,
    CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, BASE_MODEL,
    LAMBDA_ALIGNS
)
from utils import (
    get_class_folder_dicts, train_cav, evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence
)

# Configure logging
log_file = f"./results/{BASE_MODEL}_audit.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Log every executed line
def log_execution(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Executing: {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Finished: {func.__name__}")
        return result
    return wrapper

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]

train_dataset = MultiClassImageDataset(train_folders, transform=transform)
val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train"), transform=transform), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
concept_loader_list = [DataLoader(SingleClassDataLoader(path, transform=transform), batch_size=BATCH_SIZE, shuffle=True) for path in CONCEPT_FOLDER]
random_loader = DataLoader(SingleClassDataLoader(RANDOM_FOLDER, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

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
        model_trained = copy.deepcopy(MODEL).to(DEVICE)
        model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
        cav_vectors = [compute_cav(model_trained, concept_loader, random_loader, layer_name) for concept_loader in concept_loader_list]
        tcav_before = [compute_tcav_score(model_trained, layer_name, cav, class_loader, idx)
                       for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
        acc_before, precision_before, recall_before, f1_before = evaluate_accuracy(model_trained, validation_loader)
        avg_conf_before = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_LIST)
        reset_counter = 1

        for LAMBDA_ALIGN in LAMBDA_ALIGNS:
            LAMBDA_CLS = round(1.0 - LAMBDA_ALIGN, 2)
            logger.info(f"Itertion counter:  {reset_counter}")
            logger.info(f"Layer: {layer_name} - Lambda Classification: {LAMBDA_CLS} - Lambda Alignment: {LAMBDA_ALIGN}")
            model_trained = copy.deepcopy(MODEL).to(DEVICE)
            model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))

            # Freeze all but target layer
            model_trained.train()
            for name, param in model_trained.named_parameters():
                param.requires_grad = (layer_name in name)
            model_trained.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None)

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=LEARNING_RATE)
            loss_history = {"total": [], "cls": [], "align": []}
          
            for epoch in range(EPOCHS):
                total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
                for imgs, labels in dataset_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model_trained(imgs)
                    cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                    f_l = activation[layer_name].view(imgs.size(0), -1)

                    align_loss = 0.0
                    for i, target_idx in enumerate(TARGET_IDX_LIST):
                        mask = (labels == target_idx)
                        if mask.any():
                            cosine_similarity = F.cosine_similarity(f_l[mask], cav_vectors[i].unsqueeze(0), dim=1)
                            align_loss += torch.abs(-torch.mean(cosine_similarity))

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
                logger.info(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss_history['total'][-1]:.4f}")

            acc_after, precision_after, recall_after, f1_after = evaluate_accuracy(model_trained, validation_loader)
            tcav_after = [compute_tcav_score(model_trained, layer_name, cav, class_loader, idx)
                          for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
            avg_conf_after = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_LIST)
            
            # Collect metrics
            stats = {
                "Layer Name": layer_name,
                "Lambda Classification": LAMBDA_CLS,
                "Lambda Alignment": LAMBDA_ALIGN,
                "Accuracy Before": round(acc_before, 3),
                "Accuracy After": round(acc_after, 3),
                "Precision Before": round(precision_before, 3),
                "Precision After": round(precision_after, 3),
                "Recall Before": round(recall_before, 3),
                "Recall After": round(recall_after, 3),
                "F1 Before": round(f1_before, 3),
                "F1 After": round(f1_after, 3),
            }

            for i, class_name in enumerate(TARGET_CLASS_LIST):
                stats[f"TCAV Before ({class_name})"] = round(tcav_before[i], 3)
                stats[f"TCAV After ({class_name})"] = round(tcav_after[i], 3)
                stats[f"Avg Conf {class_name} Before"] = round(avg_conf_before[i], 3)
                stats[f"Avg Conf {class_name} After"] = round(avg_conf_after[i], 3)

            plot_loss_figure(loss_history["total"], loss_history["align"], loss_history["cls"], EPOCHS)
            save_statistics(stats)
            reset_counter = reset_counter + 1


if __name__ == "__main__":
    main()

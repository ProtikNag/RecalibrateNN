import os
import numpy as np
import torch
from sklearn.svm import LinearSVC

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Auto parse class folders
def get_class_folder_dicts(base_dir):
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    train_class_folders = {}
    valid_class_folders = {}
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
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector
    return cav_vector


def cosine_similarity_loss(grad, cav_vector):
    grad_norm = grad / grad.norm(dim=1, keepdim=True)
    cav_norm = cav_vector / cav_vector.norm()  # Ensure CAV is unit norm
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

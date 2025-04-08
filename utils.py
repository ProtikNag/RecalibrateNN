import os
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

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


# def evaluate_accuracy(model, loader):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#             preds = model(imgs).argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return 100.0 * correct / total if total > 0 else 0.0


def evaluate_accuracy(model, loader):
    """
    Evaluates the model on a given DataLoader `loader` and appends:
    - Overall accuracy, precision, recall, F1 score
    - Correct prediction count per class
    to './results/validation_metrics.txt'.
    """
    os.makedirs("./results", exist_ok=True)

    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Class-wise stats
    class_names = loader.dataset.class_names if hasattr(loader.dataset, "class_names") else [str(i) for i in sorted(set(all_labels))]
    class_correct_counts = {name: 0 for name in class_names}
    class_total_counts = {name: 0 for name in class_names}

    for true, pred in zip(all_labels, all_preds):
        class_name = class_names[true]
        class_total_counts[class_name] += 1
        if true == pred:
            class_correct_counts[class_name] += 1

    # Append to results file
    results_path = os.path.join("results", "validation_metrics.txt")
    with open(results_path, 'a') as f:
        f.write("\n=== New Evaluation ===\n")
        f.write(f"Overall Accuracy: {acc:.4f} ({sum(class_correct_counts.values())}/{len(all_labels)})\n")
        f.write(f"Overall Precision (macro): {precision:.4f}\n")
        f.write(f"Overall Recall (macro): {recall:.4f}\n")
        f.write(f"Overall F1-score (macro): {f1:.4f}\n\n")
        for class_name in class_names:
            correct = class_correct_counts[class_name]
            total = class_total_counts[class_name]
            f.write(f"Class: {class_name} — Correct: {correct} / {total}\n")

    print(f"Validation metrics and class counts appended to {results_path}")
    return 100.0 * acc
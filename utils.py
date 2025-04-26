import os
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import csv

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


def get_orthogonal_vector(cav_vector):
    random_vector = np.random.randn(*cav_vector.shape)
    projection = np.dot(random_vector, cav_vector) * cav_vector
    orthogonal_vector = random_vector - projection
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    # dot_product = np.dot(orthogonal_vector, cav_vector)

    return orthogonal_vector


def train_cav(concept_activations, random_activations, orthogonal=True):
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))
    clf = LinearSVC(max_iter=1500).fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector

    if orthogonal:
        cav_vector = get_orthogonal_vector(cav_vector)

    return cav_vector


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
    class_names = loader.dataset.class_names if hasattr(loader.dataset, "class_names") else [str(i) for i in
                                                                                             sorted(set(all_labels))]
    class_correct_counts = {name: 0 for name in class_names}
    class_total_counts = {name: 0 for name in class_names}

    for true, pred in zip(all_labels, all_preds):
        class_name = class_names[true]
        class_total_counts[class_name] += 1
        if true == pred:
            class_correct_counts[class_name] += 1

    # Append to results file
    for class_name in class_names:
        correct = class_correct_counts[class_name]
        total = class_total_counts[class_name]
        print(f"Class: {class_name} — Correct: {correct} / {total}\n")

    return acc, precision, recall, f1


def compute_avg_confidence(model, loader, idx_1, idx_2):
    model.eval()
    confidences_1 = []
    confidences_2 = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            confidences_1 += probs[preds == idx_1, idx_1].tolist()
            confidences_2 += probs[preds == idx_2, idx_2].tolist()
    avg_conf_1 = np.mean(confidences_1) if confidences_1 else 0.0
    avg_conf_2 = np.mean(confidences_2) if confidences_2 else 0.0
    return avg_conf_1, avg_conf_2


def plot_loss_figure(total_loss_history, align_loss_history, cls_loss_history, epochs):
    # plot total loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), total_loss_history, marker='o', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Total Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    total_loss_plot_path = os.path.join("results", "total_loss.pdf")
    plt.savefig(total_loss_plot_path)
    plt.close()

    # plot classification loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), cls_loss_history, marker='x', label='Classification Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Classification Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    cls_loss_plot_path = os.path.join("results", "classification_loss.pdf")
    plt.savefig(cls_loss_plot_path)
    plt.close()

    # plot alignment loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), align_loss_history, marker='s', label='Alignment Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Alignment Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    align_loss_plot_path = os.path.join("results", "alignment_loss.pdf")
    plt.savefig(align_loss_plot_path)
    plt.close()


def save_statistics(stat, filename="results/statistics.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stat.keys())

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(stat)

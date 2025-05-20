import os
import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import csv
from sklearn.linear_model import SGDClassifier, LogisticRegression
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dynamically determine the number of classes
def get_num_classes(base_path):
    return len([
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ])


# Get base model image size
def get_base_model_image_size(base_model_path):
    if 'inception' in base_model_path:
        image_size = 299
    else:
        image_size = 224

    return image_size



def get_model_weight_path(base_model, model_root_path =r'./model_weights'):
    base_model_path = os.path.join(model_root_path , base_model + "/" + base_model + ".pth")
    print(model_root_path, base_model_path )
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model weights not found at {base_model_path}")
    return base_model_path


def get_model_layers(model):
    layer_types = (nn.Conv2d, nn.MaxPool2d)
    layers = []
    param_names = [name for name, _ in model.named_parameters()]

    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            if any(pname.startswith(name) for pname in param_names):
                layers.append(name)
    return layers[-4:]


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


def train_cav(concept_activations, random_activations, orthogonal=False, classifier_type='LinearSVC'):
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    if classifier_type == 'LinearSVC':
        clf = LinearSVC(max_iter=1500)
    elif classifier_type == 'SGDClassifier':
        clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)  # hinge = SVM-like
    elif classifier_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000, solver='liblinear')

    clf.fit(X, y)
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


def compute_avg_confidence(model, loader, target_idx_list):
    model.eval()
    class_confidences = {idx: [] for idx in target_idx_list}

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            for idx in target_idx_list:
                confs = probs[preds == idx, idx]
                class_confidences[idx] += confs.tolist()

    avg_confidences = {
        idx: (np.mean(confs) if confs else 0.0)
        for idx, confs in class_confidences.items()
    }

    # Return as a list aligned with input index order
    return [avg_confidences[idx] for idx in target_idx_list]


def plot_loss_figure(total_loss_history, align_loss_history, cls_loss_history, epochs,
                     classification_loss="./results/classification_loss.pdf",
                     alignment_loss="./results/alignment_loss.pdf", total_loss="./results/total_loss.pdf"):
    # plot total loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), total_loss_history, marker='o', label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Total Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    total_loss_plot_path = total_loss
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
    cls_loss_plot_path = classification_loss
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
    align_loss_plot_path = alignment_loss
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

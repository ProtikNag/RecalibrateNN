import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_class_folder_dicts
from custom_dataloader import MultiClassImageDataset
import matplotlib.pyplot as plt


def evaluate_model_on_validation_set(model, val_folders, class_names, batch_size=64,
                                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluates the model on the validation dataset and saves:
    - Overall accuracy, precision, recall, F1 score
    - Correct prediction count per class (e.g., "Class A: 45 out of 50")
    """
    os.makedirs("./results", exist_ok=True)

    # Transform must match training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = MultiClassImageDataset(val_folders, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Class-wise correct count
    class_correct_counts = {name: 0 for name in class_names}
    class_total_counts = {name: 0 for name in class_names}

    for true, pred in zip(all_labels, all_preds):
        class_name = class_names[true]
        class_total_counts[class_name] += 1
        if true == pred:
            class_correct_counts[class_name] += 1

    # Save to file
    results_path = os.path.join("results", "validation_metrics.txt")
    with open(results_path, 'w') as f:
        f.write(f"Overall Accuracy: {acc:.4f} ({sum([class_correct_counts[c] for c in class_names])}/{len(all_labels)})\n")
        f.write(f"Overall Precision (macro): {precision:.4f}\n")
        f.write(f"Overall Recall (macro): {recall:.4f}\n")
        f.write(f"Overall F1-score (macro): {f1:.4f}\n\n")

        for class_name in class_names:
            correct = class_correct_counts[class_name]
            total = class_total_counts[class_name]
            f.write(f"Class: {class_name} — Correct: {correct} / {total}\n")

    print(f"Validation metrics and class counts saved to {results_path}")


def training_biased_model(
        model,
        data_path="./data/binary_classification/",
        epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains a given model from scratch on a biased dataset, evaluates on validation,
    and saves training loss plot and validation metrics.
    """

    os.makedirs("./results", exist_ok=True)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load biased data
    train_folders, val_folders, class_names = get_class_folder_dicts(data_path)
    train_dataset = MultiClassImageDataset(train_folders, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up model
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    training_losses = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        training_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save training loss plot
    plt.figure()
    plt.plot(range(1, epochs + 1), training_losses, marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join("results", "training_loss.pdf")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")

    # Final evaluation
    model.eval()
    evaluate_model_on_validation_set(model, val_folders=val_folders, class_names=class_names,
                                     batch_size=batch_size, device=device)

    return model

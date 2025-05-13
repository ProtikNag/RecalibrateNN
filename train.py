import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_dataloader import MultiClassImageDataset
from model import DeepCNN
from utils import get_class_folder_dicts
from new_config import get_num_classes

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLASSES = get_num_classes('./data/multi_class_classification/')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def training_biased_model(
        model,
        data_path="./data/multi_class_classification/",
        max_epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping_patience=10,
        lr_decay_step=3,
        lr_decay_gamma=0.5,
        loss_delta=1e-3
):
    """
    Trains a given model from scratch on an imbalanced dataset, evaluates on validation set,
    and saves training/validation loss and accuracy plots. Uses early stopping and learning rate decay.
    """

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./model_weights", exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load data
    train_folders, val_folders, class_names = get_class_folder_dicts(data_path)
    train_dataset = MultiClassImageDataset(train_folders, transform=transform)
    val_dataset = MultiClassImageDataset(val_folders, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    training_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Validation step
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val

        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{max_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss + loss_delta < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(range(1, len(training_losses) + 1), training_losses, label='Train Loss')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    axs[0].set_title("Loss over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    axs[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
    axs[1].set_title("Accuracy over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("./results/training_loss_before_recal.pdf")
    plt.close()

    return model


if __name__ == "__main__":
    model = DeepCNN(NUM_CLASSES).to(DEVICE)
    trained_model = training_biased_model(model)
    torch.save(trained_model.state_dict(), "./model_weights/imbalanced_model.pth")

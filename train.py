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


def training_biased_model(
        model,
        data_path="./data/multi_class_classification/",
        epochs=20,
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
    plt.title('Training Loss over Epochs - Before Recalibration')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join("results", "training_loss_before_recal.pdf")
    plt.savefig(loss_plot_path)
    plt.close()

    return model

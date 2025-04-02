import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataloader import MultiClassImageDataset
from utils import get_class_folder_dicts


def training_biased_model(
        model,
        data_path="./data/binary_classification/",
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Trains a given model from scratch on a biased dataset and returns the trained model.
    """

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load biased data
    train_folders, _, class_names = get_class_folder_dicts(data_path)
    train_dataset = MultiClassImageDataset(train_folders, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up model
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Final evaluation
    model.eval()

    return model

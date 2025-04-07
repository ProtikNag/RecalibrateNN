import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
log_file = "training_log.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting the training process...")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define class names
class_names = ['deer',  'horse','zebra']
logging.info(f"Class names: {class_names}")
print(f"Class names used for training: {class_names}")

# Load pre-trained GoogLeNet model
model = models.googlenet(pretrained=True)

# Modify the final fully connected layer for 3 classes
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dataset = ImageFolder(root='/home/srikanth/model_train/data/train', transform=transform)
val_dataset = ImageFolder(root='/home/srikanth/model_train/data/valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
history = {'train_loss': [], 'val_accuracy': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    history['val_accuracy'].append(val_accuracy)
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.2f}%")

# Save the trained model
model_path = 'googlenet_three_classes.pth'
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")

# Save training history plot
plt.figure()
plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plot_path = 'training_loss_plot.png'
plt.savefig(plot_path)
logging.info(f"Training loss plot saved to {plot_path}")
plt.close()

logging.info("Training process completed.")
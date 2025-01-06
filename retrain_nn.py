import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.tcav_pipeline import tcav_pipeline
import matplotlib.pyplot as plt  # Import for plotting

print(os.getcwd())

# Configuration
TARGET_TCAV_SCORE = 0.8
LEARNING_RATE = 0.1
EPOCHS = 20
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
CONCEPT_FOLDER = "./data/concept/striped"
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
MODEL_NAME = "googlenet"
LAYER_NAME = "inception4a"
RESULTS_PATH = "./results/retrained_model.pth"

# Load pre-trained model
model = models.googlenet(pretrained=True).to(DEVICE)
model.train()

# Freeze all layers except the target layer
for name, param in model.named_parameters():
    if LAYER_NAME not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Define Loss
def tcav_loss(avg_tcav_score, target_tcav_score):
    loss = (avg_tcav_score - target_tcav_score) ** 2
    return loss

# Define optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# List to store the loss values
loss_history = []

# Training loop
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    model.train()

    # Run TCAV pipeline to get the current TCAV score
    tcav_scores = tcav_pipeline(
        model=model,
        layer_name=LAYER_NAME,
        concept_folder=CONCEPT_FOLDER,
        random_folder=RANDOM_FOLDER,
        target_class=340,  # Adjust based on your dataset
        dataset_folder=DATASET_FOLDER,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    avg_tcav_score = torch.mean(torch.tensor(tcav_scores, requires_grad=True))

    print(f"Epoch {epoch}: Current avg TCAV score = {avg_tcav_score}")

    # Compute loss
    loss = tcav_loss(avg_tcav_score, TARGET_TCAV_SCORE)
    print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Store the loss value
    loss_history.append(loss.item())

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

# Save the recalibrated model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)

print(f"Model saved at {RESULTS_PATH}")

# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(EPOCHS), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.show()

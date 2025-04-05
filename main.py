import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from config import (
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    DEVICE,
    MODEL,
    LAYER_NAME,
    CONCEPT_FOLDER,
    RANDOM_FOLDER,
    BINARY_CLASSIFICATION_BASE,
    RESULTS_PATH,
    ZEBRA_CLASS_NAME,
    LAMBDA_ALIGN,
    LAMBDA_CLS,
)
from custom_dataloader import ConceptDataset, MultiClassImageDataset
from utils import get_class_folder_dicts, train_cav, cosine_similarity_loss, evaluate_accuracy
import csv

def log_config():
    logger.info("Configuration:")
    logger.info('-----------------------')
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL.__class__.__name__}")
    logger.info(f"Layer Name: {LAYER_NAME}")
    logger.info(f"Concept Folder: {CONCEPT_FOLDER}")
    logger.info(f"Random Folder: {RANDOM_FOLDER}")
    logger.info(f"Binary Classification Base: {BINARY_CLASSIFICATION_BASE}")
    logger.info(f"Results Path: {RESULTS_PATH}")
    logger.info(f"Zebra Class Name: {ZEBRA_CLASS_NAME}")
    logger.info(f"Lambda Align: {LAMBDA_ALIGN}")
    logger.info(f"Lambda Classification: {LAMBDA_CLS}")
    logger.info('-----------------------')

def verify_data():
    #Check if we have the images in the correct folders
    if not os.path.exists(CONCEPT_FOLDER):
        raise FileNotFoundError(f"Concept folder {CONCEPT_FOLDER} not found.")
    if not os.path.exists(RANDOM_FOLDER):
        raise FileNotFoundError(f"Random folder {RANDOM_FOLDER} not found.")
    if not os.path.exists(BINARY_CLASSIFICATION_BASE):
        raise FileNotFoundError(f"Binary classification base folder {BINARY_CLASSIFICATION_BASE} not found.")
    #Count the images present for each class and log the information
    train_folders, valid_folders, class_names = get_class_folder_dicts(BINARY_CLASSIFICATION_BASE)
    logger.info("Training and validation folders found.")
    logger.info(f"Classes: {class_names}")
    for folder in train_folders:
        num_images = len(os.listdir(folder))
        logger.info(f"Number of images in {folder}: {num_images}")
    for folder in valid_folders:
        num_images = len(os.listdir(folder))
        logger.info(f"Number of images in {folder}: {num_images}")


def evaluate_and_save_predictions(model, train_folders, valid_folders, transform, output_csv_path):
    model.eval()
    all_folders = train_folders + valid_folders
    results = []
    with torch.no_grad():
        for folder in all_folders:
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = plt.imread(img_path)
                img = transform(img).unsqueeze(0).to(DEVICE)
                outputs = model(img)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = class_names[predicted.item()]
                confidence_percentage = confidence.item() * 100
                results.append({
                    "Image": img_name,
                    "Folder": folder,
                    "Predicted Class": predicted_class,
                    "Confidence (%)": f"{confidence_percentage:.2f}"
                })
            # Write results to CSV
            with open(output_csv_path, mode='w', newline='') as csv_file:
                fieldnames = ["Image", "Folder", "Predicted Class", "Confidence (%)"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
    logger.info(f"Predictions saved to {output_csv_path}")

# Configure logging
logging.basicConfig(
    filename='training_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

logger.info("Starting the script...")
log_config()
verify_data()

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_folders, valid_folders, class_names = get_class_folder_dicts(BINARY_CLASSIFICATION_BASE)
evaluate_and_save_predictions(MODEL, train_folders, valid_folders, transform, 'predictions.csv')

NUM_CLASSES = len(class_names)
ZEBRA_IDX = class_names.index(ZEBRA_CLASS_NAME)

train_dataset = MultiClassImageDataset(train_folders, transform=transform)
val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = MODEL
model.train()

# Hook activations
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


model.get_submodule(LAYER_NAME).register_forward_hook(get_activation(LAYER_NAME))

# Compute CAV
logger.info("Computing CAV...")
concept_activations, random_activations = [], []
model.eval()
with torch.no_grad():
    for imgs in concept_loader:
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        acts = activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy()
        concept_activations.append(acts)
    for imgs in random_loader:
        imgs = imgs.to(DEVICE)
        _ = model(imgs)
        acts = activation[LAYER_NAME].view(imgs.size(0), -1).cpu().numpy()
        random_activations.append(acts)
concept_activations = np.vstack(concept_activations)
random_activations = np.vstack(random_activations)
cav_vector = train_cav(concept_activations, random_activations)
cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

logger.info("CAV computation completed.")


def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, k):
    model.eval()
    scores = []
    for imgs, _ in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, k]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()
            grad_flat = grad.view(grad.size(0), -1)
            S = (grad_flat * cav_vector).sum(dim=1)
            scores.append(S > 0)
    scores = torch.cat(scores)
    return scores.float().mean().item()


original_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, ZEBRA_IDX)
logger.info(f"Original TCAV Score (Zebra): {original_tcav:.4f}")
acc_before = evaluate_accuracy(model, validation_loader)
logger.info(f"Accuracy before recalibration: {acc_before:.2f}%")


def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()


def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()


model.train()
for name, param in model.named_parameters():
    param.requires_grad = (LAYER_NAME in name)
model.apply(set_batchnorm_eval)
model.apply(set_dropout_eval)

params_to_train = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(filter(lambda p: p.requires_grad, params_to_train), lr=LEARNING_RATE)

# Training loop
logger.info("Starting training...")
loss_history = []
for epoch in range(EPOCHS):
    total_loss = 0.0
    for imgs, labels in dataset_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        classification_loss = nn.CrossEntropyLoss()(outputs, labels)
        f_l = activation[LAYER_NAME]
        h_k = outputs[:, ZEBRA_IDX]
        grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0]
        grad_flat = grad.view(grad.size(0), -1)
        alignment_loss = cosine_similarity_loss(grad_flat, cav_vector)
        loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss
        logger.debug(f"Alignment Loss: {alignment_loss.item():.7f}, Total Loss: {loss.item():.7f}")
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
        optimizer.step()

    avg_loss = total_loss / len(dataset_loader)
    loss_history.append(avg_loss)
    logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.7f}")

# Save model
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
torch.save(model.state_dict(), RESULTS_PATH)
logger.info(f"Model saved at {RESULTS_PATH}")

acc_after = evaluate_accuracy(model, validation_loader)
logger.info(f"Accuracy after recalibration: {acc_after:.2f}%")

retrained_tcav = compute_tcav_score(model, LAYER_NAME, cav_vector, dataset_loader, ZEBRA_IDX)
logger.info(f"Retrained TCAV Score (Zebra): {retrained_tcav:.4f}")

# Plot loss
plt.figure(figsize=(8, 6))
plt.plot(range(EPOCHS), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Loss Curve over Epochs')
plt.grid(True)
plt.show()
plt.savefig('loss_curve.png')
logger.info("Training completed and loss curve plotted.")
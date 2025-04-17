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
    MODEL_PATH,
    DEVICE,
    MODEL,
    LAYER_NAME,
    CONCEPT_FOLDER,
    RANDOM_FOLDER,
    NUM_CLASSES,
    BINARY_CLASSIFICATION_BASE,
    RESULTS_PATH,
    RESULTS_FILE_NAME,
    TARGET_CLASS_NAME,
    LAMBDA_ALIGN,
    LAMBDA_CLS,
    TRANSFORMS,
)
from custom_dataloader import ConceptDataset, MultiClassImageDataset
from utils import get_class_folder_dicts, train_cav, cosine_similarity_loss, evaluate_accuracy
import argparse

# Configure logging
logging.basicConfig(
    filename='audit_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Program started.")




def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output

    return hook

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

def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.eval()

def set_dropout_eval(module):
    if isinstance(module, nn.Dropout):
        module.eval()

def argumentparser():
    parser = argparse.ArgumentParser(description="TCAV-based model recalibration")

    # Add arguments
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for data loaders")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run the model on (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--layer_name", type=str, default=LAYER_NAME, help="Layer name for TCAV computation")
    parser.add_argument("--concept_folder", type=str, default=CONCEPT_FOLDER, help="Path to concept images folder")
    parser.add_argument("--random_folder", type=str, default=RANDOM_FOLDER, help="Path to random images folder")
    parser.add_argument("--binary_classification_base", type=str, default=BINARY_CLASSIFICATION_BASE, help="Base path for binary classification dataset")
    parser.add_argument("--results_path", type=str, default=RESULTS_PATH, help="Path to save the recalibrated model")
    parser.add_argument("--target_class_name", type=str, default=TARGET_CLASS_NAME, help="Class name for Zebra")
    parser.add_argument("--lambda_align", type=float, default=LAMBDA_ALIGN, help="Weight for alignment loss")
    parser.add_argument("--lambda_cls", type=float, default=LAMBDA_CLS, help="Weight for classification loss")
    args = parser.parse_args()
    # Override config values with parsed arguments
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    LAYER_NAME = args.layer_name
    CONCEPT_FOLDER = args.concept_folder
    RANDOM_FOLDER = args.random_folder
    BINARY_CLASSIFICATION_BASE = args.binary_classification_base
    RESULTS_PATH = args.results_path
    TARGET_CLASS_NAME = args.target_class_name
    LAMBDA_ALIGN = args.lambda_align
    LAMBDA_CLS = args.lambda_cls
    # Log all parsed arguments
    logging.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    # Parse arguments
    return 

def loaddataset(train_folders, valid_folder, transform):
    train_dataset = MultiClassImageDataset(train_folders, transform=transform)
    val_dataset = MultiClassImageDataset(valid_folders, transform=transform)
    dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    concept_dataset = ConceptDataset(CONCEPT_FOLDER, transform=transform)
    random_dataset = ConceptDataset(RANDOM_FOLDER, transform=transform)
    concept_loader = DataLoader(concept_dataset, batch_size=BATCH_SIZE, shuffle=True)
    random_loader = DataLoader(random_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return (dataset_loader, validation_loader, concept_loader, random_loader)

def compute_tcav_score(model,activation, layer_name, cav_vector, dataset_loader, k):
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

def cav_score_calculation(model, dataset_loader, layer_name):
    # Hook activations
    activation = {}
    model.get_submodule(layer_name).register_forward_hook(get_activation(layer_name,activation))
    # Compute CAV
    logging.info("Computing CAV vector.")
    concept_activations, random_activations = [], []
    model.eval()
    with torch.no_grad():
        for imgs in concept_loader:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            acts = activation[layer_name].view(imgs.size(0), -1).cpu().numpy()
            concept_activations.append(acts)
        for imgs in random_loader:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            acts = activation[layer_name].view(imgs.size(0), -1).cpu().numpy()
            random_activations.append(acts)

    concept_activations = np.vstack(concept_activations)
    random_activations = np.vstack(random_activations)
    
    model.train()
    cav_vector = train_cav(concept_activations, random_activations)
    cav_vector = torch.tensor(cav_vector, dtype=torch.float32, device=DEVICE)

    logging.info("CAV vector computed successfully.")
    original_tcav = compute_tcav_score(model,activation, LAYER_NAME, cav_vector, dataset_loader, TARGET_IDX)
    logging.info(f"Original TCAV Score (Zebra): {original_tcav:.4f}")
    print(f"Original TCAV Score (Zebra): {original_tcav:.4f}")

    acc_before = evaluate_accuracy(model, validation_loader)
    logging.info(f"Accuracy before recalibration: {acc_before:.2f}%")
    print(f"Accuracy before recalibration: {acc_before:.2f}%")
    return model,cav_vector, activation


def recalibrate_network(model, dataset_loader, cav_vector, activation, layer_name):
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = (LAYER_NAME in name)
    model.apply(set_batchnorm_eval)
    model.apply(set_dropout_eval)

    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params_to_train), lr=LEARNING_RATE)

    # Training loop
    logging.info("Starting training loop.")
    loss_history = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for imgs, labels in dataset_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            classification_loss = nn.CrossEntropyLoss()(outputs, labels)
            f_l = activation[LAYER_NAME]

            activation_flat = f_l.view(f_l.size(0), -1)
            activation_norm = activation_flat / activation_flat.norm(dim=1, keepdim=True)
            alignment_loss = torch.mean(1 - torch.abs(torch.sum(activation_norm * cav_vector, dim=1)))
            loss = LAMBDA_ALIGN * alignment_loss + LAMBDA_CLS * classification_loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=7)
            optimizer.step()

        avg_loss = total_loss / len(dataset_loader)
        loss_history.append(avg_loss)
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.7f}")
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.7f}")
    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(EPOCHS), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Loss Curve over Epochs')
    plt.grid(True)
    #plt.show()
    plt.savefig('recalibrated_loss.png')
    logging.info("Program finished.")

    return model



if(__name__ == "__main__"):
    # Parse arguments
    #argumentparser()
    # Prepare data
    logging.info("Preparing datasets and data loaing datasets ")
    transforms = TRANSFORMS
    model = MODEL
    
    train_folders, valid_folders, class_names = get_class_folder_dicts(BINARY_CLASSIFICATION_BASE)
    if(NUM_CLASSES != len(class_names)):
        raise ValueError(f"Number of classes in dataset ({len(class_names)}) does not match expected number ({NUM_CLASSES}).")
    
    dataset_loader, validation_loader, concept_loader, random_loader = loaddataset(train_folders, valid_folders, TRANSFORMS)
    logging.info("Loaded  datasets and data loaders.")

    logging.info(f"Training dataset folder {train_folders} and validation dataset folder {valid_folders}")    
    TARGET_IDX = class_names.index(TARGET_CLASS_NAME)
    # Load model
    logging.info("Loading pre-trained model.")
    
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    logging.info("Model LoadedProceeding to start TCAV based recalibration.")
    model, cav_vector, activation = cav_score_calculation(model, dataset_loader, LAYER_NAME)
    retrained_model = recalibrate_network(model, dataset_loader, cav_vector, activation, LAYER_NAME)
    # Save model
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    
    torch.save(retrained_model, f"{RESULTS_PATH}/{RESULTS_FILE_NAME}")
    logging.info(f"Model saved at {RESULTS_PATH}")
    print(f"Model saved at {RESULTS_PATH}")

    acc_after = evaluate_accuracy(retrained_model, validation_loader)
    logging.info(f"Accuracy after recalibration: {acc_after:.2f}%")
    print(f"Accuracy after recalibration: {acc_after:.2f}%")

    retrained_tcav = compute_tcav_score(retrained_model,activation, LAYER_NAME, cav_vector, dataset_loader, TARGET_IDX)
    logging.info(f"Retrained TCAV Score (Zebra): {retrained_tcav:.4f}")
    print(f"Retrained TCAV Score (Zebra): {retrained_tcav:.4f}")

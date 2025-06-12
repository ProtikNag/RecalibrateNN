import copy
import os.path
from logger import Logger_Singleton

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import DeepCNN

from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from datetime import datetime
import argparse
from config import (
    LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
    DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
    CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
)

if(os.environ.get('PLATFORM') == "Srikanth"):
  print("overriding config paths to point to directory structure of srikanth. Note Protik will not have this parameter set ") 
  from config_modified import (
      LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
      DEVICE, RANDOM_FOLDER, CONCEPT_FOLDER_LIST, LINEAR_CLASSIFIER_TYPE,
      CLASSIFICATION_DATA_BASE_PATH, TARGET_CLASS_LIST, LAMBDA_ALIGNS
  )
  print("Taking all the required path from Srikanths folder" )


from utils import (
    get_num_classes, get_class_folder_dicts, train_cav, evaluate_accuracy, plot_loss_figure, save_statistics,
    compute_avg_confidence, get_model_weight_path, get_base_model_image_size, get_model_layers, predict_from_loader 
)

from tcav_utils import (util_compute_cav)

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None

activation = {}
output_shape = {}


def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        # This print has been added for you to visualize if the size is too large then the time taken fror convergence will be large
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
        #logging.info(f"Verify the output shape : Layername = {layer_name} ,Input.shape : {input[0].shape},  output.shape : {output.shape}")

    return hook


def compute_tcav_score(model, layer_name, cav_vector, dataset_loader, target_idx):
    logging.info(f"Computing TCAV score for layer: {layer_name}, target index: {target_idx}")
    model.eval()
    scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(DEVICE)
        with torch.enable_grad():
            outputs = model(imgs)
            f_l = activation[layer_name]
            h_k = outputs[:, target_idx]
            grad = torch.autograd.grad(h_k.sum(), f_l, retain_graph=True)[0].detach()  # check the documentation for the default values
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = F.normalize(grad_flat, p=2, dim=1)
            S = (grad_norm * cav_vector).sum(dim=1)
            scores.append(S > 0)  # additional logging for sensitivity analysis
    scores = torch.cat(scores)
    logging.info(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    print(f"TCAV score computation completed for layer: {layer_name}, target index: {target_idx}")
    return scores.float().mean().item()


def main():
    logging.info("Main function started.")
    try:
        for layer_name in LAYER_NAMES:
            logging.info(f"Processing layer: {layer_name}")
            try:

                model_trained = copy.deepcopy(MODEL).to(DEVICE)
                model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                print("Computing the cav vectors can take a while stand by")
                cav_vectors = [util_compute_cav(model_trained, concept_loader, random_loader, layer_name, activation) for concept_loader in concept_loader_list]
                print("Computing the tcav scores can take a while stand by")
                tcav_before = [compute_tcav_score(model_trained, layer_name, cav, class_loader, idx)
                            for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
                print(f"TCAV Score before : {tcav_before} for layer {layer_name}")
                logging.info(f"TCAV Score before : {tcav_before} for layer {layer_name}")
            

                acc_before, precision_before, recall_before, f1_before = evaluate_accuracy(model_trained, validation_loader)
                avg_conf_before = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_LIST)
                results_legacy, avg_confidences_legacy, class_count_legacy_before, acc_legacy_before = predict_from_loader(validation_loader, model_trained, TARGET_IDX_LIST)

                logging.info(f"Accuracy Before: {acc_before:.4f}")
                logging.info(f"Precision Before: {precision_before:.4f}")
                logging.info(f"Recall Before: {recall_before:.4f}")
                logging.info(f"F1 Score Before: {f1_before:.4f}")
                logging.info(f"Average Confidence Before: {avg_conf_before}")
                logging.info(f"TCAV Score before : {tcav_before}")
                print(f"Accuracy Before: {acc_before:.4f}, tcav_before: {tcav_before}")
            except Exception as e:
                logging.error(f"Error during initial evaluation: {e}")
                print(f"Error during initial evaluation: {e}")
                continue
            try:
                for LAMBDA_ALIGN in LAMBDA_ALIGNS:
                    LAMBDA_CLS = round(1.0 - LAMBDA_ALIGN, 2)
                    logging.info(f"Training with Lambda Align: {LAMBDA_ALIGN}, Lambda Classification: {LAMBDA_CLS}")
                    print(f"Training with Lambda Align: {LAMBDA_ALIGN}, Lambda Classification: {LAMBDA_CLS}")
                    model_trained = copy.deepcopy(MODEL).to(DEVICE)
                    model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                    model_trained.train()
                    for name, param in model_trained.named_parameters():
                        param.requires_grad = (layer_name in name)
                    model_trained.apply(lambda m: m.eval() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Dropout)) else None)

                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trained.parameters()), lr=LEARNING_RATE)
                    loss_history = {"total": [], "cls": [], "align": []}

                    for epoch in range(EPOCHS):
                        total_loss_epoch = cls_loss_epoch = align_loss_epoch = 0.0
                        for imgs, labels in dataset_loader:
                            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                            optimizer.zero_grad()
                            if (BASE_MODEL == 'inception_v3'):
                                outputs, aux_logits = model_trained(imgs)
                            else:
                                outputs = model_trained(imgs)
                            cls_loss = nn.CrossEntropyLoss()(outputs, labels)
                            f_l = activation[layer_name].view(imgs.size(0), -1)

                            align_loss = 0.0
                            for i, target_idx in enumerate(TARGET_IDX_LIST):
                                mask = (labels == target_idx)
                                if mask.any():
                                    cosine_similarity = F.cosine_similarity(f_l[mask], cav_vectors[i].unsqueeze(0), dim=1)
                                    align_loss += (1 - torch.mean(torch.abs(cosine_similarity)))
                                    logging.info(
                                        f"Epoch {epoch + 1}/{EPOCHS}, "
                                        f"Minimization parameter: {(1 - torch.mean(torch.abs(cosine_similarity)))}, "
                                        f"target_idx: {target_idx}"
                                    )
                            loss = LAMBDA_ALIGN * align_loss + LAMBDA_CLS * cls_loss
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model_trained.parameters(), max_norm=7)
                            optimizer.step()

                            total_loss_epoch += loss.item()
                            cls_loss_epoch += cls_loss.item()
                            align_loss_epoch += align_loss.item()

                        n_batches = len(dataset_loader)
                        loss_history["total"].append(total_loss_epoch / n_batches)
                        loss_history["cls"].append(cls_loss_epoch / n_batches)
                        loss_history["align"].append(align_loss_epoch / n_batches)
                        logging.info(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss_history['total'][-1]:.4f}")
                        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss_history['total'][-1]:.4f}")

                    acc_after, precision_after, recall_after, f1_after = evaluate_accuracy(model_trained, validation_loader)
                    
                    results_legacy_after, avg_confidences_legacy_after, class_count_legacy_after, acc_legacy_after = predict_from_loader(validation_loader, model_trained, TARGET_IDX_LIST)
                    
                    print("Computing the tcav scores after can take a while stand by")
                    tcav_after = [compute_tcav_score(model_trained, layer_name, cav, class_loader, idx)
                                for cav, class_loader, idx in zip(cav_vectors, class_dataloaders, TARGET_IDX_LIST)]
                    avg_conf_after = compute_avg_confidence(model_trained, validation_loader, TARGET_IDX_LIST)
                    logging.info(f"Accuracy After: {acc_after:.4f}")
                    logging.info(f"Precision After: {precision_after:.4f}")
                    logging.info(f"Recall After: {recall_after:.4f}")
                    logging.info(f"F1 Score After: {f1_after:.4f}")
                    logging.info(f"Average Confidence After: {avg_conf_after}")
                    logging.info(f"TCAV Score after : {tcav_after}")
                    print(f"Accuracy After: {acc_after:.4f}, tcav_after: {tcav_after}")

                    stats = {
                        "Layer Name": layer_name,
                        "Lambda Classification": LAMBDA_CLS,
                        "Lambda Alignment": LAMBDA_ALIGN,
                        "Accuracy Before": round(acc_before, 3),
                        "Accuracy After": round(acc_after, 3),
                        "Precision Before": round(precision_before, 3),
                        "Precision After": round(precision_after, 3),
                        "Recall Before": round(recall_before, 3),
                        "Recall After": round(recall_after, 3),
                        "F1 Before": round(f1_before, 3),
                        "F1 After": round(f1_after, 3),
                        "Legacy Accuracy Before": round(acc_legacy_before, 3),
                        "Legacy Accuracy After": round(acc_legacy_after, 3),
                        "Class count  Before": class_count_legacy_before[i],
                        "Class count After": class_count_legacy_after[i]
                    }

                    for i, class_name in enumerate(TARGET_CLASS_LIST):
                        stats[f"TCAV Before ({class_name})"] = round(tcav_before[i], 3)
                        stats[f"TCAV After ({class_name})"] = round(tcav_after[i], 3)
                        stats[f"Avg Conf {class_name} Before"] = round(avg_conf_before[i], 3)
                        stats[f"Avg Conf {class_name} After"] = round(avg_conf_after[i], 3)

                    classificationloss_filename = os.path.join(RESULTS_PATH, f"loss_{BASE_MODEL}_{layer_name}_{LAMBDA_ALIGN}.pdf")
                    alignmentloss_filename = os.path.join(RESULTS_PATH, f"alignment_loss_{BASE_MODEL}_{layer_name}_{LAMBDA_ALIGN}.pdf")
                    total_loss = os.path.join(RESULTS_PATH, f"total_loss_{BASE_MODEL}_{layer_name}_{LAMBDA_ALIGN}.pdf")
                    plot_loss_figure(loss_history["total"], loss_history["align"], loss_history["cls"], EPOCHS,
                                    classificationloss_filename, alignmentloss_filename, total_loss)
                    statistic_filename = os.path.join(RESULTS_PATH, f"statistics_{BASE_MODEL}.csv")
                    save_statistics(stats, statistic_filename)
                    logging.info(f"Training completed for Lambda Align: {LAMBDA_ALIGN}, Layer: {layer_name}")
                    modelsave_filename = os.path.join(RESULTS_PATH, f"loss_{BASE_MODEL}_{layer_name}_{LAMBDA_ALIGN}.pth")
                    torch.save(model_trained.state_dict(), modelsave_filename)
            except Exception as e:
                logging.error(f"Error during training with Lambda Align {LAMBDA_ALIGN}: {e}")
                print(f"Error during training with Lambda Align {LAMBDA_ALIGN}: {e}")

    except Exception as e:
        logging.error(f"Error in main function: {e}, layer_name :{layer_name}, LAMBDA_ALIGN{LAMBDA_ALIGN} ")
    logging.info("Main function completed.")



if __name__ == "__main__":
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Override model name and model path")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--model_path", type=str, default=None, help="Specify a model path to override the default path")
    args = parser.parse_args()

    # Check if both parameters are provided
    if not args.model_name or not args.model_path:
        print("Error: Both --model_name and --model_path must be provided.")
    else:
        BASE_MODEL = args.model_name.strip().lower()
        BASE_MODEL_PATH = args.model_path.strip()

    # Override the model name if provided
    if args.model_name:
        BASE_MODEL = args.model_name.strip().lower()
        BASE_MODEL = BASE_MODEL.strip().lower()
        MODEL_PATH = get_model_weight_path(BASE_MODEL, BASE_MODEL_PATH)

        # Configure logging
        RESULTS_PATH = './results/' + BASE_MODEL + '/'
        os.makedirs(RESULTS_PATH, exist_ok=True)
        log_filename = f"./results/{BASE_MODEL}/audit_trail_{BASE_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging = Logger_Singleton(log_filename)
        logging.info("Script started.")
        IMAGE_SIZE = get_base_model_image_size(BASE_MODEL)
        # load model
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
        MODEL.to(DEVICE)
        # Get all bottleneck layers
        LAYER_NAMES = get_model_layers(MODEL)
        logging.info(f"Layer names present in this model are {LAYER_NAMES}")
        LAYER_NAMES = get_model_layers(MODEL)[2:]
        logging.info(f"Layer names trained now in this model are {LAYER_NAMES}")
        NUM_CLASSES = get_num_classes(CLASSIFICATION_DATA_BASE_PATH)
    else:
        # Load the model
        MODEL = DeepCNN(num_classes=NUM_CLASSES)
        MODEL.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=True))
        MODEL.to(DEVICE)
        LAYER_NAMES = ["conv_block4.0"]

    # Transformations
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    VALID_TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info(f"Model overridden with: {args.model_name}")
    logging.info(f"Model path: {BASE_MODEL_PATH}")
    logging.info(f"Hyperparameters - Learning Rate: {LEARNING_RATE}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Device: {DEVICE}")
    logging.info(f"Target Classes: {TARGET_CLASS_LIST}, Lambda Aligns: {LAMBDA_ALIGNS}")

    try:
        print("Calling methods get_class_folder_dicts")
        train_folders, valid_folders, class_names = get_class_folder_dicts(CLASSIFICATION_DATA_BASE_PATH)
        TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]
        print("Loading train datasets stand by")
        train_dataset = MultiClassImageDataset(train_folders, transform=TRAIN_TRANSFORM)
        val_dataset = MultiClassImageDataset(valid_folders, transform=VALID_TRANSFORM)
        print("Loading val datasets stand by")
        dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(CLASSIFICATION_DATA_BASE_PATH, class_name + "/train"),
                                                              transform=VALID_TRANSFORM), batch_size=BATCH_SIZE) for class_name in TARGET_CLASS_LIST]
        print("Loading concept datasets stand by")
        concept_loader_list = [DataLoader(SingleClassDataLoader(path, transform=VALID_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True) for path in CONCEPT_FOLDER_LIST]
        print("Loading random datasets stand by")
        random_loader = DataLoader(SingleClassDataLoader(RANDOM_FOLDER, transform=VALID_TRANSFORM), batch_size=BATCH_SIZE, shuffle=True)
        logging.info("Data preparation completed successfully.")
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
    main()
    logging.info("Script execution finished.")

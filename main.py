import torch
import os
import csv
import numpy as np
from urllib import request
from torchvision import models
from src.tcav_pipeline import tcav_pipeline

# Define models and layers to evaluate
MODEL_CONFIGS = [
    {"name": "googlenet", "model": models.googlenet, "layers": ["inception4a", "inception5b"]},
    {"name": "resnet50", "model": models.resnet50, "layers": ["layer3", "layer4"]},
]

# Define paths
CONCEPTS_FOLDER = "./data/concept"
RANDOM_FOLDER = "./data/random"
DATASET_FOLDER = "./data/dataset/zebra"
RESULTS_FILE = "./results/tcav_scores.csv"

if __name__ == "__main__":
    # Load ImageNet class index to label mapping
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with request.urlopen(url) as f:
        class_labels = [line.strip() for line in f.readlines()]

    target_label = b"zebra"
    target_class = class_labels.index(target_label)

    # Prepare CSV file for saving results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Layer", "Concept", "Avg_TCAV_Score", "Std_TCAV_Score", "Max_TCAV", "Min_TCAV"])

        # Iterate over models
        for config in MODEL_CONFIGS:
            model_name = config["name"]
            model_fn = config["model"]
            layers = config["layers"]

            # Load pre-trained model
            model = model_fn(pretrained=True).to("cuda" if torch.cuda.is_available() else "cpu")

            # Iterate over layers
            for layer_name in layers:
                # Iterate over concepts
                for concept in os.listdir(CONCEPTS_FOLDER):
                    concept_folder = os.path.join(CONCEPTS_FOLDER, concept)
                    if not os.path.isdir(concept_folder):
                        continue  # Skip if not a directory

                    # Run TCAV pipeline
                    tcav_scores = tcav_pipeline(
                        model=model,
                        layer_name=layer_name,
                        concept_folder=concept_folder,
                        random_folder=RANDOM_FOLDER,
                        target_class=target_class,
                        dataset_folder=DATASET_FOLDER,
                    )

                    # Compute statistics
                    avg_tcav = np.mean(tcav_scores)
                    std_tcav = np.std(tcav_scores)
                    max_tcav = np.max(tcav_scores)
                    min_tcav = np.min(tcav_scores)

                    # Write result to CSV
                    writer.writerow([model_name, layer_name, concept, avg_tcav, std_tcav, max_tcav, min_tcav])
                    print(f"Saved results for {model_name}, {layer_name}, {concept}")

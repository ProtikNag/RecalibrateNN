import torch
from torchvision import models

from src.tcav_pipeline import tcav_pipeline

if __name__ == "__main__":
    # Load a pre-trained model
    model = models.googlenet(pretrained=True).to("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths
    concept_folder = "./data/concept/striped"
    random_folder = "./data/random"
    dataset_folder = "./data/dataset"
    layer_name = "inception5b"

    target_class = 243

    # Run TCAV pipeline
    tcav_score = tcav_pipeline(
        model=model,
        layer_name=layer_name,
        concept_folder=concept_folder,
        random_folder=random_folder,
        target_class=target_class,
        dataset_folder=dataset_folder,
    )

    print(f"TCAV score for class {target_class}: {tcav_score}")
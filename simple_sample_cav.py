import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import get_base_model_image_size, train_cav
#https://link.springer.com/chapter/10.1007/978-3-031-44067-0_26
#Revealing Similar Semantics Inside CNNs: An Interpretable Concept-based Comparison of Feature Spaces

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SampledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_image_paths(folder_path):
    image_paths = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def sample_without_replacement(image_paths, sample_size, rng):
    if len(image_paths) < sample_size:
        raise ValueError(
            f"Expected at least {sample_size} images, but found only {len(image_paths)} images."
        )
    return rng.sample(image_paths, sample_size)

def sample_with_replacement(image_paths, sample_size, rng):
    return [rng.choice(image_paths) for _ in range(sample_size)]



def register_activation_hook(model, layer_name, activation_store):
    def hook(_module, _inputs, output):
        activation_store[layer_name] = output

    layer = model.get_submodule(layer_name)
    return layer.register_forward_hook(hook)


def extract_activations(model, loader, layer_name, activation_store):
    collected_activations = []
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(DEVICE)
            _ = model(images)
            layer_output = activation_store[layer_name]
            collected_activations.append(layer_output.view(images.size(0), -1).cpu().numpy())
    return np.vstack(collected_activations)


def compute_cav_for_sample(
    model,
    layer_name,
    conceptlist,
    random_list,
    transform,
    batch_size,
    classifier_type,
):
    activation_store = {}
    hook_handle = register_activation_hook(model, layer_name, activation_store)

    try:
        concept_loader = DataLoader(
            SampledImageDataset(conceptlist, transform=transform),
            batch_size=batch_size,
            shuffle=False,
        )
        random_loader = DataLoader(
            SampledImageDataset(random_list, transform=transform),
            batch_size=batch_size,
            shuffle=False,
        )

        concept_activations = extract_activations(model, concept_loader, layer_name, activation_store)
        random_activations = extract_activations(model, random_loader, layer_name, activation_store)
        cav = train_cav(
            concept_activations,
            random_activations,
            orthogonal=False,
            classifier_type=classifier_type,
        )
    finally:
        hook_handle.remove()

    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)

def build_sampled_cav_list(
    model,
    layer_name,
    concept_folder,
    random_folder,
    num_cavs=30,
    sample_size=30,
    batch_size=8,
    seed=132,
    classifier_type="LinearSVC",
    image_size=224,
):
    
    transform = build_transform(image_size)

    concept_paths = get_image_paths(concept_folder)
    random_paths = get_image_paths(random_folder)

    cav_list = []
    conceptlist = []
    random_list = []

    for _ in range(num_cavs):
        rng = random.Random(seed)
        if(sample_size <= len(concept_paths)//2 or sample_size > len(random_paths)//2):
            current_conceptlist = sample_without_replacement(concept_paths, sample_size, rng)
            current_random_list = sample_without_replacement(random_paths, sample_size, rng)
        else:
            current_conceptlist = sample_with_replacement(concept_paths, sample_size, rng)
            current_random_list = sample_with_replacement(random_paths, sample_size, rng)


        cav = compute_cav_for_sample(
            model=model,
            layer_name=layer_name,
            conceptlist=current_conceptlist,
            random_list=current_random_list,
            transform=transform,
            batch_size=batch_size,
            classifier_type=classifier_type,
        )
        conceptlist.append(current_conceptlist)
        random_list.append(current_random_list)
        cav_list.append(cav)
        seed += 1  # Increment the index for the next iteration to ensure different samples
    return conceptlist, random_list, cav_list


def cosine_similarity(cav_list):
    cosine_similarities = []
    if len(cav_list) < 2:
        return cosine_similarities, float("nan")

    normalized_cavs = [F.normalize(cav.detach().flatten(), p=2, dim=0) for cav in cav_list]
    for index_a in range(len(normalized_cavs)):
        for index_b in range(index_a + 1, len(normalized_cavs)):
            similarity = torch.dot(normalized_cavs[index_a], normalized_cavs[index_b]).item()
            cosine_similarities.append(similarity)

    mean_cosine_similarity = float(np.mean(cosine_similarities))
    return cosine_similarities, mean_cosine_similarity


def compute_consistency(mean_cosine_similarity, num_cavs):
    if num_cavs < 2 or np.isnan(mean_cosine_similarity):
        return float("nan")
    if mean_cosine_similarity == 0:
        return float("inf")
    return 2 * num_cavs / (num_cavs * (num_cavs - 1) * mean_cosine_similarity)


def write_experiment_results(output_path, results):
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "concept_samples",
            "random_samples",
            "num_cavs",
            "mean_cosine_similarity",
            "consistency",
        ])
        for result in results:
            writer.writerow([
                result["concept_samples"],
                result["random_samples"],
                result["num_cavs"],
                result["mean_cosine_similarity"],
                result["consistency"],
            ])


def run_sample_size_experiment(
    model,
    layer_name,
    concept_folder,
    random_folder,
    start_sample_size,
    num_cavs,
    batch_size,
    seed,
    classifier_type,
    image_size,
    step_size=10,
):
    concept_paths = get_image_paths(concept_folder)
    random_paths = get_image_paths(random_folder)
    max_sample_size = min(len(concept_paths), len(random_paths))
    results = []

    for current_sample_size in range(start_sample_size, max_sample_size + 1, step_size):
        conceptlist, random_list, cav_list = build_sampled_cav_list(
            model=model,
            layer_name=layer_name,
            concept_folder=concept_folder,
            random_folder=random_folder,
            num_cavs=num_cavs,
            sample_size=current_sample_size,
            batch_size=batch_size,
            seed=seed,
            classifier_type=classifier_type,
            image_size=image_size,
        )
        _cosine_similarities, mean_cosine_similarity = cosine_similarity(cav_list)
        consistency = compute_consistency(mean_cosine_similarity, len(cav_list))
        results.append({
            "Number of samples": current_sample_size,
            "concept_samples": len(conceptlist[0]) if conceptlist else 0,
            "random_samples": len(random_list[0]) if random_list else 0,
            "num_cavs": len(cav_list),
            "mean_cosine_similarity": mean_cosine_similarity,
            "consistency": consistency,
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Sample concept/random images and compute a CAV.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model .pth file")
    parser.add_argument("--layer-name", required=True, help="Layer name used to extract activations")
    parser.add_argument("--concept-folder", required=True, help="Folder containing concept images")
    parser.add_argument("--random-folder", required=True, help="Folder containing random images")
    parser.add_argument("--base-model", default="resnet50", help="Base model name for image sizing")
    parser.add_argument("--num-cavs", type=int, default=30, help="Number of CAV vectors to compute")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of images to sample")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for activation extraction")
    parser.add_argument("--seed", type=int, default=132, help="Random seed for reproducibility")
    parser.add_argument(
        "--classifier-type",
        default="LinearSVC",
        choices=["LinearSVC", "SGDClassifier", "LogisticRegression"],
        help="Linear classifier used to train the CAV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    model = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)

    image_size = get_base_model_image_size(args.base_model)
    results = run_sample_size_experiment(
        model=model,
        layer_name=args.layer_name,
        concept_folder=args.concept_folder,
        random_folder=args.random_folder,
        start_sample_size=args.sample_size,
        num_cavs=args.num_cavs,
        batch_size=args.batch_size,
        seed=args.seed,
        classifier_type=args.classifier_type,
        image_size=image_size,
    )
    output_path = os.path.join(os.getcwd(), f"concept_data_check_{args.base_model}.csv")
    write_experiment_results(output_path, results)

    print(f"Saved experiment results to: {output_path}")
    for result in results:
        print(
            f"sample_size={result['concept_samples']}, "
            f"num_cavs={result['num_cavs']}, "
            f"mean_cosine_similarity={result['mean_cosine_similarity']}, "
            f"consistency={result['consistency']}"
        )


if __name__ == "__main__":
    main()
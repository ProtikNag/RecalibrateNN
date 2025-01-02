import numpy as np
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression

from src.activation_extractor import ActivationExtractor
from src.concept_dataset import ConceptDataset


def train_cav(concept_activations, random_activations):
    """
    Train a logistic regression model to separate concept and random activations
    """

    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    clf = LogisticRegression(random_state=0).fit(X, y)
    clf.fit(X, y)

    return clf.coef_.squeeze()


def compute_tcav_score(model, layer_name, cav_vector, inputs, target_class):
    """
    Compute the TCAV score by projecting the gradients onto the CAV vector
    """

    model.eval()
    extractor = ActivationExtractor(model, layer_name)
    extractor.register_hook()

    inputs.requires_grad = True
    logits = model(inputs)
    target_logits = logits[:, target_class].sum()
    target_logits.backward()

    gradients = extractor.activations.grad
    projections = (gradients.view(gradients.size(0), -1) @ torch.tensor(cav_vector).float())
    tcav_score = (projections > 0).float().mean().item()

    extractor.unregister_hook()
    return tcav_score


def tcav_pipeline(
        model,
        layer_name,
        concept_folder,
        random_folder,
        target_class,
        dataset_folder,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Data preparation
    trasform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    concept_dataset = ConceptDataset(concept_folder, transform=trasform)
    random_dataset = ConceptDataset(random_folder, transform=trasform)
    dataset = ConceptDataset(dataset_folder, transform=trasform)

    concept_loader = DataLoader(concept_dataset, batch_size=batch_size, shuffle=True)
    random_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get activations for concept and random images
    model.eval()
    concept_activations = []
    random_activations = []
    extractor = ActivationExtractor(model, layer_name)
    extractor.register_hook()

    with torch.no_grad():
        for imgs, _ in concept_loader:
            imgs = imgs.to(device)
            activations = extractor(imgs).cpu().view(imgs.size(0), -1).numpy()
            concept_activations.append(activations)

        for imgs, _ in random_loader:
            imgs = imgs.to(device)
            activations = extractor(imgs).cpu().view(imgs.size(0), -1).numpy()
            random_activations.append(activations)

    extractor.unregister_hook()

    concept_activations = np.vstack(concept_activations)
    random_activations = np.vstack(random_activations)


    # Train CAV
    cav_vector = train_cav(concept_activations, random_activations)

    # Compute TCAV score
    tcav_scores = []
    for imgs, labels in dataset_loader:
        imgs = imgs.to(device)
        score = compute_tcav_score(model, layer_name, cav_vector, imgs, target_class)
        tcav_scores.append(score)

    return tcav_scores
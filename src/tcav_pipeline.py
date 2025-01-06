import numpy as np
import torch.cuda
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision import transforms

from src.activation_extractor import ActivationExtractor
from src.concept_dataset import ConceptDataset


def train_cav(concept_activations, random_activations):
    """
    Train a logistic regression model to separate concept and random activations.
    Returns a normalized CAV vector.
    """
    X = np.vstack((concept_activations, random_activations))
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    clf = LogisticRegression(random_state=0).fit(X, y)
    cav_vector = clf.coef_.squeeze()
    cav_vector /= np.linalg.norm(cav_vector)  # Normalize the CAV vector

    return cav_vector


def compute_tcav_score(model, layer_name, cav_vector, inputs, target_class):
    """
    Compute the TCAV score by projecting the gradients onto the CAV vector.
    """

    model.eval()
    extractor = ActivationExtractor(model, layer_name)
    extractor.register_hook()

    inputs = inputs.to(next(model.parameters()).device)
    inputs.requires_grad_()
    logits = model(inputs)
    target_logits = logits[:, target_class].sum()

    gradients = torch.autograd.grad(outputs=target_logits, inputs=extractor.activations,
                                    grad_outputs=torch.ones_like(target_logits),
                                    retain_graph=True)[0]

    projections = (gradients.view(gradients.size(0), -1) @ torch.tensor(cav_vector).float().to(gradients.device))
    tcav_score = (projections > 0).float().mean().item()

    extractor.unregister_hook()
    del gradients

    return tcav_score


def tcav_pipeline(
        model,
        layer_name,
        concept_folder,
        random_folder,
        target_class,
        dataset_folder,
        batch_size=8,
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


    for imgs in concept_loader:
        imgs = imgs.to(device)
        _ = extractor(imgs)  # Perform forward pass to populate activations
        activations = extractor.activations.cpu().view(imgs.size(0), -1).detach().numpy()
        print(f"Shape of concept activations: {activations.shape}")
        concept_activations.append(activations)

    for imgs in random_loader:
        imgs = imgs.to(device)
        _ = extractor(imgs)  # Perform forward pass to populate activations
        activations = extractor.activations.cpu().view(imgs.size(0), -1).detach().numpy()
        random_activations.append(activations)

    extractor.unregister_hook()

    concept_activations = np.vstack(concept_activations)
    random_activations = np.vstack(random_activations)

    # Train CAV
    cav_vector = train_cav(concept_activations, random_activations)
    print(f"Shape of CAV vector: {cav_vector.shape}")

    # Compute TCAV score
    tcav_scores = []
    for imgs in dataset_loader:
        imgs = imgs.to(device)
        score = compute_tcav_score(model, layer_name, cav_vector, imgs, target_class)
        tcav_scores.append(score)

    return tcav_scores

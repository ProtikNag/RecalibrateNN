import numpy as np
import torch
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


def get_layer_activations(model, layer_name, inputs):
    """
    Extract activations for a specific layer given the inputs.
    """
    extractor = ActivationExtractor(model, layer_name)
    extractor.register_hook()
    _ = model(inputs)
    activations = extractor.activations.view(inputs.size(0), -1)
    extractor.unregister_hook()
    return activations
# RecalibrateNN: Improving TCAV Interpretability


## Overview

RecalibrateNN is a project that implements Testing with Concept Activation Vectors (TCAV) to interpret deep learning models using human-friendly concepts. This repository allows users to analyze, retrain, and compare TCAV scores for a given model, focusing on concept alignment with latent representations.


## Features

* Extracts layer activations from deep neural networks.
* Computes Concept Activation Vectors (CAVs) using logistic regression.
* Evaluates the TCAV score to measure conceptual sensitivity.
* Retrains a neural network to improve alignment with desired concepts.
* Compares TCAV scores before and after retraining.

## Repository Structure

```
RecalibrateNN
├── data
│   ├── concept            # Concept images for TCAV computation
│   │   ├── dotted
│   │   ├── striped
│   │   ├── zigzagged
│   ├── dataset            # Dataset used for model evaluation
│   │   ├── zebra
│   ├── random             # Random images for control comparisons
├── results                # Stores results from experiments
│   ├── retrained_model.pth
│   ├── tcav_score.csv
├── src                    # Core implementation
│   ├── activation_extractor.py  # Extracts activations from target layers
│   ├── calculate_cav.py   # Trains and computes CAV vectors
│   ├── concept_dataset.py # Dataset loader for concept images
│   ├── tcav_pipeline.py   # Pipeline for executing TCAV process
├── .gitignore             # Ignore unnecessary files
├── main.py                # Entry point for running experiments
├── requirements.txt       # Required dependencies
├── retrain_comparison.py  # Main script for retraining and comparison
├── retrain_nn.py          # Model retraining script
├── retrain_nn_new_version.py  # Alternative retraining implementation
```


## Setup

Ensure you have Python 3.8+ installed. Run the following command to install dependencies:

```aiignore
pip install -r requirements.txt
```

## Prepare Data

Ensure the required datasets are available in the data/ folder:

* Concept Data: Images that define the concept (e.g., striped, dotted, zigzagged).

* Dataset: Target class images (e.g., zebra).

* Random Data: Unrelated images for CAV training.

## Running the Experiment

To evaluate TCAV scores before and after retraining:

```aiignore
python retrain_comparison.py
```

This script:

* Loads a pre-trained GoogLeNet model.

* Extracts activations from the selected layer (inception5b).

* Computes the initial TCAV score.

* Trains the model with a latent space alignment loss.

* Computes the final TCAV score after retraining.

* Saves the retrained model.


## Expected Outputs

* Console Output: Displays TCAV scores before and after retraining.

* Loss Curve: Plots loss reduction over training epochs.

* TCAV Score Comparison: Bar chart of initial vs. final TCAV scores.

* Model Checkpoint: Stored in results/retrained_model.pth.

## Key Components

### Activation Extraction
Extracts activations from a chosen layer:
```aiignore
from src.activation_extractor import ActivationExtractor
extractor = ActivationExtractor(model, 'inception5b')
extractor.register_hook()
features = extractor.forward(images)
extractor.unregister_hook()
```

### Compute CAVs

Trains a logistic regression model to separate concept vs. random activations:

```aiignore
from src.calculate_cav import train_cav
cav_vector = train_cav(concept_activations, random_activations)
```



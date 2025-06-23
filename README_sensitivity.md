# Sensitivity Analysis and TCAV Utilities

This document provides an overview of the functionality and purpose of the following files in the project:

1. `util_sensitivity_compute.py`
2. `utils.py`
3. `tcav_utils.py`

These files collectively implement sensitivity analysis, Concept Activation Vector (CAV) training, and Testing with CAVs (TCAV) for interpretability and model evaluation.

---

## 1. `util_sensitivity_compute.py`

This script is the main driver for computing sensitivity scores and TCAV scores for a given model. It evaluates the sensitivity of a model's predictions to specific concepts and layers.

### Key Features:
- **Model Loading**: Loads the original and modified models using paths provided via command-line arguments.
- **Layer Hooking**: Registers hooks to capture activations for specified layers.
- **CAV Training**: Computes CAVs for concept and random activations.
- **Sensitivity Score Computation**: Calculates sensitivity scores for each image in the dataset.
- **TCAV Score Computation**: Computes TCAV scores for specified target classes.
- **CSV Logging**: Saves sensitivity scores and TCAV scores to a CSV file for further analysis.

### How to Run:
```bash
python util_sensitivity_compute.py --org_model_path <path_to_original_model> \
                                   --modified_model_path <path_to_modified_model> \
                                   --model_name <model_name> \
                                   --layer_name <layer_name>
```

### Key Functions:
- `get_activation(layer_name)`: Registers hooks to capture activations for a specific layer.
- `util_compute_cav`: Computes CAVs for concept and random activations.
- `util_compute_sensitivity_score`: Computes sensitivity scores for a given layer and target class.
- `util_compute_tcav_score_from_sensitivity`: Aggregates sensitivity scores to compute TCAV scores.

---

## 2. `utils.py`

This file contains utility functions for model evaluation, dataset loading, and CAV training. It is a core component used across multiple scripts.

### Key Features:
- **Dataset Utilities**: Functions to dynamically load datasets and class folders.
- **Model Utilities**: Functions to load models, retrieve layers, and compute activations.
- **CAV Training**: Implements training of CAVs using classifiers like `LinearSVC`, `SGDClassifier`, and `LogisticRegression`.
- **Evaluation Metrics**: Computes accuracy, precision, recall, and F1 scores for model predictions.
- **Visualization**: Functions to plot loss curves and save statistics.

### Key Functions:
- `get_num_classes(base_path)`: Dynamically determines the number of classes in a dataset.
- `get_model_layers(model)`: Retrieves all layers of a model for analysis.
- `train_cav(concept_activations, random_activations, orthogonal=False, classifier_type='LinearSVC')`: Trains a CAV using concept and random activations.
- `evaluate_accuracy(model, loader)`: Evaluates the model's accuracy and other metrics on a given dataset.
- `plot_loss_figure`: Plots and saves loss curves for training and validation.

---

## 3. `tcav_utils.py`

This file provides specialized utilities for TCAV computation, including CAV training, sensitivity score computation, and TCAV score calculation.

### Key Features:
- **Layer-Specific Analysis**: Computes activations and gradients for specific layers.
- **CAV Training**: Trains CAVs for specified layers and concepts.
- **TCAV Score Computation**: Calculates TCAV scores based on gradients and activations.
- **Sensitivity Analysis**: Computes sensitivity scores for individual images.

### Key Functions:
- `util_compute_cav(model, loader_positive, loader_random, layer_name, activation, orthogonal=False, dump_cav=True)`: Trains a CAV for a specific layer using concept and random activations.
- `util_compute_tcav_score(model, layer_name, cav_vector, dataset_loader, target_idx, activation)`: Computes TCAV scores for a given layer and target class.
- `util_compute_sensitivity_score(model, layer_name, cav_vector, dataset_loader, target_idx, activation)`: Computes sensitivity scores for a given layer and target class.
- `uils_getlayers(model_name)`: Retrieves predefined layer names for popular models like `vgg16`, `resnet50`, and `inception_v3`.

---

## Workflow Overview

1. **Dataset Preparation**:
   - Organize your dataset into concept and random folders.
   - Ensure the dataset structure matches the expected format in `config.py`.

2. **Model Preparation**:
   - Provide paths to the original and modified models.
   - Specify the target layer and model name.

3. **Run Sensitivity Analysis**:
   - Use `util_sensitivity_compute.py` to compute sensitivity and TCAV scores.
   - Results are saved in CSV files for further analysis.

4. **Visualize Results**:
   - Use the sensitivity scores and TCAV scores to interpret the model's behavior.
   - Plot loss curves and other metrics using functions in `utils.py`.

---

## Example Use Case

To compute sensitivity and TCAV scores for the `vgg16` model on the layer `features.17`:

```bash
python util_sensitivity_compute.py --org_model_path "/path/to/vgg16.pth" \
                                   --modified_model_path "/path/to/modified_vgg16.pth" \
                                   --model_name "vgg16" \
                                   --layer_name "features.17"
```

This will:
1. Load the original and modified models.
2. Compute CAVs for the specified layer.
3. Calculate sensitivity and TCAV scores for the target classes.
4. Save the results to a CSV file in the `results/` directory.

---

## Example Usage

### Scenario: Evaluating Sensitivity for a Custom Dataset

1. **Prepare Dataset**:
   - Organize your dataset into folders:
     ```
     dataset/
     ├── concept/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── random/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ```

2. **Train CAVs and Compute Sensitivity**:
   Run the following command:
   ```bash
   python util_sensitivity_compute.py --org_model_path "models/original_model.pth" \
                                      --modified_model_path "models/modified_model.pth" \
                                      --model_name "resnet50" \
                                      --layer_name "layer4.2"
   ```

3. **Analyze Results**:
   - Open the generated CSV file in the `results/` directory.
   - Use the sensitivity and TCAV scores to interpret the model's behavior.

---

## Notes

- Ensure that the dataset paths and model paths are correctly configured in `config.py` or `config_modified.py`.
- Use the `Logger_Singleton` class for consistent logging across scripts.
- For large datasets or models, computations may take significant time. Use a GPU for faster execution.

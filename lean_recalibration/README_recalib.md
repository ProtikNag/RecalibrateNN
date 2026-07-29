# RecalibrateNN

`RecalibrateNN` is a PyTorch-based project designed to recalibrate a pre-trained neural network (GoogleNet) for binary classification, with a focus on aligning model predictions with a specific concept (e.g., "stripes" for zebra classification) using Concept Activation Vectors (CAVs) and TCAV (Testing with CAVs) scores. The project fine-tunes a specific layer of the model to balance classification accuracy and concept alignment, making it a useful tool for interpretability and controlled model adjustment experiments.

This README provides a step-by-step guide to understanding, setting up, and running the project, as well as instructions for customizing it for your own experiments.

## Project Structure

```
RecalibrateNN/
├── data                   
├── results               
├── .gitignore              
├── config.py               
├── custom_dataloader.py    
├── main.py                 
├── README.md               
├── requirements.txt        
└── utils.py                
```


## Purpose

The project aims to:
1. Fine-tune a pre-trained GoogleNet model for binary classification (e.g., zebra vs. non-zebra).
2. Use CAVs to represent a concept (e.g., "stripes") and align the model's intermediate layer activations with this concept.
3. Evaluate the model's performance using accuracy and TCAV scores before and after recalibration.
4. Provide a framework for experimenting with concept-based model adjustments.

## Prerequisites

- Python 3.8+
- A CUDA-enabled GPU (optional but recommended for faster training; CPU fallback is available)
- Basic familiarity with PyTorch and deep learning concepts

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/RecalibrateNN.git
cd RecalibrateNN
```

### 2. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```aiignore
pip install -r requirements.txt
```

The key dependencies include:
* `torch` and `torchvision` for model training and image processing
* `numpy` for numerical operations
* `matplotlib` for plotting loss curves
* `scikit-learn` for training CAVs with LinearSVC
* `Pillow` for image loading

### 3. Preparing the Data

The `data/` folder is not included in the repository due to size constraints. You’ll need to set it up manually for binary classification. Here’s how:

Directory Structure
Create the following structure under RecalibrateNN/data/:

```
data/ 
├── binary_classification/
│   ├── class1/
│   │   ├── train/       
│   │   └── valid/       
│   └── class2/          
│       ├── train/       
│       └── valid/       
├── concept/ 
│  ├── stripes_fake/    
│  └── random/         

```

### 4. Verify Configuration

Open config.py and adjust paths or hyperparameters if needed:

* `BINARY_CLASSIFICATION_BASE`: Set to `"./data/binary_classification/"`.
* `CONCEPT_FOLDER`: Set to `"./data/concept/stripes_fake"`.
* `RANDOM_FOLDER`: Set to `"./data/concept/random"`.
* `ZEBRA_CLASS_NAME`: Set to the name of your positive class (e.g., "zebra").
* `RESULTS_PATH`: Where the retrained model will be saved (default: `"./results/retrained_model.pth"`).
* `Hyperparameters` like `LEARNING_RATE`, `EPOCHS`, `BATCH_SIZE`, `LAMBDA_ALIGN`, etc., can be tuned for your experiment.

### 5. Running the Project

Command \
Run the main script:

```bash
#If running in Srikanth platform set the environment variable PLATFORM="Srikanth" using  export PLATFORM="Srikanth"
python main.py
```

### 6. File Details

`config.py`
* Defines hyperparameters, model, and paths.
* Sets random seeds for reproducibility.

``main.py``
* Core script for data loading, CAV computation, training, and evaluation.
* Implements the recalibration logic with a custom loss function.

`custom_dataloader.py`
* MultiClassImageDataset: Loads binary classification images with labels.
* ConceptDataset: Loads concept/random images without labels.

`utils.py`
* get_class_folder_dicts: Parses class folders automatically.
* train_cav: Trains the CAV using LinearSVC.
* cosine_similarity_loss: Computes alignment loss.
* evaluate_accuracy: Calculates model accuracy.

`plot_gradcam_compare.py`
```aiignore
python new_grad.py \
  --model_name vgg \
  --model_before_path ./results/vgg16_before.pth \
  --model_after_path ./results/vgg16_after.pth \
  --target_class zebra \
  --target_layer features.28

```

---

## Detailed README: `main_recalib_custom_by_loading_cav.py`

This script performs **CAV-guided recalibration** of a pretrained classifier in two modes:

1. **Per-class mode (default)**: trains one recalibrated model per target class.  
2. **Joint multi-class mode**: trains one shared recalibrated model for all enabled target classes.

It loads concept vectors from `CAVRegistry`, computes before/after metrics + TCAV, and saves model artifacts and summaries.

### CLI usage

```bash
python main_recalib_custom_by_loading_cav.py ^
  --model_name vgg16 ^
  --model_path <weights_root> ^
  --cav_store ./cav_store ^
  --config_file config/biased/vgg16.yaml ^
  --store_results ./results
```

Joint multi-class mode:

```bash
python main_recalib_custom_by_loading_cav.py ... --multiclass_recalibration_mode
```

Optional flags:
- `--use_layer_groups`: use layer groups from manifest instead of generated layer combinations.
- `--recalibrate_classes "0,2"`: enable only selected target-class positions.
- `--manifest_file <path>`: override default `<cav_store_root>/<model_name>_manifest.json`.

---

## End-to-end interaction flow

1. `__main__` parses CLI + config, loads model/data/CAV manifest.
2. `build_class_concepts_map()` maps class indices to concept names.
3. `main()` or `main_multiclass()` is selected based on `--multiclass_recalibration_mode`.
4. Before recalibration:
   - `_compute_metrics()`
   - `_load_target_cavs()` + `_compute_tcav_scores()`
   - `_save_before_metrics()`
5. Recalibration:
   - `_resolve_layer_combos()`
   - per-class path: `_recalibrate_one_class()` -> `_run_one_lambda()`
   - joint path: `_recalibrate_multiclass()` -> `_run_one_lambda_multiclass()`
6. After recalibration:
   - `_compute_after_metrics()`
   - `_create_summary_excel()`

---

## Method-by-method reference (usage + interactions)

### Reproducibility and data utilities

- `set_seed(seed=RANDOM_STATE)`
  - **Purpose:** sets random seed for Python, NumPy, and Torch (+CUDA determinism).
  - **Used by:** module init, `main()`, `main_multiclass()`, and each lambda sweep iteration.
  - **Interaction:** ensures reproducible model training and dataloader behavior.

- `worker_init_fn(worker_id)`
  - **Purpose:** deterministic seed init for dataloader workers.
  - **Used by:** DataLoader creation in `__main__`.
  - **Interaction:** complements `set_seed()` for multi-worker loading.

- `_extract_images_and_labels(batch)`
  - **Purpose:** normalizes loader batch format to `(images, labels_or_none)`.
  - **Used by:** `_compute_tcav_scores()`.
  - **Interaction:** allows TCAV computation to work with tuple/list or tensor-only batches.

### Activation capture

- `class ActivationHook`
  - `__init__(layer_name, store)`: binds one target layer name and a dict to write into.
  - `__call__(model, inputs, output)`: stores forward output for that layer into `store[layer_name]`.
  - **Used by:** `get_activation()`, then all hook-based methods.
  - **Interaction:** shared mechanism for both TCAV and alignment-loss feature access.

- `get_activation(layer_name, store)`
  - **Purpose:** factory returning `ActivationHook`.
  - **Used by:** `_compute_tcav_scores()`, `_run_one_lambda()`, `_run_one_lambda_multiclass()`.
  - **Interaction:** central hook registration helper.

### Metrics and CAV loading

- `_compute_metrics(model, validation_loader, target_idx_list, logging)`
  - **Purpose:** computes accuracy/precision/recall/F1 and confidence stats.
  - **Calls:** `evaluate_accuracy()`, `compute_avg_confidence()`.
  - **Used by:** before/after metric stages and lambda training summaries.

- `_load_target_cavs(registry, model_name, layer_names, concept_name, logging)`
  - **Purpose:** loads + L2-normalizes CAV per layer for one concept.
  - **Returns:** `{layer_name: cav_tensor}` (missing layers skipped with warning).
  - **Used by:** `_load_class_concept_cavs()`, before/after TCAV loops.

- `_load_class_concept_cavs(registry, model_name, layer_names, concept_list, logging)`
  - **Purpose:** loads CAVs for every concept of a class.
  - **Returns:** `{concept_name: {layer_name: cav_tensor}}`.
  - **Used by:** `_recalibrate_one_class()`, `_recalibrate_multiclass()`.

### Manifest and class-concept mapping

- `load_model_manifest(manifest_path)`
  - **Purpose:** reads per-model manifest JSON produced by CAV storage pipeline.
  - **Used by:** `__main__`.
  - **Interaction:** input for class->concept grouping.

- `build_class_concepts_map(manifest_data, class_names, target_idx_list)`
  - **Purpose:** maps target class indices to matching concept names by `class_name`.
  - **Returns:** `{target_idx: [concept_name, ...]}`.
  - **Used by:** `__main__`; consumed by both recalibration modes.

### TCAV and reporting

- `_compute_tcav_scores(model, target_cavs, loader, target_idx, logging)`
  - **Purpose:** computes TCAV score per layer for one target class + concept CAV set.
  - **Returns:** `{layer_name: tcav_score}`.
  - **Used by:** `_save_before_metrics()`, `_compute_after_metrics()`, pre-run TCAV in main flows.
  - **Interaction:** uses forward hooks from `get_activation()` and autograd sensitivity wrt class logit.

- `_save_before_metrics(...)`
  - **Purpose:** saves baseline metrics + TCAV into `accuracy_results_before.txt`.
  - **Calls:** `_compute_metrics()`, `_load_target_cavs()`, `_compute_tcav_scores()`.
  - **Used by:** `main()`, `main_multiclass()`.

- `_compute_after_metrics(best_models_per_class, ...)`
  - **Purpose:** loads best saved models and computes post-recalibration metrics + TCAV.
  - **Returns:** `(after_metrics, after_tcav)`.
  - **Calls:** `_compute_metrics()`, `_load_target_cavs()`, `_compute_tcav_scores()`.
  - **Used by:** `main()`, `main_multiclass()`.

- `_create_summary_excel(...)`
  - **Purpose:** writes consolidated before/after class metrics and average TCAV deltas.
  - **Output:** `recalibration_summary_<model>.xlsx` (CSV fallback on failure).
  - **Nested helper:** `_flatten_tcav(tcav_by_concept)` flattens concept-layer score dicts.
  - **Used by:** `main()`, `main_multiclass()`.

### Training core (per-class)

- `_run_one_lambda(...)`
  - **Purpose:** one full train/eval pass for a specific `(target_class, layer_combo, lambda_align)`.
  - **Loss:** `loss = lambda_align * align_loss + (1-lambda_align) * cls_loss`.
  - **Align loss:** weighted concept cosine-alignment over hooked activations, masked to target-class samples.
  - **Calls:** `_compute_metrics()`, `plot_loss_figure()` (optional).
  - **Used by:** `_recalibrate_one_class()`.

- `_recalibrate_one_class(...)`
  - **Purpose:** sweeps all `lambda_aligns` for one class and one layer combo.
  - **Calls:** `_load_class_concept_cavs()`, `_run_one_lambda()`.
  - **Returns:** `(run_metrics, layer_combo)` for aggregation in `main()`.

### Training core (joint multi-class)

- `_run_one_lambda_multiclass(...)`
  - **Purpose:** one full train/eval pass for joint class recalibration with one shared model.
  - **Align loss:** sums class-specific weighted concept losses, each masked by its class in batch.
  - **Calls:** `_compute_metrics()`, `plot_loss_figure()` (optional).
  - **Used by:** `_recalibrate_multiclass()`.

- `_recalibrate_multiclass(...)`
  - **Purpose:** for one layer combo, loads all enabled classes’ concept CAVs and sweeps lambdas.
  - **Calls:** `_load_class_concept_cavs()`, `_run_one_lambda_multiclass()`.
  - **Returns:** `(run_metrics, layer_combo)`.

### Layer-combo resolution and orchestration

- `_resolve_layer_combos(layer_names, use_groups, registry, model_name)`
  - **Purpose:** resolves layer combinations:
    - from manifest groups when `use_groups=True`, else
    - all combinations of size 1/2/3 from `layer_names`.
  - **Used by:** `main()`, `main_multiclass()`.

- `main(...)`
  - **Purpose:** default per-class orchestration pipeline.
  - **Calls:** `_compute_metrics()`, `_load_target_cavs()`, `_compute_tcav_scores()`, `_save_before_metrics()`,
    `_resolve_layer_combos()`, `_recalibrate_one_class()`, `_compute_after_metrics()`, `_create_summary_excel()`.
  - **Model selection:** tracks best model per class by validation accuracy.

- `main_multiclass(...)`
  - **Purpose:** joint recalibration orchestration.
  - **Calls:** same baseline/reporting methods as `main()`, but recalibration path uses
    `_recalibrate_multiclass()`.
  - **Model selection:** picks one best joint model and reuses it for post-metrics per enabled class.

### Script entrypoint

- `if __name__ == "__main__":`
  - **Purpose:** CLI parsing, environment setup, config/model/data loading, class-concept map setup, weight map setup, mode dispatch.
  - **Key interactions:**
    - `ConfigSingleton` drives hyperparameters and target classes.
    - `CAVRegistry` provides CAV lookup.
    - `get_class_folder_dicts()` and `MultiClassImageDataset` define train/validation loaders.
    - selects `main` vs `main_multiclass` and executes full pipeline.

---

## Output artifacts produced by this script

- Logs: `audit_trail_<model>_<timestamp>.log`
- Layer combo index: `layer_combinations_<model>.txt`
- Before metrics: `accuracy_results_before.txt`
- Per-class results CSV: `recalibration_combo_accuracy_<model>.csv`
- Joint-mode CSV: `recalibration_multiclass_combo_accuracy_<model>.csv`
- Saved models: `model_cls...pth` or `model_multiclass...pth`
- Optional loss plots: `loss_*.pdf`
- Summary report: `recalibration_summary_<model>.xlsx` (or CSV fallback)
"""
Central constants shared between the test-data generator and the test modules.

Keeping these names/paths in one place means the generator and the tests can never
drift out of sync (e.g. a concept name changed in one file but not the other).
All paths are built with ``pathlib``/``os.path`` so everything stays OS-agnostic.
"""

import os

# ---------------------------------------------------------------------------
# Folder roots (all relative to this test folder itself, wherever it is placed)
# ---------------------------------------------------------------------------
from helpers.path_discovery import discover_paths  # noqa: E402

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../<this test folder>
# REPO_ROOT is discovered by walking up looking for ConfigSingleton.py rather than
# assuming a fixed nesting depth, so this stays correct whether the test folder lives at
# `<repo_root>/test/`, `<repo_root>/unit_test/`, or `<repo_root>/lean_recalibration/unit_test/`.
REPO_ROOT, LEAN_RECALIB_DIR = discover_paths(TEST_ROOT)
DATA_ROOT = os.path.join(TEST_ROOT, "data")
CONFIG_ROOT = os.path.join(TEST_ROOT, "config")
RESULTS_ROOT = os.path.join(TEST_ROOT, "results")

IMAGES_PER_FOLDER = 5  # user requirement: 5 random images per concept/dataset folder

# ---------------------------------------------------------------------------
# Single-class scenario (concept_mode == "single")
# ---------------------------------------------------------------------------
SINGLE_CLASSES = ["cat", "dog"]
SINGLE_CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "classification", "single")
SINGLE_CONCEPT_DIR = os.path.join(DATA_ROOT, "concepts", "single")
SINGLE_CONCEPT_NAME = "stripes"
SINGLE_RANDOM_FOLDER = os.path.join(SINGLE_CONCEPT_DIR, "random")
SINGLE_MODEL_ROOT = os.path.join(DATA_ROOT, "model_weights_single")
SINGLE_NUM_CLASSES = len(SINGLE_CLASSES)

# ---------------------------------------------------------------------------
# Multiclass scenario (concept_mode == "multiclass", --store_multiconcept_cav,
# --multiclass_recalibration_mode)
# ---------------------------------------------------------------------------
MULTI_CLASSES = ["cat", "dog", "bird"]  # class0=cat, class1=dog, class2=bird (positional)
MULTI_CLASSIFICATION_DIR = os.path.join(DATA_ROOT, "classification", "multiclass")
MULTI_CONCEPT_DIR = os.path.join(DATA_ROOT, "concepts", "multiclass")
MULTI_RANDOM_FOLDER_NAME = "random"
MULTI_RANDOM_FOLDER = os.path.join(MULTI_CONCEPT_DIR, MULTI_RANDOM_FOLDER_NAME)
# {class_idx: [concept_folder_name, ...]} -- mirrors config's multiconcept.classN.target_folders
MULTI_CLASS_CONCEPTS = {
    0: ["cat_spots", "cat_ears"],   # cat
    1: ["dog_collar"],              # dog
    2: ["bird_feathers"],           # bird
}
# The `concept:` section is still mandatory schema even when --store_multiconcept_cav is
# used, so we also generate one small "primary" concept folder for the multiclass config.
MULTI_PRIMARY_CONCEPT_NAME = "primary"
MULTI_MODEL_ROOT = os.path.join(DATA_ROOT, "model_weights_multiclass")
MULTI_NUM_CLASSES = len(MULTI_CLASSES)

# ---------------------------------------------------------------------------
# Model / layer naming (see helpers/tiny_model.py for the actual architecture)
# ---------------------------------------------------------------------------
MODEL_NAME = "vgg16"  # must be one of main_store_cav.py's SUPPORTED_MODELS
ALL_CONV_LAYER_NAMES = ["features.0", "features.2", "features.3", "features.5", "features.6"]
SKIP_LAYERS = 2  # main_store_cav.py default --skip_layers
# Layers actually processed by main_store_cav.py with the default --skip_layers=2:
PROCESSED_LAYER_NAMES = ALL_CONV_LAYER_NAMES[SKIP_LAYERS:]  # ['features.3','features.5','features.6']

# ---------------------------------------------------------------------------
# Hyperparameters (kept minimal/fast: 1 epoch, tiny batch size)
# ---------------------------------------------------------------------------
SEED = 1234
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 2
LINEAR_CLASSIFIER_TYPE = "SGDClassifier"
LAMBDA_ALIGNS = [0.0, 0.5]
LAMBDA_ALIGNS_RECALIB = [0.5]

# Layer(s) used by the *_recalib fast config variants below: a single layer means
# `_resolve_layer_combos()` in main_recalib_custom_by_loading_cav.py produces exactly ONE
# combination (instead of the full C(3,1)+C(3,2)+C(3,3)=7 combos from all 3 processed
# layers), which keeps the gradient-descent recalibration CLI tests fast. The 7-combo
# math itself is still fully covered by a direct (non-subprocess) unit test.
FAST_RECALIB_VGG_LAYERS = ["features.6"]

# CAV store roots (populated by main_store_cav.py during tests)
SINGLE_CAV_STORE = os.path.join(DATA_ROOT, "cav_store_single")
MULTI_CAV_STORE = os.path.join(DATA_ROOT, "cav_store_multiclass")

# Results output roots used by the recalibration / sensitivity scripts under test
SINGLE_RECALIB_RESULTS = os.path.join(RESULTS_ROOT, "recalib_single")
MULTI_RECALIB_RESULTS = os.path.join(RESULTS_ROOT, "recalib_multiclass")
SINGLE_SENSITIVITY_RESULTS = os.path.join(RESULTS_ROOT, "sensitivity_single")
MULTI_SENSITIVITY_RESULTS = os.path.join(RESULTS_ROOT, "sensitivity_multiclass")
BOTTLENECK_RESULTS = os.path.join(RESULTS_ROOT, "bottleneck")
STORE_CAV_RESULTS = os.path.join(RESULTS_ROOT, "store_cav")

SINGLE_CONFIG_PATH = os.path.join(CONFIG_ROOT, "config_single_class.yaml")
MULTI_CONFIG_PATH = os.path.join(CONFIG_ROOT, "config_multiclass.yaml")
# Fast variants (override_retraining=True, single VGG layer) used only by the
# main_recalib_custom_by_loading_cav.py CLI/integration tests, so main_store_cav.py's own
# tests (which use the two paths above) are never affected by this speed optimisation.
SINGLE_CONFIG_RECALIB_PATH = os.path.join(CONFIG_ROOT, "config_single_class_recalib.yaml")
MULTI_CONFIG_RECALIB_PATH = os.path.join(CONFIG_ROOT, "config_multiclass_recalib.yaml")

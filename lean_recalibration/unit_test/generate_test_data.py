"""
Generate all synthetic test data (images, tiny model checkpoints, YAML configs) needed
by the lean_recalibration test suite.

Usage
-----
    python generate_test_data.py            # generate anything missing (idempotent)
    python generate_test_data.py --force     # wipe test/data + test/config and regenerate

This script is 100% self-contained (PIL + torch + yaml only) and OS-agnostic: every path
is built with ``os.path.join`` and no shell commands are used, so it runs unmodified on
Windows, Linux, or macOS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensure `helpers` importable
from helpers import constants as C
from helpers.image_gen import generate_dataset_tree, generate_concept_folder, generate_random_folder
from helpers.tiny_model import build_and_save


def _write_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def generate_single_class_data() -> None:
    print(f"[single] classification dataset -> {C.SINGLE_CLASSIFICATION_DIR}")
    generate_dataset_tree(C.SINGLE_CLASSIFICATION_DIR, C.SINGLE_CLASSES, C.IMAGES_PER_FOLDER)

    print(f"[single] concept folder '{C.SINGLE_CONCEPT_NAME}' -> {C.SINGLE_CONCEPT_DIR}")
    generate_concept_folder(C.SINGLE_CONCEPT_DIR, C.SINGLE_CONCEPT_NAME, C.IMAGES_PER_FOLDER)
    generate_random_folder(C.SINGLE_CONCEPT_DIR, "random", C.IMAGES_PER_FOLDER)

    model_path = os.path.join(C.SINGLE_MODEL_ROOT, C.MODEL_NAME, f"{C.MODEL_NAME}.pth")
    print(f"[single] tiny model (num_classes={C.SINGLE_NUM_CLASSES}) -> {model_path}")
    if not os.path.isfile(model_path):
        build_and_save(model_path, num_classes=C.SINGLE_NUM_CLASSES, seed=C.SEED)


def generate_multiclass_data() -> None:
    print(f"[multiclass] classification dataset -> {C.MULTI_CLASSIFICATION_DIR}")
    generate_dataset_tree(C.MULTI_CLASSIFICATION_DIR, C.MULTI_CLASSES, C.IMAGES_PER_FOLDER)

    print(f"[multiclass] primary concept + random -> {C.MULTI_CONCEPT_DIR}")
    generate_concept_folder(C.MULTI_CONCEPT_DIR, C.MULTI_PRIMARY_CONCEPT_NAME, C.IMAGES_PER_FOLDER)
    generate_random_folder(C.MULTI_CONCEPT_DIR, C.MULTI_RANDOM_FOLDER_NAME, C.IMAGES_PER_FOLDER)

    for class_idx, concept_list in C.MULTI_CLASS_CONCEPTS.items():
        for concept_name in concept_list:
            print(f"[multiclass] concept '{concept_name}' (class{class_idx}) -> {C.MULTI_CONCEPT_DIR}")
            generate_concept_folder(C.MULTI_CONCEPT_DIR, concept_name, C.IMAGES_PER_FOLDER)

    model_path = os.path.join(C.MULTI_MODEL_ROOT, C.MODEL_NAME, f"{C.MODEL_NAME}.pth")
    print(f"[multiclass] tiny model (num_classes={C.MULTI_NUM_CLASSES}) -> {model_path}")
    if not os.path.isfile(model_path):
        build_and_save(model_path, num_classes=C.MULTI_NUM_CLASSES, seed=C.SEED)


def _base_recalibration_layers_to_train(vgg_layers: list = None) -> dict:
    """All 5 model keys are mandatory in ConfigSingleton regardless of which model is
    actually under test; only 'vgg16' entries matter for these tests."""
    return {
        "vgg16": list(vgg_layers if vgg_layers is not None else C.PROCESSED_LAYER_NAMES),
        "resnet50": ["layer1"],
        "inception_v3": ["Mixed_5b"],
        "mobilnet_v3_small": ["features.1"],  # NOTE: matches the (typo'd) key ConfigSingleton reads
        "mobilenet_v3_large": ["features.1"],
    }


def build_single_class_config(override_retraining: bool = False, vgg_layers: list = None) -> dict:
    return {
        "misc": {"seed": C.SEED},
        "classification": {
            "data_base_path": C.SINGLE_CLASSIFICATION_DIR,
            "target_class_list": list(C.SINGLE_CLASSES),
            "linear_classifier_type": C.LINEAR_CLASSIFIER_TYPE,
            "lambda_aligns": list(C.LAMBDA_ALIGNS),
        },
        "concept": {
            "base_path": C.SINGLE_CONCEPT_DIR,
            "target_folders": [C.SINGLE_CONCEPT_NAME],
            "random_folder": C.SINGLE_RANDOM_FOLDER,
        },
        "hyperparameters": {
            "learning_rate": C.LEARNING_RATE,
            "epochs": C.EPOCHS,
            "batch_size": C.BATCH_SIZE,
        },
        "recalibration": {
            "override_retraining": override_retraining,
            "layers_to_train": _base_recalibration_layers_to_train(vgg_layers),
            "lambda_aligns_recalib": list(C.LAMBDA_ALIGNS_RECALIB),
        },
        "xai_before_after": {
            "integrated_gradients": False,
            "grad_cam": False,
            "lime": False,
            "num_classes": 0,
            "class_images": {},
        },
    }


def build_multiclass_config(override_retraining: bool = False, vgg_layers: list = None) -> dict:
    multiconcept = {
        "base_path": C.MULTI_CONCEPT_DIR,
        "random_folder_base_path": C.MULTI_CONCEPT_DIR,
        "random_folder": C.MULTI_RANDOM_FOLDER,  # legacy fallback
    }
    for class_idx, concept_list in C.MULTI_CLASS_CONCEPTS.items():
        multiconcept[f"class{class_idx}"] = {
            "target_folders": list(concept_list),
            "random_folders": [C.MULTI_RANDOM_FOLDER_NAME] * len(concept_list),
        }

    recalibration_weights = {}
    for class_idx, concept_list in C.MULTI_CLASS_CONCEPTS.items():
        weight = round(1.0 / max(len(concept_list), 1), 4)
        recalibration_weights[f"class{class_idx}"] = {
            "class_weight": 1.0,
            "concept_weights": {name: weight for name in concept_list},
        }

    return {
        "misc": {"seed": C.SEED},
        "classification": {
            "data_base_path": C.MULTI_CLASSIFICATION_DIR,
            "target_class_list": list(C.MULTI_CLASSES),
            "linear_classifier_type": C.LINEAR_CLASSIFIER_TYPE,
            "lambda_aligns": list(C.LAMBDA_ALIGNS),
        },
        "concept": {
            "base_path": C.MULTI_CONCEPT_DIR,
            "target_folders": [C.MULTI_PRIMARY_CONCEPT_NAME],
            "random_folder": C.MULTI_RANDOM_FOLDER,
        },
        "hyperparameters": {
            "learning_rate": C.LEARNING_RATE,
            "epochs": C.EPOCHS,
            "batch_size": C.BATCH_SIZE,
        },
        "recalibration": {
            "override_retraining": override_retraining,
            "layers_to_train": _base_recalibration_layers_to_train(vgg_layers),
            "lambda_aligns_recalib": list(C.LAMBDA_ALIGNS_RECALIB),
        },
        "xai_before_after": {
            "integrated_gradients": False,
            "grad_cam": False,
            "lime": False,
            "num_classes": 0,
            "class_images": {},
        },
        "multiconcept": multiconcept,
        "recalibration_weights": recalibration_weights,
    }


def generate_configs() -> None:
    print(f"[config] writing {C.SINGLE_CONFIG_PATH}")
    _write_yaml(C.SINGLE_CONFIG_PATH, build_single_class_config())
    print(f"[config] writing {C.MULTI_CONFIG_PATH}")
    _write_yaml(C.MULTI_CONFIG_PATH, build_multiclass_config())

    # Fast variants used only by main_recalib_custom_by_loading_cav.py's CLI tests:
    # override_retraining=True + a single VGG layer collapses
    # `_resolve_layer_combos()` down to exactly ONE combination (instead of 7), which keeps
    # the real gradient-descent training loop fast. main_store_cav.py never reads the
    # `recalibration` section, so its own tests (using the two paths above) are unaffected.
    print(f"[config] writing {C.SINGLE_CONFIG_RECALIB_PATH}")
    _write_yaml(
        C.SINGLE_CONFIG_RECALIB_PATH,
        build_single_class_config(override_retraining=True, vgg_layers=C.FAST_RECALIB_VGG_LAYERS),
    )
    print(f"[config] writing {C.MULTI_CONFIG_RECALIB_PATH}")
    _write_yaml(
        C.MULTI_CONFIG_RECALIB_PATH,
        build_multiclass_config(override_retraining=True, vgg_layers=C.FAST_RECALIB_VGG_LAYERS),
    )


def main(argv=None) -> None:
    """Generate test data/configs.

    `argv` defaults to an EMPTY list (not `sys.argv[1:]`) so that calling
    `generate_test_data.main()` as a plain library function -- e.g. from the pytest
    `_ensure_test_data` fixture, which runs inside an already-running pytest process --
    never accidentally tries to parse pytest's own CLI arguments (`-v`, `-m`, ...). The
    `if __name__ == "__main__":` entry point below explicitly passes `sys.argv[1:]` so
    `python generate_test_data.py --force` still works exactly as before from the shell.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic test data/configs for lean_recalibration tests.")
    parser.add_argument("--force", action="store_true", help="Wipe test/data and test/config before regenerating.")
    args = parser.parse_args(argv if argv is not None else [])

    if args.force:
        for folder in (C.DATA_ROOT, C.CONFIG_ROOT):
            if os.path.isdir(folder):
                print(f"[force] removing {folder}")
                shutil.rmtree(folder)

    os.makedirs(C.DATA_ROOT, exist_ok=True)
    os.makedirs(C.CONFIG_ROOT, exist_ok=True)
    os.makedirs(C.RESULTS_ROOT, exist_ok=True)

    generate_single_class_data()
    generate_multiclass_data()
    generate_configs()
    print("\nTest data generation complete.")
    print(f"  Data root   : {C.DATA_ROOT}")
    print(f"  Config root : {C.CONFIG_ROOT}")


if __name__ == "__main__":
    main(sys.argv[1:])

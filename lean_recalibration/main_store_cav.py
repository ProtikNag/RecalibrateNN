# Copyright [2025] [Srikanth KS]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
/*
 * Copyright (c) 2025 Srikanth K S. All rights reserved.
 * Licensed under the APACHE2 License.
 * Author : Srikanth K S
 * Version 1.0
 */
"""
"""
Known bug
/mnt/data/python_venv/lib/python3.12/site-packages/torch/autograd/graph.py:825: UserWarning:
adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation, but you set
'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at
https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support
for this operation.
"""

import copy
import os
import random
import argparse
from datetime import datetime
import joblib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from dotenv import load_dotenv
import sys
sys.path.append("../")
from logger import Logger_Singleton
from custom_dataloader import SingleClassDataLoader, MultiClassImageDataset
from ConfigSingleton import ConfigSingleton
from cav_registry import CAVRegistry
from utils import (
    get_num_classes,
    get_class_folder_dicts,
    train_cav,
    get_model_weight_path,
    get_base_model_image_size,
    get_model_layers,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True, warn_only=True)

# ---------------------------------------------------------------------------
# Interactive usage overview
# ---------------------------------------------------------------------------
# Every public function in this module can also be driven step-by-step from a
# plain Python prompt (no CLI needed) -- each function docstring below has a
# runnable example. Typical minimal end-to-end flow:
#
#   >>> import torch
#   >>> from cav_registry import CAVRegistry
#   >>> from lean_recalibration.main_store_cav import (
#   ...     compute_cav, process_model, load_and_verify, print_model_manifest,
#   ... )
#   >>> registry = CAVRegistry.create_empty_manifest("./cav_store")
#   >>> LINEAR_CLASSIFIER_TYPE = "SGDClassifier"   # module global read by compute_cav/process_model
#   >>> model = torch.load("C:/models/vgg16/vgg16.pt", map_location="cpu", weights_only=False)
#   >>> process_model(
#   ...     model=model,
#   ...     model_name="vgg16",
#   ...     layer_names=["features.11", "features.13"],
#   ...     concept_loaders=[loader_concept1, loader_concept2],
#   ...     concept_names=["concept1", "concept2"],
#   ...     random_loader=loader_random,
#   ...     registry=registry,
#   ...     logging=my_logger,
#   ...     random_folder="C:/data/random",
#   ...     model_weight_path="C:/models/vgg16/vgg16.pt",
#   ...     concept_data_paths={"concept1": "C:/data/concept1", "concept2": "C:/data/concept2"},
#   ... )
#   >>> load_and_verify(registry, "vgg16", ["features.11", "features.13"], my_logger)
#   >>> print_model_manifest("./cav_store", "vgg16")
#
# This produces:
#   cav_store/
#     vgg16_manifest.json
#     vgg16/
#       concept1/features.11.joblib, features.13.joblib
#       concept2/features.11.joblib, features.13.joblib

# ---------------------------------------------------------------------------
# Module-level state (populated in __main__)
# ---------------------------------------------------------------------------
MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
RANDOM_STATE = 132

activation = {}
output_shape = {}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    np.random.seed(RANDOM_STATE + worker_id)
    random.seed(RANDOM_STATE + worker_id)


set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Activation hook
# ---------------------------------------------------------------------------

def get_activation(layer_name: str):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        print(
            f"Activation hook layer={layer_name}, output.shape={output.shape}"
        )
    return hook


# ---------------------------------------------------------------------------
# CAV computation
# ---------------------------------------------------------------------------

def compute_cav(
    model,
    loader_positive,
    loader_random,
    layer_name: str,
    orthogonal: bool = False,
) -> torch.Tensor:
    """
    Compute a single CAV vector for one concept loader vs. random loader.
    Returns a 1-D float32 tensor on DEVICE.

    Example (interactive Python prompt)
    ------------------------------------
    >>> from lean_recalibration.main_store_cav import compute_cav
    >>> cav_tensor = compute_cav(
    ...     model=my_model_with_hook_registered,
    ...     loader_positive=concept_dataloader,
    ...     loader_random=random_dataloader,
    ...     layer_name="features.11",
    ... )
    """
    pos_acts, rnd_acts = [], []
    model.eval()
    with torch.no_grad():
        for imgs in loader_positive:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            pos_acts.append(
                activation[layer_name].view(imgs.size(0), -1).cpu().numpy()
            )
        for imgs in loader_random:
            imgs = imgs.to(DEVICE)
            _ = model(imgs)
            rnd_acts.append(
                activation[layer_name].view(imgs.size(0), -1).cpu().numpy()
            )
    pos_acts = np.vstack(pos_acts)
    rnd_acts = np.vstack(rnd_acts)
    cav = train_cav(pos_acts, rnd_acts, orthogonal, LINEAR_CLASSIFIER_TYPE)
    return torch.tensor(cav, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# Per-concept alias helper
# ---------------------------------------------------------------------------

def _concept_aliases(concept_names, registry: CAVRegistry) -> dict:
    """
    Return {concept_name: alias} using the manifest when available,
    falling back to the concept name itself.
    """
    manifest_concepts = registry._manifest.get("concepts", {})
    return {
        name: manifest_concepts.get(name, {}).get("alias", name)
        for name in concept_names
    }


# ---------------------------------------------------------------------------
# Core loop: compute + store CAVs for all layers of one model
# ---------------------------------------------------------------------------

def process_model(
    model,
    model_name: str,
    layer_names: list,
    concept_loaders: list,
    concept_names: list,
    random_loader,
    registry: CAVRegistry,
    logging,
    random_folder: str,
    model_weight_path: str = "",
    concept_data_paths: dict = None,
    concept_class_names: dict = None,
    concept_random_loaders: dict = None,
    concept_random_folders: dict = None,
):
    """
    For every layer, compute a CAV per concept, then persist each concept's
    CAV via the registry at:
        <cav_store_root>/<model_name>/<concept_name>/<layer_name>.joblib
    and update <cav_store_root>/<model_name>_manifest.json.

    Parameters
    ----------
    model               : loaded PyTorch model (not yet moved to DEVICE here).
    model_name          : canonical model name string, e.g. "vgg16".
    layer_names         : list of layer name strings to process.
    concept_loaders     : list of DataLoaders, one per concept (same order as concept_names).
    concept_names       : list of concept name strings.
    random_loader       : DataLoader for random/negative examples, used as the fallback for any
                          concept not present in `concept_random_loaders`.
    registry            : CAVRegistry instance.
    logging             : Logger_Singleton instance.
    random_folder       : path to the random/negative image folder used as the fallback for any
                          concept not present in `concept_random_folders`.
    model_weight_path   : complete path to the model weights used to capture the CAVs.
    concept_data_paths  : (optional) {concept_name: source_images_path}, recorded in the manifest.
    concept_class_names : (optional) {concept_name: class_name}, recorded in the manifest.
    concept_random_loaders : (optional) {concept_name: DataLoader}. Lets each concept be trained
                          against its own random/negative loader (e.g. one random folder per
                          multiconcept target_folder). Takes precedence over `random_loader`.
    concept_random_folders : (optional) {concept_name: random_folder_path}, recorded in the
                          manifest per concept. Takes precedence over `random_folder`.

    Example (interactive Python prompt)
    ------------------------------------
    >>> from lean_recalibration.main_store_cav import process_model
    >>> from cav_registry import CAVRegistry
    >>> registry = CAVRegistry.create_empty_manifest("./cav_store")
    >>> process_model(
    ...     model=my_loaded_model,
    ...     model_name="vgg16",
    ...     layer_names=["features.11", "features.13"],
    ...     concept_loaders=[loader_concept1, loader_concept2],
    ...     concept_names=["concept1", "concept2"],
    ...     random_loader=loader_random,
    ...     registry=registry,
    ...     logging=my_logger,
    ...     random_folder="C:/data/random",
    ...     model_weight_path="C:/models/vgg16/vgg16.pt",
    ...     concept_data_paths={"concept1": "C:/data/concept1", "concept2": "C:/data/concept2"},
    ... )
    """
    concept_data_paths = concept_data_paths or {}
    concept_class_names = concept_class_names or {}
    concept_random_loaders = concept_random_loaders or {}
    concept_random_folders = concept_random_folders or {}
    aliases = _concept_aliases(concept_names, registry)
    logging.info(f"Processing model={model_name}, layers={layer_names}")

    for layer_name in layer_names:
        logging.info(f"  Layer: {layer_name}")
        try:
            model_copy = copy.deepcopy(model).to(DEVICE)
            model_copy.get_submodule(layer_name).register_forward_hook(
                get_activation(layer_name)
            )

            print(f"Computing CAVs for layer={layer_name}  this may take a while ")
            concept_cavs = {}
            for concept_name, concept_loader in zip(concept_names, concept_loaders):
                try:
                    loader_random = concept_random_loaders.get(concept_name, random_loader)
                    cav_tensor = compute_cav(
                        model_copy, concept_loader, loader_random, layer_name
                    )
                    concept_cavs[concept_name] = cav_tensor
                    logging.info(
                        f"    CAV computed for concept={concept_name}, "
                        f"shape={cav_tensor.shape}"
                    )
                except Exception as exc:
                    logging.error(
                        f"    CAV computation failed concept={concept_name}: {exc}"
                    )
                    print(f"    CAV computation failed concept={concept_name}: {exc}")

            if not concept_cavs:
                logging.warning(f"  No CAVs produced for layer={layer_name}, skipping.")
                continue

            saved_paths = registry.save_layer_cav(
                model=model_name,
                layer=layer_name,
                concept_cavs=concept_cavs,
                concept_aliases=aliases,
                linear_classifier_type=LINEAR_CLASSIFIER_TYPE,
                random_folder=random_folder,
                concept_random_folders=concept_random_folders,
                model_weight_path=model_weight_path,
                concept_data_paths=concept_data_paths,
                concept_class_names=concept_class_names,
            )
            logging.info(f"  Saved: {saved_paths}")
            print(f"  Saved: {saved_paths}")

        except Exception as exc:
            logging.error(f"  Layer {layer_name} failed: {exc}")
            print(f"  Layer {layer_name} failed: {exc}")
            continue

    logging.info(f"Finished model={model_name}")


# ---------------------------------------------------------------------------
# Load and display stored CAVs (verification)
# ---------------------------------------------------------------------------

def load_and_verify(registry: CAVRegistry, model_name: str, layer_names: list, logging):
    """
    Load each stored concept/layer CAV file and print a summary.

    Example (interactive Python prompt)
    ------------------------------------
    >>> from lean_recalibration.main_store_cav import load_and_verify
    >>> from cav_registry import CAVRegistry
    >>> from logger import Logger_Singleton
    >>> registry = CAVRegistry("./cav_store")
    >>> logger = Logger_Singleton("./results/verify.log")
    >>> load_and_verify(registry, "vgg16", ["features.11", "features.13"], logger)
    """
    logging.info(f"Verifying stored CAVs for model={model_name}")
    for layer_name in layer_names:
        try:
            cavs = registry.get_layer_cavs(model_name, layer_name)
            summary = {c: tuple(v.shape) for c, v in cavs.items()}
            logging.info(f"  layer={layer_name}: {summary}")
            print(f"  Verified layer={layer_name}: {summary}")
        except FileNotFoundError as exc:
            logging.warning(f"  {exc}")
            print(f"  WARNING: {exc}")


# ---------------------------------------------------------------------------
# Print the per-model manifest (<cav_store_root>/<model_name>_manifest.json)
# ---------------------------------------------------------------------------

def print_model_manifest(cav_store_root: str, model_name: str, logging=None):
    """
    Print a summary of <cav_store_root>/<model_name>_manifest.json.

    The registry (CAVRegistry.save_concept_layer_cav / save_layer_cav) keeps
    this manifest up to date automatically every time a concept CAV is saved,
    so this helper only needs to read and summarize it -- it does not need to
    rebuild anything by scanning the CAV store.

    Args
    ----
    cav_store_root : Root path containing "<model_name>/" and
                      "<model_name>_manifest.json".
    model_name     : Name of the model, e.g. "vgg16".
    logging        : Logger instance (optional).

    Example (interactive Python prompt)
    ------------------------------------
    >>> from lean_recalibration.main_store_cav import print_model_manifest
    >>> print_model_manifest("./cav_store", "vgg16")
    """
    import json
    from pathlib import Path

    manifest_path = Path(cav_store_root) / f"{model_name}_manifest.json"
    if not manifest_path.exists():
        msg = f"No manifest found for model={model_name} at {manifest_path}"
        if logging:
            logging.warning(msg)
        print(f"WARNING: {msg}")
        return

    with open(manifest_path, "r") as fh:
        manifest_data = json.load(fh)

    concepts = manifest_data.get("concepts", {})
    if logging:
        logging.info(f"Model manifest: {manifest_path}")
        logging.info(f"  Total concepts: {len(concepts)}")
    print(f"Model manifest : {manifest_path}")
    print(f"  Model weight path: {manifest_data.get('model_weight_path', '')}")
    print(f"  Total concepts   : {len(concepts)}")
    for concept_name, info in concepts.items():
        print(
            f"    - {concept_name}: data_path={info.get('data_path')} "
            f"random_folder={info.get('random_folder')} "
            f"layers={info.get('layers')}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and store CAV vectors for one or more models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=(
            "Comma-separated model name(s) to process, e.g. 'vgg16' or "
            "'vgg16,resnet50'.  Omit to process all models listed in the manifest."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Root directory that contains model weight sub-folders.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--cav_store",
        type=str,
        default="./cav_store",
        help="Root directory for the CAV store (default: ./cav_store).",
    )
    parser.add_argument(
        "--skip_layers",
        type=int,
        default=2,
        help="Skip the first N layers returned by get_model_layers (default: 2).",
    )
    parser.add_argument(
        "--store_multiconcept_cav",
        action="store_true",
        help=(
            "Read multiconcept definitions from YAML and store CAVs under "
            "<cav_store>/<class_name>/<concept_name>/ with per-scope manifests."
        ),
    )
    parser.add_argument(
    "--info",
    action="store_true",
    help="Display example usage of command line arguments.",
    )
    
    args = parser.parse_args()
    if args.info:
      print("Example usage:")
      print("  python main_store_cav.py --config_file config.yaml")
      print("  python main_store_cav.py --config_file config.yaml --model_name vgg16")
      print("  python main_store_cav.py --config_file config.yaml --model_name vgg16,resnet50 --model_path ./models --cav_store ./cav_store --skip_layers 3")
      print("  python main_store_cav.py --config_file config.yaml --store_multiconcept_cav")
      print("  python main_store_cav.py --config_file config.yaml --model_name vgg16 --model_path ./models --cav_store ./my_cavs --skip_layers 2")
      print(" Working example ")
      print(" python main_store_cav.py  --model_name vgg16 --model_path /home/srikanth/trained_models/pytorch/legacy --config_file ../config/multiclass/legacy/config_legacy_3classes.yaml --cav_store ./ --store_multiconcept_cav")
  
      sys.exit(0)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    if not args.config_file:
        parser.error("--config_file is required")

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    config = ConfigSingleton(args.config_file)

    SEED = config.SEED
    set_seed(SEED)

    CLASSIFICATION_DATA_BASE_PATH = config.CLASSIFICATION_DATA_BASE_PATH
    TARGET_CLASS_LIST = config.TARGET_CLASS_LIST
    RANDOM_FOLDER = config.RANDOM_FOLDER
    CONCEPT_FOLDER_LIST = config.CONCEPT_FOLDER_LIST
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    LINEAR_CLASSIFIER_TYPE = config.LINEAR_CLASSIFIER_TYPE

    # Derive concept names from folder basenames
    CONCEPT_NAMES = [os.path.basename(p.rstrip("/\\")) for p in CONCEPT_FOLDER_LIST]

    BASE_MODEL_PATH = args.model_path or "./model_weights"

    # ------------------------------------------------------------------
    # Concept planning
    # ------------------------------------------------------------------
    # Build a single FLAT list of (concept_name, concept_path) pairs that will
    # all be stored under one shared cav_store root, using the directory
    # layout described at the top of cav_registry.py:
    #     <cav_store>/<model_name>_manifest.json
    #     <cav_store>/<model_name>/<concept_name>/<layer_name>.joblib
    #
    # --store_multiconcept_cav lets you supply MULTIPLE concepts per class
    # (config.MULTICONCEPT_CLASS_CONCEPTS: {class_idx: [concept_folder, ...]}).
    # Concept names are prefixed with the class name (e.g. "deer__coat") so
    # that concepts from different classes never collide in the flat layout.
    def _safe_path_name(name: str) -> str:
        return str(name).replace(" ", "_").replace("/", "_").replace("\\", "_")

    def _resolve_concept_path(base_path: str, concept_name: str) -> str:
        if os.path.isabs(concept_name):
            return concept_name
        return os.path.join(base_path, concept_name)

    concept_names = []
    concept_paths = []
    concept_class_names = {}   # {concept_name: class_name}
    concept_data_paths = {}    # {concept_name: source_images_path}
    concept_random_folders = {}  # {concept_name: resolved random_folder path for this concept}

    if args.store_multiconcept_cav:
        if not getattr(config, "MULTICONCEPT_ENABLED", False):
            raise ValueError(
                "--store_multiconcept_cav was provided but multiconcept is not enabled in YAML."
            )

        # Fallback random folder used only when a target_folder has no corresponding
        # entry in its class's random_folders list.
        random_folder_to_use = (
            config.MULTICONCEPT_RANDOM_FOLDER
            if getattr(config, "MULTICONCEPT_RANDOM_FOLDER", "")
            else RANDOM_FOLDER
        )

        for class_idx, target_folders in config.MULTICONCEPT_CLASS_CONCEPTS.items():
            class_name = (
                TARGET_CLASS_LIST[class_idx]
                if class_idx < len(TARGET_CLASS_LIST)
                else f"class{class_idx}"
            )
            for position_idx, target_folder in enumerate(target_folders):
                concept_path = _resolve_concept_path(
                    config.MULTICONCEPT_BASE_PATH,
                    target_folder,
                )
                base_concept_name = _safe_path_name(
                    os.path.basename(target_folder.rstrip("/\\"))
                )
                concept_name = f"{_safe_path_name(class_name)}_{base_concept_name}"
                concept_names.append(concept_name)
                concept_paths.append(concept_path)
                concept_class_names[concept_name] = class_name
                concept_data_paths[concept_name] = concept_path

                # Each target_folder is paired 1:1 (by position) with a random_folder
                # from the same class, resolved against random_folder_base_path.
                resolved_random_path = config.get_multiconcept_random_folder_path(
                    class_idx, position_idx
                )
                concept_random_folders[concept_name] = resolved_random_path or random_folder_to_use
    else:
        random_folder_to_use = RANDOM_FOLDER
        concept_names = list(CONCEPT_NAMES)
        concept_paths = list(CONCEPT_FOLDER_LIST)
        concept_data_paths = dict(zip(concept_names, concept_paths))

    # ------------------------------------------------------------------
    # Determine which models to process
    # ------------------------------------------------------------------
    SUPPORTED_MODELS = [
        "vgg16",
        "resnet50",
        "inception_v3",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
    ]

    if args.model_name:
        models_to_process = [m.strip().lower() for m in args.model_name.split(",")]
        unknown = [m for m in models_to_process if m not in SUPPORTED_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. "
                f"Supported: {SUPPORTED_MODELS}"
            )
    else:
        models_to_process = SUPPORTED_MODELS

    print(f"Models to process: {models_to_process}")
    print(f"Concepts to process ({len(concept_names)}): {concept_names}")

    # ------------------------------------------------------------------
    # Transforms (shared; inception gets its own IMAGE_SIZE at load time)
    # ------------------------------------------------------------------
    def make_transforms(image_size: int):
        t = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return t

    # ------------------------------------------------------------------
    # Shared data loaders (concept + random)
    # ------------------------------------------------------------------
    def build_concept_loaders(transform, concept_paths):
        return [
            DataLoader(
                SingleClassDataLoader(path, transform=transform),
                batch_size=BATCH_SIZE,
                shuffle=True,
                worker_init_fn=worker_init_fn,
            )
            for path in concept_paths
        ]

    def build_random_loader(transform, random_folder):
        return DataLoader(
            SingleClassDataLoader(random_folder, transform=transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

    # ------------------------------------------------------------------
    # CAV registry (single flat store shared by every model/concept)
    # ------------------------------------------------------------------
    cav_store_root = os.path.abspath(args.cav_store)
    os.makedirs(cav_store_root, exist_ok=True)

    manifest_path = os.path.join(cav_store_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        registry = CAVRegistry.create_empty_manifest(cav_store_root)
    else:
        registry = CAVRegistry(cav_store_root)

    for concept_name, concept_path in zip(concept_names, concept_paths):
        if concept_name not in registry._manifest.get("concepts", {}):
            registry.add_concept(
                name=concept_name,
                alias=concept_name,
                data_path=concept_path,
                description=(
                    f"class={concept_class_names[concept_name]}"
                    if concept_name in concept_class_names
                    else ""
                ),
                random_folder=concept_random_folders.get(concept_name, random_folder_to_use),
                save=False,
            )
    registry._save_manifest()

    print(f"\nCAV store root : {cav_store_root}")
    print(f"Manifest       : {registry.manifest_path}")

    # ------------------------------------------------------------------
    # Main loop over models
    # ------------------------------------------------------------------
    for model_name in models_to_process:
        RESULTS_PATH = os.path.join(cav_store_root, model_name)
        os.makedirs(RESULTS_PATH, exist_ok=True)
        log_filename = os.path.join(
            RESULTS_PATH,
            f"audit_trail_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        logging = Logger_Singleton(log_filename)
        logging.info(f"=== Processing model: {model_name} ===")

        try:
            MODEL_PATH = get_model_weight_path(model_name, BASE_MODEL_PATH)
        except FileNotFoundError as exc:
            logging.error(str(exc))
            print(f"SKIP {model_name}: {exc}")
            continue

        IMAGE_SIZE = get_base_model_image_size(model_name)
        transform = make_transforms(IMAGE_SIZE)

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  |  weights: {MODEL_PATH}  |  device: {DEVICE}")
        print(f"{'='*60}")

        # Load model
        try:
            if DEVICE == "cpu":
                model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            else:
                model = torch.load(MODEL_PATH, map_location=DEVICE)
            model.to(DEVICE)
        except Exception as exc:
            logging.error(f"Failed to load model {model_name}: {exc}")
            print(f"SKIP {model_name}: failed to load  {exc}")
            continue

        registry.update_model_meta(
            model=model_name,
            alias=model_name,
            weight_path=MODEL_PATH,
        )

        all_layers = get_model_layers(model)
        logging.info(f"All layers ({len(all_layers)}): {all_layers}")
        layer_names = all_layers[args.skip_layers:]
        logging.info(f"Layers to process ({len(layer_names)}): {layer_names}")
        print(f"Layers to process: {layer_names}")

        concept_loaders = build_concept_loaders(transform, concept_paths)
        random_loader = build_random_loader(transform, random_folder_to_use)

        # Build one DataLoader per unique random folder path (avoids rebuilding
        # duplicate loaders when several concepts happen to share a random folder),
        # then map each concept to its own random loader.
        concept_random_loaders = None
        if concept_random_folders:
            unique_random_folders = set(concept_random_folders.values())
            random_loader_cache = {
                path: build_random_loader(transform, path)
                for path in unique_random_folders
            }
            concept_random_loaders = {
                concept_name: random_loader_cache[path]
                for concept_name, path in concept_random_folders.items()
            }

        process_model(
            model=model,
            model_name=model_name,
            layer_names=layer_names,
            concept_loaders=concept_loaders,
            concept_names=concept_names,
            random_loader=random_loader,
            registry=registry,
            logging=logging,
            random_folder=random_folder_to_use,
            model_weight_path=MODEL_PATH,
            concept_data_paths=concept_data_paths,
            concept_class_names=concept_class_names,
            concept_random_loaders=concept_random_loaders,
            concept_random_folders=concept_random_folders,
        )

        load_and_verify(registry, model_name, layer_names, logging)
        print_model_manifest(cav_store_root, model_name, logging)
        logging.info(f"=== Done: {model_name} ===")

    print("\nAll models processed.")
    print(f"CAV store root : {cav_store_root}")
    print(f"Models in store: {registry.list_models()}")


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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from dotenv import load_dotenv

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
):
    """
    For every layer, compute a CAV per concept, then persist via registry.

    Parameters
    ----------
    model          : loaded PyTorch model (not yet moved to DEVICE here).
    model_name     : canonical model name string, e.g. "vgg16".
    layer_names    : list of layer name strings to process.
    concept_loaders: list of DataLoaders, one per concept (same order as concept_names).
    concept_names  : list of concept name strings.
    random_loader  : DataLoader for random/negative examples.
    registry       : CAVRegistry instance.
    logging        : Logger_Singleton instance.
    """
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
                    cav_tensor = compute_cav(
                        model_copy, concept_loader, random_loader, layer_name
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

            saved_path = registry.save_layer_cav(
                model=model_name,
                layer=layer_name,
                concept_cavs=concept_cavs,
                concept_aliases=aliases,
                linear_classifier_type=LINEAR_CLASSIFIER_TYPE,
                random_folder=RANDOM_FOLDER,
            )
            logging.info(f"  Saved: {saved_path}")
            print(f"  Saved: {saved_path}")

        except Exception as exc:
            logging.error(f"  Layer {layer_name} failed: {exc}")
            print(f"  Layer {layer_name} failed: {exc}")
            continue

    logging.info(f"Finished model={model_name}")


# ---------------------------------------------------------------------------
# Load and display stored CAVs (verification)
# ---------------------------------------------------------------------------

def load_and_verify(registry: CAVRegistry, model_name: str, layer_names: list, logging):
    """Load each stored layer file and print a summary."""
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
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--store_results",
        type=str,
        default="./results",
        help="Base path for log files (default: ./results).",
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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
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
    # CAV registry
    # ------------------------------------------------------------------
    cav_store_root = os.path.abspath(args.cav_store)
    os.makedirs(cav_store_root, exist_ok=True)

    manifest_path = os.path.join(cav_store_root, "manifest.json")
    if not os.path.isfile(manifest_path):
        registry = CAVRegistry.create_empty_manifest(cav_store_root)
    else:
        registry = CAVRegistry(cav_store_root)

    # Register concepts in manifest (idempotent)
    for concept_name, concept_path in zip(CONCEPT_NAMES, CONCEPT_FOLDER_LIST):
        if concept_name not in registry._manifest.get("concepts", {}):
            registry.add_concept(
                name=concept_name,
                alias=concept_name,      # user can edit manifest to add friendlier alias
                data_path=concept_path,
                save=False,
            )
    registry._save_manifest()

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
        # Process all models listed in the manifest; fall back to full list
        manifest_models = registry.list_models()
        models_to_process = manifest_models if manifest_models else SUPPORTED_MODELS

    print(f"Models to process: {models_to_process}")

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
    # Shared data loaders (concept + random)  built once, reused
    # ------------------------------------------------------------------
    VALID_TRANSFORM_224 = make_transforms(224)
    VALID_TRANSFORM_299 = make_transforms(299)

    def build_concept_loaders(transform):
        return [
            DataLoader(
                SingleClassDataLoader(path, transform=transform),
                batch_size=BATCH_SIZE,
                shuffle=True,
                worker_init_fn=worker_init_fn,
            )
            for path in CONCEPT_FOLDER_LIST
        ]

    def build_random_loader(transform):
        return DataLoader(
            SingleClassDataLoader(RANDOM_FOLDER, transform=transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

    # ------------------------------------------------------------------
    # Main loop over models
    # ------------------------------------------------------------------
    for model_name in models_to_process:
        RESULTS_PATH = os.path.join(args.store_results, model_name)
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

        # Update manifest with model metadata
        registry.update_model_meta(
            model=model_name,
            alias=model_name,
            weight_path=MODEL_PATH,
        )

        # Determine layers
        all_layers = get_model_layers(model)
        logging.info(f"All layers ({len(all_layers)}): {all_layers}")
        layer_names = all_layers[args.skip_layers:]
        logging.info(f"Layers to process ({len(layer_names)}): {layer_names}")
        print(f"Layers to process: {layer_names}")

        # Build loaders for this model's image size
        concept_loaders = build_concept_loaders(transform)
        random_loader = build_random_loader(transform)

        process_model(
            model=model,
            model_name=model_name,
            layer_names=layer_names,
            concept_loaders=concept_loaders,
            concept_names=CONCEPT_NAMES,
            random_loader=random_loader,
            registry=registry,
            logging=logging,
        )

        load_and_verify(registry, model_name, layer_names, logging)
        logging.info(f"=== Done: {model_name} ===")

    print("\nAll models processed.")
    print(f"CAV store root : {cav_store_root}")
    print(f"Manifest       : {registry.manifest_path}")
    print(f"Models in store: {registry.list_models()}")

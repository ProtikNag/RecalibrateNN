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
CAV Registry - Load, query, and group CAV vectors for targeted recalibration.

Directory layout (one joblib file per model/concept/layer):
    <cav_store_root>/
        manifest.json                     # global bookkeeping (models, concepts, layer_groups)
        <model_name>_manifest.json         # per-model manifest (see schema below)
        <model_name>/                      # e.g. "vgg16"
            <concept_name_1>/
                <layer_name_1>.joblib
                <layer_name_2>.joblib
            <concept_name_2>/
                <layer_name_1>.joblib
                ...
        ...

Each <layer_name>.joblib file schema (schema_version "2.0") holds a SINGLE
concept's CAV for a single layer:
    {
        "schema_version": "2.0",
        "model": str,
        "concept": str,
        "layer_name": str,
        "cav_vector": torch.Tensor,   # shape (activation_dim,)
        "alias": str,
        "metadata": {
            "created_at": str,
            "linear_classifier_type": str,
            "random_folder": str
        }
    }

<model_name>_manifest.json schema (created/updated automatically whenever a
CAV is saved for that model):
    {
        "schema_version": "1.0",
        "model_name": str,
        "model_weight_path": str,          # complete path to the model weights used
        "created_at": str,
        "updated_at": str,
        "concepts": {
            "<concept_name>": {
                "concept_name": str,
                "alias": str,
                "data_path": str,          # complete path where this concept's CAVs are stored
                "model_weight_path": str,  # complete path to the model used to capture the CAV
                "source_images_path": str, # (optional) folder of images used to compute the CAV
                "random_folder": str,      # path to the random/negative folder paired with this concept
                "class_name": str,         # (optional) originating class, for multiconcept runs
                "layers": [str, ...],
                "created_at": str
            },
            ...
        }
    }

manifest.json schema (kept for backward compatibility, layer-group bookkeeping):
    {
        "schema_version": "1.0",
        "concepts": {
            "<concept_name>": {
                "alias": str,
                "data_path": str,
                "random_folder": str,      # path to the random/negative folder paired with this concept
                "description": str
            }
        },
        "models": {
            "<model_name>": {
                "alias": str,
                "weight_path": str,
                "computed_layers": [str, ...]
            }
        },
        "layer_groups": {
            "<group_name>": {
                "model": str,
                "layers": [str, ...],
                "description": str
            }
        }
    }
"""

import os
import json
import joblib
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"
# Schema version for the per-concept/per-layer joblib payloads.
LAYER_FILE_SCHEMA_VERSION = "2.0"


class CAVRegistry:
    """
    Registry for loading, querying, and grouping CAV vectors.

    Usage
    -----
    >>> registry = CAVRegistry("./cav_store")
    >>> cav = registry.get_cav("vgg16", "features.11", "deer_bias")
    >>> layer_cavs = registry.get_layer_cavs("vgg16", "features.13")
    >>> group_cavs = registry.get_group_cavs("vgg16_mid")
    """

    def __init__(self, cav_store_root: str):
        self.cav_store_root = os.path.abspath(cav_store_root)
        self.manifest_path = os.path.join(self.cav_store_root, "manifest.json")
        # Cache is keyed by (model, concept, layer) since each joblib file now
        # holds a single concept's CAV for a single layer.
        self._cache: Dict[Tuple[str, str, str], dict] = {}
        self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self):
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(
                f"Manifest not found at {self.manifest_path}. "
                "Run main_store_cav.py first or call CAVRegistry.create_empty_manifest()."
            )
        with open(self.manifest_path, "r") as fh:
            self._manifest = json.load(fh)

    def _save_manifest(self):
        os.makedirs(self.cav_store_root, exist_ok=True)
        with open(self.manifest_path, "w") as fh:
            json.dump(self._manifest, fh, indent=2)

    @classmethod
    def create_empty_manifest(cls, cav_store_root: str) -> "CAVRegistry":
        """Bootstrap a new cav_store with an empty manifest."""
        os.makedirs(cav_store_root, exist_ok=True)
        manifest_path = os.path.join(cav_store_root, "manifest.json")
        skeleton = {
            "schema_version": SCHEMA_VERSION,
            "concepts": {},
            "models": {},
            "layer_groups": {},
        }
        with open(manifest_path, "w") as fh:
            json.dump(skeleton, fh, indent=2)
        return cls(cav_store_root)

    def reload_manifest(self):
        """Re-read manifest from disk (useful after external updates)."""
        self._load_manifest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a concept/layer name for use as a folder/file name."""
        return str(name).replace(os.sep, "_").replace("/", "_")

    def _concept_dir(self, model: str, concept: str) -> str:
        return os.path.join(self.cav_store_root, model, self._safe_name(concept))

    def _layer_file(self, model: str, concept: str, layer: str) -> str:
        safe_layer = self._safe_name(layer)
        return os.path.join(self._concept_dir(model, concept), f"{safe_layer}.joblib")

    def _load_concept_layer(self, model: str, concept: str, layer: str) -> dict:
        """Load the single-concept payload stored at <model>/<concept>/<layer>.joblib."""
        key = (model, concept, layer)
        if key not in self._cache:
            path = self._layer_file(model, concept, layer)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"CAV file not found  model='{model}', concept='{concept}', "
                    f"layer='{layer}': {path}"
                )
            self._cache[key] = joblib.load(path)
        return self._cache[key]

    def _resolve_concept_name(self, concept_or_alias: str) -> str:
        """Resolve a concept name/alias to its canonical name using the global manifest."""
        concepts = self._manifest.get("concepts", {})
        if concept_or_alias in concepts:
            return concept_or_alias
        for name, info in concepts.items():
            if info.get("alias") == concept_or_alias:
                return name
        # Fall back to using the value as-is (e.g. concept registered only in a
        # per-model manifest, or manifest.json not yet populated).
        return concept_or_alias

    # ------------------------------------------------------------------
    # Core query API
    # ------------------------------------------------------------------

    def get_cav(self, model: str, layer: str, concept: str) -> torch.Tensor:
        """
        Return the CAV vector (torch.Tensor) for a specific model/layer/concept.
        `concept` may be the canonical name or its alias.
        """
        concept_name = self._resolve_concept_name(concept)
        data = self._load_concept_layer(model, concept_name, layer)
        return data["cav_vector"]

    def get_layer_cavs(self, model: str, layer: str) -> Dict[str, torch.Tensor]:
        """Return {concept_name: cav_tensor} for every concept in a layer."""
        model_dir = os.path.join(self.cav_store_root, model)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"No CAV data found for model='{model}' at {model_dir}"
            )
        result = {}
        for entry in os.listdir(model_dir):
            concept_dir = os.path.join(model_dir, entry)
            if not os.path.isdir(concept_dir):
                continue
            layer_path = self._layer_file(model, entry, layer)
            if os.path.isfile(layer_path):
                data = self._load_concept_layer(model, entry, layer)
                result[data.get("concept", entry)] = data["cav_vector"]
        return result

    def get_group_cavs(
        self, group_name: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Return CAVs for all layers belonging to a named group.
        Returns: {layer_name: {concept_name: cav_tensor}}
        """
        groups = self._manifest.get("layer_groups", {})
        if group_name not in groups:
            raise KeyError(
                f"Layer group '{group_name}' not found. "
                f"Available groups: {list(groups.keys())}"
            )
        group = groups[group_name]
        model = group["model"]
        return {layer: self.get_layer_cavs(model, layer) for layer in group["layers"]}

    def get_cav_for_group_and_concept(
        self, group_name: str, concept: str
    ) -> Dict[str, torch.Tensor]:
        """
        Return {layer_name: cav_tensor} for a single concept across all layers
        in a named group.  Useful for recalibration loops.
        """
        group_cavs = self.get_group_cavs(group_name)
        concept_name = self._resolve_concept_name(concept)
        result = {}
        for layer, concept_map in group_cavs.items():
            key = concept_name if concept_name in concept_map else None
            if key is None:
                # Fall back to matching by alias in case folder name != canonical name.
                for name in concept_map:
                    if name == concept or self.get_concept_alias(name) == concept:
                        key = name
                        break
            if key is not None:
                result[layer] = concept_map[key]
        return result

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        return list(self._manifest.get("models", {}).keys())

    def list_concepts(self) -> List[str]:
        return list(self._manifest.get("concepts", {}).keys())

    def list_layers(self, model: str) -> List[str]:
        models = self._manifest.get("models", {})
        if model not in models:
            raise KeyError(f"Model '{model}' not in manifest.")
        return models[model].get("computed_layers", [])

    def list_groups(self) -> List[str]:
        return list(self._manifest.get("layer_groups", {}).keys())

    def resolve_concept(self, alias_or_name: str) -> str:
        """Return canonical concept name from name or alias."""
        concepts = self._manifest.get("concepts", {})
        if alias_or_name in concepts:
            return alias_or_name
        for name, info in concepts.items():
            if info.get("alias") == alias_or_name:
                return name
        raise KeyError(f"Concept '{alias_or_name}' not found in manifest.")

    def get_concept_alias(self, concept_name: str) -> str:
        return self._manifest["concepts"][concept_name].get("alias", concept_name)

    # ------------------------------------------------------------------
    # Extensibility � add concepts / groups without re-running compute
    # ------------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        alias: str,
        data_path: str,
        description: str = "",
        random_folder: str = "",
        save: bool = True,
    ):
        """Register a new concept in the manifest (does not compute CAVs)."""
        self._manifest.setdefault("concepts", {})[name] = {
            "alias": alias,
            "data_path": data_path,
            "random_folder": random_folder,
            "description": description,
        }
        if save:
            self._save_manifest()

    def add_layer_group(
        self,
        group_name: str,
        model: str,
        layers: List[str],
        description: str = "",
        save: bool = True,
    ):
        """Define or update a named layer group for recalibration."""
        available = self.list_layers(model)
        missing = [l for l in layers if l not in available]
        if missing:
            raise ValueError(
                f"Layers {missing} have no computed CAVs for model '{model}'. "
                f"Run main_store_cav.py first."
            )
        self._manifest.setdefault("layer_groups", {})[group_name] = {
            "model": model,
            "layers": layers,
            "description": description,
        }
        if save:
            self._save_manifest()

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    def invalidate_cache(
        self,
        model: Optional[str] = None,
        concept: Optional[str] = None,
        layer: Optional[str] = None,
    ):
        if model is None:
            self._cache.clear()
        else:
            to_remove = [
                k for k in self._cache
                if k[0] == model
                and (concept is None or k[1] == concept)
                and (layer is None or k[2] == layer)
            ]
            for k in to_remove:
                del self._cache[k]

    # ------------------------------------------------------------------
    # Per-model manifest ("<model_name>_manifest.json")
    # ------------------------------------------------------------------

    def _model_manifest_path(self, model: str) -> str:
        return os.path.join(self.cav_store_root, f"{model}_manifest.json")

    def _load_model_manifest_file(self, model: str) -> dict:
        path = self._model_manifest_path(model)
        if os.path.isfile(path):
            with open(path, "r") as fh:
                return json.load(fh)
        return {
            "schema_version": SCHEMA_VERSION,
            "model_name": model,
            "model_weight_path": "",
            "created_at": datetime.now().isoformat(),
            "concepts": {},
        }

    def _save_model_manifest_file(self, model: str, data: dict):
        os.makedirs(self.cav_store_root, exist_ok=True)
        with open(self._model_manifest_path(model), "w") as fh:
            json.dump(data, fh, indent=2)

    def get_model_manifest(self, model: str) -> dict:
        """Return the parsed contents of <cav_store_root>/<model>_manifest.json."""
        return self._load_model_manifest_file(model)

    def _update_model_manifest_entry(
        self,
        model: str,
        concept: str,
        layer: str,
        data_path: str,
        model_weight_path: str = "",
        alias: str = "",
        source_images_path: str = "",
        class_name: str = "",
        random_folder: str = "",
    ):
        """
        Create/update the required minimum fields for one concept inside
        <cav_store_root>/<model>_manifest.json:
            model_name, concept_name, data_path, model_weight_path, random_folder
        """
        manifest = self._load_model_manifest_file(model)
        manifest["model_name"] = model
        if model_weight_path:
            manifest["model_weight_path"] = model_weight_path
        manifest["updated_at"] = datetime.now().isoformat()

        concepts = manifest.setdefault("concepts", {})
        entry = concepts.setdefault(concept, {
            "concept_name": concept,
            "alias": alias or concept,
            "data_path": data_path,
            "model_weight_path": model_weight_path,
            "source_images_path": source_images_path,
            "random_folder": random_folder,
            "class_name": class_name,
            "layers": [],
            "created_at": datetime.now().isoformat(),
        })
        entry["concept_name"] = concept
        entry["data_path"] = data_path
        if alias:
            entry["alias"] = alias
        if model_weight_path:
            entry["model_weight_path"] = model_weight_path
        if source_images_path:
            entry["source_images_path"] = source_images_path
        if random_folder:
            entry["random_folder"] = random_folder
        if class_name:
            entry["class_name"] = class_name
        if layer not in entry.setdefault("layers", []):
            entry["layers"].append(layer)

        self._save_model_manifest_file(model, manifest)

    # ------------------------------------------------------------------
    # Persistence helpers used by main_store_cav.py
    # ------------------------------------------------------------------

    def save_concept_layer_cav(
        self,
        model: str,
        concept: str,
        layer: str,
        cav_vector: torch.Tensor,
        alias: str = "",
        model_weight_path: str = "",
        linear_classifier_type: str = "",
        random_folder: str = "",
        source_images_path: str = "",
        class_name: str = "",
    ) -> str:
        """
        Persist a single concept's CAV for a single layer at:
            <cav_store_root>/<model>/<concept>/<layer>.joblib
        and update <cav_store_root>/<model>_manifest.json.

        Parameters
        ----------
        model               : Model name, e.g. "vgg16".
        concept             : Concept name, e.g. "coat" or "deer__coat".
        layer               : Layer name, e.g. "features.11".
        cav_vector          : torch.Tensor CAV vector for this concept/layer.
        alias               : Friendly display name for the concept.
        model_weight_path   : Complete path to the model weights used to capture the CAV.
        linear_classifier_type : Classifier used (LinearSVC, LogisticRegression, SGDClassifier).
        random_folder       : Path to the random-concept folder used during training.
        source_images_path  : (optional) folder of concept images used to compute the CAV.
        class_name          : (optional) originating class name (multiconcept runs).

        Returns
        -------
        str : absolute path of the saved joblib file.

        Example (interactive Python prompt)
        ------------------------------------
        >>> import torch
        >>> from cav_registry import CAVRegistry
        >>> registry = CAVRegistry("./cav_store")
        >>> registry.save_concept_layer_cav(
        ...     model="vgg16",
        ...     concept="deer__coat",
        ...     layer="features.11",
        ...     cav_vector=torch.randn(512),
        ...     alias="coat",
        ...     model_weight_path="C:/models/vgg16/vgg16.pt",
        ...     linear_classifier_type="SGDClassifier",
        ...     random_folder="C:/data/random",
        ...     source_images_path="C:/data/concepts/deer/coat",
        ...     class_name="deer",
        ... )
        """
        payload = {
            "schema_version": LAYER_FILE_SCHEMA_VERSION,
            "model": model,
            "concept": concept,
            "layer_name": layer,
            "cav_vector": cav_vector,
            "alias": alias or concept,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "linear_classifier_type": linear_classifier_type,
                "random_folder": random_folder,
            },
        }
        dest = self._layer_file(model, concept, layer)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        joblib.dump(payload, dest)

        # Update the global manifest.json bookkeeping (models / concepts / layer_groups).
        models_entry = self._manifest.setdefault("models", {}).setdefault(model, {
            "alias": model,
            "weight_path": model_weight_path,
            "computed_layers": [],
        })
        if model_weight_path:
            models_entry["weight_path"] = model_weight_path
        if layer not in models_entry.get("computed_layers", []):
            models_entry.setdefault("computed_layers", []).append(layer)

        if concept not in self._manifest.get("concepts", {}):
            self._manifest.setdefault("concepts", {})[concept] = {
                "alias": alias or concept,
                "data_path": source_images_path or os.path.dirname(dest),
                "random_folder": random_folder,
                "description": f"class={class_name}" if class_name else "",
            }
        else:
            # Keep the recorded random_folder current even if the concept entry
            # already existed (e.g. from an earlier layer of the same concept).
            if random_folder:
                self._manifest["concepts"][concept]["random_folder"] = random_folder
        self._save_manifest()

        # Update the required per-model manifest ("<model>_manifest.json").
        self._update_model_manifest_entry(
            model=model,
            concept=concept,
            layer=layer,
            data_path=os.path.dirname(dest),
            model_weight_path=model_weight_path,
            alias=alias or concept,
            source_images_path=source_images_path,
            class_name=class_name,
            random_folder=random_folder,
        )

        self.invalidate_cache(model, concept, layer)
        return dest

    def save_layer_cav(
        self,
        model: str,
        layer: str,
        concept_cavs: Dict[str, torch.Tensor],
        concept_aliases: Dict[str, str],
        linear_classifier_type: str = "",
        random_folder: str = "",
        concept_random_folders: Optional[Dict[str, str]] = None,
        model_weight_path: str = "",
        concept_data_paths: Optional[Dict[str, str]] = None,
        concept_class_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Backward-compatible convenience wrapper around save_concept_layer_cav().
        Persists one CAV file per concept under:
            <cav_store_root>/<model>/<concept>/<layer>.joblib

        Parameters
        ----------
        model               : Model name, e.g. "vgg16".
        layer               : Layer name, e.g. "features.11".
        concept_cavs        : {concept_name: cav_tensor}
        concept_aliases     : {concept_name: alias_string}
        linear_classifier_type : Classifier used (LinearSVC, LogisticRegression, SGDClassifier).
        random_folder       : Path to the random-concept folder used during training. Used as a
                              fallback for any concept not present in `concept_random_folders`.
        concept_random_folders : (optional) {concept_name: random_folder_path}. Lets each concept
                              be paired with its own random folder (e.g. one random folder per
                              multiconcept target_folder). Takes precedence over `random_folder`.
        model_weight_path   : Complete path to the model weights used to capture the CAV.
        concept_data_paths  : (optional) {concept_name: source_images_path}
        concept_class_names : (optional) {concept_name: class_name}

        Returns
        -------
        dict : {concept_name: absolute_path_of_saved_file}
        """
        concept_data_paths = concept_data_paths or {}
        concept_class_names = concept_class_names or {}
        concept_random_folders = concept_random_folders or {}
        saved_paths = {}
        for name, tensor in concept_cavs.items():
            saved_paths[name] = self.save_concept_layer_cav(
                model=model,
                concept=name,
                layer=layer,
                cav_vector=tensor,
                alias=concept_aliases.get(name, name),
                model_weight_path=model_weight_path,
                linear_classifier_type=linear_classifier_type,
                random_folder=concept_random_folders.get(name, random_folder),
                source_images_path=concept_data_paths.get(name, ""),
                class_name=concept_class_names.get(name, ""),
            )
        return saved_paths

    def update_model_meta(
        self,
        model: str,
        alias: str = "",
        weight_path: str = "",
        save: bool = True,
    ):
        """Set or update display metadata for a model entry in the manifest."""
        entry = self._manifest.setdefault("models", {}).setdefault(model, {
            "alias": alias or model,
            "weight_path": weight_path,
            "computed_layers": [],
        })
        if alias:
            entry["alias"] = alias
        if weight_path:
            entry["weight_path"] = weight_path
        if save:
            self._save_manifest()


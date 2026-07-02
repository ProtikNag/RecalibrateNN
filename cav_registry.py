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

Directory layout:
    <cav_store_root>/
        manifest.json
        <model_name>/
            <layer_name>.joblib        # one file per layer
            ...
        ...

Each .joblib file schema (schema_version "1.0"):
    {
        "schema_version": "1.0",
        "model": str,
        "layer_name": str,
        "concepts": {
            "<concept_name>": {
                "cav_vector": torch.Tensor,   # shape (activation_dim,)
                "alias": str
            },
            ...
        },
        "metadata": {
            "created_at": str,
            "linear_classifier_type": str,
            "random_folder": str
        }
    }

manifest.json schema:
    {
        "schema_version": "1.0",
        "concepts": {
            "<concept_name>": {
                "alias": str,
                "data_path": str,
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
        self._cache: Dict[Tuple[str, str], dict] = {}
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

    def _layer_file(self, model: str, layer: str) -> str:
        safe_layer = layer.replace(os.sep, "_")
        return os.path.join(self.cav_store_root, model, f"{safe_layer}.joblib")

    def _load_layer(self, model: str, layer: str) -> dict:
        key = (model, layer)
        if key not in self._cache:
            path = self._layer_file(model, layer)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"CAV file not found  model='{model}', layer='{layer}': {path}"
                )
            self._cache[key] = joblib.load(path)
        return self._cache[key]

    def _resolve_concept_key(self, concept_or_alias: str, layer_data: dict) -> str:
        concepts = layer_data.get("concepts", {})
        if concept_or_alias in concepts:
            return concept_or_alias
        for name, info in concepts.items():
            if info.get("alias") == concept_or_alias:
                return name
        raise KeyError(
            f"Concept '{concept_or_alias}' not found. "
            f"Available: {list(concepts.keys())}"
        )

    # ------------------------------------------------------------------
    # Core query API
    # ------------------------------------------------------------------

    def get_cav(self, model: str, layer: str, concept: str) -> torch.Tensor:
        """
        Return the CAV vector (torch.Tensor) for a specific model/layer/concept.
        `concept` may be the canonical name or its alias.
        """
        data = self._load_layer(model, layer)
        key = self._resolve_concept_key(concept, data)
        return data["concepts"][key]["cav_vector"]

    def get_layer_cavs(self, model: str, layer: str) -> Dict[str, torch.Tensor]:
        """Return {concept_name: cav_tensor} for every concept in a layer."""
        data = self._load_layer(model, layer)
        return {name: info["cav_vector"] for name, info in data["concepts"].items()}

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
        result = {}
        for layer, concept_map in group_cavs.items():
            data = self._load_layer(
                self._manifest["layer_groups"][group_name]["model"], layer
            )
            key = self._resolve_concept_key(concept, data)
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
    # Extensibility — add concepts / groups without re-running compute
    # ------------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        alias: str,
        data_path: str,
        description: str = "",
        save: bool = True,
    ):
        """Register a new concept in the manifest (does not compute CAVs)."""
        self._manifest.setdefault("concepts", {})[name] = {
            "alias": alias,
            "data_path": data_path,
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
        layer: Optional[str] = None,
    ):
        if model is None:
            self._cache.clear()
        else:
            to_remove = [
                k for k in self._cache
                if k[0] == model and (layer is None or k[1] == layer)
            ]
            for k in to_remove:
                del self._cache[k]

    # ------------------------------------------------------------------
    # Persistence helpers used by main_store_cav.py
    # ------------------------------------------------------------------

    def save_layer_cav(
        self,
        model: str,
        layer: str,
        concept_cavs: Dict[str, torch.Tensor],
        concept_aliases: Dict[str, str],
        linear_classifier_type: str = "",
        random_folder: str = "",
    ) -> str:
        """
        Persist a layer's CAV data and update the manifest.

        Parameters
        ----------
        model : str
            Model name, e.g. "vgg16".
        layer : str
            Layer name, e.g. "features.11".
        concept_cavs : dict
            {concept_name: cav_tensor}
        concept_aliases : dict
            {concept_name: alias_string}
        linear_classifier_type : str
            Classifier used (LinearSVC, LogisticRegression,SGDClassifier ).
        random_folder : str
            Path to the random-concept folder used during training.

        Returns
        -------
        str : absolute path of the saved file.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "layer_name": layer,
            "concepts": {
                name: {
                    "cav_vector": tensor,
                    "alias": concept_aliases.get(name, name),
                }
                for name, tensor in concept_cavs.items()
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "linear_classifier_type": linear_classifier_type,
                "random_folder": random_folder,
            },
        }
        dest = self._layer_file(model, layer)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        joblib.dump(payload, dest)

        # Update manifest
        models_entry = self._manifest.setdefault("models", {}).setdefault(model, {
            "alias": model,
            "weight_path": "",
            "computed_layers": [],
        })
        if layer not in models_entry.get("computed_layers", []):
            models_entry.setdefault("computed_layers", []).append(layer)

        self._save_manifest()
        # Invalidate stale cache entry
        self.invalidate_cache(model, layer)
        return dest

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

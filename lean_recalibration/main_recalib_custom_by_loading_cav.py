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

import copy
import os
import json
import argparse
import random
from datetime import datetime
from itertools import combinations

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd

import sys
sys.path.append("../")

from logger import Logger_Singleton
from custom_dataloader import MultiClassImageDataset
from ConfigSingleton import ConfigSingleton
from cav_registry import CAVRegistry
from utils import (
    get_class_folder_dicts,
    evaluate_accuracy,
    plot_loss_figure,
    save_statistics,
    compute_avg_confidence,
    get_model_weight_path,
    get_base_model_image_size,
    get_model_layers,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

RANDOM_STATE = 132


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    np.random.seed(RANDOM_STATE + worker_id)
    random.seed(RANDOM_STATE + worker_id)


set_seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Activation hook — no global state; stores into a caller-supplied dict
# ---------------------------------------------------------------------------

class ActivationHook:
    """Captures activations for a specific layer into a provided dictionary."""
    def __init__(self, layer_name: str, store: dict):
        self.layer_name = layer_name
        self.store = store
    
    def __call__(self, model, inputs, output):
        self.store[self.layer_name] = output


def get_activation(layer_name: str, store: dict):
    """Returns an ActivationHook callable for capturing layer outputs."""
    return ActivationHook(layer_name, store)


def _extract_images_and_labels(batch):
    if isinstance(batch, (tuple, list)):
        imgs = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return imgs, labels
    return batch, None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(model, validation_loader, target_idx_list, logging):
    try:
        accuracy, precision, recall, f1, class_results = evaluate_accuracy(
            model, validation_loader
        )
        avg_conf = compute_avg_confidence(model, validation_loader, target_idx_list)
        return accuracy, precision, recall, f1, avg_conf, class_results
    except Exception as exc:
        logging.error(f"Metrics computation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# CAV loading via registry
# ---------------------------------------------------------------------------

def _load_target_cavs(
    registry: CAVRegistry,
    model_name: str,
    layer_names: list,
    concept_name: str,
    logging,
) -> dict:
    target_cavs = {}
    for layer in layer_names:
        try:
            raw = registry.get_cav(model_name, layer, concept_name)
            if not isinstance(raw, torch.Tensor):
                raw = torch.tensor(raw, dtype=torch.float32)
            cav = F.normalize(
                raw.to(DEVICE, dtype=torch.float32).view(1, -1), p=2, dim=1
            )
            target_cavs[layer] = cav
            logging.info(
                f"  CAV loaded: model={model_name} layer={layer} "
                f"concept={concept_name} shape={cav.shape}"
            )
        except (FileNotFoundError, KeyError) as exc:
            logging.warning(
                f"  CAV not found model={model_name} layer={layer} "
                f"concept={concept_name}: {exc}"
            )
    return target_cavs


def _load_class_concept_cavs(
    registry: CAVRegistry,
    model_name: str,
    layer_names: list,
    concept_list: list,
    logging,
) -> dict:
    """
    Load CAVs for every concept belonging to a class, keyed by concept name:
        {concept_name: {layer_name: cav_tensor}}
    Concepts with no CAVs found for any of `layer_names` are omitted entirely.
    """
    concept_cavs = {}
    for concept_name in concept_list:
        cavs = _load_target_cavs(registry, model_name, layer_names, concept_name, logging)
        if cavs:
            concept_cavs[concept_name] = cavs
        else:
            logging.warning(
                f"  No CAVs found for concept={concept_name} model={model_name} "
                f"layers={layer_names} — this concept will be skipped."
            )
    return concept_cavs


# ---------------------------------------------------------------------------
# Per-model manifest loading (<model_name>_manifest.json) and class/concept
# grouping for targeted, weighted multi-concept recalibration.
# ---------------------------------------------------------------------------

def load_model_manifest(manifest_path: str) -> dict:
    """
    Load a per-model manifest file, e.g. "vgg16_manifest.json", as produced by
    CAVRegistry.save_concept_layer_cav()/save_layer_cav() in main_store_cav.py.

    Returns the parsed manifest dict with (at least) a "concepts" key mapping
    concept_name -> {class_name, layers, data_path, random_folder, ...}.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(manifest_path, "r") as fh:
        return json.load(fh)


def build_class_concepts_map(
    manifest_data: dict,
    class_names: list,
    target_idx_list: list,
) -> dict:
    """
    Group every concept in the manifest by its recorded "class_name", matching
    against `class_names[target_idx]` for each target_idx in `target_idx_list`.

    Returns {target_idx: [concept_name, ...]} (insertion order preserved, so
    config-driven weighting can be applied deterministically).
    """
    concepts = manifest_data.get("concepts", {})
    class_concepts_map = {idx: [] for idx in target_idx_list}
    for concept_name, info in concepts.items():
        concept_class_name = info.get("class_name", "")
        for target_idx in target_idx_list:
            if target_idx < len(class_names) and class_names[target_idx] == concept_class_name:
                class_concepts_map[target_idx].append(concept_name)
    return class_concepts_map



# ---------------------------------------------------------------------------
# TCAV scores
# ---------------------------------------------------------------------------

def _compute_tcav_scores(
    model,
    target_cavs: dict,
    loader,
    target_idx: int,
    logging,
) -> dict:
    scores = {}
    activation_dict = {}
    hook_handles = []
    try:
        for ln in target_cavs:
            h = model.get_submodule(ln).register_forward_hook(
                get_activation(ln, activation_dict)
            )
            hook_handles.append(h)

        model.eval()
        for ln, cav in target_cavs.items():
            layer_scores = []
            for batch in loader:
                imgs, _ = _extract_images_and_labels(batch)
                imgs = imgs.to(DEVICE)
                with torch.enable_grad():
                    outputs = model(imgs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    if ln not in activation_dict:
                        raise KeyError(f"Missing activation for layer '{ln}'.")
                    f_l = activation_dict[ln]
                    h_k = outputs[:, target_idx]
                    grad = torch.autograd.grad(
                        h_k.sum(), f_l,
                        retain_graph=False, allow_unused=True
                    )[0]
                    if grad is None:
                        layer_scores.append(
                            torch.zeros(imgs.size(0), dtype=torch.bool, device=DEVICE)
                        )
                    else:
                        grad_flat = grad.detach().view(grad.size(0), -1)
                        grad_norm = F.normalize(grad_flat, p=2, dim=1)
                        sensitivity = torch.sum(grad_norm * cav, dim=1)
                        layer_scores.append((sensitivity > 0).detach())
                        del grad, grad_flat, grad_norm, sensitivity
                activation_dict.clear()
                del imgs, outputs, h_k

            if layer_scores:
                tcav = torch.cat(layer_scores).float().mean().item()
                scores[ln] = tcav
                logging.info(
                    f"  TCAV layer={ln} target={target_idx} score={tcav:.4f}"
                )
                print(
                    f"  TCAV layer={ln} target={target_idx} score={tcav:.4f}"
                )
    finally:
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        activation_dict.clear()
    return scores


# ---------------------------------------------------------------------------
# Baseline metrics before recalibration
# ---------------------------------------------------------------------------

def _save_before_metrics(
    model,
    layer_names: list,
    registry: CAVRegistry,
    model_name: str,
    class_concepts_map: dict,
    validation_loader,
    target_idx_list: list,
    results_path: str,
    logging,
):
    acc, prec, rec, f1, conf, class_res = _compute_metrics(
        model, validation_loader, target_idx_list, logging
    )
    logging.info(
        f"Before acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
    )
    print(f"Before acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    # tcav_before: {target_idx: {concept_name: {layer: score}}}
    tcav_before = {}
    for target_idx in target_idx_list:
        concept_list = class_concepts_map.get(target_idx, [])
        if not concept_list:
            continue
        tcav_before[target_idx] = {}
        for concept_name in concept_list:
            try:
                cavs = _load_target_cavs(
                    registry, model_name, layer_names, concept_name, logging
                )
                if cavs:
                    tcav_before[target_idx][concept_name] = _compute_tcav_scores(
                        model, cavs, validation_loader, target_idx, logging
                    )
                del cavs
            except Exception as exc:
                logging.warning(
                    f"TCAV before-metrics failed target={target_idx} "
                    f"concept={concept_name}: {exc}"
                )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    out_path = os.path.join(results_path, "accuracy_results_before.txt")
    with open(out_path, "w") as fh:
        fh.write("Accuracy Metrics Before Recalibration\n")
        fh.write("=" * 50 + "\n")
        fh.write(f"Accuracy : {acc:.4f}\n")
        fh.write(f"Precision: {prec:.4f}\n")
        fh.write(f"Recall   : {rec:.4f}\n")
        fh.write(f"F1 Score : {f1:.4f}\n")
        if isinstance(conf, dict):
            fh.write(f"Avg Conf : {conf}\n")
        if isinstance(class_res, dict):
            fh.write("\nClass-wise Results:\n")
            for cls_name, res in class_res.items():
                fh.write(
                    f"  {cls_name}: correct={res['correct']} total={res['total']}\n"
                )
        fh.write("\nTCAV Scores Before Recalibration:\n")
        for tidx, concept_scores in tcav_before.items():
            for concept_name, layer_scores in concept_scores.items():
                for ln, s in layer_scores.items():
                    fh.write(
                        f"  target={tidx} concept={concept_name} layer={ln}: {s:.4f}\n"
                    )
    logging.info(f"Before-metrics saved: {out_path}")
    print(f"Before-metrics saved: {out_path}")


# ---------------------------------------------------------------------------
# After-recalibration metrics computation and Excel summary
# ---------------------------------------------------------------------------

def _compute_after_metrics(
    best_models_per_class: dict,
    registry: CAVRegistry,
    model_name: str,
    class_concepts_map: dict,
    validation_loader,
    target_idx_list: list,
    results_path: str,
    logging,
) -> tuple:
    """
    Load best model for each target class, compute accuracy and TCAV scores.
    
    best_models_per_class: {target_class: {best_model_path, metrics}}
    
    Returns: (after_metrics_dict, after_tcav_dict)
             after_tcav_dict: {target_class: {concept_name: {layer: score}}}
    """
    after_metrics = {}
    after_tcav = {}
    
    for target_class in target_idx_list:
        if target_class not in best_models_per_class:
            logging.warning(f"No best model found for class {target_class}")
            continue
        
        model_info = best_models_per_class[target_class]
        best_model_path = model_info.get("path")
        
        if not os.path.isfile(best_model_path):
            logging.warning(f"Best model file not found: {best_model_path}")
            continue
        
        logging.info(f"Loading best model for class {target_class}: {best_model_path}")
        print(f"Loading best model for class {target_class}...")
        
        try:
            if DEVICE == "cpu":
                best_model = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
            else:
                best_model = torch.load(best_model_path, map_location=DEVICE)
            best_model.to(DEVICE)
            best_model.eval()
            
            # Compute metrics
            acc, prec, rec, f1, conf, class_res = _compute_metrics(
                best_model, validation_loader, [target_class], logging
            )
            after_metrics[target_class] = {
                "accuracy": round(float(acc), 6),
                "precision": round(float(prec), 6),
                "recall": round(float(rec), 6),
                "f1": round(float(f1), 6),
            }
            logging.info(
                f"After — class {target_class}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
            )
            
            # Compute TCAV scores for every concept belonging to this class
            concept_list = class_concepts_map.get(target_class, [])
            if concept_list:
                after_tcav[target_class] = {}
                for concept_name in concept_list:
                    try:
                        cavs = _load_target_cavs(
                            registry, model_name, model_info.get("layers", []),
                            concept_name, logging
                        )
                        if cavs:
                            after_tcav[target_class][concept_name] = _compute_tcav_scores(
                                best_model, cavs, validation_loader, target_class, logging
                            )
                        del cavs
                    except Exception as exc:
                        logging.warning(
                            f"TCAV after-metrics failed class {target_class} "
                            f"concept={concept_name}: {exc}"
                        )
            
            # Free model to prevent CUDA OOM
            del best_model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        except Exception as exc:
            logging.error(f"Failed to compute after-metrics for class {target_class}: {exc}")
            print(f"Error: {exc}")
    
    return after_metrics, after_tcav


def _create_summary_excel(
    before_metrics: dict,
    before_tcav: dict,
    after_metrics: dict,
    after_tcav: dict,
    target_idx_list: list,
    target_class_names: list,
    class_concepts_map: dict,
    results_path: str,
    model_name: str,
    logging,
    concept_weights_map: dict = None,
    class_weights_map: dict = None,
):
    """
    Create an Excel summary with before/after metrics and TCAV scores.
    
    before_metrics: {target_class: {accuracy, precision, recall, f1}}
    before_tcav: {target_class: {concept_name: {layer_name: score}}}
    after_metrics: {target_class: {accuracy, precision, recall, f1}}
    after_tcav: {target_class: {concept_name: {layer_name: score}}}
    target_class_names: human-readable class names
    class_concepts_map: {target_class: [concept_name, ...]}
    concept_weights_map: (optional) {target_class: {concept_name: weight}}
    class_weights_map: (optional) {target_class: class_weight}
    """
    concept_weights_map = concept_weights_map or {}
    class_weights_map = class_weights_map or {}
    summary_data = []

    def _flatten_tcav(tcav_by_concept: dict) -> list:
        """Flatten {concept_name: {layer: score}} into a flat list of scores."""
        vals = []
        for layer_scores in (tcav_by_concept or {}).values():
            vals.extend(layer_scores.values())
        return vals

    for idx, target_class in enumerate(target_idx_list):
        class_name = target_class_names[idx] if idx < len(target_class_names) else f"class_{target_class}"
        concept_list = class_concepts_map.get(target_class, [])
        concept_weights = concept_weights_map.get(target_class, {})
        class_weight = class_weights_map.get(target_class, 1.0)
        concepts_str = ", ".join(
            f"{c}({concept_weights.get(c, 0):.2f})" if concept_weights else c
            for c in concept_list
        ) or "unknown"

        before_acc = before_metrics.get(target_class, {}).get("accuracy", "N/A")
        before_prec = before_metrics.get(target_class, {}).get("precision", "N/A")
        before_rec = before_metrics.get(target_class, {}).get("recall", "N/A")
        before_f1 = before_metrics.get(target_class, {}).get("f1", "N/A")
        
        after_acc = after_metrics.get(target_class, {}).get("accuracy", "N/A")
        after_prec = after_metrics.get(target_class, {}).get("precision", "N/A")
        after_rec = after_metrics.get(target_class, {}).get("recall", "N/A")
        after_f1 = after_metrics.get(target_class, {}).get("f1", "N/A")
        
        # Average TCAV before (flattened across all concepts and layers for this class)
        before_tcav_vals = _flatten_tcav(before_tcav.get(target_class, {}))
        avg_tcav_before = (
            round(sum(before_tcav_vals) / len(before_tcav_vals), 6)
            if before_tcav_vals
            else "N/A"
        )
        
        # Average TCAV after
        after_tcav_vals = _flatten_tcav(after_tcav.get(target_class, {}))
        avg_tcav_after = (
            round(sum(after_tcav_vals) / len(after_tcav_vals), 6)
            if after_tcav_vals
            else "N/A"
        )
        
        # Accuracy improvement
        acc_improvement = "N/A"
        if isinstance(before_acc, (int, float)) and isinstance(after_acc, (int, float)):
            acc_improvement = round(after_acc - before_acc, 6)
        
        summary_data.append({
            "Class Index": target_class,
            "Class Name": class_name,
            "Concepts": concepts_str,
            "Class Weight": class_weight,
            "Before Accuracy": before_acc,
            "After Accuracy": after_acc,
            "Accuracy Improvement": acc_improvement,
            "Before Precision": before_prec,
            "After Precision": after_prec,
            "Before Recall": before_rec,
            "After Recall": after_rec,
            "Before F1": before_f1,
            "After F1": after_f1,
            "Avg TCAV Before": avg_tcav_before,
            "Avg TCAV After": avg_tcav_after,
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(summary_data)
    excel_path = os.path.join(results_path, f"recalibration_summary_{model_name}.xlsx")
    
    try:
        # Use pandas ExcelWriter for more control
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Add formatting (optional, requires openpyxl)
            try:
                from openpyxl.styles import PatternFill, Font, Alignment
                workbook = writer.book
                worksheet = writer.sheets["Summary"]
                
                # Header formatting
                header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF")
                
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except Exception:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            except Exception as fmt_exc:
                logging.warning(f"Could not apply Excel formatting: {fmt_exc}")
        
        logging.info(f"Summary Excel saved: {excel_path}")
        print(f"Summary Excel saved: {excel_path}")
    except ImportError:
        logging.error("openpyxl not installed. Installing...")
        os.system("pip install openpyxl")
        # Retry
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Summary", index=False)
            logging.info(f"Summary Excel saved: {excel_path}")
            print(f"Summary Excel saved: {excel_path}")
        except Exception as exc:
            logging.error(f"Failed to save Excel: {exc}")
            # Fallback: save as CSV
            csv_path = excel_path.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"Fallback: saved as CSV: {csv_path}")
            print(f"Fallback: saved as CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Single lambda training pass — one deep copy, explicit CUDA cleanup
# ---------------------------------------------------------------------------

def _run_one_lambda(
    base_model,
    base_model_name: str,
    layer_names: list,
    concept_cavs: dict,
    concept_weights: dict,
    class_weight: float,
    training_loader,
    validation_loader,
    target_class: int,
    lambda_align: float,
    epochs: int,
    lr: float,
    results_path: str,
    model_save_name: str,
    save_plots: bool,
    logging,
) -> dict:
    """
    concept_cavs    : {concept_name: {layer_name: cav_tensor}} — every concept
                      that belongs to `target_class`, each possibly aligned at
                      a different subset of `layer_names`.
    concept_weights : {concept_name: float} — relative weight of each concept's
                      alignment loss (typically normalized to sum to 1.0).
    class_weight    : float — overall scaling applied to the combined
                      (weighted) alignment loss for this class.
    """
    lambda_cls = round(1.0 - lambda_align, 2)
    logging.info(f"  lambda_align={lambda_align} lambda_cls={lambda_cls}")
    print(f"  lambda_align={lambda_align} lambda_cls={lambda_cls}")
    logging.info(
        f"  class_weight={class_weight} concept_weights={concept_weights}"
    )

    model_trained = copy.deepcopy(base_model).to(DEVICE)
    activation_dict = {}
    hook_handles = []
    run_result = {}

    # Union of every layer referenced by any concept — one hook per unique layer.
    all_layers = sorted({ln for layer_cavs in concept_cavs.values() for ln in layer_cavs})

    try:
        for ln in all_layers:
            h = model_trained.get_submodule(ln).register_forward_hook(
                get_activation(ln, activation_dict)
            )
            hook_handles.append(h)

        for pname, param in model_trained.named_parameters():
            param.requires_grad = any(ln in pname for ln in layer_names)
        for module in model_trained.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                module.eval()

        trainable_params = [p for p in model_trained.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(f"No trainable parameters for layers {layer_names}.")

        optimizer = optim.Adam(trainable_params, lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        loss_history = {"total": [], "cls": [], "align": []}
        model_trained.train()

        for epoch in range(epochs):
            total_l = cls_l_sum = align_l_sum = 0.0
            n_batches = 0

            for imgs, labels in training_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                outputs = model_trained(imgs)
                if isinstance(outputs, (tuple, list)):
                    main_out = outputs[0]
                    aux_out = outputs[1] if len(outputs) > 1 else None
                else:
                    main_out = outputs
                    aux_out = None

                if base_model_name == "inception_v3" and aux_out is not None:
                    cls_loss = (
                        0.6 * ce_loss_fn(main_out, labels)
                        + 0.4 * ce_loss_fn(aux_out, labels)
                    )
                else:
                    cls_loss = ce_loss_fn(main_out, labels)

                align_loss = torch.tensor(0.0, device=DEVICE)
                target_mask = labels == target_class

                if target_mask.any():
                    for concept_name, layer_cavs in concept_cavs.items():
                        concept_weight = concept_weights.get(concept_name, 0.0)
                        if concept_weight == 0.0:
                            continue
                        concept_align_loss = torch.tensor(0.0, device=DEVICE)
                        for ln, cav in layer_cavs.items():
                            if ln not in activation_dict:
                                raise KeyError(f"Missing activation for '{ln}'.")
                            f_l = activation_dict[ln]
                            f_flat = F.normalize(
                                f_l.view(f_l.size(0), -1)[target_mask], p=2, dim=1
                            )
                            if f_flat.size(1) != cav.size(1):
                                raise ValueError(
                                    f"CAV/feature dim mismatch at '{ln}' "
                                    f"(concept={concept_name}): "
                                    f"feat={f_flat.size(1)} cav={cav.size(1)}"
                                )
                            cosine_sim = torch.sum(f_flat * cav, dim=1)
                            concept_align_loss = concept_align_loss + (1 - cosine_sim.mean())
                            del f_l, f_flat, cosine_sim
                        align_loss = align_loss + concept_weight * concept_align_loss
                        del concept_align_loss

                align_loss = class_weight * align_loss
                loss = lambda_align * align_loss + lambda_cls * cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=7.0)
                optimizer.step()

                total_l += loss.item()
                cls_l_sum += cls_loss.item()
                align_l_sum += align_loss.item()
                n_batches += 1

                # Free activation state and all batch tensors before next iteration
                activation_dict.clear()
                del imgs, labels, outputs, main_out, loss, cls_loss, align_loss
                if aux_out is not None:
                    del aux_out

            if n_batches > 0:
                loss_history["total"].append(total_l / n_batches)
                loss_history["cls"].append(cls_l_sum / n_batches)
                loss_history["align"].append(align_l_sum / n_batches)
                msg = (
                    f"  Epoch {epoch+1}/{epochs} "
                    f"total={loss_history['total'][-1]:.4f} "
                    f"align={loss_history['align'][-1]:.4f}"
                )
                logging.info(msg)
                print(msg)

        acc, prec, rec, f1, conf, class_res = _compute_metrics(
            model_trained, validation_loader, [target_class], logging
        )
        logging.info(
            f"  After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )
        print(
            f"  After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )

        run_result = {
            "Lambda Alignment": lambda_align,
            "Lambda Classification": lambda_cls,
            "Class Weight": class_weight,
            "Concept Weights": str(concept_weights),
            "Accuracy": round(float(acc), 6),
            "Precision": round(float(prec), 6),
            "Recall": round(float(rec), 6),
            "F1 Score": round(float(f1), 6),
        }
        if isinstance(conf, dict):
            run_result.update(
                {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                 for k, v in conf.items()}
            )
        if isinstance(class_res, dict):
            for cid, cm in class_res.items():
                run_result[f"class{cid}_correct"] = cm.get("correct")
                run_result[f"class{cid}_total"] = cm.get("total")

        save_path = os.path.join(results_path, model_save_name)
        torch.save(model_trained, save_path)
        logging.info(f"  Saved model: {save_path}")

        if save_plots:
            layers_str = "_".join(layer_names)
            plot_loss_figure(
                loss_history["total"],
                loss_history["align"],
                loss_history["cls"],
                epochs,
                os.path.join(
                    results_path,
                    f"loss_cls_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_align_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_total_{base_model_name}_c{target_class}_{layers_str}_{lambda_align}.pdf",
                ),
            )

    except Exception as exc:
        logging.error(f"  Error at lambda={lambda_align}: {exc}")
        print(f"  Error at lambda={lambda_align}: {exc}")

    finally:
        # Always remove hooks and free model copy — prevents CUDA OOM across lambda iterations
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        activation_dict.clear()
        del model_trained
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return run_result


# ---------------------------------------------------------------------------
# Recalibrate one class across all lambda values for one layer combo
# ---------------------------------------------------------------------------

def _recalibrate_one_class(
    base_model,
    base_model_name: str,
    layer_combo: list,
    registry: CAVRegistry,
    concept_list: list,
    concept_weights: dict,
    class_weight: float,
    training_loader,
    validation_loader,
    target_class: int,
    lambda_aligns: list,
    combo_idx: int,
    epochs: int,
    lr: float,
    results_path: str,
    save_plots: bool,
    logging,
) -> tuple:
    logging.info(
        f"class={target_class} concepts={concept_list} "
        f"weights={concept_weights} class_weight={class_weight} combo={layer_combo}"
    )
    concept_cavs = _load_class_concept_cavs(
        registry, base_model_name, layer_combo, concept_list, logging
    )
    if not concept_cavs:
        logging.warning(
            f"No CAVs for class={target_class} concepts={concept_list} "
            f"layers={layer_combo} — skipping."
        )
        return [], layer_combo

    run_metrics = []
    for lambda_idx, lambda_align in enumerate(lambda_aligns):
        set_seed(RANDOM_STATE + lambda_idx)
        result = _run_one_lambda(
            base_model=base_model,
            base_model_name=base_model_name,
            layer_names=layer_combo,
            concept_cavs=concept_cavs,
            concept_weights=concept_weights,
            class_weight=class_weight,
            training_loader=training_loader,
            validation_loader=validation_loader,
            target_class=target_class,
            lambda_align=lambda_align,
            epochs=epochs,
            lr=lr,
            results_path=results_path,
            model_save_name=(
                f"model_cls{target_class}_combo{combo_idx}"
                f"_lambda{lambda_align}.pth"
            ),
            save_plots=save_plots,
            logging=logging,
        )
        if result:
            result["Target Class"] = target_class
            result["Layers"] = str(layer_combo)
            result["Combination"] = "|".join(layer_combo)
            result["Layer Count"] = len(layer_combo)
            run_metrics.append(result)

    # Free CAV tensors once all lambdas for this combo are done
    del concept_cavs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Return both metrics and best model info
    return run_metrics, layer_combo


# ---------------------------------------------------------------------------
# Joint multi-class training pass — ONE shared model copy is recalibrated
# simultaneously against every enabled target class's weighted concepts.
# ---------------------------------------------------------------------------

def _run_one_lambda_multiclass(
    base_model,
    base_model_name: str,
    layer_names: list,
    class_concept_cavs: dict,
    concept_weights_map: dict,
    class_weights_map: dict,
    training_loader,
    validation_loader,
    target_idx_list: list,
    lambda_align: float,
    epochs: int,
    lr: float,
    results_path: str,
    model_save_name: str,
    save_plots: bool,
    logging,
) -> dict:
    """
    class_concept_cavs   : {target_class: {concept_name: {layer: cav_tensor}}}
                            — every enabled class's concepts, keyed by class.
    concept_weights_map  : {target_class: {concept_name: weight}}
    class_weights_map    : {target_class: weight}

    Unlike `_run_one_lambda` (which trains one independent model per class),
    this trains a SINGLE `copy.deepcopy(base_model)` whose alignment loss is
    the sum, across every class in `target_idx_list`, of that class's own
    weighted concept-alignment loss (masked to that class's samples in the
    batch), scaled by its `class_weight`. The classification loss is computed
    over the full batch as usual.
    """
    lambda_cls = round(1.0 - lambda_align, 2)
    logging.info(f"  [multiclass] lambda_align={lambda_align} lambda_cls={lambda_cls}")
    print(f"  [multiclass] lambda_align={lambda_align} lambda_cls={lambda_cls}")
    logging.info(
        f"  [multiclass] classes={target_idx_list} "
        f"class_weights={class_weights_map} concept_weights={concept_weights_map}"
    )

    model_trained = copy.deepcopy(base_model).to(DEVICE)
    activation_dict = {}
    hook_handles = []
    run_result = {}

    # Union of every layer referenced by any concept of any class.
    all_layers = sorted({
        ln
        for concept_cavs in class_concept_cavs.values()
        for layer_cavs in concept_cavs.values()
        for ln in layer_cavs
    })

    try:
        for ln in all_layers:
            h = model_trained.get_submodule(ln).register_forward_hook(
                get_activation(ln, activation_dict)
            )
            hook_handles.append(h)

        for pname, param in model_trained.named_parameters():
            param.requires_grad = any(ln in pname for ln in layer_names)
        for module in model_trained.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                module.eval()

        trainable_params = [p for p in model_trained.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(f"No trainable parameters for layers {layer_names}.")

        optimizer = optim.Adam(trainable_params, lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        loss_history = {"total": [], "cls": [], "align": []}
        model_trained.train()

        for epoch in range(epochs):
            total_l = cls_l_sum = align_l_sum = 0.0
            n_batches = 0

            for imgs, labels in training_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                outputs = model_trained(imgs)
                if isinstance(outputs, (tuple, list)):
                    main_out = outputs[0]
                    aux_out = outputs[1] if len(outputs) > 1 else None
                else:
                    main_out = outputs
                    aux_out = None

                if base_model_name == "inception_v3" and aux_out is not None:
                    cls_loss = (
                        0.6 * ce_loss_fn(main_out, labels)
                        + 0.4 * ce_loss_fn(aux_out, labels)
                    )
                else:
                    cls_loss = ce_loss_fn(main_out, labels)

                align_loss = torch.tensor(0.0, device=DEVICE)

                # Sum a weighted alignment loss across EVERY enabled class,
                # each masked to its own samples within this batch.
                for target_class in target_idx_list:
                    concept_cavs = class_concept_cavs.get(target_class, {})
                    if not concept_cavs:
                        continue
                    target_mask = labels == target_class
                    if not target_mask.any():
                        continue

                    concept_weights = concept_weights_map.get(target_class, {})
                    class_weight = class_weights_map.get(target_class, 1.0)

                    class_align_loss = torch.tensor(0.0, device=DEVICE)
                    for concept_name, layer_cavs in concept_cavs.items():
                        concept_weight = concept_weights.get(concept_name, 0.0)
                        if concept_weight == 0.0:
                            continue
                        concept_align_loss = torch.tensor(0.0, device=DEVICE)
                        for ln, cav in layer_cavs.items():
                            if ln not in activation_dict:
                                raise KeyError(f"Missing activation for '{ln}'.")
                            f_l = activation_dict[ln]
                            f_flat = F.normalize(
                                f_l.view(f_l.size(0), -1)[target_mask], p=2, dim=1
                            )
                            if f_flat.size(1) != cav.size(1):
                                raise ValueError(
                                    f"CAV/feature dim mismatch at '{ln}' "
                                    f"(class={target_class}, concept={concept_name}): "
                                    f"feat={f_flat.size(1)} cav={cav.size(1)}"
                                )
                            cosine_sim = torch.sum(f_flat * cav, dim=1)
                            concept_align_loss = concept_align_loss + (1 - cosine_sim.mean())
                            del f_l, f_flat, cosine_sim
                        class_align_loss = class_align_loss + concept_weight * concept_align_loss
                        del concept_align_loss

                    align_loss = align_loss + class_weight * class_align_loss
                    del class_align_loss

                loss = lambda_align * align_loss + lambda_cls * cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=7.0)
                optimizer.step()

                total_l += loss.item()
                cls_l_sum += cls_loss.item()
                align_l_sum += align_loss.item()
                n_batches += 1

                # Free activation state and all batch tensors before next iteration
                activation_dict.clear()
                del imgs, labels, outputs, main_out, loss, cls_loss, align_loss
                if aux_out is not None:
                    del aux_out

            if n_batches > 0:
                loss_history["total"].append(total_l / n_batches)
                loss_history["cls"].append(cls_l_sum / n_batches)
                loss_history["align"].append(align_l_sum / n_batches)
                msg = (
                    f"  [multiclass] Epoch {epoch+1}/{epochs} "
                    f"total={loss_history['total'][-1]:.4f} "
                    f"align={loss_history['align'][-1]:.4f}"
                )
                logging.info(msg)
                print(msg)

        acc, prec, rec, f1, conf, class_res = _compute_metrics(
            model_trained, validation_loader, target_idx_list, logging
        )
        logging.info(
            f"  [multiclass] After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )
        print(
            f"  [multiclass] After: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
        )

        run_result = {
            "Lambda Alignment": lambda_align,
            "Lambda Classification": lambda_cls,
            "Class Weights": str(class_weights_map),
            "Concept Weights": str(concept_weights_map),
            "Accuracy": round(float(acc), 6),
            "Precision": round(float(prec), 6),
            "Recall": round(float(rec), 6),
            "F1 Score": round(float(f1), 6),
        }
        if isinstance(conf, dict):
            run_result.update(
                {k: round(float(v), 6) if isinstance(v, (int, float)) else v
                 for k, v in conf.items()}
            )
        if isinstance(class_res, dict):
            for cid, cm in class_res.items():
                run_result[f"class{cid}_correct"] = cm.get("correct")
                run_result[f"class{cid}_total"] = cm.get("total")

        save_path = os.path.join(results_path, model_save_name)
        torch.save(model_trained, save_path)
        logging.info(f"  [multiclass] Saved model: {save_path}")

        if save_plots:
            layers_str = "_".join(layer_names)
            classes_str = "-".join(str(c) for c in target_idx_list)
            plot_loss_figure(
                loss_history["total"],
                loss_history["align"],
                loss_history["cls"],
                epochs,
                os.path.join(
                    results_path,
                    f"loss_cls_{base_model_name}_multiclass{classes_str}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_align_{base_model_name}_multiclass{classes_str}_{layers_str}_{lambda_align}.pdf",
                ),
                os.path.join(
                    results_path,
                    f"loss_total_{base_model_name}_multiclass{classes_str}_{layers_str}_{lambda_align}.pdf",
                ),
            )

    except Exception as exc:
        logging.error(f"  [multiclass] Error at lambda={lambda_align}: {exc}")
        print(f"  [multiclass] Error at lambda={lambda_align}: {exc}")

    finally:
        # Always remove hooks and free model copy — prevents CUDA OOM across lambda iterations
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        activation_dict.clear()
        del model_trained
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return run_result


# ---------------------------------------------------------------------------
# Recalibrate ALL enabled classes jointly (one shared model) across all
# lambda values, for one layer combo.
# ---------------------------------------------------------------------------

def _recalibrate_multiclass(
    base_model,
    base_model_name: str,
    layer_combo: list,
    registry: CAVRegistry,
    class_concepts_map: dict,
    concept_weights_map: dict,
    class_weights_map: dict,
    training_loader,
    validation_loader,
    target_idx_list: list,
    lambda_aligns: list,
    combo_idx: int,
    epochs: int,
    lr: float,
    results_path: str,
    save_plots: bool,
    logging,
) -> tuple:
    """
    Joint counterpart to `_recalibrate_one_class`: loads every enabled
    class's concept CAVs and trains ONE shared model per lambda value that
    simultaneously targets every class in `target_idx_list`.
    """
    class_concept_cavs = {}
    for target_class in target_idx_list:
        concept_list = class_concepts_map.get(target_class, [])
        if not concept_list:
            logging.warning(
                f"[multiclass] No concepts found for class={target_class} — "
                "it will be excluded from this joint run."
            )
            continue
        cavs = _load_class_concept_cavs(
            registry, base_model_name, layer_combo, concept_list, logging
        )
        if cavs:
            class_concept_cavs[target_class] = cavs
        else:
            logging.warning(
                f"[multiclass] No CAVs for class={target_class} "
                f"layers={layer_combo} — it will be excluded from this joint run."
            )

    if not class_concept_cavs:
        logging.warning(
            f"[multiclass] No CAVs for any class in {target_idx_list} "
            f"layers={layer_combo} — skipping combo."
        )
        return [], layer_combo

    joint_classes = list(class_concept_cavs.keys())
    run_metrics = []
    for lambda_idx, lambda_align in enumerate(lambda_aligns):
        set_seed(RANDOM_STATE + lambda_idx)
        result = _run_one_lambda_multiclass(
            base_model=base_model,
            base_model_name=base_model_name,
            layer_names=layer_combo,
            class_concept_cavs=class_concept_cavs,
            concept_weights_map=concept_weights_map,
            class_weights_map=class_weights_map,
            training_loader=training_loader,
            validation_loader=validation_loader,
            target_idx_list=joint_classes,
            lambda_align=lambda_align,
            epochs=epochs,
            lr=lr,
            results_path=results_path,
            model_save_name=(
                f"model_multiclass_combo{combo_idx}_lambda{lambda_align}.pth"
            ),
            save_plots=save_plots,
            logging=logging,
        )
        if result:
            result["Target Classes"] = str(joint_classes)
            result["Layers"] = str(layer_combo)
            result["Combination"] = "|".join(layer_combo)
            result["Layer Count"] = len(layer_combo)
            run_metrics.append(result)

    # Free CAV tensors once all lambdas for this combo are done
    del class_concept_cavs
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return run_metrics, layer_combo


# ---------------------------------------------------------------------------
# Layer combination / group resolution
# ---------------------------------------------------------------------------

def _resolve_layer_combos(
    layer_names: list,
    use_groups: bool,
    registry: CAVRegistry,
    model_name: str,
) -> list:
    if use_groups:
        groups = registry.list_groups()
        model_groups = [
            g for g in groups
            if registry._manifest["layer_groups"][g]["model"] == model_name
        ]
        if model_groups:
            combos = [
                registry._manifest["layer_groups"][g]["layers"]
                for g in model_groups
            ]
            print(
                f"Using {len(combos)} manifest groups for {model_name}: {model_groups}"
            )
            return combos
        print(
            f"No manifest groups for {model_name}; falling back to combinations."
        )

    combos = []
    for size in (1, 2, 3):
        if len(layer_names) >= size:
            combos.extend(
                [list(c) for c in combinations(layer_names, size)]
            )
    return combos


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    base_model,
    base_model_name: str,
    registry: CAVRegistry,
    class_concepts_map: dict,
    concept_weights_map: dict,
    class_weights_map: dict,
    layer_names: list,
    training_loader,
    validation_loader,
    target_idx_list: list,
    target_class_names: list,
    lambda_aligns: list,
    recalibrate_flags: list,
    epochs: int,
    lr: float,
    results_path: str,
    use_groups: bool,
    save_plots: bool,
    logging,
):
    """
    class_concepts_map  : {target_idx: [concept_name, ...]} — every concept
                          belonging to a class (as read from the per-model
                          manifest, e.g. vgg16_manifest.json).
    concept_weights_map : {target_idx: {concept_name: weight}} — configurable
                          per-concept alignment weights (config file driven).
    class_weights_map   : {target_idx: class_weight} — configurable per-class
                          alignment weight (config file driven).
    """
    set_seed(RANDOM_STATE)
    logging.info("Recalibration started.")
    best_models_per_class = {}  # Track best model per class: {target_class: {path, layers, metrics}}

    # Capture before metrics for later summary
    before_metrics = {}
    before_tcav = {}
    
    # Compute before metrics
    acc, prec, rec, f1, conf, class_res = _compute_metrics(
        base_model, validation_loader, target_idx_list, logging
    )
    for target_idx in target_idx_list:
        before_metrics[target_idx] = {
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
        }
    
    # Compute TCAV scores before recalibration (per concept, per class)
    for target_idx in target_idx_list:
        concept_list = class_concepts_map.get(target_idx, [])
        if not concept_list:
            continue
        before_tcav[target_idx] = {}
        for concept_name in concept_list:
            try:
                cavs = _load_target_cavs(registry, base_model_name, layer_names, concept_name, logging)
                if cavs:
                    before_tcav[target_idx][concept_name] = _compute_tcav_scores(
                        base_model, cavs, validation_loader, target_idx, logging
                    )
                del cavs
            except Exception as exc:
                logging.warning(
                    f"TCAV before-metrics failed target={target_idx} "
                    f"concept={concept_name}: {exc}"
                )
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Save before metrics to text file
    _save_before_metrics(
        base_model, layer_names, registry, base_model_name,
        class_concepts_map, validation_loader, target_idx_list,
        results_path, logging,
    )

    layer_combos = _resolve_layer_combos(
        layer_names, use_groups, registry, base_model_name
    )
    if not layer_combos:
        logging.error("No layer combinations available. Aborting.")
        print("No layer combinations available. Aborting.")
        return

    combo_index = {idx + 1: "|".join(c) for idx, c in enumerate(layer_combos)}
    with open(
        os.path.join(results_path, f"layer_combinations_{base_model_name}.txt"), "w"
    ) as fh:
        fh.write(str(combo_index))

    combo_csv = os.path.join(
        results_path, f"recalibration_combo_accuracy_{base_model_name}.csv"
    )
    all_results = []

    for class_idx, target_class in enumerate(target_idx_list):
        flag = (
            recalibrate_flags[class_idx]
            if class_idx < len(recalibrate_flags)
            else 1
        )
        if flag != 1:
            logging.info(
                f"Skipping class idx={class_idx} target={target_class} (flag=0)"
            )
            print(
                f"Skipping class idx={class_idx} target={target_class} (flag=0)"
            )
            continue

        concept_list = class_concepts_map.get(target_class, [])
        if not concept_list:
            logging.warning(
                f"No concepts found for target_class={target_class} — skipping."
            )
            continue

        concept_weights = concept_weights_map.get(target_class, {})
        class_weight = class_weights_map.get(target_class, 1.0)
        logging.info(
            f"Class={target_class} concepts={concept_list} "
            f"concept_weights={concept_weights} class_weight={class_weight}"
        )
        print(
            f"Class={target_class} concepts={concept_list} "
            f"concept_weights={concept_weights} class_weight={class_weight}"
        )

        class_results = []
        for combo_idx, combo_layers in enumerate(layer_combos):
            logging.info(
                f"Class={target_class} combo {combo_idx+1}/{len(layer_combos)}: "
                f"{combo_layers}"
            )
            print(
                f"\nClass={target_class} combo {combo_idx+1}/{len(layer_combos)}: "
                f"{combo_layers}"
            )

            combo_results, combo_layers_returned = _recalibrate_one_class(
                base_model=base_model,
                base_model_name=base_model_name,
                layer_combo=combo_layers,
                registry=registry,
                concept_list=concept_list,
                concept_weights=concept_weights,
                class_weight=class_weight,
                training_loader=training_loader,
                validation_loader=validation_loader,
                target_class=target_class,
                lambda_aligns=lambda_aligns,
                combo_idx=combo_idx,
                epochs=epochs,
                lr=lr,
                results_path=results_path,
                save_plots=save_plots,
                logging=logging,
            )

            for r in combo_results:
                r["Class Index"] = class_idx
                save_statistics(r, combo_csv)
                
                # Track best model for this class
                current_acc = r.get("Accuracy", 0.0)
                if target_class not in best_models_per_class or current_acc > best_models_per_class[target_class].get("accuracy", 0.0):
                    model_file = os.path.join(
                        results_path,
                        f"model_cls{target_class}_combo{combo_idx}_lambda{r['Lambda Alignment']}.pth"
                    )
                    if os.path.isfile(model_file):
                        best_models_per_class[target_class] = {
                            "path": model_file,
                            "layers": combo_layers_returned,
                            "accuracy": current_acc,
                            "lambda": r["Lambda Alignment"],
                            "combination": r.get("Combination", ""),
                        }

            class_results.extend(combo_results)
            all_results.extend(combo_results)

        if class_results:
            best = max(class_results, key=lambda r: r.get("Accuracy", 0.0))
            logging.info(
                f"Best for class={target_class}: layers={best['Layers']} "
                f"lambda={best['Lambda Alignment']} acc={best['Accuracy']:.6f}"
            )
            print(
                f"Best for class={target_class}: layers={best['Layers']} "
                f"lambda={best['Lambda Alignment']} acc={best['Accuracy']:.6f}"
            )

    if all_results:
        best_overall = max(all_results, key=lambda r: r.get("Accuracy", 0.0))
        msg = (
            f"\nBest overall: class={best_overall['Target Class']} "
            f"layers={best_overall['Layers']} "
            f"lambda={best_overall['Lambda Alignment']} "
            f"acc={best_overall['Accuracy']:.6f}"
        )
        logging.info(msg)
        print(msg)

    # Compute after-recalibration metrics using best models
    logging.info("Computing post-recalibration metrics...")
    print("\n" + "="*60)
    print("Computing post-recalibration metrics...")
    print("="*60)
    
    after_metrics, after_tcav = _compute_after_metrics(
        best_models_per_class, registry, base_model_name,
        class_concepts_map, validation_loader, target_idx_list,
        results_path, logging,
    )
    
    # Create summary Excel
    logging.info("Creating summary Excel file...")
    _create_summary_excel(
        before_metrics, before_tcav, after_metrics, after_tcav,
        target_idx_list, target_class_names, class_concepts_map,
        results_path, base_model_name, logging,
        concept_weights_map=concept_weights_map,
        class_weights_map=class_weights_map,
    )
    
    logging.info("Recalibration complete.")


# ---------------------------------------------------------------------------
# Joint multi-class recalibration entry point (used when
# --multiclass_recalibration_mode is set): trains ONE shared model per
# (layer_combo, lambda_align) using the weighted alignment loss of every
# enabled target class simultaneously, instead of one independent model per
# class as `main()` does.
# ---------------------------------------------------------------------------

def main_multiclass(
    base_model,
    base_model_name: str,
    registry: CAVRegistry,
    class_concepts_map: dict,
    concept_weights_map: dict,
    class_weights_map: dict,
    layer_names: list,
    training_loader,
    validation_loader,
    target_idx_list: list,
    target_class_names: list,
    lambda_aligns: list,
    recalibrate_flags: list,
    epochs: int,
    lr: float,
    results_path: str,
    use_groups: bool,
    save_plots: bool,
    logging,
):
    """
    class_concepts_map  : {target_idx: [concept_name, ...]}
    concept_weights_map : {target_idx: {concept_name: weight}}
    class_weights_map   : {target_idx: class_weight}

    Only classes with recalibrate_flags[i] == 1 are included in the joint
    training run. All of them share a single recalibrated model per
    (layer_combo, lambda_align); the best such joint model (by overall
    validation accuracy) is then reused as "the" after-recalibration model
    for every enabled class when computing after-metrics and the summary.
    """
    set_seed(RANDOM_STATE)
    logging.info("Multi-class joint recalibration started.")

    enabled_classes = [
        target_idx_list[i]
        for i in range(len(target_idx_list))
        if (recalibrate_flags[i] if i < len(recalibrate_flags) else 1) == 1
    ]
    if not enabled_classes:
        logging.error("No classes enabled for multi-class recalibration. Aborting.")
        print("No classes enabled for multi-class recalibration. Aborting.")
        return
    logging.info(f"Classes recalibrated jointly: {enabled_classes}")
    print(f"Classes recalibrated jointly: {enabled_classes}")

    before_metrics = {}
    before_tcav = {}

    acc, prec, rec, f1, conf, class_res = _compute_metrics(
        base_model, validation_loader, target_idx_list, logging
    )
    for target_idx in target_idx_list:
        before_metrics[target_idx] = {
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
        }

    for target_idx in enabled_classes:
        concept_list = class_concepts_map.get(target_idx, [])
        if not concept_list:
            continue
        before_tcav[target_idx] = {}
        for concept_name in concept_list:
            try:
                cavs = _load_target_cavs(
                    registry, base_model_name, layer_names, concept_name, logging
                )
                if cavs:
                    before_tcav[target_idx][concept_name] = _compute_tcav_scores(
                        base_model, cavs, validation_loader, target_idx, logging
                    )
                del cavs
            except Exception as exc:
                logging.warning(
                    f"TCAV before-metrics failed target={target_idx} "
                    f"concept={concept_name}: {exc}"
                )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    _save_before_metrics(
        base_model, layer_names, registry, base_model_name,
        class_concepts_map, validation_loader, target_idx_list,
        results_path, logging,
    )

    layer_combos = _resolve_layer_combos(
        layer_names, use_groups, registry, base_model_name
    )
    if not layer_combos:
        logging.error("No layer combinations available. Aborting.")
        print("No layer combinations available. Aborting.")
        return

    combo_index = {idx + 1: "|".join(c) for idx, c in enumerate(layer_combos)}
    with open(
        os.path.join(results_path, f"layer_combinations_{base_model_name}.txt"), "w"
    ) as fh:
        fh.write(str(combo_index))

    combo_csv = os.path.join(
        results_path, f"recalibration_multiclass_combo_accuracy_{base_model_name}.csv"
    )
    all_results = []
    best_joint = None  # {path, layers, accuracy, lambda, combination}

    for combo_idx, combo_layers in enumerate(layer_combos):
        logging.info(f"Joint combo {combo_idx+1}/{len(layer_combos)}: {combo_layers}")
        print(f"\nJoint combo {combo_idx+1}/{len(layer_combos)}: {combo_layers}")

        combo_results, combo_layers_returned = _recalibrate_multiclass(
            base_model=base_model,
            base_model_name=base_model_name,
            layer_combo=combo_layers,
            registry=registry,
            class_concepts_map=class_concepts_map,
            concept_weights_map=concept_weights_map,
            class_weights_map=class_weights_map,
            training_loader=training_loader,
            validation_loader=validation_loader,
            target_idx_list=enabled_classes,
            lambda_aligns=lambda_aligns,
            combo_idx=combo_idx,
            epochs=epochs,
            lr=lr,
            results_path=results_path,
            save_plots=save_plots,
            logging=logging,
        )

        for r in combo_results:
            save_statistics(r, combo_csv)

            current_acc = r.get("Accuracy", 0.0)
            if best_joint is None or current_acc > best_joint.get("accuracy", 0.0):
                model_file = os.path.join(
                    results_path,
                    f"model_multiclass_combo{combo_idx}_lambda{r['Lambda Alignment']}.pth"
                )
                if os.path.isfile(model_file):
                    best_joint = {
                        "path": model_file,
                        "layers": combo_layers_returned,
                        "accuracy": current_acc,
                        "lambda": r["Lambda Alignment"],
                        "combination": r.get("Combination", ""),
                    }

        all_results.extend(combo_results)

    if all_results:
        best_overall = max(all_results, key=lambda r: r.get("Accuracy", 0.0))
        msg = (
            f"\nBest overall joint model: "
            f"layers={best_overall['Layers']} "
            f"lambda={best_overall['Lambda Alignment']} "
            f"acc={best_overall['Accuracy']:.6f}"
        )
        logging.info(msg)
        print(msg)

    if best_joint is None:
        logging.error("No joint model was successfully trained. Aborting after-metrics.")
        print("No joint model was successfully trained. Aborting after-metrics.")
        return

    # Reuse the single best joint model as "the" after-recalibration model for
    # every enabled class, so the existing per-class after-metrics/summary
    # machinery can be reused unchanged.
    best_models_per_class = {
        target_idx: best_joint for target_idx in enabled_classes
    }

    logging.info("Computing post-recalibration metrics...")
    print("\n" + "="*60)
    print("Computing post-recalibration metrics (joint multi-class model)...")
    print("="*60)

    after_metrics, after_tcav = _compute_after_metrics(
        best_models_per_class, registry, base_model_name,
        class_concepts_map, validation_loader, target_idx_list,
        results_path, logging,
    )

    logging.info("Creating summary Excel file...")
    _create_summary_excel(
        before_metrics, before_tcav, after_metrics, after_tcav,
        target_idx_list, target_class_names, class_concepts_map,
        results_path, base_model_name, logging,
        concept_weights_map=concept_weights_map,
        class_weights_map=class_weights_map,
    )

    logging.info("Multi-class joint recalibration complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAV-guided model recalibration (loads CAVs from cav_store)."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name, e.g. vgg16")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Root directory containing model weight sub-folders")
    parser.add_argument("--cav_store", type=str, default="./cav_store",
                        help="CAV store root (contains manifest.json)")
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to YAML config, e.g. config/biased/vgg16.yaml")
    parser.add_argument("--store_results", type=str, default="./results",
                        help="Base directory for results and logs")
    parser.add_argument("--use_layer_groups", action="store_true",
                        help="Use layer groups from manifest instead of combinations")
    parser.add_argument(
        "--recalibrate_classes", type=str, default=None,
        help="Comma-separated class indices to recalibrate, e.g. '0,2'. "
             "Defaults to all target classes.",
    )
    parser.add_argument(
        "--manifest_file", type=str, default=None,
        help=(
            "Path to the per-model manifest JSON (e.g. 'vgg16_manifest.json') "
            "produced by main_store_cav.py. Its 'concepts' section is grouped "
            "by 'class_name' to build the multi-concept-per-class map used for "
            "targeted recalibration. Defaults to "
            "'<cav_store_root>/<model_name>_manifest.json', where "
            "<cav_store_root> is --cav_store itself, or its parent directory "
            "if --cav_store points directly at the model's own CAV folder "
            "(e.g. --cav_store './vgg16' resolves to './vgg16_manifest.json')."
        ),
    )
    parser.add_argument(
        "--multiclass_recalibration_mode", action="store_true",
        help=(
            "If set, jointly recalibrate ALL enabled classes at the same time "
            "using a single shared model per (layer_combo, lambda_align) — the "
            "weighted alignment loss of every enabled class's concepts is "
            "summed (each masked to its own samples) before combining with "
            "the classification loss. If NOT set (default), classes are "
            "recalibrated one at a time, each producing its own independent "
            "model (the original behaviour)."
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    config = ConfigSingleton(args.config_file)
    set_seed(config.SEED)

    BASE_MODEL = args.model_name.strip().lower()
    MODEL_PATH = get_model_weight_path(BASE_MODEL, args.model_path.strip())

    RESULTS_PATH = os.path.join(args.store_results, BASE_MODEL)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    log_filename = os.path.join(
        RESULTS_PATH,
        f"audit_trail_{BASE_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    LOGGING = Logger_Singleton(log_filename)
    LOGGING.info("Script started.")

    # ------------------------------------------------------------------
    # Resolve the true CAV store root. --cav_store may point either at the
    # common store root (containing "manifest.json" and a "<model>/"
    # sub-folder), or directly at the model-specific CAV folder itself
    # (e.g. ".../vgg16"). In the latter case, both "manifest.json" and the
    # per-model "<model>_manifest.json" live one directory up, so step up
    # automatically when the last path segment matches the model name.
    # ------------------------------------------------------------------
    cav_store_input = os.path.abspath(args.cav_store)
    if os.path.basename(cav_store_input.rstrip(os.sep)) == BASE_MODEL:
        CAV_STORE_ROOT = os.path.dirname(cav_store_input.rstrip(os.sep))
        LOGGING.info(
            f"--cav_store points at the model folder '{cav_store_input}'; "
            f"using its parent as the CAV store root: '{CAV_STORE_ROOT}'."
        )
    else:
        CAV_STORE_ROOT = cav_store_input

    registry = CAVRegistry(CAV_STORE_ROOT)

    IMAGE_SIZE = get_base_model_image_size(BASE_MODEL)
    LOGGING.info(f"Loading {BASE_MODEL} from {MODEL_PATH}")
    print(f"Loading {BASE_MODEL} from {MODEL_PATH} on {DEVICE}")

    if DEVICE == "cpu":
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    else:
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
    MODEL.to(DEVICE)
    MODEL.eval()

    # Layer names — config override or all layers
    all_layers = get_model_layers(MODEL)
    if config.OVERRIDE_RECALIB:
        layer_map = {
            "vgg16":              config.VGG_RECALIB,
            "resnet50":           config.RESNET50_RECALIB,
            "inception_v3":       config.INCEPTION_V3_RECALIB,
            "mobilenet_v3_small": config.MOBILENET_V3_SMALL_RECALIB,
            "mobilenet_v3_large": config.MOBILENET_V3_LARGE_RECALIB,
        }
        LAYER_NAMES = layer_map.get(BASE_MODEL, all_layers[2:])
    else:
        LAYER_NAMES = all_layers[2:]
    LOGGING.info(f"Recalibration layers: {LAYER_NAMES}")
    print(f"Recalibration layers: {LAYER_NAMES}")

    TRANSFORM = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    try:
        train_folders, valid_folders, class_names = get_class_folder_dicts(
            config.CLASSIFICATION_DATA_BASE_PATH
        )
        TARGET_IDX_LIST = [
            class_names.index(cls) for cls in config.TARGET_CLASS_LIST
        ]

        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE)

        TRAINING_LOADER = DataLoader(
            MultiClassImageDataset(train_folders, transform=TRANSFORM),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        VALIDATION_LOADER = DataLoader(
            MultiClassImageDataset(valid_folders, transform=TRANSFORM),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
        )
        LOGGING.info("Data loaded successfully.")
    except Exception as exc:
        LOGGING.error(f"Data preparation failed: {exc}")
        raise

    # ------------------------------------------------------------------
    # Class -> concepts mapping, built from the per-model manifest file
    # (e.g. "vgg16_manifest.json") rather than a single concept per class.
    # ------------------------------------------------------------------
    manifest_path = args.manifest_file or os.path.join(
        CAV_STORE_ROOT, f"{BASE_MODEL}_manifest.json"
    )
    LOGGING.info(f"Loading per-model manifest: {manifest_path}")
    print(f"Loading per-model manifest: {manifest_path}")
    manifest_data = load_model_manifest(manifest_path)

    CLASS_CONCEPTS_MAP = build_class_concepts_map(
        manifest_data, class_names, TARGET_IDX_LIST
    )

    # Fallback to the legacy single-concept-per-class mapping (config's
    # CONCEPT_FOLDER_LIST, positional against TARGET_CLASS_LIST) for any
    # class the manifest has no concepts for.
    LEGACY_CONCEPT_NAMES = [
        os.path.basename(p.rstrip("/\\"))
        for p in config.CONCEPT_FOLDER_LIST
    ]
    for i, target_idx in enumerate(TARGET_IDX_LIST):
        if not CLASS_CONCEPTS_MAP.get(target_idx) and i < len(LEGACY_CONCEPT_NAMES):
            LOGGING.warning(
                f"No manifest concepts found for class={target_idx}; "
                f"falling back to legacy concept '{LEGACY_CONCEPT_NAMES[i]}'."
            )
            CLASS_CONCEPTS_MAP[target_idx] = [LEGACY_CONCEPT_NAMES[i]]

    LOGGING.info(f"Class-concepts map: {CLASS_CONCEPTS_MAP}")
    print(f"Class-concepts map: {CLASS_CONCEPTS_MAP}")

    # ------------------------------------------------------------------
    # Configurable per-class and per-concept alignment weights (config file
    # driven — see ConfigSingleton._read_recalibration_weights_variables).
    # ------------------------------------------------------------------
    CLASS_WEIGHTS_MAP = {
        target_idx: config.get_class_weight(idx)
        for idx, target_idx in enumerate(TARGET_IDX_LIST)
    }
    CONCEPT_WEIGHTS_MAP = {
        target_idx: config.get_concept_weights(idx, CLASS_CONCEPTS_MAP.get(target_idx, []))
        for idx, target_idx in enumerate(TARGET_IDX_LIST)
    }
    LOGGING.info(f"Class weights: {CLASS_WEIGHTS_MAP}")
    LOGGING.info(f"Concept weights: {CONCEPT_WEIGHTS_MAP}")
    print(f"Class weights: {CLASS_WEIGHTS_MAP}")
    print(f"Concept weights: {CONCEPT_WEIGHTS_MAP}")

    # Recalibrate flags
    if args.recalibrate_classes is not None:
        enabled = set(int(x) for x in args.recalibrate_classes.split(","))
        RECALIBRATE_FLAGS = [
            1 if i in enabled else 0
            for i in range(len(TARGET_IDX_LIST))
        ]
    else:
        RECALIBRATE_FLAGS = [1] * len(TARGET_IDX_LIST)

    LOGGING.info(f"Recalibrate flags: {RECALIBRATE_FLAGS}")

    # Normalise LAMBDA_ALIGNS to a list (config may return dict or list)
    _raw_lambdas = config.LAMBDA_ALIGNS
    LAMBDA_ALIGNS_LIST = (
        list(_raw_lambdas.values())
        if isinstance(_raw_lambdas, dict)
        else list(_raw_lambdas)
    )

    LOGGING.info(f"multiclass_recalibration_mode={args.multiclass_recalibration_mode}")
    print(f"multiclass_recalibration_mode={args.multiclass_recalibration_mode}")

    recalib_fn = main_multiclass if args.multiclass_recalibration_mode else main
    recalib_fn(
        base_model=MODEL,
        base_model_name=BASE_MODEL,
        registry=registry,
        class_concepts_map=CLASS_CONCEPTS_MAP,
        concept_weights_map=CONCEPT_WEIGHTS_MAP,
        class_weights_map=CLASS_WEIGHTS_MAP,
        layer_names=LAYER_NAMES,
        training_loader=TRAINING_LOADER,
        validation_loader=VALIDATION_LOADER,
        target_idx_list=TARGET_IDX_LIST,
        target_class_names=class_names,
        lambda_aligns=LAMBDA_ALIGNS_LIST,
        recalibrate_flags=RECALIBRATE_FLAGS,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        results_path=RESULTS_PATH,
        use_groups=args.use_layer_groups,
        save_plots=getattr(config, "SAVE_PLOTS", False),
        logging=LOGGING,
    )

    LOGGING.info("Script execution finished.")

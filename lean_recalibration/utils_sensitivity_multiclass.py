import os
import os.path
import sys
import ast
import pickle
import json
import math
import argparse
import random
from datetime import datetime
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from openpyxl.styles import Font, PatternFill, Alignment
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from logger import Logger_Singleton
from custom_dataloader import SingleClassDataLoader

from ConfigSingleton import ConfigSingleton
from utils import (
    get_model_layers, load_model, 
    load_train_valid_dataset,
)

from tcav_utils import (util_compute_sensitivity_score, util_compute_tcav_score_from_sensitivity)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


RANDOM_STATE = 132

def load_model_statedict(model, model_path, trusted_checkpoint=False):
    """Load a state dict, or extract one from a trusted full-model checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=not trusted_checkpoint)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "This checkpoint contains pickled objects (e.g. a torchvision VGG), "
            "not just weights. If you created and trust this file, call "
            "load_model_statedict(model, path, trusted_checkpoint=True). "
            "Never enable this for an untrusted checkpoint."
        ) from exc

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    model.load_state_dict(state_dict)
    return model


class _ActivationHookStub:
    """Placeholder used only to unpickle checkpoints saved with ActivationHook forward hooks attached."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return None


def load_model_with_trained_weights(base_model_path, weights_path, device='cpu'):
    """Load the base model architecture and update it with weights from a trained checkpoint.

    Only use with trusted checkpoints: both files are full pickled models.
    """
    # Recalibrated models were saved from a script run as __main__ with ActivationHook registered
    main_module = sys.modules['__main__']
    if not hasattr(main_module, 'ActivationHook'):
        main_module.ActivationHook = _ActivationHookStub

    model = torch.load(base_model_path, map_location='cpu', weights_only=False)
    model = load_model_statedict(model, weights_path, trusted_checkpoint=True)
    model.to(device)
    model.eval()
    print(f"Loaded base model from {base_model_path} with weights from {weights_path}")
    return model


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Set global seed
set_seed(RANDOM_STATE)

# Columns exported to every per-concept sheet (and used to build the raw CSV backup).
# This is deliberately wide so that the Excel workbook alone is enough to answer
# follow-up questions without re-running the (expensive) sensitivity computation.
EXPORT_COLUMNS = [
    'model_name', 'concept_name', 'class_name', 'target_class_index', 'stage', 'combo_name',
    'recalibrated_layers', 'layer_name',
    'lambda_align', 'sensitivity_score', 'positive_sensitivity', 'file_path',
    'concept_folder_path', 'random_folder_path', 'model_weight_path', 'cav_source_path',
    'dataset_split', 'concept_mode', 'cav_random_state', 'batch_size', 'linear_classifier_type',
    'manifest_file', 'config_file', 'run_timestamp',
]

# Stay comfortably under Excel's hard limit of 1,048,576 rows per sheet.
EXCEL_MAX_ROWS_PER_SHEET = 1_048_500

# Characters that are not allowed inside an Excel worksheet name.
_INVALID_SHEET_CHARS = set('[]:*?/\\')


def get_model_path(MODEL_NAME, layer_name, lambda_val, recalibrated_model_base_path='./results'):
    model_filename = f"loss_{MODEL_NAME.strip()}_{layer_name.strip()}_{lambda_val}.pth"
    base_path = os.path.join(recalibrated_model_base_path, MODEL_NAME.strip(), model_filename)
    print("Looking for the recalibrated model at ", base_path)
    if not os.path.isfile(base_path):
        print("File not found in the given path ")
        raise FileNotFoundError(f"The file '{base_path}' was not found.")
    print("Base path where the model is located ", base_path)
    return base_path


def get_model_path_multiclass(MODEL_NAME, layer_name, lambda_val, recalibrated_model_base_path='./results'):
    model_filename = f"model_multiclass_{layer_name.strip()}_lambda{lambda_val}.pth"
    base_path = os.path.join(recalibrated_model_base_path, MODEL_NAME.strip(), model_filename)
    print("Looking for the recalibrated model at ", base_path)
    if not os.path.isfile(base_path):
        print("File not found in the given path ")
        raise FileNotFoundError(f"The file '{base_path}' was not found.")
    print("Base path where the model is located ", base_path)
    return base_path

MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
OVERRIDE_RECALIB = None


def load_layer_combinations(layer_combinations_file):
    """
    Read layer_combinations_<model>.txt written by main_recalib_custom_by_loading_cav.py,
    e.g. {1: 'L1|L2', 2: 'L3'}, and return [(combo_name, [layers])] in combo order.
    Keys are 1-based while saved model files are 0-based (key 1 -> combo0).
    """
    if not os.path.isfile(layer_combinations_file):
        raise FileNotFoundError(f"Layer combinations file '{layer_combinations_file}' was not found.")
    with open(layer_combinations_file, 'r') as file:
        combo_index = ast.literal_eval(file.read())
    scenarios = []
    for key in sorted(combo_index, key=int):
        combo_layers = [ln.strip() for ln in str(combo_index[key]).split('|') if ln.strip()]
        scenarios.append((f"combo{int(key) - 1}", combo_layers))
    return scenarios


activation = {}
output_shape = {}

def get_activation(layer_name):
    def hook(model, input, output):
        activation[layer_name] = output
        output_shape[layer_name] = output.shape
        # This print has been added for you to visualize if the size is too large then the time taken fror convergence will be large
        print(f"Verify the output shape : Layername = {layer_name} , output.shape : {output.shape}")
    return hook


def get_layernames_override(MODEL_NAME, config):
    if(MODEL_NAME == 'vgg16'):
        return (config.VGG_RECALIB)
    if(MODEL_NAME == 'resnet50'):
        return (config.RESNET50_RECALIB)
    if(MODEL_NAME == 'inception_v3'):
        return (config.INCEPTION_V3_RECALIB)
    if(MODEL_NAME == 'mobilenet_v3_small'):
        return (config.MOBILENET_V3_SMALL_RECALIB)
    if(MODEL_NAME == 'mobilenet_v3_large'):
        return (config.MOBILENET_V3_LARGE_RECALIB)


def _safe_name(name):
    """Sanitize a class/folder name for use inside a concept name or Excel sheet name."""
    return str(name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_singleclass_filelist(dataloader):
    """
    Return the absolute file paths for every image a SingleClassDataLoader-backed
    DataLoader will yield, in exactly the order it will yield them.

    This relies on the loader being built with shuffle=False (the default for
    plain DataLoader, and how `class_dataloaders` is constructed below), so the
    iteration order matches `dataset.image_files` order exactly - which is what
    lets us line up each row of a sensitivity-score tensor with the file that
    produced it, without touching custom_dataloader.py.
    """
    dataset = dataloader.dataset
    return [os.path.abspath(os.path.join(dataset.folder_path, fname)) for fname in dataset.image_files]


def build_concept_specs(config, concept_mode, target_class_list, target_idx_list, class_dataloaders,
                         concept_folder_list, random_folder):
    """
    Resolve a flat list of "concept spec" dicts regardless of concept_mode, so the
    rest of the pipeline (CAV computation, sensitivity scoring, row recording) can
    stay mode-agnostic. Each spec has:
        concept_name       - unique, Excel/CSV-safe name (e.g. "deer_coat")
        class_name          - the owning target class name (e.g. "deer")
        target_class_index  - index into the model's output logits for that class
        concept_folder       - absolute path to the concept's positive-example images
        random_folder        - absolute path to that concept's random/negative images
        class_dataloader     - the (shared, per-class) DataLoader used to compute
                                sensitivity scores; concepts of the same class share
                                the exact same DataLoader instance.

    concept_mode:
        'single'      -> one concept per class, from config.CONCEPT_FOLDER_LIST /
                         config.RANDOM_FOLDER (the original behaviour).
        'multiclass'  -> multiple concepts per class, from
                         config.MULTICONCEPT_CLASS_CONCEPTS, each with its own
                         random folder resolved via config.get_multiconcept_random_folder_path.
    """
    concept_specs = []

    if concept_mode == 'multiclass':
        if not getattr(config, 'MULTICONCEPT_ENABLED', False):
            raise ValueError(
                "concept_mode='multiclass' was requested but the YAML config has no "
                "(or an empty) 'multiconcept' section."
            )
        for class_idx, target_folders in config.MULTICONCEPT_CLASS_CONCEPTS.items():
            if class_idx >= len(target_class_list):
                print(f"[WARN] multiconcept class index {class_idx} has no matching "
                      f"entry in target_class_list - skipping its concepts.")
                continue
            class_name = target_class_list[class_idx]
            target_idx = target_idx_list[class_idx]
            class_loader = class_dataloaders[class_idx]
            for position_idx, target_folder in enumerate(target_folders):
                concept_folder = (
                    target_folder if os.path.isabs(target_folder)
                    else os.path.join(config.MULTICONCEPT_BASE_PATH, target_folder)
                )
                resolved_random_folder = config.get_multiconcept_random_folder_path(class_idx, position_idx)
                base_concept_name = _safe_name(os.path.basename(target_folder.rstrip('/\\')))
                concept_specs.append({
                    'concept_name': f"{_safe_name(class_name)}_{base_concept_name}",
                    'class_name': class_name,
                    'target_class_index': target_idx,
                    'concept_folder': os.path.abspath(concept_folder),
                    'random_folder': os.path.abspath(resolved_random_folder),
                    'class_dataloader': class_loader,
                })
    elif concept_mode == 'single':
        if len(concept_folder_list) != len(target_class_list):
            print(f"[WARN] concept.target_folders has {len(concept_folder_list)} entries "
                  f"but target_class_list has {len(target_class_list)}; zipping to the "
                  f"shorter of the two - please double check the YAML config.")
        for class_name, target_idx, concept_folder, class_loader in zip(
                target_class_list, target_idx_list, concept_folder_list, class_dataloaders):
            base_concept_name = _safe_name(os.path.basename(concept_folder.rstrip('/\\')))
            concept_specs.append({
                'concept_name': f"{_safe_name(class_name)}_{base_concept_name}",
                'class_name': class_name,
                'target_class_index': target_idx,
                'concept_folder': os.path.abspath(concept_folder),
                'random_folder': os.path.abspath(random_folder),
                'class_dataloader': class_loader,
            })
    else:
        raise ValueError(f"Unknown concept_mode '{concept_mode}'; expected 'single' or 'multiclass'.")

    # De-duplicate concept names (e.g. two classes that happen to share a basename)
    seen_counts = {}
    for spec in concept_specs:
        base_name = spec['concept_name']
        seen_counts[base_name] = seen_counts.get(base_name, 0) + 1
        if seen_counts[base_name] > 1:
            spec['concept_name'] = f"{base_name}_{seen_counts[base_name]}"

    if not concept_specs:
        raise ValueError("No concepts were resolved - check the 'concept'/'multiconcept' "
                          "sections of the YAML config and the --concept_mode argument.")
    return concept_specs


def _safe_layer_name(layer_name):
    """Sanitize a layer name for use as a file name (mirrors cav_registry.CAVRegistry._safe_name,
    so joblib file names line up exactly with what main_store_cav.py / CAVRegistry wrote)."""
    return str(layer_name).replace(os.sep, "_").replace("/", "_")


def load_manifest(manifest_path):
    """
    Load a per-model CAV manifest (e.g. vgg16_manifest.json, produced by
    main_store_cav.py / cav_registry.CAVRegistry). See vgg16_manifest.json in
    this folder for an example of the expected schema:
        {
          "model_name": ..., "model_weight_path": ...,
          "concepts": {
            "<concept_name>": {
              "data_path": "<folder containing '<layer_name>.joblib' files>",
              "layers": ["<layer_name>", ...],
              ...
            }, ...
          }
        }
    """
    with open(manifest_path, "r") as fh:
        return json.load(fh)


def get_cav_from_manifest(manifest_data, concept_name, layer_name, device):
    """
    Look up a precomputed CAV vector for `concept_name`/`layer_name` from an
    already-parsed manifest (see `load_manifest`). The CAV itself is read from
    the joblib file at <concept's data_path>/<layer_name>.joblib, which is
    exactly where main_store_cav.py / CAVRegistry.save_concept_layer_cav wrote it.

    Returns a 3-tuple (cav_tensor, cav_source_path, reason):
        - On success: (torch.Tensor moved to `device`, absolute joblib path, None)
        - On failure: (None, None, a human-readable reason string) - the caller
          is expected to warn and skip rather than raise, so one missing CAV
          never aborts the whole run.
    """
    concepts = (manifest_data or {}).get("concepts", {})
    concept_entry = concepts.get(concept_name)
    if concept_entry is None:
        return None, None, f"concept '{concept_name}' not found in manifest"
    if layer_name not in (concept_entry.get("layers") or []):
        return None, None, f"layer '{layer_name}' not listed for concept '{concept_name}' in manifest"
    data_path = concept_entry.get("data_path")
    if not data_path:
        return None, None, f"concept '{concept_name}' has no 'data_path' recorded in manifest"
    cav_file = os.path.join(data_path, f"{_safe_layer_name(layer_name)}.joblib")
    if not os.path.isfile(cav_file):
        return None, None, f"cav file not found on disk: {cav_file}"
    try:
        payload = joblib.load(cav_file)
        cav_vector = payload["cav_vector"]
    except Exception as e:
        return None, None, f"failed to load cav file '{cav_file}': {e}"
    cav_tensor = cav_vector if isinstance(cav_vector, torch.Tensor) else torch.tensor(cav_vector)
    cav_tensor = cav_tensor.to(device=device, dtype=torch.float32)
    return cav_tensor, os.path.abspath(cav_file), None


def _sanitize_sheet_name(name, used_names):
    """
    Make `name` a valid, unique Excel worksheet name (<=31 chars, none of
    [ ] : * ? / \\), tracking already-used names case-insensitively in
    `used_names` (mutated in place).
    """
    cleaned = ''.join(ch for ch in str(name) if ch not in _INVALID_SHEET_CHARS).strip()
    if not cleaned:
        cleaned = "Sheet"
    cleaned = cleaned[:31]
    base = cleaned
    suffix = 1
    while cleaned.lower() in used_names:
        suffix += 1
        tail = f"_{suffix}"
        cleaned = f"{base[:31 - len(tail)]}{tail}"
    used_names.add(cleaned.lower())
    return cleaned


def _style_worksheet(worksheet, max_col_width=50):
    """Apply the header styling convention already used elsewhere in this repo
    (see main_recalib_custom_by_loading_cav.py), plus a frozen header row and
    autofilter so the workbook is pleasant to explore by hand."""
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
    worksheet.freeze_panes = "A2"
    try:
        worksheet.auto_filter.ref = worksheet.dimensions
    except Exception:
        pass
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            try:
                if cell.value is not None and len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except Exception:
                pass
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, max_col_width)


def compute_concept_summary(records_df):
    """
    Build the 'Summary' sheet: one row per concept with the literal TCAV score
    the user asked for (positive-sensitivity rows / total rows, across every
    layer/stage/lambda gathered so far for that concept) plus a handful of
    useful breakdowns (before-only, after-only, per-lambda) and descriptive
    stats so the sheet is useful on its own without opening the raw data.
    """
    rows = []
    for concept_name, group in records_df.groupby('concept_name', sort=False):
        total_rows = len(group)
        positive_rows = int((group['sensitivity_score'] > 0).sum())
        tcav_score = positive_rows / total_rows if total_rows else float('nan')

        before_group = group[group['stage'] == 'before']
        before_total = len(before_group)
        before_positive = int((before_group['sensitivity_score'] > 0).sum())
        tcav_before = before_positive / before_total if before_total else float('nan')

        after_group = group[group['stage'] == 'after']
        after_total = len(after_group)
        after_positive = int((after_group['sensitivity_score'] > 0).sum())
        tcav_after_overall = after_positive / after_total if after_total else float('nan')

        first = group.iloc[0]
        row = {
            'concept_name': concept_name,
            'model_name': first['model_name'],
            'class_name': first['class_name'],
            'target_class_index': first['target_class_index'],
            'concept_mode': first['concept_mode'],
            'concept_folder_path': first['concept_folder_path'],
            'random_folder_path': first['random_folder_path'],
            'cav_source_paths': ", ".join(sorted(group['cav_source_path'].dropna().unique())),
            'total_rows': total_rows,
            'positive_rows': positive_rows,
            'non_positive_rows': total_rows - positive_rows,
            'tcav_score': tcav_score,
            'tcav_score_before': tcav_before,
            'tcav_score_after_overall': tcav_after_overall,
            'mean_sensitivity_overall': group['sensitivity_score'].mean(),
            'std_sensitivity_overall': group['sensitivity_score'].std(),
            'min_sensitivity_overall': group['sensitivity_score'].min(),
            'max_sensitivity_overall': group['sensitivity_score'].max(),
            'num_layers_processed': group['layer_name'].nunique(),
            'num_combos_processed': int(after_group['combo_name'].nunique()) if after_total else 0,
            'num_lambdas_processed': int(after_group['lambda_align'].nunique()) if after_total else 0,
        }
        for (combo_name, lambda_val), lam_group in after_group.groupby(['combo_name', 'lambda_align'], sort=True):
            lam_total = len(lam_group)
            lam_positive = int((lam_group['sensitivity_score'] > 0).sum())
            row[f'tcav_score_after_{combo_name}_lambda_{lambda_val}'] = lam_positive / lam_total if lam_total else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def compute_layer_summary(records_df):
    """
    Build the 'Summary_By_Layer' sheet: one row per (concept, layer, stage,
    lambda) combination with its own TCAV score/counts/stats - a granular,
    reproduces-the-per-layer-TCAV-you'd-expect breakdown, computed entirely
    from the stored rows (no recomputation against the model needed).
    """
    group_cols = ['concept_name', 'model_name', 'class_name', 'layer_name', 'stage', 'combo_name',
                  'recalibrated_layers', 'lambda_align']
    rows = []
    for keys, group in records_df.groupby(group_cols, sort=False, dropna=False):
        concept_name, model_name, class_name, layer_name, stage, combo_name, recalibrated_layers, lambda_align = keys
        total_rows = len(group)
        positive_rows = int((group['sensitivity_score'] > 0).sum())
        tcav_score = positive_rows / total_rows if total_rows else float('nan')
        rows.append({
            'concept_name': concept_name,
            'model_name': model_name,
            'class_name': class_name,
            'layer_name': layer_name,
            'stage': stage,
            'combo_name': combo_name,
            'recalibrated_layers': recalibrated_layers,
            'lambda_align': lambda_align,
            'model_weight_path': group['model_weight_path'].iloc[0],
            'cav_source_path': group['cav_source_path'].iloc[0],
            'total_rows': total_rows,
            'positive_rows': positive_rows,
            'tcav_score': tcav_score,
            'mean_sensitivity': group['sensitivity_score'].mean(),
            'std_sensitivity': group['sensitivity_score'].std(),
            'min_sensitivity': group['sensitivity_score'].min(),
            'max_sensitivity': group['sensitivity_score'].max(),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(['concept_name', 'layer_name', 'stage', 'combo_name', 'lambda_align'],
                                    ascending=[True, True, False, True, True]).reset_index(drop=True)
    return result


def compute_classification_metrics(model, loader, class_names, device, stage, lambda_align, model_weight_path,
                                   combo_name='base', recalibrated_layers=''):
    """
    Evaluate `model` on `loader` and return (overall_row, per_class_rows).

    overall_row    : accuracy + macro precision/recall/F1 (same averaging as utils.evaluate_accuracy)
    per_class_rows : one row per class with support, correct count, class accuracy, precision, recall, F1
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            outputs = model(imgs.to(device))
            # Inception-style models may return (logits, aux_logits)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    base = {'stage': stage, 'combo_name': combo_name, 'recalibrated_layers': recalibrated_layers,
            'lambda_align': lambda_align, 'model_weight_path': model_weight_path}
    if not all_labels:
        return {**base, 'num_samples': 0, 'accuracy': float('nan'), 'precision_macro': float('nan'),
                'recall_macro': float('nan'), 'f1_macro': float('nan')}, []

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    class_indices = list(range(len(class_names)))
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=class_indices, average='macro', zero_division=0)
    overall_row = {
        **base,
        'num_samples': int(len(all_labels)),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision_macro': float(precision_m),
        'recall_macro': float(recall_m),
        'f1_macro': float(f1_m),
    }

    precision_c, recall_c, f1_c, support_c = precision_recall_fscore_support(
        all_labels, all_preds, labels=class_indices, average=None, zero_division=0)
    per_class_rows = []
    for idx, class_name in enumerate(class_names):
        mask = all_labels == idx
        support = int(support_c[idx])
        correct = int((all_preds[mask] == idx).sum())
        per_class_rows.append({
            **base,
            'class_name': class_name,
            'class_index': idx,
            'support': support,
            'correct': correct,
            'class_accuracy': correct / support if support else float('nan'),
            'precision': float(precision_c[idx]),
            'recall': float(recall_c[idx]),
            'f1_score': float(f1_c[idx]),
        })
    return overall_row, per_class_rows


def _paired_ttest(scores_a, scores_b, alpha):
    """Pair two {file_path, sensitivity_score} frames on file_path and run ttest_rel(b, a)."""
    paired = scores_a.merge(scores_b, on='file_path', suffixes=('_a', '_b'))
    n_pairs = len(paired)
    vals_a = paired['sensitivity_score_a'].to_numpy(dtype=float)
    vals_b = paired['sensitivity_score_b'].to_numpy(dtype=float)
    diff = vals_b - vals_a
    t_stat = p_value = float('nan')
    # ttest_rel is undefined for <2 pairs or when every difference is identical
    if n_pairs >= 2 and np.std(diff) > 0:
        result = ttest_rel(vals_b, vals_a)
        t_stat, p_value = float(result.statistic), float(result.pvalue)
    significant = bool(p_value < alpha) if not np.isnan(p_value) else False
    if not significant:
        direction = 'no significant change'
    else:
        direction = 'increased' if diff.mean() > 0 else 'decreased'
    return {
        'n_pairs': n_pairs,
        'mean_a': float(vals_a.mean()) if n_pairs else float('nan'),
        'mean_b': float(vals_b.mean()) if n_pairs else float('nan'),
        'mean_diff_b_minus_a': float(diff.mean()) if n_pairs else float('nan'),
        'std_diff': float(diff.std(ddof=1)) if n_pairs >= 2 else float('nan'),
        't_statistic': t_stat,
        'p_value': p_value,
        f'significant_p<{alpha}': significant,
        'direction': direction,
    }


def compute_paired_ttest(records_df, alpha=0.05):
    """
    Paired t-tests (scipy.stats.ttest_rel) of sensitivity scores for every
    recalibration scenario, one row per (class, concept, combo, layer, comparison):

        before vs lambda=X      - base model vs each recalibrated model of the combo
        lambda=X vs lambda=Y    - every pair of lambdas within the same combo

    Scores are paired on file_path, so each image is compared against itself.
    Group A is the reference and group B the comparison; the difference and
    direction are B relative to A (e.g. 'increased' = B has higher sensitivity).
    """
    columns = [
        'model_name', 'class_name', 'target_class_index', 'concept_name', 'combo_name', 'recalibrated_layers',
        'layer_name', 'comparison', 'group_a', 'group_b', 'lambda_a', 'lambda_b',
        'model_weight_path_a', 'model_weight_path_b', 'n_pairs', 'mean_a', 'mean_b', 'mean_diff_b_minus_a',
        'std_diff', 't_statistic', 'p_value', f'significant_p<{alpha}', 'direction',
    ]
    rows = []
    group_cols = ['model_name', 'class_name', 'target_class_index', 'concept_name', 'layer_name']
    for keys, group in records_df.groupby(group_cols, sort=False, dropna=False):
        model_name, class_name, target_class_index, concept_name, layer_name = keys
        before_group = group[group['stage'] == 'before']
        before = before_group.drop_duplicates('file_path')[['file_path', 'sensitivity_score']]
        before_path = before_group['model_weight_path'].iloc[0] if not before_group.empty else ''
        after_all = group[group['stage'] == 'after']
        for (combo_name, recalibrated_layers), combo_group in after_all.groupby(
                ['combo_name', 'recalibrated_layers'], sort=True):
            # {lambda: (scores, model_weight_path)} for this combo at this layer
            by_lambda = {
                lam: (lam_group.drop_duplicates('file_path')[['file_path', 'sensitivity_score']],
                      lam_group['model_weight_path'].iloc[0])
                for lam, lam_group in combo_group.groupby('lambda_align', sort=True)
            }
            lambdas = sorted(by_lambda)
            comparisons = []
            if not before.empty:
                comparisons += [('before', np.nan, before, before_path, f'lambda={lam}', lam, *by_lambda[lam])
                                for lam in lambdas]
            comparisons += [(f'lambda={lam_a}', lam_a, *by_lambda[lam_a], f'lambda={lam_b}', lam_b, *by_lambda[lam_b])
                            for i, lam_a in enumerate(lambdas) for lam_b in lambdas[i + 1:]]
            for order, (name_a, lam_a, scores_a, path_a, name_b, lam_b, scores_b, path_b) in enumerate(comparisons):
                rows.append({
                    'model_name': model_name,
                    'class_name': class_name,
                    'target_class_index': target_class_index,
                    'concept_name': concept_name,
                    'combo_name': combo_name,
                    'recalibrated_layers': recalibrated_layers,
                    'layer_name': layer_name,
                    'comparison': f'{name_a} vs {name_b}',
                    'group_a': name_a,
                    'group_b': name_b,
                    'lambda_a': lam_a,
                    'lambda_b': lam_b,
                    'model_weight_path_a': path_a,
                    'model_weight_path_b': path_b,
                    '_order': order,
                    **_paired_ttest(scores_a, scores_b, alpha),
                })
    result = pd.DataFrame(rows, columns=columns + ['_order'])
    if not result.empty:
        result = result.sort_values(
            ['class_name', 'concept_name', 'combo_name', 'layer_name', '_order']).reset_index(drop=True)
    return result.drop(columns='_order')


ACCURACY_METRIC_COLS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
CLASS_METRIC_COLS = ['class_accuracy', 'precision', 'recall', 'f1_score']


def _sort_scenarios(df, leading_cols=()):
    """Order rows as: before first, then each combo/lambda scenario."""
    df = df.copy()
    df['_stage_order'] = (df['stage'] != 'before').astype(int)
    df = df.sort_values(list(leading_cols) + ['_stage_order', 'combo_name', 'lambda_align'], na_position='first')
    return df.drop(columns='_stage_order').reset_index(drop=True)


def build_accuracy_overall(accuracy_df):
    """One row per model (before + every combo/lambda) with metrics and the change versus before."""
    if accuracy_df is None or accuracy_df.empty:
        return pd.DataFrame(columns=['stage', 'combo_name', 'recalibrated_layers', 'lambda_align', 'num_samples']
                            + ACCURACY_METRIC_COLS + [f'{c}_change_vs_before' for c in ACCURACY_METRIC_COLS]
                            + ['model_weight_path'])
    df = _sort_scenarios(accuracy_df)
    before = df[df['stage'] == 'before']
    for col in ACCURACY_METRIC_COLS:
        df[f'{col}_change_vs_before'] = df[col] - before.iloc[0][col] if not before.empty else float('nan')
    ordered = (['stage', 'combo_name', 'recalibrated_layers', 'lambda_align', 'num_samples'] + ACCURACY_METRIC_COLS
               + [f'{c}_change_vs_before' for c in ACCURACY_METRIC_COLS] + ['model_weight_path'])
    return df[ordered]


def build_accuracy_by_class(class_accuracy_df):
    """One row per (class, model) with per-class metrics and the change versus the base model."""
    ordered = (['class_name', 'class_index', 'stage', 'combo_name', 'recalibrated_layers', 'lambda_align',
                'support', 'correct'] + CLASS_METRIC_COLS + [f'{c}_change_vs_before' for c in CLASS_METRIC_COLS]
               + ['model_weight_path'])
    if class_accuracy_df is None or class_accuracy_df.empty:
        return pd.DataFrame(columns=ordered)
    df = _sort_scenarios(class_accuracy_df, leading_cols=['class_index'])
    before = df[df['stage'] == 'before'].drop_duplicates('class_index').set_index('class_index')
    for col in CLASS_METRIC_COLS:
        df[f'{col}_change_vs_before'] = df[col] - df['class_index'].map(before[col])
    return df[ordered]


def add_accuracy_to_summary(summary_df, accuracy_df):
    """Append before and per-(combo, lambda) accuracy, precision, recall and F1 columns to every Summary row."""
    if accuracy_df is None or accuracy_df.empty or summary_df.empty:
        return summary_df
    extra = {}
    before = accuracy_df[accuracy_df['stage'] == 'before']
    if not before.empty:
        for col in ACCURACY_METRIC_COLS:
            extra[f'{col}_before'] = before.iloc[0][col]
    after = accuracy_df[accuracy_df['stage'] == 'after'].sort_values(['combo_name', 'lambda_align'])
    for _, row in after.drop_duplicates(['combo_name', 'lambda_align']).iterrows():
        for col in ACCURACY_METRIC_COLS:
            extra[f'{col}_after_{row["combo_name"]}_lambda_{row["lambda_align"]}'] = row[col]
    summary_df = summary_df.copy()
    for col, value in extra.items():
        summary_df[col] = value
    return summary_df


def write_excel_report(records_df, run_info, excel_path, accuracy_df=None, class_accuracy_df=None):
    """
    (Re)build the full Excel workbook from scratch:
        Run_Info          - key/value dump of every run parameter
        Summary           - one row per concept, TCAV score + stats, plus model
                            accuracy/precision/recall/F1 before and after each combo/lambda
        Summary_By_Layer  - one row per concept/layer/stage/combo/lambda, TCAV score + stats
        Accuracy_Overall  - one row per model (before + each combo/lambda) with metrics and change vs before
        Accuracy_By_Class - per-class accuracy/precision/recall/F1 for every model evaluated
        TTest_<class>     - one sheet per class: paired t-tests of sensitivity per concept/combo/layer,
                            before vs each lambda, and every lambda vs lambda pair
        <concept sheets>  - full row-level detail, one sheet per concept
                            (split into _p1/_p2/... parts if a concept would
                            otherwise exceed Excel's per-sheet row limit)

    Writes to a temporary '.xlsx'-suffixed file first, then atomically
    replaces the target path, so a crash mid-write never corrupts the
    previous good report.
    """
    directory = os.path.dirname(excel_path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".~{os.path.basename(excel_path)}.tmp.xlsx")

    concept_names_in_order = list(dict.fromkeys(records_df['concept_name'].tolist()))
    concept_row_counts = records_df['concept_name'].value_counts()

    # Resolve every sheet name up front (reserving the fixed sheet names first)
    # so Summary/Summary_By_Layer can reference the exact per-concept sheet
    # name(s), and so sheets can be written in final reading order in one pass.
    used_sheet_names = {"run_info", "summary", "summary_by_layer", "accuracy_overall", "accuracy_by_class"}
    # One paired t-test sheet per class, reserved before the concept sheets so its name stays stable
    class_names_in_order = list(dict.fromkeys(records_df['class_name'].tolist()))
    ttest_sheet_names = {cls: _sanitize_sheet_name(f"TTest_{_safe_name(cls)}", used_sheet_names)
                         for cls in class_names_in_order}
    concept_sheet_names = {}
    for concept_name in concept_names_in_order:
        total_rows = int(concept_row_counts[concept_name])
        num_parts = max(1, math.ceil(total_rows / EXCEL_MAX_ROWS_PER_SHEET))
        names = []
        for part_idx in range(num_parts):
            base_name = concept_name if num_parts == 1 else f"{concept_name}_p{part_idx + 1}"
            names.append(_sanitize_sheet_name(base_name, used_sheet_names))
        concept_sheet_names[concept_name] = names

    summary_df = compute_concept_summary(records_df)
    summary_df.insert(1, 'excel_sheet_names', summary_df['concept_name'].map(
        lambda c: ", ".join(concept_sheet_names.get(c, []))))
    summary_df = add_accuracy_to_summary(summary_df, accuracy_df)
    layer_summary_df = compute_layer_summary(records_df)
    ttest_df = compute_paired_ttest(records_df)
    accuracy_overall_df = build_accuracy_overall(accuracy_df)
    class_accuracy_df = build_accuracy_by_class(class_accuracy_df)
    # Stringify every value: run_info deliberately mixes bools/ints/floats/strings
    # in one column, and writing that mix straight to Excel triggers a pandas/
    # openpyxl round-trip bug where some ints get silently reinterpreted as bools
    # (e.g. epochs=1 comes back as True) - this is a plain key/value reference
    # sheet for humans, so stringifying avoids the bug with no loss of meaning.
    run_info_df = pd.DataFrame([(k, str(v)) for k, v in run_info.items()], columns=["Parameter", "Value"])

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        run_info_df.to_excel(writer, sheet_name="Run_Info", index=False)
        _style_worksheet(writer.sheets["Run_Info"])

        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        _style_worksheet(writer.sheets["Summary"])

        layer_summary_df.to_excel(writer, sheet_name="Summary_By_Layer", index=False)
        _style_worksheet(writer.sheets["Summary_By_Layer"])

        accuracy_overall_df.to_excel(writer, sheet_name="Accuracy_Overall", index=False)
        _style_worksheet(writer.sheets["Accuracy_Overall"])

        class_accuracy_df.to_excel(writer, sheet_name="Accuracy_By_Class", index=False)
        _style_worksheet(writer.sheets["Accuracy_By_Class"])

        for class_name in class_names_in_order:
            sheet_name = ttest_sheet_names[class_name]
            class_ttest_df = ttest_df[ttest_df['class_name'] == class_name].reset_index(drop=True)
            class_ttest_df.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_worksheet(writer.sheets[sheet_name])

        for concept_name in concept_names_in_order:
            concept_df = records_df[records_df['concept_name'] == concept_name]
            concept_df = concept_df.sort_values(['layer_name', 'stage', 'combo_name', 'lambda_align', 'file_path'],
                                                ascending=[True, False, True, True, True])
            concept_df = concept_df[EXPORT_COLUMNS].reset_index(drop=True)
            names = concept_sheet_names[concept_name]
            for part_idx, sheet_name in enumerate(names):
                start = part_idx * EXCEL_MAX_ROWS_PER_SHEET
                end = start + EXCEL_MAX_ROWS_PER_SHEET
                part_df = concept_df.iloc[start:end]
                part_df.to_excel(writer, sheet_name=sheet_name, index=False)
                _style_worksheet(writer.sheets[sheet_name])

    os.replace(tmp_path, excel_path)
    return excel_path


def flush_raw_csv(records, csv_path, flushed_count):
    """
    Append any newly-accumulated rows (records: dict of column-name -> list of
    values) to a CSV backup, starting at index `flushed_count`, writing the
    header only the first time. This is far cheaper than rewriting the whole
    CSV on every call (as the original script did) since only new rows are
    ever written. Returns the new flushed_count.
    """
    if not records:
        return flushed_count
    total_count = len(next(iter(records.values())))
    if total_count <= flushed_count:
        return flushed_count
    new_rows = {col: values[flushed_count:total_count] for col, values in records.items()}
    new_df = pd.DataFrame(new_rows)
    write_header = not os.path.exists(csv_path)
    new_df.to_csv(csv_path, mode='a', header=write_header, index=False)
    return total_count


def main():
    ############## Parser #################################
    # Argument parser to override the model name and model path
    parser = argparse.ArgumentParser(description="Obtainthe original model path and the revised model path")
    parser.add_argument("--org_model_path", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--recal_model_basepath", type=str, default=None, help="Specify a location of the recalibrated model base path")
    parser.add_argument("--model_name", type=str, default=None, help="Specify a model name to override the default model")
    parser.add_argument("--before_after", action='store_true', help="Default parameter for before after comparison if its true then before after comparison will be done")
    parser.add_argument("--config_file", type=str, default=None, help="specify the yaml file ")
    parser.add_argument("--store_results", type=str, default=None, help="specify the location where teh results should be stored ")
    parser.add_argument("--load_validation_dataset", action='store_true', help="If passed the validation dataset will be loaded instead (default: False)")
    parser.add_argument("--concept_mode", type=str, default="auto", choices=["auto", "single", "multiclass"],
                         help="'single' = one concept per class (config 'concept' section). "
                              "'multiclass' = multiple concepts per class (config 'multiconcept' section). "
                              "'auto' (default) picks 'multiclass' when the YAML config has multiconcept "
                              "enabled, otherwise falls back to 'single'.")
    parser.add_argument("--manifest_file", type=str, default=None,
                         help="Path to a per-model CAV manifest JSON (e.g. vgg16_manifest.json, produced by "
                              "main_store_cav.py / cav_registry.CAVRegistry). Sensitivity scores are computed "
                              "using CAV vectors loaded from this manifest instead of being trained on the fly; "
                              "if a concept/layer's CAV is not present in the manifest, that layer is skipped "
                              "for that concept with a warning.")

    args = parser.parse_args()
    config_file = args.config_file
    save_dir = args.store_results
    load_validationdataset = args.load_validation_dataset
    print(config_file)
    if config_file is not None:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")
    else:
        raise FileNotFoundError(f"Config file parameter not provided in the command line")
    if not save_dir:
        raise ValueError("Please provide a --store_results directory to save the log/CSV/Excel outputs.")
    manifest_file = args.manifest_file
    if not manifest_file:
        raise ValueError(
            "Please provide a --manifest_file file (e.g. vgg16_manifest.json, produced by main_store_cav.py) "
            "so precomputed CAV vectors can be loaded - sensitivity scores are no longer computed on the fly."
        )
    if not os.path.isfile(manifest_file):
        raise FileNotFoundError(f"Manifest file '{manifest_file}' does not exist.")
    manifest_data = load_manifest(manifest_file)
    config = ConfigSingleton(config_file)
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    if(DEVICE =='cuda'):
      torch.cuda.manual_seed(RANDOM_STATE)
      torch.cuda.manual_seed_all(RANDOM_STATE)  # For multi-GPU setups
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    if(DEVICE =='cuda'):
      # Ensure deterministic behavior (may impact performance)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
    CLASSIFICATION_DATA_BASE_PATH = config.CLASSIFICATION_DATA_BASE_PATH
    TARGET_CLASS_LIST = config.TARGET_CLASS_LIST
    RANDOM_FOLDER = config.RANDOM_FOLDER
    CONCEPT_FOLDER_LIST = config.CONCEPT_FOLDER_LIST
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    NUM_CLASSES = config.NUM_CLASSES
    LAMBDA_ALIGNS = config.LAMBDA_ALIGNS
    LINEAR_CLASSIFIER_TYPE = config.LINEAR_CLASSIFIER_TYPE

    print("Config file loaded successfully.")

    concept_mode = args.concept_mode
    if concept_mode == 'auto':
        concept_mode = 'multiclass' if getattr(config, 'MULTICONCEPT_ENABLED', False) else 'single'
    if concept_mode == 'multiclass' and not getattr(config, 'MULTICONCEPT_ENABLED', False):
        raise ValueError("--concept_mode multiclass was requested but the YAML config has no "
                          "(or an empty) 'multiconcept' section.")
    print(f"Concept mode resolved to: {concept_mode}")

    # Get the training dataset and the validation dataset folders

    before_after = args.before_after
    BASE_MODEL_PATH = args.org_model_path
    MODEL_NAME = args.model_name
    lambda_val_list = LAMBDA_ALIGNS
    if(args.recal_model_basepath):
        recal_model_basepath = args.recal_model_basepath
    else:
        recal_model_basepath = None


    if not BASE_MODEL_PATH or not MODEL_NAME:
        raise ValueError("Please provide valid paths for org_model_path, and model_name")
    print(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    formatted_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dataset_split = "valid" if load_validationdataset else "train"
    ############## Logging #################################

    save_folder = os.path.join(save_dir, MODEL_NAME)
    os.makedirs(save_folder, exist_ok=True)
    log_filename = os.path.join(save_folder, f"{formatted_datetime}_sensitivity_compute.log")
    csv_filename = os.path.join(save_folder, f"sensitivity_audit_trail_{MODEL_NAME}_{formatted_datetime}.csv")
    excel_filename = os.path.join(save_folder, f"sensitivity_report_{MODEL_NAME}_{formatted_datetime}.xlsx")


    logger = Logger_Singleton(log_filename)
    logger.info(f"Using org_model_path: {BASE_MODEL_PATH}, model_name: {MODEL_NAME}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Concept mode: {concept_mode}")
    print(f"Results are stored in {log_filename}, {csv_filename}, {excel_filename}")
    # Set the device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    ############## Model BEFORE #################################
    #Load the model
    print(MODEL_NAME, BASE_MODEL_PATH)
    model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
    model_trained.to(device)
    if before_after and config.OVERRIDE_RECALIB:
        layers = get_layernames_override(MODEL_NAME, config)
    else:
        layers = get_model_layers(model_trained)
    print(layers)
    try:
        del model_trained
        torch.cuda.empty_cache()
    except Exception as e:
        print("Model trained variable not yet defined ")
    ############## Load data #################################
    dataset_loader, val_loader, TRAIN_TRANSFORM, VALID_TRANSFORM, class_names = load_train_valid_dataset(MODEL_NAME, CLASSIFICATION_DATA_BASE_PATH, BATCH_SIZE, random_state=RANDOM_STATE)
    TARGET_IDX_LIST = [class_names.index(cls) for cls in TARGET_CLASS_LIST]

    data_root = os.path.join(CLASSIFICATION_DATA_BASE_PATH, dataset_split)
    class_dataloaders = [DataLoader(SingleClassDataLoader(os.path.join(data_root, class_name), transform=VALID_TRANSFORM), batch_size=BATCH_SIZE)
                          for class_name in TARGET_CLASS_LIST]

    concept_specs = build_concept_specs(
        config, concept_mode, TARGET_CLASS_LIST, TARGET_IDX_LIST, class_dataloaders,
        CONCEPT_FOLDER_LIST, RANDOM_FOLDER,
    )
    concept_names_resolved = [spec['concept_name'] for spec in concept_specs]
    print(f"Resolved {len(concept_specs)} concept(s) in '{concept_mode}' mode: {concept_names_resolved}")
    logger.info(f"Resolved {len(concept_specs)} concept(s) in '{concept_mode}' mode: {concept_names_resolved}")

    # File list per concept, derived directly from that concept's own class
    # dataloader (deterministic order, shuffle=False) - this is what lets every
    # sensitivity-score row be matched back to the exact image file that produced it.
    concept_file_lists = {spec['concept_name']: get_singleclass_filelist(spec['class_dataloader']) for spec in concept_specs}

    # Every recalibration scenario = one combo of layers from layer_combinations_<model>.txt x every lambda
    recalibration_scenarios = []
    if before_after:
        if recal_model_basepath is None:
            msg = "--before_after was set without --recal_model_basepath; only the 'before' stage will be computed."
            print(f"WARNING: {msg}")
            logger.warning(msg)
        else:
            layer_combinations_file = os.path.join(
                recal_model_basepath, MODEL_NAME, f"layer_combinations_{MODEL_NAME}.txt")
            recalibration_scenarios = load_layer_combinations(layer_combinations_file)
            logger.info(f"Recalibration combos from {layer_combinations_file}: {recalibration_scenarios}")
            print(f"Recalibration combos: {recalibration_scenarios}")
    # 'before' sensitivity is needed at every layer that any combo recalibrated
    before_layers = list(layers)
    for _, combo_layers in recalibration_scenarios:
        before_layers.extend(ln for ln in combo_layers if ln not in before_layers)

    run_info = {
        "model_name": MODEL_NAME,
        "org_model_path": os.path.abspath(BASE_MODEL_PATH),
        "recal_model_basepath": os.path.abspath(recal_model_basepath) if recal_model_basepath else "",
        "config_file": os.path.abspath(config_file),
        "manifest_file": os.path.abspath(manifest_file),
        "concept_mode": concept_mode,
        "before_after": before_after,
        "dataset_split": dataset_split,
        "classification_data_base_path": os.path.abspath(CLASSIFICATION_DATA_BASE_PATH),
        "num_concepts": len(concept_specs),
        "concept_names": ", ".join(concept_names_resolved),
        "target_class_list": ", ".join(TARGET_CLASS_LIST),
        "num_classes_in_dataset": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "linear_classifier_type": LINEAR_CLASSIFIER_TYPE,
        "lambda_aligns": ", ".join(str(v) for v in lambda_val_list),
        "cav_random_state": RANDOM_STATE,
        "device": DEVICE,
        "run_timestamp": formatted_datetime,
        "log_file": os.path.abspath(log_filename),
        "csv_backup_file": os.path.abspath(csv_filename),
        "excel_report_file": os.path.abspath(excel_filename),
        "layers_to_process": ", ".join(before_layers),
        "num_layers_to_process": len(before_layers),
        "recalibration_combos": "; ".join(f"{name}: {'|'.join(cl)}" for name, cl in recalibration_scenarios),
        "num_recalibration_scenarios": len(recalibration_scenarios) * len(lambda_val_list),
    }

    records = defaultdict(list)
    flushed_count = 0
    accuracy_rows = []
    class_accuracy_rows = []
    evaluated_model_paths = set()

    def record_model_metrics(model, stage, lambda_align, combo_name, recalibrated_layers, model_weight_path):
        """Evaluate a model on the validation set once per weight file and store the metrics."""
        model_weight_path = os.path.abspath(model_weight_path)
        if model_weight_path in evaluated_model_paths:
            return
        overall_row, per_class_rows = compute_classification_metrics(
            model, val_loader, class_names, device, stage, lambda_align, model_weight_path,
            combo_name=combo_name, recalibrated_layers=recalibrated_layers)
        accuracy_rows.append(overall_row)
        class_accuracy_rows.extend(per_class_rows)
        evaluated_model_paths.add(model_weight_path)
        msg = (f"[{stage}][{combo_name}][lambda={lambda_align}] accuracy={overall_row['accuracy']:.4f} "
               f"precision={overall_row['precision_macro']:.4f} recall={overall_row['recall_macro']:.4f} "
               f"f1={overall_row['f1_macro']:.4f} ({model_weight_path})")
        print(msg)
        logger.info(msg)

    def append_records(concept_name, class_name, target_class_index, stage, layer_name, lambda_align,
                        sensitivity_scores, file_paths, concept_folder_path, random_folder_path, model_weight_path,
                        cav_source_path, combo_name, recalibrated_layers):
        n = len(sensitivity_scores)
        records['model_name'].extend([MODEL_NAME] * n)
        records['concept_name'].extend([concept_name] * n)
        records['class_name'].extend([class_name] * n)
        records['target_class_index'].extend([target_class_index] * n)
        records['stage'].extend([stage] * n)
        records['combo_name'].extend([combo_name] * n)
        records['recalibrated_layers'].extend([recalibrated_layers] * n)
        records['layer_name'].extend([layer_name] * n)
        records['lambda_align'].extend([lambda_align] * n)
        records['sensitivity_score'].extend([float(s) for s in sensitivity_scores])
        records['positive_sensitivity'].extend([1 if float(s) > 0 else 0 for s in sensitivity_scores])
        records['file_path'].extend(file_paths)
        records['concept_folder_path'].extend([concept_folder_path] * n)
        records['random_folder_path'].extend([random_folder_path] * n)
        records['model_weight_path'].extend([model_weight_path] * n)
        records['cav_source_path'].extend([cav_source_path] * n)
        records['dataset_split'].extend([dataset_split] * n)
        records['concept_mode'].extend([concept_mode] * n)
        records['cav_random_state'].extend([RANDOM_STATE] * n)
        records['batch_size'].extend([BATCH_SIZE] * n)
        records['linear_classifier_type'].extend([LINEAR_CLASSIFIER_TYPE] * n)
        records['manifest_file'].extend([os.path.abspath(manifest_file)] * n)
        records['config_file'].extend([os.path.abspath(config_file)] * n)
        records['run_timestamp'].extend([formatted_datetime] * n)

    def checkpoint(reason="checkpoint", rebuild_excel=True):
        nonlocal flushed_count
        try:
            flushed_count = flush_raw_csv(records, csv_filename, flushed_count)
        except Exception as e:
            print(f"Warning: failed to flush raw CSV backup ({reason}): {e}")
            logger.info(f"Failed to flush raw CSV backup ({reason}): {e}")
        if not rebuild_excel:
            return
        try:
            if records['model_name']:
                records_df = pd.DataFrame(records)
                write_excel_report(records_df, run_info, excel_filename,
                                   accuracy_df=pd.DataFrame(accuracy_rows),
                                   class_accuracy_df=pd.DataFrame(class_accuracy_rows))
                print(f"Excel report rebuilt ({reason}): {excel_filename}")
                logger.info(f"Excel report rebuilt ({reason}): {excel_filename}")
        except Exception as e:
            print(f"Warning: failed to rebuild Excel report ({reason}): {e}")
            logger.info(f"Failed to rebuild Excel report ({reason}): {e}")

    cav_cache = {}

    def get_layer_cavs(layer_name):
        """Resolve every concept's precomputed CAV for a layer once: ({concept: (cav, source)}, active_specs)."""
        if layer_name in cav_cache:
            return cav_cache[layer_name]
        cav_lookup = {}
        for spec in concept_specs:
            cav_tensor, cav_source_path, reason = get_cav_from_manifest(
                manifest_data, spec['concept_name'], layer_name, device)
            if cav_tensor is None:
                msg = (f"Sensitivity score for layer '{layer_name}' (concept "
                       f"'{spec['concept_name']}') was not processed due to non-availability "
                       f"of cav in the manifest ({reason}).")
                print(f"WARNING: {msg}")
                logger.warning(msg)
                continue
            cav_lookup[spec['concept_name']] = (cav_tensor, cav_source_path)
        active_specs = [spec for spec in concept_specs if spec['concept_name'] in cav_lookup]
        cav_cache[layer_name] = (cav_lookup, active_specs)
        return cav_cache[layer_name]

    def compute_and_record_sensitivity(model, layer_name, stage, lambda_align, combo_name, recalibrated_layers,
                                       model_weight_path):
        """Compute per-image sensitivity at one layer for every concept and append the rows."""
        cav_lookup, active_specs = get_layer_cavs(layer_name)
        tag = f"[{layer_name}]" if stage == 'before' else f"[{combo_name}][{layer_name}][lambda={lambda_align}]"
        if not active_specs:
            msg = f"{tag} No CAVs available in manifest for any concept - skipping this layer."
            print(f"WARNING: {msg}")
            logger.warning(msg)
            return
        hook_handle = model.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
        try:
            logger.info(f"{tag} Computing the sensitivity score ({stage}) using manifest CAVs, can take a while stand by")
            sensitivity_scores = [
                util_compute_sensitivity_score(model, layer_name, cav_lookup[spec['concept_name']][0],
                                               spec['class_dataloader'], spec['target_class_index'], activation)
                for spec in active_specs
            ]
            tcav_legacy = util_compute_tcav_score_from_sensitivity(sensitivity_scores)
            logger.info(f"{tag} legacy mean-sensitivity 'tcav' ({stage}) per concept is {tcav_legacy}")
            sensitivity_scores = [cpudata.detach().cpu().numpy() for cpudata in sensitivity_scores]

            for spec, scores in zip(active_specs, sensitivity_scores):
                file_paths = concept_file_lists[spec['concept_name']]
                if len(file_paths) != len(scores):
                    msg = (f"{tag} concept '{spec['concept_name']}': file-list length "
                           f"({len(file_paths)}) != sensitivity-score length ({len(scores)}); "
                           f"truncating to the shorter of the two.")
                    print("WARNING " + msg)
                    logger.info("WARNING " + msg)
                n = min(len(file_paths), len(scores))
                positive_fraction = float(np.mean(scores[:n] > 0)) if n else float('nan')
                mean_score = float(np.mean(scores[:n])) if n else float('nan')
                print(f"{tag} {stage.upper()} concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score:.4f}, positive_fraction={positive_fraction:.4f}")
                logger.info(f"{tag} {stage.upper()} concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score}, positive_fraction={positive_fraction}")
                append_records(
                    concept_name=spec['concept_name'], class_name=spec['class_name'],
                    target_class_index=spec['target_class_index'], stage=stage, layer_name=layer_name,
                    lambda_align=lambda_align, sensitivity_scores=scores[:n], file_paths=file_paths[:n],
                    concept_folder_path=spec['concept_folder'], random_folder_path=spec['random_folder'],
                    model_weight_path=os.path.abspath(model_weight_path),
                    cav_source_path=cav_lookup[spec['concept_name']][1],
                    combo_name=combo_name, recalibrated_layers=recalibrated_layers,
                )
        finally:
            hook_handle.remove()
            activation.clear()  # Clear activations to free memory
            torch.cuda.empty_cache()

    ############## BEFORE: base model, loaded once #################################
    base_model = None
    try:
        base_model = load_model(MODEL_NAME, BASE_MODEL_PATH)
        base_model.to(device)
        # Accuracy is computed before any forward hook is attached
        try:
            record_model_metrics(base_model, 'before', np.nan, 'base', '', BASE_MODEL_PATH)
        except Exception as e:
            print(f"Warning: failed to compute accuracy of the base model: {e}")
            logger.warning(f"Failed to compute accuracy of the base model: {e}")
        for layer_name in before_layers:
            try:
                compute_and_record_sensitivity(base_model, layer_name, 'before', np.nan, 'base', '', BASE_MODEL_PATH)
                checkpoint(reason=f"before/{layer_name}", rebuild_excel=False)
            except Exception as e:
                checkpoint(reason=f"before/{layer_name}/exception", rebuild_excel=False)
                print(f"Obtained exception while processing Layer{layer_name} (before), Exception details {e}")
                logger.info(f"Obtained exception while processing Layer{layer_name} (before), Exception details {e}")
    except Exception as e:
        print(f"Obtained exception while loading/processing the base model: {e}")
        logger.info(f"Obtained exception while loading/processing the base model: {e}")
    finally:
        del base_model
        torch.cuda.empty_cache()
    checkpoint(reason="before/end", rebuild_excel=True)

    ############## AFTER: every combo x every lambda #################################
    for combo_name, combo_layers in recalibration_scenarios:
        recalibrated_layers = "|".join(combo_layers)
        for lambda_val in lambda_val_list:
            scenario = f"{combo_name}/lambda={lambda_val}"
            try:
                if concept_mode == 'multiclass':
                    modified_model_path = get_model_path_multiclass(MODEL_NAME, combo_name, lambda_val, recal_model_basepath)
                else:
                    modified_model_path = get_model_path(MODEL_NAME, combo_name, lambda_val, recal_model_basepath)
            except FileNotFoundError as e:
                print(f"WARNING: [{scenario}] recalibrated model not found, skipping: {e}")
                logger.warning(f"[{scenario}] recalibrated model not found, skipping: {e}")
                continue

            model_trained = None
            try:
                model_trained = load_model_with_trained_weights(BASE_MODEL_PATH, modified_model_path, device=device)
                try:
                    record_model_metrics(model_trained, 'after', lambda_val, combo_name, recalibrated_layers,
                                         modified_model_path)
                except Exception as e:
                    print(f"Warning: failed to compute accuracy for {modified_model_path}: {e}")
                    logger.warning(f"[{scenario}] Failed to compute accuracy for {modified_model_path}: {e}")
                for layer_name in combo_layers:
                    try:
                        compute_and_record_sensitivity(model_trained, layer_name, 'after', lambda_val, combo_name,
                                                       recalibrated_layers, modified_model_path)
                    except Exception as e:
                        print(f"Exception obtained while computing sensitivity score [{scenario}][{layer_name}]: {e}")
                        logger.info(f"[{scenario}][{layer_name}] Exception obtained while computing sensitivity score {e}")
            except Exception as e:
                print(f"Obtained exception while processing {scenario}: {e}")
                logger.info(f"Obtained exception while processing {scenario}: {e}")
            finally:
                del model_trained
                torch.cuda.empty_cache()
                checkpoint(reason=scenario, rebuild_excel=False)
        checkpoint(reason=f"{combo_name}/end_of_combo", rebuild_excel=True)

    checkpoint(reason="final", rebuild_excel=True)
    if os.path.isfile(csv_filename):
        print(f"DONE. Raw CSV backup written to: {csv_filename}")
        logger.info(f"DONE. Raw CSV backup written to: {csv_filename}")
    else:
        msg = "No rows were collected (no CAVs were available in the manifest for any layer/concept) - no CSV backup was written."
        print(f"WARNING: {msg}")
        logger.warning(msg)
    if os.path.isfile(excel_filename):
        print(f"DONE. Excel report written to: {excel_filename}")
        logger.info(f"DONE. Excel report written to: {excel_filename}")
    else:
        msg = "No rows were collected (no CAVs were available in the manifest for any layer/concept) - no Excel report was written."
        print(f"WARNING: {msg}")
        logger.warning(msg)


if __name__ == "__main__":
    main()

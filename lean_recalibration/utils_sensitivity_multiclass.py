import os
import os.path
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
from torch.utils.data import DataLoader
from openpyxl.styles import Font, PatternFill, Alignment

from logger import Logger_Singleton
from custom_dataloader import SingleClassDataLoader

from ConfigSingleton import ConfigSingleton
from utils import (
    get_model_layers, load_model, load_model_statedict,
    load_train_valid_dataset,
)

from tcav_utils import (util_compute_sensitivity_score, util_compute_tcav_score_from_sensitivity)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


RANDOM_STATE = 132
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
    'model_name', 'concept_name', 'class_name', 'target_class_index', 'stage', 'layer_name',
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


MODEL = None
TRAIN_TRANSFORM = None
VALID_TRANSFORM = None
LAYER_NAMES = None
OVERRIDE_RECALIB = None


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
            'num_lambdas_processed': int(after_group['lambda_align'].nunique()) if after_total else 0,
        }
        for lambda_val in sorted(after_group['lambda_align'].dropna().unique()):
            lam_group = after_group[after_group['lambda_align'] == lambda_val]
            lam_total = len(lam_group)
            lam_positive = int((lam_group['sensitivity_score'] > 0).sum())
            row[f'tcav_score_after_lambda_{lambda_val}'] = lam_positive / lam_total if lam_total else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


def compute_layer_summary(records_df):
    """
    Build the 'Summary_By_Layer' sheet: one row per (concept, layer, stage,
    lambda) combination with its own TCAV score/counts/stats - a granular,
    reproduces-the-per-layer-TCAV-you'd-expect breakdown, computed entirely
    from the stored rows (no recomputation against the model needed).
    """
    group_cols = ['concept_name', 'model_name', 'class_name', 'layer_name', 'stage', 'lambda_align']
    rows = []
    for keys, group in records_df.groupby(group_cols, sort=False, dropna=False):
        concept_name, model_name, class_name, layer_name, stage, lambda_align = keys
        total_rows = len(group)
        positive_rows = int((group['sensitivity_score'] > 0).sum())
        tcav_score = positive_rows / total_rows if total_rows else float('nan')
        rows.append({
            'concept_name': concept_name,
            'model_name': model_name,
            'class_name': class_name,
            'layer_name': layer_name,
            'stage': stage,
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
        result = result.sort_values(['concept_name', 'layer_name', 'stage', 'lambda_align']).reset_index(drop=True)
    return result


def write_excel_report(records_df, run_info, excel_path):
    """
    (Re)build the full Excel workbook from scratch:
        Run_Info          - key/value dump of every run parameter
        Summary           - one row per concept, TCAV score + stats
        Summary_By_Layer  - one row per concept/layer/stage/lambda, TCAV score + stats
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
    used_sheet_names = {"run_info", "summary", "summary_by_layer"}
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
    layer_summary_df = compute_layer_summary(records_df)
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

        for concept_name in concept_names_in_order:
            concept_df = records_df[records_df['concept_name'] == concept_name]
            concept_df = concept_df.sort_values(['layer_name', 'stage', 'lambda_align', 'file_path'])
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
    parser.add_argument("--config", type=str, default=None, help="specify the yaml file ")
    parser.add_argument("--store_results", type=str, default=None, help="specify the location where teh results should be stored ")
    parser.add_argument("--load_validation_dataset", action='store_true', help="If passed the validation dataset will be loaded instead (default: False)")
    parser.add_argument("--concept_mode", type=str, default="auto", choices=["auto", "single", "multiclass"],
                         help="'single' = one concept per class (config 'concept' section). "
                              "'multiclass' = multiple concepts per class (config 'multiconcept' section). "
                              "'auto' (default) picks 'multiclass' when the YAML config has multiconcept "
                              "enabled, otherwise falls back to 'single'.")
    parser.add_argument("--manifest", type=str, default=None,
                         help="Path to a per-model CAV manifest JSON (e.g. vgg16_manifest.json, produced by "
                              "main_store_cav.py / cav_registry.CAVRegistry). Sensitivity scores are computed "
                              "using CAV vectors loaded from this manifest instead of being trained on the fly; "
                              "if a concept/layer's CAV is not present in the manifest, that layer is skipped "
                              "for that concept with a warning.")

    args = parser.parse_args()
    config_file = args.config
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
    manifest_file = args.manifest
    if not manifest_file:
        raise ValueError(
            "Please provide a --manifest file (e.g. vgg16_manifest.json, produced by main_store_cav.py) "
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
        "layers_to_process": ", ".join(layers),
        "num_layers_to_process": len(layers),
    }

    records = defaultdict(list)
    flushed_count = 0

    def append_records(concept_name, class_name, target_class_index, stage, layer_name, lambda_align,
                        sensitivity_scores, file_paths, concept_folder_path, random_folder_path, model_weight_path,
                        cav_source_path):
        n = len(sensitivity_scores)
        records['model_name'].extend([MODEL_NAME] * n)
        records['concept_name'].extend([concept_name] * n)
        records['class_name'].extend([class_name] * n)
        records['target_class_index'].extend([target_class_index] * n)
        records['stage'].extend([stage] * n)
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
                write_excel_report(records_df, run_info, excel_filename)
                print(f"Excel report rebuilt ({reason}): {excel_filename}")
                logger.info(f"Excel report rebuilt ({reason}): {excel_filename}")
        except Exception as e:
            print(f"Warning: failed to rebuild Excel report ({reason}): {e}")
            logger.info(f"Failed to rebuild Excel report ({reason}): {e}")

    for layer_name in layers:
        try:
            # Resolve every concept's CAV vector for this layer from the manifest
            # up front (CAVs are precomputed/fixed - they don't change between
            # 'before' and each recalibrated 'after' stage, so this is done once
            # per layer instead of being retrained per stage/lambda as before).
            cav_lookup = {}  # concept_name -> (cav_tensor, cav_source_path)
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
            if not active_specs:
                msg = f"[{layer_name}] No CAVs available in manifest for any concept - skipping this layer entirely."
                print(f"WARNING: {msg}")
                logger.warning(msg)
                continue

            ############## BEFORE Do it once #################################
            model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
            model_trained.to(device)
            hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
            logger.info(f"[{layer_name}] Computing the sensitivity score (before) using manifest CAVs, can take a while stand by")
            sensitivity_scores = [
                util_compute_sensitivity_score(model_trained, layer_name, cav_lookup[spec['concept_name']][0],
                                                spec['class_dataloader'], spec['target_class_index'], activation)
                for spec in active_specs
            ]
            tcav_before_legacy = util_compute_tcav_score_from_sensitivity(sensitivity_scores)
            logger.info(f"[{layer_name}] legacy mean-sensitivity 'tcav' (before) per concept is {tcav_before_legacy}")
            sensitivity_scores = [cpudata.detach().cpu().numpy() for cpudata in sensitivity_scores]

            for spec, scores in zip(active_specs, sensitivity_scores):
                file_paths = concept_file_lists[spec['concept_name']]
                if len(file_paths) != len(scores):
                    msg = (f"[{layer_name}] concept '{spec['concept_name']}': file-list length "
                           f"({len(file_paths)}) != sensitivity-score length ({len(scores)}); "
                           f"truncating to the shorter of the two.")
                    print("WARNING " + msg)
                    logger.info("WARNING " + msg)
                n = min(len(file_paths), len(scores))
                positive_fraction = float(np.mean(scores[:n] > 0)) if n else float('nan')
                mean_score = float(np.mean(scores[:n])) if n else float('nan')
                print(f"[{layer_name}] BEFORE concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score:.4f}, positive_fraction={positive_fraction:.4f}")
                logger.info(f"[{layer_name}] BEFORE concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score}, positive_fraction={positive_fraction}")
                append_records(
                    concept_name=spec['concept_name'], class_name=spec['class_name'],
                    target_class_index=spec['target_class_index'], stage='before', layer_name=layer_name,
                    lambda_align=np.nan, sensitivity_scores=scores[:n], file_paths=file_paths[:n],
                    concept_folder_path=spec['concept_folder'], random_folder_path=spec['random_folder'],
                    model_weight_path=os.path.abspath(BASE_MODEL_PATH),
                    cav_source_path=cav_lookup[spec['concept_name']][1],
                )

            hook_handle.remove()
            activation.clear()  # Clear activations to free memory
            torch.cuda.empty_cache()
            try:
                del model_trained
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Model trained variable not yet defined  ")

            if(before_after == True):
                for lambda_val in lambda_val_list:
                    try:
                        ###########AFTER######################
                        if(recal_model_basepath != None):
                            modified_model_path = get_model_path(MODEL_NAME, layer_name, lambda_val, recal_model_basepath)
                            model_trained = load_model(MODEL_NAME, BASE_MODEL_PATH)
                            model_trained = load_model_statedict(model_trained, modified_model_path)
                            hook_handle = model_trained.get_submodule(layer_name).register_forward_hook(get_activation(layer_name))
                            model_trained.to(device)
                            logger.info(f"[{layer_name}][lambda={lambda_val}] Computing the sensitivity score (after) using manifest CAVs, can take a while stand by")
                            try:
                                sensitivity_scores = [
                                    util_compute_sensitivity_score(model_trained, layer_name, cav_lookup[spec['concept_name']][0],
                                                                    spec['class_dataloader'], spec['target_class_index'], activation)
                                    for spec in active_specs
                                ]
                                tcav_after_legacy = util_compute_tcav_score_from_sensitivity(sensitivity_scores)
                                logger.info(f"[{layer_name}][lambda={lambda_val}] legacy mean-sensitivity 'tcav' (after) per concept is {tcav_after_legacy}")
                                sensitivity_scores = [cpudata.detach().cpu().numpy() for cpudata in sensitivity_scores]

                                for spec, scores in zip(active_specs, sensitivity_scores):
                                    file_paths = concept_file_lists[spec['concept_name']]
                                    n = min(len(file_paths), len(scores))
                                    positive_fraction = float(np.mean(scores[:n] > 0)) if n else float('nan')
                                    mean_score = float(np.mean(scores[:n])) if n else float('nan')
                                    print(f"[{layer_name}][lambda={lambda_val}] AFTER concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score:.4f}, positive_fraction={positive_fraction:.4f}")
                                    logger.info(f"[{layer_name}][lambda={lambda_val}] AFTER concept={spec['concept_name']}: n={n}, mean_sensitivity={mean_score}, positive_fraction={positive_fraction}")
                                    append_records(
                                        concept_name=spec['concept_name'], class_name=spec['class_name'],
                                        target_class_index=spec['target_class_index'], stage='after', layer_name=layer_name,
                                        lambda_align=lambda_val, sensitivity_scores=scores[:n], file_paths=file_paths[:n],
                                        concept_folder_path=spec['concept_folder'], random_folder_path=spec['random_folder'],
                                        model_weight_path=os.path.abspath(modified_model_path),
                                        cav_source_path=cav_lookup[spec['concept_name']][1],
                                    )

                                hook_handle.remove()
                                activation.clear()  # Clear activations to free memory
                                torch.cuda.empty_cache()
                                checkpoint(reason=f"{layer_name}/lambda={lambda_val}", rebuild_excel=False)
                            except Exception as e:
                                print(f"Exception obtained while computing sensitivity score {e}")
                                logger.info(f"[{layer_name}][lambda={lambda_val}] Exception obtained while computing sensitivity score {e}")
                                continue
                    except Exception as e:
                        checkpoint(reason=f"{layer_name}/lambda={lambda_val}/exception", rebuild_excel=False)
                        print(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}")
                        logger.info(f"Obtained exception while processing Layer{layer_name}, with Lambda value {lambda_val}: {e}")
                        continue
                try:
                    del model_trained
                except Exception as e:
                    print(f"Model trained variable not yet defined  ")
                    continue
            checkpoint(reason=f"{layer_name}/end_of_layer", rebuild_excel=True)
        except Exception as e:
            checkpoint(reason=f"{layer_name}/exception", rebuild_excel=True)
            print(f"Obtained exception while processing Layer{layer_name} , Exception details {e}")
            logger.info(f"Obtained exception while processing Layer{layer_name} , Exception details {e}")
            continue

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

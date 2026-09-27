"""
Builders for synthetic sensitivity-report workbooks/CSVs matching the exact schema that
``bottleneck_detection.py`` expects (see its module docstring / ``REQUIRED_LONG_COLUMNS``),
and that ``utils_sensitivity_multiclass.py`` produces in ``write_excel_report``.

Kept separate from the real pipeline so ``test_bottleneck_detection.py`` can unit test the
CSPI/bottleneck math in complete isolation (no model/CAV/training dependency at all), while
``test_integration_pipeline.py`` instead feeds it the REAL workbook produced by
``utils_sensitivity_multiclass.py``.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


def _synthetic_scores(n_samples: int, layers: list, seed: int, drift: float = 0.15) -> np.ndarray:
    """Deterministic (sample x layer) matrix with real variance in every column/row and a
    generally increasing-then-drifting trend across layers (loosely mimics real CAV
    sensitivity scores) so alignment/correlation computations are well-defined (non-zero
    variance, non-singular)."""
    rng = np.random.RandomState(seed)
    base = rng.normal(loc=0.5, scale=0.2, size=(n_samples, len(layers)))
    trend = np.linspace(0, drift * len(layers), len(layers))
    return base + trend[np.newaxis, :]


def build_sensitivity_workbook(
    path: str,
    concept_names: Iterable[str],
    layers: list,
    n_samples: int = 5,
    lambdas: Iterable[float] = (0.5,),
    include_before: bool = True,
    include_after: bool = True,
    class_name_for_concept: dict | None = None,
    seed: int = 0,
) -> str:
    """Write a synthetic multiclass sensitivity-report .xlsx at `path`.

    Structure mirrors utils_sensitivity_multiclass.write_excel_report(): 'Run_Info',
    'Summary' (+ 'Summary_By_Layer', omitted here as bottleneck_detection.py does not
    require it), and one long-format sheet per concept with columns
    {class_name, stage, layer_name, lambda_align, sensitivity_score, file_path}.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    class_name_for_concept = class_name_for_concept or {}
    concept_names = list(concept_names)

    run_info_df = pd.DataFrame(
        [
            ("model_name", "vgg16"),
            ("layers_to_process", ",".join(layers)),
            ("num_samples_per_stage", n_samples),
        ],
        columns=["Parameter", "Value"],
    )

    summary_rows = []
    sheet_frames = {}
    for c_idx, concept in enumerate(concept_names):
        class_name = class_name_for_concept.get(concept, "class0")
        rows = []
        seed_here = seed + c_idx * 97

        if include_before:
            scores = _synthetic_scores(n_samples, layers, seed_here)
            positive_before = 0
            for s in range(n_samples):
                for l_idx, layer in enumerate(layers):
                    val = float(scores[s, l_idx])
                    positive_before += int(val > 0)
                    rows.append({
                        "class_name": class_name, "stage": "before", "layer_name": layer,
                        "lambda_align": np.nan, "sensitivity_score": val,
                        "file_path": f"{concept}_sample_{s:02d}.jpg",
                    })
            tcav_before = positive_before / max(1, n_samples * len(layers))
        else:
            tcav_before = None

        tcav_after = {}
        if include_after:
            for lam in lambdas:
                scores = _synthetic_scores(n_samples, layers, seed_here + 1000 + int(lam * 100), drift=0.05)
                positive_after = 0
                for s in range(n_samples):
                    for l_idx, layer in enumerate(layers):
                        val = float(scores[s, l_idx])
                        positive_after += int(val > 0)
                        rows.append({
                            "class_name": class_name, "stage": "after", "layer_name": layer,
                            "lambda_align": float(lam), "sensitivity_score": val,
                            "file_path": f"{concept}_sample_{s:02d}.jpg",
                        })
                tcav_after[lam] = positive_after / max(1, n_samples * len(layers))

        sheet_frames[concept] = pd.DataFrame(rows)

        summary_row = {
            "concept_name": concept, "class_name": class_name,
            "excel_sheet_names": concept, "n_samples": n_samples,
        }
        if tcav_before is not None:
            summary_row["tcav_score_before"] = tcav_before
        if tcav_after:
            summary_row["tcav_score_after_overall"] = float(np.mean(list(tcav_after.values())))
            for lam, val in tcav_after.items():
                summary_row[f"tcav_score_after_lambda_{int(round(lam * 100)):02d}"] = val
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        run_info_df.to_excel(writer, sheet_name="Run_Info", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for concept, df in sheet_frames.items():
            df.to_excel(writer, sheet_name=concept[:31], index=False)

    return path


def build_legacy_csv(path: str, n_classes: int = 2, n_samples: int = 5, layers: list = None, seed: int = 0) -> str:
    """Write a synthetic legacy wide-format CSV matching load_and_prepare_data()'s schema:
    'Full Class Index', 'Full filepath', and one 'sensitivityscore_before_<layer>' column
    per layer (plus throw-away 'sensitivityscore_After_<layer>' columns that the loader
    explicitly drops, to exercise that code path too)."""
    layers = layers or ["features.3", "features.5", "features.6"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng = np.random.RandomState(seed)
    rows = []
    for class_idx in range(n_classes):
        scores = _synthetic_scores(n_samples, layers, seed + class_idx * 13)
        after_scores = _synthetic_scores(n_samples, layers, seed + class_idx * 13 + 500)
        for s in range(n_samples):
            row = {
                "Full Class Index": class_idx,
                "Full filepath": f"class{class_idx}_sample_{s:02d}.jpg",
            }
            for l_idx, layer in enumerate(layers):
                row[f"sensitivityscore_before_{layer}"] = float(scores[s, l_idx])
                row[f"sensitivityscore_After_{layer}"] = float(after_scores[s, l_idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

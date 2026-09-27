"""
Unit + CLI/integration tests for lean_recalibration/bottleneck_detection.py.

bottleneck_detection.py is fully standalone (no model/CAV/training dependency - it only
consumes a sensitivity Excel report or legacy CSV), so every test here uses purely
synthetic, in-memory-generated data via test/helpers/synthetic_reports.py. This is the
fastest and most deterministic of the four target modules to test.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

import bottleneck_detection as bd
from helpers.synthetic_reports import build_sensitivity_workbook, build_legacy_csv
from helpers import constants as C

LAYERS = list(C.PROCESSED_LAYER_NAMES)  # ['features.3', 'features.5', 'features.6']


# ---------------------------------------------------------------------------
# Pure unit tests: small string/sort helpers
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSmallUtilities:
    def test_natural_sort_key_handles_embedded_digits(self):
        items = ["features.10", "features.3", "features.2"]
        ordered = sorted(items, key=bd.natural_sort_key)
        assert ordered == ["features.2", "features.3", "features.10"]

    def test_sanitize_sheet_name_strips_invalid_chars(self):
        used = set()
        name = bd.sanitize_sheet_name("bad:name/with*chars?", used)
        assert set(name) & set("[]:*?/\\") == set()

    def test_sanitize_sheet_name_truncates_to_31_chars(self):
        used = set()
        long_name = "x" * 50
        name = bd.sanitize_sheet_name(long_name, used)
        assert len(name) <= 31

    def test_sanitize_sheet_name_dedupes_case_insensitively(self):
        used = set()
        first = bd.sanitize_sheet_name("Concept", used)
        second = bd.sanitize_sheet_name("concept", used)
        assert first.lower() != second.lower()

    def test_fs_safe_replaces_special_chars(self):
        assert bd._fs_safe("deer coat/legs") == "deer_coat_legs"

    def test_parse_csv_list_none_and_empty(self):
        assert bd._parse_csv_list(None) is None
        assert bd._parse_csv_list("") is None

    def test_parse_csv_list_splits_and_strips(self):
        assert bd._parse_csv_list("a, b ,c") == ["a", "b", "c"]

    def test_stage_tag_label_before(self):
        assert bd.stage_tag_label("before") == ("before", "Before")

    def test_stage_tag_label_legacy(self):
        assert bd.stage_tag_label("legacy") == ("legacy", "Legacy")

    def test_stage_tag_label_after_no_lambda(self):
        assert bd.stage_tag_label("after", None) == ("after", "After")

    def test_stage_tag_label_after_with_lambda(self):
        tag, label = bd.stage_tag_label("after", 0.5)
        assert tag == "afterL50"
        assert "0.5" in label

    def test_strip_part_suffix_removes_pN(self):
        assert bd._strip_part_suffix("deer_coat_p1") == "deer_coat"
        assert bd._strip_part_suffix("deer_coat_p12") == "deer_coat"
        assert bd._strip_part_suffix("deer_coat") == "deer_coat"

    def test_default_output_path_naming(self):
        out = bd.default_output_path(os.path.join("some", "dir", "sensitivity_report_vgg16.xlsx"))
        assert os.path.basename(out) == "bottleneck_report_sensitivity_report_vgg16.xlsx"


# ---------------------------------------------------------------------------
# Pure unit tests: CSPI / alignment / correlation math
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestCspiMath:
    def test_compute_gradient_alignment_self_alignment_is_one(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["L1", "L2", "L3"])
        alignment = bd.compute_gradient_alignment(df)
        diag = np.diag(alignment.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-8)

    def test_compute_cspi_weak_layer_labeled_correctly(self):
        # Layer 0 has ~zero alignment with everything downstream -> "Weak"
        layers = ["L0", "L1", "L2"]
        alignment = pd.DataFrame(
            [[1.0, 0.01, -0.01], [0.01, 1.0, 0.9], [-0.01, 0.9, 1.0]],
            index=layers, columns=layers,
        )
        result = bd.compute_cspi(alignment, tau_abs=0.1, tau_rel=0.1)
        row0 = result[result["Layer"] == "L0"].iloc[0]
        assert row0["Layer_Type"] == "Weak"
        assert row0["Is_Bottleneck"] == 0

    def test_compute_cspi_positive_aligned(self):
        layers = ["L0", "L1"]
        alignment = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=layers, columns=layers)
        result = bd.compute_cspi(alignment, tau_abs=0.1, tau_rel=0.1)
        row0 = result[result["Layer"] == "L0"].iloc[0]
        assert row0["Layer_Type"] == "Positive_Aligned"
        assert row0["Is_Bottleneck"] == 0

    def test_compute_cspi_negative_aligned_is_bottleneck(self):
        layers = ["L0", "L1"]
        alignment = pd.DataFrame([[1.0, -0.8], [-0.8, 1.0]], index=layers, columns=layers)
        result = bd.compute_cspi(alignment, tau_abs=0.1, tau_rel=0.1)
        row0 = result[result["Layer"] == "L0"].iloc[0]
        assert row0["Layer_Type"] == "Negative_Aligned"
        assert row0["Is_Bottleneck"] == 1

    def test_compute_cspi_last_layer_has_no_downstream(self):
        layers = ["L0", "L1"]
        alignment = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=layers, columns=layers)
        result = bd.compute_cspi(alignment)
        last = result[result["Layer"] == "L1"].iloc[0]
        assert last["CSPI_M"] == 0.0
        assert last["CSPI_D"] == 0.0

    def test_compute_adjacent_correlation_length(self):
        corr = np.eye(4)
        R = bd.compute_adjacent_correlation(corr)
        assert len(R) == 3

    def test_compute_delta_length_and_pairs(self):
        R = np.array([0.9, 0.5, 0.4])
        delta, pairs = bd.compute_delta(R, ["A", "B", "C", "D"])
        assert len(delta) == 2
        assert pairs == [("A", "B"), ("B", "C")]
        np.testing.assert_allclose(delta, [0.5 - 0.9, 0.4 - 0.5])

    def test_compute_moving_average_short_series_passthrough(self):
        out = bd.compute_moving_average([0.1, -0.2], ["A->B", "B->C"], window_size=3)
        assert len(out) == 2
        assert list(out["Window_Layers"]) == ["A->B", "B->C"]

    def test_compute_moving_average_empty(self):
        out = bd.compute_moving_average([], [], window_size=3)
        assert len(out) == 0

    def test_detect_bottlenecks_adjacent_flags_sign_flip(self):
        R = np.array([-0.5, 0.8])
        delta = np.array([1.3])
        flags, reasons = bd.detect_bottlenecks_adjacent(R, delta, cspi_m=None)
        assert flags[0] == 1
        assert "sign flip" in reasons[0]

    def test_detect_bottlenecks_adjacent_gate_blocks_flag(self):
        # Weak correlation would normally trigger, but a high cspi_m gate suppresses it.
        R = np.array([0.05])
        delta = np.array([])
        flags, _ = bd.detect_bottlenecks_adjacent(R, delta, cspi_m=[0.9], tau_cspi=0.6)
        assert flags[0] == 0

    def test_compute_top_subset_returns_none_when_not_enough_positive(self):
        corr = pd.DataFrame([[1.0, -0.5], [-0.5, 1.0]], columns=["A", "B"], index=["A", "B"])
        assert bd.compute_top_subset(corr, top_k=5) is None

    def test_compute_top_subset_selects_top_k(self):
        layers = ["A", "B", "C", "D"]
        # last row (D) has descending positive correlation with A > B > C
        data = [
            [1.0, 0.1, 0.2, 0.9],
            [0.1, 1.0, 0.3, 0.5],
            [0.2, 0.3, 1.0, 0.7],
            [0.9, 0.5, 0.7, 1.0],
        ]
        corr = pd.DataFrame(data, columns=layers, index=layers)
        subset = bd.compute_top_subset(corr, top_k=2)
        assert list(subset.columns) == ["D", "A"]

    def test_build_layer_table_end_to_end_shapes(self):
        rng = np.random.RandomState(1)
        wide = pd.DataFrame(rng.normal(loc=1.0, size=(5, 3)), columns=["L0", "L1", "L2"])
        built = bd.build_layer_table(wide, tau_abs=0.1, tau_rel=0.1, tau_delta=0.2,
                                      tau_corr=0.2, tau_cspi=0.6, window_size=3)
        assert set(built.keys()) == {"table", "corr", "alignment", "delta_df"}
        assert len(built["table"]) == 3
        # R (adjacent correlation) has n_layers-1=2 entries; delta_df holds diff(R), which
        # is one entry shorter again -> n_layers-2 == 1 row for 3 layers.
        assert len(built["delta_df"]) == 1


# ---------------------------------------------------------------------------
# Legacy CSV loader
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLegacyCsvLoader:
    def test_load_and_prepare_data_splits_by_class_and_renames_columns(self, tmp_path):
        csv_path = str(tmp_path / "legacy.csv")
        build_legacy_csv(csv_path, n_classes=2, n_samples=5, layers=LAYERS)
        class_groups = bd.load_and_prepare_data(csv_path)
        assert set(class_groups.keys()) == {0, 1}
        for _, df in class_groups.items():
            assert df.shape[0] == 5
            # 'features.' should have been renamed to 'Layer.', and *_After_* columns dropped
            assert all(c.startswith("Layer.") for c in df.columns)
            assert df.shape[1] == len(LAYERS)


# ---------------------------------------------------------------------------
# Workbook discovery/loading helpers (against a synthetic .xlsx)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_workbook(tmp_path):
    path = str(tmp_path / "sensitivity_report_test.xlsx")
    build_sensitivity_workbook(
        path,
        concept_names=["concept_a", "concept_b"],
        layers=LAYERS,
        n_samples=5,
        lambdas=[0.5],
        class_name_for_concept={"concept_a": "cat", "concept_b": "dog"},
        seed=7,
    )
    return path


@pytest.mark.unit
class TestWorkbookLoading:
    def test_load_run_info(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        info = bd.load_run_info(xls)
        assert info["model_name"] == "vgg16"
        assert "features.3" in info["layers_to_process"]

    def test_build_summary_lookup(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        lookup = bd.build_summary_lookup(xls)
        assert "concept_a" in lookup
        assert "tcav_score_before" in lookup["concept_a"]

    def test_resolve_concept_sheet_map_uses_summary(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        mapping = bd.resolve_concept_sheet_map(xls)
        assert mapping["concept_a"] == ["concept_a"]
        assert mapping["concept_b"] == ["concept_b"]

    def test_load_concept_long_df_has_required_columns(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        df = bd.load_concept_long_df(xls, ["concept_a"])
        assert bd.REQUIRED_LONG_COLUMNS.issubset(set(df.columns))
        assert set(df["stage"].unique()) == {"before", "after"}

    def test_resolve_layer_order_prefers_run_info(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        run_info = bd.load_run_info(xls)
        order = bd.resolve_layer_order(run_info, list(reversed(LAYERS)))
        assert order == LAYERS

    def test_lookup_tcav_score_before_and_after(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        lookup = bd.build_summary_lookup(xls)
        before = bd.lookup_tcav_score(lookup, "concept_a", "before", None)
        after = bd.lookup_tcav_score(lookup, "concept_a", "after", 0.5)
        assert before is not None
        assert after is not None

    def test_pivot_stage_matrix_shapes(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        df = bd.load_concept_long_df(xls, ["concept_a"])
        pivoted = bd.pivot_stage_matrix(df, "before", None, LAYERS)
        assert pivoted is not None
        wide, dropped_rows, dropped_layers = pivoted
        assert wide.shape == (5, len(LAYERS))
        assert dropped_rows == 0

    def test_pivot_stage_matrix_none_when_stage_absent(self, synthetic_workbook):
        xls = pd.ExcelFile(synthetic_workbook, engine="openpyxl")
        df = bd.load_concept_long_df(xls, ["concept_a"])
        # request a lambda that doesn't exist for the after stage
        assert bd.pivot_stage_matrix(df, "after", 0.99, LAYERS) is None


# ---------------------------------------------------------------------------
# --min-samples boundary: exactly matches the "5 images" requirement
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestMinSamplesBoundary:
    def _args(self, min_samples):
        class Args:
            pass
        a = Args()
        a.min_samples = min_samples
        a.no_charts = True
        a.top_k = 10
        a.window_size = 3
        return a

    def test_five_samples_passes_default_min_samples(self, tmp_path):
        path = str(tmp_path / "five.xlsx")
        build_sensitivity_workbook(path, ["concept_a"], LAYERS, n_samples=5, seed=3)
        xls = pd.ExcelFile(path, engine="openpyxl")
        run_info = bd.load_run_info(xls)
        summary_lookup = bd.build_summary_lookup(xls)
        long_df = bd.load_concept_long_df(xls, ["concept_a"])
        layer_order = bd.resolve_layer_order(run_info, long_df["layer_name"].dropna().unique().tolist())
        thresholds = {"tau_abs": 0.1, "tau_rel": 0.1, "tau_delta": 0.2, "tau_corr": 0.2, "tau_cspi": 0.6}
        result = bd.process_concept_stage("concept_a", long_df, "before", None, layer_order,
                                           self._args(5), thresholds, str(tmp_path), summary_lookup)
        assert result is not None
        assert result["n_samples"] == 5

    def test_four_samples_skipped_below_default_min_samples(self, tmp_path):
        path = str(tmp_path / "four.xlsx")
        build_sensitivity_workbook(path, ["concept_a"], LAYERS, n_samples=4, seed=3)
        xls = pd.ExcelFile(path, engine="openpyxl")
        run_info = bd.load_run_info(xls)
        summary_lookup = bd.build_summary_lookup(xls)
        long_df = bd.load_concept_long_df(xls, ["concept_a"])
        layer_order = bd.resolve_layer_order(run_info, long_df["layer_name"].dropna().unique().tolist())
        thresholds = {"tau_abs": 0.1, "tau_rel": 0.1, "tau_delta": 0.2, "tau_corr": 0.2, "tau_cspi": 0.6}
        result = bd.process_concept_stage("concept_a", long_df, "before", None, layer_order,
                                           self._args(5), thresholds, str(tmp_path), summary_lookup)
        assert result is None


# ---------------------------------------------------------------------------
# Full CLI / integration tests (subprocess) -- single-concept and multi-concept ("single
# class" vs "multiclass" verification for this standalone script)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestBottleneckDetectionCli:
    def test_single_concept_xlsx_end_to_end(self, run_script, tmp_path):
        input_xlsx = str(tmp_path / "sensitivity_report_single.xlsx")
        build_sensitivity_workbook(
            input_xlsx, ["stripes"], LAYERS, n_samples=5, lambdas=[0.5],
            class_name_for_concept={"stripes": "cat"}, seed=11,
        )
        output_xlsx = str(tmp_path / "bottleneck_single.xlsx")
        result = run_script("bottleneck_detection.py", [input_xlsx, output_xlsx, "--no-charts"])
        result.assert_success()
        assert os.path.isfile(output_xlsx)

        xls = pd.ExcelFile(output_xlsx, engine="openpyxl")
        assert "Overview" in xls.sheet_names
        assert any(sn.startswith("stripes_") for sn in xls.sheet_names)

    def test_multiclass_concepts_with_charts_end_to_end(self, run_script, tmp_path):
        input_xlsx = str(tmp_path / "sensitivity_report_multi.xlsx")
        concepts = ["cat_spots", "cat_ears", "dog_collar", "bird_feathers"]
        class_map = {"cat_spots": "cat", "cat_ears": "cat", "dog_collar": "dog", "bird_feathers": "bird"}
        build_sensitivity_workbook(
            input_xlsx, concepts, LAYERS, n_samples=5, lambdas=[0.5],
            class_name_for_concept=class_map, seed=23,
        )
        output_xlsx = str(tmp_path / "bottleneck_multi.xlsx")
        result = run_script("bottleneck_detection.py", [input_xlsx, output_xlsx])
        result.assert_success()
        assert os.path.isfile(output_xlsx)
        charts_dir = tmp_path / "charts"
        assert charts_dir.is_dir()
        assert any(charts_dir.rglob("*.png"))

        xls = pd.ExcelFile(output_xlsx, engine="openpyxl")
        for concept in concepts:
            assert any(sn.startswith(concept[:20]) for sn in xls.sheet_names)

    def test_legacy_csv_end_to_end(self, run_script, tmp_path):
        input_csv = str(tmp_path / "legacy_report.csv")
        build_legacy_csv(input_csv, n_classes=2, n_samples=5, layers=LAYERS)
        output_xlsx = str(tmp_path / "bottleneck_legacy.xlsx")
        result = run_script("bottleneck_detection.py", [input_csv, output_xlsx, "--no-charts"])
        result.assert_success()
        assert os.path.isfile(output_xlsx)

    def test_missing_input_file_errors_cleanly(self, run_script, tmp_path):
        missing = str(tmp_path / "does_not_exist.xlsx")
        result = run_script("bottleneck_detection.py", [missing])
        assert result.returncode != 0

    def test_concepts_filter_option(self, run_script, tmp_path):
        input_xlsx = str(tmp_path / "sensitivity_report_filter.xlsx")
        concepts = ["cat_spots", "dog_collar"]
        build_sensitivity_workbook(input_xlsx, concepts, LAYERS, n_samples=5, seed=5)
        output_xlsx = str(tmp_path / "bottleneck_filter.xlsx")
        result = run_script("bottleneck_detection.py", [
            input_xlsx, output_xlsx, "--concepts", "cat_spots", "--no-charts",
        ])
        result.assert_success()
        xls = pd.ExcelFile(output_xlsx, engine="openpyxl")
        assert any(sn.startswith("cat_spots") for sn in xls.sheet_names)
        assert not any(sn.startswith("dog_collar") for sn in xls.sheet_names)

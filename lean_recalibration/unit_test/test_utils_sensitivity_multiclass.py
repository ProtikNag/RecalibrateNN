"""
Unit + CLI/integration tests for lean_recalibration/utils_sensitivity_multiclass.py.

``main()`` parses ``sys.argv`` directly (no reusable programmatic entry point), so full
pipeline behaviour is exercised via subprocess (the ``run_script`` fixture), reusing the CAV
stores + per-model manifests pre-populated ONCE per session by ``main_store_cav.py`` (see
``populated_single_cav_store`` / ``populated_multi_cav_store`` in conftest.py). Pure helper
functions (``_safe_name``, ``_safe_layer_name``, ``get_model_path``, ``get_layernames_override``,
``get_singleclass_filelist``, ``build_concept_specs``, ``load_manifest``,
``get_cav_from_manifest``, ``_sanitize_sheet_name``, ``compute_concept_summary``,
``compute_layer_summary``, ``flush_raw_csv``) are unit-tested in-process for speed and
precision.

NOTE on ``--before_after``: it loads a *recalibrated* checkpoint via
``get_model_path()``/``load_model_statedict()``, which expects a file named
``loss_<model>_<layer>_<lambda>.pth`` under ``--recal_model_basepath/<model>/`` -- a naming
convention that predates (and does not match) the checkpoint names actually written by the
current ``main_recalib_custom_by_loading_cav.py`` (``model_cls*_combo*_lambda*.pth`` /
``model_multiclass_combo*_lambda*.pth``). This is a pre-existing mismatch between the two
scripts, not something introduced by this test suite, so ``--before_after`` is exercised here
by manually creating a checkpoint at the path ``get_model_path()`` itself expects (this is
exactly ``utils_sensitivity_multiclass.py``'s own documented contract), rather than by piping
in real output from the recalibration script.

NOTE on single-class mode's end-to-end CLI test: testing against a REAL manifest produced by
``main_store_cav.py`` uncovered a second, separate pre-existing mismatch: in single
("non-multiconcept") mode, ``main_store_cav.py`` stores each concept's CAV in the manifest
under the BARE concept-folder name (e.g. ``"stripes"``), while
``utils_sensitivity_multiclass.py``'s ``build_concept_specs()`` always prefixes the concept
name with the owning class (e.g. ``"cat_stripes"``) in single mode. As a result, single-class
sensitivity runs currently complete successfully but silently collect zero rows (no CSV/Excel
is written). This was confirmed against the real pipeline and reported; per explicit maintainer
direction, this test suite does NOT modify production code to fix it -- see
``test_single_class_end_to_end_documents_concept_naming_mismatch`` below, which pins/documents
this exact current behaviour instead of asserting real output. Multiclass mode is unaffected
(both scripts already agree on class-prefixed naming there), so ``--before_after`` and the
"real output produced" assertions are exercised via multiclass mode instead.
"""

import json
import os

import joblib
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import utils_sensitivity_multiclass as usm
from custom_dataloader import SingleClassDataLoader

from helpers import constants as C


class _FakeConfig:
    """Minimal stand-in for ConfigSingleton, just enough for get_layernames_override()."""
    def __init__(self, **kwargs):
        self.VGG_RECALIB = kwargs.get("VGG_RECALIB", ["features.3"])
        self.RESNET50_RECALIB = kwargs.get("RESNET50_RECALIB", ["layer1"])
        self.INCEPTION_V3_RECALIB = kwargs.get("INCEPTION_V3_RECALIB", ["Mixed_5b"])
        self.MOBILENET_V3_SMALL_RECALIB = kwargs.get("MOBILENET_V3_SMALL_RECALIB", ["features.1"])
        self.MOBILENET_V3_LARGE_RECALIB = kwargs.get("MOBILENET_V3_LARGE_RECALIB", ["features.1"])
        self.MULTICONCEPT_ENABLED = kwargs.get("MULTICONCEPT_ENABLED", False)
        self.MULTICONCEPT_CLASS_CONCEPTS = kwargs.get("MULTICONCEPT_CLASS_CONCEPTS", {})
        self.MULTICONCEPT_BASE_PATH = kwargs.get("MULTICONCEPT_BASE_PATH", "")

    def get_multiconcept_random_folder_path(self, class_idx, position_idx):
        return self._random_paths.get((class_idx, position_idx), "") if hasattr(self, "_random_paths") else ""


# ---------------------------------------------------------------------------
# _safe_name / _safe_layer_name
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSafeName:
    def test_replaces_spaces_and_slashes(self):
        assert usm._safe_name("cat spots/v2\\x") == "cat_spots_v2_x"

    def test_strips_surrounding_whitespace(self):
        assert usm._safe_name("  cat  ") == "cat"


@pytest.mark.unit
class TestSafeLayerName:
    def test_replaces_path_separators(self):
        assert usm._safe_layer_name("features/3") == "features_3"
        assert usm._safe_layer_name("features.3") == "features.3"  # dots untouched


# ---------------------------------------------------------------------------
# get_model_path
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetModelPath:
    def test_returns_path_when_checkpoint_exists(self, tmp_path):
        model_dir = tmp_path / "vgg16"
        model_dir.mkdir()
        ckpt = model_dir / "loss_vgg16_features.3_0.5.pth"
        ckpt.write_bytes(b"not a real checkpoint")
        result = usm.get_model_path("vgg16", "features.3", 0.5, str(tmp_path))
        assert result == str(ckpt)

    def test_raises_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            usm.get_model_path("vgg16", "features.3", 0.5, str(tmp_path))


# ---------------------------------------------------------------------------
# get_layernames_override
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetLayernamesOverride:
    def test_returns_correct_layers_per_model(self):
        config = _FakeConfig(VGG_RECALIB=["features.6"], RESNET50_RECALIB=["layer2"])
        assert usm.get_layernames_override("vgg16", config) == ["features.6"]
        assert usm.get_layernames_override("resnet50", config) == ["layer2"]

    def test_unknown_model_returns_none(self):
        config = _FakeConfig()
        assert usm.get_layernames_override("not_a_model", config) is None


# ---------------------------------------------------------------------------
# get_singleclass_filelist
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetSingleClassFilelist:
    def test_returns_absolute_paths_in_dataset_order(self):
        folder = os.path.join(C.SINGLE_CONCEPT_DIR, C.SINGLE_CONCEPT_NAME)
        loader = DataLoader(SingleClassDataLoader(folder), batch_size=2, shuffle=False)
        file_list = usm.get_singleclass_filelist(loader)
        assert len(file_list) == C.IMAGES_PER_FOLDER
        assert all(os.path.isabs(p) for p in file_list)
        assert file_list == [os.path.abspath(os.path.join(folder, f)) for f in loader.dataset.image_files]


# ---------------------------------------------------------------------------
# build_concept_specs
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestBuildConceptSpecs:
    def test_single_mode_builds_one_spec_per_zipped_class(self):
        specs = usm.build_concept_specs(
            config=_FakeConfig(),
            concept_mode="single",
            target_class_list=["cat", "dog"],
            target_idx_list=[0, 1],
            class_dataloaders=["loader_cat", "loader_dog"],
            concept_folder_list=["/data/concepts/stripes"],  # shorter than target_class_list on purpose
            random_folder="/data/concepts/random",
        )
        # zip() truncates to the shortest iterable -> only "cat" gets a spec
        assert len(specs) == 1
        assert specs[0]["concept_name"] == "cat_stripes"
        assert specs[0]["class_name"] == "cat"
        assert specs[0]["target_class_index"] == 0
        assert specs[0]["class_dataloader"] == "loader_cat"

    def test_multiclass_mode_builds_one_spec_per_concept_folder(self):
        config = _FakeConfig(
            MULTICONCEPT_ENABLED=True,
            MULTICONCEPT_CLASS_CONCEPTS={0: ["cat_spots", "cat_ears"], 1: ["dog_collar"]},
            MULTICONCEPT_BASE_PATH="/data/multiconcept",
        )
        config._random_paths = {}
        specs = usm.build_concept_specs(
            config=config,
            concept_mode="multiclass",
            target_class_list=["cat", "dog"],
            target_idx_list=[0, 1],
            class_dataloaders=["loader_cat", "loader_dog"],
            concept_folder_list=[],
            random_folder="/data/random",
        )
        names = sorted(s["concept_name"] for s in specs)
        assert names == ["cat_cat_ears", "cat_cat_spots", "dog_dog_collar"]

    def test_multiclass_mode_without_multiconcept_enabled_raises(self):
        with pytest.raises(ValueError, match="multiclass"):
            usm.build_concept_specs(
                config=_FakeConfig(MULTICONCEPT_ENABLED=False),
                concept_mode="multiclass",
                target_class_list=["cat"],
                target_idx_list=[0],
                class_dataloaders=["loader_cat"],
                concept_folder_list=[],
                random_folder="/data/random",
            )

    def test_unknown_concept_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown concept_mode"):
            usm.build_concept_specs(
                config=_FakeConfig(),
                concept_mode="bogus",
                target_class_list=["cat"],
                target_idx_list=[0],
                class_dataloaders=["loader_cat"],
                concept_folder_list=["/data/concepts/stripes"],
                random_folder="/data/random",
            )

    def test_no_resolved_concepts_raises_value_error(self):
        with pytest.raises(ValueError, match="No concepts were resolved"):
            usm.build_concept_specs(
                config=_FakeConfig(),
                concept_mode="single",
                target_class_list=[],
                target_idx_list=[],
                class_dataloaders=[],
                concept_folder_list=[],
                random_folder="/data/random",
            )

    def test_duplicate_concept_names_are_deduplicated_with_suffix(self):
        # Two target folders under the SAME class sharing a basename -> genuine collision
        # ("cat_spots" would be produced twice) -> the second occurrence must be renamed.
        config = _FakeConfig(
            MULTICONCEPT_ENABLED=True,
            MULTICONCEPT_CLASS_CONCEPTS={0: ["spots", "spots"]},
            MULTICONCEPT_BASE_PATH="/data/multiconcept",
        )
        config._random_paths = {}
        specs = usm.build_concept_specs(
            config=config,
            concept_mode="multiclass",
            target_class_list=["cat", "dog"],
            target_idx_list=[0, 1],
            class_dataloaders=["loader_cat", "loader_dog"],
            concept_folder_list=[],
            random_folder="/data/random",
        )
        names = sorted(s["concept_name"] for s in specs)
        assert names == ["cat_spots", "cat_spots_2"]


# ---------------------------------------------------------------------------
# load_manifest / get_cav_from_manifest
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLoadManifestAndGetCav:
    def _write_manifest_and_cav(self, tmp_path, layers=("features.3",)):
        concept_dir = tmp_path / "vgg16" / "concept_a"
        concept_dir.mkdir(parents=True)
        cav_file = concept_dir / "features.3.joblib"
        joblib.dump({"cav_vector": torch.tensor([1.0, 0.0, 0.0])}, str(cav_file))
        manifest = {
            "concepts": {
                "concept_a": {"class_name": "cat", "layers": list(layers), "data_path": str(concept_dir)},
            }
        }
        manifest_path = tmp_path / "vgg16_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return str(manifest_path)

    def test_load_manifest_parses_json(self, tmp_path):
        manifest_path = self._write_manifest_and_cav(tmp_path)
        data = usm.load_manifest(manifest_path)
        assert "concept_a" in data["concepts"]

    def test_get_cav_from_manifest_success(self, tmp_path):
        manifest_path = self._write_manifest_and_cav(tmp_path)
        data = usm.load_manifest(manifest_path)
        cav, source_path, reason = usm.get_cav_from_manifest(data, "concept_a", "features.3", "cpu")
        assert reason is None
        assert isinstance(cav, torch.Tensor)
        assert torch.allclose(cav, torch.tensor([1.0, 0.0, 0.0]))
        assert os.path.isfile(source_path)

    def test_get_cav_from_manifest_unknown_concept(self, tmp_path):
        manifest_path = self._write_manifest_and_cav(tmp_path)
        data = usm.load_manifest(manifest_path)
        cav, source_path, reason = usm.get_cav_from_manifest(data, "no_such_concept", "features.3", "cpu")
        assert cav is None and source_path is None
        assert "not found in manifest" in reason

    def test_get_cav_from_manifest_layer_not_listed(self, tmp_path):
        manifest_path = self._write_manifest_and_cav(tmp_path, layers=("features.5",))
        data = usm.load_manifest(manifest_path)
        cav, source_path, reason = usm.get_cav_from_manifest(data, "concept_a", "features.3", "cpu")
        assert cav is None
        assert "not listed" in reason

    def test_get_cav_from_manifest_file_missing_on_disk(self, tmp_path):
        manifest_path = self._write_manifest_and_cav(tmp_path)
        data = usm.load_manifest(manifest_path)
        # Ask for a layer that IS listed but whose joblib file was never written.
        data["concepts"]["concept_a"]["layers"].append("features.6")
        cav, source_path, reason = usm.get_cav_from_manifest(data, "concept_a", "features.6", "cpu")
        assert cav is None
        assert "not found on disk" in reason


# ---------------------------------------------------------------------------
# _sanitize_sheet_name
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSanitizeSheetName:
    def test_strips_invalid_excel_characters(self):
        used = set()
        assert usm._sanitize_sheet_name("a[b]:c*d?e/f\\g", used) == "abcdefg"

    def test_empty_after_cleaning_falls_back_to_sheet(self):
        used = set()
        assert usm._sanitize_sheet_name("[]:*?/\\", used) == "Sheet"

    def test_truncates_to_31_chars(self):
        used = set()
        long_name = "x" * 50
        result = usm._sanitize_sheet_name(long_name, used)
        assert len(result) <= 31

    def test_duplicate_names_get_numeric_suffix(self):
        used = set()
        first = usm._sanitize_sheet_name("concept", used)
        second = usm._sanitize_sheet_name("concept", used)
        assert first == "concept"
        assert second != first
        assert second.startswith("concept")


# ---------------------------------------------------------------------------
# compute_concept_summary / compute_layer_summary
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestComputeSummaries:
    def _sample_records_df(self):
        return pd.DataFrame({
            "concept_name": ["c1", "c1", "c1", "c1"],
            "model_name": ["vgg16"] * 4,
            "class_name": ["cat"] * 4,
            "target_class_index": [0] * 4,
            "concept_mode": ["single"] * 4,
            "concept_folder_path": ["/x"] * 4,
            "random_folder_path": ["/y"] * 4,
            "cav_source_path": ["/z.joblib"] * 4,
            "layer_name": ["features.3", "features.3", "features.5", "features.5"],
            "stage": ["before", "after", "before", "after"],
            "lambda_align": [float("nan"), 0.5, float("nan"), 0.5],
            "sensitivity_score": [0.1, -0.2, 0.3, 0.4],
            "model_weight_path": ["/m.pth"] * 4,
        })

    def test_concept_summary_computes_tcav_fractions(self):
        df = self._sample_records_df()
        summary = usm.compute_concept_summary(df)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["concept_name"] == "c1"
        assert row["total_rows"] == 4
        assert row["positive_rows"] == 3  # 0.1, 0.3, 0.4 are positive; -0.2 is not
        assert row["tcav_score"] == pytest.approx(0.75)
        assert row["tcav_score_before"] == pytest.approx(1.0)  # both before rows positive (0.1, 0.3)
        assert row["tcav_score_after_overall"] == pytest.approx(0.5)  # -0.2 negative, 0.5 positive

    def test_layer_summary_has_one_row_per_layer_stage_lambda(self):
        df = self._sample_records_df()
        layer_summary = usm.compute_layer_summary(df)
        # 2 layers x 2 stages (before has no lambda split, after has lambda=0.5) = 4 groups
        assert len(layer_summary) == 4
        assert set(layer_summary["layer_name"]) == {"features.3", "features.5"}


# ---------------------------------------------------------------------------
# flush_raw_csv
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestFlushRawCsv:
    def test_writes_header_on_first_flush_and_appends_on_second(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        records = {"a": [1, 2], "b": ["x", "y"]}
        flushed = usm.flush_raw_csv(records, csv_path, 0)
        assert flushed == 2
        assert os.path.isfile(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 2

        records["a"].append(3)
        records["b"].append("z")
        flushed = usm.flush_raw_csv(records, csv_path, flushed)
        assert flushed == 3
        df2 = pd.read_csv(csv_path)
        assert len(df2) == 3  # appended, not rewritten from scratch

    def test_no_new_rows_is_a_no_op(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        records = {"a": [1]}
        flushed = usm.flush_raw_csv(records, csv_path, 0)
        flushed_again = usm.flush_raw_csv(records, csv_path, flushed)
        assert flushed_again == flushed

    def test_empty_records_returns_flushed_count_unchanged(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        assert usm.flush_raw_csv({}, csv_path, 5) == 5
        assert not os.path.isfile(csv_path)


# ---------------------------------------------------------------------------
# CLI / integration tests (real subprocess invocations against pre-built CAV stores)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
class TestUtilsSensitivityCli:
    def _model_path(self, model_root):
        return os.path.join(model_root, C.MODEL_NAME, f"{C.MODEL_NAME}.pth")

    def test_missing_config_errors_cleanly(self, run_script):
        result = run_script(
            "utils_sensitivity_multiclass.py",
            ["--org_model_path", self._model_path(C.SINGLE_MODEL_ROOT), "--model_name", C.MODEL_NAME,
             "--store_results", "results", "--manifest", "manifest.json"],
        )
        assert result.returncode != 0
        assert "Config file parameter not provided" in result.stderr

    def test_missing_manifest_errors_cleanly(self, run_script, tmp_path):
        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.SINGLE_CONFIG_PATH,
                "--org_model_path", self._model_path(C.SINGLE_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--store_results", str(tmp_path / "results"),
            ],
        )
        assert result.returncode != 0
        assert "--manifest file" in result.stderr

    def test_missing_store_results_errors_cleanly(self, run_script, populated_single_cav_store):
        manifest_path = os.path.join(populated_single_cav_store, f"{C.MODEL_NAME}_manifest.json")
        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.SINGLE_CONFIG_PATH,
                "--org_model_path", self._model_path(C.SINGLE_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--manifest", manifest_path,
            ],
        )
        assert result.returncode != 0
        assert "--store_results directory" in result.stderr

    def test_single_class_end_to_end_documents_concept_naming_mismatch(self, run_script, tmp_path, populated_single_cav_store):
        """Single-class mode currently produces an EMPTY (but non-crashing) report.

        KNOWN, CONFIRMED PRE-EXISTING BEHAVIOUR (not a bug in this test suite): in single
        ("non-multiconcept") mode, main_store_cav.py stores each concept's CAV in the manifest
        under the BARE concept-folder name (e.g. "stripes" -- see its `concept_names =
        list(CONCEPT_NAMES)` branch, which never incorporates the class name). But
        utils_sensitivity_multiclass.py's build_concept_specs() ALWAYS prefixes the concept
        name with the owning class (e.g. "cat_stripes") in single mode. Because of this, the
        manifest lookup in get_cav_from_manifest() never finds a match for any layer/concept
        in single-class mode, and the run completes successfully but collects zero rows (no
        CSV/Excel is written at all -- see the "No rows were collected" warning below).

        This was confirmed against the real pipeline (main_store_cav.py's real output manifest)
        and reported to the maintainer, who asked that this test pin/document the current
        behaviour rather than have this test suite alter production code. Multiclass mode is
        NOT affected -- see test_multiclass_end_to_end below, which does assert real rows are
        produced, because both scripts already agree on class-prefixed naming there.
        """
        manifest_path = os.path.join(populated_single_cav_store, f"{C.MODEL_NAME}_manifest.json")
        results_dir = str(tmp_path / "results")
        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.SINGLE_CONFIG_PATH,
                "--org_model_path", self._model_path(C.SINGLE_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--store_results", results_dir,
                "--manifest", manifest_path,
                "--concept_mode", "single",
            ],
            timeout=600,
        )
        result.assert_success()  # the script itself does not crash / does not error out
        assert "Concept mode resolved to: single" in result.stdout
        assert "Resolved 1 concept(s) in 'single' mode: ['cat_stripes']" in result.stdout
        assert "not found in manifest" in result.stdout
        assert "No rows were collected" in result.stdout

        model_dir = os.path.join(results_dir, C.MODEL_NAME)
        assert not os.path.isdir(model_dir) or not [
            f for f in os.listdir(model_dir) if f.endswith(".csv") or f.endswith(".xlsx")
        ], "Expected NO CSV/Excel output for single-class mode given the documented naming mismatch"

    def test_multiclass_end_to_end(self, run_script, tmp_path, populated_multi_cav_store):
        manifest_path = os.path.join(populated_multi_cav_store, f"{C.MODEL_NAME}_manifest.json")
        results_dir = str(tmp_path / "results")
        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.MULTI_CONFIG_PATH,
                "--org_model_path", self._model_path(C.MULTI_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--store_results", results_dir,
                "--manifest", manifest_path,
                "--concept_mode", "multiclass",
            ],
            timeout=600,
        )
        result.assert_success()
        assert "Concept mode resolved to: multiclass" in result.stdout

        model_dir = os.path.join(results_dir, C.MODEL_NAME)
        csv_files = [f for f in os.listdir(model_dir) if f.startswith("sensitivity_audit_trail_") and f.endswith(".csv")]
        xlsx_files = [f for f in os.listdir(model_dir) if f.startswith("sensitivity_report_") and f.endswith(".xlsx")]
        assert csv_files, f"No sensitivity CSV backup found in {model_dir}"
        assert xlsx_files, f"No sensitivity Excel report found in {model_dir}"

        import openpyxl
        wb = openpyxl.load_workbook(os.path.join(model_dir, xlsx_files[0]), read_only=True)
        assert "Run_Info" in wb.sheetnames
        assert "Summary" in wb.sheetnames
        assert "Summary_By_Layer" in wb.sheetnames
        # One extra sheet per multiclass concept, beyond the 3 fixed sheets.
        assert len(wb.sheetnames) >= 3 + len(C.MULTI_CLASS_CONCEPTS)

        df = pd.read_csv(os.path.join(model_dir, csv_files[0]))
        assert (df["stage"] == "before").all()  # --before_after was not passed
        assert set(df["layer_name"]) == set(C.PROCESSED_LAYER_NAMES)
        expected_concepts = set()
        for class_idx, folders in C.MULTI_CLASS_CONCEPTS.items():
            class_name = C.MULTI_CLASSES[class_idx]
            for folder in folders:
                expected_concepts.add(f"{class_name}_{folder}")
        assert set(df["concept_name"]) == expected_concepts

    def test_auto_concept_mode_resolves_to_multiclass_when_enabled(self, run_script, tmp_path, populated_multi_cav_store):
        manifest_path = os.path.join(populated_multi_cav_store, f"{C.MODEL_NAME}_manifest.json")
        results_dir = str(tmp_path / "results")
        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.MULTI_CONFIG_PATH,
                "--org_model_path", self._model_path(C.MULTI_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--store_results", results_dir,
                "--manifest", manifest_path,
                # --concept_mode omitted -> defaults to "auto"
            ],
            timeout=600,
        )
        result.assert_success()
        assert "Concept mode resolved to: multiclass" in result.stdout

    def test_before_after_loads_recalibrated_checkpoint(self, run_script, tmp_path, populated_multi_cav_store):
        """Exercise the --before_after path by manually creating a checkpoint at exactly the
        path get_model_path() documents/expects (see module docstring for why this is not
        piped in from main_recalib_custom_by_loading_cav.py's real output).

        Uses multiclass mode (not single) because that is where concept-name lookups against
        a real main_store_cav.py manifest actually succeed (see the single-class test above,
        which documents a separate, unrelated naming mismatch specific to single-class mode)."""
        manifest_path = os.path.join(populated_multi_cav_store, f"{C.MODEL_NAME}_manifest.json")
        results_dir = str(tmp_path / "results")
        recal_base = str(tmp_path / "recal_models")
        recal_model_dir = os.path.join(recal_base, C.MODEL_NAME)
        os.makedirs(recal_model_dir, exist_ok=True)

        base_model = torch.load(self._model_path(C.MULTI_MODEL_ROOT), map_location="cpu", weights_only=False)
        state_dict = base_model.state_dict()
        for layer in C.PROCESSED_LAYER_NAMES:
            for lam in C.LAMBDA_ALIGNS:
                ckpt_path = os.path.join(recal_model_dir, f"loss_{C.MODEL_NAME}_{layer}_{lam}.pth")
                torch.save(state_dict, ckpt_path)

        result = run_script(
            "utils_sensitivity_multiclass.py",
            [
                "--config", C.MULTI_CONFIG_PATH,
                "--org_model_path", self._model_path(C.MULTI_MODEL_ROOT),
                "--model_name", C.MODEL_NAME,
                "--store_results", results_dir,
                "--manifest", manifest_path,
                "--concept_mode", "multiclass",
                "--before_after",
                "--recal_model_basepath", recal_base,
            ],
            timeout=600,
        )
        result.assert_success()

        model_dir = os.path.join(results_dir, C.MODEL_NAME)
        csv_files = [f for f in os.listdir(model_dir) if f.startswith("sensitivity_audit_trail_") and f.endswith(".csv")]
        assert csv_files, f"No sensitivity CSV backup found in {model_dir}"
        df = pd.read_csv(os.path.join(model_dir, csv_files[0]))
        assert set(df["stage"]) == {"before", "after"}
        assert set(df.loc[df["stage"] == "after", "lambda_align"].unique()) == set(C.LAMBDA_ALIGNS)

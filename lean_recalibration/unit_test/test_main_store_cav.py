"""
Unit + CLI/integration tests for lean_recalibration/main_store_cav.py.

main_store_cav.py has no reusable ``main()`` -- its orchestration lives directly under
``if __name__ == "__main__":``, so full-pipeline behavior can only be exercised faithfully
via subprocess (the ``run_script`` fixture from conftest.py). Pure helper functions
(``_concept_aliases``, ``get_concepts_to_update``, ``compute_cav``, ``load_and_verify``,
``print_model_manifest``) are still unit-tested in-process for speed and precision.
"""

import json
import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import main_store_cav as msc
from cav_registry import CAVRegistry
from custom_dataloader import SingleClassDataLoader

from helpers import constants as C


# ---------------------------------------------------------------------------
# _concept_aliases
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestConceptAliases:
    def test_falls_back_to_concept_name_when_no_manifest_entry(self, tmp_path):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        aliases = msc._concept_aliases(["concept_a", "concept_b"], registry)
        assert aliases == {"concept_a": "concept_a", "concept_b": "concept_b"}

    def test_uses_manifest_alias_when_present(self, tmp_path):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        registry.add_concept(name="concept_a", alias="Concept A", data_path="/x", save=False)
        aliases = msc._concept_aliases(["concept_a", "concept_b"], registry)
        assert aliases["concept_a"] == "Concept A"
        assert aliases["concept_b"] == "concept_b"


# ---------------------------------------------------------------------------
# get_concepts_to_update
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGetConceptsToUpdate:
    def _registry_with_concept_a_on_features3(self, tmp_path):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        registry.save_layer_cav(
            model="vgg16",
            layer="features.3",
            concept_cavs={"concept_a": torch.zeros(8)},
            concept_aliases={"concept_a": "concept_a"},
        )
        return registry

    def test_rebuild_true_returns_everything_regardless_of_store(self, tmp_path):
        registry = self._registry_with_concept_a_on_features3(tmp_path)
        result = msc.get_concepts_to_update(
            registry, "vgg16", ["concept_a", "concept_b"], ["features.3"], rebuild_cav=True,
        )
        assert result == ["concept_a", "concept_b"]

    def test_empty_registry_returns_all_concepts(self, tmp_path):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        result = msc.get_concepts_to_update(
            registry, "vgg16", ["concept_a", "concept_b"], ["features.3", "features.5"], rebuild_cav=False,
        )
        assert result == ["concept_a", "concept_b"]

    def test_concept_with_all_requested_layers_stored_is_skipped(self, tmp_path):
        registry = self._registry_with_concept_a_on_features3(tmp_path)
        result = msc.get_concepts_to_update(
            registry, "vgg16", ["concept_a", "concept_b"], ["features.3"], rebuild_cav=False,
        )
        assert result == ["concept_b"]

    def test_concept_missing_a_newly_requested_layer_is_included(self, tmp_path):
        registry = self._registry_with_concept_a_on_features3(tmp_path)
        # concept_a only has features.3 stored; features.5 is now also requested, so it
        # must be re-included even though it is not a brand new concept.
        result = msc.get_concepts_to_update(
            registry, "vgg16", ["concept_a", "concept_b"], ["features.3", "features.5"], rebuild_cav=False,
        )
        assert result == ["concept_a", "concept_b"]


# ---------------------------------------------------------------------------
# compute_cav (real forward passes through the tiny model + real linear-classifier CAV fit)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestComputeCav:
    def test_returns_flat_float_tensor_of_expected_length(self):
        # LINEAR_CLASSIFIER_TYPE is normally set as a module global inside main_store_cav's
        # `if __name__ == "__main__":` block; set it explicitly for in-process unit testing.
        msc.LINEAR_CLASSIFIER_TYPE = C.LINEAR_CLASSIFIER_TYPE

        model = msc.copy.deepcopy(
            torch.load(
                os.path.join(C.SINGLE_MODEL_ROOT, C.MODEL_NAME, f"{C.MODEL_NAME}.pth"),
                map_location="cpu",
                weights_only=False,
            )
        )
        model.eval()
        layer_name = C.PROCESSED_LAYER_NAMES[0]  # "features.3"
        model.get_submodule(layer_name).register_forward_hook(msc.get_activation(layer_name))

        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        concept_loader = DataLoader(
            SingleClassDataLoader(os.path.join(C.SINGLE_CONCEPT_DIR, C.SINGLE_CONCEPT_NAME), transform=transform),
            batch_size=2, shuffle=False,
        )
        random_loader = DataLoader(
            SingleClassDataLoader(C.SINGLE_RANDOM_FOLDER, transform=transform),
            batch_size=2, shuffle=False,
        )

        cav = msc.compute_cav(model, concept_loader, random_loader, layer_name)

        assert isinstance(cav, torch.Tensor)
        assert cav.ndim == 1
        # features.3 sits right after the FIRST MaxPool2d (features.2): 32x32 input ->
        # 16x16 spatial, 16 channels -> flattened length 16*16*16 = 4096.
        assert cav.shape[0] == 16 * 16 * 16
        assert torch.isfinite(cav).all()


# ---------------------------------------------------------------------------
# load_and_verify
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestLoadAndVerify:
    class _CollectingLogger:
        def __init__(self):
            self.lines = []

        def info(self, msg):
            self.lines.append(("info", msg))

        def warning(self, msg):
            self.lines.append(("warning", msg))

        def error(self, msg):
            self.lines.append(("error", msg))

    def test_logs_shape_summary_for_stored_layer(self, tmp_path, capsys):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        registry.save_layer_cav(
            model="vgg16", layer="features.3",
            concept_cavs={"concept_a": torch.zeros(8)},
            concept_aliases={"concept_a": "concept_a"},
        )
        logger = self._CollectingLogger()
        msc.load_and_verify(registry, "vgg16", ["features.3"], logger)

        captured = capsys.readouterr()
        assert "Verified layer=features.3" in captured.out
        assert any("layer=features.3" in msg for level, msg in logger.lines if level == "info")

    def test_warns_when_model_has_no_stored_cavs_at_all(self, tmp_path, capsys):
        registry = CAVRegistry.create_empty_manifest(str(tmp_path / "cav_store"))
        logger = self._CollectingLogger()
        msc.load_and_verify(registry, "vgg16", ["features.3"], logger)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert any(level == "warning" for level, _ in logger.lines)


# ---------------------------------------------------------------------------
# print_model_manifest
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPrintModelManifest:
    def test_prints_concept_summary_when_manifest_exists(self, tmp_path, capsys):
        cav_store_root = str(tmp_path / "cav_store")
        registry = CAVRegistry.create_empty_manifest(cav_store_root)
        registry.save_layer_cav(
            model="vgg16", layer="features.3",
            concept_cavs={"concept_a": torch.zeros(8)},
            concept_aliases={"concept_a": "concept_a"},
            model_weight_path="C:/models/vgg16/vgg16.pth",
        )
        msc.print_model_manifest(cav_store_root, "vgg16")

        captured = capsys.readouterr()
        assert "Total concepts   : 1" in captured.out
        assert "concept_a" in captured.out

    def test_warns_when_manifest_missing(self, tmp_path, capsys):
        msc.print_model_manifest(str(tmp_path / "no_such_cav_store"), "vgg16")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "No manifest found" in captured.out


# ---------------------------------------------------------------------------
# CLI / integration tests (real subprocess invocations)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
class TestMainStoreCavCli:
    def test_info_flag_prints_usage_and_exits_zero(self, run_script):
        result = run_script("main_store_cav.py", ["--info"])
        result.assert_success()
        assert "Example usage" in result.stdout

    def test_missing_config_file_arg_errors_cleanly(self, run_script):
        result = run_script("main_store_cav.py", ["--model_name", "vgg16"])
        assert result.returncode != 0
        assert "--config_file is required" in result.stderr

    def test_nonexistent_config_file_path_errors_cleanly(self, run_script, tmp_path):
        bogus = str(tmp_path / "does_not_exist.yaml")
        result = run_script("main_store_cav.py", ["--config_file", bogus])
        assert result.returncode != 0
        assert "Config file not found" in result.stderr

    def test_unknown_model_name_errors_cleanly(self, run_script):
        result = run_script(
            "main_store_cav.py",
            ["--config_file", C.SINGLE_CONFIG_PATH, "--model_name", "not_a_real_model"],
        )
        assert result.returncode != 0
        assert "Unknown model(s)" in result.stderr

    def test_multiconcept_flag_without_multiconcept_config_errors_cleanly(self, run_script):
        result = run_script(
            "main_store_cav.py",
            ["--config_file", C.SINGLE_CONFIG_PATH, "--store_multiconcept_cav"],
        )
        assert result.returncode != 0
        assert "multiconcept is not enabled" in result.stderr

    def test_single_concept_end_to_end(self, run_script, tmp_path):
        cav_store = str(tmp_path / "cav_store_single")
        result = run_script(
            "main_store_cav.py",
            [
                "--config_file", C.SINGLE_CONFIG_PATH,
                "--model_name", C.MODEL_NAME,
                "--model_path", C.SINGLE_MODEL_ROOT,
                "--cav_store", cav_store,
                "--skip_layers", str(C.SKIP_LAYERS),
            ],
        )
        result.assert_success()

        assert os.path.isfile(os.path.join(cav_store, "manifest.json"))
        assert os.path.isfile(os.path.join(cav_store, f"{C.MODEL_NAME}_manifest.json"))
        for layer in C.PROCESSED_LAYER_NAMES:
            joblib_path = os.path.join(cav_store, C.MODEL_NAME, C.SINGLE_CONCEPT_NAME, f"{layer}.joblib")
            assert os.path.isfile(joblib_path), f"missing {joblib_path}"

        with open(os.path.join(cav_store, "manifest.json")) as fh:
            manifest = json.load(fh)
        assert C.SINGLE_CONCEPT_NAME in manifest.get("concepts", {})

    def test_multiclass_multiconcept_end_to_end(self, run_script, tmp_path):
        cav_store = str(tmp_path / "cav_store_multiclass")
        result = run_script(
            "main_store_cav.py",
            [
                "--config_file", C.MULTI_CONFIG_PATH,
                "--model_name", C.MODEL_NAME,
                "--model_path", C.MULTI_MODEL_ROOT,
                "--cav_store", cav_store,
                "--store_multiconcept_cav",
                "--skip_layers", str(C.SKIP_LAYERS),
            ],
        )
        result.assert_success()

        # Re-derive expected concept names using main_store_cav.py's naming convention:
        # f"{class_name}_{basename(target_folder)}".
        expected_concepts = []
        for class_idx, folders in C.MULTI_CLASS_CONCEPTS.items():
            class_name = C.MULTI_CLASSES[class_idx]
            for folder in folders:
                expected_concepts.append(f"{class_name}_{folder}")

        for concept in expected_concepts:
            for layer in C.PROCESSED_LAYER_NAMES:
                joblib_path = os.path.join(cav_store, C.MODEL_NAME, concept, f"{layer}.joblib")
                assert os.path.isfile(joblib_path), f"missing {joblib_path}"

        # The single-concept "primary" folder must NOT be processed in --store_multiconcept_cav mode.
        assert not os.path.isdir(os.path.join(cav_store, C.MODEL_NAME, C.MULTI_PRIMARY_CONCEPT_NAME))

    def test_rerun_without_rebuild_skips_already_stored_concepts(self, run_script, tmp_path):
        cav_store = str(tmp_path / "cav_store_rerun")
        common_args = [
            "--config_file", C.SINGLE_CONFIG_PATH,
            "--model_name", C.MODEL_NAME,
            "--model_path", C.SINGLE_MODEL_ROOT,
            "--cav_store", cav_store,
            "--skip_layers", str(C.SKIP_LAYERS),
        ]
        first = run_script("main_store_cav.py", common_args)
        first.assert_success()

        second = run_script("main_store_cav.py", common_args)
        second.assert_success()
        assert "Already up to date" in second.stdout or "No new/changed concepts" in second.stdout

    def test_rebuild_cav_flag_recomputes_even_when_up_to_date(self, run_script, tmp_path):
        cav_store = str(tmp_path / "cav_store_rebuild")
        common_args = [
            "--config_file", C.SINGLE_CONFIG_PATH,
            "--model_name", C.MODEL_NAME,
            "--model_path", C.SINGLE_MODEL_ROOT,
            "--cav_store", cav_store,
            "--skip_layers", str(C.SKIP_LAYERS),
        ]
        first = run_script("main_store_cav.py", common_args)
        first.assert_success()

        second = run_script("main_store_cav.py", common_args + ["--rebuild_cav"])
        second.assert_success()
        assert "ALL concepts will be recomputed" in second.stdout
        assert "Already up to date" not in second.stdout

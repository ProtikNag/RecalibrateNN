# lean_recalibration Test Suite

Automated pytest suite for the four core `lean_recalibration` scripts:

| Script                                      | What it does                                                   |
|----------------------------------------------|------------------------------------------------------------------|
| `main_store_cav.py`                          | Trains linear CAVs (Concept Activation Vectors) per layer/concept and writes them + a manifest to a CAV store. |
| `main_recalib_custom_by_loading_cav.py`      | Loads CAVs from the store and recalibrates (fine-tunes) a model against them, per class or jointly (multiclass). |
| `utils_sensitivity_multiclass.py`            | Computes TCAV-style sensitivity/alignment scores for each concept/layer, before/after recalibration, and writes CSV + Excel reports. |
| `bottleneck_detection.py`                    | Reads a sensitivity Excel report and flags layer-wise "bottleneck" concepts using statistical thresholds; writes a summary workbook (+ optional charts). |

Everything needed to run the tests — synthetic images, tiny model checkpoints, YAML configs,
helper code, and the tests themselves — lives in this one self-contained folder. Nothing here
depends on any real dataset, real pretrained weights, or GPU.

**Design goals:** OS-agnostic (Windows/Linux/macOS, no shell scripts), fully self-contained,
relocatable (can be moved/renamed to anywhere inside the project checkout and will self-heal),
and covers both **single-class** and **multiclass** concept verification for every script.

---

## 1. Prerequisites

- Python 3.9+ (verified with 3.12)
- The repo's own dependencies (torch, torchvision, pandas, openpyxl, PyYAML, scikit-learn,
  matplotlib, Pillow) — already listed in the project root `requirements.txt`.
- `pytest` (the only addition this test suite itself needs on top of the above).

If you already have the project's main `requirements.txt` installed, you only need to add pytest:

```bash
pip install pytest
```

Otherwise, install everything this test suite needs directly:

```bash
pip install -r requirements-test.txt
```

No GPU is required — everything runs on CPU with tiny (a few KB) synthetic tensors/images.

---

## 2. Quick start — command lines

Run these from **this folder** (`lean_recalibration/unit_test/`, or wherever you have moved it —
see [Relocating this folder](#6-relocating-this-folder-os-agnostic--self-healing) below).

```bash
# Full suite: every unit + integration + pipeline test (~120 tests, ~4-6 minutes)
python run_tests.py

# Same thing, calling pytest directly (equivalent to `run_tests.py` with no flags)
python -m pytest -v

# Fast subset only: in-process unit tests, no subprocess/training (~seconds)
python run_tests.py --unit

# Only the subprocess/CLI end-to-end tests (slower, spawns real training subprocesses)
python run_tests.py --integration

# Just one target module's tests
python run_tests.py --module bottleneck      # or: store_cav | recalib | sensitivity | integration

# Exclude the slowest (training-loop) tests
python run_tests.py --skip-slow

# List what would run without executing anything
python run_tests.py --list

# Force-regenerate all synthetic data/configs first (rarely needed - see below)
python run_tests.py --regenerate-data

# Forward extra args straight to pytest, e.g. run a single test by keyword
python run_tests.py -- -k "multiclass"
```

`run_tests.py` is a thin, dependency-free wrapper around `python -m pytest` (see its own
`--help` for the full option list). Its exit code is pytest's exit code (`0` = all passed), so
it can be used directly as a CI step. You can always fall back to calling `pytest` yourself with
any options it supports (`-k`, `-x`, `--pdb`, `-n auto` with pytest-xdist, etc.) — nothing here
is required to go through `run_tests.py`.

---

## 3. What gets tested — single-class vs multiclass

The scripts' own vocabulary is reused directly:

- **Single-class / `concept_mode=single`**: one concept per class, via the config's `concept:`
  section. Exercised with 2 synthetic classes (`cat`, `dog`) and 1 concept (`stripes`).
- **Multiclass / `concept_mode=multiclass`**: multiple concepts per class, via the config's
  `multiconcept:` section, `--store_multiconcept_cav`, and `--multiclass_recalibration_mode`.
  Exercised with 3 synthetic classes (`cat`, `dog`, `bird`) and 4 concepts
  (`cat_spots`, `cat_ears`, `dog_collar`, `bird_feathers`).

Every target script's test file includes both scenarios; `test_integration_pipeline.py`
additionally chains all four scripts together using each stage's REAL output as the next
stage's real input (not hand-crafted fixtures) for the multiclass case, and stages 1-2 only for
the single-class case (see that file's module docstring for why — short version: a real,
pre-existing naming-mismatch bug described in section 5 below makes it a dead end for
single-class after stage 2).

Each concept/dataset folder contains **5 generated images** (`helpers/image_gen.py`, seeded, tiny
`64x64` PNGs) — this matches `bottleneck_detection.py`'s own default `--min-samples 5` exactly.

### Test inventory

| File                                              | Unit tests | Integration/CLI tests | Total |
|-----------------------------------------------------|:---:|:---:|:---:|
| `test_bottleneck_detection.py`                      | 38 | 5  | 43  |
| `test_main_store_cav.py`                            | 11 | 9  | 20  |
| `test_main_recalib_custom_by_loading_cav.py`        | 16 | 4  | 20  |
| `test_utils_sensitivity_multiclass.py`              | 28 | 7  | 35  |
| `test_integration_pipeline.py`                      | 0  | 2  | 2   |
| **Total**                                            | **93** | **27** | **120** |

("Unit" = fast, in-process, no subprocess/training. "Integration" = real subprocess invocation
of the actual CLI script, some including a small real training loop — marked `slow` where applicable.)

---

## 4. Folder layout

```
unit_test/
  README.md                  <- this file
  requirements-test.txt      <- pip install -r requirements-test.txt
  pytest.ini                 <- marker registration (unit / integration / slow)
  conftest.py                <- shared fixtures (see comments inside for design notes)
  generate_test_data.py      <- (re)generates all synthetic images/models/configs
  run_tests.py                <- OS-agnostic test runner (see section 2)
  helpers/
    path_discovery.py        <- marker-file upward search that locates the repo root
    tiny_model.py             <- builds+saves a tiny CNN checkpoint (no custom classes)
    image_gen.py              <- generates the 5-images-per-folder synthetic datasets
    synthetic_reports.py      <- hand-crafted sensitivity-report workbooks (bottleneck tests only)
    constants.py               <- single source of truth for every test path/name
  config/                     <- generated YAML configs (see section 5)
  data/                       <- generated synthetic images, tiny model checkpoints, CAV stores
  results/                    <- generated script outputs (recalibration results, sensitivity
                                  reports, bottleneck reports) - written here, NOT to a temp dir,
                                  so you can inspect them after a run
  test_main_store_cav.py
  test_main_recalib_custom_by_loading_cav.py
  test_bottleneck_detection.py
  test_utils_sensitivity_multiclass.py
  test_integration_pipeline.py
```

`data/`, `config/`, and `results/` (and `.pytest_cache/`, `__pycache__/`) are all generated
locally on first run; safe to delete entirely at any time — everything regenerates automatically
on the next `pytest`/`run_tests.py` invocation (see `conftest.py`'s `_ensure_test_data` fixture).

---

## 5. Known, pre-existing production behaviours documented (not fixed) by this suite

While building these tests, two genuine mismatches were found between the target scripts. Per
explicit direction, **production code was intentionally left unchanged**; instead, dedicated
tests pin/document the CURRENT behaviour so any future fix is a deliberate, visible change:

1. **Single-class concept-naming mismatch** (`main_store_cav.py` vs `utils_sensitivity_multiclass.py`):
   `main_store_cav.py`'s single-concept path stores CAVs under the bare concept-folder name
   (e.g. `"stripes"`), but `utils_sensitivity_multiclass.py`'s `build_concept_specs()` always
   looks single-mode concepts up under a class-prefixed name (e.g. `"cat_stripes"`). A real
   single-class sensitivity run against a real CAV store therefore **exits 0 but silently
   collects zero rows** (no CSV/Excel produced). Multiclass mode is unaffected — both scripts
   already agree on class-prefixed naming there.
   Pinned by: `test_utils_sensitivity_multiclass.py::TestUtilsSensitivityCli::
   test_single_class_end_to_end_documents_concept_naming_mismatch`.

2. **`--before_after` checkpoint-naming mismatch**: `utils_sensitivity_multiclass.py`'s
   `get_model_path()` expects recalibrated checkpoints named `loss_{model}_{layer}_{lambda}.pth`,
   but `main_recalib_custom_by_loading_cav.py` actually saves
   `model_cls{class}_combo{idx}_lambda{val}.pth` / `model_multiclass_combo{idx}_lambda{val}.pth`.
   Tested in isolation by manually creating checkpoints under the exact name
   `get_model_path()` itself expects (a legitimate way to test that function's own contract).
   See: `test_utils_sensitivity_multiclass.py::TestUtilsSensitivityCli::
   test_before_after_loads_recalibrated_checkpoint`.

If/when either of these is fixed in production code, the corresponding test's assertions will
need to be updated to match the new (correct) behaviour — search for `NOTE:` / the test names
above in `test_utils_sensitivity_multiclass.py` for the exact details.

---

## 6. Relocating this folder (OS-agnostic + self-healing)

This folder can be moved anywhere inside the project checkout (directly under the repo root,
nested under `lean_recalibration/`, renamed to `test/`/`unit_test/`/anything else, etc.).
`helpers/path_discovery.py` walks upward from wherever the folder actually is, looking for
`main_store_cav.py` and `ConfigSingleton.py` as markers, instead of assuming a fixed relative
nesting depth. `conftest.py` uses this to put the correct repo root on `sys.path`/`PYTHONPATH`
automatically every session, and unconditionally rewrites the generated YAML configs (which
embed absolute paths) on every run so they can never point at a stale location.

If you ever see `ModuleNotFoundError: No module named 'ConfigSingleton'` (or `cav_registry`,
`custom_dataloader`, `utils`, `logger`), it means the folder was moved to somewhere those
marker files genuinely cannot be found above it — e.g. copied outside the project checkout
entirely. Moving it back under (or alongside) `lean_recalibration/` resolves it automatically;
no code changes are needed.

---

## 7. Cleaning up / regenerating

```bash
# Wipe and regenerate everything from scratch (images, models, configs)
python run_tests.py --regenerate-data

# Or manually:
python generate_test_data.py --force

# Or just delete the generated folders yourself - they are fully regenerated next run:
#   data/, config/, results/, .pytest_cache/, __pycache__/ (and helpers/__pycache__/)
```

---

## 8. Troubleshooting

- **Tests hang or time out**: the integration/CLI tests spawn real (tiny) training subprocesses;
  the first run of a session is slower because CAV stores are built once per session and
  reused. Subsequent runs of the same session are fast. If a run gets interrupted mid-way,
  just re-run — `--regenerate-data` is only needed if generated files look corrupted.
- **`ModuleNotFoundError: No module named 'ConfigSingleton'`**: see section 6 above.
- **Stale absolute paths after moving the folder**: handled automatically (config files are
  rewritten every session) — no action needed.
- **Windows vs Linux path separators in errors**: not a bug — all paths in this suite are built
  with `os.path.join`/`pathlib`, so forward vs backslash differences you might see in printed
  paths are simply how each OS reports its own paths; behavior is identical on both.

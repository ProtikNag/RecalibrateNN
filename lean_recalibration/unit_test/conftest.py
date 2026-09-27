"""
Shared pytest fixtures/configuration for the lean_recalibration test suite.

Design notes
------------
* ``ConfigSingleton`` and ``Logger_Singleton`` (in the repo root) are TRUE process-wide
  singletons (class-level ``_instance``). Any in-process unit test that touches either
  class must not leak state into the next test, so we reset both ``_instance`` attributes
  before AND after every test function (autouse fixture below).
* The four target scripts are primarily *command line tools*. The most faithful way to
  verify them end-to-end is to invoke them exactly as a user would: as a subprocess
  (``python <script>.py --arg ...``). This sidesteps singleton/sys.argv/sys.path global
  state entirely and works identically on Windows and Linux. The ``run_script`` fixture
  below provides this.
* For pure-logic unit tests (helper functions), the modules are imported in-process. The
  repo root and the ``lean_recalibration`` folder are added to ``sys.path`` here so bare
  imports (``logger``, ``custom_dataloader``, ``ConfigSingleton``, ``cav_registry``,
  ``utils``) used inside the target scripts resolve correctly regardless of the current
  working directory pytest was launched from.
* All paths are constructed with ``os.path`` and passed through ``str(Path(...))`` where
  needed so the suite runs unmodified on Windows or Linux.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# sys.path setup (must happen before any `import <repo module>` in test files)
# ---------------------------------------------------------------------------
# NOTE: we deliberately do NOT assume a fixed relative nesting between this test folder
# and the actual repo (e.g. "the test folder is always a direct sibling of
# lean_recalibration/"). The folder may be placed anywhere inside the project checkout,
# e.g. directly under the repo root (`<repo_root>/test/`) or nested one level deeper
# (`<repo_root>/lean_recalibration/unit_test/`). See helpers/path_discovery.py for the
# marker-file-based upward search that makes this work regardless of location -- this is
# what fixes `ModuleNotFoundError: No module named 'ConfigSingleton'` when the folder is
# moved/nested differently than originally assumed.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)  # needed to import helpers.path_discovery itself

from helpers.path_discovery import discover_paths  # noqa: E402

REPO_ROOT, LEAN_RECALIB_DIR = discover_paths(TEST_DIR)

for _p in (REPO_ROOT, LEAN_RECALIB_DIR, TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe plotting everywhere

from helpers import constants as C  # noqa: E402


# ---------------------------------------------------------------------------
# Ensure synthetic test data + configs exist before the session starts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _ensure_test_data():
    """(Re)generate synthetic images/models/configs once per test session.

    generate_test_data.main() is itself idempotent/cheap: it SKIPS regenerating any
    image/model file that already exists (fast os.path.isfile/Path.exists checks only),
    but it ALWAYS rewrites the YAML config files with paths computed from THIS run's
    discovered TEST_ROOT/REPO_ROOT. That rewrite is essential (not just a nice-to-have):
    the YAML configs embed absolute filesystem paths, so if this test folder is ever
    moved/copied/renamed (e.g. `test/` -> `unit_test/` -> `lean_recalibration/unit_test/`),
    any config file left over from a previous location would silently point at paths that
    no longer exist. Always calling main() (instead of only when files are "missing")
    guarantees the configs self-heal on the very next test run after a move, with no
    perceptible cost.
    """
    import generate_test_data
    generate_test_data.main()
    os.makedirs(C.RESULTS_ROOT, exist_ok=True)
    yield


# ---------------------------------------------------------------------------
# Singleton isolation for in-process unit tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset ConfigSingleton/Logger_Singleton before and after every test.

    Without this, whichever config/log file was used to construct the singleton FIRST
    across the whole pytest session would silently "stick" for every subsequent test
    that constructs `ConfigSingleton(other_path)` or `Logger_Singleton(other_path)`.
    """
    from ConfigSingleton import ConfigSingleton
    from logger import Logger_Singleton

    ConfigSingleton._instance = None
    Logger_Singleton._instance = None
    yield
    ConfigSingleton._instance = None
    Logger_Singleton._instance = None


# ---------------------------------------------------------------------------
# Subprocess-based CLI runner (used for full-pipeline / integration tests)
# ---------------------------------------------------------------------------
class ScriptResult:
    def __init__(self, proc: subprocess.CompletedProcess):
        self.returncode = proc.returncode
        self.stdout = proc.stdout
        self.stderr = proc.stderr

    def assert_success(self):
        assert self.returncode == 0, (
            f"Script exited with code {self.returncode}\n"
            f"--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"
        )


def _run_script(script_name: str, args: list, cwd: str | None = None, timeout: int = 900, env_overrides: dict | None = None):
    """Run a lean_recalibration script as a real subprocess and return a ScriptResult.

    Module-level so it can be shared by both the function-scoped ``run_script`` fixture
    (used directly by tests) and any session-scoped setup fixtures (e.g. CAV-store
    pre-population) that a session-scoped fixture is not allowed to depend on a
    function-scoped one for in pytest.
    """
    script_path = os.path.join(LEAN_RECALIB_DIR, script_name)
    assert os.path.isfile(script_path), f"Script not found: {script_path}"

    env = os.environ.copy()
    # Repo root on PYTHONPATH makes the target script's bare `import logger`,
    # `import utils`, etc. resolve regardless of the subprocess's cwd, and makes
    # its own `sys.path.append("../")` line harmless.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT, existing_pp]) if existing_pp else REPO_ROOT
    env["MPLBACKEND"] = "Agg"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.run(
        [sys.executable, script_path] + args,
        cwd=cwd or REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ScriptResult(proc)


@pytest.fixture
def run_script():
    """Return a callable that runs a lean_recalibration script as a real subprocess.

    Usage:
        result = run_script("main_store_cav.py", ["--config_file", cfg, ...])
        result.assert_success()
    """
    return _run_script


# ---------------------------------------------------------------------------
# Session-scoped CAV store pre-population (used by recalibration/sensitivity tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def populated_single_cav_store(_ensure_test_data):
    """Run main_store_cav.py ONCE per session for the single-concept scenario and return
    the resulting CAV store root. Downstream scripts (recalibration, sensitivity) only
    ever READ this store, so building it once and sharing it across every test that needs
    it avoids re-computing identical CAVs dozens of times."""
    if not os.path.isfile(os.path.join(C.SINGLE_CAV_STORE, "manifest.json")):
        result = _run_script(
            "main_store_cav.py",
            [
                "--config_file", C.SINGLE_CONFIG_PATH,
                "--model_name", C.MODEL_NAME,
                "--model_path", C.SINGLE_MODEL_ROOT,
                "--cav_store", C.SINGLE_CAV_STORE,
                "--skip_layers", str(C.SKIP_LAYERS),
            ],
        )
        result.assert_success()
    return C.SINGLE_CAV_STORE


@pytest.fixture(scope="session")
def populated_multi_cav_store(_ensure_test_data):
    """Run main_store_cav.py --store_multiconcept_cav ONCE per session for the multiclass
    scenario and return the resulting CAV store root."""
    if not os.path.isfile(os.path.join(C.MULTI_CAV_STORE, "manifest.json")):
        result = _run_script(
            "main_store_cav.py",
            [
                "--config_file", C.MULTI_CONFIG_PATH,
                "--model_name", C.MODEL_NAME,
                "--model_path", C.MULTI_MODEL_ROOT,
                "--cav_store", C.MULTI_CAV_STORE,
                "--store_multiconcept_cav",
                "--skip_layers", str(C.SKIP_LAYERS),
            ],
        )
        result.assert_success()
    return C.MULTI_CAV_STORE


@pytest.fixture
def tmp_results_dir(tmp_path):
    """A fresh temp directory for a test's output artifacts (auto-cleaned by pytest)."""
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture(scope="session")
def single_config_path():
    return C.SINGLE_CONFIG_PATH


@pytest.fixture(scope="session")
def multi_config_path():
    return C.MULTI_CONFIG_PATH


@pytest.fixture(scope="session")
def single_config_recalib_path():
    """Fast config variant (1 VGG layer, override_retraining=True) for recalibration CLI tests."""
    return C.SINGLE_CONFIG_RECALIB_PATH


@pytest.fixture(scope="session")
def multi_config_recalib_path():
    """Fast config variant (1 VGG layer, override_retraining=True) for recalibration CLI tests."""
    return C.MULTI_CONFIG_RECALIB_PATH


@pytest.fixture(scope="session")
def single_model_root():
    return C.SINGLE_MODEL_ROOT


@pytest.fixture(scope="session")
def multi_model_root():
    return C.MULTI_MODEL_ROOT

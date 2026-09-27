#!/usr/bin/env python3
"""
OS-agnostic test runner for the lean_recalibration test suite.

Thin wrapper around ``python -m pytest`` with a few convenience presets. Uses
``sys.executable`` and ``subprocess`` (no shell commands, no ``.sh``/``.bat`` scripts), so the
exact same command works unmodified on Windows, Linux, and macOS.

Examples
--------
    python run_tests.py                      # full suite: all unit + integration + pipeline tests
    python run_tests.py --unit                # fast, in-process unit tests only (seconds)
    python run_tests.py --integration          # subprocess/CLI end-to-end tests only (minutes)
    python run_tests.py --skip-slow            # exclude tests marked @pytest.mark.slow
    python run_tests.py --module bottleneck    # just one target module's test file
    python run_tests.py --regenerate-data      # force-regenerate synthetic images/configs first
    python run_tests.py --list                 # list collected tests only, do not execute
    python run_tests.py -- -k "multiclass"     # anything after `--` is forwarded to pytest as-is

Exit code is pytest's own exit code (0 = all tests passed), so this script is safe to use
directly as a CI step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Maps a short, memorable name to each target module's test file, so
# `--module bottleneck` is easier to type/remember than the full filename.
MODULE_FILES = {
    "store_cav": "test_main_store_cav.py",
    "recalib": "test_main_recalib_custom_by_loading_cav.py",
    "bottleneck": "test_bottleneck_detection.py",
    "sensitivity": "test_utils_sensitivity_multiclass.py",
    "integration": "test_integration_pipeline.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--unit", action="store_true",
        help="Run only fast, in-process unit tests (pytest marker: unit).",
    )
    scope.add_argument(
        "--integration", action="store_true",
        help="Run only subprocess/CLI end-to-end tests (pytest marker: integration).",
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Exclude tests marked @pytest.mark.slow (small-training-loop tests).",
    )
    parser.add_argument(
        "--module", choices=sorted(MODULE_FILES) + ["all"], default="all",
        help="Restrict the run to one target module's test file (default: all).",
    )
    parser.add_argument(
        "--regenerate-data", action="store_true",
        help="Force-regenerate synthetic images/models/configs before running tests "
             "(equivalent to running `python generate_test_data.py --force` first). "
             "Not required for normal use: the suite already self-heals stale/missing "
             "data automatically every session.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List collected tests only (pytest --collect-only -q); do not execute anything.",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER,
        help="Extra arguments forwarded verbatim to pytest. Prefix with `--` "
             "e.g. `run_tests.py -- -k multiclass`.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.regenerate_data:
        print("=== Regenerating synthetic test data (--force) ===", flush=True)
        gen_cmd = [sys.executable, os.path.join(THIS_DIR, "generate_test_data.py"), "--force"]
        gen_result = subprocess.run(gen_cmd, cwd=THIS_DIR)
        if gen_result.returncode != 0:
            return gen_result.returncode

    cmd = [sys.executable, "-m", "pytest"]

    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.skip_slow:
        markers.append("not slow")
    if markers:
        cmd += ["-m", " and ".join(markers)]

    if args.module != "all":
        cmd.append(MODULE_FILES[args.module])

    if args.list:
        cmd += ["--collect-only", "-q"]
    else:
        cmd.append("-v")

    extra = list(args.pytest_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    print("=== Running:", " ".join(cmd), "===", flush=True)
    print(f"(working directory: {THIS_DIR})", flush=True)
    result = subprocess.run(cmd, cwd=THIS_DIR)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

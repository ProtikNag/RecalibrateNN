"""
Robust filesystem discovery so the test suite works no matter where the test folder
(``unit_test/``, a sibling ``test/`` directly under the repo root, or any other
copy/rename/relocation of it) is placed relative to the actual lean_recalibration
project checkout.

Instead of assuming a fixed relative nesting (e.g. "the test folder is always a direct
sibling of lean_recalibration/", or "the repo root is always two levels up"), we walk UP
the directory tree from this file's location looking for two marker files:

  * ``main_store_cav.py``  -> marks the ``lean_recalibration`` folder (the 4 target
    scripts under test).
  * ``ConfigSingleton.py`` -> marks the actual repo root (shared root-level modules
    such as ``ConfigSingleton``, ``logger``, ``utils``, ``cav_registry``).

This means the test suite keeps working unmodified whether it lives at
``<repo_root>/test/``, ``<repo_root>/unit_test/`` or
``<repo_root>/lean_recalibration/unit_test/`` -- which directly fixes:
``ModuleNotFoundError: No module named 'ConfigSingleton'`` when the test folder is
nested one level deeper (or shallower) than whatever fixed assumption was hardcoded.
"""

import os


def _find_ancestor_containing(start_dir: str, filename: str, max_levels: int = 12):
    """Walk upward from start_dir (inclusive) looking for a directory containing `filename`.
    Returns the absolute directory path if found, else None."""
    current = os.path.abspath(start_dir)
    for _ in range(max_levels):
        if os.path.isfile(os.path.join(current, filename)):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # reached filesystem root
            break
        current = parent
    return None


def discover_paths(start_dir: str):
    """Return (repo_root, lean_recalibration_dir) discovered relative to `start_dir`
    (typically the test folder's own directory, wherever it has been placed).
    """
    lean_recalib_dir = _find_ancestor_containing(start_dir, "main_store_cav.py")
    search_base_for_repo_root = lean_recalib_dir or start_dir
    repo_root = _find_ancestor_containing(search_base_for_repo_root, "ConfigSingleton.py")

    if repo_root is None:
        # Fall back to the suite's original layout assumption (test folder directly
        # under repo root) so behavior is unchanged if discovery somehow can't find
        # the marker files (e.g. ConfigSingleton.py was renamed/removed).
        repo_root = os.path.dirname(os.path.abspath(start_dir))
    if lean_recalib_dir is None:
        candidate = os.path.join(repo_root, "lean_recalibration")
        lean_recalib_dir = candidate if os.path.isdir(candidate) else repo_root

    return os.path.abspath(repo_root), os.path.abspath(lean_recalib_dir)

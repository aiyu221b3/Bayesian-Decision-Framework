"""Acceptance tests for the supported Python 3.11 synthetic test foundation."""

import importlib.metadata
import os
import sys
from pathlib import Path

import matplotlib

from test_support import (
    DEFECT_TRACEABILITY,
    HYPOTHESIS_SEED,
    NUMERIC_TOLERANCE,
    PRESERVATION_TRACEABILITY,
    PROPERTY_SEEDS,
    ROOT,
)

EXPECTED_DEPENDENCIES = {
    "numpy": "2.2.6",
    "matplotlib": "3.10.7",
    "seaborn": "0.13.2",
    "pytest": "9.0.2",
    "hypothesis": "6.112.1",
}


def _locked_dependencies(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split("==", 1) for line in lines if line and not line.startswith("#"))


def test_supported_python_and_dependency_versions_match_lock():
    """**Validates: Requirements 2.9, 3.8**"""
    assert sys.version_info[:2] == (3, 11)
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.9"
    assert _locked_dependencies(ROOT / "requirements.txt") == EXPECTED_DEPENDENCIES
    for name, version in EXPECTED_DEPENDENCIES.items():
        assert importlib.metadata.version(name) == version


def test_numeric_tolerance_and_property_seeds_are_central_and_valid():
    assert 0.0 < NUMERIC_TOLERANCE < 1.0
    assert PROPERTY_SEEDS and len(PROPERTY_SEEDS) == len(set(PROPERTY_SEEDS))
    assert all(seed >= 0 for seed in (*PROPERTY_SEEDS, HYPOTHESIS_SEED))


def test_all_defect_and_preservation_clauses_have_test_mappings():
    assert set(DEFECT_TRACEABILITY) == {f"1.{index}" for index in range(1, 12)}
    assert set(PRESERVATION_TRACEABILITY) == {f"3.{index}" for index in range(1, 10)}
    assert all(tests for tests in (*DEFECT_TRACEABILITY.values(), *PRESERVATION_TRACEABILITY.values()))


def test_matplotlib_uses_the_noninteractive_backend():
    assert os.environ["MPLBACKEND"] == "Agg"
    assert matplotlib.get_backend().lower() == "agg"

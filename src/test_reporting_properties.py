"""Red-phase Properties 11, 12, and 14 for auditable reporting.

The production reporting API is intentionally introduced by Tasks 3.13/3.14.
These tests must not be weakened when that API is absent.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, seed, settings, strategies as st

from test_support import HYPOTHESIS_SEED, NUMERIC_TOLERANCE

ATOL = NUMERIC_TOLERANCE
METHODS = ("M0", "M1", "M2", "M3", "MT")
CATEGORIES = ("synthetic evidence", "empirical approximation", "future pilot")
STATUSES = ("passed", "failed", "inconclusive")
DIRECTIONS = ("positive", "negative", "null", "mixed")
INTERVAL_METHOD = "two-sided 95% paired-seed percentile bootstrap"
ARTIFACT_NAMES = ("frontier.png", "perseveration.png", "reliability.png", "tree_specificity.png")

def _reporting():
    """Load the Task 3.13 API while preserving a useful red-phase failure."""
    try:
        return importlib.import_module("reporting")
    except ModuleNotFoundError as error:
        if error.name == "reporting":
            pytest.fail(
                "Task 3.13 reporting API is absent: expected reporting.py with "
                "build_headline_report, replay_headline_report, "
                "validate_claim_artifact_registry, and validate_outcome_retention"
            )
        raise


def _effect_spec():
    return {
        "effect_id": "M2_minus_M0.boundary_coverage",
        "left_method": "M2",
        "right_method": "M0",
        "metric": "boundary_coverage",
        "units": "proportion of eligible boundaries",
    }


@st.composite
def headline_seed_records(draw, minimum=10, maximum=15):
    seeds = draw(st.lists(
        st.integers(min_value=1, max_value=2**31 - 1),
        min_size=minimum,
        max_size=maximum,
        unique=True,
    ))
    rows = []
    for evaluation_seed in seeds:
        eligible = draw(st.integers(min_value=2, max_value=40))
        methods = {}
        for method in METHODS:
            hits = draw(st.integers(min_value=0, max_value=eligible))
            methods[method] = {
                "values": {"boundary_coverage": hits / eligible},
                "counts": {
                    "boundary_hits": hits,
                    "boundary_eligible": eligible,
                },
            }
        rows.append({
            "seed": evaluation_seed,
            "scenario": "abrupt-change",
            "ordered_test_stream_ids": tuple(
                f"test-{evaluation_seed}-{index}" for index in range(eligible)
            ),
            "methods": methods,
        })
    return rows

def _provenance(records):
    from protocol import content_hash

    config = {
        "profile": "headline",
        "methods": METHODS,
        "scenarios": ("abrupt-change",),
        "interval": INTERVAL_METHOD,
    }
    seeds = tuple(row["seed"] for row in records)
    test_ids = tuple(
        stream_id
        for row in records
        for stream_id in row["ordered_test_stream_ids"]
    )
    return {
        "schema_version": "1",
        "code_revision": "0123456789abcdef0123456789abcdef01234567",
        "environment_id": "cpython-3.11-requirements-sha256:fixture",
        "metric_definition_version": "metric-record-v1",
        "freeze_timestamp": "2025-01-01T00:00:00+00:00",
        "complete_config": config,
        "config_hash": content_hash(config),
        "predeclared_evaluation_seeds": seeds,
        "role_stream_ids": {
            "fit": ("fit-fixture",),
            "development": ("development-fixture",),
            "stress": ("stress-fixture",),
            "test": test_ids,
        },
        "scenarios": ("abrupt-change",),
        "methods": METHODS,
        "numeric_rerun_tolerance": ATOL,
    }


def _bootstrap_oracle(differences, bootstrap_seed):
    values = np.asarray(differences, dtype=float)
    assert len(values) >= 10 and np.all(np.isfinite(values))
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _assert_replay_close(expected, observed):
    if isinstance(expected, Mapping):
        assert set(observed) == set(expected)
        for key in expected:
            _assert_replay_close(expected[key], observed[key])
    elif isinstance(expected, (tuple, list)):
        assert len(observed) == len(expected)
        for left, right in zip(expected, observed):
            _assert_replay_close(left, right)
    elif isinstance(expected, float):
        assert math.isclose(expected, observed, rel_tol=0.0, abs_tol=ATOL)
    else:
        assert observed == expected

@seed(HYPOTHESIS_SEED)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
@given(
    records=headline_seed_records(),
    bootstrap_seed=st.integers(min_value=0, max_value=2**32 - 1),
    artifact_payload=st.binary(min_size=1, max_size=64),
)
def test_property_11_generated_headline_reports_have_complete_replayable_provenance(
    records, bootstrap_seed, artifact_payload, tmp_path
):
    """Property 11. **Validates: Requirements 2.7, 2.8, 2.9, 3.7**"""
    reporting = _reporting()
    artifact_paths = {}
    for name in ARTIFACT_NAMES:
        path = tmp_path / name
        path.write_bytes(artifact_payload + name.encode("utf-8"))
        artifact_paths[name] = path

    provenance = _provenance(records)
    effect_spec = _effect_spec()
    report = reporting.build_headline_report(
        records,
        provenance=provenance,
        artifact_paths=artifact_paths,
        paired_effects=(effect_spec,),
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=10_000,
        confidence_level=0.95,
    )

    assert report["profile"] == "headline"
    assert tuple(report["provenance"]["predeclared_evaluation_seeds"]) == tuple(
        row["seed"] for row in records
    )
    assert len(set(report["provenance"]["predeclared_evaluation_seeds"])) >= 10
    assert len(report["per_seed"]) == len(records)
    for source, retained in zip(records, report["per_seed"]):
        assert retained["seed"] == source["seed"]
        assert retained["ordered_test_stream_ids"] == source["ordered_test_stream_ids"]
        for method in METHODS:
            assert retained["methods"][method]["counts"] == source["methods"][method]["counts"]
            assert retained["methods"][method]["values"] == source["methods"][method]["values"]

    differences = [
        row["methods"]["M2"]["values"]["boundary_coverage"]
        - row["methods"]["M0"]["values"]["boundary_coverage"]
        for row in records
    ]
    expected_mean, expected_lower, expected_upper = _bootstrap_oracle(
        differences, bootstrap_seed
    )
    effect = report["paired_effects"][effect_spec["effect_id"]]
    assert effect["units"] == effect_spec["units"]
    assert effect["finite_pair_count"] == len(records)
    assert effect["interval_method"] == INTERVAL_METHOD
    assert effect["bootstrap_seed"] == bootstrap_seed
    assert effect["bootstrap_replicates"] == 10_000
    assert effect["confidence_level"] == 0.95
    assert math.isclose(effect["mean"], expected_mean, rel_tol=0.0, abs_tol=ATOL)
    assert math.isclose(effect["lower"], expected_lower, rel_tol=0.0, abs_tol=ATOL)
    assert math.isclose(effect["upper"], expected_upper, rel_tol=0.0, abs_tol=ATOL)
    assert all(math.isfinite(effect[key]) for key in ("mean", "lower", "upper"))

    assert report["provenance"]["config_hash"] == provenance["config_hash"]
    assert set(report["artifacts"]) == set(ARTIFACT_NAMES)
    for name, path in artifact_paths.items():
        assert report["artifacts"][name]["path"] == str(path)
        assert report["artifacts"][name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    replay = reporting.replay_headline_report(
        report,
        seed_records=records,
        artifact_paths=artifact_paths,
    )
    _assert_replay_close(report, replay)


@seed(HYPOTHESIS_SEED)
@given(
    records=headline_seed_records(minimum=1, maximum=9),
    bootstrap_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_property_11_rejects_fewer_than_ten_predeclared_evaluation_seeds(
    records, bootstrap_seed
):
    """Property 11. **Validates: Requirements 2.8, 2.9**"""
    reporting = _reporting()
    with pytest.raises(reporting.ReportValidationError, match="ten|10|seed"):
        reporting.build_headline_report(
            records,
            provenance=_provenance(records),
            artifact_paths={},
            paired_effects=(_effect_spec(),),
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=10_000,
            confidence_level=0.95,
        )


@st.composite
def claim_registries(draw):
    count = draw(st.integers(min_value=1, max_value=8))
    claims = []
    artifacts = {}
    for index in range(count):
        category = draw(st.sampled_from(CATEGORIES))
        artifact_id = f"artifact-{index}"
        artifact_hash = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        artifacts[artifact_id] = {
            "path": f"figures/{artifact_id}.png",
            "sha256": artifact_hash,
            "report_hash": "a" * 64,
            "config_hash": "b" * 64,
        }
        claims.append({
            "claim_id": f"claim-{index}",
            "text": f"bounded generated claim {index}",
            "evidence_categories": (category,),
            "artifact_id": artifact_id,
            "adjacent": {
                "limitations": ("limited to the declared evidence scope",),
                "counts": {"seeds": 10, "samples": draw(st.integers(1, 1000))},
                "uncertainty": {
                    "method": INTERVAL_METHOD if category != "future pilot" else "not estimable: deferred",
                    "status": "reported" if category != "future pilot" else "not demonstrated",
                },
                "provenance": {
                    "report_hash": "a" * 64,
                    "config_hash": "b" * 64,
                },
            },
        })
    return claims, artifacts

@seed(HYPOTHESIS_SEED)
@given(claim_registries())
def test_property_12_generated_claims_have_one_category_and_adjacent_scope_metadata(registry):
    """Property 12. **Validates: Requirements 2.10, 3.9**"""
    reporting = _reporting()
    claims, artifacts = registry
    validated = reporting.validate_claim_artifact_registry(claims, artifacts)

    assert len(validated) == len(claims)
    for claim in validated:
        assert tuple(claim["evidence_categories"]) in tuple((item,) for item in CATEGORIES)
        assert claim["adjacent"]["limitations"]
        assert claim["adjacent"]["counts"]
        assert claim["adjacent"]["uncertainty"]
        assert claim["adjacent"]["provenance"]
        artifact = artifacts[claim["artifact_id"]]
        assert claim["adjacent"]["provenance"]["report_hash"] == artifact["report_hash"]
        assert claim["adjacent"]["provenance"]["config_hash"] == artifact["config_hash"]


@seed(HYPOTHESIS_SEED)
@given(
    registry=claim_registries(),
    defect=st.sampled_from((
        "no-category", "multiple-categories", "limitations", "counts",
        "uncertainty", "provenance",
    )),
)
def test_property_12_rejects_ambiguous_or_nonadjacent_claim_evidence(registry, defect):
    """Property 12. **Validates: Requirement 2.10**"""
    reporting = _reporting()
    claims, artifacts = copy.deepcopy(registry)
    if defect == "no-category":
        claims[0]["evidence_categories"] = ()
    elif defect == "multiple-categories":
        claims[0]["evidence_categories"] = CATEGORIES[:2]
    else:
        del claims[0]["adjacent"][defect]

    with pytest.raises(reporting.ReportValidationError):
        reporting.validate_claim_artifact_registry(claims, artifacts)


@st.composite
def outcome_matrices(draw):
    scenarios = draw(st.lists(
        st.sampled_from((
            "no-change", "abrupt-change", "gradual-coarticulated-change",
            "correlated-evidence", "miscalibration", "duration-hazard-mismatch",
        )),
        min_size=1,
        max_size=6,
        unique=True,
    ))
    predeclared = []
    outcomes = []
    for scenario in scenarios:
        for method in METHODS:
            hypothesis_id = f"{scenario}:{method}:primary"
            predeclared.append({
                "hypothesis_id": hypothesis_id,
                "scenario": scenario,
                "method": method,
                "applicable": True,
            })
            outcomes.append({
                "hypothesis_id": hypothesis_id,
                "scenario": scenario,
                "method": method,
                "status": draw(st.sampled_from(STATUSES)),
                "observed_direction": draw(st.sampled_from(DIRECTIONS)),
            })
    return predeclared, outcomes


@seed(HYPOTHESIS_SEED)
@given(outcome_matrices())
def test_property_14_generated_scenario_method_outcomes_are_retained_exactly_once(matrix):
    """Property 14. **Validates: Requirements 2.7, 3.7**"""
    reporting = _reporting()
    predeclared, outcomes = matrix
    retained = reporting.validate_outcome_retention(predeclared, outcomes)

    expected_keys = {
        (row["hypothesis_id"], row["scenario"], row["method"])
        for row in predeclared
        if row["applicable"]
    }
    retained_keys = [
        (row["hypothesis_id"], row["scenario"], row["method"])
        for row in retained
    ]
    assert set(retained_keys) == expected_keys
    assert len(retained_keys) == len(expected_keys)
    assert {row["method"] for row in retained} == set(METHODS)
    source = {
        (row["hypothesis_id"], row["scenario"], row["method"]): row
        for row in outcomes
    }
    for row in retained:
        key = (row["hypothesis_id"], row["scenario"], row["method"])
        assert row["status"] in STATUSES
        assert row["status"] == source[key]["status"]
        assert row["observed_direction"] == source[key]["observed_direction"]


@seed(HYPOTHESIS_SEED)
@given(
    matrix=outcome_matrices(),
    defect=st.sampled_from(("missing", "duplicate", "invalid-status")),
)
def test_property_14_rejects_selective_omission_duplicates_and_status_relabeling(
    matrix, defect
):
    """Property 14. **Validates: Requirements 2.7, 3.7**"""
    reporting = _reporting()
    predeclared, outcomes = copy.deepcopy(matrix)
    if defect == "missing":
        outcomes.pop()
    elif defect == "duplicate":
        outcomes.append(copy.deepcopy(outcomes[0]))
    else:
        outcomes[0]["status"] = "omitted-because-adverse"

    with pytest.raises(reporting.ReportValidationError):
        reporting.validate_outcome_retention(predeclared, outcomes)

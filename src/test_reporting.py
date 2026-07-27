"""Focused unit checks for versioned JSON-first reporting."""

import hashlib

import pytest

from protocol import content_hash
from reporting import (
    METHODS,
    ReportValidationError,
    build_reduced_report,
    load_report_json,
    validate_claim_artifact_registry,
    write_report_json,
)


def _fixture(count=3):
    rows = []
    for seed in range(1, count + 1):
        methods = {
            method: {
                "values": {"coverage": seed / 10},
                "counts": {"hits": seed, "eligible": 10},
            }
            for method in METHODS
        }
        rows.append({
            "seed": seed,
            "scenario": "abrupt-change",
            "ordered_test_stream_ids": (f"test-{seed}",),
            "methods": methods,
        })
    config = {"profile": "reduced", "methods": METHODS}
    provenance = {
        "schema_version": "1",
        "code_revision": "revision",
        "environment_id": "cpython-3.11-fixture",
        "metric_definition_version": "metrics-v1",
        "freeze_timestamp": "2025-01-01T00:00:00+00:00",
        "complete_config": config,
        "config_hash": content_hash(config),
        "predeclared_evaluation_seeds": tuple(range(1, count + 1)),
        "role_stream_ids": {
            "fit": ("fit",), "development": ("development",),
            "stress": ("stress",),
            "test": tuple(f"test-{seed}" for seed in range(1, count + 1)),
        },
        "scenarios": ("abrupt-change",),
        "methods": METHODS,
        "numeric_rerun_tolerance": 1e-6,
    }
    effects = ({
        "effect_id": "M2-minus-M0", "left_method": "M2",
        "right_method": "M0", "metric": "coverage", "units": "proportion",
    },)
    return rows, provenance, effects


def test_reduced_report_is_immutable_non_headline_and_json_round_trips(tmp_path):
    rows, provenance, effects = _fixture()
    report = build_reduced_report(
        rows, provenance=provenance, paired_effects=effects, bootstrap_seed=7
    )
    assert report["profile"] == "non_headline"
    assert not report["artifacts"]
    with pytest.raises(TypeError):
        report["provenance"]["config_hash"] = "changed"

    path = write_report_json(report, tmp_path / "report.json")
    loaded = load_report_json(path)
    assert loaded["report_hash"] == report["report_hash"]
    assert loaded["paired_effects"] == report["paired_effects"]


def test_claim_registry_rejects_unsupported_superiority_language():
    digest = hashlib.sha256(b"artifact").hexdigest()
    artifacts = {"plot": {
        "path": "figures/plot.png", "sha256": digest,
        "report_hash": "a" * 64, "config_hash": "b" * 64,
    }}
    claims = [{
        "claim_id": "claim", "text": "M2 is superior to M0",
        "evidence_categories": ("synthetic evidence",),
        "artifact_id": "plot",
        "adjacent": {
            "limitations": ("synthetic only",), "counts": {"seeds": 10},
            "uncertainty": {"status": "reported", "direction_supported": False},
            "provenance": {"report_hash": "a" * 64, "config_hash": "b" * 64},
        },
    }]
    with pytest.raises(ReportValidationError, match="superiority"):
        validate_claim_artifact_registry(claims, artifacts)

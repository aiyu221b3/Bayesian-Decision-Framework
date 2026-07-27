"""Exploration property for the sequential-Bayes remediation bug condition."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from changepoint import BOCPD
from evidence import Simcon, concat_stream
from metrics import emission_metrics, pool
from policies import BayesThresh, BocpdThresh, RawThresh
from stopping import evaluate, fit_dp
from tree_belief import TreeBel
from test_support import NUMERIC_TOLERANCE


ROOT = Path(__file__).resolve().parents[1]
COUNTEREXAMPLES = (
    ROOT / ".kiro" / "specs" / "sequential-bayes-remediation"
    / "bug_condition_counterexamples.json"
)
ATOL = NUMERIC_TOLERANCE


def _entry(rows, clause, fixture, expected, observed, corrected):
    rows.append({
        "clause": clause,
        "fixture": fixture,
        "expected": expected,
        "observed": observed,
        "status": "corrected_on_current_code" if corrected else "bug_reproduced",
    })


def _bounded_bocpd_cases():
    """Deterministic bounded generator used because the project has no PBT dependency."""
    for classes in range(2, 6):
        for hazard in (0.02, 0.05, 0.2):
            for prefix_length in (2, 7, 15):
                yield classes, hazard, prefix_length


def test_property_1_complete_sequential_bayes_remediation():
    """**Validates: Requirements 1.1-1.11, 2.1-2.11**"""
    rows = []

    # 1.1: generated posterior invariants plus stable/surprising and two-threshold probes.
    generated_ok = True
    for classes, hazard, prefix_length in _bounded_bocpd_cases():
        detector = BOCPD(classes, hazard=hazard, alpha=1.0)
        trace = [0] * prefix_length + [classes - 1, 0]
        for observation in trace:
            posterior = detector.update(observation).posterior
            generated_ok &= bool(
                np.all(np.isfinite(posterior))
                and np.all(posterior >= 0)
                and np.isclose(posterior.sum(), 1.0, atol=ATOL)
            )

    stable = BOCPD(3, hazard=0.05)
    surprising = BOCPD(3, hazard=0.05)
    for _ in range(15):
        stable.update(0)
        surprising.update(0)
    stable_score = stable.update(0).cp_posterior
    surprising_score = surprising.update(1).cp_posterior
    low_threshold = (stable_score + surprising_score) / 2
    high_threshold = (surprising_score + 1.0) / 2
    trace = np.vstack([
        np.tile([0.98, 0.01, 0.01], (15, 1)),
        [0.01, 0.98, 0.01],
    ])
    low_policy = BocpdThresh(3, cp_thr=low_threshold, hazard=0.05)
    high_policy = BocpdThresh(3, cp_thr=high_threshold, hazard=0.05)
    low_policy.run(trace)
    high_policy.run(trace)
    boundary_ok = (
        generated_ok
        and surprising_score > stable_score
        and low_policy.bd == [15]
        and high_policy.bd == []
    )
    _entry(
        rows, "1.1", "15 stable class-0 observations then class 1; thresholds straddle score",
        "normalized evidence-sensitive posterior; only lower cp_thr accepts frame 15",
        {"stable_score": stable_score, "surprising_score": surprising_score,
         "low_threshold_boundaries": low_policy.bd,
         "high_threshold_boundaries": high_policy.bd},
        boundary_ok,
    )

    # 1.2: generated permanent-high traces must be one event for M0-MT,
    # regardless of refractory duration.
    high2 = np.tile([0.99, 0.01], (40, 1))
    similarity = np.array([
        [1.0, 0.8, 0.1, 0.1], [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8], [0.1, 0.1, 0.8, 1.0],
    ])
    high4 = np.tile([0.97, 0.01, 0.01, 0.01], (40, 1))
    emission_counts = {}
    for refr in (0, 1, 3, 8):
        runs = {
            "M0": RawThresh(2, theta=0.8, refr=refr).run(high2),
            "M1": BayesThresh(2, lam=1.0, rho=0.0, tau=0.8, refr=refr).run(high2),
            "M2": BayesThresh(2, lam=1.0, rho=0.05, tau=0.8, refr=refr).run(high2),
            "M3": BocpdThresh(2, lam=1.0, tau=0.8, refr=refr).run(high2),
            "MT": TreeBel(4, similarity, lam=1.0, rho=0.05,
                          tau=0.8, refr=refr).run(high4),
        }
        emission_counts[str(refr)] = {name: len(value) for name, value in runs.items()}
    emitter_ok = all(count == 1 for row in emission_counts.values()
                     for count in row.values())
    _entry(rows, "1.2", "40 permanently-high frames; M0-MT; refr in {0,1,3,8}",
           "each method emits exactly once and changing only refr creates no event",
           {"emission_counts_by_refractory": emission_counts}, emitter_ok)

    # 1.3: exact empty/singleton/hand-counted schemas and pooled rates.
    z = np.array([0, 0, 1, 1])
    empty = emission_metrics([], z, [0, 1], [0, 2], K=2)
    singleton = emission_metrics([(0, 0)], z, [0, 1], [0, 2], K=2)
    repeat = emission_metrics([(0, 0), (1, 0), (2, 1)], np.array([0, 0, 1]),
                              [0, 1], [0, 2], K=2)
    perseveration = emission_metrics(
        [(3, 0), (4, 1), (5, 1)], np.array([0, 0, 0, 1, 1, 1]),
        [0, 1], [0, 3], K=2,
    )
    emission_error = emission_metrics(
        [(0, 0), (1, 0), (2, 0), (3, 1)], z, [0, 1], [0, 2], K=2,
    )
    first = emission_metrics([(0, 0), (1, 0)], np.array([0, 0]), [0], [0], K=4)
    second = emission_metrics([(0, 0), (1, 1), (2, 2), (3, 3)],
                              np.array([0, 1, 2, 3]), [0, 1, 2, 3],
                              [0, 1, 2, 3], K=4)
    pooled = pool([first, second])
    same_schema = all(set(item) == set(empty) for item in
                      (singleton, repeat, perseveration, emission_error))
    metric_ok = (same_schema and empty["rep_n"] == empty["rep_d"] == 0
                 and math.isnan(empty["rep_r"])
                 and singleton["rep_d"] == 0 and math.isnan(singleton["rep_r"])
                 and (repeat["rep_n"], repeat["rep_d"], repeat["rep_r"]) == (1, 2, 0.5)
                 and (perseveration["per_n"], perseveration["per_d"]) == (1, 3)
                 and perseveration["per_r"] == 1 / 3
                 and (emission_error["err_n"], emission_error["err_d"],
                      emission_error["err_r"]) == (1, 4, 0.25)
                 and (pooled["rep_n"], pooled["rep_d"], pooled["rep_r"]) == (1, 4, 0.25))
    _entry(rows, "1.3", "empty, singleton, repeat 1/2, persev 1/3, error 1/4, pooled 1/4",
           "stable schema; explicit counts; zero-denominator NaN; count-pooled rates",
           {"same_schema": same_schema, "empty_repeat": empty["rep_r"],
            "singleton_repeat_denominator": singleton["rep_d"],
            "repeat_counts_rate": [repeat["rep_n"], repeat["rep_d"], repeat["rep_r"]],
            "perseveration_counts_rate": [perseveration["per_n"], perseveration["per_d"],
                                           perseveration["per_r"]],
            "error_counts_rate": [emission_error["err_n"], emission_error["err_d"],
                                   emission_error["err_r"]],
            "pooled_repeat_counts_rate": [pooled["rep_n"], pooled["rep_d"], pooled["rep_r"]]},
           metric_ok)

    # 1.4: every noninitial boundary is retained as a hit or censored miss,
    # and replacing a hit with a miss cannot improve the primary result.
    z15 = np.array([0] * 5 + [1] * 5 + [2] * 5)
    censored = emission_metrics([(7, 1)], z15, [0, 1, 2], [0, 5, 10], K=3)
    all_missed = emission_metrics([], z15, [0, 1, 2], [0, 5, 10], K=3)
    expected_censored = {
        "bnd_n": 2, "hit_n": 1, "miss_n": 1, "cov": 0.5,
        "hdel": 2.0, "rdel": 3.5,
    }
    censored_ok = (all(censored[key] == value for key, value in expected_censored.items())
                   and all_missed["cov"] <= censored["cov"]
                   and all_missed["rdel"] >= censored["rdel"])
    _entry(rows, "1.4", "15 frames; onsets [0,5,10]; hit at 7; final miss",
           "eligible=2, hits=1, misses=1, coverage=1/2, hit delay=2, restricted delay=3.5; miss is no better",
           {**{key: censored[key] for key in expected_censored},
            "all_missed_coverage": all_missed["cov"],
            "all_missed_restricted_delay": all_missed["rdel"]}, censored_ok)

    # 1.5: leaf/family/root/disjoint fixtures retain hierarchy meaning. A
    # derived hierarchy row must keep exact node identity and emission time.
    def hierarchy_score(nd, truth=0, time=1):
        return emission_metrics([(time, frozenset(nd))], np.array([truth, truth]),
                                [truth], [0], K=4)

    leaf_case = hierarchy_score({0})
    family_case = hierarchy_score({0, 1})
    root_case = hierarchy_score({0, 1, 2, 3})
    disjoint_case = hierarchy_score({1, 2})
    permutation_case = hierarchy_score({2, 3}, truth=3)
    hierarchy_rows = family_case.get("hierarchy_rows")
    exact_row_ok = False
    if isinstance(hierarchy_rows, list) and len(hierarchy_rows) == 1:
        row = hierarchy_rows[0]
        try:
            exact_row_ok = (int(row["time"]) == 1
                            and frozenset(row["node"]) == frozenset({0, 1}))
        except (KeyError, TypeError, ValueError):
            exact_row_ok = False
    permutation_ok = all(family_case[key] == permutation_case[key]
                         for key in ("hcov_r", "card", "spec"))
    hierarchy_ok = (
        (leaf_case["hcov_r"], leaf_case["card"]) == (1, 1)
        and (family_case["hcov_r"], family_case["card"]) == (1, 2)
        and (root_case["hcov_r"], root_case["card"]) == (1, 4)
        and (disjoint_case["hcov_r"], disjoint_case["card"]) == (0, 2)
        and not family_case["flat_ok"] and math.isnan(family_case["per_r"])
        and math.isnan(family_case["seq_r"])
        and permutation_ok and exact_row_ok
    )
    _entry(rows, "1.5", "leaf {0}, family {0,1}, root, disjoint {1,2}, permuted family {2,3}",
           "coverage/cardinality/specificity are permutation-invariant; nonleaf flat metrics are N/A; exact node and time are retained",
           {"leaf_coverage_cardinality": [leaf_case["hcov_r"], leaf_case["card"]],
            "family_coverage_cardinality": [family_case["hcov_r"], family_case["card"]],
            "root_coverage_cardinality": [root_case["hcov_r"], root_case["card"]],
            "disjoint_coverage_cardinality": [disjoint_case["hcov_r"], disjoint_case["card"]],
            "nonleaf_perseveration": family_case["per_r"],
            "nonleaf_sequence": family_case["seq_r"],
            "permutation_invariant": permutation_ok,
            "hierarchy_rows_present": hierarchy_rows is not None,
            "exact_node_and_time_retained": exact_row_ok}, hierarchy_ok)

    # 1.6: deliberate fit/test ID overlap must abort before scoring.
    shared = [{"stream_id": "same", "bmax": np.array([0.4, 0.9]),
               "pred": np.array([0, 1]), "truth": 1}]
    overlap_error = None
    overlap_score = None
    try:
        fitted = fit_dp(shared, cw=0.05, ce=1.0, nb=3)
        overlap_score = evaluate(shared, fitted, cw=0.05, ce=1.0)
    except Exception as error:  # approved behavior is a specific protocol rejection
        overlap_error = f"{type(error).__name__}: {error}"
    overlap_ok = overlap_error is not None and any(
        word in overlap_error.lower() for word in ("overlap", "protocol", "role")
    )
    _entry(rows, "1.6", "fit IDs=['same']; test IDs=['same']",
           "abort before scoring with a specific overlap/protocol error",
           {"error": overlap_error, "score_was_produced": overlap_score is not None,
            "score": overlap_score}, overlap_ok)

    # 1.7: generated streams require an auditable role manifest and a policy
    # view that raises a protocol error on simulator-secret access.
    generated = concat_stream(Simcon(K=4, fami=2, dur_r=(4, 6), blend=1),
                              np.random.default_rng(7), n_signs=2)
    required_manifest = {"stream_id", "role", "scenario", "seed",
                         "policy_visible", "simulator_only"}
    missing_manifest = sorted(required_manifest - set(generated))
    secret_access_error = None
    secret_was_read = False
    try:
        policy_view = generated.get("policy_visible", generated)
        if isinstance(policy_view, dict):
            _ = policy_view["z"]
        else:
            _ = getattr(policy_view, "truth")
        secret_was_read = True
    except Exception as error:
        secret_access_error = f"{type(error).__name__}: {error}"
    secret_guarded = (secret_access_error is not None and any(
        word in secret_access_error.lower() for word in ("secret", "protocol", "simulator")
    ))
    manifest_ok = (not missing_manifest and secret_guarded
                   and set(generated["policy_visible"]) <= {"stream_id", "p"})
    _entry(rows, "1.7", "generated two-sign stream at seed 7 plus attempted truth read",
           "manifest records ID/role/scenario/seed; policy view exposes only ID/posteriors and rejects simulator secrets",
           {"keys": sorted(generated), "missing_manifest_fields": missing_manifest,
            "secret_was_read": secret_was_read, "secret_access_error": secret_access_error},
           manifest_ok)

    # 1.8: current headline artifact requires complete paired-effect provenance.
    report_path = ROOT / "results" / "full_synthetic_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = report.get("meta", {})
    required_meta = {"schema_version", "code_revision", "environment_id", "config_hash",
                     "split_stream_ids", "rerun_tolerance", "artifact_paths",
                     "metric_definition_version"}
    missing_meta = sorted(required_meta - set(meta))
    has_paired_effect = "paired_effects" in report.get("summary", {})
    report_ok = len(meta.get("seeds", [])) >= 10 and not missing_meta and has_paired_effect
    _entry(rows, "1.8", "results/full_synthetic_report.json",
           "at least ten paired seeds plus revision/environment/splits/hash/tolerance/artifacts and paired effects",
           {"seed_count": len(meta.get("seeds", [])), "missing_meta": missing_meta,
            "paired_effects_present": has_paired_effect}, report_ok)

    # 1.9: the supported clean environment must be constrained and the README
    # must run the complete noninteractive suite, not one legacy test file.
    environment_files = [ROOT / "requirements.txt", ROOT / "pyproject.toml",
                         ROOT / "environment.yml", ROOT / "environment.yaml"]
    existing_environment = [path.name for path in environment_files if path.exists()]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    full_test_command = ("pytest -q" in readme or "pytest src -q" in readme
                         or "pytest src/ -q" in readme)
    environment_ok = bool(existing_environment) and full_test_command
    _entry(rows, "1.9", "clean Python 3.11 checkout and README supported commands",
           "version-constrained environment exists and README runs the complete noninteractive suite",
           {"environment_specifications": existing_environment,
            "complete_test_command_documented": full_test_command}, environment_ok)

    # 1.10: current claims must not promote a future pilot or describe the
    # empirical comparator as RL, a universal/global optimum, or exact DP.
    design = (ROOT / "SEQUENTIAL_BAYES_DESIGN.md").read_text(encoding="utf-8")
    combined_docs = readme + "\n" + design
    unsupported_snippets = [snippet for snippet in (
        "method built and validated", "DP-opt risk", "DP optimum", "DP-optimum",
        "globally optimal", "global optimum", "reinforcement-learning comparator",
        "RL comparator", "demonstrated on synthetic and real streams",
        "Validated on synthetic streams **and**",
    ) if snippet.lower() in combined_docs.lower()]
    comparator_label = meta.get("dp_label")
    claims_ok = (not unsupported_snippets and comparator_label ==
                 "empirical finite-horizon dynamic-programming approximation")
    _entry(rows, "1.10", "README.md, SEQUENTIAL_BAYES_DESIGN.md, and current report",
           "future pilot is not demonstrated; comparator is empirical finite-horizon DP approximation, never RL or globally optimal",
           {"unsupported_snippets": unsupported_snippets,
            "report_comparator_label": comparator_label}, claims_ok)

    # 1.11: invoking real evaluation without artifacts must stop at a deferred boundary.
    invocation = subprocess.run(
        [sys.executable, str(ROOT / "src" / "real_eval.py"),
         "__missing_cache__", "__missing_model__"],
        cwd=ROOT, text=True, capture_output=True, timeout=20, check=False,
    )
    invocation_text = (invocation.stdout + invocation.stderr).strip()
    deferred_ok = (invocation.returncode != 0 and "deferred" in invocation_text.lower()
                   and "traceback" not in invocation_text.lower())
    _entry(rows, "1.11", "python src/real_eval.py __missing_cache__ __missing_model__",
           "fail before optional imports/artifact access with explicit deferred status and no metric",
           {"returncode": invocation.returncode,
            "output_tail": invocation_text[-500:]}, deferred_ok)

    payload = {
        "property": "Property 1: Bug Condition - Complete Sequential Bayes Remediation",
        "validates": [f"1.{index}" for index in range(1, 12)],
        "outcomes": rows,
        "failing_clauses": [row["clause"] for row in rows if row["status"] == "bug_reproduced"],
    }
    COUNTEREXAMPLES.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n",
                               encoding="utf-8")

    failures = [row for row in rows if row["status"] == "bug_reproduced"]
    assert not failures, "Minimized bug-condition counterexamples:\n" + json.dumps(
        failures, indent=2, allow_nan=True
    )

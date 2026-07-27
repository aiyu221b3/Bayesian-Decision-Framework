"""Observation-first preservation tests for supported pre-remediation behavior.

Property 2: Preservation - Supported Behavior Outside the Bug Condition.
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9**
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from belief import FlatBel
from evidence import (
    Simcon,
    _ar1_noise,
    _label_seq,
    _mean_logits,
    build_sim,
    concat_stream,
)
from experiments import mk_pols
from metrics import emission_metrics, levenshtein
from policies import BayesThresh
from test_support import NUMERIC_TOLERANCE
from tree_belief import specificity


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".kiro" / "specs" / "sequential-bayes-remediation"
CONFIG = json.loads((SPEC / "preservation_test_config.json").read_text())
GOLDEN = json.loads((SPEC / "preservation_golden.json").read_text())
ATOL = NUMERIC_TOLERANCE


def _assert_probability_rows(values):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.allclose(values.sum(axis=-1), 1.0, atol=ATOL)


def _serialise_emissions(emissions):
    return [
        [time, sorted(identity) if isinstance(identity, frozenset) else identity]
        for time, identity in emissions
    ]


def test_golden_belief_trajectories_replay_with_central_tolerance():
    """Fixed observations preserve static and switching numerical trajectories."""
    posterior = np.asarray(GOLDEN["belief_input"], dtype=float)
    expected = GOLDEN["belief_trajectories"]

    static = FlatBel(4, lam=0.5, rho=0.0).run(posterior)
    switching = FlatBel(4, lam=0.5, rho=0.05).run(posterior)

    assert np.allclose(static, expected["static"], atol=ATOL)
    assert np.allclose(switching, expected["switching"], atol=ATOL)


def test_golden_generator_output_replays_exact_structure_and_numbers():
    """The captured configuration and seed reproduce every generator field."""
    cfg = Simcon(**GOLDEN["generator_config"])
    observed = concat_stream(
        cfg,
        np.random.default_rng(CONFIG["seeds"]["generator"]),
        n_signs=3,
    )
    expected = GOLDEN["generator_output"]

    assert np.allclose(observed["p"], expected["p"], atol=ATOL)
    assert np.array_equal(observed["z"], expected["z"])
    assert observed["labels"] == expected["labels"]
    assert observed["durss"] == expected.get("durss", expected.get("durations"))
    assert observed["onsets"] == expected["onsets"]
    assert observed["bounds"] == expected["bounds"]
    assert np.array_equal(observed["families"], expected["families"])


def test_golden_first_qualifying_emission_time_and_identity_for_m0_through_mt():
    """One genuine crossing preserves its first exact flat label or tree node."""
    posterior = np.asarray(GOLDEN["belief_input"], dtype=float)
    similarity = np.array([
        [1.0, 0.8, 0.1, 0.1],
        [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ])
    policies = mk_pols(Simcon(K=4, fami=2), similarity)
    observed = {
        name: _serialise_emissions(policy.run(posterior))
        for name, policy in policies.items()
    }

    assert sorted(policies) == CONFIG["core_methods"]
    assert observed == GOLDEN["first_crossing_emissions"]


def test_golden_flat_correctness_and_unit_cost_levenshtein_examples():
    """Ordinary flat correctness and insertion/deletion/substitution costs stay one."""
    metrics = emission_metrics(
        [(0, 0), (1, 1)], np.array([0, 0]), [0], [0], K=4
    )
    expected = GOLDEN["flat_examples"]

    assert (metrics["err_n"], metrics["err_d"], metrics["err_r"]) == (
        expected["err_n"], expected["err_d"], expected["err_r"]
    )
    observed_edit = {
        "equal": levenshtein([0, 1], [0, 1]),
        "insert": levenshtein([0, 2], [0, 1, 2]),
        "delete": levenshtein([0, 1, 2], [0, 2]),
        "substitute": levenshtein([0, 1, 2], [0, 3, 2]),
    }
    assert observed_edit == expected["levenshtein"]


def test_golden_hierarchy_membership_and_specificity_only():
    """Preserve supported membership/order, not the known missing identity-row behavior."""
    metrics = emission_metrics(
        [(0, frozenset({0, 1}))], np.array([1, 1]), [1], [0], K=4
    )
    expected = GOLDEN["hierarchy_examples"]

    assert metrics["hcov_r"] == expected["family_coverage"]
    assert metrics["card"] == expected["family_cardinality"]
    assert np.isclose(metrics["spec"], expected["family_specificity"], atol=ATOL)
    assert specificity(frozenset({0}), 4) == expected["leaf_specificity"]
    assert specificity(frozenset(range(4)), 4) == expected["root_specificity"]
    assert expected["leaf_specificity"] > expected["family_specificity"] > expected["root_specificity"]


def _posterior_cases():
    """Bounded valid posterior generator with deterministic, persisted seeds."""
    for seed in CONFIG["seeds"]["properties"]:
        for classes, horizon in ((2, 1), (3, 9), (6, 17)):
            rng = np.random.default_rng(seed + classes + horizon)
            values = rng.dirichlet(np.linspace(0.5, 1.5, classes), size=horizon)
            if horizon > 1:
                values[0] = 0.0
                values[0, seed % classes] = 1.0
            yield seed, classes, values


def test_property_valid_posteriors_are_causal_finite_normalized_and_rho_zero_equivalent():
    """Property 2 refinement. **Validates: Requirements 3.1, 3.2**"""
    for _, classes, posterior in _posterior_cases():
        for lam in (0.0, 0.37, 1.0):
            for rho in (0.0, 0.08, 1.0):
                full = FlatBel(classes, lam=lam, rho=rho).run(posterior)
                _assert_probability_rows(full)

                cuts = sorted({1, len(posterior) // 2, len(posterior)})
                for cut in cuts:
                    prefix = FlatBel(classes, lam=lam, rho=rho).run(posterior[:cut])
                    assert np.allclose(prefix, full[:cut], atol=ATOL)

                if len(posterior) > 1:
                    changed_future = posterior.copy()
                    changed_future[-1] = np.roll(changed_future[-1], 1)
                    changed = FlatBel(classes, lam=lam, rho=rho).run(changed_future)
                    assert np.allclose(changed[:-1], full[:-1], atol=ATOL)

            m1 = FlatBel(classes, lam=lam, rho=0.0).run(posterior)
            m2_at_zero = FlatBel(classes, lam=lam, rho=0.0).run(posterior)
            assert np.allclose(m1, m2_at_zero, atol=ATOL)

            m1_policy = BayesThresh(classes, lam=lam, rho=0.0, tau=0.8, refr=0)
            m2_at_zero_policy = BayesThresh(
                classes, lam=lam, rho=0.0, tau=0.8, refr=0
            )
            assert m1_policy.run(posterior) == m2_at_zero_policy.run(posterior)


def test_property_positive_rho_uses_the_documented_causal_forgetting_update():
    """Property 2 refinement. **Validates: Requirement 3.2**"""
    for _, classes, posterior in _posterior_cases():
        lam = 0.43
        rho = 0.12
        model = FlatBel(classes, lam=lam, rho=rho)
        previous = np.full(classes, 1.0 / classes)

        for observation in posterior:
            predicted = (1.0 - rho) * previous + rho / classes
            expected = predicted * np.power(np.clip(observation, 1e-12, None), lam)
            expected /= expected.sum()
            observed = model.update(observation)
            assert np.allclose(observed, expected, atol=ATOL)
            previous = observed


def _simulator_cases():
    """Bounded valid configuration generator spanning correlation/blend settings."""
    shapes = (
        (3, 1, -0.4, 0),
        (4, 2, 0.0, 2),
        (6, 3, 0.75, 3),
    )
    for seed in CONFIG["seeds"]["properties"]:
        for classes, families, corr, blend in shapes:
            yield seed, Simcon(
                K=classes,
                fami=families,
                sig=3.8,
                gamma=1.3,
                corr=corr,
                noise=0.3,
                temp=0.9,
                dur_r=(5, 9),
                blend=blend,
            )


def test_property_generator_replay_normalization_labels_lengths_onsets_and_bounds():
    """Property 2 refinement. **Validates: Requirement 3.3**"""
    for seed, cfg in _simulator_cases():
        first = concat_stream(cfg, np.random.default_rng(seed), n_signs=4)
        second = concat_stream(cfg, np.random.default_rng(seed), n_signs=4)

        assert np.array_equal(first["p"], second["p"])
        assert np.array_equal(first["z"], second["z"])
        assert first["labels"] == second["labels"]
        assert first["durss"] == second["durss"]
        assert first["onsets"] == second["onsets"]
        assert first["bounds"] == second["bounds"]
        assert np.array_equal(first["families"], second["families"])

        _assert_probability_rows(first["p"])
        assert np.all(first["p"] <= 1.0)
        assert all(a != b for a, b in zip(first["labels"], first["labels"][1:]))
        assert len(first["z"]) == len(first["p"]) == sum(first["durss"])
        assert first["onsets"] == [0] + list(np.cumsum(first["durss"])[:-1])
        assert first["bounds"] == first["onsets"][1:]
        for label, duration, onset in zip(
            first["labels"], first["durss"], first["onsets"]
        ):
            assert np.all(first["z"][onset:onset + duration] == label)


def test_property_ar1_recurrence_onset_ramps_and_configured_coarticulation():
    """Property 2 refinement. **Validates: Requirement 3.3**"""
    for seed, cfg in _simulator_cases():
        actual_noise = _ar1_noise(
            np.random.default_rng(seed), cfg.corr, cfg.noise, 12, cfg.K
        )
        replay_rng = np.random.default_rng(seed)
        expected_noise = np.zeros_like(actual_noise)
        expected_noise[0] = replay_rng.normal(0.0, cfg.noise, cfg.K)
        scale = np.sqrt(1.0 - cfg.corr**2)
        for frame in range(1, len(expected_noise)):
            epsilon_n = replay_rng.normal(0.0, cfg.noise, cfg.K)
            expected_noise[frame] = (
                cfg.corr * expected_noise[frame - 1] + scale * epsilon_n
            )
        assert np.allclose(actual_noise, expected_noise, atol=ATOL)

        rng = np.random.default_rng(seed)
        similarity, _ = build_sim(cfg, rng)
        truth, labels, durss, onsets, bounds = _label_seq(cfg, 3, rng)
        means = _mean_logits(cfg, similarity, truth, labels, durss, onsets)

        first_label = labels[0]
        first_curve = means[:durss[0], first_label]
        assert np.isclose(first_curve[0], 0.0, atol=ATOL)
        assert np.isclose(first_curve[-1], cfg.sig, atol=ATOL)
        assert np.all(np.diff(first_curve) >= -ATOL)

        for boundary in bounds:
            if cfg.blend > 0:
                assert np.allclose(means[boundary], means[boundary - 1], atol=ATOL)
            else:
                assert np.allclose(means[boundary], 0.0, atol=ATOL)


def test_supported_synthetic_smoke_has_all_methods_without_real_data_or_network():
    """Synthetic execution remains independent. **Validates: Requirements 3.7, 3.8**"""
    code = r'''
import builtins
import socket
import sys
import numpy as np

forbidden = {"torch", "classifier", "practice", "real_eval"}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError("forbidden synthetic dependency: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

def blocked_network(*args, **kwargs):
    raise AssertionError("synthetic smoke attempted network access")
socket.socket = blocked_network

from evidence import Simcon
from experiments import mk_pols

similarity = np.array([
    [1.0, 0.8, 0.1, 0.1], [0.8, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.8], [0.1, 0.1, 0.8, 1.0],
])
posterior = np.array([
    [0.25, 0.25, 0.25, 0.25], [0.55, 0.15, 0.15, 0.15],
    [0.90, 0.04, 0.03, 0.03],
])
policies = mk_pols(Simcon(K=4, fami=2), similarity)
assert sorted(policies) == ["M0", "M1", "M2", "M3", "MT"]
for policy in policies.values():
    policy.run(posterior)
assert forbidden.isdisjoint(sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / "src",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_beginner_friendly_explanatory_scope_is_preserved_without_claim_oracles():
    """Positive educational scope only. **Validates: Requirement 3.9**"""
    explainer = (ROOT / "THEORY_EXPLAINER.tex").read_text(encoding="utf-8")
    lower = " ".join(explainer.lower().split())

    for concept in CONFIG["educational_concepts"]:
        assert concept.lower() in lower
    for marker in CONFIG["pedagogical_markers"]:
        assert marker.lower() in lower

    assert "theory (from the ground up)" in lower
    assert "python techniques used" in lower
    assert "every file below is one gear in that machine" in lower


def test_preservation_oracle_records_all_known_exclusions():
    """The golden baseline cannot accidentally bless remaining defects."""
    exclusions = " ".join(GOLDEN["excluded_oracles"]).lower()
    for required in (
        "repeats", "rates", "misses", "identity", "leakage",
        "artifacts", "real-data", "claims",
    ):
        assert required in exclusions

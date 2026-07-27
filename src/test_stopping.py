import copy

import numpy as np
import pytest

from stopping import (
    DPPol,
    best_fixed_threshold,
    calibration,
    evaluate,
    fit_dp,
    myopic_tau,
    threshold_decide,
)


def mk(bm, pr, y):
    return {
        "bmax": np.asarray(bm, dtype=float),
        "pred": np.asarray(pr, dtype=int),
        "truth": int(y),
    }


def test_myopic_tau():
    assert myopic_tau(0.1, 1.0) == pytest.approx(0.9)
    assert myopic_tau(0.02, 1.0) == pytest.approx(0.98)


def test_eval_uses_first_crossing():
    ss = [
        mk(
            [0.4, 0.8, 0.9],
            [0, 1, 1],
            1,
        )
    ]

    m = evaluate(
        ss,
        threshold_decide(0.7),
        cw=0.1,
        ce=1.0,
    )

    assert m["acc"] == 1.0
    assert m["latency"] == 1.0
    assert m["risk"] == pytest.approx(0.1)


def test_no_crossing_commits_at_final_frame():
    ss = [
        mk(
            [0.2, 0.3, 0.4],
            [0, 0, 1],
            1,
        )
    ]

    m = evaluate(
        ss,
        threshold_decide(0.9),
        cw=0.1,
        ce=1.0,
    )

    assert m["acc"] == 1.0
    assert m["latency"] == 2.0
    assert m["risk"] == pytest.approx(0.2)


def test_fixed_threshold_comes_from_grid():
    dev = [
        mk([0.4, 0.8], [0, 1], 1),
        mk([0.3, 0.9], [1, 1], 1),
    ]

    grid = np.array([0.5, 0.7, 0.85])

    tau, m = best_fixed_threshold(
        dev,
        cw=0.05,
        ce=1.0,
        grid=grid,
    )

    assert tau in grid
    assert set(m) == {"risk", "acc", "latency", "n"}


def test_dp_returns_frozen_policy_shape():
    fit = [
        mk([0.3, 0.6, 0.9], [0, 1, 1], 1),
        mk([0.2, 0.7, 0.8], [1, 1, 1], 1),
        mk([0.4, 0.5, 0.95], [0, 0, 0], 0),
    ]

    pol = fit_dp(
        fit,
        cw=0.05,
        ce=1.0,
        nb=10,
    )

    assert isinstance(pol, DPPol)
    assert pol.reg.shape == (3, 10)
    assert pol.edges.shape == (11,)


def test_dp_fit_does_not_mutate_streams():
    fit = [
        mk([0.3, 0.6, 0.9], [0, 1, 1], 1),
        mk([0.2, 0.7, 0.8], [1, 1, 1], 1),
    ]

    old = copy.deepcopy(fit)

    fit_dp(
        fit,
        cw=0.05,
        ce=1.0,
        nb=8,
    )

    for a, b in zip(fit, old):
        assert np.array_equal(a["bmax"], b["bmax"])
        assert np.array_equal(a["pred"], b["pred"])
        assert a["truth"] == b["truth"]


def test_dp_can_evaluate_new_test_streams():
    fit = [
        mk([0.2, 0.7, 0.9], [0, 1, 1], 1),
        mk([0.3, 0.8, 0.95], [1, 1, 1], 1),
    ]

    test = [
        mk([0.4, 0.85], [0, 1], 1),
    ]

    pol = fit_dp(
        fit,
        cw=0.05,
        ce=1.0,
        nb=8,
    )

    m = evaluate(
        test,
        pol,
        cw=0.05,
        ce=1.0,
    )

    assert m["n"] == 1
    assert 0 <= m["acc"] <= 1
    assert m["latency"] >= 0


def test_calibration_counts_every_frame():
    ss = [
        mk([0.2, 0.8], [0, 1], 1),
        mk([0.4, 0.9, 0.95], [0, 1, 1], 1),
    ]

    ctr, acc, tot = calibration(ss, nb=5)

    assert len(ctr) == 5
    assert len(acc) == 5
    assert tot.sum() == 5


def protocol_mk(role, stream_id, bm, pr, y, seed=7):
    from protocol import StreamManifestEntry

    manifest = StreamManifestEntry(
        stream_id=stream_id,
        role=role,
        scenario="no-change",
        seed=seed,
        within_seed_index=0,
        policy_visible_field_names=("stream_id", "posteriors"),
        simulator_only_field_names=("truth",),
        generator_config_hash="a" * 64,
        data_hash="b" * 64,
    )
    return {
        "stream_id": stream_id,
        "role": role,
        "manifest": manifest,
        "bmax": np.asarray(bm, dtype=float),
        "pred": np.asarray(pr, dtype=int),
        "truth": int(y),
    }


def frozen_config(**updates):
    config = {
        "wait_cost": 0.05,
        "error_cost": 1.0,
        "bin_count": 5,
        "empty_bin_fallback": {
            "accuracy": "bin_center",
            "transition": "self",
        },
        "minimum_bin_visits": 2,
        "minimum_transition_visits": 2,
        "accuracy_prior_strength": 1.0,
    }
    config.update(updates)
    return config


def test_separated_stopping_apis_are_serializable_diagnostic_and_paired():
    import json
    from stopping import (
        COMPARATOR_LABEL,
        evaluate_stopping,
        fit_empirical_dp,
        freeze_stopping_policies,
        select_fixed_threshold,
    )

    fit = [
        protocol_mk("fit", "fit-1", [0.2, 0.7, 0.9], [0, 1, 1], 1),
        protocol_mk("fit", "fit-2", [0.3, 0.8, 0.95], [1, 1, 1], 1),
    ]
    development = [
        protocol_mk("development", "dev-1", [0.4, 0.85], [0, 1], 1),
        protocol_mk("development", "dev-2", [0.5, 0.9], [1, 1], 1),
    ]
    test = [
        protocol_mk("test", "test-1", [0.45, 0.88], [0, 1], 1, seed=31),
        protocol_mk("test", "test-2", [0.6, 0.92], [0, 0], 0, seed=31),
    ]
    before = copy.deepcopy(test)
    config = frozen_config()
    dp = fit_empirical_dp(fit, config)
    fixed = select_fixed_threshold(development, {
        **config, "thresholds": [0.5, 0.7, 0.9],
    })
    policies = freeze_stopping_policies(config, dp, fixed)
    result = evaluate_stopping(test, policies)

    json.dumps(dp, allow_nan=False)
    json.dumps(fixed, allow_nan=False)
    json.dumps(result, allow_nan=False)
    assert dp["label"] == COMPARATOR_LABEL
    diagnostics = dp["diagnostics"]
    assert sum(diagnostics["bin_visit_counts"]) == 6
    assert len(diagnostics["occupancy_counts"]) == 3
    assert "fallback_usage" in diagnostics
    assert "stop_cost" in diagnostics
    assert "wait_cost" in diagnostics
    assert "bellman_margin_wait_minus_stop" in diagnostics
    assert result["methods"]["immediate"]["latency"] == 0.0
    assert result["methods"]["empirical_dp"]["label"] == COMPARATOR_LABEL
    assert set(result["paired_differences"]) == {
        "dp_minus_myopic", "dp_minus_fixed"
    }
    expected_ids = ["test-1", "test-2"]
    for metrics in result["per_seed"][0]["methods"].values():
        assert metrics["ordered_test_stream_ids"] == expected_ids
        assert metrics["correct_count"] + metrics["error_count"] == 2
    for current, old in zip(test, before):
        assert np.array_equal(current["bmax"], old["bmax"])
        assert np.array_equal(current["pred"], old["pred"])
        assert current["truth"] == old["truth"]


def test_new_apis_enforce_roles_freeze_fallback_and_reject_overlap():
    from protocol import ProtocolViolationError
    from stopping import (
        evaluate_stopping,
        fit_empirical_dp,
        freeze_stopping_policies,
        select_fixed_threshold,
    )

    fit = [protocol_mk("fit", "shared", [0.2, 0.9], [0, 1], 1)]
    development = [
        protocol_mk("development", "dev", [0.3, 0.9], [0, 1], 1)
    ]
    with pytest.raises(ValueError, match="empty_bin_fallback"):
        fit_empirical_dp(fit, {
            "wait_cost": 0.05, "error_cost": 1.0, "bin_count": 5,
        })
    with pytest.raises(ProtocolViolationError, match="fit operation"):
        fit_empirical_dp(development, frozen_config())

    dp = fit_empirical_dp(fit, frozen_config())
    fixed = select_fixed_threshold(development, {
        **frozen_config(), "thresholds": [0.5, 0.8],
    })
    policies = freeze_stopping_policies(frozen_config(), dp, fixed)
    overlapping_test = [
        protocol_mk("test", "shared", [0.4, 0.95], [0, 1], 1)
    ]
    with pytest.raises(ProtocolViolationError, match="overlap"):
        evaluate_stopping(overlapping_test, policies)


def test_fit_size_and_bin_count_sensitivity_are_predeclared_fit_only():
    from stopping import fit_empirical_dp_sensitivity

    fit = [
        protocol_mk("fit", f"fit-{index}", [0.2, 0.6, 0.9], [0, 1, 1], 1)
        for index in range(4)
    ]
    configs = [frozen_config(bin_count=3), frozen_config(bin_count=7)]
    records = fit_empirical_dp_sensitivity(fit, configs, fit_sizes=[2, 4])
    assert [(row["fit_size"], row["bin_count"]) for row in records] == [
        (2, 3), (4, 3), (2, 7), (4, 7),
    ]
    assert all(
        row["policy"]["diagnostics"]["fit_stream_count"] == row["fit_size"]
        for row in records
    )

"""Property 3 tests for coherent BOCPD and accepted boundary events.

**Validates: Requirements 2.1**
"""

from dataclasses import fields

import numpy as np
import pytest
from hypothesis import given, seed, strategies as st

from changepoint import BOCPD, BocpdSnapshot
from gates import Cross
from policies import BocpdThresh
from test_support import HYPOTHESIS_SEED, NUMERIC_TOLERANCE

ATOL = NUMERIC_TOLERANCE


def _reference_update(posterior, counts, observation, classes, alpha, hazard):
    """One independent implementation of the documented recurrence."""
    totals = counts.sum(axis=1)
    run_predictive = (counts[:, observation] + alpha) / (
        totals + classes * alpha
    )
    weights = np.concatenate((
        [posterior.sum() * hazard / classes],
        posterior * (1.0 - hazard) * run_predictive,
    ))
    predictive_probability = float(weights.sum())
    updated = weights / predictive_probability

    updated_counts = np.zeros((len(counts) + 1, classes), dtype=float)
    updated_counts[0, observation] = 1.0
    updated_counts[1:] = counts
    updated_counts[1:, observation] += 1.0
    return updated, updated_counts, predictive_probability


def _assert_probability_vector(values):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.isclose(values.sum(), 1.0, atol=ATOL, rtol=0.0)


def _run_reference_checked(trace, classes=3, alpha=1.0, hazard=0.05):
    detector = BOCPD(classes, hazard=hazard, alpha=alpha)
    posterior = np.array([1.0])
    counts = np.zeros((1, classes), dtype=float)
    snapshots = []

    for observation in trace:
        expected, counts, predictive = _reference_update(
            posterior, counts, observation, classes, alpha, hazard
        )
        snapshot = detector.update(observation)
        assert np.allclose(snapshot.posterior, expected, atol=ATOL, rtol=0.0)
        assert snapshot.cp_posterior == pytest.approx(expected[0], abs=ATOL)
        assert snapshot.predictive_probability == pytest.approx(predictive, abs=ATOL)
        assert snapshot.map_run_length == int(np.argmax(expected))
        _assert_probability_vector(snapshot.posterior)
        snapshots.append(snapshot)
        posterior = expected

    return snapshots


@pytest.mark.parametrize(
    "kwargs",
    (
        {"K": 0},
        {"K": 1},
        {"K": 3, "alpha": 0.0},
        {"K": 3, "alpha": -0.1},
        {"K": 3, "hazard": -0.01},
        {"K": 3, "hazard": 1.01},
    ),
)
def test_bocpd_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BOCPD(**kwargs)


@pytest.mark.parametrize(
    ("observation", "error"),
    (
        (-1, ValueError),
        (3, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
    ),
)
def test_bocpd_rejects_invalid_observations_without_mutating_state(
    observation, error
):
    detector = BOCPD(3, hazard=0.1, alpha=0.7)
    detector.update(1)
    posterior_before = detector.R.copy()
    counts_before = detector.counts.copy()

    with pytest.raises(error):
        detector.update(observation)

    assert np.array_equal(detector.R, posterior_before)
    assert np.array_equal(detector.counts, counts_before)


def test_no_information_first_observation_has_symmetric_declared_snapshot():
    """A fresh symmetric prior cannot favor one observed category over another."""
    classes = 4
    hazard = 0.2

    for observation in range(classes):
        snapshot = BOCPD(classes, hazard=hazard, alpha=3.0).update(observation)
        assert isinstance(snapshot, BocpdSnapshot)
        assert {field.name for field in fields(snapshot)} == {
            "cp_posterior",
            "posterior",
            "map_run_length",
            "predictive_probability",
            "predictive_surprise",
        }
        assert np.allclose(
            snapshot.posterior, [hazard, 1.0 - hazard], atol=ATOL, rtol=0.0
        )
        assert snapshot.cp_posterior == pytest.approx(hazard, abs=ATOL)
        assert snapshot.predictive_probability == pytest.approx(1.0 / classes)
        assert snapshot.predictive_surprise == pytest.approx(np.log(classes))
        assert snapshot.map_run_length == 1


def test_snapshot_is_structured_and_detached_from_detector_state():
    detector = BOCPD(3, hazard=0.1)
    snapshot = detector.update(2)
    detector_state = detector.R.copy()

    snapshot.posterior[0] = 999.0

    assert np.array_equal(detector.R, detector_state)
    assert snapshot.cp_posterior == pytest.approx(detector_state[0])


def test_stable_and_abrupt_change_fixtures_follow_declared_recurrence():
    stable_trace = [0] * 16
    abrupt_trace = [0] * 15 + [1]

    stable = _run_reference_checked(stable_trace)
    abrupt = _run_reference_checked(abrupt_trace)

    assert np.allclose(
        [item.cp_posterior for item in stable[:-1]],
        [item.cp_posterior for item in abrupt[:-1]],
        atol=ATOL,
        rtol=0.0,
    )
    assert stable[-1].cp_posterior < 0.05
    assert abrupt[-1].cp_posterior > 0.05
    assert abrupt[-1].cp_posterior > stable[-1].cp_posterior
    assert abrupt[-1].predictive_surprise > stable[-1].predictive_surprise


def _posterior_trace(labels, classes):
    values = np.full((len(labels), classes), 0.01 / (classes - 1))
    values[np.arange(len(labels)), labels] = 0.99
    return values


def test_between_threshold_fixture_makes_cp_thr_control_boundary_acceptance():
    classes = 3
    hazard = 0.05
    labels = [0] * 15 + [1]
    posterior_trace = _posterior_trace(labels, classes)

    stable_detector = BOCPD(classes, hazard=hazard)
    change_detector = BOCPD(classes, hazard=hazard)
    for _ in range(15):
        stable_detector.update(0)
        change_detector.update(0)
    stable_score = stable_detector.update(0).cp_posterior
    change_score = change_detector.update(1).cp_posterior

    lower_threshold = (stable_score + change_score) / 2.0
    upper_threshold = (change_score + 1.0) / 2.0
    release = lower_threshold / 2.0
    assert stable_score < lower_threshold < change_score < upper_threshold

    lower = BocpdThresh(
        classes, hazard=hazard, cp_thr=lower_threshold, cp_rel=release
    )
    upper = BocpdThresh(
        classes, hazard=hazard, cp_thr=upper_threshold, cp_rel=release
    )
    lower.run(posterior_trace)
    upper.run(posterior_trace)

    assert lower.bd == [15]
    assert upper.bd == []
    assert lower.cp == pytest.approx(upper.cp, abs=ATOL)


def test_boundary_hysteresis_requires_release_before_a_second_event():
    gate = Cross(up=0.20, lo=0.08)
    scores = [0.05, 0.20, 0.35, 0.19, 0.09, 0.08, 0.21]

    assert [gate.step(score) for score in scores] == [
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def _boundary_counts(accepted, true_boundaries, horizon):
    """Exact-frame oracle for the task's deterministic count fixtures."""
    accepted = tuple(int(time) for time in accepted)
    true_boundaries = frozenset(int(time) for time in true_boundaries)
    detected = sum(time in true_boundaries for time in accepted)
    false_alarms = len(accepted) - detected
    eligible_nonboundary_frames = horizon - len(true_boundaries)
    return {
        "boundary_n": len(true_boundaries),
        "detection_n": detected,
        "miss_n": len(true_boundaries) - detected,
        "false_alarm_n": false_alarms,
        "false_alarm_d": eligible_nonboundary_frames,
        "false_alarm_r": false_alarms / eligible_nonboundary_frames,
    }


def test_boundary_false_alarm_and_detection_count_semantics():
    assert _boundary_counts([3, 10], [10], horizon=16) == {
        "boundary_n": 1,
        "detection_n": 1,
        "miss_n": 0,
        "false_alarm_n": 1,
        "false_alarm_d": 15,
        "false_alarm_r": 1 / 15,
    }
    assert _boundary_counts([], [], horizon=16) == {
        "boundary_n": 0,
        "detection_n": 0,
        "miss_n": 0,
        "false_alarm_n": 0,
        "false_alarm_d": 16,
        "false_alarm_r": 0.0,
    }

    labels = [0] * 15 + [1]
    values = _posterior_trace(labels, classes=3)
    detector = BOCPD(3, hazard=0.05)
    for observation in labels[:15]:
        detector.update(observation)
    change_score = detector.update(1).cp_posterior

    policy = BocpdThresh(
        3,
        hazard=0.05,
        cp_thr=(0.05 + change_score) / 2.0,
        cp_rel=0.02,
    )
    policy.run(values)

    assert policy.bd == [15]
    assert _boundary_counts(policy.bd, [15], horizon=16) == {
        "boundary_n": 1,
        "detection_n": 1,
        "miss_n": 0,
        "false_alarm_n": 0,
        "false_alarm_d": 15,
        "false_alarm_r": 0.0,
    }


@st.composite
def _categorical_bocpd_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=6))
    alpha = draw(st.sampled_from((0.05, 0.2, 1.0, 3.0, 10.0)))
    hazard = draw(st.sampled_from((0.0, 0.01, 0.05, 0.2, 0.75, 1.0)))
    trace = draw(
        st.lists(
            st.integers(min_value=0, max_value=classes - 1),
            min_size=1,
            max_size=25,
        )
    )
    return classes, alpha, hazard, trace


@seed(HYPOTHESIS_SEED)
@given(_categorical_bocpd_cases())
def test_property_3_bocpd_recurrence_normalization_for_generated_inputs(case):
    """Property 3. **Validates: Requirements 2.1**"""
    classes, alpha, hazard, trace = case
    detector = BOCPD(classes, hazard=hazard, alpha=alpha)
    expected = np.array([1.0])
    counts = np.zeros((1, classes), dtype=float)

    for observation in trace:
        expected, counts, predictive = _reference_update(
            expected, counts, observation, classes, alpha, hazard
        )
        snapshot = detector.update(observation)

        assert isinstance(snapshot, BocpdSnapshot)
        _assert_probability_vector(snapshot.posterior)
        assert np.allclose(snapshot.posterior, expected, atol=ATOL, rtol=0.0)
        assert snapshot.cp_posterior == pytest.approx(expected[0], abs=ATOL)
        assert snapshot.predictive_probability == pytest.approx(
            predictive, abs=ATOL
        )
        assert snapshot.map_run_length == int(np.argmax(expected))
        assert np.isfinite(snapshot.predictive_surprise)
        assert snapshot.predictive_surprise >= 0.0

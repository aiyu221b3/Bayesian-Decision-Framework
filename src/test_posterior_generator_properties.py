"""Properties 10 and 13 for posterior and generator preservation.

The generated checks refine the observation-first Task 2 preservation oracle;
they do not redefine or modify the validated generator mathematics.
"""

import numpy as np
from hypothesis import given, seed, strategies as st

from belief import FlatBel
from changepoint import BOCPD
from evidence import (
    Simcon,
    _ar1_noise,
    _label_seq,
    _mean_logits,
    build_sim,
    concat_stream,
)
from policies import BayesThresh
from test_support import HYPOTHESIS_SEED, NUMERIC_TOLERANCE

ATOL = NUMERIC_TOLERANCE


def _assert_probability_vector(values):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.isclose(values.sum(), 1.0, atol=ATOL, rtol=0.0)


def _assert_probability_rows(values):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0 + ATOL)
    assert np.allclose(values.sum(axis=1), 1.0, atol=ATOL, rtol=0.0)


def _draw_probability(draw, classes):
    kind = draw(st.sampled_from(("one-hot", "dense", "sparse")))
    if kind == "one-hot":
        vector = np.zeros(classes, dtype=float)
        vector[draw(st.integers(min_value=0, max_value=classes - 1))] = 1.0
        return vector

    minimum = 1 if kind == "dense" else 0
    weights = np.asarray(draw(st.lists(
        st.integers(min_value=minimum, max_value=1000),
        min_size=classes,
        max_size=classes,
    )), dtype=float)
    if weights.sum() == 0.0:
        weights[0] = 1.0
    return weights / weights.sum()


@st.composite
def posterior_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=7))
    horizon = draw(st.integers(min_value=1, max_value=25))
    posteriors = np.stack([
        _draw_probability(draw, classes) for _ in range(horizon)
    ])
    prior = _draw_probability(draw, classes)
    return {
        "classes": classes,
        "posteriors": posteriors,
        "prior": prior,
        "lam": draw(st.sampled_from((0.0, 0.1, 0.5, 1.0, 2.0))),
        "rho": draw(st.sampled_from((0.01, 0.08, 0.4, 1.0))),
        "hazard": draw(st.sampled_from((0.0, 0.01, 0.05, 0.3, 1.0))),
        "alpha": draw(st.sampled_from((0.05, 0.5, 1.0, 5.0))),
        "cut": draw(st.integers(min_value=1, max_value=horizon)),
        "tau": draw(st.sampled_from((0.5, 0.8, 0.95))),
        "refr": draw(st.integers(min_value=0, max_value=5)),
    }


@seed(HYPOTHESIS_SEED)
@given(posterior_cases())
def test_property_10_generated_ablation_and_posterior_invariants(case):
    """Property 10. **Validates: Requirements 2.7, 3.1, 3.2**"""
    classes = case["classes"]
    posteriors = case["posteriors"]
    common = {
        "K": classes,
        "lam": case["lam"],
        "prior": case["prior"],
    }

    m1 = FlatBel(rho=0.0, **common).run(posteriors)
    m2_at_zero = FlatBel(rho=0.0, **common).run(posteriors)
    switching = FlatBel(rho=case["rho"], **common).run(posteriors)

    # The zero-switch ablation follows the identical arithmetic path.
    assert np.array_equal(m1, m2_at_zero)
    for trajectory in (m1, m2_at_zero, switching):
        _assert_probability_rows(trajectory)

    cut = case["cut"]
    for rho, full in ((0.0, m1), (case["rho"], switching)):
        prefix = FlatBel(rho=rho, **common).run(posteriors[:cut])
        assert np.allclose(prefix, full[:cut], atol=ATOL, rtol=0.0)

    policy_settings = {
        "K": classes,
        "lam": case["lam"],
        "rho": 0.0,
        "tau": case["tau"],
        "refr": case["refr"],
    }
    m1_policy = BayesThresh(**policy_settings)
    m2_zero_policy = BayesThresh(**policy_settings)
    assert m1_policy.run(posteriors) == m2_zero_policy.run(posteriors)
    assert np.array_equal(m1_policy.bel.belief, m2_zero_policy.bel.belief)

    observations = np.argmax(posteriors, axis=1)
    detector = BOCPD(
        classes, hazard=case["hazard"], alpha=case["alpha"]
    )
    snapshots = [detector.update(int(value)) for value in observations]
    for snapshot in snapshots:
        _assert_probability_vector(snapshot.posterior)
        assert np.isclose(
            snapshot.cp_posterior,
            snapshot.posterior[0],
            atol=ATOL,
            rtol=0.0,
        )

    prefix_detector = BOCPD(
        classes, hazard=case["hazard"], alpha=case["alpha"]
    )
    prefix_snapshots = [
        prefix_detector.update(int(value)) for value in observations[:cut]
    ]
    for expected, observed in zip(snapshots[:cut], prefix_snapshots):
        assert np.allclose(
            observed.posterior, expected.posterior, atol=ATOL, rtol=0.0
        )
        assert np.isclose(
            observed.cp_posterior,
            expected.cp_posterior,
            atol=ATOL,
            rtol=0.0,
        )


@st.composite
def simulator_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=7))
    duration_low = draw(st.integers(min_value=3, max_value=8))
    duration_high = draw(
        st.integers(min_value=duration_low + 1, max_value=duration_low + 6)
    )
    return {
        "seed": draw(st.integers(min_value=0, max_value=2**32 - 1)),
        "n_signs": draw(st.integers(min_value=1, max_value=6)),
        "cfg": Simcon(
            K=classes,
            fami=draw(st.integers(min_value=1, max_value=classes)),
            w_f=draw(st.sampled_from((0.35, 0.55, 0.8))),
            a_f=draw(st.sampled_from((0.0, 0.08, 0.2))),
            sig=draw(st.sampled_from((0.5, 3.8, 7.0))),
            gamma=draw(st.sampled_from((0.5, 1.0, 1.7, 3.0))),
            corr=draw(st.sampled_from((-0.8, -0.25, 0.0, 0.6, 0.9))),
            noise=draw(st.sampled_from((0.0, 0.1, 0.6, 1.2))),
            temp=draw(st.sampled_from((0.3, 0.7, 1.5))),
            dur_r=(duration_low, duration_high),
            blend=draw(st.integers(min_value=0, max_value=duration_low)),
        ),
    }


def _assert_same_stream(first, second):
    for field in ("p", "z", "families"):
        assert np.array_equal(first[field], second[field])
    for field in ("labels", "durss", "onsets", "bounds"):
        assert first[field] == second[field]


def _assert_ar1_recurrence(cfg, seed_value, horizon):
    observed = _ar1_noise(
        np.random.default_rng(seed_value),
        cfg.corr,
        cfg.noise,
        horizon,
        cfg.K,
    )
    replay_rng = np.random.default_rng(seed_value)
    expected = np.zeros_like(observed)
    expected[0] = replay_rng.normal(0.0, cfg.noise, cfg.K)
    inn_scl = np.sqrt(1.0 - cfg.corr**2)
    for frame in range(1, horizon):
        epsilon_n = replay_rng.normal(0.0, cfg.noise, cfg.K)
        expected[frame] = (
            cfg.corr * expected[frame - 1]
            + inn_scl * epsilon_n
        )
    assert np.allclose(observed, expected, atol=ATOL, rtol=0.0)


def _assert_ramps_and_coarticulation(cfg, seed_value, n_signs):
    rng = np.random.default_rng(seed_value)
    similarity, _ = build_sim(cfg, rng)
    truth, labels, durss, onsets, boundaries = _label_seq(
        cfg, n_signs, rng
    )
    means = _mean_logits(
        cfg, similarity, truth, labels, durss, onsets
    )

    for segment, (label, duration, onset) in enumerate(
        zip(labels, durss, onsets)
    ):
        for within in range(duration):
            progress = within / max(duration - 1, 1)
            current = cfg.sig * progress**cfg.gamma * similarity[:, label]
            expected = current
            if segment > 0 and cfg.blend > 0 and within < cfg.blend:
                weight = within / cfg.blend
                previous = cfg.sig * similarity[:, labels[segment - 1]]
                expected = (1.0 - weight) * previous + weight * current
            assert np.allclose(
                means[onset + within], expected, atol=ATOL, rtol=0.0
            )

        ramp_start = 0 if segment == 0 else min(cfg.blend, duration)
        own_curve = means[
            onset + ramp_start:onset + duration, label
        ]
        if len(own_curve) > 1:
            assert np.all(np.diff(own_curve) >= -ATOL)

    for segment, boundary in enumerate(boundaries, start=1):
        if cfg.blend == 0:
            assert np.allclose(means[boundary], 0.0, atol=ATOL, rtol=0.0)
        else:
            expected = cfg.sig * similarity[:, labels[segment - 1]]
            assert np.allclose(
                means[boundary], expected, atol=ATOL, rtol=0.0
            )


@seed(HYPOTHESIS_SEED)
@given(simulator_cases())
def test_property_13_generated_generator_structural_preservation(case):
    """Property 13. **Validates: Requirement 3.3**"""
    cfg = case["cfg"]
    seed_value = case["seed"]
    n_signs = case["n_signs"]
    first = concat_stream(
        cfg, np.random.default_rng(seed_value), n_signs=n_signs
    )
    second = concat_stream(
        cfg, np.random.default_rng(seed_value), n_signs=n_signs
    )

    _assert_same_stream(first, second)
    _assert_probability_rows(first["p"])
    assert len(first["p"]) == len(first["z"]) == sum(first["durss"])
    assert len(first["labels"]) == len(first["durss"]) == n_signs
    assert all(
        left != right
        for left, right in zip(first["labels"], first["labels"][1:])
    )

    expected_onsets = [0] + list(np.cumsum(first["durss"])[:-1])
    assert first["onsets"] == expected_onsets
    assert first["bounds"] == expected_onsets[1:]
    low, high = cfg.dur_r
    assert all(low <= duration < high for duration in first["durss"])
    for label, duration, onset in zip(
        first["labels"], first["durss"], first["onsets"]
    ):
        assert np.all(first["z"][onset:onset + duration] == label)

    noise_seed = seed_value ^ 0xA5A5A5A5
    _assert_ar1_recurrence(
        cfg, noise_seed, max(3, min(40, len(first["z"])))
    )
    _assert_ramps_and_coarticulation(cfg, seed_value, n_signs)

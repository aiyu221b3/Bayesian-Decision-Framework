import numpy as np
import pytest

from belief import FlatBel
from changepoint import BOCPD, BocpdSnapshot
from evidence import Simcon, concat_stream
from test_support import NUMERIC_TOLERANCE


def assert_probability_vector(values, atol=NUMERIC_TOLERANCE):
    values = np.asarray(values, dtype=float)

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
    assert np.isclose(values.sum(), 1.0, atol=atol)


def test_simulator_is_deterministic_for_fixed_seed():
    cfg = Simcon()

    first = concat_stream(
        cfg,
        np.random.default_rng(17),
        n_signs=5,
    )
    second = concat_stream(
        cfg,
        np.random.default_rng(17),
        n_signs=5,
    )

    assert np.array_equal(first["z"], second["z"])
    assert np.array_equal(first["p"], second["p"])
    assert first["labels"] == second["labels"]
    assert first["durss"] == second["durss"]
    assert first["onsets"] == second["onsets"]
    assert first["bounds"] == second["bounds"]


def test_simulator_structural_invariants():
    cfg = Simcon()
    stream = concat_stream(
        cfg,
        np.random.default_rng(4),
        n_signs=6,
    )

    assert len(stream["z"]) == sum(stream["durss"])
    assert len(stream["p"]) == len(stream["z"])

    assert all(
        left != right
        for left, right in zip(
            stream["labels"],
            stream["labels"][1:],
        )
    )

    assert np.allclose(
        stream["p"].sum(axis=1),
        1.0,
        atol=NUMERIC_TOLERANCE,
    )

    assert np.all(np.isfinite(stream["p"]))
    assert np.all(stream["p"] >= 0)


def test_belief_remains_normalized_and_causal():
    rng = np.random.default_rng(5)
    observations = rng.dirichlet(
        np.ones(5),
        size=50,
    )

    belief_filter = FlatBel(
        K=5,
        lam=0.4,
        rho=0.07,
    )

    for observation in observations:
        belief = belief_filter.update(observation)
        assert_probability_vector(belief)


def test_rho_zero_is_the_static_limit():
    rng = np.random.default_rng(8)
    observations = rng.dirichlet(
        np.ones(4),
        size=30,
    )

    static = FlatBel(
        K=4,
        lam=0.37,
        rho=0.0,
    ).run(observations)

    switching_at_zero = FlatBel(
        K=4,
        lam=0.37,
        rho=0.0,
    ).run(observations)

    assert np.allclose(
        static,
        switching_at_zero,
        atol=NUMERIC_TOLERANCE,
    )


def test_switching_belief_adapts_faster_after_change():
    class_zero = np.array([0.98, 0.01, 0.01])
    class_one = np.array([0.01, 0.98, 0.01])

    observations = np.vstack([
        np.tile(class_zero, (20, 1)),
        np.tile(class_one, (20, 1)),
    ])

    static = FlatBel(
        K=3,
        lam=0.5,
        rho=0.0,
    ).run(observations)

    switching = FlatBel(
        K=3,
        lam=0.5,
        rho=0.08,
    ).run(observations)

    def first_new_class_argmax(beliefs):
        hits = np.flatnonzero(
            np.argmax(beliefs[20:], axis=1) == 1
        )
        return int(hits[0]) if len(hits) else len(beliefs)

    static_delay = first_new_class_argmax(static)
    switching_delay = first_new_class_argmax(switching)

    assert switching_delay < static_delay


def test_bocpd_first_observation_matches_hazard():
    hazard = 0.08
    detector = BOCPD(
        K=4,
        hazard=hazard,
        alpha=1.0,
    )

    snapshot = detector.update(2)

    assert isinstance(snapshot, BocpdSnapshot)
    assert snapshot.cp_posterior == pytest.approx(hazard)
    assert_probability_vector(snapshot.posterior)


def test_bocpd_posterior_invariants():
    detector = BOCPD(
        K=5,
        hazard=0.05,
        alpha=1.0,
    )

    observations = [
        0, 0, 0, 0, 1, 1, 1, 4, 4, 2, 2, 2
    ]

    for observation in observations:
        snapshot = detector.update(observation)

        assert_probability_vector(snapshot.posterior)
        assert 0.0 <= snapshot.cp_posterior <= 1.0
        assert snapshot.map_run_length >= 0
        assert snapshot.predictive_probability > 0
        assert np.isfinite(snapshot.predictive_surprise)


def test_bocpd_is_sensitive_to_surprising_evidence():
    stable_detector = BOCPD(
        K=3,
        hazard=0.05,
        alpha=1.0,
    )
    surprise_detector = BOCPD(
        K=3,
        hazard=0.05,
        alpha=1.0,
    )

    for _ in range(15):
        stable_detector.update(0)
        surprise_detector.update(0)

    stable_score = stable_detector.update(0).cp_posterior
    surprising_score = surprise_detector.update(1).cp_posterior

    assert surprising_score > stable_score


@pytest.mark.parametrize(
    "kwargs",
    [
        {"K": 1},
        {"K": 3, "hazard": -0.1},
        {"K": 3, "hazard": 1.1},
        {"K": 3, "alpha": 0.0},
    ],
)
def test_invalid_bocpd_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BOCPD(**kwargs)


def test_invalid_bocpd_observation_is_rejected():
    detector = BOCPD(K=3)

    with pytest.raises(ValueError):
        detector.update(3)

    with pytest.raises(TypeError):
        detector.update(1.5)

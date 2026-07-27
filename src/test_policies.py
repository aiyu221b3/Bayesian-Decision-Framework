import numpy as np
import pytest

from gates import Cross, Gate
from policies import BayesThresh, BocpdThresh, RawThresh
from tree_belief import TreeBel


def test_high_level_emits_once():
    g = Gate(up=0.8, lo=0.7, ref=0)

    out = [
        g.step(t, 0, 0.9, 0.9)
        for t in range(20)
    ]

    assert [y for y in out if y is not None] == [0]


def test_release_and_recross_emit_twice():
    g = Gate(up=0.8, lo=0.7, ref=0)

    out = [
        g.step(0, 0, 0.90, None),
        g.step(1, 0, 0.76, 0.76),
        g.step(2, 0, 0.65, 0.65),
        g.step(3, 0, 0.86, 0.86),
    ]

    assert [y for y in out if y is not None] == [0, 0]


def test_timer_alone_does_not_rearm():
    g = Gate(up=0.8, lo=0.7, ref=3)

    out = [
        g.step(t, 0, 0.9, 0.9)
        for t in range(30)
    ]

    assert [y for y in out if y is not None] == [0]


def test_disjoint_class_can_emit():
    g = Gate(up=0.8, lo=0.7, ref=0)

    a = g.step(0, 0, 0.9, None)
    b = g.step(1, 1, 0.9, 0.8)

    assert a == 0
    assert b == 1


def test_boundary_allows_same_class_again():
    g = Gate(up=0.8, lo=0.7, ref=0)

    a = g.step(0, 0, 0.9, None)
    b = g.step(5, 0, 0.9, 0.9, bd=True)

    assert a == 0
    assert b == 0


def test_cross_needs_release():
    cg = Cross(up=0.1, lo=0.04)

    vals = [0.03, 0.11, 0.15, 0.08, 0.03, 0.12]
    out = [cg.step(x) for x in vals]

    assert out == [False, True, False, False, False, True]


@pytest.mark.parametrize(
    ("up", "lo"),
    ((0.0, 0.0), (0.5, 0.5), (0.4, 0.5), (1.1, 0.5)),
)
def test_cross_rejects_invalid_hysteresis_thresholds(up, lo):
    with pytest.raises(ValueError):
        Cross(up=up, lo=lo)


def test_cross_reset_rearms_detector():
    cg = Cross(up=0.8, lo=0.7)

    assert cg.step(0.8) is True
    assert cg.step(0.9) is False

    cg.reset()

    assert cg.step(0.8) is True


@pytest.mark.parametrize("refr", (-1, 1.5, 2.0, "2", True, None))
def test_gate_rejects_non_integer_refractory_values(refr):
    with pytest.raises(ValueError, match="nonnegative integer"):
        Gate(up=0.8, lo=0.7, ref=refr)


def test_gate_preserves_exact_node_identity_and_support():
    node = frozenset({2, 4})
    gate = Gate(up=0.8, lo=0.7, ref=0)

    assert gate.step(0, node, 0.9) == node
    assert gate.last_output == node
    assert gate.last_support == node
    assert gate.last == node


def test_raw_high_trace_emits_once():
    P = np.tile([0.95, 0.05], (40, 1))
    em = RawThresh(2, theta=0.8, refr=3).run(P)

    assert em == [(0, 0)]


def test_bayes_high_trace_emits_once():
    P = np.tile([0.99, 0.01], (40, 1))
    em = BayesThresh(
        2,
        lam=1.0,
        rho=0.0,
        tau=0.8,
        refr=3,
    ).run(P)

    assert len(em) == 1
    assert em[0][1] == 0


def test_bocpd_high_trace_emits_once():
    P = np.tile([0.99, 0.01], (40, 1))
    pol = BocpdThresh(
        2,
        lam=1.0,
        tau=0.8,
        cp_thr=0.1,
        refr=3,
    )

    em = pol.run(P)

    assert len(em) == 1
    assert em[0][1] == 0


def test_raw_a_to_b_emits_both():
    A = np.tile([0.95, 0.05], (10, 1))
    B = np.tile([0.05, 0.95], (10, 1))
    P = np.vstack([A, B])

    em = RawThresh(
        2,
        theta=0.8,
        refr=0,
    ).run(P)

    assert em == [(0, 0), (10, 1)]


def test_tree_never_emits_root():
    S = np.array([
        [1.0, 0.8, 0.1, 0.1],
        [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ])

    P = np.tile(
        [0.92, 0.04, 0.02, 0.02],
        (30, 1),
    )

    mt = TreeBel(
        4,
        S,
        lam=1.0,
        rho=0.05,
        tau=0.8,
        refr=0,
    )

    em = mt.run(P)

    assert len(em) == 1
    assert len(em[0][1]) < 4

"""Property 4 tests for the shared hysteretic emission state machine.

**Validates: Requirements 2.2, 3.4, 3.6**
"""

import numpy as np
from hypothesis import given, seed, strategies as st

from gates import Gate
from policies import BayesThresh, BocpdThresh, RawThresh
from test_support import HYPOTHESIS_SEED, NUMERIC_TOLERANCE
from tree_belief import TreeBel

ATOL = NUMERIC_TOLERANCE
UP = 0.80
LO = 0.70


def _sp(y):
    if isinstance(y, (set, frozenset)):
        return frozenset(y)
    return frozenset((int(y),))


def _mass(v, nd):
    if nd is None:
        return None
    return float(np.asarray(v)[list(nd)].sum())


class _Ref:
    """Independent reference implementation of the gate rules."""

    def __init__(self, up=UP, lo=LO, ref=0):
        self.up = float(up)
        self.lo = float(lo)
        self.ref = int(ref)
        self.arm = True
        self.last = None
        self.t0 = None

    def step(self, t, y, s, ls=None, bd=False):
        s = float(s)
        ls = None if ls is None else float(ls)

        if y is None:
            if not self.arm and (
                bd or (ls is not None and ls <= self.lo)
            ):
                self.arm = True
            return None

        ys = _sp(y)

        if not self.arm:
            rel = (
                bd
                or (ls is not None and ls <= self.lo)
                or (
                    self.last is not None
                    and ys.isdisjoint(self.last)
                    and s >= self.up
                )
            )

            if rel:
                self.arm = True

        if not self.arm or s < self.up:
            return None

        if self.t0 is not None and t - self.t0 <= self.ref:
            return None

        self.arm = False
        self.last = ys
        self.t0 = int(t)

        return y


_ID = st.sampled_from((
    None,
    0,
    1,
    frozenset({0}),
    frozenset({1}),
    frozenset({0, 1}),
    frozenset({1, 2}),
))

_SC = st.sampled_from((
    0.0,
    0.69,
    0.70,
    0.71,
    0.79,
    0.80,
    0.81,
    0.95,
    1.0,
))

_LS = st.one_of(st.none(), _SC)


@st.composite
def _gate_cases(draw):
    n = draw(st.integers(min_value=1, max_value=30))
    ys = draw(st.lists(_ID, min_size=n, max_size=n))
    ss = draw(st.lists(_SC, min_size=n, max_size=n))
    ls = draw(st.lists(_LS, min_size=n, max_size=n))
    bd = draw(st.lists(
        st.booleans(),
        min_size=n,
        max_size=n,
    ))
    ref = draw(st.integers(min_value=0, max_value=8))

    return list(zip(ys, ss, ls, bd)), ref


@seed(HYPOTHESIS_SEED)
@given(_gate_cases())
def test_property_4_gate_matches_reference_for_generated_traces(case):
    """Property 4. **Validates: Requirements 2.2, 3.4, 3.6**"""
    rows, ref = case
    g = Gate(up=UP, lo=LO, ref=ref)
    r = _Ref(up=UP, lo=LO, ref=ref)

    got = []
    want = []

    for t, (y, s, ls, bd) in enumerate(rows):
        a = g.step(t, y, s, ls, bd)
        b = r.step(t, y, s, ls, bd)

        if a is not None:
            got.append((t, a))
        if b is not None:
            want.append((t, b))

    assert got == want
    assert g.arm == r.arm
    assert g.last == r.last
    assert g.t0 == r.t0


@st.composite
def _posterior_cases(draw):
    n = draw(st.integers(min_value=1, max_value=25))
    ys = draw(st.lists(
        st.integers(min_value=0, max_value=3),
        min_size=n,
        max_size=n,
    ))
    cs = draw(st.lists(
        _SC.filter(lambda x: x >= 0.40),
        min_size=n,
        max_size=n,
    ))

    rows = []

    for y, c in zip(ys, cs):
        q = np.full(4, (1.0 - c) / 3.0)
        q[y] = c
        rows.append(q)

    ref = draw(st.integers(min_value=0, max_value=8))

    return np.asarray(rows), ref


_S = np.array([
    [1.0, 0.8, 0.1, 0.1],
    [0.8, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.8],
    [0.1, 0.1, 0.8, 1.0],
])


def _pols(ref):
    return {
        "M0": RawThresh(
            4,
            theta=UP,
            low=LO,
            refr=ref,
        ),
        "M1": BayesThresh(
            4,
            lam=1.0,
            rho=0.0,
            tau=UP,
            low=LO,
            refr=ref,
        ),
        "M2": BayesThresh(
            4,
            lam=1.0,
            rho=0.05,
            tau=UP,
            low=LO,
            refr=ref,
        ),
        "M3": BocpdThresh(
            4,
            lam=1.0,
            tau=UP,
            low=LO,
            hazard=0.05,
            cp_thr=0.20,
            cp_release=0.05,
            refr=ref,
        ),
        "MT": TreeBel(
            4,
            _S,
            lam=1.0,
            rho=0.05,
            tau=UP,
            low=LO,
            refr=ref,
        ),
    }


def _expected(p, P):
    r = _Ref(p.g.up, p.g.lo, p.g.ref)
    em = []

    if isinstance(p, RawThresh):
        for t, q in enumerate(P):
            y = int(np.argmax(q))
            s = float(q[y])
            o = r.step(t, y, s, _mass(q, r.last))

            if o is not None:
                em.append((t, o))

        return em

    if isinstance(p, BocpdThresh):
        p.bel.reset()
        p.cpd.reset()
        p.cg.reset()

        for t, q in enumerate(P):
            cp = p.cpd.update(
                int(np.argmax(q))
            ).cp_posterior
            bd = p.cg.step(cp)

            if bd:
                p.bel.reset()

            b = p.bel.update(q)
            y = int(np.argmax(b))
            s = float(b[y])
            o = r.step(
                t,
                y,
                s,
                _mass(b, r.last),
                bd,
            )

            if o is not None:
                em.append((t, o))

        return em

    if isinstance(p, BayesThresh):
        p.bel.reset()

        for t, q in enumerate(P):
            b = p.bel.update(q)
            y = int(np.argmax(b))
            s = float(b[y])
            o = r.step(
                t,
                y,
                s,
                _mass(b, r.last),
            )

            if o is not None:
                em.append((t, o))

        return em

    p.bel.reset()

    for t, q in enumerate(P):
        b = p.bel.update(q)
        y, s = p.pick()
        o = r.step(
            t,
            y,
            s,
            _mass(b, r.last),
        )

        if o is not None:
            em.append((t, o))

    return em


@seed(HYPOTHESIS_SEED)
@given(_posterior_cases())
def test_property_4_m0_through_mt_match_reference_state_machine(case):
    """Property 4. **Validates: Requirements 2.2, 3.4, 3.6**"""
    P, ref = case

    for p in _pols(ref).values():
        want = _expected(p, P)
        got = p.run(P)
        assert got == want


def test_property_4_exact_fixture_and_threshold_noise_do_not_chatter():
    """Property 4. **Validates: Requirements 2.2, 3.4**"""
    eps = 10.0 * ATOL
    ss = [
        0.79,
        0.80,
        0.81,
        0.79,
        0.71,
        LO + eps,
        0.70,
        0.79,
        0.80,
    ]

    g = Gate(up=UP, lo=LO, ref=0)
    em = []

    for t, s in enumerate(ss):
        y = g.step(t, 0, s, s)

        if y is not None:
            em.append((t, y))

    assert em == [(1, 0), (8, 0)]


def test_property_4_overlapping_hierarchical_refinement_is_not_new_event():
    """Property 4. **Validates: Requirements 2.2, 3.4, 3.6**"""
    fam = frozenset({0, 1})
    leaf = frozenset({0})
    other = frozenset({2})

    g = Gate(up=UP, lo=LO, ref=0)

    out = [
        g.step(0, fam, 0.90, None),
        g.step(1, leaf, 0.90, 0.95),
        g.step(2, other, 0.90, 0.05),
    ]

    assert out == [fam, None, other]

    P = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.90, 0.05, 0.025, 0.025],
    ])

    mt = TreeBel(
        4,
        _S,
        lam=1.0,
        rho=1.0,
        tau=UP,
        low=LO,
        refr=0,
    )

    assert mt.run(P) == [(0, fam)]

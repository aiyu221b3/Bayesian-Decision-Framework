import numpy as np

from belief import FlatBel
from changepoint import BOCPD
from gates import Cross, EmitterGate


def _mass(v, nd):
    if nd is None:
        return None
    return float(np.asarray(v)[list(nd)].sum())


class RawThresh:
    def __init__(
        self,
        K,
        theta=0.6,
        refr=8,
        low=None,
    ):
        self.K = K
        self.theta = float(theta)
        self.low = max(0.0, self.theta - 0.1) if low is None else float(low)
        self.g = EmitterGate(self.theta, self.low, refr)
        self.refr = self.g.refr

    def run(self, P):
        self.g.reset()
        self.sc = []
        em = []
        for t, p in enumerate(P):
            p = np.asarray(p, dtype=float)
            y = int(np.argmax(p))
            s = float(p[y])
            ls = _mass(p, self.g.last_support)
            self.sc.append(s)
            out = self.g.step(t, y, s, ls)
            if out is not None:
                em.append((t, out))
        return em


class BayesThresh:
    def __init__(
        self,
        K,
        lam=0.5,
        rho=0.0,
        tau=0.8,
        refr=8,
        low=None,
    ):
        self.K = K
        self.bel = FlatBel(K, lam, rho)
        self.tau = float(tau)
        self.low = max(0.0, self.tau - 0.1) if low is None else float(low)
        self.g = EmitterGate(self.tau, self.low, refr)
        self.refr = self.g.refr

    def run(self, P):
        self.bel.reset()
        self.g.reset()
        self.sc = []
        em = []
        for t, p in enumerate(P):
            b = self.bel.update(p)
            y = int(np.argmax(b))
            s = float(b[y])
            ls = _mass(b, self.g.last_support)
            self.sc.append(s)
            out = self.g.step(t, y, s, ls)
            if out is not None:
                em.append((t, out))
        return em


class BocpdThresh:
    def __init__(
        self,
        K,
        lam=0.5,
        tau=0.8,
        hazard=1 / 30.0,
        cp_thr=0.1,
        refr=8,
        cp_rel=None,
        low=None,
        alpha=1.0,
        cp_release=None,
    ):
        self.K = K
        self.bel = FlatBel(K, lam, 0.0)
        self.cpd = BOCPD(K, hazard, alpha)
        self.tau = float(tau)
        self.low = max(0.0, self.tau - 0.1) if low is None else float(low)
        if cp_rel is not None and cp_release is not None:
            raise ValueError("specify only one of cp_release and cp_rel")
        release = cp_release if cp_release is not None else cp_rel
        self.cp_thr = float(cp_thr)
        self.cp_release = (
            min(self.cp_thr * 0.5, self.cpd.h * 1.2)
            if release is None
            else float(release)
        )
        # compat spelling
        self.cp_rel = self.cp_release
        self.g = EmitterGate(self.tau, self.low, refr)
        self.refr = self.g.refr
        self.cg = Cross(self.cp_thr, self.cp_release)

    def run(self, P):
        self.bel.reset()
        self.cpd.reset()
        self.g.reset()
        self.cg.reset()
        self.sc = []
        self.cp = []
        self.bd = []
        em = []
        for t, p in enumerate(P):
            snap = self.cpd.update(int(np.argmax(p)))
            cp = snap.cp_posterior
            bd = self.cg.step(cp)
            self.cp.append(cp)
            if bd:
                self.bd.append(t)
                self.bel.reset()
            b = self.bel.update(p)
            y = int(np.argmax(b))
            s = float(b[y])
            ls = _mass(b, self.g.last_support)
            self.sc.append(s)
            out = self.g.step(t, y, s, ls, bd)
            if out is not None:
                em.append((t, out))
        return em

import itertools

import numpy as np

from belief import FlatBel
from gates import EmitterGate


def estimate_confusion(Ps, Zs, K):
    C = np.zeros((K, K), dtype=float)
    for P, Z in zip(Ps, Zs):
        pr = np.argmax(P, axis=1)
        for z, y in zip(Z, pr):
            C[int(z), int(y)] += 1
    den = C.sum(axis=1, keepdims=True)
    return np.divide(C, den, out=np.zeros_like(C), where=den > 0)


def confusion_to_sim(C):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")
    S = 0.5 * (C + C.T)
    S = np.clip(S, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def build_tree(S):
    S = np.asarray(S, dtype=float)
    K = len(S)
    if S.shape != (K, K):
        raise ValueError("S must be square")
    if K < 2:
        raise ValueError("tree needs at least 2 leaves")
    mem = {i: [i] for i in range(K)}
    nodes = [frozenset([i]) for i in range(K)]
    act = list(range(K))
    nxt = K
    while len(act) > 1:
        best = None
        for a, b in itertools.combinations(act, 2):
            ds = [1.0 - S[i, j] for i in mem[a] for j in mem[b]]
            d = float(np.mean(ds))
            key = (d, a, b)
            if best is None or key < best:
                best = key
        _, a, b = best
        mem[nxt] = mem[a] + mem[b]
        nodes.append(frozenset(mem[nxt]))
        act.remove(a)
        act.remove(b)
        act.append(nxt)
        nxt += 1
    return nodes


def specificity(nd, K):
    n = len(nd) if isinstance(nd, (set, frozenset)) else 1
    return (K - n) / (K - 1)


def _mass(v, nd):
    if nd is None:
        return None
    return float(np.asarray(v)[list(nd)].sum())


class TreeBel:
    def __init__(
        self,
        K,
        sim,
        lam=0.5,
        rho=0.05,
        tau=0.8,
        refr=8,
        low=None,
        min_sp=0.25,
    ):
        self.K = K
        self.bel = FlatBel(K, lam, rho)
        self.nodes = build_tree(sim)
        self.tau = float(tau)
        self.low = max(0.0, self.tau - 0.1) if low is None else float(low)
        self.min_sp = float(min_sp)
        self.g = EmitterGate(self.tau, self.low, refr)
        self.refr = self.g.refr

    def pick(self):
        b = self.bel.belief
        out = []
        for nd in self.nodes:
            # skip root
            if len(nd) == self.K:
                continue
            sp = specificity(nd, self.K)
            if sp < self.min_sp:
                continue
            s = float(b[list(nd)].sum())
            if s >= self.tau:
                out.append((len(nd), tuple(sorted(nd)), nd, s))
        if not out:
            return None, 0.0
        _, _, nd, s = min(out)
        return nd, s

    def commit_node(self):
        nd, _ = self.pick()
        return nd

    def run(self, P):
        self.bel.reset()
        self.g.reset()
        self.sc = []
        em = []
        for t, p in enumerate(P):
            b = self.bel.update(p)
            nd, s = self.pick()
            ls = _mass(b, self.g.last_support)
            self.sc.append(s)
            out = self.g.step(t, nd, s, ls)
            if out is not None:
                em.append((t, out))
        return em

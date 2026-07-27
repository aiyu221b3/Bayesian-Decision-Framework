import numpy as np


_EPS = 1e-12


def logsumexp(x):
    """log(sum(exp(x))) stable."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        raise ValueError("logsumexp received no finite values")
    return m + np.log(np.exp(x - m).sum())


def _probability_vector(values, K, name):
    """val & norm 1d prob vec."""
    v = np.asarray(values, dtype=float)
    if v.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},)")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(v < 0):
        raise ValueError(f"{name} cannot contain negative values")
    tot = v.sum()
    if tot <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return v / tot


class FlatBel:
    """tempered cat bayes filter. rho=0 static, rho>0 sticky hmm jump."""

    def __init__(self, K, lam=0.32, rho=0.0, prior=None):
        if K < 2:
            raise ValueError("K must be at least 2")
        if lam < 0:
            raise ValueError("lam cannot be negative")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        self.K = int(K)
        self.lam = float(lam)
        self.rho = float(rho)
        if prior is None:
            p_vec = np.full(self.K, 1.0 / self.K)
        else:
            p_vec = _probability_vector(prior, self.K, "prior")
        self.log_prior = np.log(np.clip(p_vec, _EPS, None))
        self.log_prior -= logsumexp(self.log_prior)
        self.reset()

    def reset(self):
        self.logb = self.log_prior.copy()
        return self.belief

    def update(self, p):
        obs = _probability_vector(p, self.K, "p")
        if self.rho > 0.0:
            prev = self.belief
            pred = (1.0 - self.rho) * prev + self.rho / self.K
            log_pred = np.log(np.clip(pred, _EPS, None))
        else:
            log_pred = self.logb
        log_obs = np.log(np.clip(obs, _EPS, None))
        self.logb = log_pred + self.lam * log_obs
        self.logb -= logsumexp(self.logb)
        return self.belief

    def run(self, p_sequence, reset=True):
        seq = np.asarray(p_sequence, dtype=float)
        if seq.ndim != 2 or seq.shape[1] != self.K:
            raise ValueError(f"p_sequence must have shape (T, {self.K})")
        if reset:
            self.reset()
        if len(seq) == 0:
            return np.empty((0, self.K), dtype=float)
        return np.stack([self.update(p) for p in seq])

    @property
    def belief(self):
        return np.exp(self.logb)

    @property
    def bmax(self):
        return float(self.belief.max())

    @property
    def argmax(self):
        return int(np.argmax(self.logb))


if __name__ == "__main__":
    from evidence import Simcon, concat_stream

    cfg = Simcon()
    rng = np.random.default_rng(1)
    st = concat_stream(cfg, rng, n_signs=2)
    stat = FlatBel(cfg.K, lam=0.5, rho=0.0).run(st["p"])
    sw = FlatBel(cfg.K, lam=0.5, rho=0.05).run(st["p"])
    bnd = st["bounds"][0]
    prv, nw = st["labels"][:2]
    print(f"boundary={bnd}: {prv} -> {nw}")
    prb = min(bnd + 10, len(st["p"]) - 1)
    print("static previous-label belief:", round(float(stat[prb, prv]), 4))
    print("switching previous-label belief:", round(float(sw[prb, prv]), 4))

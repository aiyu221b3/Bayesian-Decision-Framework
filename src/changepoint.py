from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BocpdSnapshot:
    """bocpd step state."""

    cp_posterior: float
    posterior: np.ndarray
    map_run_length: int
    predictive_probability: float
    predictive_surprise: float


class BOCPD:
    """cat bocpd. run length counts & posterior growth/reset."""

    def __init__(self, K, hazard=1 / 30.0, alpha=1.0):
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)):
            raise TypeError("K must be an integer")
        if K < 2:
            raise ValueError("K must be at least 2")
        try:
            hazard = float(hazard)
            alpha = float(alpha)
        except (TypeError, ValueError) as error:
            raise TypeError("hazard and alpha must be real numbers") from error
        if not np.isfinite(hazard) or not 0.0 <= hazard <= 1.0:
            raise ValueError("hazard must be finite and lie in [0, 1]")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.K = int(K)
        self.h = hazard
        self.alpha = alpha
        self.reset()

    def reset(self):
        # all mass at r=0
        self.R = np.array([1.0], dtype=float)
        self.counts = np.zeros((1, self.K), dtype=float)

    def _validate_observation(self, obs):
        if isinstance(obs, bool) or not isinstance(obs, (int, np.integer)):
            raise TypeError("obs must be an integer class index")
        obs = int(obs)
        if not 0 <= obs < self.K:
            raise ValueError(f"obs must lie in [0, {self.K - 1}]")
        return obs

    def update(self, obs):
        obs = self._validate_observation(obs)
        tot = self.counts.sum(axis=1)
        run_p = (self.counts[:, obs] + self.alpha) / (tot + self.K * self.alpha)
        prior_p = 1.0 / self.K
        # cp reset & growth branches
        cp_un = prior_p * self.h * self.R.sum()
        gro_un = self.R * (1.0 - self.h) * run_p
        new_post = np.empty(len(self.R) + 1, dtype=float)
        new_post[0] = cp_un
        new_post[1:] = gro_un
        p_prob = float(new_post.sum())
        if not np.isfinite(p_prob) or p_prob <= 0:
            raise FloatingPointError("BOCPD predictive probability is invalid")
        new_post /= p_prob
        if not np.all(np.isfinite(new_post)) or np.any(new_post < 0):
            raise FloatingPointError("BOCPD produced an invalid posterior")
        new_c = np.zeros((len(self.counts) + 1, self.K), dtype=float)
        new_c[0, obs] = 1.0
        new_c[1:] = self.counts
        new_c[1:, obs] += 1.0
        self.R = new_post
        self.counts = new_c
        map_r = int(np.argmax(self.R))
        cp_p = float(self.R[0])
        surp = float(-np.log(max(p_prob, 1e-300)))
        return BocpdSnapshot(
            cp_posterior=cp_p,
            posterior=self.R.copy(),
            map_run_length=map_r,
            predictive_probability=p_prob,
            predictive_surprise=surp,
        )

    def map_runlength(self):
        return int(np.argmax(self.R))

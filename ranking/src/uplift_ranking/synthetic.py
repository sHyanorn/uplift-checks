"""Synthetic randomised trials with known effects, for tests and timings only."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class Trial:
    x: np.ndarray  # features, float32
    w: np.ndarray  # treatment, 0 or 1
    y: np.ndarray  # binary outcome
    mu0: np.ndarray  # P(Y = 1 | x, control)
    mu1: np.ndarray  # P(Y = 1 | x, treated)

    @property
    def tau(self):
        return self.mu1 - self.mu0


def rct(n, seed, d=6, treated=0.5, base=-1.0):
    """Randomised trial: the outcome depends on x0 and x1, the effect on x2 (and its sign on x3)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    mu0 = expit(base + 0.8 * x[:, 0] - 0.5 * x[:, 1])
    tau = 0.25 * expit(2.0 * x[:, 2]) - 0.05 * (x[:, 3] > 1.5)
    mu1 = np.clip(mu0 + tau, 0.0, 1.0)
    w = (rng.random(n) < treated).astype(np.int64)
    y = (rng.random(n) < np.where(w == 1, mu1, mu0)).astype(np.int64)
    return Trial(x, w, y, mu0, mu1)


def tied(values, rng, levels, noise=0.0):
    """A score from values plus noise, rounded to about `levels` distinct values (heavy ties)."""
    s = values + noise * rng.normal(size=len(values))
    edges = np.quantile(s, np.linspace(0, 1, levels + 1)[1:-1])
    return np.searchsorted(edges, s).astype(np.float64) / levels

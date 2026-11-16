"""The two metric libraries import and compute with the pinned numpy, pandas and scikit-learn."""

import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, qini_score, rate_score
from sklift.metrics import qini_auc_score, uplift_auc_score


def trial(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tau = rng.uniform(-0.1, 0.3, n)
    w = rng.integers(0, 2, n)
    y = (rng.uniform(size=n) < 0.3 + w * tau).astype(int)
    return y, w, tau


def test_scikit_uplift_scores_the_perfect_ranking_as_one():
    y, w, tau = trial()
    assert qini_auc_score(y, y * w - y * (1 - w), w) == 1.0
    assert qini_auc_score(y, tau, w) > qini_auc_score(y, -tau, w)
    assert uplift_auc_score(y, tau, w) > uplift_auc_score(y, -tau, w)


def test_causalml_orders_the_true_effect_above_its_negation():
    y, w, tau = trial()
    for score in (qini_score, auuc_score, rate_score):
        good = score(pd.DataFrame({"y": y, "w": w, "s": tau}))["s"]
        bad = score(pd.DataFrame({"y": y, "w": w, "s": -tau}))["s"]
        assert np.isfinite(good) and np.isfinite(bad) and good > bad

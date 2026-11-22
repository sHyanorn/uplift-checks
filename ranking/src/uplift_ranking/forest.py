"""Causal forest (Section 6): EconML 0.17.0 CausalForestDML with the outcome model of 5.3.

The treatment model is a LogisticRegression, the two folds are EconML's, one job, and
random_state is the fit's seed. On Criteo the forest is fitted on a fixed draw of training rows.
"""

import numpy as np
from econml.dml import CausalForestDML
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from .learners import FIXED, booster

CRITEO_TRAINING_ROWS, CRITEO_FOREST_ROWS = 8_385_214, 1_000_000
SEARCH_TREES, FINAL_TREES = 200, 500


def criteo_rows():
    """Indices into Criteo's training rows (file order) of the forest's rows (Section 6), kept in file
    order; the order of rows sets EconML's folds."""
    draw = np.random.default_rng([20261115, 3]).choice(CRITEO_TRAINING_ROWS, CRITEO_FOREST_ROWS, replace=False)
    return np.sort(draw)


class OutcomeModel(ClassifierMixin, BaseEstimator):
    """LightGBM with the settings of 5.3 and early stopping on the validation rows, as a classifier.

    The class holds the validation rows while a forest is fitted: EconML keeps four copies of the
    estimator, and rows held by an instance would be frozen four times with every forest."""

    valid = None

    def fit(self, x, y, sample_weight=None):
        self.booster_ = booster(x, y, self.valid.x, self.valid.y, FIXED, 0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        p = self.booster_.predict(x)
        return np.c_[1 - p, p]

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] > 0.5).astype(int)


def fit(train, valid, config, seed, trees):
    """Fit on `train` (a learners.Split); `config` holds min_samples_leaf and max_features."""
    forest = CausalForestDML(
        model_y=OutcomeModel(),
        model_t=LogisticRegression(),
        discrete_outcome=True,
        discrete_treatment=True,
        cv=2,
        n_estimators=trees,
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        n_jobs=1,
        random_state=seed,
    )
    OutcomeModel.valid = valid
    try:
        return forest.fit(train.y, train.w, X=train.x)
    finally:
        OutcomeModel.valid = None


def outcome_trees(forest):
    """Trees of each fold's outcome model, chosen by early stopping; the forest's own size is fixed."""
    return {f"outcome fold {f}": model.booster_.best_iteration for f, model in enumerate(forest.models_y[0])}


def effect(forest, x):
    return forest.effect(x).ravel()

"""LightGBM meta-learners, outcome models, nuisances and the R-loss (Sections 5.3, 5.4 and 6).

Every component is fitted on training rows with early stopping on its own loss on the validation
rows. Two folds, where used, are those of `roles.first_fold` with the fit's seed; a cross-fitted
model predicts a training row with the fold model that did not see it, and a validation row with
the mean of both fold models.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from .roles import first_fold

THREADS = 4
MAX_TREES, PATIENCE = 1000, 50
FIXED = {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 100, "feature_fraction": 0.8}
COMMON = {"bagging_fraction": 0.8, "bagging_freq": 1, "deterministic": True, "force_row_wise": True,
          "num_threads": THREADS, "verbosity": -1}
CLIP = (0.01, 0.99)


@dataclass
class Split:
    x: np.ndarray
    w: np.ndarray
    y: np.ndarray

    def rows(self, keep):
        return Split(self.x[keep], self.w[keep], self.y[keep])


def log_loss(p, valid):
    y, p = valid.get_label(), np.clip(p, 1e-15, 1 - 1e-15)
    return "log loss", float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), False


def l2(p, valid):
    weight = valid.get_weight()
    return "l2", float(np.average((valid.get_label() - p) ** 2, weights=weight)), False


def booster(x, target, valid_x, valid_target, config, seed, objective="binary", weight=None, valid_weight=None):
    """One LightGBM component: up to 1,000 trees, early stopping after 50 rounds on the validation loss.

    The loss (Sections 5.3 and 6) is the log-loss, or for a regression component the weighted squared
    error, computed in numpy: LightGBM's own metrics vary in their last bits between identical runs,
    which can move the best iteration where the loss is flat."""
    params = {**COMMON, **config, "objective": objective, "seed": seed, "metric": "None"}
    train = lgb.Dataset(x, target, weight=weight)
    valid = lgb.Dataset(valid_x, valid_target, weight=valid_weight, reference=train)
    return lgb.train(params, train, MAX_TREES, valid_sets=[valid], feval=log_loss if objective == "binary" else l2,
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


@dataclass
class Fitted:
    """Boosters by name and the rule combining their predictions into an effect."""

    kind: str
    boosters: dict = field(default_factory=dict)
    share: float | None = None  # constant e of a learner on Criteo and Hillstrom

    def predict(self, name, x):
        return self.boosters[name].predict(x)

    def mean(self, name, x):
        return (self.predict(f"{name} fold 0", x) + self.predict(f"{name} fold 1", x)) / 2

    def propensity(self, x):
        return np.full(len(x), self.share) if self.share is not None else np.clip(self.mean("e", x), *CLIP)

    def effect(self, x):
        if self.kind == "S":
            return self.predict("S", np.c_[x, np.ones(len(x))]) - self.predict("S", np.c_[x, np.zeros(len(x))])
        if self.kind == "T":
            return self.predict("mu1", x) - self.predict("mu0", x)
        if self.kind == "X":
            e = self.propensity(x)
            return e * self.predict("tau0", x) + (1 - e) * self.predict("tau1", x)
        return self.predict("tau", x)

    @property
    def trees(self):
        return {name: b.best_iteration for name, b in self.boosters.items()}


def cross_fit(fitted, name, train, valid, target, valid_target, config, seed, folds):
    """Fit `name` in two folds; return its out-of-fold training predictions and mean validation predictions."""
    out = np.empty(len(target))
    for f in (0, 1):
        own = folds == (f == 0)
        b = booster(train.x[own], target[own], valid.x, valid_target, config, seed)
        fitted.boosters[f"{name} fold {f}"] = b
        out[~own] = b.predict(train.x[~own])
    return out, fitted.mean(name, valid.x)


def propensity(fitted, train, valid, config, seed, folds, cross_fitted):
    """e of a learner: the training share, or (Lazada) a cross-fitted classifier clipped to [0.01, 0.99].

    The classifier uses the learner's configuration; Section 5.4 clips its e, and a learner's e is
    clipped the same way (Section 6)."""
    if not cross_fitted:
        fitted.share = float(train.w.mean())
        return np.full(len(train.w), fitted.share), np.full(len(valid.w), fitted.share)
    e_train, e_valid = cross_fit(fitted, "e", train, valid, train.w, valid.w, config, seed, folds)
    return np.clip(e_train, *CLIP), np.clip(e_valid, *CLIP)


def fit(kind, train, valid, config, seed, cross_fitted_e):
    """Fit meta-learner S, T, X, R or DR with one grid configuration shared by all its components.

    X (Section 6): the T stage is cross-fitted in the two folds; a training row's imputed effect uses
    the fold model that did not see it, a validation row's the mean of both; the effect is
    e tau0 + (1 - e) tau1."""
    fitted = Fitted(kind)
    folds = first_fold(train.x, seed)
    if kind == "S":
        fitted.boosters["S"] = booster(np.c_[train.x, train.w], train.y, np.c_[valid.x, valid.w], valid.y, config, seed)
        return fitted
    if kind == "T":
        for a in (0, 1):
            t, v = train.rows(train.w == a), valid.rows(valid.w == a)
            fitted.boosters[f"mu{a}"] = booster(t.x, t.y, v.x, v.y, config, seed)
        return fitted
    e_train, e_valid = propensity(fitted, train, valid, config, seed, folds, cross_fitted_e)
    if kind == "R":
        m_train, m_valid = cross_fit(fitted, "m", train, valid, train.y, valid.y, config, seed, folds)
        r_train, r_valid = train.w - e_train, valid.w - e_valid
        fitted.boosters["tau"] = booster(train.x, (train.y - m_train) / r_train, valid.x, (valid.y - m_valid) / r_valid,
                                         config, seed, "regression", r_train**2, r_valid**2)
        return fitted
    mu_train, mu_valid = {}, {}
    for a in (0, 1):
        # An arm's fold model is fitted on that arm's rows of one fold and predicts the other fold's rows.
        mu_train[a] = np.empty(len(train.w))
        v = valid.rows(valid.w == a)
        for f in (0, 1):
            own = folds == (f == 0)
            fit_rows = own & (train.w == a)
            b = fitted.boosters[f"mu{a} fold {f}"] = booster(train.x[fit_rows], train.y[fit_rows], v.x, v.y, config, seed)
            mu_train[a][~own] = b.predict(train.x[~own])
        mu_valid[a] = fitted.mean(f"mu{a}", valid.x)
    if kind == "X":
        for a, sign in ((1, 1.0), (0, -1.0)):
            t, v = train.w == a, valid.w == a
            d_train = sign * (train.y[t] - mu_train[1 - a][t])
            d_valid = sign * (valid.y[v] - mu_valid[1 - a][v])
            fitted.boosters[f"tau{a}"] = booster(train.x[t], d_train, valid.x[v], d_valid, config, seed, "regression")
        return fitted
    pseudo = [aipw(s.w, s.y, mu[0], mu[1], e) for s, mu, e in ((train, mu_train, e_train), (valid, mu_valid, e_valid))]
    fitted.boosters["tau"] = booster(train.x, pseudo[0], valid.x, pseudo[1], config, seed, "regression")
    return fitted


def aipw(w, y, mu0, mu1, e):
    return mu1 - mu0 + w * (y - mu1) / e - (1 - w) * (y - mu0) / (1 - e)


def outcome_models(train, valid, arms):
    """5.3: one classifier per arm with the fixed settings and seed 0; `arms` maps a name to the arm codes it pools."""
    out = {}
    for name, codes in arms.items():
        t, v = train.rows(np.isin(train.w, codes)), valid.rows(np.isin(valid.w, codes))
        out[name] = booster(t.x, t.y, v.x, v.y, FIXED, 0)
    return out


def nuisances(train, valid, cross_fitted_e):
    """5.4: m-hat on all training rows and e (training share, or two fold classifiers), seed 0.

    Returns the fitted models and their validation predictions."""
    fitted = Fitted("nuisances")
    fitted.boosters["m"] = booster(train.x, train.y, valid.x, valid.y, FIXED, 0)
    _, e = propensity(fitted, train, valid, FIXED, 0, first_fold(train.x, 0), cross_fitted_e)
    return fitted, fitted.predict("m", valid.x), e


def r_loss(effect, y, w, m, e):
    """Mean of [(Y - m(X)) - (W - e(X)) tau(X)]^2 over the validation rows; inf if any prediction is not finite."""
    if not np.all(np.isfinite(effect)):
        return float("inf")
    return float(np.mean(((y - m) - (w - e) * effect) ** 2))

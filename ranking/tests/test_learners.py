import numpy as np
import pytest

from uplift_ranking import learners
from uplift_ranking.learners import Split
from uplift_ranking.synthetic import rct

CONFIG = {"num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 100, "feature_fraction": 1.0, "lambda_l2": 0.0}
KINDS = ("S", "T", "X", "R", "DR")


def splits(n, seed, **kw):
    t, v = rct(n, seed, **kw), rct(n // 2, seed + 100, **kw)
    return Split(t.x, t.w, t.y), Split(v.x, v.w, v.y), v.tau


@pytest.fixture(scope="module")
def imbalanced():
    return splits(30000, 1, treated=0.2)


@pytest.fixture(scope="module")
def selective():
    return splits(30000, 2, treated=0.3, selective=1.0)


@pytest.mark.parametrize("kind", KINDS)
def test_effects_track_the_true_effect(kind, imbalanced, selective):
    for (train, valid, tau), cross_fitted in ((imbalanced, False), (selective, True)):
        fitted = learners.fit(kind, train, valid, CONFIG, seed=0, cross_fitted_e=cross_fitted)
        assert np.corrcoef(fitted.effect(valid.x), tau)[0, 1] > 0.6, (kind, cross_fitted)
        assert all(t > 0 for t in fitted.trees.values())


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_repeats_and_another_seed_differs(kind, imbalanced):
    train, valid, _ = imbalanced
    first = learners.fit(kind, train, valid, CONFIG, seed=3, cross_fitted_e=False).effect(valid.x)
    assert np.array_equal(first, learners.fit(kind, train, valid, CONFIG, seed=3, cross_fitted_e=False).effect(valid.x))
    assert not np.array_equal(first, learners.fit(kind, train, valid, CONFIG, seed=4, cross_fitted_e=False).effect(valid.x))


class Recorder:
    """Stands in for a LightGBM component: remembers its training rows and refuses to predict them."""

    log = []
    value = 0.4

    def __init__(self, x, target, valid_x, valid_target, config, seed, objective="binary", weight=None, valid_weight=None):
        self.rows = {r.tobytes() for r in x}
        self.target, self.weight, self.valid_target, self.valid_weight = target, weight, valid_target, valid_weight
        self.best_iteration = 1
        self.objective = objective
        Recorder.log.append(self)

    def predict(self, x):
        assert not any(r.tobytes() in self.rows for r in x), "a model predicted a row it was trained on"
        return np.full(len(x), self.value)


@pytest.mark.parametrize("kind", ("X", "R", "DR"))
def test_cross_fitted_predictions_never_use_their_own_row(kind, monkeypatch):
    train, valid, _ = splits(3000, 5, treated=0.3, selective=1.0)
    monkeypatch.setattr(learners, "booster", Recorder)
    Recorder.log = []
    fitted = learners.fit(kind, train, valid, CONFIG, seed=1, cross_fitted_e=True)
    folds = learners.first_fold(train.x, 1)
    for name, b in fitted.boosters.items():
        if " fold " in name:
            f = int(name[-1])
            assert b.rows <= {r.tobytes() for r in train.x[folds == (f == 0)]}, name
            if name.startswith("mu"):
                arm = int(name[2])
                assert b.rows == {r.tobytes() for r in train.x[(folds == (f == 0)) & (train.w == arm)]}, name


def test_outcome_models_pool_the_named_arms(monkeypatch):
    train, valid, _ = splits(3000, 6)
    arms = np.random.default_rng(0).integers(0, 3, 3000)
    monkeypatch.setattr(learners, "booster", Recorder)
    models = learners.outcome_models(Split(train.x, arms, train.y), Split(valid.x, arms[:1500], valid.y), {"mu0": [0], "mu1": [1, 2]})
    assert models["mu0"].rows == {r.tobytes() for r in train.x[arms == 0]}
    assert models["mu1"].rows == {r.tobytes() for r in train.x[arms > 0]}


def test_nuisances(imbalanced, selective):
    train, valid, _ = imbalanced
    _, m, e = learners.nuisances(train, valid, cross_fitted_e=False)
    assert np.all(e == train.w.mean()) and len(m) == len(valid.y)
    train, valid, _ = selective
    fitted, _, e = learners.nuisances(train, valid, cross_fitted_e=True)
    assert set(fitted.boosters) == {"m", "e fold 0", "e fold 1"}
    assert np.array_equal(e, np.clip(fitted.mean("e", valid.x), 0.01, 0.99))
    assert np.corrcoef(e, valid.x[:, 1])[0, 1] > 0.5


def test_r_loss_prefers_the_true_effect(selective):
    train, valid, tau = selective
    _, m, e = learners.nuisances(train, valid, cross_fitted_e=True)
    loss = lambda t: learners.r_loss(t, valid.y, valid.w, m, e)
    assert loss(tau) < loss(np.zeros_like(tau)) < loss(-tau)
    assert loss(np.r_[tau[:-1], np.nan]) == float("inf")


def test_pseudo_outcomes_follow_their_formulas(monkeypatch):
    """Stand-ins fitted on control rows only predict 0.3, on treated rows only 0.6 and on both arms 0.4,
    so the regressions' targets tell the arms apart; e is the training share."""
    train, valid, _ = splits(2000, 7, treated=0.3)
    arm = {r.tobytes(): a for r, a in zip(train.x, train.w)}

    class ByArm(Recorder):
        def __init__(self, x, *args, **kwargs):
            super().__init__(x, *args, **kwargs)
            self.value = {(0,): 0.3, (1,): 0.6}.get(tuple(sorted({arm[r.tobytes()] for r in x})), 0.4)

    monkeypatch.setattr(learners, "booster", ByArm)
    e = train.w.mean()
    r = learners.fit("R", train, valid, CONFIG, seed=0, cross_fitted_e=False).boosters["tau"]
    assert np.allclose(r.target, (train.y - 0.4) / (train.w - e)) and np.allclose(r.weight, (train.w - e) ** 2)
    assert np.allclose(r.valid_target, (valid.y - 0.4) / (valid.w - e)) and np.allclose(r.valid_weight, (valid.w - e) ** 2)
    dr = learners.fit("DR", train, valid, CONFIG, seed=0, cross_fitted_e=False).boosters["tau"]
    assert np.allclose(dr.target, 0.6 - 0.3 + train.w * (train.y - 0.6) / e - (1 - train.w) * (train.y - 0.3) / (1 - e))
    x = learners.fit("X", train, valid, CONFIG, seed=0, cross_fitted_e=False).boosters
    assert np.allclose(x["tau1"].target, train.y[train.w == 1] - 0.3)
    assert np.allclose(x["tau0"].target, 0.6 - train.y[train.w == 0])


class Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return self.value(x) if callable(self.value) else np.full(len(x), self.value)


def test_effect_rules():
    x = np.zeros((3, 2))
    s = learners.Fitted("S", {"S": Constant(lambda z: 0.1 + 0.5 * z[:, -1])})
    assert np.allclose(s.effect(x), 0.5)
    t = learners.Fitted("T", {"mu1": Constant(0.7), "mu0": Constant(0.2)})
    assert np.allclose(t.effect(x), 0.5)
    xl = learners.Fitted("X", {"tau0": Constant(1.0), "tau1": Constant(2.0)}, share=0.3)
    assert np.allclose(xl.effect(x), 0.3 * 1.0 + 0.7 * 2.0)
    xl = learners.Fitted("X", {"tau0": Constant(1.0), "tau1": Constant(2.0), "e fold 0": Constant(0.0), "e fold 1": Constant(0.002)})
    assert np.allclose(xl.effect(x), 0.01 * 1.0 + 0.99 * 2.0)


def test_cross_fitted_propensity_is_clipped(monkeypatch):
    train, valid, _ = splits(2000, 8, treated=0.3, selective=1.0)
    monkeypatch.setattr(learners, "booster", Recorder)
    monkeypatch.setattr(Recorder, "value", 0.001)
    r = learners.fit("R", train, valid, CONFIG, seed=0, cross_fitted_e=True).boosters["tau"]
    assert np.allclose(r.weight, (train.w - 0.01) ** 2) and np.allclose(r.valid_weight, (valid.w - 0.01) ** 2)


def test_early_stopping_is_reproducible_where_the_loss_is_flat(imbalanced):
    """With min_child_samples above the rows no split is possible and the validation loss is flat."""
    train, valid, _ = imbalanced
    small = train.rows(np.arange(800))
    fits = [learners.booster(small.x, small.y, valid.x, valid.y, dict(CONFIG, min_child_samples=2000), 0) for _ in range(6)]
    assert {b.best_iteration for b in fits} == {1}
    import pickle
    assert len({pickle.dumps(b) for b in fits}) == 1


def test_validation_losses_are_those_of_the_component(imbalanced):
    train, valid, _ = imbalanced
    b = learners.booster(train.x, train.y, valid.x, valid.y, CONFIG, 0)
    p = np.clip(b.predict(valid.x), 1e-15, 1 - 1e-15)
    assert b.best_score["valid_0"]["log loss"] == pytest.approx(-np.mean(valid.y * np.log(p) + (1 - valid.y) * np.log(1 - p)), rel=1e-8)
    weight = np.linspace(0.1, 2, len(valid.y))
    r = learners.booster(train.x, train.y - 0.3, valid.x, valid.y - 0.3, CONFIG, 0, "regression", np.ones(len(train.y)), weight)
    assert r.best_score["valid_0"]["l2"] == pytest.approx(np.average((valid.y - 0.3 - r.predict(valid.x)) ** 2, weights=weight), rel=1e-8)

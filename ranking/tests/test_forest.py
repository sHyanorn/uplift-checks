import pickle

import numpy as np
import pytest

from uplift_ranking import forest
from uplift_ranking.learners import Split
from uplift_ranking.synthetic import rct

CONFIG = {"min_samples_leaf": 50, "max_features": 1.0}
pytestmark = pytest.mark.filterwarnings("ignore:Could not generate out-of-bag:UserWarning")


@pytest.fixture(scope="module")
def data():
    t, v = rct(8000, 1, treated=0.3), rct(3000, 2, treated=0.3)
    return Split(t.x, t.w, t.y), Split(v.x, v.w, v.y), v.tau


@pytest.fixture(scope="module")
def fitted(data):
    train, valid, _ = data
    return forest.fit(train, valid, CONFIG, seed=0, trees=40)


def test_criteo_rows_follow_the_registered_draw():
    rows = forest.criteo_rows()
    draw = np.random.default_rng([20261115, 3]).choice(8385214, 1000000, replace=False)
    assert np.array_equal(rows, np.sort(draw))
    assert len(np.unique(rows)) == 1_000_000 and rows[-1] < 8_385_214


def test_effect_tracks_the_true_effect(data, fitted):
    _, valid, tau = data
    assert np.corrcoef(forest.effect(fitted, valid.x), tau)[0, 1] > 0.5


def test_refits_repeat_exactly_with_one_job(data, fitted):
    train, valid, _ = data
    first = forest.effect(fitted, valid.x)
    assert np.array_equal(first, forest.effect(forest.fit(train, valid, CONFIG, seed=0, trees=40), valid.x))
    assert not np.array_equal(first, forest.effect(forest.fit(train, valid, CONFIG, seed=1, trees=40), valid.x))


def test_settings_reach_econml(fitted):
    assert fitted.n_jobs == 1 and fitted.cv == 2 and fitted.n_estimators == 40
    assert fitted.min_samples_leaf == 50 and fitted.max_features == 1.0 and fitted.random_state == 0
    outcome = [m for fold in fitted.models_y for m in fold]
    assert len(outcome) == 2 and all(isinstance(m, forest.OutcomeModel) for m in outcome)
    assert all(0 < m.booster_.best_iteration < 1000 for m in outcome)
    assert all(m.booster_.params["num_leaves"] == 63 and m.booster_.params["seed"] == 0 for m in outcome)


def test_a_pickled_forest_holds_no_rows(data, fitted):
    train, valid, _ = data
    raw = pickle.dumps(fitted, protocol=5)
    assert train.x.tobytes()[:4096] not in raw and valid.x.tobytes()[:4096] not in raw
    assert np.array_equal(forest.effect(pickle.loads(raw), valid.x), forest.effect(fitted, valid.x))

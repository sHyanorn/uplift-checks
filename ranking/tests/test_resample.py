import numpy as np
import pytest

from uplift_ranking import metrics, resample
from uplift_ranking.synthetic import rct, tied


def duplicated_trial(n, seed):
    """A trial where about a fifth of the rows repeat an earlier row's features (resampling groups)."""
    t = rct(n, seed, treated=0.35)
    rng = np.random.default_rng(seed)
    source = rng.integers(0, n, n)
    repeat = rng.random(n) < 0.2
    source[~repeat] = np.arange(n)[~repeat]
    source = np.minimum(source, np.arange(n))
    _, first, group = np.unique(t.x[source], axis=0, return_index=True, return_inverse=True)
    group = np.argsort(np.argsort(first))[group]  # numbered by first row in file order
    return t, source, group.ravel()


def subset_values(score, t, rows, perm):
    """Section 5.2 and H5 statistics computed from their definitions on the given rows."""
    w, y, s = t.w[rows], t.y[rows], score[rows]
    gamma = metrics.aipw(w, y, t.mu0[rows], t.mu1[rows])
    index = np.full(len(score), -1)
    index[rows] = np.arange(len(rows))
    sub_perm = index[perm][index[perm] >= 0]
    own = metrics.own_versions(s, w, y, gamma, sub_perm)
    budget = [metrics.budget_value(s, g, k) for g in (gamma, metrics.ipw(w, y)) for k in resample.BUDGETS]
    return np.array([own[v] for v in resample.STATISTICS[:8]] + budget)


def test_every_replicate_equals_the_definitions_on_its_rows():
    t, source, group = duplicated_trial(2500, 4)
    rng = np.random.default_rng(9)
    scores = [
        tied(t.tau[source], rng, 3, noise=0.05),
        tied(t.tau[source], rng, 40, noise=0.05),
        t.tau[source] + 0.05 * rng.normal(size=2500),
    ]
    perm = np.random.default_rng([20261115, 4, 3]).permutation(2500)
    models = [resample.Ranked(s, t.w, t.y, t.mu0, t.mu1, perm) for s in scores]
    values = resample.replicate_values(models, group, unit=3, replicates=25)
    assert values.shape == (25, 3, len(resample.STATISTICS))
    for r in range(25):
        rows = np.flatnonzero(resample.replicate_rows(group, 3, r))
        for j, s in enumerate(scores):
            np.testing.assert_allclose(values[r, j], subset_values(s, t, rows, perm), rtol=1e-9, atol=1e-13)
    full = np.arange(2500)
    for j, s in enumerate(scores):
        np.testing.assert_allclose(resample.point_values(models)[j], subset_values(s, t, full, perm), rtol=1e-9, atol=1e-13)


def test_parallel_replicates_equal_serial_ones():
    t = rct(1500, 2)
    group = np.arange(1500)
    models = [resample.Ranked(t.tau + 0.1 * t.x[:, j], t.w, t.y, t.mu0, t.mu1, np.arange(1500)) for j in range(2)]
    serial = resample.replicate_values(models, group, unit=1, replicates=12)
    assert np.array_equal(serial, resample.replicate_values(models, group, unit=1, replicates=12, processes=3))


def test_replicates_draw_half_the_groups_with_the_registered_generator():
    _, _, group = duplicated_trial(2000, 1)
    n_groups = group.max() + 1
    assert np.array_equal(np.unique(group[np.unique(group, return_index=True)[1]]), np.arange(n_groups))
    for r in (0, 1, 1999):
        rows = resample.replicate_rows(group, 2, r)
        drawn = np.random.default_rng([20261115, 1, 2, r]).choice(n_groups, n_groups // 2, replace=False)
        assert np.array_equal(np.flatnonzero(rows), np.flatnonzero(np.isin(group, drawn)))
        assert all(rows[group == g].all() or not rows[group == g].any() for g in range(n_groups))
    assert not np.array_equal(resample.replicate_rows(group, 2, 0), resample.replicate_rows(group, 3, 0))


def brute_critical_value(point, values):
    se = [np.sqrt(sum((v - np.mean(values[:, p])) ** 2 for v in values[:, p]) / (len(values) - 1)) for p in range(len(point))]
    maxima = sorted(
        max(abs(values[r, p] - point[p]) / se[p] for p in range(len(point)) if se[p] > 0) for r in range(len(values))
    )
    h = 0.95 * (len(maxima) - 1)
    return maxima[int(h)] + (h - int(h)) * (maxima[int(h) + 1] - maxima[int(h)])


def test_critical_value_is_the_95th_percentile_of_the_max_t():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(301, 6)) @ rng.normal(size=(6, 6)) * 0.01
    point = values.mean(axis=0) + rng.normal(size=6) * 0.001
    c, se = resample.critical_value(point, values)
    assert c == pytest.approx(brute_critical_value(point, values), rel=1e-12)
    assert 1.96 < c < 3.0
    with_zero = np.c_[values, np.zeros(301)]
    c0, se0 = resample.critical_value(np.r_[point, 0.0], with_zero)
    assert c0 == c and se0[-1] == 0
    assert list(resample.identical(np.r_[point, 0.0], se0)) == [False] * 6 + [True]


def test_seed_rule_and_separation():
    assert resample.seed_rule(np.array([0.1, 0.2, 0.05, 0.3, 0.01]))
    assert not resample.seed_rule(np.array([0.1, 0.2, -0.05, 0.3, 0.01]))
    assert not resample.seed_rule(np.array([0.0, 0.2, 0.05, 0.3, 0.01]))
    assert resample.seed_rule(np.zeros(5))
    seeds = np.array([0.5, 0.4, 0.6, 0.5, 0.5])
    assert resample.separated(0.5, 0.2, 2.4, seeds)
    assert not resample.separated(0.5, 0.21, 2.4, seeds)
    assert not resample.separated(0.5, 0.2, 2.4, np.array([0.5, 0.4, -0.1, 0.5, 0.5]))


def test_detectable_differences():
    assert resample.mdd(0.01) == pytest.approx(0.028)
    assert resample.mdd(0.01, c=2.9) == pytest.approx(0.0374)
    assert resample.at_size(0.028, groups=4_000_000, n=1_000_000) == pytest.approx(0.056)


def test_half_sample_intervals_cover_at_the_nominal_rate():
    """Over independent trials, the marginal 95% interval of a paired V6 difference covers its value on a very large sample."""
    def difference(t, rows=None):
        rows = np.arange(len(t.w)) if rows is None else rows
        gamma = metrics.aipw(t.w[rows], t.y[rows], t.mu0[rows], t.mu1[rows])
        return metrics.rate_qini(t.tau[rows], gamma) - metrics.rate_qini(t.x[rows, 0].astype(float), gamma)

    truth = difference(rct(1_000_000, 0))
    covered, ses, points = [], [], []
    for seed in range(1, 201):
        t = rct(1000, seed)
        models = [resample.Ranked(s, t.w, t.y, t.mu0, t.mu1, np.arange(1000)) for s in (t.tau, t.x[:, 0].astype(float))]
        v6 = resample.STATISTICS.index("V6")
        d = resample.replicate_values(models, np.arange(1000), unit=1, replicates=50)[:, :, v6] @ [1.0, -1.0]
        point, se = difference(t), resample.standard_error(d)
        covered.append(abs(point - truth) <= 1.96 * se)
        ses.append(se)
        points.append(point)
    assert 0.92 <= np.mean(covered) <= 0.98
    assert 0.85 <= np.mean(ses) / np.std(points, ddof=1) <= 1.15


def test_a_non_finite_score_is_not_ranked():
    t = rct(300, 2)
    score = t.tau.copy()
    score[40] = np.nan
    with pytest.raises(ValueError, match="a score is NaN or infinite"):
        resample.Ranked(score, t.w, t.y, t.mu0, t.mu1, np.arange(300))

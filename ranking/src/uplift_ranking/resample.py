"""Half-sample replicates, standard errors, simultaneous intervals and detectable differences (Section 5.5).

Each model is ranked once on the full set of rows. A replicate keeps a subset of the rows in that
ranking, and every statistic is recomputed on it (including the treated share and the perfect
curve of V2*) from cumulative sums over the kept rows.
"""

import multiprocessing

import numpy as np

from .metrics import area, permuted_order, ratio, tie_blocks

REPLICATES = 2000
BUDGETS = (0.1, 0.3, 0.5)
STATISTICS = ("V2*", "V3", "V4", "V5", "V6", "V9", "V10", "V11") + tuple(
    f"{scores} {k}" for scores in ("AIPW", "IPW") for k in BUDGETS
)
PERCENT = np.arange(1, 101) / 100


def replicate_rows(group, unit, r):
    """Rows of replicate r of unit u: the groups default_rng([20261115, 1, u, r]).choice(G, G // 2, replace=False)."""
    n_groups = int(group.max()) + 1
    chosen = np.zeros(n_groups, dtype=bool)
    chosen[np.random.default_rng([20261115, 1, unit, r]).choice(n_groups, n_groups // 2, replace=False)] = True
    return chosen[group]


def kept_ends(kept, ends):
    """For the blocks ending at `ends` that keep a row: the index in `kept` of their last kept row."""
    last = np.searchsorted(kept, ends, side="right") - 1
    return last[np.diff(last, prepend=-1) > 0]


class Ranked:
    """One model's score on a fixed set of rows, ranked once for evaluation on any subset of them."""

    def __init__(self, score, w, y, mu0, mu1, perm):
        w, y = w.astype(float), y.astype(float)
        self.order, self.ends = tie_blocks(score)
        columns = np.stack([w, w * y, (1 - w) * y, mu1 - mu0, w * (y - mu1), (1 - w) * (y - mu0)])
        self.columns = columns[:, self.order]
        self.order4 = permuted_order(score, perm)
        self.columns4 = columns[:3, self.order4]
        self.arms = []
        for arm in (1, 0):
            rows = np.flatnonzero(w == arm)
            order, ends = tie_blocks(score[rows])
            self.arms.append((rows[order], ends, y[rows[order]]))

    def statistics(self, rows):
        """The values of STATISTICS on the rows where `rows` (boolean, file order) is True."""
        kept = np.flatnonzero(rows[self.order])
        at = kept_ends(kept, self.ends)
        n_t, y_t, y_c, s_a, s_b, s_c = np.cumsum(self.columns[:, kept], axis=1)[:, at]
        k = at + 1.0
        n, treated = k[-1], n_t[-1]
        n_c, e = k - n_t, treated / n

        q = (y_t - y_c * ratio(n_t, n_c)) / n
        v3 = area(k / n, q)
        responders_t, responders_c = y_t[-1], y_c[-1]
        perfect_k = np.array([responders_t, n - responders_c, n])
        perfect_q = np.array([responders_t, responders_t, responders_t - responders_c * ratio(treated, n - treated)]) / n
        v9 = area(k / n, (ratio(y_t, n_t) - ratio(y_c, n_c)) * k / n)

        # Tie blocks at their mean gamma, summed row by row in closed form: block b holds rows
        # before + 1, ..., before + size, where the cumulative sum is g_before + j * mean.
        g = s_a + s_b / e - s_c / (1 - e)
        size = np.diff(k, prepend=0)
        before, g_before = k - size, g - np.diff(g, prepend=0)
        mean, g_all = (g - g_before) / size, g[-1] / n
        triangle = size * (size + 1) / 2
        v6 = np.sum(size * g_before + mean * triangle - g_all * (size * before + triangle)) / n**2
        harmonic = np.diff(np.cumsum(1.0 / np.arange(1, len(kept) + 1))[at], prepend=0)
        v11 = np.sum(size * mean + (g_before - before * mean) * harmonic - size * g_all) / n

        kept4 = np.flatnonzero(rows[self.order4])
        n_t4, y_t4, y_c4 = np.cumsum(self.columns4[:, kept4], axis=1)
        k4 = np.arange(1.0, len(kept4) + 1)
        v4 = area(k4 / n, (y_t4 - y_c4 * ratio(n_t4, k4 - n_t4)) / n)

        v = np.zeros(100)
        for (arm_rows, ends, y_arm), sign in zip(self.arms, (1.0, -1.0)):
            kept_arm = np.flatnonzero(rows[arm_rows])
            last = kept_ends(kept_arm, ends)
            responders = np.cumsum(y_arm[kept_arm])[last]
            v += sign * ratio(np.interp(PERCENT * len(kept_arm), np.r_[0, last + 1], np.r_[0.0, responders]), len(kept_arm))
        v10 = np.mean(v - PERCENT * v[-1])

        budget = []
        for gamma in (g, y_t / e - y_c / (1 - e)):
            budget += list(np.interp(np.array(BUDGETS) * n, np.r_[0, k], np.r_[0.0, gamma]) / n)
        return np.array([v3 / area(perfect_k / n, perfect_q), v3, v4, ratio(v3, q[-1]), v6, v9, v10, v11, *budget])


_models = None


def _replicates(job):
    group, unit, start, stop = job
    out = np.empty((stop - start, len(_models), len(STATISTICS)))
    for i, r in enumerate(range(start, stop)):
        rows = replicate_rows(group, unit, r)
        out[i] = [model.statistics(rows) for model in _models]
    return out


def replicate_values(models, group, unit, replicates=REPLICATES, processes=1):
    """Array (replicate, model, statistic) for a list of Ranked models on the same rows.

    Workers are forked so that they share the ranked columns with this process; they run numpy
    only, and every replicate's value is fixed by its index, whatever the number of processes.
    """
    global _models
    _models = models
    bounds = np.linspace(0, replicates, 4 * processes + 1).astype(int)
    jobs = [(group, unit, a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    try:
        if processes == 1:
            return np.concatenate([_replicates(job) for job in jobs])
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            return np.concatenate(pool.map(_replicates, jobs))
    finally:
        _models = None


def point_values(models):
    return np.array([model.statistics(np.ones(len(model.order), dtype=bool)) for model in models])


def standard_error(values):
    """Standard deviation over replicates (ddof 1), without rescaling."""
    return np.std(values, axis=0, ddof=1)


def critical_value(point, values):
    """c* of a family: 95th percentile over replicates of max_p |d*_rp - d_p| / SE_p, differences with SE 0 left out.

    `point` holds the family's values (P,), `values` their replicates (R, P). Returns c* and the SEs.
    """
    se = standard_error(values)
    live = se > 0
    if not live.any():
        return 0.0, se
    t = np.abs(values[:, live] - point[live]) / se[live]
    return float(np.percentile(t.max(axis=1), 95)), se


def interval(point, se, c):
    return point - c * se, point + c * se


def identical(point, se):
    return (se == 0) & (point == 0)


def seed_rule(values):
    """Five values of a difference (seeds 0 to 4): all have the sign of the seed-0 value (Section 5.5).

    A seed-0 value of exactly 0 holds only if all five are 0."""
    return bool(np.all(np.sign(values) == np.sign(values[0])))


def separated(point, se, c, seeds):
    low, high = interval(point, se, c)
    return (low > 0 or high < 0) and seed_rule(seeds)


def mdd(se, c=None):
    """2.80 SE, or (c* + 0.84) SE within a family."""
    return (2.80 if c is None else c + 0.84) * se


def at_size(value, groups, n):
    """A detectable difference of our test set (G resampling groups) at another test size n."""
    return value * np.sqrt(groups / n)

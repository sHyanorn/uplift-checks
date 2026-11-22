import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from causalml.metrics import auuc_score, qini_score, rate_score
from sklift.metrics import qini_auc_score, uplift_auc_score

from uplift_ranking import metrics as m
from uplift_ranking.synthetic import rct, tied


def sample(n, seed, levels, treated=0.5):
    """A trial and a score with about `levels` distinct values (None: no ties)."""
    t = rct(n, seed, treated=treated)
    rng = np.random.default_rng(seed + 1000)
    score = t.tau + 0.05 * rng.normal(size=n) if levels is None else tied(t.tau, rng, levels, noise=0.05)
    return t, score


# Definitions written row by row from Section 5 of the registration, O(n^2).

def slow_blocks(score):
    return [[i for i in range(len(score)) if score[i] == v] for v in sorted(set(score.tolist()), reverse=True)]


def slow_counts(rows, w, y):
    n_t = sum(w[i] for i in rows)
    y_t = sum(y[i] for i in rows if w[i] == 1)
    y_c = sum(y[i] for i in rows if w[i] == 0)
    return n_t, len(rows) - n_t, y_t, y_c


def slow_q(rows, w, y):
    n_t, n_c, y_t, y_c = slow_counts(rows, w, y)
    return y_t - (y_c * n_t / n_c if n_c else 0.0)


def slow_u(rows, w, y):
    n_t, n_c, y_t, y_c = slow_counts(rows, w, y)
    return ((y_t / n_t if n_t else 0.0) - (y_c / n_c if n_c else 0.0)) * len(rows)


def slow_area(blocks, curve, w, y):
    """Curve at block ends, joined linearly; integral of c(phi) - phi c(1) by the trapezoid rule over every row."""
    n = sum(len(b) for b in blocks)
    at = {0: 0.0}
    for b in range(len(blocks)):
        top = [i for block in blocks[: b + 1] for i in block]
        at[len(top)] = curve(top, w, y) / n
    ends = sorted(at)
    c = [np.interp(k, ends, [at[e] for e in ends]) for k in range(n + 1)]
    return sum((c[k - 1] + c[k]) / 2 for k in range(1, n + 1)) / n - c[n] / 2


def slow_separate(score, w, y):
    v = np.zeros(100)
    for arm, sign in ((1, 1), (0, -1)):
        rows = [i for i in range(len(w)) if w[i] == arm]
        blocks = [[rows[j] for j in b] for b in slow_blocks(score[rows])]
        for p in range(1, 101):
            cut, taken, responders = p * len(rows) / 100, 0, 0.0
            for b in blocks:
                part = min(len(b), cut - taken)
                if part <= 0:
                    break
                responders += part / len(b) * sum(y[i] for i in b)
                taken += part
            v[p - 1] += sign * responders / len(rows)
    return sum(v[p - 1] - p / 100 * v[99] for p in range(1, 101)) / 100


def slow_toc(score, gamma):
    n = len(score)
    value = np.empty(n)
    for b in slow_blocks(score):
        value[b] = np.mean(gamma[b])
    ranked = [value[i] for b in slow_blocks(score) for i in b]
    return [sum(ranked[:k]) / k - np.mean(gamma) for k in range(1, n + 1)]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("levels", [1, 3, 12, None])
def test_own_versions_follow_the_definitions(seed, levels):
    t, score = sample(240, seed, levels, treated=0.3 + 0.1 * seed)
    w, y = t.w, t.y
    gamma = m.aipw(w, y, t.mu0, t.mu1)
    blocks = slow_blocks(score)
    v3 = slow_area(blocks, slow_q, w, y)
    assert m.qini(score, w, y) == pytest.approx(v3, abs=1e-12)
    assert m.qini_share(score, w, y) == pytest.approx(v3 / (slow_q(range(len(w)), w, y) / len(w)), abs=1e-12)
    assert m.uplift_joint(score, w, y) == pytest.approx(slow_area(blocks, slow_u, w, y), abs=1e-12)
    assert m.uplift_separate(score, w, y) == pytest.approx(slow_separate(score, w, y), abs=1e-12)
    toc = slow_toc(score, gamma)
    n = len(toc)
    assert m.rate_qini(score, gamma) == pytest.approx(sum(k / n * toc[k - 1] for k in range(1, n + 1)) / n, abs=1e-12)
    assert m.rate_autoc(score, gamma) == pytest.approx(sum(toc) / n, abs=1e-12)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(len(w))
    position = np.argsort(perm)
    ranked = sorted(range(len(w)), key=lambda i: (-score[i], position[i]))
    assert m.qini(score, w, y, perm) == pytest.approx(slow_area([[i] for i in ranked], slow_q, w, y), abs=1e-12)


def test_first_rows_of_one_arm_use_the_zero_ratio():
    w, y = np.array([1, 1, 0, 0, 1, 0]), np.array([1, 0, 1, 0, 1, 1])
    score = np.array([6.0, 5, 4, 3, 2, 1])
    blocks = slow_blocks(score)
    assert m.qini(score, w, y) == pytest.approx(slow_area(blocks, slow_q, w, y), abs=1e-15)
    assert m.uplift_joint(score, w, y) == pytest.approx(slow_area(blocks, slow_u, w, y), abs=1e-15)
    assert list(m.ratio([1.0, 2.0], [0.0, 4.0])) == [0.0, 0.5]


@pytest.mark.parametrize("seed", range(50))
def test_v2_star_equals_scikit_uplift(seed):
    t, score = sample(400 + 37 * seed, seed, levels=[1, 2, 4, 8, 30][seed % 5])
    assert abs(m.qini_scaled(score, t.w, t.y) - qini_auc_score(t.y, score, t.w, negative_effect=True)) <= 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_v9_over_scikit_uplift_perfect_curve_is_v8(seed):
    t, score = sample(400 + 37 * seed, seed, levels=[2, 5, 30, None][seed % 4], treated=0.2 + 0.06 * seed)
    w, y = t.w, t.y
    # sklift 0.5.1 perfect_uplift_curve: score 2 (y == w) + (y if control responders outnumber treated non-responders else w)
    summand = y if np.sum((y == 1) & (w == 0)) > np.sum((y == 0) & (w == 1)) else w
    v8 = m.uplift_joint(score, w, y) / m.uplift_joint(2.0 * (y == w) + summand, w, y)
    assert abs(v8 - uplift_auc_score(y, score, w)) <= 1e-12


def test_grouped_versions_ignore_the_order_of_rows():
    t, score = sample(3000, 7, levels=6)
    gamma = m.aipw(t.w, t.y, t.mu0, t.mu1)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(3000)
    before = m.own_versions(score, t.w, t.y, gamma, perm)
    shuffle = np.random.default_rng(1).permutation(3000)
    after = m.own_versions(score[shuffle], t.w[shuffle], t.y[shuffle], gamma[shuffle], np.argsort(shuffle)[perm])
    for v in m.OWN:
        assert after[v] == pytest.approx(before[v], rel=1e-12, abs=1e-15), v
    other = np.random.default_rng([20261115, 4, 2]).permutation(3000)
    assert m.qini(score, t.w, t.y, other) != pytest.approx(before["V4"], rel=1e-6)


def test_v4_equals_v3_without_ties():
    t, score = sample(3000, 3, levels=None)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(3000)
    assert m.qini(score, t.w, t.y, perm) == pytest.approx(m.qini(score, t.w, t.y), rel=1e-13)


@pytest.mark.parametrize("levels", [4, None])
def test_causalml_qini_is_v4_rescaled_when_the_top_row_is_a_control(levels):
    """causalml 0.18.0 qini_score: Q(k) / |Q(n)| at k = 0..n, minus the line to Q(n) / |Q(n)|, averaged over n + 1 points.

    Before the first control row of the ranking causalml gets 0 / 0 and joins Q linearly, where V4 has
    Q(k) = Y_T(k). The two agree exactly when the first row is a control, as here, or when the leading
    treated rows and the first control row all have no responder.
    """
    t, score = sample(5000, 11, levels)
    w, y = t.w, t.y
    score[np.flatnonzero(w == 0)[0]] = score.max() + 1.0
    n = len(w)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(n)
    v1 = m.library_versions(score, w, y, perm)["V1"]
    assert v1 == pytest.approx(m.qini(score, w, y, perm) * n / ((n + 1) * abs(m.qini_end(w, y))), rel=1e-12)


def library_toc(order, w, y):
    """causalml 0.18.0 get_toc over a ranking: difference in means over the first k rows minus the overall one, 0 while one arm is absent."""
    n_t = np.cumsum(w[order])
    n_c = np.arange(1, len(w) + 1) - n_t
    y_t, y_c = np.cumsum((w * y)[order]), np.cumsum(((1 - w) * y)[order])
    both = (n_t > 0) & (n_c > 0)
    dim = m.ratio(y_t, n_t) - m.ratio(y_c, n_c)
    return np.where(both, dim - dim[-1], 0.0)


@pytest.mark.parametrize("levels", [4, None])
def test_v13_and_v12_weight_the_difference_in_means_toc_in_v4_order(levels):
    t, score = sample(4000, 5, levels)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(4000)
    lib = m.library_versions(score, t.w, t.y, perm)
    toc = library_toc(m.permuted_order(score, perm), t.w, t.y)
    n = len(toc)
    k = np.arange(1, n + 1)
    # V13: midpoints of TOC weighted by q, normalised; equal to twice V6's weighting of the same curve.
    assert lib["V13"] == pytest.approx(2 * np.mean(k / n * toc), rel=1e-10)
    # V12: weights 1/q at the midpoints, normalised; V11 weights every row alike.
    mid, q_mid = (np.r_[0.0, toc[:-1]] + toc) / 2, (k - 0.5) / n
    assert lib["V12"] == pytest.approx(np.sum(mid / q_mid) / np.sum(1 / q_mid), rel=1e-10)
    assert lib["V12"] != pytest.approx(np.mean(toc), rel=0.05)


def test_without_ties_causalml_gets_the_order_of_the_score():
    t, score = sample(4000, 9, levels=None)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(4000)
    lib = m.library_versions(score, t.w, t.y, perm)
    frame = pd.DataFrame({"y": t.y, "w": t.w, "score": score})
    assert lib["V1"] == float(qini_score(frame, normalize=True)["score"])
    assert lib["V7"] == float(auuc_score(frame, normalize=True)["score"])
    assert lib["V12"] == float(rate_score(frame, weighting="autoc")["score"])
    assert lib["V13"] == float(rate_score(frame, weighting="qini")["score"])


PROBE = """
import json
import numpy as np
from uplift_ranking import metrics as m
from uplift_ranking.synthetic import rct, tied
t = rct(20000, 2)
score = tied(t.tau, np.random.default_rng(1002), 4, noise=0.05)
values = m.library_versions(score, t.w, t.y, np.random.default_rng([20261115, 4, 1]).permutation(20000))
print(json.dumps({v: values[v].hex() for v in ("V1", "V7", "V12", "V13")}))
"""


def test_library_versions_are_the_same_at_every_cpu_dispatch_level():
    """numpy picks its sort kernel by CPU instruction set, and the kernels order equal keys differently."""
    out = set()
    for disabled in ("", "X86_V4", "X86_V3 X86_V4"):
        env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=disabled)
        out.add(subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(out) == 1


def test_v11_and_v12_can_order_two_rules_oppositely():
    # Rule a finds a small group with a large effect and is random elsewhere; rule b ranks a broad effect.
    rng = np.random.default_rng(3)
    n = 40000
    group, z = rng.random(n) < 0.02, rng.random(n)
    tau = np.where(group, 0.6, 0.3 * z)
    w = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + w * tau).astype(np.int64)
    a, b = np.where(group, 2 + rng.random(n), rng.random(n)), z
    gamma, perm = m.ipw(w, y), rng.permutation(n)
    assert m.library_versions(a, w, y, perm)["V12"] > m.library_versions(b, w, y, perm)["V12"]
    assert m.rate_autoc(a, gamma) < m.rate_autoc(b, gamma)


def test_library_versions_ignore_the_order_of_rows():
    t, score = sample(6000, 2, levels=4)
    perm = np.random.default_rng([20261115, 4, 1]).permutation(6000)
    before = m.library_versions(score, t.w, t.y, perm)
    treated_first = np.argsort(-t.w, kind="stable")  # as in long runs of the Criteo file
    w, y, s = t.w[treated_first], t.y[treated_first], score[treated_first]
    assert m.library_versions(s, w, y, np.argsort(treated_first)[perm]) == before


@pytest.mark.parametrize("where", ["one row", "every row", "infinite"])
def test_a_non_finite_score_has_no_value(where):
    t, score = sample(400, 5, levels=4)
    if where == "every row":
        score[:] = np.nan
    else:
        score[17] = np.nan if where == "one row" else np.inf
    gamma, perm = m.aipw(t.w, t.y, t.mu0, t.mu1), np.random.default_rng([20261115, 4, 1]).permutation(400)
    for version in (lambda: m.qini(score, t.w, t.y), lambda: m.own_versions(score, t.w, t.y, gamma, perm),
                    lambda: m.library_versions(score, t.w, t.y, perm)):
        with pytest.raises(ValueError, match="a score is NaN or infinite"):
            version()


def test_tie_shares():
    score = np.array([3.0, 2, 2, 2, 1, 0, 0])
    w = np.array([1, 1, 0, 0, 1, 1, 1])
    assert m.ties(score, w) == (5 / 7, 3 / 5)
    assert m.ties(np.arange(4.0), w[:4]) == (0.0, 0.0)


@pytest.mark.parametrize("levels", [2, 9, None])
def test_budget_value_follows_the_definition(levels):
    t, score = sample(300, 4, levels)
    gamma = m.aipw(t.w, t.y, t.mu0, t.mu1)
    for k in (0.1, 0.3, 0.5):
        cut, taken, total = k * 300, 0, 0.0
        for b in slow_blocks(score):
            part = min(len(b), cut - taken)
            if part <= 0:
                break
            total += part / len(b) * sum(gamma[i] for i in b)
            taken += part
        assert m.budget_value(score, gamma, k) == pytest.approx(total / 300, abs=1e-12)


RELEASE = """
import weakref
import numpy as np
from uplift_ranking import metrics as m
from uplift_ranking.synthetic import rct


class Held:
    pass


def first_call(t, perm):
    held = Held()
    alive = weakref.ref(held)
    m.library_versions(t.tau, t.w, t.y, perm)
    return alive


print(first_call(rct(2000, 0), np.random.default_rng(0).permutation(2000))() is None)
"""


def test_the_first_library_call_keeps_no_frame_of_its_caller():
    """A frame kept after the first call would hold a whole unit's scores and rankings through the next unit."""
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", RELEASE], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[-1] == "True"

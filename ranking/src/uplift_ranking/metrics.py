"""Metric versions of Section 5.2 on one set of rows.

Rows are ranked by score, largest first. Rows with equal scores form a tie block; a curve is
evaluated at the ends of blocks and joined linearly between them (Section 5.1). A ratio with a
zero denominator is 0. `w` is the treatment (0 or 1), `y` the binary outcome, `gamma` the AIPW
or IPW scores of the rows.
"""

import numpy as np
import pandas as pd

# Imported here and not at the first call: shap, which causalml imports, keeps a caught ImportError,
# and its traceback would hold the calling frames, with the first unit's arrays, for the rest of the run.
from causalml.metrics import auuc_score, qini_score, rate_score
from sklift.metrics import qini_auc_score, uplift_auc_score

OWN = ("V2*", "V3", "V4", "V5", "V6", "V9", "V10", "V11")
LIBRARY = ("V1", "V2", "V7", "V8", "V12", "V13")


def ratio(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b != 0)


def finite(score):
    """The score, or ValueError if a value is NaN or infinite: such scores have no ranking."""
    if not np.all(np.isfinite(score)):
        raise ValueError("a score is NaN or infinite")
    return score


def tie_blocks(score):
    """Rows ranked by score, largest first, and the position of the last row of each tie block."""
    order = np.argsort(-finite(score), kind="stable")
    ranked = score[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(score) - 1]
    return order, ends


def permuted_order(score, perm):
    """V4's ranking: by score, largest first, ties in the order the rows take in `perm`.

    Section 5.2 permutes the test rows; a tied row's position in that sequence orders it."""
    return perm[np.argsort(-finite(score)[perm], kind="stable")]


def area(x, f):
    """Integral over [0, 1] of f(phi) - phi f(1), f linear between (0, 0) and the points (x, f), x[-1] = 1."""
    x, f = np.r_[0.0, x], np.r_[0.0, f]
    return float(np.sum(np.diff(x) * (f[1:] + f[:-1])) / 2 - f[-1] / 2)


def counts(order, ends, w, y):
    """k, N_T(k), N_C(k), Y_T(k), Y_C(k) at the given ends of the ranking `order`."""
    k = ends + 1
    n_t = np.cumsum(w[order])[ends]
    y_t = np.cumsum((w * y)[order])[ends]
    y_c = np.cumsum(((1 - w) * y)[order])[ends]
    return k, n_t, k - n_t, y_t, y_c


def qini(score, w, y, perm=None):
    """V3: Radcliffe's Qini curve, ties grouped. With `perm`, V4: ties ordered by the permutation."""
    if perm is None:
        order, ends = tie_blocks(score)
    else:
        order, ends = permuted_order(score, perm), np.arange(len(score))
    k, n_t, n_c, y_t, y_c = counts(order, ends, w, y)
    n = len(score)
    return area(k / n, (y_t - y_c * ratio(n_t, n_c)) / n)


def qini_end(w, y):
    """q(1) = Q(n) / n."""
    n_t = w.sum()
    return float((np.sum(w * y) - np.sum((1 - w) * y) * ratio(n_t, len(w) - n_t)) / len(w))


def qini_share(score, w, y):
    """V5: Singh's Qini, G(phi) = Q(phi n) / Q(n)."""
    return qini(score, w, y) / qini_end(w, y)


def qini_scaled(score, w, y):
    """V2*: scikit-uplift's qini_auc_score, the area of V3 over that of the perfect curve from the labels."""
    return qini(score, w, y) / qini(y * w - y * (1 - w), w, y)


def uplift_joint(score, w, y):
    """V9: U(k) = (Y_T(k) / N_T(k) - Y_C(k) / N_C(k)) k, both arms in one ranking."""
    order, ends = tie_blocks(score)
    k, n_t, n_c, y_t, y_c = counts(order, ends, w, y)
    n = len(score)
    return area(k / n, (ratio(y_t, n_t) - ratio(y_c, n_c)) * k / n)


def uplift_separate(score, w, y):
    """V10: arms ranked separately; a tie block cut by p |arm| counts in proportion."""
    p = np.arange(1, 101) / 100
    v = np.zeros(100)
    for arm, sign in ((1, 1.0), (0, -1.0)):
        rows = np.flatnonzero(w == arm)
        order, ends = tie_blocks(score[rows])
        responders = np.interp(p * len(rows), np.r_[0, ends + 1], np.r_[0.0, np.cumsum(y[rows][order])[ends]])
        v += sign * ratio(responders, len(rows))
    return float(np.mean(v - p * v[-1]))


def aipw(w, y, mu0, mu1):
    """AIPW scores (Section 5.3), with e the treated share of the rows given."""
    e = w.mean()
    return mu1 - mu0 + w * (y - mu1) / e - (1 - w) * (y - mu0) / (1 - e)


def ipw(w, y):
    e = w.mean()
    return w * y / e - (1 - w) * y / (1 - e)


def toc(score, gamma):
    """TOC(k), k = 1..n: mean of gamma over the first k rows minus the mean of gamma, ties at block means."""
    order, ends = tie_blocks(score)
    size = np.diff(np.r_[0, ends + 1])
    block_mean = np.diff(np.r_[0.0, np.cumsum(gamma[order])[ends]]) / size
    return np.cumsum(np.repeat(block_mean, size)) / np.arange(1, len(score) + 1) - gamma.mean()


def rate_qini(score, gamma):
    """V6: (1/n) sum_k (k/n) TOC(k)."""
    t = toc(score, gamma)
    return float(np.mean(np.arange(1, len(t) + 1) / len(t) * t))


def rate_autoc(score, gamma):
    """V11: (1/n) sum_k TOC(k)."""
    return float(np.mean(toc(score, gamma)))


def budget_value(score, gamma, k):
    """H5: (1/n) sum of gamma over the top k n rows by score, a tie block at the cut in proportion."""
    order, ends = tie_blocks(score)
    cut = np.interp(k * len(score), np.r_[0, ends + 1], np.r_[0.0, np.cumsum(gamma[order])[ends]])
    return float(cut / len(score))


def own_versions(score, w, y, gamma, perm):
    return {
        "V2*": qini_scaled(score, w, y),
        "V3": qini(score, w, y),
        "V4": qini(score, w, y, perm),
        "V5": qini_share(score, w, y),
        "V6": rate_qini(score, gamma),
        "V9": uplift_joint(score, w, y),
        "V10": uplift_separate(score, w, y),
        "V11": rate_autoc(score, gamma),
    }


def library_versions(score, w, y, perm):
    """V1, V2, V7, V8, V12, V13 (Section 5.2).

    causalml uses the score only through pandas' default sort, which is not stable and orders tied
    rows differently on different CPU instruction sets. It gets a score column without ties instead:
    in V4's ranking the first row gets n and the last 1, so V1, V7, V12 and V13 order tied scores as
    V4 does on every machine. scikit-uplift groups ties and gets the score itself, with the rows in
    file order; the frame given to causalml has the columns y, w and score.
    """
    rank = np.empty(len(score))
    rank[permuted_order(score, perm)] = np.arange(len(score), 0, -1)
    frame = pd.DataFrame({"y": y, "w": w, "score": rank})
    return {
        "V1": float(qini_score(frame, normalize=True)["score"]),
        "V2": float(qini_auc_score(y, score, w, negative_effect=True)),
        "V7": float(auuc_score(frame, normalize=True)["score"]),
        "V8": float(uplift_auc_score(y, score, w)),
        "V12": float(rate_score(frame, weighting="autoc")["score"]),
        "V13": float(rate_score(frame, weighting="qini")["score"]),
    }


def ties(score, w):
    """Share of rows in tie blocks of more than one row, and the treated share within those blocks."""
    _, inverse, size = np.unique(score, return_inverse=True, return_counts=True)
    tied = size[inverse] > 1
    return float(tied.mean()), float(w[tied].mean()) if tied.any() else 0.0

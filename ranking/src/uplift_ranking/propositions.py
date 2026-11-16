"""Labels of H1 to H7 (Sections 7 and 8), as functions of summary statistics only.

A family is a set of paired differences with their half-sample replicates and seed values; the
rules are applied in the registered order.
"""

import itertools
import math

import numpy as np
from scipy.stats import kendalltau

from .resample import critical_value, identical, mdd, seed_rule

MODELS = ("S", "T", "X", "R", "DR", "causal forest", "TARNet", "DESCN", "EFIN")
PAIRS = list(itertools.combinations(MODELS, 2))
META, DEEP = set(MODELS[:5]), set(MODELS[6:])
H1A = ("V1", "V2", "V3", "V4", "V6", "V7", "V8", "V9", "V10", "V13")
H1A_OWN = ("V2*", "V3", "V4", "V6", "V9", "V10")
H1B = (("V6", "V11"), ("V11", "V12"), ("V6", "V12"))
PUBLISHED_GAIN = {(1, "TARNet"): 0.0010, (1, "EFIN"): 0.0167, (3, "DESCN"): 0.0021}  # Table 4 of Liu et al. 2024


def differences(values):
    """M(a) - M(b) for the 36 pairs, from a model -> value mapping."""
    return np.array([values[a] - values[b] for a, b in PAIRS])


def family(point, replicates, seeds, marginal=False):
    """Simultaneous (or marginal) 95% intervals over a family, its separated differences (-1, 0 or +1)
    and its identical ones (0 with SE 0, Section 5.5).

    `point` (P,), `replicates` (R, P), `seeds` (5, P): the five seed values of each difference. Marginal
    intervals (H2's marginal counts, Section 7) are d +- 1.96 SE, with the seed rule as well."""
    c, se = critical_value(point, replicates)
    c = 1.96 if marginal else c
    low, high = point - c * se, point + c * se
    rule = np.array([seed_rule(seeds[:, p]) for p in range(len(point))])
    direction = np.where(rule & (low > 0), 1, np.where(rule & (high < 0), -1, 0))
    return {"c": c, "se": se, "low": low, "high": high, "seed_rule": rule, "separated": direction,
            "identical": identical(point, se)}


def by_pair(found):
    """{"a b": ["u v", ...]} from (pair index, u, v) triples, model pairs in the order of PAIRS."""
    out = {}
    for p, u, v in sorted(found, key=lambda f: f[0]):
        out.setdefault(" ".join(PAIRS[p]), []).append(f"{u} {v}")
    return out


def reversals(points):
    """Model pairs whose differences are nonzero with opposite signs under two versions (H1, reported under every label)."""
    signs = {v: np.sign(d) for v, d in points.items()}
    return by_pair([(p, u, v) for u, v in itertools.combinations(points, 2) for p in range(len(PAIRS))
                    if signs[u][p] * signs[v][p] < 0])


def ordering(points, separated):
    """H1 label of a group of versions, and the pairs that rules 2 and 3 name.

    `points`: version -> 36 differences; `separated`: own version -> directions. Under rule 2 an entry
    "u v" is separated upwards under u and downwards under v; under rule 3, separated under u and
    reversed by v."""
    signs = {v: np.sign(d) for v, d in points.items()}
    if all(np.array_equal(s, next(iter(signs.values()))) for s in signs.values()):
        return "identical orderings", {}
    own = list(separated)
    contradictions = [(p, u, v) for u, v in itertools.permutations(own, 2) for p in range(len(PAIRS))
                      if separated[u][p] == 1 and separated[v][p] == -1]
    if contradictions:
        return "contradictory separations", by_pair(contradictions)
    reversed_separated = [(p, u, v) for u in own for v in points for p in range(len(PAIRS))
                          if v != u and separated[u][p] != 0 and signs[v][p] == -separated[u][p]]
    if reversed_separated:
        return "a separated pair reversed", by_pair(reversed_separated)
    if not any(np.any(s != 0) for s in separated.values()):
        return f"no separated pair; {len(reversals(points))} reversals of 36", {}
    return "agree on every separated pair", {}


def kendall(values):
    """Kendall's tau_b between every two versions; `values`: version -> model -> value."""
    out = {}
    for u, v in itertools.combinations(values, 2):
        out[(u, v)] = float(kendalltau([values[u][m] for m in MODELS], [values[v][m] for m in MODELS]).statistic)
    return out


def distinguishable(n):
    """H2 label from N, the number of the 36 pairs separated in the family of 72."""
    return "mostly indistinguishable" if n <= 9 else "mixed" if n <= 26 else "mostly distinguishable"


def separated_pairs(direction72):
    """Pairs separated under V6 or V11 in the family of 72 (V6's 36 differences first)."""
    return (direction72[:36] != 0) | (direction72[36:] != 0)


def counts(pairs_separated):
    meta = [i for i, (a, b) in enumerate(PAIRS) if a in META and b in META]
    mixed = [i for i, (a, b) in enumerate(PAIRS) if (a in DEEP) != (b in DEEP)]
    return {"N": int(pairs_separated.sum()), "meta-learners": int(pairs_separated[meta].sum()),
            "deep against non-deep": int(pairs_separated[mixed].sum())}


def gain(low, high, seeds_hold):
    """H3 label of a contrast D - R* from its simultaneous interval."""
    if low > 0 and seeds_hold:
        return "gain shown"
    if high < 0 and seeds_hold:
        return "loss shown"
    return "undecided"


def deep_result(labels):
    """The V2* label, with "metric-dependent" if V6 or V11 gives the opposite resolved label."""
    opposite = {"gain shown": "loss shown", "loss shown": "gain shown"}.get(labels["V2*"])
    if opposite and opposite in (labels["V6"], labels["V11"]):
        return labels["V2*"] + "; metric-dependent"
    return labels["V2*"]


def published_gain(g, delta, se, groups, n_published, c, seeds, gain_shown):
    """H3: our D - T against the published gain g; `seeds` are the five seed values of D - T."""
    se_published = se * math.sqrt(groups / n_published)
    z = (g - delta) / math.sqrt(se**2 + se_published**2)
    if z > c and np.all(np.asarray(seeds) < g):
        return "published gain contradicted", z
    if gain_shown and -c <= z <= c:
        return "published gain reproduced", z
    return "published gain not tested at this size", z


def gaps_below(s):
    """H4 label from s_L."""
    return "most published gaps below the detectable difference" if s > 0.5 else "half or fewer below"


def value(low, high):
    """H5 and H7 label of a marginal interval of a difference in value."""
    return "higher" if low > 0 else "lower" if high < 0 else "not distinguishable"


def marginal_interval(point, se):
    return point - 1.96 * se, point + 1.96 * se


def exact_and_dual(status, v_exact, v_dual):
    """H6 label of one instance."""
    if status != "Optimal":
        return "exact not certified"
    if abs(v_exact - v_dual) <= 1e-4 * max(v_exact, 1e-12):
        return "dual within 1e-4 of exact"
    return "dual below exact"


def detectable(se, c=None):
    """Median, minimum and maximum MDD over a family's differences (marginal, or (c* + 0.84) SE with c)."""
    m = mdd(np.asarray(se), c)
    return {"median": float(np.median(m)), "min": float(m.min()), "max": float(m.max())}

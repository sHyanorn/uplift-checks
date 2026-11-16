import numpy as np
import pytest

from uplift_ranking import literature
from uplift_ranking import propositions as P


def test_pairs_and_counts():
    assert len(P.PAIRS) == 36
    everything = P.counts(np.ones(36, bool))
    assert everything == {"N": 36, "meta-learners": 10, "deep against non-deep": 18}


def family_inputs(point, spread=0.01, seeds=None, reps=400, seed=0):
    rng = np.random.default_rng(seed)
    point = np.asarray(point, float)
    replicates = point + spread * rng.normal(size=(reps, len(point)))
    seeds = np.tile(point, (5, 1)) if seeds is None else np.asarray(seeds, float)
    return point, replicates, seeds


def test_family_separates_by_interval_and_seed_rule():
    point, reps, seeds = family_inputs([0.05, -0.05, 0.005, 0.05])
    seeds[2:, 3] = -0.01  # the seed rule fails for the last difference
    f = P.family(point, reps, seeds)
    assert list(f["separated"]) == [1, -1, 0, 0]
    assert f["c"] > 1.96 and list(P.family(point, reps, seeds, marginal=True)["separated"]) == [1, -1, 0, 0]
    big = P.family(*family_inputs(np.r_[point, np.zeros(68)]))
    assert big["c"] > f["c"]  # a family of 72 needs a larger critical value than one of 4


def test_a_difference_that_is_zero_in_every_replicate_is_identical():
    point, reps, seeds = family_inputs([0.05, 0.0, 0.01])
    reps[:, 1], seeds[:, 1] = 0.0, 0.0
    f = P.family(point, reps, seeds)
    assert list(f["identical"]) == [False, True, False] and f["separated"][1] == 0


def directions(*pairs):
    d = np.zeros(36, int)
    for p, s in pairs:
        d[p] = s
    return d


def test_ordering_rules_in_order():
    base = np.linspace(-1, 1, 36)
    same = {"V1": base, "V2": base * 2, "V3": base + 0}
    assert P.ordering(same, {"V3": directions((0, -1))})[0] == "identical orderings"
    flipped = dict(same, V1=np.where(np.arange(36) == 0, 1.0, base))
    # pair 0 separated in opposite directions under two own versions: rule 2 before rule 3
    label, detail = P.ordering(flipped, {"V3": directions((0, -1)), "V1": directions((0, 1))})
    assert label == "contradictory separations" and detail == {"S T": ["V1 V3"]}
    label, detail = P.ordering(flipped, {"V3": directions((0, -1))})
    assert label == "a separated pair reversed" and "V3 V1" in detail["S T"]
    assert P.ordering(flipped, {"V3": directions()}) == ("no separated pair; 1 reversals of 36", {})
    zero = dict(same, V1=np.where(np.arange(36) == 0, 0.0, base))
    assert P.ordering(zero, {"V3": directions((0, -1))})[0] == "agree on every separated pair"
    assert P.ordering(zero, {"V3": directions()}) == ("no separated pair; 0 reversals of 36", {})  # a zero is no reversal


def test_reversed_pairs_are_listed_whatever_the_label():
    base = np.linspace(-1, 1, 36)
    points = {"V1": np.where(np.isin(np.arange(36), [1, 2, 3]), -base, base), "V3": base}
    assert P.ordering(points, {"V3": directions((0, -1))})[0] == "agree on every separated pair"
    assert P.reversals(points) == {" ".join(P.PAIRS[p]): ["V1 V3"] for p in (1, 2, 3)}
    assert P.reversals({"V1": base, "V9": 2 * base, "V3": np.where(np.arange(36) == 5, 0.0, -base)}) == {
        " ".join(P.PAIRS[p]): ["V1 V3", "V9 V3"] for p in range(36) if p != 5}


def test_kendall_tau_b():
    up = {m: i for i, m in enumerate(P.MODELS)}
    tau = P.kendall({"V3": up, "V9": up, "V11": {m: -v for m, v in up.items()}})
    assert tau[("V3", "V9")] == pytest.approx(1.0) and tau[("V3", "V11")] == pytest.approx(-1.0)


@pytest.mark.parametrize("n, label", [(0, "mostly indistinguishable"), (9, "mostly indistinguishable"), (10, "mixed"),
                                      (26, "mixed"), (27, "mostly distinguishable"), (36, "mostly distinguishable")])
def test_distinguishable(n, label):
    assert P.distinguishable(n) == label


def test_pairs_separated_under_either_metric():
    d = np.zeros(72, int)
    d[[0, 36, 40]] = [1, -1, 1]
    assert list(np.flatnonzero(P.separated_pairs(d))) == [0, 4]


def test_gain_and_metric_dependence():
    assert P.gain(0.01, 0.02, True) == "gain shown" and P.gain(0.01, 0.02, False) == "undecided"
    assert P.gain(-0.02, -0.01, True) == "loss shown" and P.gain(-0.01, 0.01, True) == "undecided"
    assert P.deep_result({"V2*": "loss shown", "V6": "gain shown", "V11": "undecided"}) == "loss shown; metric-dependent"
    assert P.deep_result({"V2*": "undecided", "V6": "gain shown", "V11": "loss shown"}) == "undecided"
    assert P.deep_result({"V2*": "gain shown", "V6": "undecided", "V11": "gain shown"}) == "gain shown"


def test_published_gains_are_those_of_table_4():
    for unit, dataset in ((1, "Criteo"), (3, "Lazada")):
        table = {row: v for g, _, _, _, ds, row, _, v in literature.TABLE if g == "L1" and ds == dataset}
        for model in ("TARNet", "DESCN", "EFIN"):
            g = round(table[model] - table["T-Learner"], 4)
            assert P.PUBLISHED_GAIN.get((unit, model)) == (g if g > 0 else None)


def test_published_gain_z_test():
    # se 0.001 at G = n_L: z = (g - delta) / (0.001 sqrt 2)
    label, z = P.published_gain(0.0167, 0.0, 0.001, 4_000_000, 4_000_000, 2.5, [0.0] * 5, gain_shown=False)
    assert z == pytest.approx(0.0167 / (0.001 * np.sqrt(2))) and label == "published gain contradicted"
    assert P.published_gain(0.0167, 0.0, 0.001, 4_000_000, 4_000_000, 2.5, [0.0, 0.0, 0.02, 0.0, 0.0], False)[0] == "published gain not tested at this size"
    assert P.published_gain(0.0167, 0.015, 0.001, 4_000_000, 4_000_000, 2.5, [0.015] * 5, True)[0] == "published gain reproduced"
    assert P.published_gain(0.0167, 0.015, 0.001, 4_000_000, 4_000_000, 2.5, [0.015] * 5, False)[0] == "published gain not tested at this size"
    assert P.published_gain(0.0167, 0.0, 0.001, 4_000_000, 400_000, 2.5, [0.0] * 5, False)[1] == pytest.approx(0.0167 / (0.001 * np.sqrt(11)))


def test_other_labels():
    assert P.gaps_below(0.51) == "most published gaps below the detectable difference" and P.gaps_below(0.5) == "half or fewer below"
    assert (P.value(0.1, 0.2), P.value(-0.2, -0.1), P.value(-0.1, 0.1)) == ("higher", "lower", "not distinguishable")
    assert P.exact_and_dual("Time limit reached", 1.0, 1.0) == "exact not certified"
    assert P.exact_and_dual("Optimal", 1.0, 1.0 - 0.9e-4) == "dual within 1e-4 of exact"
    assert P.exact_and_dual("Optimal", 1.0, 1.0 - 1.1e-4) == "dual below exact"
    assert P.exact_and_dual("Optimal", 0.0, 0.0) == "dual within 1e-4 of exact"
    assert P.marginal_interval(1.0, 0.5) == pytest.approx((0.02, 1.98))
    assert P.detectable([0.001, 0.002, 0.003]) == pytest.approx({"median": 0.0056, "min": 0.0028, "max": 0.0084})
    assert P.detectable([0.001], c=2.16)["median"] == pytest.approx(0.003)

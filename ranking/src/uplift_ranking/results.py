"""Every registered output of a unit (H1 to H5, and H6, H7 on Hillstrom) from predictions on a set of rows.

The same functions serve the dry run on validation rows and the test read. Outputs are aggregate
statistics and labels only.
"""

import numpy as np

from . import allocation, literature, metrics, resample
from . import propositions as P

OWN = ("V2*", "V3", "V4", "V5", "V6", "V9", "V10", "V11")
LIBRARY = ("V1", "V2", "V7", "V8", "V12", "V13")
NON_DEEP = P.MODELS[:6]
LITERATURE = {1: ("L1 Criteo", "L2", "L4"), 3: ("L1 Lazada", "L3")}
N_PUBLISHED = {1: literature.GROUPS["L1 Criteo"][3], 3: literature.GROUPS["L1 Lazada"][3]}  # test size of Table 4


def column(statistic):
    return resample.STATISTICS.index(statistic)


class Differences:
    """Point, replicate and seed values of every statistic for each model, and their paired differences."""

    def __init__(self, point, replicates, seeds):
        self.point, self.replicates, self.seeds = point, replicates, seeds  # (M, S), (R, M, S), (5, M, S)

    def of(self, pairs, statistic):
        i, j = [P.MODELS.index(a) for a, _ in pairs], [P.MODELS.index(b) for _, b in pairs]
        s = column(statistic)
        return (self.point[i, s] - self.point[j, s], self.replicates[:, i, s] - self.replicates[:, j, s],
                self.seeds[:, i, s] - self.seeds[:, j, s])


def rounded(x):
    return [float(v) for v in np.asarray(x).ravel()]


def by_difference(seeds):
    """The five seed values (seeds 0 to 4) of each difference, from a (5, P) array: what the seed rule reads."""
    return [rounded(s) for s in np.asarray(seeds).T]


def unit_outputs(unit, scores, w, y, mu0, mu1, groups, r_star, labels=True, replicates=resample.REPLICATES, processes=4):
    """`scores`: model -> five arrays (seeds 0 to 4) of predicted effects on the rows."""
    perm = np.random.default_rng([20261115, 4, unit]).permutation(len(w))
    ranked = lambda s: (resample.Ranked(scores[m][s], w, y, mu0, mu1, perm) for m in P.MODELS)
    seed0 = list(ranked(0))
    point = resample.point_values(seed0)
    resampled = resample.replicate_values(seed0, groups, unit, replicates, processes)
    del seed0  # the workers needed all nine rankings; seeds 1 to 4 take one model's at a time
    diff = Differences(point, resampled, np.array([point] + [resample.point_values(ranked(s)) for s in range(1, 5)]))
    library = {m: metrics.library_versions(scores[m][0], w, y, perm) for m in P.MODELS}
    values = {v: {m: float(point[k, column(v)]) for k, m in enumerate(P.MODELS)} for v in OWN}
    values |= {v: {m: library[m][v] for m in P.MODELS} for v in LIBRARY}
    out = {"unit": unit, "rows": len(w), "groups": int(groups.max()) + 1, "R*": r_star, "values": values,
           # H1 (Section 7): Kendall's tau_b leaves out V5 and V2*, whose values are V2's.
           "kendall": {f"{a} {b}": t for (a, b), t in P.kendall({v: values[v] for v in values if v not in ("V5", "V2*")}).items()},
           "ties": {m: metrics.ties(scores[m][0], w) for m in P.MODELS},
           "distinct predictions over seeds": {m: len({s.tobytes() for s in scores[m]}) == 5 for m in P.MODELS}}

    pairs = {v: diff.of(P.PAIRS, v) for v in OWN}
    families = {v: P.family(*pairs[v]) for v in OWN}
    out["H1 families"] = {v: {"c": f["c"], "se": rounded(f["se"]), "low": rounded(f["low"]), "high": rounded(f["high"]),
                              "seeds": by_difference(pairs[v][2]), "separated": [int(d) for d in f["separated"]],
                              "identical": [int(i) for i in f["identical"]]} for v, f in families.items()}
    points = {v: np.array([values[v][a] - values[v][b] for a, b in P.PAIRS]) for v in values}
    groups_h1 = {"H1a": (P.H1A, P.H1A_OWN)} | {f"H1b {a} {b}": ((a, b), [v for v in (a, b) if v in families]) for a, b in P.H1B}
    for name, (versions, own) in groups_h1.items():
        group = {v: points[v] for v in versions}
        out[name] = {"reversed pairs": P.reversals(group)}
        if labels:
            out[name]["label"], out[name]["named"] = P.ordering(group, {v: families[v]["separated"] for v in own})

    point72, reps72, seeds72 = (np.concatenate(z, axis=-1) for z in zip(diff.of(P.PAIRS, "V6"), diff.of(P.PAIRS, "V11")))
    f72 = P.family(point72, reps72, seeds72)
    marginal = P.family(point72, reps72, seeds72, marginal=True)
    both = P.separated_pairs(f72["separated"])
    out["H2"] = {"c": f72["c"], "counts": P.counts(both), "V6 alone": int((f72["separated"][:36] != 0).sum()),
                 "V11 alone": int((f72["separated"][36:] != 0).sum()),
                 "marginal": P.counts(P.separated_pairs(marginal["separated"])),
                 "low": rounded(f72["low"]), "high": rounded(f72["high"]), "seeds": by_difference(seeds72),
                 "identical": [int(i) for i in f72["identical"]],
                 "median MDD_sim": {"V6": float(np.median(resample.mdd(f72["se"][:36], f72["c"]))),
                                    "V11": float(np.median(resample.mdd(f72["se"][36:], f72["c"])))},
                 "median MDD": {"V6": float(np.median(resample.mdd(f72["se"][:36]))),
                                "V11": float(np.median(resample.mdd(f72["se"][36:])))}}
    if labels:
        out["H2"]["label"] = P.distinguishable(out["H2"]["counts"]["N"])

    deep = P.MODELS[6:]
    contrasts = [((d, r_star), v) for v in ("V2*", "V6", "V11") for d in deep] + [((d, "T"), "V2*") for d in deep]
    parts = [diff.of([pair], v) for pair, v in contrasts]
    f12 = P.family(*(np.concatenate(z, axis=-1) for z in zip(*parts)))
    out["H3"] = {"c": f12["c"], "contrasts": []}
    for k, ((d, base), v) in enumerate(contrasts):
        item = {"model": d, "against": base, "version": v, "value": float(parts[k][0][0]), "low": float(f12["low"][k]),
                "high": float(f12["high"][k]), "seeds": rounded(parts[k][2][:, 0]),
                "MDD_sim": float(resample.mdd(f12["se"][k], f12["c"])), "MDD": float(resample.mdd(f12["se"][k])),
                "identical": bool(f12["identical"][k])}
        if labels:
            item["label"] = P.gain(f12["low"][k], f12["high"][k], f12["seed_rule"][k])
        out["H3"]["contrasts"].append(item)
    if labels:
        by = {(c["model"], c["against"], c["version"]): c for c in out["H3"]["contrasts"]}
        out["H3"]["results"] = {d: P.deep_result({v: by[(d, r_star, v)]["label"] for v in ("V2*", "V6", "V11")}) for d in deep}
        for d in deep:
            g = P.PUBLISHED_GAIN.get((unit, d))
            if g is not None:
                k = contrasts.index(((d, "T"), "V2*"))
                label, z = P.published_gain(g, parts[k][0][0], f12["se"][k], out["groups"], N_PUBLISHED[unit], f12["c"],
                                            parts[k][2][:, 0], by[(d, "T", "V2*")]["label"] == "gain shown")
                out["H3"]["results"][f"{d} published gain"] = {"g": g, "z": z, "label": label}

    se = {v: families[v]["se"] for v in OWN}
    out["H4"] = {"MDD": {v: P.detectable(se[v]) for v in OWN}}
    pair_mdd = lambda v, n: {frozenset(p): resample.at_size(m, out["groups"], n) for p, m in zip(P.PAIRS, resample.mdd(se[v]))}
    for group in LITERATURE.get(unit, ()):
        _, version, _, n_l = literature.GROUPS[group]
        mdd = resample.mdd(se[version])
        mdd_l = literature.mdd_at_size(mdd, out["groups"], n_l)
        item = {"MDD_L": mdd_l, "s_L": literature.share_below(group, mdd_l),
                "s_L min": literature.share_below(group, resample.at_size(mdd.min(), out["groups"], n_l)),
                "s_L max": literature.share_below(group, resample.at_size(mdd.max(), out["groups"], n_l)),
                "s_L matched": literature.share_below_matched(group, pair_mdd(version, n_l))}
        if group == "L1 Criteo":
            item["reading B"] = {"MDD_L": mdd_l * literature.READING_B,
                                 "s_L": literature.share_below(group, mdd_l * literature.READING_B)}
        if group == "L3":
            released = literature.mdd_at_size(mdd, out["groups"], 181_669)
            item["released size"] = {"MDD_L": released, "s_L": literature.share_below(group, released)}
        if labels:
            item["label"] = P.gaps_below(item["s_L"])
        out["H4"][group] = item

    out["H5"] = []
    widths = []
    for k in resample.BUDGETS:
        for d in P.MODELS:
            if d == r_star:
                continue
            item = {"model": d, "k": k}
            for scores_kind in ("AIPW", "IPW"):
                p, reps, _ = diff.of([(d, r_star)], f"{scores_kind} {k}")
                low, high = P.marginal_interval(p[0], resample.standard_error(reps[:, 0]))
                item[scores_kind] = {"value": float(p[0]), "low": float(low), "high": float(high)}
                if labels:
                    item[scores_kind]["label"] = P.value(low, high)
            ipw_width = item["IPW"]["high"] - item["IPW"]["low"]
            if ipw_width > 0:  # 0 only for a model identical to R* on these rows
                widths.append((item["AIPW"]["high"] - item["AIPW"]["low"]) / ipw_width)
            out["H5"].append(item)
    out["H5 width ratio"] = float(np.median(widths)) if widths else None
    return out


def allocation_outputs(v, arm, y, mu, score, groups, labels=True, replicates=resample.REPLICATES, time_limit=1800.0):
    """H6 and H7 on Hillstrom: `v` (n, 2) predicted gains of arms 1 and 2, `mu` (n, 3), `score` R*'s U4 score."""
    out = []
    for instance, (c, k) in enumerate(allocation.INSTANCES):
        budget = k * len(v)
        exact, dual = allocation.exact(v, c, budget, time_limit), allocation.dual(v, c, budget)
        item = {"instance": instance, "c": c, "B": budget, "status": exact["status"], "V_E": exact["value"],
                "V_D": dual["value"], "gap": exact["gap"], "dual bound": exact["bound"], "seconds exact": exact["seconds"],
                "seconds dual": dual["seconds"], "bound ratio": float(np.c_[np.zeros(len(v)), v].max() / exact["value"])
                if exact["value"] > 0 else None}
        if labels:
            item["label"] = P.exact_and_dual(exact["status"], exact["value"], dual["value"])
        policies = [exact["assignment"], *allocation.baselines(v, c, budget, score, instance)]
        point = allocation.policy_values(policies, arm, y, mu)
        reps = allocation.replicate_policy_values(policies, arm, y, mu, groups, replicates=replicates)
        for b, name in ((1, "b1"), (2, "b2")):
            item[name] = {}
            for e, estimator in enumerate(("IPW", "AIPW")):
                d = point[0, e] - point[b, e]
                low, high = P.marginal_interval(d, resample.standard_error(reps[:, 0, e] - reps[:, b, e]))
                item[name][estimator] = {"value": float(d), "low": float(low), "high": float(high)}
                if labels:
                    item[name][estimator]["label"] = P.value(low, high)
        out.append(item)
    return out

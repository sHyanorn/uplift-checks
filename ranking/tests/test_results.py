import numpy as np

from uplift_ranking import propositions as P
from uplift_ranking import results
from uplift_ranking.resample import seed_rule
from uplift_ranking.roles import groups
from uplift_ranking.synthetic import rct


def test_identical_differences_and_reversed_pairs_are_reported():
    t = rct(3000, 4)
    rng = np.random.default_rng(0)
    scores = {m: [t.tau + 0.05 * rng.normal(size=3000) for _ in range(5)] for m in P.MODELS}
    scores["T"] = scores["S"]
    out = results.unit_outputs(1, scores, t.w, t.y, t.mu0, t.mu1, groups(t.x), "S", replicates=30, processes=1)
    # Each separation, gain and published-gain label follows from the interval and five seed values in the output.
    def separated(f):
        rule = [seed_rule(np.array(s)) for s in f["seeds"]]
        return np.array([int(r and lo > 0) - int(r and hi < 0) for r, lo, hi in zip(rule, f["low"], f["high"])])

    for v in results.OWN:
        assert separated(out["H1 families"][v]).tolist() == out["H1 families"][v]["separated"]
    assert out["H2"]["counts"] == P.counts(P.separated_pairs(separated(out["H2"])))
    by = {(c["model"], c["against"], c["version"]): c for c in out["H3"]["contrasts"]}
    for c in by.values():
        assert c["seeds"][0] == c["value"] and c["label"] == P.gain(c["low"], c["high"], seed_rule(np.array(c["seeds"])))
    for d in ("TARNet", "EFIN"):
        r, c, crit = out["H3"]["results"][f"{d} published gain"], by[(d, "T", "V2*")], out["H3"]["c"]
        assert (r["label"] == "published gain contradicted") == (r["z"] > crit and max(c["seeds"]) < r["g"])
    st = P.PAIRS.index(("S", "T"))
    for v in results.OWN:
        assert np.flatnonzero(out["H1 families"][v]["identical"]).tolist() == [st]
    assert np.flatnonzero(out["H2"]["identical"]).tolist() == [st, 36 + st]
    assert not any(c["identical"] for c in out["H3"]["contrasts"])
    points = {v: np.array([out["values"][v][a] - out["values"][v][b] for a, b in P.PAIRS]) for v in out["values"]}
    assert out["H1a"]["reversed pairs"] == P.reversals({v: points[v] for v in P.H1A}) != {}
    for a, b in P.H1B:
        assert out[f"H1b {a} {b}"]["reversed pairs"] == P.reversals({a: points[a], b: points[b]})

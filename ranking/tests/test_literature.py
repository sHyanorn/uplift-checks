import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pytest

from uplift_ranking import literature as lit

REGISTRATION = (Path(__file__).resolve().parents[1] / "PREREGISTRATION.md").read_text()


def test_table_is_appendix_a():
    a = REGISTRATION[REGISTRATION.index("## Appendix A") : REGISTRATION.index("## Appendix B")]
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in a.splitlines() if re.match(r"\| L\d ", line)]
    assert len(rows) == len(lit.TABLE) == 50
    for row, (g, paper, page, table, dataset, name, column, value) in zip(rows, lit.TABLE):
        assert row[:7] == [g, paper, str(page), table, dataset, name, column]
        assert Decimal(row[7]) == Decimal(repr(value))


def test_gap_counts_and_medians_match_appendix_b():
    b = REGISTRATION[REGISTRATION.index("Median absolute published gap") :].split("\n")[0]
    registered = dict(zip(("L1 Criteo", "L1 Lazada", "L2", "L3", "L4"), re.findall(r"(0\.\d{4}) \(", b)))
    assert {g: len(lit.gaps(g)) for g in lit.GROUPS} == {"L1 Criteo": 78, "L1 Lazada": 78, "L2": 11, "L3": 7, "L4": 3}
    for group, median in registered.items():
        exact = np.median([Decimal(repr(abs(g))) for _, _, g in lit.gaps(group)])
        assert Decimal(exact).quantize(Decimal("0.0001"), ROUND_HALF_UP) == Decimal(median), group


def test_gaps_are_the_registered_differences():
    assert ("EFIN", "S-Learner", round(0.1285 - 0.0857, 4)) in lit.gaps("L2")
    assert all(a == "DESCN" for a, _, _ in lit.gaps("L3"))
    assert {(a, b) for a, b, _ in lit.gaps("L4")} == {("S-Learner", "X-Learner"), ("S-Learner", "T-Learner"), ("X-Learner", "T-Learner")}
    assert lit.gaps("L1 Criteo")[0] == ("S-Learner", "T-Learner", -0.0102)


def test_reading_b_coefficient_follows_from_the_overlap():
    k, overlap = 10, 0.1
    variance = (1 + (k - 1) * overlap) / k
    assert variance == pytest.approx(0.19)
    assert round(np.sqrt(variance), 2) == lit.READING_B


def test_share_below_and_rescaling():
    mdd = lit.mdd_at_size([0.002, 0.004, 0.006], groups=4_000_000, n=1_000_000)
    assert mdd == pytest.approx(0.008)
    gaps = sorted(abs(g) for _, _, g in lit.gaps("L1 Criteo"))
    assert lit.share_below("L1 Criteo", gaps[39]) == 39 / 78
    assert lit.share_below("L4", 1.0) == 1.0 and lit.share_below("L4", 0.0) == 0.0


def test_matched_pairs():
    assert {frozenset((a, b)) for a, b, _ in lit.matched("L4")} == {frozenset(p) for p in [("S", "X"), ("S", "T"), ("X", "T")]}
    l1 = {frozenset((a, b)) for a, b, _ in lit.matched("L1 Criteo")}
    assert l1 == {frozenset((a, b)) for a in ("S", "T") for b in ("TARNet", "DESCN", "EFIN")} | {frozenset(("S", "T"))}
    assert {frozenset((a, b)) for a, b, _ in lit.matched("L2")} == {frozenset(("EFIN", "S")), frozenset(("EFIN", "T"))}
    assert lit.matched("L3") == []
    pair_mdd = {p: 0.01 for p in l1}
    expected = np.mean([abs(g) < 0.01 for _, _, g in lit.matched("L1 Criteo")])
    assert lit.share_below_matched("L1 Criteo", pair_mdd) == expected


def test_a_group_without_a_matched_pair_writes_null():
    """L3 has no matched pair; its share is null, so the results stay valid JSON (amendment 10)."""
    assert lit.share_below_matched("L3", {}) is None
    assert json.dumps({"s_L matched": lit.share_below_matched("L3", {})}, allow_nan=False) == '{"s_L matched": null}'

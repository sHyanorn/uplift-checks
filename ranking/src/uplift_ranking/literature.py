"""Published figures (Appendix A), the gaps of each literature group and their share below the MDD (H4)."""

import itertools
import math

import numpy as np

# Appendix A: group, paper, PDF page, table, dataset, row, column, value.
TABLE = (
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'S-Learner', 'Test QINI', 0.0951),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'T-Learner', 'Test QINI', 0.1053),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'BNN', 'Test QINI', 0.1017),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'TARNet', 'Test QINI', 0.1063),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'CFRNet', 'Test QINI', 0.1085),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'CEVAE', 'Test QINI', 0.0881),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'GANITE', 'Test QINI', 0.1192),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'DragonNet', 'Test QINI', 0.1099),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'FlexTENet', 'Test QINI', 0.1164),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'SNet', 'Test QINI', 0.1044),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'EUEN', 'Test QINI', 0.1143),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'DESCN', 'Test QINI', 0.0795),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Criteo', 'EFIN', 'Test QINI', 0.1220),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'S-Learner', 'Test QINI', 0.0178),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'T-Learner', 'Test QINI', 0.0231),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'BNN', 'Test QINI', 0.0199),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'TARNet', 'Test QINI', 0.0177),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'CFRNet', 'Test QINI', 0.0234),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'CEVAE', 'Test QINI', 0.0078),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'GANITE', 'Test QINI', 0.0187),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'DragonNet', 'Test QINI', 0.0177),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'FlexTENet', 'Test QINI', 0.0215),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'SNet', 'Test QINI', 0.0190),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'EUEN', 'Test QINI', 0.0205),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'DESCN', 'Test QINI', 0.0252),
    ('L1', 'Liu et al. 2024 2406.00335v1', 7, 'Table 4', 'Lazada', 'EFIN', 'Test QINI', 0.0169),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'S-Learner', 'CRITEO-UPLIFT QINI', 0.0857),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'T-Learner', 'CRITEO-UPLIFT QINI', 0.1083),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'TarNet', 'CRITEO-UPLIFT QINI', 0.1027),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'CFRNet', 'CRITEO-UPLIFT QINI', 0.1052),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'DragonNet', 'CRITEO-UPLIFT QINI', 0.1096),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'GANITE', 'CRITEO-UPLIFT QINI', 0.1170),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'CEVAE', 'CRITEO-UPLIFT QINI', 0.0951),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'FlexTENet', 'CRITEO-UPLIFT QINI', 0.1108),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'SNet', 'CRITEO-UPLIFT QINI', 0.1112),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'EUEN', 'CRITEO-UPLIFT QINI', 0.1153),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'DESCN', 'CRITEO-UPLIFT QINI', 0.1129),
    ('L2', 'Liu et al. 2023 (EFIN) 2306.00315v1', 7, 'Table 3', 'Criteo', 'EFIN', 'CRITEO-UPLIFT QINI', 0.1285),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'X-learner (NN)', 'Production Dataset AUUC (mean)', 0.0234),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'Causal Forest', 'Production Dataset AUUC (mean)', 0.0132),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'BART', 'Production Dataset AUUC (mean)', 0.0222),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'TARNet', 'Production Dataset AUUC (mean)', 0.0309),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'CFR_wass', 'Production Dataset AUUC (mean)', 0.0261),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'CFR_mmd', 'Production Dataset AUUC (mean)', 0.0324),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'X-network', 'Production Dataset AUUC (mean)', 0.0324),
    ('L3', 'Zhong et al. 2022 (DESCN) 2207.09920v3', 6, 'Table 2', 'Lazada (production sample, 0.91M test rows)', 'DESCN', 'Production Dataset AUUC (mean)', 0.0340),
    ('L4', 'Singh 2026 2604.06123v1', 3, 'Table 2', 'Criteo', 'S-Learner', 'Qini', 0.3759),
    ('L4', 'Singh 2026 2604.06123v1', 3, 'Table 2', 'Criteo', 'X-Learner', 'Qini', 0.3615),
    ('L4', 'Singh 2026 2604.06123v1', 3, 'Table 2', 'Criteo', 'T-Learner', 'Qini', 0.3503),
    ('L4', 'Singh 2026 2604.06123v1', 3, 'Table 2', 'Criteo', 'Causal Forest', 'Qini', 0.2524),
)
# Group: (Appendix A group, dataset), version, unit, published test size n_L.
GROUPS = {
    "L1 Criteo": (("L1", "Criteo"), "V2*", 1, 1_397_959),
    "L1 Lazada": (("L1", "Lazada"), "V2*", 3, 181_669),
    "L2": (("L2", None), "V2*", 1, 2_795_918),
    "L3": (("L3", None), "V2*", 3, 910_000),
    "L4": (("L4", None), "V5", 1, 2_795_918),
}
OURS = {"S-Learner": "S", "T-Learner": "T", "X-Learner": "X", "TARNet": "TARNet", "TarNet": "TARNet", "DESCN": "DESCN", "EFIN": "EFIN"}
# Reading B (L1 Criteo): the mean over ten test sets of 10% overlapping by about 10% pairwise has variance
# (1 + 9 x 0.1) / 10 = 0.19 times that of one, so the MDD scales by its square root, registered as 0.44.
READING_B = 0.44


def values(group):
    (name, dataset), *_ = GROUPS[group]
    return {row: value for g, _, _, _, ds, row, _, value in TABLE if g == name and dataset in (None, ds)}


def gaps(group):
    """(row a, row b, value a - value b) for the group's pairs, rounded to the table's four decimals."""
    v = values(group)
    if group.startswith("L1"):
        pairs = itertools.combinations(v, 2)
    elif group == "L2":
        pairs = (("EFIN", r) for r in v if r != "EFIN")
    elif group == "L3":
        pairs = (("DESCN", r) for r in v if r != "DESCN")
    else:
        pairs = itertools.combinations([r for r in v if r != "Causal Forest"], 2)
    return [(a, b, round(v[a] - v[b], 4)) for a, b in pairs]


def share_below(group, mdd):
    """s_L: share of the group's gaps whose absolute value is below `mdd`."""
    return float(np.mean([abs(g) < mdd for _, _, g in gaps(group)]))


def mdd_at_size(pair_mdds, groups, n):
    """MDD_L: the median over our pairs' MDDs, rescaled from G resampling groups to the test size n."""
    return float(np.median(pair_mdds)) * math.sqrt(groups / n)


def matched(group):
    """Gaps between two of our models: L4 S, X, T; L1 and L2 S or T against TARNet, DESCN or EFIN, and S against T."""
    if group == "L4":
        allowed = {frozenset(p) for p in itertools.combinations(("S", "X", "T"), 2)}
    elif group in ("L1 Criteo", "L1 Lazada", "L2"):
        allowed = {frozenset((a, b)) for a in ("S", "T") for b in ("TARNet", "DESCN", "EFIN")} | {frozenset(("S", "T"))}
    else:
        return []
    return [(OURS[a], OURS[b], g) for a, b, g in gaps(group)
            if a in OURS and b in OURS and frozenset((OURS[a], OURS[b])) in allowed]


def share_below_matched(group, pair_mdd):
    """Share of matched gaps below the MDD of the same pair of our models; `pair_mdd` maps frozenset pairs to MDDs.
    None (JSON null) for a group without a matched pair, L3 (amendment 10)."""
    rows = matched(group)
    return float(np.mean([abs(g) < pair_mdd[frozenset((a, b))] for a, b, g in rows])) if rows else None

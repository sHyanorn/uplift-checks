import itertools

import numpy as np
import pytest

from uplift_ranking import allocation as al


def brute_force(v, c, budget):
    feasible = [a for a in itertools.product(range(3), repeat=len(v)) if al.costs(c)[list(a)].sum() <= budget]
    return max(al.value(v, np.array(a)) for a in feasible)


@pytest.mark.parametrize("seed", range(3))
def test_exact_solution_is_the_best_of_all_3_to_the_8_assignments(seed):
    v = np.random.default_rng(seed).normal(0.02, 0.05, size=(8, 2))
    for c, k in al.INSTANCES:
        result = al.exact(v, c, k * 8)
        assert result["status"] == "Optimal"
        assert result["value"] == pytest.approx(brute_force(v, c, k * 8), abs=1e-12)
        assert al.costs(c)[result["assignment"]].sum() <= k * 8 + 1e-9


def test_dual_is_feasible_and_never_above_exact():
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = rng.normal(0.02, 0.05, size=(300, 2))
        for c, k in al.INSTANCES:
            budget = k * 300
            d, e = al.dual(v, c, budget), al.exact(v, c, budget)
            assert al.costs(c)[d["assignment"]].sum() <= budget
            assert d["value"] <= e["value"] + 1e-12


def test_dual_falls_below_exact_on_an_integer_gap():
    # Budget 2: user 0 gains 3 at cost 2 (arm 2), user 1 gains 1.6 at cost 1 (arm 1). Exact sends user 0;
    # at every lambda the dual either exceeds the budget or keeps only user 1.
    v = np.array([[0.0, 3.0], [1.6, 0.0]])
    e, d = al.exact(v, 2.0, 2.0), al.dual(v, 2.0, 2.0)
    assert e["value"] == 3.0 and list(e["assignment"]) == [2, 0]
    assert d["value"] == 1.6 and list(d["assignment"]) == [0, 1]


def test_dual_breaks_ties_to_the_lower_arm():
    v = np.array([[0.2, 0.2], [0.0, 0.0]])
    assert list(al.dual(v, 1.0, 10.0)["assignment"]) == [1, 0]


def test_baselines():
    rng = np.random.default_rng(2)
    v = rng.normal(0.02, 0.05, size=(1000, 2)) + [0.0, 0.01]
    score = np.round(rng.normal(size=1000), 1)
    assert al.email_arm(v) == 2 and al.email_arm(np.zeros((3, 2))) == 1
    for instance, (c, k) in enumerate(al.INSTANCES):
        budget = k * 1000
        b1, b2 = al.baselines(v, c, budget, score, instance)
        size = int(np.floor(budget / c))
        assert set(np.unique(b1)) <= {0, 2} and (b1 == 2).sum() == size and al.costs(c)[b1].sum() <= budget
        drawn = np.random.default_rng([20261115, 5, instance]).choice(1000, size, replace=False)
        assert np.array_equal(np.flatnonzero(b1), np.sort(drawn))
        assert np.array_equal(np.flatnonzero(b2), np.sort(np.argsort(-score, kind="stable")[:size]))
    b1, _ = al.baselines(v[:10], 0.5, 100.0, score[:10], 0)  # a budget above n users: everyone is sent
    assert (b1 == al.email_arm(v[:10])).all()


def test_policy_estimators_are_unbiased_over_500_trials():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    mu = np.clip(np.c_[0.2 + 0.05 * x, 0.25 + 0.1 * x, 0.22 - 0.05 * x], 0.01, 0.99)
    policy = np.where(x > 0.3, 1, np.where(x < -0.5, 2, 0))
    truth = mu[np.arange(n), policy].mean()
    biased = np.clip(mu + 0.05, 0, 1)  # AIPW stays unbiased with wrong outcome models under randomisation
    estimates = []
    for _ in range(500):
        arm = rng.choice(3, n, p=[0.4, 0.3, 0.3])
        y = (rng.random(n) < mu[np.arange(n), arm]).astype(float)
        estimates.append(al.policy_values([policy], arm, y, biased)[0])
    estimates = np.array(estimates)
    se = estimates.std(axis=0, ddof=1) / np.sqrt(500)
    assert np.all(np.abs(estimates.mean(axis=0) - truth) < 3 * se)
    assert se[1] < se[0]


def test_replicates_recompute_the_arm_shares():
    rng = np.random.default_rng(4)
    arm, y = rng.choice(3, 400), rng.integers(0, 2, 400).astype(float)
    mu, policy, group = np.full((400, 3), 0.3), rng.choice(3, 400), np.arange(400)
    values = al.replicate_policy_values([policy], arm, y, mu, group, replicates=5)
    for r in range(5):
        rows = al.replicate_rows(group, 4, r)
        shares = np.bincount(arm[rows], minlength=3) / rows.sum()
        taken = (arm[rows] == policy[rows]) / shares[policy[rows]]
        assert values[r, 0, 0] == pytest.approx(np.mean(taken * y[rows]), rel=1e-12)

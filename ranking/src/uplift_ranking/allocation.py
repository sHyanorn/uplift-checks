"""Three-arm choice under a budget on Hillstrom (H6) and the estimated value of policies (H7).

Arms: 0 no e-mail, 1 men's e-mail, 2 women's e-mail. `v` is the (n, 2) matrix of v_i1 and v_i2,
the predicted gains over arm 0; arm 0 gains 0 and costs 0.
"""

import time

import highspy
import numpy as np

from .resample import replicate_rows

COSTS = (0.5, 1.0, 2.0)  # c, the cost of arm 2; arm 1 costs 1
BUDGETS = (0.1, 0.3, 0.5)  # B as a share of the number of test rows
# Instance i of H6 and H7 (Section 8) is INSTANCES[i], c outer and B inner as grids are enumerated in
# Section 6; i seeds the draw of baseline b1.
INSTANCES = [(c, k) for c in COSTS for k in BUDGETS]


def costs(c):
    return np.array([0.0, 1.0, c])


def gains(v):
    return np.c_[np.zeros(len(v)), v]


def value(v, assignment):
    return float(gains(v)[np.arange(len(v)), assignment].sum())


def exact(v, c, budget, time_limit=1800.0):
    """HiGHS MILP: maximise the gain with one arm per user and total cost at most the budget.

    H6 (Section 8): a user's arm is the argmax of the user's column values; HiGHS uses its default
    number of threads."""
    n, cost = len(v), costs(c)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 3 * n, n + 1
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.col_cost_ = gains(v).ravel()
    lp.col_lower_, lp.col_upper_ = np.zeros(3 * n), np.ones(3 * n)
    lp.row_lower_ = np.r_[np.ones(n), -highspy.kHighsInf]
    lp.row_upper_ = np.r_[np.ones(n), budget]
    user, arm = np.repeat(np.arange(n), 3), np.tile(np.arange(3), n)
    keep = np.c_[np.ones(3 * n, bool), cost[arm] > 0]  # column (i, j): row i, and the budget row if c_j > 0
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.r_[0, np.cumsum(keep.sum(axis=1))]
    lp.a_matrix_.index_ = np.c_[user, np.full(3 * n, n)][keep]
    lp.a_matrix_.value_ = np.c_[np.ones(3 * n), cost[arm]][keep]
    lp.integrality_ = [highspy.HighsVarType.kInteger] * (3 * n)
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("mip_rel_gap", 1e-6)
    solver.setOptionValue("time_limit", time_limit)
    solver.passModel(lp)
    start = time.perf_counter()
    solver.run()
    seconds = time.perf_counter() - start
    info = solver.getInfo()
    assignment = np.asarray(solver.getSolution().col_value).reshape(n, 3).argmax(axis=1)
    return {
        "assignment": assignment,
        "value": value(v, assignment),
        "status": solver.modelStatusToString(solver.getModelStatus()),
        "gap": info.mip_gap,
        "bound": info.mip_dual_bound,
        "seconds": seconds,
    }


def dual(v, c, budget, halvings=60):
    """Bisection on lambda over [0, max v / min positive cost]; the assignment at the final upper end.

    max v includes arm 0's gain of 0 (H6, Section 8)."""
    cost, g = costs(c), gains(v)
    choose = lambda lam: np.argmax(g - lam * cost, axis=1)  # ties to the lower arm
    start = time.perf_counter()
    low, high = 0.0, g.max() / cost[cost > 0].min()
    for _ in range(halvings):
        mid = (low + high) / 2
        if cost[choose(mid)].sum() > budget:
            low = mid
        else:
            high = mid
    assignment = choose(high)
    return {"assignment": assignment, "value": value(v, assignment), "lambda": high, "seconds": time.perf_counter() - start}


def email_arm(v):
    """The e-mail arm with the larger mean gain over test rows, ties to arm 1."""
    return 2 if v[:, 1].mean() > v[:, 0].mean() else 1


def baselines(v, c, budget, score, instance):
    """(b1) random users and (b2) users ranked by the score (ties by row order), all sent the e-mail arm a."""
    n, a = len(v), email_arm(v)
    size = min(n, int(np.floor(budget / costs(c)[a])))
    b1, b2 = np.zeros(n, np.int64), np.zeros(n, np.int64)
    b1[np.random.default_rng([20261115, 5, instance]).choice(n, size, replace=False)] = a
    b2[np.argsort(-score, kind="stable")[:size]] = a
    return b1, b2


def policy_values(policies, arm, y, mu, rows=None):
    """IPW and AIPW values of each policy (rows: policies; columns: IPW, AIPW), with the arm shares of the rows used.

    `arm` is the observed arm, `mu` the (n, 3) predictions of the three outcome models. For the intervals
    of H7 (Section 8) the policies are fixed on all test rows; a replicate recomputes the estimators,
    the arm shares included."""
    if rows is not None:
        policies, arm, y, mu = [p[rows] for p in policies], arm[rows], y[rows], mu[rows]
    shares = np.bincount(arm, minlength=3) / len(arm)
    out = []
    for policy in policies:
        taken = (arm == policy) / shares[policy]
        fitted = mu[np.arange(len(arm)), policy]
        out.append([np.mean(taken * y), np.mean(fitted + taken * (y - fitted))])
    return np.array(out)


def replicate_policy_values(policies, arm, y, mu, group, unit=4, replicates=2000):
    """Array (replicate, policy, estimator) on the half-sample replicates of Section 5.5."""
    return np.array([policy_values(policies, arm, y, mu, replicate_rows(group, unit, r)) for r in range(replicates)])

"""Monte Carlo check of the closed-form truths in dgp.py.

Counterfactual outcomes are simulated under each treatment arm with shared
noise, and their differences are regressed on the conditioning history.

    python truth.py [--n 2000000] [--seed 1]
"""

import argparse

import numpy as np

import dgp
from config import Params


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def contrast(n, seed, p, arm, arm_ref):
    """Simulate both arms with shared noise; return (reference data, Y difference)."""
    d = dgp.simulate(n, np.random.default_rng(seed), p, arm)
    d_ref = dgp.simulate(n, np.random.default_rng(seed), p, arm_ref)
    return d_ref, d["Y"] - d_ref["Y"]


def one(*_):
    return 1.0


def zero(*_):
    return 0.0


def check(p, n, seed):
    """Return rows (quantity, closed form, Monte Carlo)."""
    rows = []

    def add(name, closed, mc):
        for label, c, m in zip(name, closed, mc):
            rows.append((label, c, m))

    # Trial 0: Y^{1,1,1} - Y^{0,1,1} given L0
    d, diff = contrast(n, seed, p, (one, one, one), (zero, one, one))
    X0 = np.column_stack([np.ones(n), d["L0"]])
    add(["gamma0: 1", "gamma0: L0"], dgp.true_psi(p), ols(X0, diff))

    # Naive g-estimand: Y^{1,1,0} - Y^{0,1,0} given L0
    d, diff = contrast(n, seed, p, (one, one, zero), (zero, one, zero))
    add(["naive limit: 1", "naive limit: L0"], dgp.naive_limit(p), ols(X0, diff))

    # Trial 1: Y^{A0,1,0} - Y^{A0,0,1} given S1 (L0 and A0 should not enter)
    d, diff = contrast(n, seed, p, (None, one, zero), (None, zero, one))
    X1 = np.column_stack([np.ones(n), d["L1"], d["L0"], d["A0"]])
    add(
        ["gamma1: 1", "gamma1: L1", "gamma1: L0", "gamma1: A0"],
        [p.t10, p.t11, 0.0, 0.0],
        ols(X1, diff),
    )

    # Trial 2: Y^{A0,A1,1} - Y^{A0,A1,0} given S2
    d, diff = contrast(n, seed, p, (None, None, one), (None, None, zero))
    X2 = np.column_stack([np.ones(n), d["L2"], d["A0"], d["A0"] * d["L0"]])
    add(
        ["gamma2: 1", "gamma2: L2", "gamma2: A0", "gamma2: A0*L0"],
        [p.t20, p.t21, p.t22, p.t23],
        ols(X2, diff),
    )

    # mu0(L0, a2) = E(Y^{0,1,a2} | L0)
    for a2, arm in ((1, (zero, one, one)), (0, (zero, one, zero))):
        d = dgp.simulate(n, np.random.default_rng(seed), p, arm)
        intercept = dgp.mu0(p, 0.0, a2)
        slope = dgp.mu0(p, 1.0, a2) - intercept
        add(
            [f"mu0(a2={a2}): 1", f"mu0(a2={a2}): L0"],
            [intercept, slope],
            ols(X0, d["Y"]),
        )

    # q(S1) = E{gamma2(S2) | S1, A1 = 1}, from observational data
    d = dgp.simulate(n, np.random.default_rng(seed), p)
    keep = d["A1"] == 1
    Xq = np.column_stack([np.ones(n), d["L1"], d["A0"], d["A0"] * d["L0"]])
    add(
        ["q: 1", "q: L1", "q: A0", "q: A0*L0"],
        [p.t20 + p.t21 * p.l22, p.t21 * p.l21, p.t21 * p.l23 + p.t22, p.t23],
        ols(Xq[keep], dgp.gamma2(p, d["L0"], d["A0"], d["L2"])[keep]),
    )

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"{'quantity':<20}{'closed form':>12}{'Monte Carlo':>13}{'diff':>9}")
    for name, closed, mc in check(Params(), args.n, args.seed):
        print(f"{name:<20}{closed:>12.4f}{mc:>13.4f}{mc - closed:>9.4f}")

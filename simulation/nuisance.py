"""Nuisance functions for the Trial 0 estimating equations.

gamma1 and gamma2 are treated as known, so the nuisances are pi0, pi1 and q.
The outcome regression mu0 depends on the pseudo-outcome and is fitted in
estimators.py.
"""

import numpy as np

import dgp


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_logistic(X, y, max_iter=50, tol=1e-10):
    """Logistic regression by Newton-Raphson."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        prob = dgp.expit(X @ beta)
        step = np.linalg.solve(X.T @ (X * (prob * (1 - prob))[:, None]), X.T @ (y - prob))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def estimated(data, p):
    """Correctly specified parametric fits of pi0, pi1 and q."""
    L0, A0, L1, A1, L2 = (data[k] for k in ("L0", "A0", "L1", "A1", "L2"))
    one = np.ones_like(L0)

    X_pi0 = np.column_stack([one, L0])
    pi0 = dgp.expit(X_pi0 @ fit_logistic(X_pi0, A0))

    X_pi1 = np.column_stack([one, L1, A0])
    pi1 = dgp.expit(X_pi1 @ fit_logistic(X_pi1, A1))

    # q(S1) = E{gamma2(S2) | S1, A1 = 1}: regress gamma2 on S1 among A1 = 1
    X_q = np.column_stack([one, L1, A0, A0 * L0])
    treated = A1 == 1
    q = X_q @ ols(X_q[treated], dgp.gamma2(p, L0, A0, L2)[treated])

    return dict(pi0=pi0, pi1=pi1, q=q)


def oracle(data, p):
    """True pi0, pi1 and q."""
    L0, A0, L1 = data["L0"], data["A0"], data["L1"]
    return dict(pi0=dgp.pi0(p, L0), pi1=dgp.pi1(p, A0, L1), q=dgp.q(p, L0, A0, L1))

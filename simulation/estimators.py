"""Naive and bias-corrected g-estimation of gamma0(L0; psi) = psi0 + psi1 L0.

All estimators solve

    P_n[ c(L0) {A0 - pi0(L0)} {H - A0 gamma0(L0; psi) - mu0(L0)} ] = 0,

with c(L0) = (1, L0)', i.e. the influence function of g-estimation, and differ
only in the pseudo-outcome H:

    naive      H = Y - (A2 - h_2^1) gamma2 - (A1 - h_1^0) gamma1        (Equation 4)
    corrected  H = naive + R (Delta gamma2 - q) + q                     (Theorem 4)

Here h_1^0 = 1, h_2^1 = 1 - A1, R = A1 / pi1(S1), and Delta = h_2^0 - h_2^1 = 1
whenever R is nonzero.
"""

import numpy as np

import dgp
import nuisance

METHODS = ("naive", "corrected")


def pseudo_outcomes(data, p, pi1, q):
    L0, A0, L1, A1, L2, A2, Y = (data[k] for k in ("L0", "A0", "L1", "A1", "L2", "A2", "Y"))
    g2 = dgp.gamma2(p, L0, A0, L2)
    g1 = dgp.gamma1(p, L1)
    R = A1 / pi1

    naive = Y - (A2 - (1 - A1)) * g2 - (A1 - 1) * g1
    return {
        "naive": naive,
        "corrected": naive + R * (g2 - q) + q,
    }


def solve(L0, A0, H, pi0, mu0):
    """Solve the estimating equation; return (psi_hat, sandwich se).

    The standard error treats the nuisance functions as fixed.
    """
    n = len(L0)
    X = np.column_stack([np.ones(n), L0])
    W = X * (A0 - pi0)[:, None]

    bread = W.T @ (X * A0[:, None]) / n
    psi = np.linalg.solve(bread, W.T @ (H - mu0) / n)

    U = W * (H - A0 * (X @ psi) - mu0)[:, None]
    bread_inv = np.linalg.inv(bread)
    cov = bread_inv @ (U.T @ U / n) @ bread_inv.T / n
    return psi, np.sqrt(np.diag(cov))


def fit_mu0(L0, A0, H):
    """E(H | L0, A0 = 0), linear in L0, fitted among A0 = 0."""
    X = np.column_stack([np.ones_like(L0), L0])
    untreated = A0 == 0
    return X @ nuisance.ols(X[untreated], H[untreated])


def estimate(data, p, nuisance_mode="estimated"):
    """Return {method: (psi_hat, se)} for all methods."""
    L0, A0 = data["L0"], data["A0"]
    if nuisance_mode == "oracle":
        nu = nuisance.oracle(data, p)
    else:
        nu = nuisance.estimated(data, p)

    out = {}
    for method, H in pseudo_outcomes(data, p, nu["pi1"], nu["q"]).items():
        if nuisance_mode == "oracle":
            mu0 = dgp.mu0(p, L0, a2=0 if method == "naive" else 1)
        else:
            mu0 = fit_mu0(L0, A0, H)
        out[method] = solve(L0, A0, H, nu["pi0"], mu0)
    return out

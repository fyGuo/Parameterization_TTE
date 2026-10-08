"""Data generating mechanism and its closed-form truths.

The outcome mean is

    E(Y | S2, A2) = b(L) + A2 gamma2(S2) + A1 d1(S1) + A0 d0(S0),

where d1 and d0 are back-solved so that the Trial 1 and Trial 0 contrasts equal
the target blips gamma1 and gamma0 (the construction in the proof of Theorem A.1).
"""

import numpy as np


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- blips

def gamma2(p, L0, A0, L2):
    return p.t20 + p.t21 * L2 + A0 * (p.t22 + p.t23 * L0)


def gamma1(p, L1):
    return p.t10 + p.t11 * L1


def gamma0(p, L0):
    return p.psi0 + p.psi1 * L0


# ------------------------------------------------- pieces of the outcome mean

def mean_gamma2_given_s1(p, L0, A0, L1, a1):
    """E{gamma2(S2) | S1, A1 = a1}."""
    mean_L2 = p.l21 * L1 + p.l22 * a1 + p.l23 * A0
    return gamma2(p, L0, A0, mean_L2)


def d1(p, L0, A0, L1):
    return gamma1(p, L1) - p.b3 * p.l22 + mean_gamma2_given_s1(p, L0, A0, L1, 0)


def kappa0(p, L0):
    """Trial 0 contrast when d0 = 0."""
    return (
        p.b2 * p.l12
        + p.b3 * p.tau
        + p.t11 * p.l12
        + 2 * (p.t21 * p.tau + p.t22 + p.t23 * L0)
    )


def d0(p, L0):
    return gamma0(p, L0) - kappa0(p, L0)


def outcome_mean(p, L0, A0, L1, A1, L2, A2):
    return (
        p.b0 + p.b1 * L0 + p.b2 * L1 + p.b3 * L2
        + A2 * gamma2(p, L0, A0, L2)
        + A1 * d1(p, L0, A0, L1)
        + A0 * d0(p, L0)
    )


# ---------------------------------------------------------- true nuisances

def pi0(p, L0):
    return expit(p.a00 + p.a01 * L0)


def pi1(p, A0, L1):
    return expit(p.a10 + p.a11 * L1 + p.a12 * A0)


def pi2(p, A1, L2):
    return expit(p.a20 + p.a21 * L2 + p.a22 * A1)


def q(p, L0, A0, L1):
    """q(S1) = E{Delta gamma2(S2) | S1, A1 = h_1^0}, with Delta = 1 and h_1^0 = 1."""
    return mean_gamma2_given_s1(p, L0, A0, L1, 1)


def mu0(p, L0, a2):
    """E(Y^{0,1,a2} | L0): a2 = 1 is mu_0^0 of the paper, a2 = 0 its naive analogue."""
    mean_L1 = p.l11 * L0
    mean_L2 = p.l21 * mean_L1 + p.l22
    return (
        p.b0 + p.b1 * L0 + p.b2 * mean_L1 + p.b3 * mean_L2
        + a2 * (p.t20 + p.t21 * mean_L2)
        + d1(p, L0, 0, mean_L1)
    )


# ------------------------------------------------------------ true targets

def true_psi(p):
    """(intercept, slope) of gamma0(L0) = E(Y^{1,1,1} - Y^{0,1,1} | L0)."""
    return np.array([p.psi0, p.psi1])


def naive_limit(p):
    """(intercept, slope) of E(Y^{1,1,0} - Y^{0,1,0} | L0), the naive g-estimand."""
    return np.array([p.psi0 - p.t21 * p.tau - p.t22, p.psi1 - p.t23])


# ---------------------------------------------------------------- simulate

def simulate(n, rng, p, regime=(None, None, None)):
    """Draw n i.i.d. observations.

    regime = (g0, g1, g2): each entry is None (treatment follows its propensity)
    or a function of the treatment history so far, e.g. `lambda A0, A1: 1 - A1`,
    returning the value the treatment is set to. All randomness is drawn up
    front, so calls with the same seed share noise across regimes.
    """
    e0, e1, e2, ey = rng.standard_normal((4, n))
    u0, u1, u2 = rng.uniform(size=(3, n))
    g0, g1, g2 = regime

    L0 = e0
    A0 = (u0 < pi0(p, L0)).astype(float) if g0 is None else g0() * np.ones(n)
    L1 = p.l11 * L0 + p.l12 * A0 + e1
    A1 = (u1 < pi1(p, A0, L1)).astype(float) if g1 is None else g1(A0) * np.ones(n)
    L2 = p.l21 * L1 + p.l22 * A1 + p.l23 * A0 + e2
    A2 = (u2 < pi2(p, A1, L2)).astype(float) if g2 is None else g2(A0, A1) * np.ones(n)
    Y = outcome_mean(p, L0, A0, L1, A1, L2, A2) + p.sigma_y * ey

    return dict(L0=L0, A0=A0, L1=L1, A1=A1, L2=L2, A2=A2, Y=Y)

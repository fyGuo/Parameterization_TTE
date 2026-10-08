"""Parameters of the three-time simulation (Figure 4 of the paper).

Data: (L0, A0, L1, A1, L2, A2, Y), with S0 = L0, S1 = (L0, A0, L1),
S2 = (L0, A0, L1, A1, L2).

Causal contrasts (variation-independent, but not an SNMM):
    Trial 2: E(Y^{A0,A1,1} - Y^{A0,A1,0} | S2) = gamma2(S2)
    Trial 1: E(Y^{A0,1,0}  - Y^{A0,0,1}  | S1) = gamma1(S1),  h_2^1 = 1 - A1
    Trial 0: E(Y^{1,1,1}   - Y^{0,1,1}   | S0) = gamma0(S0),  h_1^0 = h_2^0 = 1
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # A0 | L0 ~ Bern(expit(a00 + a01 L0))
    a00: float = 0.0
    a01: float = 0.5
    # L1 | L0, A0 ~ N(l11 L0 + l12 A0, 1)
    l11: float = 0.5
    l12: float = 0.5
    # A1 | S1 ~ Bern(expit(a10 + a11 L1 + a12 A0))
    a10: float = 0.0
    a11: float = 0.5
    a12: float = -0.3
    # L2 | S1, A1 ~ N(l21 L1 + l22 A1 + l23 A0, 1)
    l21: float = 0.5
    l22: float = 0.5
    l23: float = 0.3
    # A2 | S2 ~ Bern(expit(a20 + a21 L2 + a22 A1))
    a20: float = 0.0
    a21: float = 0.5
    a22: float = -0.3
    # Treatment-free part of E(Y | S2, A2): b0 + b1 L0 + b2 L1 + b3 L2
    b0: float = 0.0
    b1: float = 1.0
    b2: float = 1.0
    b3: float = 1.0
    # gamma2(S2) = t20 + t21 L2 + A0 (t22 + t23 L0)
    t20: float = 1.0
    t21: float = 0.5
    t22: float = 1.0
    t23: float = 0.5
    # gamma1(S1) = t10 + t11 L1
    t10: float = 1.0
    t11: float = 0.5
    # gamma0(S0) = psi0 + psi1 L0, the target of estimation
    psi0: float = 1.0
    psi1: float = 0.5
    # sd of the outcome error
    sigma_y: float = 1.0

    @property
    def tau(self):
        """Effect of A0 on E(L2) with A1 held fixed."""
        return self.l21 * self.l12 + self.l23

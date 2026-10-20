"""
Global minimum at L -> infinity.

b = 0 (exact): for M ≤ mu the potential is M² [1/(2g) + mu/(2π)], for M ≥ mu its derivative is
M (1/g + M/π). Hence
    mu < π/|g|:  vacuum phase V, M* = π/|g|;
    mu > π/|g|:  trivial phase T, M* = 0;
    mu = π/|g|:  Omega is flat on [0, π/|g|] — the minimum is degenerate.
"""

import math
from dataclasses import dataclass

from njl3.inf.formulas import M_vacuum, omega_rel_b0


@dataclass
class ResultB0:
    mu: float
    g: float
    phase: str           # 'V' or 'T'
    M_star: float
    omega_star: float    # Omega(M*) - Omega(0)
    degenerate: bool     # mu = π/|g|: every M in [0, π/|g|] is a minimum


def find_minimum_b0(mu, g, tol=1e-12):
    Mv = M_vacuum(g)
    if abs(mu - Mv) <= tol * max(1.0, Mv):
        return ResultB0(mu, g, 'T', 0.0, 0.0, True)
    if mu < Mv:
        return ResultB0(mu, g, 'V', Mv, omega_rel_b0(Mv, mu, g), False)
    return ResultB0(mu, g, 'T', 0.0, 0.0, False)

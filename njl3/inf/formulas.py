"""
Closed forms for NJL3 at L -> infinity.

b = 0. The two branches coincide, E = sqrt(M² + p²) in the (p1, p2) plane, and
    W(M) = -(1/π) ∫_M^mu E (mu - E) dE = -(mu - M)² (mu + 2M) / (6π)   (M < mu),  0 (M ≥ mu),
    ΔV(M) = M³ / (3π).
For M ≤ mu the cubic terms cancel and the potential is purely quadratic:
    Omega(M) - Omega(0) = M² [1/(2g) + mu/(2π)],
and the gap function G = (1/M) dOmega/dM is constant there:
    G(M) = 1/g + mu/π   (M < mu),   1/g + M/π   (M ≥ mu).
"""

import math

PI = math.pi


def W_b0(M, mu):
    """W_mu(M, b = 0)."""
    if mu <= 0.0 or M >= mu:
        return 0.0
    return -(mu - M)**2 * (mu + 2.0 * M) / (6.0 * PI)


def omega_rel_b0(M, mu, g):
    """Omega(M, 0) - Omega(0, 0)."""
    return M * M / (2.0 * g) + M**3 / (3.0 * PI) + W_b0(M, mu) - W_b0(0.0, mu)


def gap_G_b0(M, mu, g):
    """Gap function G(M) at b = 0 (dOmega/dM = M G)."""
    return 1.0 / g + (mu if M < mu else M) / PI


def M_vacuum(g):
    """Vacuum mass (mu = 0): the root of 1/g + M/π = 0."""
    return -PI / g

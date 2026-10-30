"""
NJL3 at L -> infinity (R²) and temperature T > 0: the matter term with the Fermi–Dirac logarithms,

    W_T(M, b) = -T ∫ dp1/(2π) ∫ dp2/(2π) Σ± Σ_{s=±1} ln(1 + exp(-(E± - s mu)/T)),
    E± = sqrt((E1 ± b)² + p2²),  E1 = sqrt(M² + p1²),

Omega(M, b) - Omega(0, 0) = M²/(2g) + ΔV(M, b) + W_T(M, b) - W_T(0, 0)  (ΔV is the T = 0 vacuum term, dU).
As T -> 0, -T ln(1 + e^{-(E - mu)/T}) -> -(mu - E)₊ and W_T -> W of njl3.inf.formulas.

The p2 integral gives J_T(A) = ∫ dq T Σ_s ln(1 + e^{-(sqrt(A² + q²) - s mu)/T}) (even in A, smooth), so
    W_T(M, b) = -(1/(2π²)) Σ± ∫_0^∞ dp J_T(E(p) ± b)          (as at T = 0 with J -> J_T).
b = 0 in closed form (polylogarithms, z± = e^{(±mu - M)/T}):
    W_T(M, 0) = (T/π) Σ± [M T Li2(-z±) + T² Li3(-z±)],
    G(M) = (1/M) dOmega/dM = 1/g + M/π + (T/π) Σ± ln(1 + z±),
G(0) = 0 gives the boundary of the broken phase  M_vac = 2T ln(2 cosh(mu/(2T)))  (M_vac = π/|g|).
"""

import math

import mpmath as mp
from scipy.integrate import quad

PI = math.pi


def _lnfd(x):
    """ln(1 + e^{-x}) without overflow."""
    return math.log1p(math.exp(-x)) if x > 0 else -x + math.log1p(math.exp(x))


def W_T_b0(M, mu, T):
    """W_T(M, b = 0) in closed form."""
    s = 0.0
    for sg in (1.0, -1.0):
        z = mp.e ** ((sg * mu - M) / T)
        s += M * T * float(mp.polylog(2, -z)) + T * T * float(mp.polylog(3, -z))
    return T / PI * s


def gap_G_T_b0(M, mu, T, g):
    """G(M) = (1/M) dOmega/dM at b = 0."""
    s = sum(_lnfd((M - sg * mu) / T) for sg in (1.0, -1.0))
    return 1.0 / g + M / PI + T / PI * s


def omega_rel_T_b0(M, mu, T, g):
    """Omega(M, 0) - Omega(0, 0) at temperature T."""
    return M * M / (2.0 * g) + M ** 3 / (3.0 * PI) + W_T_b0(M, mu, T) - W_T_b0(0.0, mu, T)


def boundary_mu(T, g):
    """mu on the boundary of the broken phase (b = 0): M_vac = 2T ln(2 cosh(mu/2T)); None if T >= T_c(0)."""
    Mv = -PI / g
    x = math.exp(Mv / (2.0 * T)) / 2.0
    if x < 1.0:
        return None
    return 2.0 * T * math.acosh(x)


def J_T(A, mu, T):
    """∫_R dq T Σ_s ln(1 + e^{-(sqrt(A² + q²) - s mu)/T})."""
    a = abs(A)
    qmax = math.sqrt(max((mu + 40.0 * T) ** 2 - a * a, 0.0)) + 40.0 * T
    f = lambda q: T * sum(_lnfd((math.sqrt(a * a + q * q) - sg * mu) / T) for sg in (1.0, -1.0))
    pts = [math.sqrt(mu * mu - a * a)] if a < mu else None
    return 2.0 * quad(f, 0.0, qmax, points=pts, limit=200, epsabs=1e-14, epsrel=1e-11)[0]


def W_T(M, b, mu, T):
    """W_T(M, b) by quadrature (b ≥ 0)."""
    Emax = mu + b + 45.0 * T
    pmax = math.sqrt(max(Emax * Emax - M * M, 0.0)) + 45.0 * T
    tot = 0.0
    for sg in (1.0, -1.0):
        f = lambda p: J_T(math.sqrt(M * M + p * p) + sg * b, mu, T)
        pts = [x for x in (math.sqrt(max((mu - sg * b) ** 2 - M * M, 0.0)),
                           math.sqrt(max(b * b - M * M, 0.0))) if 0.0 < x < pmax]
        tot += quad(f, 0.0, pmax, points=pts or None, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
    return -tot / (2.0 * PI * PI)


def omega_rel_T(M, b, mu, T, g):
    """Omega(M, b) - Omega(0, 0) at temperature T."""
    from njl3.full.formulas import dU
    if M == 0.0:
        return 0.0
    return M * M / (2.0 * g) + dU(M, b) + W_T(M, b, mu, T) - W_T_b0(0.0, mu, T)

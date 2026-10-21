"""
Homogeneous NJL3 / GN potential on R² × S¹ (b = 0) for an arbitrary twist phi of the boundary
condition and an arbitrary renormalized coupling g.

Modes: k_n = 2π(n + φ)/L, n ∈ Z; m_n = sqrt(M² + k_n²), P_n = sqrt(mu² - m_n²),
thresholds S_n = sqrt(mu² - k_n²). φ and 1 - φ give the same set |k_n|, so φ is reduced to [0, 1/2].

    Omega(M) = M²/(2g) + M³/(3π) + Omega_L(M) + Omega_muL(M),
    Omega_L  = (2/(π L³)) Re[ LM Li₂(z) + Li₃(z) ],   z = e^{-LM + 2πiφ}   (Poisson sum with the twist),
    Omega_muL = -(2/(π L)) Σ_{n∈Z, m_n<mu} I(m_n),    I(m) = mu P/2 - (m²/2) ln((mu + P)/m).

Gap equation dOmega/dM = M G(M):

    G(M) = 1/g + M/π + (1/(π L)) ln D(LM) + (2/(π L)) Σ_{n∈Z, m_n<mu} arccosh(mu/m_n),
    D(x) = 1 - 2 e^{-x} cos 2πφ + e^{-2x} = (1 - e^{-x})² + 4 e^{-x} sin² πφ.

φ = 0 reproduces njl3.b0 (D = (1 - e^{-x})², the modes ±n pair up with C_n = 2); at φ = 0 this module
delegates to njl3.b0, so the results coincide exactly.

Monotone splitting for φ > 0 (all terms finite at M = 0):
    A(M) = M/π + (1/(π L)) ln D(LM),  A'(M) = (1 - e^{-2x}) / (π D) > 0  — increasing;
    B(M) = (2/(π L)) Σ arccosh(mu/m_n)                                — decreasing.
For M ≥ S_max = max S_n no mode is filled and G = 1/g + A(M) is increasing (at most one root).
"""

import math

import mpmath
import numpy as np

from njl3.b0 import formulas as F0

PI = math.pi


def reduce_phi(phi):
    """φ → [0, 1/2] (φ, φ + 1 and 1 - φ are equivalent)."""
    p = phi % 1.0
    return min(p, 1.0 - p)


# ==============================================================================
# Modes
# ==============================================================================

def modes(mu, L, phi):
    """Modes that can be filled (|k_n| < mu): list of k_n (signed), ascending in |k_n|."""
    phi = reduce_phi(phi)
    if mu <= 0.0:
        return []
    nmax = int(math.floor(L * mu / (2.0 * PI))) + 1
    ks = [2.0 * PI * (n + phi) / L for n in range(-nmax - 1, nmax + 1)]
    return sorted((k for k in ks if abs(k) < mu), key=abs)


def thresholds(mu, L, phi):
    """Distinct thresholds S_n = sqrt(mu² - k_n²), ascending."""
    return sorted(set(round(math.sqrt(mu * mu - k * k), 15) for k in modes(mu, L, phi)))


def levels(mu, L, phi):
    """Distinct |k_n| of the fillable modes, ascending (level j holds 1 or 2 modes)."""
    return sorted(set(round(abs(k), 13) for k in modes(mu, L, phi)))


def filled_level(M, mu, L, phi):
    """Index j of the highest filled level |k| at mass M (-1 if none)."""
    j = -1
    for i, k in enumerate(levels(mu, L, phi)):
        if M * M + k * k < mu * mu:
            j = i
    return j


def S_max(mu, L, phi):
    """Largest threshold (0 if no mode can be filled)."""
    ks = modes(mu, L, phi)
    return math.sqrt(mu * mu - ks[0] ** 2) if ks else 0.0


# ==============================================================================
# Potential
# ==============================================================================

def dU(M):
    return M**3 / (3.0 * PI)


def omega_L(M, L, phi):
    """Omega_L(M) = (2/(π L³)) Re[LM Li₂(z) + Li₃(z)], z = e^{-LM + 2πiφ}."""
    z = mpmath.exp(mpmath.mpc(-L * M, 2.0 * PI * phi))
    return 2.0 / (PI * L**3) * float(mpmath.re(L * M * mpmath.polylog(2, z) + mpmath.polylog(3, z)))


def _I(m, mu):
    if m >= mu:
        return 0.0
    P = math.sqrt(mu * mu - m * m)
    if m == 0.0:
        return mu * P / 2.0
    return mu * P / 2.0 - m * m / 2.0 * math.log((mu + P) / m)


def omega_muL(M, mu, L, phi):
    return -2.0 / (PI * L) * sum(_I(math.sqrt(M * M + k * k), mu) for k in modes(mu, L, phi))


def omega_rel_closed(M, mu, L, g, phi):
    """Omega(M) - Omega(0) from the closed forms (complex polylogarithms)."""
    phi = reduce_phi(phi)
    return (M * M / (2.0 * g) + dU(M) + omega_L(M, L, phi) - omega_L(0.0, L, phi)
            + omega_muL(M, mu, L, phi) - omega_muL(0.0, mu, L, phi))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def omega_rel(M, mu, L, g, phi, M_switch=0.5):
    """Omega(M) - Omega(0): ∫_0^M m G dm (substitution m = S - u² near thresholds) for M < M_switch,
    closed form beyond (as njl3.b0.omega_rel)."""
    phi = reduce_phi(phi)
    if phi == 0.0:
        return F0.omega_rel(M, mu, L, g, M_switch)
    if M == 0.0:
        return 0.0
    if M >= M_switch:
        return omega_rel_closed(M, mu, L, g, phi)
    S_all = thresholds(mu, L, phi)
    edges = [0.0] + [S for S in S_all if 0.0 < S < M] + [M]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        S = next((x for x in S_all if x >= b), None)
        if S is None or S - b >= b - a:
            m = 0.5 * (b - a) * _GL_X + 0.5 * (a + b)
            total += 0.5 * (b - a) * np.dot(_GL_W, m * gap_G_vec(m, mu, L, g, phi))
        else:
            ua, ub = math.sqrt(S - b), math.sqrt(S - a)
            u = 0.5 * (ub - ua) * _GL_X + 0.5 * (ua + ub)
            m = np.maximum(S - u * u, 1e-300)
            total += 0.5 * (ub - ua) * np.dot(_GL_W, 2.0 * u * m * gap_G_vec(m, mu, L, g, phi))
    return float(total)


# ==============================================================================
# Gap function
# ==============================================================================

def _lnD(x, phi):
    """ln D(x), D = (1 - e^{-x})² + 4 e^{-x} sin² πφ (no cancellation)."""
    x = np.asarray(x, dtype=float)
    return np.log(np.expm1(-x) ** 2 + 4.0 * np.exp(-x) * math.sin(PI * phi) ** 2)


def gap_G_vec(M, mu, L, g, phi):
    """G(M) for an array M ≥ 0 (M > 0 at φ = 0)."""
    phi = reduce_phi(phi)
    if phi == 0.0:
        return F0.gap_G_vec(M, mu, L, g)
    M = np.asarray(M, dtype=float)
    ks = modes(mu, L, phi)
    x = L * M
    if ks and abs(ks[0]) * L < 1e-2:
        # tiny φ: combine ln D with the lowest mode, ln D(x) + 2 arccosh(mu/m_0)
        #   = ln[D/(x² + θ²)] + 2 ln(L (mu + P_0)) for m_0 < mu,  θ = 2πφ = L k_0
        k0 = ks[0]
        th2 = (L * k0) ** 2
        m0sq = M * M + k0 * k0
        P0 = np.sqrt(np.clip(mu * mu - m0sq, 0.0, None))
        s = np.where(m0sq < mu * mu,
                     _lnD(x, phi) - np.log(x * x + th2) + 2.0 * np.log(L * (mu + P0)),
                     _lnD(x, phi))
        rest = ks[1:]
    else:
        s = _lnD(x, phi)
        rest = ks
    for k in rest:
        m = np.sqrt(M * M + k * k)
        P = np.sqrt(np.clip(mu * mu - m * m, 0.0, None))
        s = s + 2.0 * np.where(m < mu, np.log((mu + P) / m), 0.0)
    return 1.0 / g + M / PI + s / (PI * L)


def gap_G(M, mu, L, g, phi):
    return float(gap_G_vec(np.array([M]), mu, L, g, phi)[0])


def gap_G0(mu, L, g, phi):
    """G(0⁺)."""
    phi = reduce_phi(phi)
    if phi == 0.0:
        return F0.gap_G0(mu, L, g)
    return gap_G(0.0, mu, L, g, phi)


def part_A(M, L, phi):
    """A(M) = M/π + (1/(πL)) ln D(LM) — increasing (φ > 0)."""
    return M / PI + float(_lnD(L * M, phi)) / (PI * L)


def part_B(M, mu, L, phi):
    """B(M) = (2/(πL)) Σ arccosh(mu/m_n) — decreasing."""
    s = 0.0
    for k in modes(mu, L, phi):
        m = math.sqrt(M * M + k * k)
        if m < mu:
            s += math.acosh(mu / m)
    return 2.0 * s / (PI * L)

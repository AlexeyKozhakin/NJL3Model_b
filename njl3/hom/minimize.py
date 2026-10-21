"""
Certified global minimum of the homogeneous potential for arbitrary φ and g (generalizes njl3.b0).

On [0, S_max] G = 1/g + A(M) + B(M) (A increasing, B decreasing): bisection with pruning by
1/g + A(p) + B(q) ≤ G ≤ 1/g + A(q) + B(p) finds all roots. On [S_max, ∞) G = 1/g + A(M) is increasing.
At φ = 0 everything is delegated to njl3.b0.

Phases: 'T' (M* = 0), 'V' (M* > 0, no mode filled), 'I<j>' (levels |k| 0..j filled).
"""

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from njl3.b0 import minimize as B0
from njl3.hom import formulas as F


def isolate_roots(mu, L, g, phi, tol=1e-10):
    """All roots (M, kind) of G for M > 0, kind ∈ {'min', 'max', 'touch'}."""
    phi = F.reduce_phi(phi)
    if phi == 0.0:
        return B0.isolate_roots(mu, L, g, tol)
    G = lambda x: F.gap_G(x, mu, L, g, phi)
    inv_g = 1.0 / g
    roots = []
    Smax = F.S_max(mu, L, phi)
    if Smax > 0.0:
        windows = []
        A = lambda x: F.part_A(x, L, phi)
        Bf = lambda x: F.part_B(x, mu, L, phi)
        stack = [(0.0, Smax, A(0.0), A(Smax), Bf(0.0), Bf(Smax))]
        while stack:
            p, q, Ap, Aq, Bp, Bq = stack.pop()
            if inv_g + Ap + Bq > 0.0 or inv_g + Aq + Bp < 0.0:
                continue
            if q - p <= tol:
                windows.append((p, q))
                continue
            c = 0.5 * (p + q)
            Ac, Bc = A(c), Bf(c)
            stack.append((c, q, Ac, Aq, Bc, Bq))
            stack.append((p, c, Ap, Ac, Bp, Bc))
        windows.sort()
        merged = []
        for p, q in windows:
            if merged and p <= merged[-1][1] + 1e-15:
                merged[-1] = (merged[-1][0], max(merged[-1][1], q))
            else:
                merged.append((p, q))
        for p, q in merged:
            if p == 0.0 and q <= tol:
                continue
            Gp, Gq = G(p), G(q)
            if Gp < 0.0 < Gq:
                roots.append((brentq(G, p, q, xtol=1e-15), 'min'))
            elif Gp > 0.0 > Gq:
                roots.append((brentq(G, p, q, xtol=1e-15), 'max'))
            else:
                roots.append((0.5 * (p + q), 'touch'))
    a = max(Smax, 1e-300)
    if G(a) < 0.0:
        b = 2.0 * a + 1.0
        while G(b) < 0.0:
            b *= 2.0
        roots.append((brentq(G, a, b, xtol=1e-15), 'min'))
    return sorted(roots)


def find_minimum(mu, L, g, phi, tol=1e-10, roots=None):
    """(M*, Omega(M*) - Omega(0), [(M, Omega - Omega(0)) for each local minimum M > 0])."""
    phi = F.reduce_phi(phi)
    if roots is None:
        roots = isolate_roots(mu, L, g, phi, tol)
    minima = [(r, F.omega_rel(r, mu, L, g, phi)) for r, kind in roots if kind in ('min', 'touch')]
    best_M, best_Om = 0.0, 0.0
    for r, om in minima:
        if om < best_Om:
            best_M, best_Om = r, om
    return best_M, best_Om, minima


def phase_label(M_star, mu, L, phi):
    """'T', 'V' or 'I<j>' (j — highest filled level of |k_n|)."""
    if M_star == 0.0:
        return 'T'
    j = F.filled_level(M_star, mu, L, phi)
    return 'V' if j < 0 else f'I{j}'


def phase_at(mu, L, g, phi):
    M, _, _ = find_minimum(mu, L, g, phi)
    return phase_label(M, mu, L, phi), M


def M_vacuum(L, g, phi):
    """Vacuum mass (mu = 0): root of 1/g + A(M) = 0, or 0 if 1/g + A(0) ≥ 0 (φ > 0)."""
    phi = F.reduce_phi(phi)
    if phi == 0.0:
        return B0.M_vacuum(L, g)
    f = lambda M: 1.0 / g + F.part_A(M, L, phi)
    if f(0.0) >= 0.0:
        return 0.0
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 0.0, hi, xtol=1e-15, rtol=1e-15)


@dataclass
class PointResult:
    mu: float
    L: float
    g: float
    phi: float
    phase: str
    M_star: float
    omega_star: float
    G0: float
    roots: list = field(default_factory=list)
    local_minima: list = field(default_factory=list)


def analyze_point(mu, L, g, phi):
    phi = F.reduce_phi(phi)
    roots = isolate_roots(mu, L, g, phi)
    M, Om, minima = find_minimum(mu, L, g, phi, roots=roots)
    return PointResult(mu=mu, L=L, g=g, phi=phi, phase=phase_label(M, mu, L, phi), M_star=M,
                       omega_star=Om, G0=F.gap_G0(mu, L, g, phi), roots=roots, local_minima=minima)

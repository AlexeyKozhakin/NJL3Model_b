"""
Analytic small-M asymptotics of a channel (method M2 of PLAN.md).

At the threshold b = S of a level |k_j| (S = sqrt(mu² - k_j²), multiplicity C_j):

    Omega(M, S) - Omega(0, 0) = M² [A - c ln(1/M)] + O(M⁴ ln M),   c = C_j S / (2π L mu),

    A = 1/(2g) + S/4 + K_L - (1/(π L)) Σ_{l ≠ j} C_l K_l - (C_j/(π L)) K_ch.

Derivation (b = S fixed). Write E = sqrt(M² + p²). Every term but the channel level is an even, smooth
function g(E) integrated over p (up to kinks, harmless at this order), so its change is
M² ∫_0^∞ g'(p)/(2p) dp = M² ∫_0^∞ (g(p) - g(0))/(2p²) dp (integration by parts; no derivatives needed):

    ΔU(M, S) = M² S/4                                        (exact for S ≥ M),
    K_L      = (2/(π² L²)) ∫_0^∞ (g_L(p) - g_L(0))/(2p²) dp,  g_L(p) = s_φ(L(p + S)) + s_φ(L|p - S|),
    K_l      = ∫_0^∞ (h_l(p) - h_l(0))/(2p²) dp,              h_l(p) = Σ± (mu - sqrt((p ± S)² + k_l²))₊.

The channel level: its plus branch is empty (E + S < S impossible), the minus branch covers 0 < E < 2S,
I(M) = ∫_0^{sqrt(4S² - M²)} f(E) dp with f(E) = mu - sqrt((E - S)² + k_j²), f(0) = 0, f'(0) = S/mu.
Splitting f = (S/mu) E + r(E):

    I(M) - I(0) = M² { (S/(2 mu)) [ln(1/M) + ln(4S) - 1/2] + K_r } + O(M⁴ ln M),
    K_r = ∫_0^{2S} r'(p)/(2p) dp + r(2S) dP/d(M²) = ∫_0^{2S} r(p)/(2p²) dp
(the upper limit P = sqrt(4S² - M²) moves by -M²/(4S) and r(2S) = -2S²/mu ≠ 0: this term cancels the
boundary term r(2S)/(4S) of the integration by parts),

so K_ch = (S/(2 mu)) (ln(4S) - 1/2) + K_r, and Omega_μL = -(1/(π L)) Σ C (...) gives the terms above.
"""

import math

import numpy as np
from scipy.integrate import quad

from njl3.inh import formulas as F

PI = math.pi
_Q = dict(limit=400, epsabs=1e-15, epsrel=1e-13)


def _s(x, phi):
    return float(F.s_func(np.array([x]), phi)[0])


def K_L(S, L, phi):
    g0 = 2.0 * _s(L * S, phi)
    gL = lambda p: _s(L * (p + S), phi) + _s(L * abs(p - S), phi)
    f = lambda p: (gL(p) - g0) / (2 * p * p) if p > 0 else 0.0
    # beyond p = S + 64/L the table gives s = 0 for both terms: the tail is -g0/(2p²)
    pe = S + 64.0 / L
    I = quad(f, 0.0, S, **_Q)[0] + quad(f, S, pe, **_Q)[0] - g0 / (2 * pe)
    return 2.0 / (PI**2 * L * L) * I


def K_level(S, k, mu):
    """Regular level |k| ≠ k_j at b = S."""
    def h(p):
        v = 0.0
        for A in (p + S, p - S):
            v += max(mu - math.sqrt(A * A + k * k), 0.0)
        return v
    h0 = h(0.0)
    Sk = math.sqrt(max(mu * mu - k * k, 0.0))
    pts = sorted({x for x in (Sk - S, S - Sk, S + Sk) if x > 0})
    edges = [0.0] + pts + [S + Sk]
    f = lambda p: (h(p) - h0) / (2 * p * p) if p > 0 else 0.0
    I = sum(quad(f, a, b, **_Q)[0] for a, b in zip(edges[:-1], edges[1:]) if b > a)
    return I - h0 / (2 * (S + Sk))            # tail p > S + S_k: h = 0


def K_channel(S, k, mu):
    r = lambda E: mu - math.sqrt((E - S)**2 + k * k) - S / mu * E
    f = lambda p: r(p) / (2 * p * p) if p > 0 else 0.0
    pts = [S] if k == 0.0 else None              # k = 0: f has a kink at E = S
    Kr = quad(f, 0.0, 2 * S, points=pts, **_Q)[0]
    return S / (2 * mu) * (math.log(4 * S) - 0.5) + Kr


def channel_A(j, mu, L, g, phi):
    """(A, c) of the channel at the threshold of level j (levels of njl3.inh.formulas.levels)."""
    lv = F.levels(mu, L, phi)
    kj, Cj = lv[j]
    S = math.sqrt(mu * mu - kj * kj)
    A = 1.0 / (2 * g) + S / 4 + K_L(S, L, phi)
    A -= sum(C * K_level(S, k, mu) for i, (k, C) in enumerate(lv) if i != j) / (PI * L)
    A -= Cj * K_channel(S, kj, mu) / (PI * L)
    return A, Cj * S / (2 * PI * L * mu)

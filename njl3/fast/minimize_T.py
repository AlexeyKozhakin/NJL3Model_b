"""
Global minimum of Omega_T(M, b) on R² × S¹ at temperature T > 0 (njl3.fast.thermal, formula (A)).

The temperature only smears the matter term, so the minima at T > 0 continue the T = 0 ones (temperature lowers the
gaps). Candidates are taken from the T = 0 search (njl3.fast.minimize_nb with T = 0):

  - b = 0: every local minimum of Omega_T(M, 0) on a log grid of M and every T = 0 homogeneous minimum (njl3.hom)
    refined by bounded Brent (several homogeneous minima compete near first-order lines; a narrow well, e.g. V at
    M ≈ π for mu ≈ 3, may fall between grid points);
  - every channel B_n with a gap at T = 0 (and the T = 0 global well, if it won), refined by Nelder–Mead in
    (ln M, t = (b - S_n)/M) on Omega_T. A channel whose T = 0 gap is far below the temperature
    (T > DEAD · T_c,n, T_c,n = (e^γ/π)(S_n/mu) M_n) is not refined — its gap does not survive.

The minimum must be deeper than OMEGA_NOISE, otherwise the point is in phase T (M* = 0).
"""

import math

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from njl3.fast import minimize_nb
from njl3.fast.thermal import CtxT
from njl3.full.minimize import FullResult, OMEGA_NOISE, TC_RATIO
from njl3.hom import find_minimum as hom_find_minimum
from njl3.hom import phase_label as hom_phase_label
from njl3.inh import formulas as F

DEAD = 3.0                        # channel skipped if T > DEAD * T_c,n (T_c,n of the T = 0 gap)
LNM_LO = math.log(1e-6)
B_MIN = 1e-5                     # b below this: the channel slid to the b = 0 solution
_GM_B0 = np.geomspace(1e-4, 8.0, 90)


def _hom(ctx, starts=()):
    """Best b = 0 minimum of Omega_T: every local minimum of a ln M grid and every T = 0 homogeneous minimum
    (starts) refined by bounded Brent in a window around it."""
    om = np.array([ctx.omega(float(M), 0.0) for M in _GM_B0])
    x_grid = np.log(_GM_B0)
    h = x_grid[1] - x_grid[0]
    seeds = [(x_grid[i], 1.5 * h) for i in range(len(om))
             if om[i] < 0.0 and (i == 0 or om[i] <= om[i - 1]) and (i == len(om) - 1 or om[i] <= om[i + 1])]
    # T = 0 minima move little: a narrow window, so that Brent does not slide into a neighbouring well
    seeds += [(math.log(M), 0.03) for M in starts if M > 0]
    best_M, best_o = 0.0, 0.0
    for x0, w in seeds:
        r = minimize_scalar(lambda x: ctx.omega(math.exp(x), 0.0), bounds=(x0 - w, x0 + w),
                            method='bounded', options=dict(xatol=1e-11))
        if r.fun < best_o:
            best_M, best_o = math.exp(float(r.x)), float(r.fun)
    return (best_M, best_o) if best_o < -OMEGA_NOISE else (0.0, 0.0)


def _channel(ctx, S, M0, b0):
    """Nelder–Mead on Omega_T in (ln M, t) from the T = 0 well (M0, b0)."""
    def f(x):
        lm = max(x[0], LNM_LO)
        M = math.exp(lm)
        b = S + x[1] * M
        return ctx.omega(M, b) if b >= 0 else 1e3
    x0 = np.array([math.log(M0), (b0 - S) / M0])
    simplex = np.array([x0, x0 + [-0.3, 0.0], x0 + [0.0, 0.3]])
    r = minimize(f, x0, method='Nelder-Mead',
                 options=dict(initial_simplex=simplex, xatol=1e-7, fatol=1e-15, maxiter=800, maxfev=1600))
    lm = max(float(r.x[0]), LNM_LO)
    M = math.exp(lm)
    b = S + float(r.x[1]) * M
    if r.fun < -OMEGA_NOISE and lm > LNM_LO + 1e-6 and b > 1e-8:
        return M, b, float(r.fun)
    return 0.0, b, 0.0


def find_minimum(mu, L, g, phi=0.0, T=1e-3):
    phi = F.check_phi(phi)
    r0 = minimize_nb.find_minimum(mu, L, g, phi, T=0.0)
    ctx = CtxT(mu, L, g, phi, T)
    _, _, hom_minima = hom_find_minimum(mu, L, g, phi)
    Mh, Oh = _hom(ctx, [M for M, _ in hom_minima])
    cand = [dict(phase=hom_phase_label(Mh, mu, L, phi), M=Mh, b=0.0, omega=Oh)]
    starts = []
    for ch in r0.channels:
        if ch['M'] > 0 and ch['b'] > 1e-8:
            Tc = TC_RATIO * (ch['S'] / mu) * ch['M']
            if T <= DEAD * Tc:
                starts.append((ch['S'], ch['M'], ch['b']))
    if r0.phase.startswith('B') and r0.M_star > 0:
        S = min((ch['S'] for ch in r0.channels), key=lambda s: abs(s - r0.b_star))
        if not any(abs(s - S) < 1e-12 and abs(m - r0.M_star) < 1e-9 for s, m, _ in starts):
            starts.append((S, r0.M_star, r0.b_star))
    for S, M0, b0 in starts:
        M, b, o = _channel(ctx, S, M0, b0)
        if M > 0:
            cand.append(dict(phase='B', M=M, b=b, omega=o))
    # a channel that slid to b ≈ 0 is the b = 0 solution; a channel beats the (exact) b = 0 minimum only by a margin
    best = cand[0]
    for c in sorted(cand[1:], key=lambda c: c['omega']):
        if c['b'] > B_MIN and c['omega'] < best['omega'] - max(OMEGA_NOISE, 1e-10 * abs(best['omega'])):
            best = c
            break
    if best['omega'] >= -OMEGA_NOISE:
        best = dict(phase='T', M=0.0, b=float('nan'), omega=0.0)
    if best['phase'] == 'B':
        n_near = min(r0.channels, key=lambda q: abs(q['S'] - best['b']))['n']
        best = dict(best, phase=f'B{n_near}')
    if best['phase'] == 'T':
        best = dict(best, b=float('nan'))
    lnM = math.log(best['M']) if best['M'] > 0 else -math.inf
    return FullResult(mu=mu, L=L, g=g, phase=best['phase'], M_star=best['M'], ln_M_star=lnM,
                      b_star=best['b'], omega_star=best['omega'], asymptotic=False,
                      channels=r0.channels, b0=dict(M=Mh, omega=Oh, T0_phase=r0.phase, T0_M=r0.M_star))

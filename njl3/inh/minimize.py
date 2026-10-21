"""
Global minimum of Omega(M, b) for a twist φ (tabulated values) and any g — the algorithm of
njl3.full.minimize (channels at the thresholds S_j = sqrt(mu² - k_j²) of the levels |k_j|, log
asymptotics at small M, temperature cut-off T_c,j = (e^γ/π)(S_j/mu) M*_j, noise threshold), with the
b = 0 candidate from njl3.hom. At φ = 0 it delegates to njl3.full.

Phases: 'T' (M* = 0), 'V' / 'I<j>' (b = 0, njl3.hom labels), 'B<j>' (b* ≈ S_j, level j).
"""

import math

import numpy as np
from scipy.optimize import minimize

from njl3.full import minimize as FM
from njl3.full.minimize import (FullResult, M_ASYMP, M_ASYMP_VALID, M_PROBE, OMEGA_NOISE, PRUNE_FACTOR,
                                SKIP_GRID_ASYMP, TC_RATIO, channel_m_min)
from njl3.hom import find_minimum as hom_find_minimum
from njl3.hom import phase_label as hom_phase_label
from njl3.inh import formulas as F

_LNM_LO, _LNM_HI = math.log(M_ASYMP), math.log(6.0)
_LNM = np.linspace(math.log(M_ASYMP), math.log(4.5), 40)
_T = np.concatenate([np.linspace(-4.0, -1.0, 13), np.linspace(-0.9, 0.9, 19), np.linspace(1.0, 3.0, 5)])


def thresholds(mu, L, phi):
    """[(j, S_j, C_j)] for the levels |k_j| < mu (j — level index)."""
    return [(j, math.sqrt(mu * mu - k * k), float(C)) for j, (k, C) in enumerate(F.levels(mu, L, phi))]


def channel_asymptotics(S, C, mu, L, g, phi):
    c = C * S / (2 * math.pi * L * mu)
    A = F.omega_rel(M_PROBE, S, mu, L, g, phi) / M_PROBE**2 + c * math.log(1.0 / M_PROBE)
    lnM = -A / c - 0.5
    # lnM > 700: S -> 0 (level at the Fermi surface), c -> 0 — the asymptotics does not apply (M_as = ∞,
    # the channel goes to the numerical search; as Math.exp in the JS port)
    M_as = 0.0 if lnM < -700 else (math.inf if lnM > 700 else math.exp(lnM))
    return dict(c=c, A=A, lnM=lnM, M=M_as, omega=-0.5 * c * M_as * M_as)


def channel_minimum(S, C, mu, L, g, phi, T=0.0, asy=None):
    if asy is None:
        asy = channel_asymptotics(S, C, mu, L, g, phi)
    m_min = channel_m_min(S, mu, T)
    lnm_lo = max(_LNM_LO, math.log(m_min)) if m_min > 0 else _LNM_LO
    if lnm_lo >= math.log(4.5):
        return dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=False, c=asy['c'], A=asy['A'])
    lnm_grid = _LNM if lnm_lo == _LNM_LO else np.linspace(
        lnm_lo, math.log(4.5), max(8, int(round(40 * (math.log(4.5) - lnm_lo) / (math.log(4.5) - _LNM_LO)))))
    if SKIP_GRID_ASYMP and asy['M'] < M_ASYMP_VALID and asy['lnM'] > lnm_lo:
        lg, tg = np.array([[asy['lnM']]]), np.array([[0.0]])
    else:
        lg, tg = np.meshgrid(lnm_grid, _T)
    Mg = np.exp(lg)
    bg = S + tg * Mg
    ok = bg >= 0
    Om = np.full(Mg.shape, np.inf)
    Om[ok] = F.omega_rel_vec(Mg[ok], bg[ok], mu, L, g, phi)
    i, j = np.unravel_index(np.argmin(Om), Om.shape)
    best = dict(M=float(Mg[i, j]), lnM=float(lg[i, j]), b=float(bg[i, j]), omega=float(Om[i, j]), asym=False)

    def f(x):
        lm = min(max(x[0], lnm_lo), _LNM_HI)
        M = math.exp(lm)
        b = S + x[1] * M
        return F.omega_rel(M, b, mu, L, g, phi) if b >= 0 else 1e3
    x0 = np.array([lg[i, j], tg[i, j]])
    simplex = np.array([x0, x0 + [0.3, 0.0], x0 + [0.0, 0.3]])
    r = minimize(f, x0, method='Nelder-Mead',
                 options=dict(initial_simplex=simplex, xatol=1e-6 if T > 0 else 1e-9,
                              fatol=(1e-10 if T > 0 else 1e-13) * abs(best['omega']) + 1e-300,
                              maxiter=600, maxfev=1200))
    if r.fun < best['omega']:
        lm = min(max(float(r.x[0]), lnm_lo), _LNM_HI)
        M = math.exp(lm)
        best = dict(M=M, lnM=lm, b=S + r.x[1] * M, omega=float(r.fun), asym=False)
    if best['omega'] > -OMEGA_NOISE or best['M'] < m_min * (1 + 1e-9):
        best = dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=False)
    if asy['M'] < M_ASYMP_VALID and asy['M'] > m_min and asy['omega'] < best['omega']:
        best = dict(M=asy['M'], lnM=asy['lnM'], b=S, omega=asy['omega'], asym=True)
    best.update(c=asy['c'], A=asy['A'])
    return best


def find_minimum(mu, L, g, phi=0.0, T=0.0):
    """Global minimum over (M, b) at (mu, L); T > 0 — survival threshold of the gaps."""
    phi = F.check_phi(phi)
    if phi == 0.0:
        return FM.find_minimum(mu, L, g, T=T)
    Mb, Ob, _ = hom_find_minimum(mu, L, g, phi)
    b0 = dict(M=Mb, omega=Ob, phase=hom_phase_label(Mb, mu, L, phi))
    if T > 0 and 0 < Mb <= T / TC_RATIO:
        Mb, Ob = 0.0, 0.0
    cand = [dict(phase=hom_phase_label(Mb, mu, L, phi), M=Mb, lnM=math.log(Mb) if Mb > 0 else -math.inf, b=0.0,
                 omega=Ob, asym=False)]
    thr = thresholds(mu, L, phi)
    asys = [channel_asymptotics(S, C, mu, L, g, phi) for _, S, C in thr]
    order = sorted(range(len(thr)), key=lambda k: asys[k]['omega'] if asys[k]['M'] < M_ASYMP_VALID else -math.inf)
    channels = [None] * len(thr)
    best_om = min(Ob, 0.0)
    for k in order:
        (n, S, C), asy = thr[k], asys[k]
        small = asy['M'] < M_ASYMP_VALID
        if T > 0 and small and asy['M'] <= channel_m_min(S, mu, T):
            ch = dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=True, c=asy['c'], A=asy['A'], pruned=True)
        elif T > 0 and small and asy['omega'] > PRUNE_FACTOR * best_om:
            ch = dict(M=asy['M'], lnM=asy['lnM'], b=S, omega=asy['omega'], asym=True, c=asy['c'], A=asy['A'],
                      pruned=True)
        else:
            ch = channel_minimum(S, C, mu, L, g, phi, T=T, asy=asy)
        ch.update(n=n, S=S)
        channels[k] = ch
        if ch['M'] > 0:
            best_om = min(best_om, ch['omega'])
            cand.append(dict(phase=f'B{n}', M=ch['M'], lnM=ch['lnM'], b=ch['b'], omega=ch['omega'], asym=ch['asym']))
    cand = [c for c in cand if c['phase'][0] != 'B' or c['b'] > 1e-8]
    best = cand[0]
    for c in sorted(cand[1:], key=lambda c: c['omega']):
        margin = 0.0 if c['asym'] else OMEGA_NOISE
        if c['omega'] < best['omega'] - margin:
            best = c
            break
    if best['phase'][0] == 'B':
        n_near, _, _ = min(thr, key=lambda q: abs(q[1] - best['b']))
        best = dict(best, phase=f'B{n_near}')
    if best['phase'] == 'T':
        best = dict(best, b=float('nan'))
    return FullResult(mu=mu, L=L, g=g, phase=best['phase'], M_star=best['M'], ln_M_star=best['lnM'], b_star=best['b'],
                      omega_star=best['omega'], asymptotic=best['asym'], channels=channels, b0=b0)

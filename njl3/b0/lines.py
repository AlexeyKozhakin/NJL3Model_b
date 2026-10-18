"""
Phase-transition lines, special points and maps of the phase diagram at b = 0.

Analytic lines:
    mu_s(L)  — stability edge of M = 0: G0(mu_s, L) = 0 (G0 is strictly increasing in mu);
               the transition across it is second order if G2 > 0;
    mu_VT(L) — vacuum–trivial transition: Omega_muL(0; mu, L) = W(L);
    peaks    — mu_s(L) ∩ (mu = 2 pi n / L), explicitly:
               L_n = (2|g|/pi) [ln(4 pi n) + 2 sum_{m=1}^{n-1} arccosh(n/m)],  mu_n = 2 pi n / L_n.

Tracing: at fixed L, a scan in mu with the certified minimiser; every phase change is
refined by bisection, and several changes within one step are found one after another.
Order of an I–T transition: second if the change coincides with mu_s(L) and G2 > 0.

The derivation is given in the accompanying paper.
"""

import math
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

from njl3.b0 import formulas as F
from njl3.b0.minimize import M_vacuum, find_minimum, phase_at, phase_label


# ==============================================================================
# Analytic lines and points
# ==============================================================================

def mu_stability(L, g, mu_max=50.0):
    """mu_s(L): the unique root of G0(mu, L) = 0; None if there is no root below mu_max."""
    f = lambda mu: F.gap_G0(mu, L, g)
    lo, hi = 1e-12, 1.0
    while f(hi) < 0.0:
        lo, hi = hi, 2.0 * hi
        if hi > mu_max:
            return None
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)


def mu_VT(L, g):
    """
    Solution of Omega(M_vac) = Omega(0). Omega(M_vac) does not depend on mu, so the condition
    is Omega_muL(0; mu, L) = W(L), whose left-hand side is strictly decreasing in mu.
    It is a phase boundary only if no I phase is deeper (checked by the tracing).
    """
    Mv = M_vacuum(L, g)
    W = Mv**2 / (2 * g) + F.dU(Mv) + F.omega_L(Mv, L) - F.omega_L(0.0, L)
    f = lambda mu: F.omega_muL(0.0, mu, L) - W
    hi = 1.0
    while f(hi) > 0.0:
        hi *= 2.0
    return brentq(f, 1e-12, hi, xtol=1e-15, rtol=1e-15)


def peak(n, g):
    """
    Peak n (explicit): (L_n, mu_n). As n -> ∞:
    mu_n = (pi/|g|) [1 - (2√2 zeta(-1/2)/pi) n^{-3/2} + O(n^{-5/2})].
    """
    s = math.log(4.0 * math.pi * n) + 2.0 * sum(math.acosh(n / m) for m in range(1, n))
    L = -2.0 * g / math.pi * s
    return L, 2.0 * math.pi * n / L


def peaks(g, L_max):
    """All peaks with L_n ≤ L_max: list of (n, L_n, mu_n)."""
    out, n = [], 1
    while True:
        L, mu = peak(n, g)
        if L > L_max:
            return out
        out.append((n, L, mu))
        n += 1


def tricritical_points(g, L_lo, L_hi, n_scan=4000):
    """
    Points on mu = mu_s(L) where G2 crosses zero continuously (the jumps G2 -> -inf at mode
    thresholds are peaks and are discarded). Whether such points lie on a phase boundary is
    checked by the tracing (for g = -1 and L ∈ [0.1, 10] they do not).
    """
    Ls = np.linspace(L_lo, L_hi, n_scan)
    mus = [mu_stability(L, g) for L in Ls]
    vals = np.array([F.G2(mu, L) if mu is not None else np.nan for mu, L in zip(mus, Ls)])
    nm = [len(F.modes(mu, L)) if mu is not None else -1 for mu, L in zip(mus, Ls)]
    h = lambda L: F.G2(mu_stability(L, g), L)
    pts = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        if nm[i] != nm[i + 1] or not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        Lt = brentq(h, Ls[i], Ls[i + 1], xtol=1e-14)
        pts.append((Lt, mu_stability(Lt, g)))
    return pts


# ==============================================================================
# Tracing at fixed L
# ==============================================================================

def _bisect_switch(a, la, b, L, g, tol):
    while b - a > tol:
        c = 0.5 * (a + b)
        if phase_at(c, L, g)[0] == la:
            a = c
        else:
            b = c
    return 0.5 * (a + b)


def _order(L, g, mu_c, frm, to):
    """Second order only for I–T on the line mu_s(L) with G2 > 0; everything else is first order."""
    if 'T' in (frm, to) and (frm.startswith('I') or to.startswith('I')):
        mu_s = mu_stability(L, g)
        if mu_s is not None and abs(mu_c - mu_s) < 1e-7 and F.G2(mu_s, L) > 0.0:
            return 2
    return 1


def trace_L(L, g, mu_lo=0.05, mu_hi=7.0, n_mu=400, tol=1e-12, eps=1e-9):
    """
    All phase changes in mu at fixed L: a list of dicts
    {L, mu, frm, to, order, M_left, M_right}. Back-and-forth changes within one scan step
    (A→B→A) are not seen — a limitation of the step (mu_hi - mu_lo)/(n_mu - 1).
    """
    mus = np.linspace(mu_lo, mu_hi, n_mu)
    labels = [phase_at(mu, L, g)[0] for mu in mus]
    out = []
    for k in range(n_mu - 1):
        a, la, b, lb = mus[k], labels[k], mus[k + 1], labels[k + 1]
        while la != lb:
            mu_c = _bisect_switch(a, la, b, L, g, tol)
            to, Mr = phase_at(mu_c + eps, L, g)
            _, Ml = phase_at(mu_c - eps, L, g)
            out.append(dict(L=float(L), mu=float(mu_c), frm=la, to=to,
                            order=_order(L, g, mu_c, la, to), M_left=Ml, M_right=Mr))
            a, la = mu_c + eps, to
    return out


def _pattern(L, g, mu_lo, mu_hi, n_mu):
    return tuple((r['frm'], r['to']) for r in trace_L(L, g, mu_lo, mu_hi, n_mu))


def _refine_pattern_change(La, Lb, pa, g, mu_lo, mu_hi, n_mu, tol=1e-8):
    while Lb - La > tol:
        c = 0.5 * (La + Lb)
        if _pattern(c, g, mu_lo, mu_hi, n_mu) == pa:
            La = c
        else:
            Lb = c
    return 0.5 * (La + Lb)


def trace_lines(L_grid, g, mu_lo=0.05, mu_hi=7.0, n_mu=400, jobs=None, special=True):
    """
    Transition lines on an L grid, and special points.

    Returns a dict (JSON-serialisable):
        g, mu_range, L_range,
        lines:   {"V-T": [...], "I0-T": [...], ...}  points {L, mu, order, M_left, M_right},
        special: [{kind: "peak" | "critical_end" | "tricritical", L, mu, ...}].
    critical_end — the point where sub-phase I_n is born (a second-order line ends on a
    first-order line); located by bisection in L to 1e-8.
    """
    L_grid = np.asarray(L_grid, dtype=float)
    tr = partial(trace_L, g=g, mu_lo=mu_lo, mu_hi=mu_hi, n_mu=n_mu)
    with Pool(jobs) as pool:
        traces = pool.map(tr, L_grid)

    lines = {}
    for t in traces:
        for r in t:
            lines.setdefault(f"{r['frm']}-{r['to']}", []).append(
                {k: r[k] for k in ('L', 'mu', 'order', 'M_left', 'M_right')})
    out = dict(g=g, mu_range=[mu_lo, mu_hi], L_range=[float(L_grid[0]), float(L_grid[-1])],
               lines=lines, special=[])
    if not special:
        return out

    sp = []
    for n, Ln, mun in peaks(g, L_grid[-1]):
        if Ln >= L_grid[0]:
            sp.append(dict(kind='peak', n=n, L=Ln, mu=mun))
    for Lt, mut in tricritical_points(g, L_grid[0], L_grid[-1]):
        on = any('T' in (r['frm'], r['to']) and abs(r['mu'] - mut) < 1e-6 for r in tr(Lt))
        sp.append(dict(kind='tricritical', L=Lt, mu=mut, on_phase_boundary=bool(on)))

    pats = [tuple((r['frm'], r['to']) for r in t) for t in traces]
    step = (mu_hi - mu_lo) / (n_mu - 1)
    jobs_ce = []
    for i in range(len(L_grid) - 1):
        if pats[i] and len(pats[i + 1]) > len(pats[i]):
            # scan only in a window around the transitions (same step in mu)
            mus = [r['mu'] for r in traces[i] + traces[i + 1]]
            lo, hi = max(mu_lo, min(mus) - 0.2), min(mu_hi, max(mus) + 0.2)
            n_w = max(3, int(round((hi - lo) / step)) + 1)
            win = (lo, hi, n_w)
            pa = _pattern(L_grid[i], g, *win)
            jobs_ce.append((L_grid[i], L_grid[i + 1], pa, g, *win))
    with Pool(jobs) as pool:
        Lcs = pool.starmap(_refine_pattern_change, jobs_ce)
    for Lc in Lcs:
        muT = [r['mu'] for r in tr(Lc + 1e-6) if r['to'] == 'T']
        sp.append(dict(kind='critical_end', L=Lc, mu=muT[0] if muT else None))

    out['special'] = sorted(sp, key=lambda x: x['L'])
    return out


# ==============================================================================
# Map M*(mu, L)
# ==============================================================================

def _grid_row(L, mu_grid, g):
    row = []
    for mu in mu_grid:
        M, Om, minima = find_minimum(mu, L, g)
        row.append((M, Om, phase_label(M, mu, L), len(minima)))
    return row


def compute_grid(mu_grid, L_grid, g, jobs=None):
    """
    Map of the global minimum on a grid. Returns a dict of arrays of shape
    (len(L_grid), len(mu_grid)): M_star, omega_star, phase (strings), n_minima.
    """
    mu_grid = np.asarray(mu_grid, dtype=float)
    L_grid = np.asarray(L_grid, dtype=float)
    with Pool(jobs) as pool:
        rows = pool.map(partial(_grid_row, mu_grid=mu_grid, g=g), L_grid)
    return dict(mu=mu_grid, L=L_grid,
                M_star=np.array([[c[0] for c in r] for r in rows]),
                omega_star=np.array([[c[1] for c in r] for r in rows]),
                phase=np.array([[c[2] for c in r] for r in rows]),
                n_minima=np.array([[c[3] for c in r] for r in rows]))

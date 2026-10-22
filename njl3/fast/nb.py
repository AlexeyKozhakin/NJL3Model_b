"""
Compiled (numba) core of njl3.fast: Omega(M, b) with the same quadratures and s_φ tables as njl3.inh
(results agree to rounding), and the inner loops of the search (channel A, scouting + Newton, dense scan),
so that no Python call is made per Omega evaluation.

    ctx = context(mu, L, g, phi)        # tables, mode levels, Omega(0, ·) terms
    omega(M, b, ctx)                    # Omega(M, b) - Omega(0, 0)
"""

import math

import numpy as np
from numba import njit

from njl3.full import formulas as FF
from njl3.inh import formulas as F

PI = math.pi

# quadrature templates of njl3.full (two-sided graded, graded, uniform pieces)
TSX, TSW = np.ascontiguousarray(FF._TS_X), np.ascontiguousarray(FF._TS_W)
GTX, GTW = np.ascontiguousarray(FF._GT_X), np.ascontiguousarray(FF._GT_W)
CHX, CHW = np.ascontiguousarray(FF._CH_X), np.ascontiguousarray(FF._CH_W)

_TABLES = {}


def table(phi):
    """(edges, coef, mid, half, phi0, x_min, s0) of s_φ as arrays."""
    if phi not in _TABLES:
        if phi == 0.0:
            pieces = [(a, b, c) for a, b, c in FF._PIECES]
            x_min, s0 = 0.0, math.pi**2 / 6
        else:
            T = F._table(phi)
            import json, os
            with open(os.path.join(os.path.dirname(F.__file__), 'data', f's_table_phi{phi:g}.json')) as f:
                t = json.load(f)
            pieces = [(p['a'], p['b'], np.array(p['coef'])) for p in t['pieces']]
            x_min, s0 = t['x_min'], t['s0']
        nc = max(len(c) for _, _, c in pieces)
        coef = np.zeros((len(pieces), nc))
        for i, (_, _, c) in enumerate(pieces):
            coef[i, :len(c)] = c
        edges = np.array([a for a, _, _ in pieces] + [pieces[-1][1]])
        mid = np.array([0.5 * (a + b) for a, b, _ in pieces])
        half = np.array([0.5 * (b - a) for a, b, _ in pieces])
        _TABLES[phi] = (edges, coef, mid, half, phi == 0.0, x_min, s0)
    return _TABLES[phi]


@njit(cache=True, fastmath=False)
def s_nb(x, edges, coef, mid, half, phi0, x_min, s0):
    n = edges.shape[0] - 1
    if (not phi0) and x < x_min:
        return s0
    if x >= edges[n] or x < edges[0]:
        return 0.0
    lo, hi = 0, n - 1
    while lo < hi:
        m = (lo + hi) // 2
        if x < edges[m + 1]:
            hi = m
        else:
            lo = m + 1
    k = lo
    t2 = 2.0 * (x - mid[k]) / half[k]
    b1 = 0.0
    b2 = 0.0
    for j in range(coef.shape[1] - 1, 0, -1):
        b1, b2 = coef[k, j] + t2 * b1 - b2, b1
    v = coef[k, 0] + 0.5 * t2 * b1 - b2
    if phi0 and k == 0 and x > 0.0:
        v -= x * x / 4.0 * math.log(x)
    return v


@njit(cache=True)
def omega_L_nb(M, b, L, edges, coef, mid, half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW):
    p_b = math.sqrt(max(b * b - M * M, 0.0))
    E_hi = b + 64.0 / L
    p_hi = math.sqrt(max(E_hi * E_hi - M * M, 0.0))
    c = min(p_b + 4.0 / L, p_hi)
    tot = 0.0
    for i in range(TSX.shape[0]):
        p = p_b * TSX[i]
        E = math.sqrt(M * M + p * p)
        tot += p_b * TSW[i] * (s_nb(L * (E + b), edges, coef, mid, half, phi0, x_min, s0)
                                + s_nb(L * abs(E - b), edges, coef, mid, half, phi0, x_min, s0))
    for i in range(GTX.shape[0]):
        p = p_b + (c - p_b) * GTX[i]
        E = math.sqrt(M * M + p * p)
        tot += (c - p_b) * GTW[i] * (s_nb(L * (E + b), edges, coef, mid, half, phi0, x_min, s0)
                                     + s_nb(L * abs(E - b), edges, coef, mid, half, phi0, x_min, s0))
    for i in range(CHX.shape[0]):
        p = c + (p_hi - c) * CHX[i]
        E = math.sqrt(M * M + p * p)
        tot += (p_hi - c) * CHW[i] * (s_nb(L * (E + b), edges, coef, mid, half, phi0, x_min, s0)
                                      + s_nb(L * abs(E - b), edges, coef, mid, half, phi0, x_min, s0))
    return 2.0 / (PI * PI * L * L) * tot


@njit(cache=True)
def _pE(E, M):
    return math.sqrt(max(E * E - M * M, 0.0))


@njit(cache=True)
def _J(p, M):
    E = math.sqrt(M * M + p * p)
    ash = math.asinh(p / M) if M > 0 else 0.0
    return 0.5 * (p * E + M * M * ash)


@njit(cache=True)
def mode0_nb(mu, b, M):
    tot = 0.0
    if mu - b > M:
        pr = _pE(mu - b, M)
        tot += (mu - b) * pr - _J(pr, M)
    lo, hi = max(M, b - mu), b + mu
    e2 = min(b, hi)
    if e2 > lo:
        p1, p2 = _pE(lo, M), _pE(e2, M)
        tot += (mu - b) * (p2 - p1) + _J(p2, M) - _J(p1, M)
    e3 = max(b, lo)
    if hi > e3:
        p3, p4 = _pE(e3, M), _pE(hi, M)
        tot += (mu + b) * (p4 - p3) - (_J(p4, M) - _J(p3, M))
    return tot


@njit(cache=True)
def _seg(a, c, M, bs, k, mu, TSX, TSW):
    h = c - a
    if h <= 0.0:
        return 0.0
    s = 0.0
    for i in range(TSX.shape[0]):
        p = a + h * TSX[i]
        A = math.sqrt(M * M + p * p) + bs
        s += TSW[i] * (mu - math.sqrt(A * A + k * k))
    return h * s


@njit(cache=True)
def modeK_nb(M, b, mu, k, TSX, TSW):
    S = math.sqrt(mu * mu - k * k)
    I = 0.0
    if S - b > M:
        I += _seg(0.0, _pE(S - b, M), M, b, k, mu, TSX, TSW)
    lo, hi = max(M, b - S), b + S
    pl = _pE(lo, M)
    pr = _pE(max(hi, lo), M)
    mid = min(max(_pE(b, M), pl), pr)
    return I + _seg(pl, mid, M, -b, k, mu, TSX, TSW) + _seg(mid, pr, M, -b, k, mu, TSX, TSW)


@njit(cache=True)
def omega_muL_nb(M, b, mu, L, levk, levC, TSX, TSW):
    tot = 0.0
    for i in range(levk.shape[0]):
        if levk[i] == 0.0:
            tot += levC[i] * mode0_nb(mu, b, M)
        else:
            tot += levC[i] * modeK_nb(M, b, mu, levk[i], TSX, TSW)
    return -tot / (PI * L)


@njit(cache=True)
def dU_nb(M, b):
    if M == 0.0:
        return 0.0
    if b >= M:
        return M * M * b / 4.0
    return ((2 * M * M + b * b) * math.sqrt(M * M - b * b) / 2.0 + 1.5 * M * M * b * math.asin(b / M)) / (3 * PI)


@njit(cache=True)
def omega_nb(M, b, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
             TSX, TSW, GTX, GTW, CHX, CHW):
    """Omega(M, b) - Omega(0, 0); z0 = Omega_L(0, 0) + Omega_muL(0, 0)."""
    if M == 0.0:
        return 0.0
    if b < 0.0:
        return 1e3
    return (M * M / (2 * g) + dU_nb(M, b)
            + omega_L_nb(M, b, L, edges, coef, mid, half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW)
            + omega_muL_nb(M, b, mu, L, levk, levC, TSX, TSW) - z0)


class Ctx:
    """Everything omega_nb needs at a point (mu, L, g, phi)."""

    def __init__(self, mu, L, g, phi):
        self.mu, self.L, self.g, self.phi = mu, L, g, phi
        lv = F.levels(mu, L, phi)
        self.levk = np.array([k for k, _ in lv], dtype=float)
        self.levC = np.array([float(c) for _, c in lv], dtype=float)
        self.tab = table(phi)
        e, c, m, h, p0, xm, s0 = self.tab
        self.z0 = (omega_L_nb(0.0, 0.0, L, e, c, m, h, p0, xm, s0, TSX, TSW, GTX, GTW, CHX, CHW)
                   + omega_muL_nb(0.0, 0.0, mu, L, self.levk, self.levC, TSX, TSW))
        self.args = (mu, L, g, self.z0, self.levk, self.levC, e, c, m, h, p0, xm, s0, TSX, TSW, GTX, GTW, CHX, CHW)

    def omega(self, M, b):
        return omega_nb(M, b, *self.args)


# ==============================================================================
# Inner loops of the search
# ==============================================================================

@njit(cache=True)
def oxt(x, t, S, lnm_max, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
        TSX, TSW, GTX, GTW, CHX, CHW):
    M = math.exp(min(x, lnm_max))
    return omega_nb(M, S + t * M, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
                    TSX, TSW, GTX, GTW, CHX, CHW)


@njit(cache=True)
def refine_nb(x0, t0, scout, S, lnm_lo, lnm_max, SX, ST, steps, mu, L, g, z0, levk, levC, edges, coef, mid,
              half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW):
    """Scouting grid (if scout) or start (x0, t0), then damped Newton with a 3×3 stencil. Returns (x, t, Omega)."""
    H = 1e-3
    if scout:
        fx = 1e300
        x, t = x0, 0.0
        for i in range(SX.shape[0]):
            xx = min(max(x0 + SX[i], lnm_lo), lnm_max)
            for j in range(ST.shape[0]):
                f = oxt(xx, ST[j], S, lnm_max, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
                        TSX, TSW, GTX, GTW, CHX, CHW)
                if f < fx:
                    fx, x, t = f, xx, ST[j]
    else:
        x, t = x0, t0
        fx = oxt(x, t, S, lnm_max, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
                 TSX, TSW, GTX, GTW, CHX, CHW)
    f = np.empty((3, 3))
    for _ in range(steps):
        for a in range(3):
            for c in range(3):
                if a == 1 and c == 1:
                    f[a, c] = fx
                else:
                    f[a, c] = oxt(x + (a - 1) * H, t + (c - 1) * H, S, lnm_max, mu, L, g, z0, levk, levC, edges,
                                  coef, mid, half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW)
        gx = (f[2, 1] - f[0, 1]) / (2 * H)
        gt = (f[1, 2] - f[1, 0]) / (2 * H)
        hxx = (f[2, 1] - 2 * f[1, 1] + f[0, 1]) / (H * H)
        htt = (f[1, 2] - 2 * f[1, 1] + f[1, 0]) / (H * H)
        hxt = (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / (4 * H * H)
        det = hxx * htt - hxt * hxt
        if hxx > 0 and det > 0:
            dx = -(htt * gx - hxt * gt) / det
            dt = -(hxx * gt - hxt * gx) / det
        else:
            sc = 0.1 / max(abs(gx), abs(gt), 1e-300)
            dx, dt = -gx * sc, -gt * sc
        lam = 1.0
        ok = False
        xn, tn, fn = x, t, fx
        for _k in range(12):
            xn = min(max(x + lam * dx, lnm_lo), lnm_max)
            tn = t + lam * dt
            fn = oxt(xn, tn, S, lnm_max, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
                     TSX, TSW, GTX, GTW, CHX, CHW)
            if fn <= fx:
                ok = True
                break
            lam *= 0.5
        if not ok:
            break
        step = max(abs(xn - x), abs(tn - t))
        x, t, fx = xn, tn, fn
        if step < 1e-7:
            break
    return x, t, fx


@njit(cache=True)
def grid_nb(Mg, bg, mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW):
    out = np.empty((Mg.shape[0], bg.shape[0]))
    for i in range(Mg.shape[0]):
        for k in range(bg.shape[0]):
            out[i, k] = omega_nb(Mg[i], bg[k], mu, L, g, z0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
                                 TSX, TSW, GTX, GTW, CHX, CHW)
    return out

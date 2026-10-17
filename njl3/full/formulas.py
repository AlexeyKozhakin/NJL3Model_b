"""
Потенциал Omega(M, b) модели NJL3 на R² × S¹ при phi = 0, b ≥ 0.

Обозначения: E = sqrt(M² + p²) (p ≡ p1), A± = E ± b, k_n = 2πn/L,
S_n = sqrt(mu² − k_n²), C_0 = 1, C_{n≥1} = 2.

    Omega_ren(M, b) − Omega_ren(0, 0) = M²/(2g) + Σ_X [X(M, b) − X(0, b)],   X ∈ {ΔU, Ω_L, Ω_μL}

(перенормировка X_phys = X(M,b) − X(0,b) + X(0,0); поэтому Omega(0, b) = Omega(0, 0) при любом b).

    ΔU(M, b)  = M² b / 4                                                   (b ≥ M)
              = (1/3π) [ (2M² + b²) sqrt(M² − b²)/2 + (3/2) M² b arcsin(b/M) ]   (b ≤ M)
    Ω_L(M, b) = (2/(π² L²)) ∫_0^∞ dp Σ± s(L |A±|),
                s(x) = −∫_0^∞ ln(1 − e^{−sqrt(t² + x²)}) dt   (таблица Чебышёва)
    Ω_μL(M, b) = −(1/(π L)) Σ_n C_n Σ± ∫_{F>0} (mu − sqrt(A±² + k_n²)) dp

Мода n = 0 в Ω_μL берётся в элементарных функциях, остальные — Гаусс–Лежандр.
"""

import json
import math
import os

import numpy as np

PI = math.pi
_HERE = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# s(x): кусочно-чебышёвская таблица
# ==============================================================================

with open(os.path.join(_HERE, 'data', 's_table.json')) as _f:
    _TAB = json.load(_f)
_PIECES = [(p['a'], p['b'], np.array(p['coef'])) for p in _TAB['pieces']]
_XMAX = _TAB['x_max']


def s_func(x):
    """s(x) для массива x ≥ 0 (абсолютная точность ~1e-15; s = 0 при x ≥ 64)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i, (a, b, c) in enumerate(_PIECES):
        m = (x >= a) & (x < b) if i else (x >= a) & (x < b)
        if not np.any(m):
            continue
        xm = x[m]
        t = (2.0 * xm - (a + b)) / (b - a)
        v = np.polynomial.chebyshev.chebval(t, c)
        if i == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = v - np.where(xm > 0, xm * xm / 4.0 * np.log(xm), 0.0)
        out[m] = v
    return out


# ==============================================================================
# Квадратуры
# ==============================================================================

def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


_GL16 = _gl(16)
_GL32 = _gl(32)


def _nodes(segments, rule):
    """Узлы и веса Гаусса–Лежандра на наборе отрезков [(a, b), ...]."""
    x0, w0 = rule
    xs, ws = [], []
    for a, b in segments:
        if b <= a:
            continue
        h = 0.5 * (b - a)
        xs.append(h * x0 + 0.5 * (a + b))
        ws.append(h * w0)
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ws)


def _graded(c, lo, hi, levels=22):
    """Отрезки [lo, hi], геометрически сгущённые к точке c ∈ [lo, hi]."""
    segs = []
    for side_lo, side_hi, toward_hi in ((lo, c, True), (c, hi, False)):
        L = side_hi - side_lo
        if L <= 0:
            continue
        # точки c ± L·2^{-j}
        pts = [L * 2.0**(-j) for j in range(levels)] + [0.0]
        for d1, d2 in zip(pts[:-1], pts[1:]):
            if toward_hi:
                segs.append((side_hi - d1, side_hi - d2))
            else:
                segs.append((side_lo + d2, side_lo + d1))
    return segs


# ==============================================================================
# Вклады
# ==============================================================================

def dU(M, b):
    """Вакуумный вклад ΔU(M, b) (замкнутая форма); ΔU(0, b) = 0."""
    if M == 0.0:
        return 0.0
    if b >= M:
        return M * M * b / 4.0
    return ((2 * M * M + b * b) * math.sqrt(M * M - b * b) / 2.0 + 1.5 * M * M * b * math.asin(b / M)) / (3 * PI)


def omega_L(M, b, L):
    """Компактификационный вклад Ω_L(M, b)."""
    p_b = math.sqrt(b * b - M * M) if b > M else 0.0          # излом |E − b|
    E_hi = b + 64.0 / L                                       # дальше s(L|A|) = 0
    p_hi = math.sqrt(max(E_hi * E_hi - M * M, 0.0))
    segs = _graded(p_b, 0.0, min(p_b + 4.0 / L, p_hi))
    a = min(p_b + 4.0 / L, p_hi)
    step = 4.0 / L
    while a < p_hi:
        segs.append((a, min(a + step, p_hi)))
        a += step
    p, w = _nodes(segs, _GL16)
    E = np.sqrt(M * M + p * p)
    f = s_func(L * (E + b)) + s_func(L * np.abs(E - b))
    return 2.0 / (PI**2 * L * L) * float(np.dot(w, f))


def _J(p, M):
    """∫_0^p sqrt(M² + q²) dq."""
    if M == 0.0:
        return 0.5 * p * p
    E = math.sqrt(M * M + p * p)
    return 0.5 * (p * E + M * M * math.asinh(p / M))


def _p_of_E(E, M):
    return math.sqrt(max(E * E - M * M, 0.0))


def _mode0(mu, b, M):
    """n = 0: Σ± ∫_{F>0} (mu − |A±|) dp в элементарных функциях."""
    tot = 0.0
    # плюс: E ≤ mu − b
    if mu - b > M:
        pr = _p_of_E(mu - b, M)
        tot += (mu - b) * pr - _J(pr, M)
    # минус: |E − b| ≤ mu, E ≥ M
    lo, hi = max(M, b - mu), b + mu
    if hi > lo:
        e1, e2 = lo, min(b, hi)                     # E < b: mu − b + E
        if e2 > e1:
            p1, p2 = _p_of_E(e1, M), _p_of_E(e2, M)
            tot += (mu - b) * (p2 - p1) + _J(p2, M) - _J(p1, M)
        e3, e4 = max(b, lo), hi                     # E > b: mu + b − E
        if e4 > e3:
            p3, p4 = _p_of_E(e3, M), _p_of_E(e4, M)
            tot += (mu + b) * (p4 - p3) - (_J(p4, M) - _J(p3, M))
    return tot


def _mode_n(mu, b, M, k):
    """n ≥ 1: Σ± ∫_{F>0} (mu − sqrt(A±² + k²)) dp (Гаусс–Лежандр)."""
    S = math.sqrt(mu * mu - k * k)
    segs_p, segs_m = [], []
    if S - b > M:
        segs_p.append((0.0, _p_of_E(S - b, M)))
    lo, hi = max(M, b - S), b + S
    if hi > lo:
        pl, pr = _p_of_E(lo, M), _p_of_E(hi, M)
        pb = _p_of_E(b, M) if b > M else None
        if pb is not None and pl < pb < pr:
            segs_m += [(pl, pb), (pb, pr)]
        else:
            segs_m.append((pl, pr))
    tot = 0.0
    for segs, sgn in ((segs_p, 1.0), (segs_m, -1.0)):
        p, w = _nodes(segs, _GL32)
        if p.size:
            A = np.sqrt(M * M + p * p) + sgn * b
            tot += float(np.dot(w, mu - np.sqrt(A * A + k * k)))
    return tot


def omega_muL(M, b, mu, L):
    """Термический вклад Ω_μL(M, b)."""
    if mu <= 0.0:
        return 0.0
    n_max = int(math.floor(L * mu / (2 * PI)))
    tot = _mode0(mu, b, M)
    for n in range(1, n_max + 1):
        tot += 2.0 * _mode_n(mu, b, M, 2 * PI * n / L)
    return -tot / (PI * L)


def omega_rel(M, b, mu, L, g):
    """Omega_ren(M, b) − Omega_ren(0, 0)."""
    if M == 0.0:
        return 0.0
    return (M * M / (2.0 * g) + dU(M, b)
            + omega_L(M, b, L) - omega_L(0.0, b, L)
            + omega_muL(M, b, mu, L) - omega_muL(0.0, b, mu, L))


# ==============================================================================
# Векторные версии (много точек (M, b) за один вызов)
# ==============================================================================

def _graded_template(levels, rule):
    """Узлы/веса на [0, 1], сгущённые к 0: отрезки [2^{-(j+1)}, 2^{-j}] и [0, 2^{-levels}]."""
    segs = [(2.0**-(j + 1), 2.0**-j) for j in range(levels)] + [(0.0, 2.0**-levels)]
    return _nodes(segs, rule)


_GT_X, _GT_W = _graded_template(16, _GL16)          # расстояния до излома в долях стороны
_CH_X, _CH_W = _nodes([(i / 16, (i + 1) / 16) for i in range(16)], _GL16)


def omega_L_vec(M, b, L):
    """Ω_L для массивов M, b одной формы (L — скаляр). ~770 узлов на точку."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    M, b = np.broadcast_arrays(M, b)
    shape = M.shape
    M, b = M.ravel()[:, None], b.ravel()[:, None]
    p_b = np.sqrt(np.clip(b * b - M * M, 0.0, None))
    E_hi = b + 64.0 / L
    p_hi = np.sqrt(np.clip(E_hi * E_hi - M * M, 0.0, None))
    c = np.minimum(p_b + 4.0 / L, p_hi)
    # левая сторона [0, p_b] к p_b, правая [p_b, c] к p_b, хвост [c, p_hi] равными кусками
    pL = p_b - p_b * _GT_X;  wL = p_b * _GT_W
    pR = p_b + (c - p_b) * _GT_X;  wR = (c - p_b) * _GT_W
    pT = c + (p_hi - c) * _CH_X;  wT = (p_hi - c) * _CH_W
    p = np.concatenate([pL, pR, pT], axis=1)
    w = np.concatenate([wL, wR, wT], axis=1)
    E = np.sqrt(M * M + p * p)
    f = s_func(L * (E + b)) + s_func(L * np.abs(E - b))
    return (2.0 / (PI**2 * L * L) * np.sum(w * f, axis=1)).reshape(shape)


def _J_vec(p, M):
    """∫_0^p sqrt(M² + q²) dq для массивов."""
    E = np.sqrt(M * M + p * p)
    with np.errstate(divide='ignore', invalid='ignore'):
        ash = np.where(M > 0, np.arcsinh(p / np.where(M > 0, M, 1.0)), 0.0)
    return 0.5 * (p * E + M * M * ash)


def _pE(E, M):
    return np.sqrt(np.clip(E * E - M * M, 0.0, None))


def _mode0_vec(mu, b, M):
    tot = np.where(mu - b > M, (mu - b) * _pE(mu - b, M) - _J_vec(_pE(mu - b, M), M), 0.0)
    lo, hi = np.maximum(M, b - mu), b + mu
    e2 = np.minimum(b, hi)
    p1, p2 = _pE(lo, M), _pE(np.maximum(e2, lo), M)
    tot = tot + np.where(e2 > lo, (mu - b) * (p2 - p1) + _J_vec(p2, M) - _J_vec(p1, M), 0.0)
    e3 = np.maximum(b, lo)
    p3, p4 = _pE(e3, M), _pE(np.maximum(hi, e3), M)
    tot = tot + np.where(hi > e3, (mu + b) * (p4 - p3) - (_J_vec(p4, M) - _J_vec(p3, M)), 0.0)
    return tot


_G32X, _G32W = _GL32


def _seg_int(a, c, M, b_sgn, k, mu):
    """Σ по точкам ∫_a^c (mu − sqrt((E + b_sgn)² + k²)) dp, a, c — столбцы (N, 1)."""
    h = 0.5 * (c - a)
    p = h * _G32X + 0.5 * (a + c)
    A = np.sqrt(M * M + p * p) + b_sgn
    return np.sum(h * _G32W * (mu - np.sqrt(A * A + k * k)), axis=1)


def omega_muL_vec(M, b, mu, L):
    """Ω_μL для массивов M, b (mu, L — скаляры)."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    M, b = np.broadcast_arrays(M, b)
    shape = M.shape
    M, b = M.ravel(), b.ravel()
    if mu <= 0.0:
        return np.zeros(shape)
    tot = _mode0_vec(mu, b, M)
    Mc, bc = M[:, None], b[:, None]
    n_max = int(math.floor(L * mu / (2 * PI)))
    for n in range(1, n_max + 1):
        k = 2 * PI * n / L
        S = math.sqrt(mu * mu - k * k)
        # плюс: p ∈ [0, sqrt((S − b)² − M²)], если S − b > M
        pr = np.where(S - bc > Mc, _pE(S - bc, Mc), 0.0)
        I = _seg_int(np.zeros_like(pr), pr, Mc, bc, k, mu)
        # минус: E ∈ [max(M, b − S), b + S], разрез в E = b
        lo, hi = np.maximum(Mc, bc - S), bc + S
        pl, prm = _pE(lo, Mc), _pE(np.maximum(hi, lo), Mc)
        mid = np.clip(_pE(bc, Mc), pl, prm)
        I = I + _seg_int(pl, mid, Mc, -bc, k, mu) + _seg_int(mid, prm, Mc, -bc, k, mu)
        tot = tot + 2.0 * I
    return (-tot / (PI * L)).reshape(shape)


def omega_rel_vec(M, b, mu, L, g, OL0=None):
    """Omega(M, b) − Omega(0, 0) для массивов M, b. OL0 — готовый Ω_L(M, b) − Ω_L(0, b) (не зависит от mu)."""
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    M, b = np.broadcast_arrays(M, b)
    if OL0 is None:
        OL0 = omega_L_vec(M, b, L) - omega_L_vec(np.zeros_like(b), b, L)
    dUv = np.vectorize(dU)(M, b)
    out = M * M / (2 * g) + dUv + OL0 + omega_muL_vec(M, b, mu, L) - omega_muL_vec(np.zeros_like(b), b, mu, L)
    return np.where(M == 0, 0.0, out)

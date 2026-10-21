"""
Omega(M, b) of NJL3 on R² × S¹ (chiral spiral, b ≥ 0) for a twist φ of the boundary condition
(generalizes njl3.full, which is φ = 0; at φ = 0 everything is delegated to njl3.full).

Modes k_n = 2π(n + φ)/L; distinct levels |k| with multiplicity C (φ = 0: C_0 = 1, C_{n≥1} = 2;
0 < φ < 1/2: every mode separate, C = 1; φ = 1/2: pairs, C = 2).

    Omega(M, b) - Omega(0, 0) = M²/(2g) + Σ_X [X(M, b) - X(0, b)],   X ∈ {ΔU, Ω_L, Ω_μL},
    ΔU — as in njl3.full (independent of L, φ),
    Ω_L(M, b) = (2/(π² L²)) ∫_0^∞ dp Σ± s_φ(L |A±|),   A± = sqrt(M² + p²) ± b,
    s_φ(x) = -∫_0^∞ Re ln(1 - e^{-sqrt(t² + x²) + 2πiφ}) dt   (tables njl3/inh/data, make_tables.py),
    Ω_μL(M, b) = -(1/(π L)) Σ_levels C Σ± ∫_{F>0} (mu - sqrt(A±² + k²)) dp.
"""

import functools
import json
import math
import os

import numpy as np

from njl3.full import formulas as F

PI = math.pi
_HERE = os.path.dirname(os.path.abspath(__file__))
PHIS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def check_phi(phi):
    """φ must be one of the tabulated values (φ and 1 - φ are equivalent)."""
    p = phi % 1.0
    p = round(min(p, 1.0 - p), 12)
    if not any(abs(p - q) < 1e-12 for q in PHIS):
        raise ValueError(f'phi = {phi}: tables exist for {PHIS} (and 1 - phi)')
    return min(PHIS, key=lambda q: abs(q - p))


# ==============================================================================
# s_φ(x)
# ==============================================================================

@functools.lru_cache(maxsize=8)
def _table(phi):
    with open(os.path.join(_HERE, 'data', f's_table_phi{phi:g}.json')) as f:
        t = json.load(f)
    pieces = t['pieces']
    edges = np.array([p['a'] for p in pieces] + [pieces[-1]['b']])
    nc = max(len(p['coef']) for p in pieces)
    coef = np.zeros((len(pieces) + 1, nc))
    for i, p in enumerate(pieces):
        coef[i, :len(p['coef'])] = p['coef']
    mid = np.array([0.5 * (p['a'] + p['b']) for p in pieces] + [0.0])
    half = np.array([0.5 * (p['b'] - p['a']) for p in pieces] + [1.0])
    return dict(edges=edges, coefT=[np.ascontiguousarray(coef[:, j]) for j in range(nc)], mid=mid, half=half,
                n=len(pieces), s0=t['s0'], x_min=t['x_min'])


def s_func(x, phi):
    """s_φ(x) for an array x ≥ 0."""
    if phi == 0.0:
        return F.s_func(x)
    T = _table(phi)
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x = x.ravel()
    k = np.searchsorted(T['edges'], x, side='right') - 1
    k = np.where((k < 0) | (k >= T['n']), T['n'], k)              # outside: dummy row (0)
    t2 = 2.0 * (x - T['mid'][k]) / T['half'][k]
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    cT = T['coefT']
    for j in range(len(cT) - 1, 0, -1):
        b1, b2 = cT[j][k] + t2 * b1 - b2, b1
    v = cT[0][k] + 0.5 * t2 * b1 - b2
    v = np.where(x < T['x_min'], T['s0'], v)
    return v.reshape(shape)


# ==============================================================================
# Modes
# ==============================================================================

def levels(mu, L, phi):
    """[(|k|, C)] for the modes with |k_n| < mu, ascending in |k|."""
    if mu <= 0.0:
        return []
    out = {}
    nmax = int(math.floor(L * mu / (2.0 * PI))) + 1
    for n in range(-nmax - 1, nmax + 1):
        k = abs(2.0 * PI * (n + phi) / L)
        if k < mu:
            key = round(k, 12)
            out[key] = out.get(key, 0) + 1
    return sorted(out.items())


# ==============================================================================
# Contributions
# ==============================================================================

dU = F.dU


def omega_L_vec(M, b, L, phi):
    if phi == 0.0:
        return F.omega_L_vec(M, b, L)
    M = np.atleast_1d(np.asarray(M, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    M, b = np.broadcast_arrays(M, b)
    shape = M.shape
    M, b = M.ravel()[:, None], b.ravel()[:, None]
    p_b = np.sqrt(np.clip(b * b - M * M, 0.0, None))
    E_hi = b + 64.0 / L
    p_hi = np.sqrt(np.clip(E_hi * E_hi - M * M, 0.0, None))
    c = np.minimum(p_b + 4.0 / L, p_hi)
    pL = p_b * F._TS_X; wL = p_b * F._TS_W
    pR = p_b + (c - p_b) * F._GT_X; wR = (c - p_b) * F._GT_W
    pT = c + (p_hi - c) * F._CH_X; wT = (p_hi - c) * F._CH_W
    p = np.concatenate([pL, pR, pT], axis=1)
    w = np.concatenate([wL, wR, wT], axis=1)
    E = np.sqrt(M * M + p * p)
    f = s_func(L * (E + b), phi) + s_func(L * np.abs(E - b), phi)
    return (2.0 / (PI**2 * L * L) * np.sum(w * f, axis=1)).reshape(shape)


def _mode_k_vec(Mc, bc, k, mu):
    """Σ± ∫_{F>0} (mu - sqrt(A±² + k²)) dp for one mode with k > 0 (columns Mc, bc)."""
    S = math.sqrt(mu * mu - k * k)
    pr = np.where(S - bc > Mc, F._pE(S - bc, Mc), 0.0)
    I = F._seg_int(np.zeros_like(pr), pr, Mc, bc, k, mu)
    lo, hi = np.maximum(Mc, bc - S), bc + S
    pl, prm = F._pE(lo, Mc), F._pE(np.maximum(hi, lo), Mc)
    mid = np.clip(F._pE(bc, Mc), pl, prm)
    return I + F._seg_int(pl, mid, Mc, -bc, k, mu) + F._seg_int(mid, prm, Mc, -bc, k, mu)


def omega_muL_vec(M, b, mu, L, phi):
    if phi == 0.0:
        return F.omega_muL_vec(M, b, mu, L)
    M = np.atleast_1d(np.asarray(M, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    M, b = np.broadcast_arrays(M, b)
    shape = M.shape
    M, b = M.ravel(), b.ravel()
    tot = np.zeros(M.shape)
    Mc, bc = M[:, None], b[:, None]
    for k, C in levels(mu, L, phi):
        tot = tot + (C * F._mode0_vec(mu, b, M) if k == 0.0 else C * _mode_k_vec(Mc, bc, k, mu))
    return (-tot / (PI * L)).reshape(shape)


@functools.lru_cache(maxsize=4096)
def _omega_L0(L, phi):
    return float(omega_L_vec(0.0, 0.0, L, phi)[0])


@functools.lru_cache(maxsize=65536)
def _omega_muL0(mu, L, phi):
    return float(omega_muL_vec(0.0, 0.0, mu, L, phi)[0])


def omega_rel_vec(M, b, mu, L, g, phi):
    """Omega(M, b) - Omega(0, 0) for arrays M, b."""
    if phi == 0.0:
        return F.omega_rel_vec(M, b, mu, L, g)
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    M, b = np.broadcast_arrays(M, b)
    if M.size > F._CHUNK:
        Mf, bf = M.ravel(), b.ravel()
        return np.concatenate([omega_rel_vec(Mf[k:k + F._CHUNK], bf[k:k + F._CHUNK], mu, L, g, phi)
                               for k in range(0, Mf.size, F._CHUNK)]).reshape(M.shape)
    OL = omega_L_vec(M, b, L, phi) - _omega_L0(float(L), phi)
    dUv = np.vectorize(dU)(M, b)
    out = M * M / (2 * g) + dUv + OL + omega_muL_vec(M, b, mu, L, phi) - _omega_muL0(float(mu), float(L), phi)
    return np.where(M == 0, 0.0, out)


def omega_rel(M, b, mu, L, g, phi):
    if M == 0.0:
        return 0.0
    return float(omega_rel_vec(np.array([M]), np.array([b]), mu, L, g, phi)[0])

"""
Stability of the homogeneous solution at temperature T > 0 (publication note note_stab_T).

The homogeneous solution is the global minimum M = M(mu, L, T) of Omega_T(M, b = 0) (formula (A),
njl3.fast.thermal). The second variation, without the gap equation, is

    Phi_full(k, q1) = ½ G_T(M) + Phi_T(k, q1; M),

G_T(M) = (1/M) dOmega_T/dM (= 0 at M ≠ 0; G_T(0⁺) ≥ 0 at M = 0), Phi_T — the loop of formulas.phi with
θ(E - mu) -> 1 - n_F(E - mu) - n_F(E + mu). Both are smearings of the T = 0 quantities over the chemical potential
nu at the same M (f_T(E) = ∫_0^∞ w_T(nu) θ(E - nu) dnu, w_T = K_T(nu - mu) + K_T(nu + mu), ∫ w_T = 1):

    G_T(M; mu) = ∫ w_T(nu) G(M; nu) dnu,     Phi_T(k, q1; M, mu) = ∫ w_T(nu) Phi(k, q1; M, nu) dnu.

G_T: adaptive quadrature in nu with break points at the thresholds nu = m_n (the T = 0 gap function has
arccosh(nu/m_n) cusps there). Phi_T on a q1 grid: the same double-exponential nodes as formula (A) applied to the
vectorized T = 0 closed form (phi_mat_vec); its integrable logarithmic singularities in nu make this approximate near
the dips, so the minimum is refined with the exact Phi_T (formulas.phi, quadrature in p1).
"""

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from njl3.fast.thermal import UX, UW, CtxT
from njl3.fast.minimize_T import _hom
from njl3.hom import formulas as H
from njl3.hom import find_minimum as hom_find_minimum
from njl3.hom import phase_label as hom_phase_label
from njl3.stab import formulas as S

PI = math.pi
_KEEP = UW > 1e-13
_UX, _UW = UX[_KEEP], UW[_KEEP]


def _K(x, T):
    return 1.0 / (4.0 * T * math.cosh(min(abs(x) / (2.0 * T), 350.0)) ** 2)


def _masses(M, nu_max, L, phi):
    out, n = [], 0
    while True:
        ks = {abs(2.0 * PI * (n + phi) / L), abs(2.0 * PI * (-n - 1 + phi) / L)}
        ms = [math.hypot(M, k) for k in ks]
        out += ms
        if min(ms) > nu_max:
            return sorted(set(out))
        n += 1


def gap_G_T(M, mu, L, g, T, phi=0.0):
    """G_T(M) = (1/M) dOmega_T/dM; M = 0: G_T(0⁺). T = 0: njl3.hom."""
    if T == 0.0:
        return H.gap_G0(mu, L, g, phi) if M == 0.0 else H.gap_G(M, mu, L, g, phi)
    lo, hi = max(0.0, mu - 40.0 * T), mu + 40.0 * T
    if M == 0.0:
        F = lambda nu: H.gap_G0(nu, L, g, phi) if nu > 0.0 else 0.0
    else:
        F = lambda nu: H.gap_G(M, nu, L, g, phi)
    br = [m for m in _masses(M, hi, L, phi) if lo < m < hi]
    if lo < mu < hi:
        br.append(mu)
    return quad(lambda nu: (_K(nu - mu, T) + _K(nu + mu, T)) * F(nu), lo, hi,
                points=sorted(set(br)) or None, limit=4000, epsabs=1e-12, epsrel=1e-10)[0]


def phi_mat_T_vec(k, q1, M, mu, L, T, phi=0.0):
    """Matter part of Phi_T on an array q1 > 0: smearing of phi_mat_vec over nu (approximate near the dips)."""
    q1 = np.asarray(q1, float)
    s = np.zeros_like(q1)
    for x, w in zip(T * _UX, _UW):
        for nu in (mu + x, -mu + x):
            if nu > M:
                v = S.phi_mat_vec(k, q1, M, nu, L, phi)
                s += w * np.where(np.isfinite(v), v, 0.0)
    return s


def hom_T(mu, L, g, T, phi=0.0):
    """Global minimum of Omega_T(M, 0): (M, Omega, phase label V / I_j / T)."""
    ctx = CtxT(mu, L, g, phi, T)
    _, _, minima = hom_find_minimum(mu, L, g, phi)
    M, om = _hom(ctx, [m for m, _ in minima])
    return M, om, hom_phase_label(M, mu, L, phi)


def q_max_for(mu, T):
    """All T = 0 singular points lie at q1 ≤ 2 mu; at T > 0 the dips are smeared over ~T."""
    return max(10.0, 2.0 * mu + 10.0 * T + 2.0)


def scan_min_T(mu, L, g, T, M=None, phi=0.0, n_q=400, q_max=None, k_max=None, n_ref=4):
    """
    min over k ≥ 0, q1 ∈ (0, q_max] of Phi_full at temperature T > 0 for the homogeneous minimum M (computed if
    None). Grid: Phi_vac (exact) + smeared matter part; the n_ref lowest grid local minima (over all k) are refined
    by bounded Brent on the exact Phi_T in a window of ±1.5 grid steps (±max(1.5h, 10T) around the T = 0 singular
    points that are taken as extra candidates).
    Returns dict(phi_min, k, q1, M, const, phase).
    """
    if M is None:
        M, _, phase = hom_T(mu, L, g, T, phi)
    else:
        phase = hom_phase_label(M, mu, L, phi)
    zero = M <= 0.0
    m = S.M_ZERO if zero else M
    const = 0.5 * gap_G_T(0.0, mu, L, g, T, phi) if zero else 0.0
    q_max = q_max or q_max_for(mu, T)
    k_max = k_max if k_max is not None else int(q_max * L / (2.0 * PI))
    q = np.linspace(q_max / n_q, q_max, n_q)
    h = q[1] - q[0]
    cands, edge = [], []
    for k in range(0, k_max + 1):
        if zero:
            vac = S._vac_grid(k, float(m), float(L), float(phi), float(q_max), int(n_q))
        else:
            vac = S.phi_vac_vec(k, q, m, L, phi=phi)
        v = const + vac + phi_mat_T_vec(k, q, m, mu, L, T, phi)
        for i in range(n_q):
            if (i == 0 or v[i] <= v[i - 1]) and (i == n_q - 1 or v[i] <= v[i + 1]):
                cands.append((float(v[i]), k, float(q[i]), 1.5 * h))
        edge.append((float(v[0]), k, float(q[0]), float(q[0])))        # q1 -> 0 (k ≥ 1: q1 = 0 allowed)
        for qs in S.singular_q1(m, mu, L, k, phi):
            if 0.0 < qs <= q_max:
                cands.append((float(np.interp(qs, q, v)), k, qs, max(1.5 * h, 10.0 * T)))
    cands.sort()
    edge.sort()
    cands = edge[:2] + cands            # the two lowest q1 -> 0 edges are always refined
    best = None
    seen = []
    for v0, k, q0, w in cands:
        if len(seen) >= n_ref + 2:
            break
        if any(k == kk and abs(q0 - qq) < ww for kk, qq, ww in seen):
            continue
        seen.append((k, q0, w))
        f = lambda x, k=k: const + S.phi(k, x, m, mu, L, T, phi)
        a, b = max(q0 - w, 1e-6), min(q0 + w, q_max)
        r = minimize_scalar(f, bounds=(a, b), method='bounded', options=dict(xatol=1e-7))
        val, qq = (float(r.fun), float(r.x))
        fa = f(q0)
        if fa < val:
            val, qq = fa, q0
        if best is None or val < best['phi_min']:
            best = dict(phi_min=val, k=k, q1=qq)
    best.update(M=M, const=const, phase=phase)
    return best

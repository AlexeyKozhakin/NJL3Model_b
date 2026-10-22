"""
Fast global minimum of Omega(M, b) (methods M1 + M2 + M4 of PLAN.md).

Candidates: the homogeneous solution (njl3.hom, exact) and one well per channel j at the threshold
S_j = sqrt(mu² - k_j²). The channel well is predicted by the leading asymptotics
M_j = exp(-A_j/c_j - 1/2), b = S_j (accurate to |Δ ln M| ~ 1e-3 up to M ~ 0.5, see PLAN.md); for M_j > M_REFINE
it is refined by Newton steps in (ln M, t), b = S_j + t M, on the exact Omega (finite-difference
derivatives, one vectorized call per step). Temperature cut-off and candidate selection as in
njl3.inh.minimize. A_j are taken from Omega(M_PROBE, S_j) of all channels in one vectorized call.
"""

import math

import numpy as np

from njl3.full.minimize import FullResult, M_PROBE, OMEGA_NOISE, TC_RATIO, channel_m_min
from njl3.hom import find_minimum as hom_find_minimum
from njl3.hom import phase_label as hom_phase_label
from njl3.inh import formulas as F

M_REFINE = 1e-2          # below: the asymptotic well is used as is (unless it is a rival of the best)
M_RIVAL, RIVAL = 1e-3, 0.9   # asymptotic wells with M > M_RIVAL and Omega ≤ 0.9·best are refined
LNM_MAX = math.log(6.0)
_H = 1e-3                # finite-difference step in (ln M, t)


def channels(mu, L, g, phi):
    """[(j, S, C, A, c)] for all levels, A from Omega(M_PROBE, S) (one vectorized call)."""
    lv = F.levels(mu, L, phi)
    if not lv:
        return []
    S = np.array([math.sqrt(mu * mu - k * k) for k, _ in lv])
    C = np.array([float(c) for _, c in lv])
    c = C * S / (2 * math.pi * L * mu)
    Om = F.omega_rel_vec(np.full(len(S), M_PROBE), S, mu, L, g, phi)
    A = Om / M_PROBE**2 + c * math.log(1.0 / M_PROBE)
    return [(j, S[j], C[j], A[j], c[j]) for j in range(len(lv))]


def _omega_xt(x, t, S, mu, L, g, phi):
    x = np.asarray(x, dtype=float)
    M = np.exp(np.minimum(x, LNM_MAX))
    b = S + np.asarray(t, dtype=float) * M
    out = F.omega_rel_vec(M, np.maximum(b, 0.0), mu, L, g, phi)
    return np.where(b >= 0, out, 1e3)


_SX = np.linspace(-1.5, 1.5, 7)                                     # scouting grid around the asymptotics
_ST = np.concatenate([np.linspace(-4.0, -1.0, 4), np.linspace(-0.6, 0.6, 5), [1.0, 2.0, 3.0]])


def refine(x0, S, mu, L, g, phi, lnm_lo, steps=12, t0=None):
    """Local minimum of the channel near (ln M, t) = (x0, 0): one vectorized scouting call on a coarse
    (ln M, t) grid, then damped Newton with a 3×3 stencil (gradient step where the Hessian is not
    positive definite)."""
    if t0 is None:
        X, T = np.meshgrid(np.clip(x0 + _SX, lnm_lo, LNM_MAX), _ST, indexing='ij')
        f0 = _omega_xt(X.ravel(), T.ravel(), S, mu, L, g, phi)
        k = int(np.argmin(f0))
        x, t, fx = float(X.ravel()[k]), float(T.ravel()[k]), float(f0[k])
    else:
        x, t = x0, t0
        fx = float(_omega_xt([x], [t], S, mu, L, g, phi)[0])
    d = np.array([-_H, 0.0, _H])
    for _ in range(steps):
        Xs, Ts = np.meshgrid(x + d, t + d, indexing='ij')
        f = _omega_xt(Xs.ravel(), Ts.ravel(), S, mu, L, g, phi).reshape(3, 3)
        gx = (f[2, 1] - f[0, 1]) / (2 * _H)
        gt = (f[1, 2] - f[1, 0]) / (2 * _H)
        hxx = (f[2, 1] - 2 * f[1, 1] + f[0, 1]) / _H**2
        htt = (f[1, 2] - 2 * f[1, 1] + f[1, 0]) / _H**2
        hxt = (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) / (4 * _H**2)
        det = hxx * htt - hxt * hxt
        if hxx > 0 and det > 0:
            dx, dt = -(htt * gx - hxt * gt) / det, -(hxx * gt - hxt * gx) / det
        else:                                               # not convex: scaled gradient step
            sc = 0.1 / max(abs(gx), abs(gt), 1e-300)
            dx, dt = -gx * sc, -gt * sc
        lam = 1.0
        for _ in range(12):                                 # damping: accept only a decrease
            xn, tn = min(max(x + lam * dx, lnm_lo), LNM_MAX), t + lam * dt
            fn = float(_omega_xt([xn], [tn], S, mu, L, g, phi)[0])
            if fn <= fx:
                break
            lam *= 0.5
        else:
            break
        step = max(abs(xn - x), abs(tn - t))
        x, t, fx = xn, tn, fn
        if step < 1e-7:
            break
    return x, t, fx


_GM = np.geomspace(0.02, 4.0, 40)          # dense scan in the strip near mu ~ pi (variant B of PLAN.md)
GLOBAL_TRIGGER = 0.3      # global scouting if some channel well is large or the b = 0 solution is intermediate (0 < M < mu)


def global_wells(mu, L, g, phi, thr, m_min, n_best=5):
    """Coarse (M, b) scan in one vectorized call; the best grid-local minima, refined by Newton relative to the
    nearest threshold. Returns [(phase, M, lnM, b, omega)]."""
    bg = np.linspace(0.0, mu + 0.5, 60)
    Mg, Bg = np.meshgrid(_GM, bg, indexing='ij')
    Om = F.omega_rel_vec(Mg.ravel(), Bg.ravel(), mu, L, g, phi).reshape(Mg.shape)
    P = np.pad(Om, 1, constant_values=np.inf)
    loc = np.ones(Om.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= Om <= P[1 + di:1 + di + Om.shape[0], 1 + dj:1 + dj + Om.shape[1]]
    idx = [tuple(q) for q in np.argwhere(loc & (Om < -OMEGA_NOISE) & (Bg > 0))]
    idx = sorted(idx, key=lambda q: Om[q])[:n_best]
    out = []
    lnm_lo = math.log(max(m_min, 1e-6))
    for i, k in idx:
        M, b = Mg[i, k], Bg[i, k]
        j, S, _ = min(thr, key=lambda q: abs(q[1] - b))
        x, t, om = refine(max(math.log(M), lnm_lo), S, mu, L, g, phi, lnm_lo, t0=(b - S) / M)
        Mr = math.exp(x)
        if om < -OMEGA_NOISE and Mr > m_min and S + t * Mr > 1e-8:
            br = S + t * Mr
            jn = min(thr, key=lambda q: abs(q[1] - br))[0]
            out.append((f'B{jn}', Mr, x, br, om))
    return out


def find_minimum(mu, L, g, phi=0.0, T=0.0):
    phi = F.check_phi(phi)
    Mb, Ob, _ = hom_find_minimum(mu, L, g, phi)
    b0 = dict(M=Mb, omega=Ob, phase=hom_phase_label(Mb, mu, L, phi))
    if T > 0 and 0 < Mb <= T / TC_RATIO:
        Mb, Ob = 0.0, 0.0
    cand = [dict(phase=hom_phase_label(Mb, mu, L, phi), M=Mb, lnM=math.log(Mb) if Mb > 0 else -math.inf, b=0.0,
                 omega=Ob, asym=False)]
    chans = []
    for j, S, C, A, c in channels(mu, L, g, phi):
        lnM = -A / c - 0.5
        m_min = channel_m_min(S, mu, T)
        ch = dict(n=j, S=S, c=c, A=A, M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=True)
        if lnM < math.log(max(m_min, 1e-300)) or lnM < -700:
            chans.append(ch)                                    # gap does not survive T
            continue
        if lnM < math.log(M_REFINE):
            M = math.exp(lnM)
            ch.update(M=M, lnM=lnM, omega=-0.5 * c * M * M)
        else:
            lnm_lo = math.log(m_min) if m_min > 0 else math.log(1e-6)
            x0 = min(max(lnM, lnm_lo), LNM_MAX)
            x, t, om = refine(x0, S, mu, L, g, phi, lnm_lo)
            M = math.exp(x)
            if om < -OMEGA_NOISE and M > m_min:
                ch.update(M=M, lnM=x, b=S + t * M, omega=om, asym=False)
        chans.append(ch)
        if ch['M'] > 0 and ch['b'] > 1e-8:
            cand.append(dict(phase=f'B{j}', M=ch['M'], lnM=ch['lnM'], b=ch['b'], omega=ch['omega'], asym=ch['asym']))
    # near-degenerate channels: asymptotic wells within RIVAL of the best are refined (the asymptotics alone may
    # pick the wrong one of two channels whose Omega differ by ~1%)
    best_om = min([q['omega'] for q in cand] + [0.0])
    for ch in chans:
        if ch['asym'] and ch['M'] > M_RIVAL and ch['omega'] <= RIVAL * best_om:
            m_min = channel_m_min(ch['S'], mu, T)
            lnm_lo = math.log(m_min) if m_min > 0 else math.log(1e-6)
            x, t, om = refine(max(ch['lnM'], lnm_lo), ch['S'], mu, L, g, phi, lnm_lo)
            M = math.exp(x)
            if om < ch['omega'] and M > m_min:
                ch.update(M=M, lnM=x, b=ch['S'] + t * M, omega=om, asym=False)
                for q in cand:
                    if q['phase'] == f"B{ch['n']}":
                        q.update(M=M, lnM=x, b=ch['S'] + t * M, omega=om, asym=False)
    thr = [(ch['n'], ch['S'], 0) for ch in chans]
    big = any(ch['lnM'] > math.log(GLOBAL_TRIGGER) for ch in chans) or 0 < b0['M'] < mu
    if thr and big:
        m_min_all = min(channel_m_min(S, mu, T) for _, S, _ in thr)
        for ph, M, lnM_, b, om in global_wells(mu, L, g, phi, thr, m_min_all):
            cand.append(dict(phase=ph, M=M, lnM=lnM_, b=b, omega=om, asym=False))
    best = cand[0]
    for c_ in sorted(cand[1:], key=lambda q: q['omega']):
        if c_['omega'] < best['omega'] - (0.0 if c_['asym'] else OMEGA_NOISE):
            best = c_
            break
    if best['phase'][0] == 'B':
        n_near = min(chans, key=lambda q: abs(q['S'] - best['b']))['n']
        best = dict(best, phase=f'B{n_near}')
    if best['phase'] == 'T':
        best = dict(best, b=float('nan'))
    return FullResult(mu=mu, L=L, g=g, phase=best['phase'], M_star=best['M'], ln_M_star=best['lnM'], b_star=best['b'],
                      omega_star=best['omega'], asymptotic=best['asym'], channels=chans, b0=b0)

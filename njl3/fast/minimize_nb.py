"""
njl3.fast.minimize with the compiled core (njl3.fast.nb): the same candidates and rules, Omega and the inner
loops (channel A, scouting + Newton, dense scan) in numba. Results agree with njl3.fast.minimize to rounding.
"""

import math

import numpy as np

from njl3.fast import nb
from njl3.fast.minimize import GLOBAL_TRIGGER, LNM_MAX, M_REFINE, M_RIVAL, RIVAL, _GM, _ST, _SX
from njl3.full.minimize import FullResult, M_PROBE, OMEGA_NOISE, TC_RATIO, channel_m_min
from njl3.hom import find_minimum as hom_find_minimum
from njl3.hom import phase_label as hom_phase_label
from njl3.inh import formulas as F


def _refine(ctx, x0, S, lnm_lo, t0=None):
    return nb.refine_nb(x0, 0.0 if t0 is None else t0, t0 is None, S, lnm_lo, LNM_MAX, _SX, _ST, 12, *ctx.args)


def channels(ctx):
    c0 = math.log(1.0 / M_PROBE)
    out = []
    for j, (k, C) in enumerate(zip(ctx.levk, ctx.levC)):
        S = math.sqrt(ctx.mu ** 2 - k * k)
        c = C * S / (2 * math.pi * ctx.L * ctx.mu)
        A = ctx.omega(M_PROBE, S) / M_PROBE**2 + c * c0
        out.append((j, S, C, A, c))
    return out


BG_STEP = 0.06                       # fixed b step of the dense scan: the Omega_L grid is shared by all mu of one L
_OL_CACHE = {}                       # (L, phi) -> Omega_L on _GM × b grid (per process)


def _ol_grid(ctx, nb_needed):
    key = (float(ctx.L), ctx.phi)
    have = _OL_CACHE.get(key)
    if have is None or have.shape[1] < nb_needed:
        n = max(nb_needed, 0 if have is None else have.shape[1])
        bg = np.arange(n) * BG_STEP
        e, c, m, h, p0, xm, s0 = ctx.tab
        start = 0 if have is None else have.shape[1]
        new = nb.grid_OL_nb(_GM, bg[start:], ctx.L, e, c, m, h, p0, xm, s0, nb.TSX, nb.TSW, nb.GTX, nb.GTW, nb.CHX, nb.CHW)
        have = new if have is None else np.concatenate([have, new], axis=1)
        if len(_OL_CACHE) > 512:
            _OL_CACHE.clear()
        _OL_CACHE[key] = have
    return have[:, :nb_needed]


def global_wells(ctx, thr, m_min, n_best=5):
    mu = ctx.mu
    nbg = int(math.ceil((mu + 0.5) / BG_STEP)) + 1
    bg = np.arange(nbg) * BG_STEP
    OL = _ol_grid(ctx, nbg)
    Om = nb.grid_rest_nb(_GM, bg, OL, mu, ctx.L, ctx.g, ctx.z0, ctx.levk, ctx.levC, nb.TSX, nb.TSW)
    Mg, Bg = np.meshgrid(_GM, bg, indexing='ij')
    P = np.pad(Om, 1, constant_values=np.inf)
    loc = np.ones(Om.shape, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                loc &= Om <= P[1 + di:1 + di + Om.shape[0], 1 + dj:1 + dj + Om.shape[1]]
    idx = sorted([tuple(q) for q in np.argwhere(loc & (Om < -OMEGA_NOISE) & (Bg > 0))], key=lambda q: Om[q])[:n_best]
    out = []
    lnm_lo = math.log(max(m_min, 1e-6))
    for i, k in idx:
        M, b = Mg[i, k], Bg[i, k]
        j, S, _ = min(thr, key=lambda q: abs(q[1] - b))
        x, t, om = _refine(ctx, max(math.log(M), lnm_lo), S, lnm_lo, t0=(b - S) / M)
        Mr = math.exp(x)
        if om < -OMEGA_NOISE and Mr > m_min and S + t * Mr > 1e-8:
            br = S + t * Mr
            jn = min(thr, key=lambda q: abs(q[1] - br))[0]
            out.append((f'B{jn}', Mr, x, br, om))
    return out


def find_minimum(mu, L, g, phi=0.0, T=0.0):
    phi = F.check_phi(phi)
    ctx = nb.Ctx(mu, L, g, phi)
    Mb, Ob, _ = hom_find_minimum(mu, L, g, phi)
    b0 = dict(M=Mb, omega=Ob, phase=hom_phase_label(Mb, mu, L, phi))
    if T > 0 and 0 < Mb <= T / TC_RATIO:
        Mb, Ob = 0.0, 0.0
    cand = [dict(phase=hom_phase_label(Mb, mu, L, phi), M=Mb, lnM=math.log(Mb) if Mb > 0 else -math.inf, b=0.0,
                 omega=Ob, asym=False)]
    chans = []
    for j, S, C, A, c in channels(ctx):
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
            x, t, om = _refine(ctx, x0, S, lnm_lo)
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
            x, t, om = _refine(ctx, max(ch['lnM'], lnm_lo), ch['S'], lnm_lo)
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
        for ph, M, lnM_, b, om in global_wells(ctx, thr, m_min_all):
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

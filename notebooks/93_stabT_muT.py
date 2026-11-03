"""
Stability of the homogeneous solution in (mu, T) at fixed L (njl3.stab.thermal, note note_stab_T):
in every point the global minimum M(mu, L, T) of Omega_T(M, 0) and min over (k, q1) of
Phi_full = ½ G_T(M) + Phi_T(k, q1; M) (M ≠ 0: G_T = 0; M = 0: ½ G_T(0⁺) + the loop at M -> 0).

Grid as notebooks/92_muT_maps.py: mu linear on [MU0, MU1], T logarithmic on [T0, T1]; base N × N, then K halvings
of the cells whose corners differ in the sign of min Phi or in the homogeneous phase or by > REFINE_DPHI in min Phi; inside uniform cells —
bilinear interpolation of min Phi (k, q1 and the phase from a corner).

    python notebooks/93_stabT_muT.py --L L [--phi PHI] [--g G] [--n N] [--refine K] [--mu MU0 MU1] [--T T0 T1]
→ notebooks/output/93_stabT_L<L>_phi<φ>_g<g>.json and .png (log T) / _linT.png (linear T)
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'notebooks', 'output')

ap = argparse.ArgumentParser()
ap.add_argument('--L', type=float, required=True)
ap.add_argument('--phi', type=float, default=0.0)
ap.add_argument('--g', type=float, default=-1.0)
ap.add_argument('--n', type=int, default=60)
ap.add_argument('--refine', type=int, default=2)
ap.add_argument('--mu', type=float, nargs=2, default=(0.05, 7.0))
ap.add_argument('--T', type=float, nargs=2, default=(1e-3, 5.0))
ap.add_argument('--plot-only', action='store_true')
A = ap.parse_args()
R = 2 ** A.refine
NF = R * (A.n - 1) + 1
MU = np.linspace(A.mu[0], A.mu[1], NF)
TT = np.geomspace(A.T[0], A.T[1], NF)
NAME = os.path.join(OUT, f'93_stabT_L{A.L:g}_phi{A.phi:g}_g{A.g:g}')


def solve(p):
    warnings.simplefilter('ignore')
    from njl3.stab.thermal import scan_min_T
    i, j = p
    r = scan_min_T(float(MU[j]), A.L, A.g, float(TT[i]), phi=A.phi)
    return i, j, (r['phase'], r['M'], r['phi_min'], r['k'], r['q1'])


REFINE_DPHI = 0.3       # also refine cells with a strong jump of min Phi (subband openings: van Hove dips)


def needs_refine(res, corners):
    v = [res[c][2] for c in corners]
    return (len({res[c][0] for c in corners}) > 1
            or len({x < 0 for x in v}) > 1
            or max(v) - min(v) > REFINE_DPHI)


def compute():
    t0 = time.time()
    res = {}
    base = [(i, j) for j in range(0, NF, R) for i in range(0, NF, R)]
    with Pool() as pool:
        for n, (i, j, o) in enumerate(pool.imap_unordered(solve, base, chunksize=4), 1):
            res[(i, j)] = o
            if n % 400 == 0:
                print(f'  базовая сетка: {n}/{len(base)}, {time.time() - t0:.0f} с', flush=True)
        print(f'  базовая сетка: {len(res)} точек, {time.time() - t0:.0f} с', flush=True)
        cells = [(i, j) for i in range(0, NF - 1, R) for j in range(0, NF - 1, R)]
        s, unrefined = R, []
        while s > 1:
            ref = []
            for i, j in cells:
                corners = [(i, j), (i + s, j), (i, j + s), (i + s, j + s)]
                (ref if needs_refine(res, corners) else unrefined).append((i, j, s))
            h = s // 2
            pts = sorted({(i + a, j + c) for i, j, _ in ref for a in (0, h, s) for c in (0, h, s)} - set(res))
            for i, j, o in pool.imap_unordered(solve, pts, chunksize=4):
                res[(i, j)] = o
            print(f'  уточнение шаг {h}: {len(ref)} ячеек, {len(pts)} точек, всего {len(res)}, '
                  f'{time.time() - t0:.0f} с', flush=True)
            cells = [(i + a, j + c) for i, j, _ in ref for a in (0, h) for c in (0, h)]
            s = h
    phase = np.empty((NF, NF), dtype=object)
    M, P, K, Q = (np.full((NF, NF), np.nan) for _ in range(4))
    interp = np.ones((NF, NF), dtype=bool)
    for (i, j), o in res.items():
        phase[i, j], M[i, j], P[i, j], K[i, j], Q[i, j] = o
        interp[i, j] = False
    for i, j, s in unrefined:
        C = [res[(i, j)], res[(i + s, j)], res[(i, j + s)], res[(i + s, j + s)]]
        for a in range(s + 1):
            for c in range(s + 1):
                if not interp[i + a, j + c]:
                    continue
                u, v = a / s, c / s
                w = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
                near = C[int(round(u)) + 2 * int(round(v))]
                phase[i + a, j + c] = near[0]
                M[i + a, j + c] = sum(wk * Ck[1] for wk, Ck in zip(w, C))
                P[i + a, j + c] = sum(wk * Ck[2] for wk, Ck in zip(w, C))
                K[i + a, j + c], Q[i + a, j + c] = near[3], near[4]
    r6 = lambda x: None if not np.isfinite(x) else float(f'{x:.6g}')
    with open(NAME + '.json', 'w') as f:
        json.dump(dict(L=A.L, phi=A.phi, g=A.g, refine=A.refine, mu=[float(f'{x:.9g}') for x in MU],
                       T=[float(f'{x:.9g}') for x in TT], phase=phase.astype(str).tolist(),
                       M=[[r6(x) for x in r] for r in M], phi_min=[[r6(x) for x in r] for r in P],
                       k=[[None if not np.isfinite(x) else int(x) for x in r] for r in K],
                       q1=[[r6(x) for x in r] for r in Q], interp=interp.astype(int).tolist()), f,
                  separators=(',', ':'))
    print(f'сетка {NF}×{NF}: посчитано {len(res)} точек ({100 * len(res) / interp.size:.0f}%), '
          f'{time.time() - t0:.0f} с ({NAME}.json)', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, SymLogNorm
    d = json.load(open(NAME + '.json'))
    mu, T = np.array(d['mu']), np.array(d['T'])
    ph = np.array(d['phase'])
    P = np.array([[np.nan if x is None else x for x in r] for r in d['phi_min']], float)
    code = np.full(ph.shape, 2.0)
    code[ph == 'V'] = 0
    code[np.char.startswith(ph, 'I')] = 1
    # устойчивость × фаза: 0..2 устойчиво (V, I, T), 3..5 неустойчиво
    sc = code + 3 * (P < 0)
    colors = ['#2166ac', '#67a9cf', '#bdbdbd', '#67001f', '#d6604d', '#f4a582']
    labels = ['V, устойчиво', 'I$_j$, устойчиво', 'T (M = 0), устойчиво',
              'V, неустойчиво', 'I$_j$, неустойчиво', 'T (M = 0), неустойчиво']
    vmax = np.nanmax(np.abs(P))
    for logT, suffix in ((True, ''), (False, '_linT')):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
        im = ax[0].pcolormesh(mu, T, P, cmap='RdBu', norm=SymLogNorm(linthresh=1e-2, vmin=-vmax, vmax=vmax),
                              shading='nearest', rasterized=True)
        fig.colorbar(im, ax=ax[0], label=r'$\min_{k,q_1}\Phi^{\rm full}_T$')
        ax[0].set_title(r'$\min_{k,q_1}\Phi^{\rm full}_T$ (синее > 0 устойчиво, красное < 0); линии — граница фаз '
                        r'и $\Phi=0$', fontsize=9)
        ax[1].pcolormesh(mu, T, sc, cmap=ListedColormap(colors), vmin=-0.5, vmax=5.5, shading='nearest',
                         rasterized=True)
        ax[1].set_title('устойчивость однородного минимума × его фаза', fontsize=10)
        for a in ax:
            a.contour(mu, T, P, levels=[0.0], colors='k', linewidths=1.0)
            a.contour(mu, T, code, levels=[0.5, 1.5], colors='0.35', linewidths=0.6, linestyles='--')
            a.set_xlabel(r'$\mu$'); a.set_ylabel('$T$')
            if logT:
                a.set_yscale('log')
        fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)], loc='upper center',
                   ncol=6, fontsize=8)
        fig.suptitle(f'Устойчивость однородного решения в (μ, T) при L = {A.L:g}, φ = {A.phi:g}, g = {A.g:g}; '
                     f'шкала T {"логарифмическая" if logT else "линейная"}', y=0.005, va='bottom', fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(NAME + suffix + '.png', dpi=130)
        plt.close(fig)
    print(NAME + '.png', NAME + '_linT.png')


if __name__ == '__main__':
    if not A.plot_only:
        compute()
    plot()

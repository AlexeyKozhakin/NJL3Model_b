"""
Phase diagram in (mu, T) at fixed L: global minimum of Omega_T(M, b) (formula (A), njl3.fast.minimize_T).

Grid: mu linear on [MU0, MU1], T logarithmic on [T0, T1]; base N × N, then K halvings of the cells where the
corners differ in phase or ln M* (as notebooks/90_inh_maps.py); inside uniform cells — bilinear interpolation.
Robustness at large T (the T = 0 wells move far):
  - the T = 0 candidates depend on mu only — computed once per mu column (cached in the worker);
  - continuation in T: a mu column of the base grid is computed with T increasing, the minimum at the previous T is
    an extra start for the next one;
  - refinement points start also from the minima at the corners of their cell.

    python notebooks/92_muT_maps.py --L L [--phi PHI] [--g G] [--n N] [--refine K] [--mu MU0 MU1] [--T T0 T1]
→ notebooks/output/92_muT_L<L>_phi<φ>_g<g>.json and .png (log T) / _linT.png (linear T)
"""
import argparse
import functools
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
REFINE_DLNM = 1.0

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
NAME = os.path.join(OUT, f'92_muT_L{A.L:g}_phi{A.phi:g}_g{A.g:g}')


@functools.lru_cache(maxsize=4096)
def _r0(mu):
    from njl3.fast import minimize_nb
    return minimize_nb.find_minimum(mu, A.L, A.g, A.phi, T=0.0)


def _solve(mu, T, extra):
    warnings.simplefilter('ignore')
    from njl3.fast.minimize_T import find_minimum
    r = find_minimum(mu, A.L, A.g, A.phi, T, extra=extra, r0=_r0(mu))
    return (r.phase, r.M_star, r.ln_M_star, r.b_star, r.omega_star)


def column(j):
    """Base-grid column mu = MU[j], T increasing (continuation)."""
    out, prev = [], ()
    for i in range(0, NF, R):
        o = _solve(float(MU[j]), float(TT[i]), prev)
        out.append((i, j, o))
        prev = ((o[1], o[3] if math.isfinite(o[3]) else 0.0),) if o[1] > 0 else ()
    return out


def refine_job(job):
    """Refinement points of one mu column; starts — the minima at the cell corners."""
    j, items = job
    return [(i, j, _solve(float(MU[j]), float(TT[i]), extra)) for i, extra in items]


def needs_refine(res, corners):
    if len({res[c][0] for c in corners}) > 1:
        return True
    lm = [res[c][2] for c in corners if np.isfinite(res[c][2])]
    return len(lm) > 1 and max(lm) - min(lm) > REFINE_DLNM


def compute():
    t0 = time.time()
    res = {}
    with Pool() as pool:
        for out in pool.imap_unordered(column, range(0, NF, R)):
            for i, j, o in out:
                res[(i, j)] = o
        print(f'  базовая сетка: {len(res)} точек, {time.time() - t0:.0f} с', flush=True)
        cells = [(i, j) for i in range(0, NF - 1, R) for j in range(0, NF - 1, R)]
        s, unrefined = R, []
        while s > 1:
            ref = []
            for i, j in cells:
                corners = [(i, j), (i + s, j), (i, j + s), (i + s, j + s)]
                (ref if needs_refine(res, corners) else unrefined).append((i, j, s))
            h = s // 2
            jobs = {}
            for i, j, _ in ref:
                corners = [(i, j), (i + s, j), (i, j + s), (i + s, j + s)]
                extra = tuple((res[c][1], res[c][3] if math.isfinite(res[c][3]) else 0.0) for c in corners if res[c][1] > 0)
                for a in (0, h, s):
                    for c in (0, h, s):
                        p = (i + a, j + c)
                        if p not in res:
                            jobs.setdefault(p[1], {})[p[0]] = extra
            batch = [(j, sorted(d.items())) for j, d in jobs.items()]
            n = sum(len(d) for _, d in batch)
            for out in pool.imap_unordered(refine_job, batch):
                for i, j, o in out:
                    res[(i, j)] = o
            print(f'  уточнение шаг {h}: {len(ref)} ячеек, {n} точек, всего {len(res)}, {time.time() - t0:.0f} с', flush=True)
            cells = [(i + a, j + c) for i, j, _ in ref for a in (0, h) for c in (0, h)]
            s = h
    phase = np.empty((NF, NF), dtype=object)
    lnM, b, Om = (np.full((NF, NF), np.nan) for _ in range(3))
    interp = np.ones((NF, NF), dtype=bool)
    for (i, j), o in res.items():
        phase[i, j], _, lnM[i, j], b[i, j], Om[i, j] = o
        interp[i, j] = False
    for i, j, s in unrefined:
        C = [res[(i, j)], res[(i + s, j)], res[(i, j + s)], res[(i + s, j + s)]]
        for a in range(s + 1):
            for c in range(s + 1):
                if not interp[i + a, j + c]:
                    continue
                u, v = a / s, c / s
                w = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
                bil = lambda k: sum(wk * Ck[k] for wk, Ck in zip(w, C)) if all(np.isfinite(Ck[k]) for Ck in C) else (
                    C[0][k] if not np.isfinite(C[0][k]) else np.nan)
                phase[i + a, j + c] = C[0][0]
                lnM[i + a, j + c], b[i + a, j + c], Om[i + a, j + c] = bil(2), bil(3), bil(4)
    r6 = lambda x: None if not np.isfinite(x) else float(f'{x:.6g}')
    with open(NAME + '.json', 'w') as f:
        json.dump(dict(L=A.L, phi=A.phi, g=A.g, refine=A.refine, mu=[float(f'{x:.9g}') for x in MU],
                       T=[float(f'{x:.9g}') for x in TT], phase=phase.astype(str).tolist(),
                       lnM=[[r6(x) for x in r] for r in lnM], b=[[r6(x) for x in r] for r in b],
                       omega=[[r6(x) for x in r] for r in Om], interp=interp.astype(int).tolist()), f,
                  separators=(',', ':'))
    print(f'сетка {NF}×{NF}: посчитано {len(res)} точек ({100 * len(res) / interp.size:.0f}%), '
          f'{time.time() - t0:.0f} с ({NAME}.json)', flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    d = json.load(open(NAME + '.json'))
    mu, T = np.array(d['mu']), np.array(d['T'])
    ph = np.array(d['phase'])
    lnM = np.array([[np.nan if x is None else x for x in r] for r in d['lnM']], float)
    b = np.array([[np.nan if x is None else x for x in r] for r in d['b']], float)
    code = np.full(ph.shape, 3.0)
    code[ph == 'V'] = 0
    code[np.char.startswith(ph, 'I')] = 1
    code[np.char.startswith(ph, 'B')] = 2
    colors = ['#2166ac', '#67a9cf', '#ef8a62', '#e3e2dd']
    labels = ['V', 'I$_j$ (b = 0)', 'B$_n$ (спираль)', 'T (M* = 0)']
    for logT, suffix in ((True, ''), (False, '_linT')):
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        ax[0].pcolormesh(mu, T, code, cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5, shading='nearest', rasterized=True)
        ax[0].set_title('фаза', fontsize=10)
        im = ax[1].pcolormesh(mu, T, lnM / np.log(10), cmap='viridis', vmin=-3, vmax=1, shading='nearest', rasterized=True)
        fig.colorbar(im, ax=ax[1], label='$\\log_{10}M^*$')
        ax[1].set_title('$\\log_{10}M^*$ (серое — M* = 0)', fontsize=10)
        im = ax[2].pcolormesh(mu, T, b, cmap='Oranges', vmin=0, shading='nearest', rasterized=True)
        fig.colorbar(im, ax=ax[2], label='$b^*$')
        ax[2].set_title('$b^*$ (спираль; белое — b = 0 или M = 0)', fontsize=10)
        for a in ax[1:]:
            a.set_facecolor(colors[3])
        for a in ax:
            a.set_xlabel('$\\mu$'); a.set_ylabel('$T$')
            if logT:
                a.set_yscale('log')
        fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)], loc='upper center',
                   ncol=4, fontsize=9)
        fig.suptitle(f'Фазовая диаграмма в (μ, T) при L = {A.L:g}, φ = {A.phi:g}, g = {A.g:g}: минимум $\\Omega_T(M,b)$ '
                     f'(формула A); шкала T {"логарифмическая" if logT else "линейная"}', y=0.005, va='bottom', fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        fig.savefig(NAME + suffix + '.png', dpi=130)
        plt.close(fig)
    print(NAME + '.png', NAME + '_linT.png')


if __name__ == '__main__':
    if not A.plot_only:
        compute()
    plot()

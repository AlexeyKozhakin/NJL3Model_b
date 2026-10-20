"""
Как фазы B_n на R²×S¹ исчезают при L -> ∞ (связь задачи L = ∞ со статьёй 2).

Для фиксированных mu прослеживается глобальный минимум njl3.full при росте L и асимптотика
лучшего канала M*_n = exp(-A_n/c_n - 1/2), c_n = C_n S_n /(2π L mu) ∝ 1/L.

    python notebooks/71_Bn_vs_L.py      ->  notebooks/output/71_Bn_vs_L.png, .json
"""

import json
import math
import os
import sys
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full.minimize import channel_asymptotics, find_minimum, thresholds

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
MUS = [3.6, 4.5, 6.0]
LS = np.geomspace(1.0, 60.0, 26)


def point(args):
    mu, L = args
    r = find_minimum(mu, L, G)
    # лучший канал по асимптотике: наибольшая глубина -c M*²/2
    best = None
    for n, S, C in thresholds(mu, L):
        a = channel_asymptotics(S, C, mu, L, G)
        if best is None or a['omega'] < best[1]['omega']:
            best = (n, a)
    return dict(mu=mu, L=float(L), phase=r.phase, lnM=r.ln_M_star, b=r.b_star, asym=r.asymptotic,
                n_best=best[0], lnM_asym=best[1]['lnM'], c=best[1]['c'], A=best[1]['A'])


def main():
    jobs = [(mu, L) for mu in MUS for L in LS]
    with Pool() as pool:
        res = pool.map(point, jobs)
    with open(os.path.join(OUT, '71_Bn_vs_L.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(f"mu={r['mu']:.1f} L={r['L']:6.2f} phase={r['phase']:3s} lnM*={r['lnM']:9.3f} "
              f"best channel n={r['n_best']:2d}: c={r['c']:.4f} A={r['A']:+.4f} lnM_asym={r['lnM_asym']:9.3f}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mu, c in zip(MUS, ('tab:blue', 'tab:orange', 'tab:green')):
        rr = [r for r in res if r['mu'] == mu]
        L = np.array([r['L'] for r in rr])
        lnM = np.array([r['lnM'] for r in rr])
        axes[0].plot(L, lnM / math.log(10), 'o-', color=c, ms=3, label=f'$\\mu={mu}$: глобальный минимум')
        axes[0].plot(L, np.array([r['lnM_asym'] for r in rr]) / math.log(10), '--', color=c, lw=1,
                     label=f'$\\mu={mu}$: асимптотика лучшего канала')
        axes[1].plot(L, [r['c'] * r['L'] for r in rr], 'o-', color=c, ms=3, label=f'$\\mu={mu}$')
    axes[0].set_xscale('log'); axes[0].set_xlabel('$L$'); axes[0].set_ylabel('$\\log_{10} M^*$')
    axes[0].set_title('Щель фаз $B_n$ при росте $L$ (на $\\mathbb{R}^2\\times S^1$)', fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].set_xscale('log'); axes[1].set_xlabel('$L$'); axes[1].set_ylabel('$c_n\\cdot L$')
    axes[1].set_title('Коэффициент логарифма $c_n\\propto 1/L$', fontsize=10); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '71_Bn_vs_L.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()

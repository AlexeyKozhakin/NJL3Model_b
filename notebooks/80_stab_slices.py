"""
Slices of the second variation Phi(k, q1) of the GN action around the homogeneous solution M*(mu, L)
(njl3.stab). Phi < 0 — the homogeneous phase is unstable against the fluctuation (k, q1).

80_stab_slices.png — Phi(q1) for k = 0..3 at three points of the intermediate phases I_j;
                     T = 0 (lines) and T = 1e-3 for k = 0 (dots near the 2k_F singularity).
"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from njl3.b0 import phase_at
from njl3.stab import phi
from njl3.stab.formulas import phi_vec, singular_q1

warnings.simplefilter('ignore')
OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
POINTS = [(3.03, 2.0), (3.13, 5.0), (3.15, 10.0)]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    q = np.linspace(0.005, 10.0, 4000)
    for ax, (mu, L) in zip(axes, POINTS):
        ph, M = phase_at(mu, L, G)
        for k, col in zip(range(4), COLORS):
            v = phi_vec(k, q, M, mu, L)
            ax.plot(q, v, color=col, lw=1.4, label=f'$k={k}$')
        qs = [x for x in singular_q1(M, mu, L, 0) if x > 0]
        for x in qs:
            ax.axvline(x, color='0.6', lw=0.6, ls=':')
            qq = np.linspace(x - 0.05, x + 0.05, 41)
            ax.plot(qq, [phi(0, y, M, mu, L, 1e-3) for y in qq], '.', ms=2.5, color='k')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylim(-1.0, 1.2)
        ax.set_xlim(0, 10)
        ax.set_xlabel('$q_1$')
        ax.set_title(f'$\\mu={mu}$, $L={L:g}$: фаза {ph}, $M^*={M:.3f}$', fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('$\\Phi(k, q_1)$')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Вторая вариация вокруг однородного решения: $\\Phi<0$ — неустойчивость '
                 '(точки: $T=10^{-3}$, $k=0$; пунктир: $q_1=2k_{F,n}$)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '80_stab_slices.png'), dpi=150)


if __name__ == '__main__':
    main()

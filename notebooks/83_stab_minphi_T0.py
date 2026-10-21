"""
The map of 82_stab_minphi at T = 0: min over q1 ∈ (0, 10] of Phi. In the intermediate phases I_j every
point has a 2k_F singularity inside (0, 10] (filled channel n, q1 = 2 P_n ≤ 2 mu), so min Phi = -∞ there;
it is replaced by PHI_INF = -10. In phase V there are no filled channels and Phi = Phi_vac does not
depend on T, so the values are taken from 82_stab_minphi_<NAME>.npz.

    python notebooks/83_stab_minphi_T0.py   (after 82_stab_minphi.py for 'full' and 'peaks')
→ output/83_stab_minphi_T0_<NAME>.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm

from njl3.stab.formulas import singular_q1

OUT = os.path.join(os.path.dirname(__file__), 'output')
PHI_INF = -10.0


def main(name):
    d = np.load(os.path.join(OUT, f'82_stab_minphi_{name}.npz'))
    mus, Ls, phase, M = d['mu'], d['L'], d['phase'].astype(str), d['M']
    v = d['phi_min'].copy()
    for i, j in zip(*np.where(M > 0)):
        if any(0.0 < x <= 10.0 for x in singular_q1(M[i, j], mus[i], Ls[j], 0)):
            v[i, j] = PHI_INF
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_facecolor('#e6e6e6')
    fin = np.where(v > PHI_INF, v, np.nan)
    hi = min(1.0, np.nanmax(fin))
    im = ax.pcolormesh(Ls, mus, fin, cmap='Blues', vmin=0.0, vmax=hi, shading='auto', rasterized=True)
    ax.pcolormesh(Ls, mus, np.where(v == PHI_INF, 1.0, np.nan), cmap=ListedColormap(['#b2182b']),
                  shading='auto', rasterized=True)
    code = np.zeros(phase.shape)
    code[np.char.startswith(phase, 'I')] = 1
    code[phase == 'T'] = 2
    ax.contour(Ls, mus, code, levels=[0.5, 1.5], colors='k', linewidths=0.7)
    fig.colorbar(im, ax=ax, label='$\\min\\Phi$ (фаза V)', extend='max' if np.nanmax(fin) > hi else 'neither')
    ax.set_xlabel('$L$'); ax.set_ylabel('$\\mu$')
    ax.set_title('$T=0$: минимум $\\Phi$ по $q_1\\in(0,10]$ и $k$\n'
                 f'красное — $-\\infty$ (логарифм у $2k_F$, принято $\\Phi={PHI_INF:g}$); серое — $M^*=0$',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'83_stab_minphi_T0_{name}.png'), dpi=150)
    np.savez(os.path.join(OUT, f'83_stab_minphi_T0_{name}.npz'), mu=mus, L=Ls, phase=phase, phi_min=v)


if __name__ == '__main__':
    for name in ('full', 'peaks'):
        main(name)

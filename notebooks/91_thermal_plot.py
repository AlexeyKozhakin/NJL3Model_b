"""
Picture of a map at temperature T (web/lab/data/inhT/maps_T<T>_phi<φ>_g<g>_<region>.json, notebooks/90_inh_maps.py
--thermal) next to the T = 0 map with the survival threshold (web/lab/data/inh/...).

    python notebooks/91_thermal_plot.py T PHI G REGION   → notebooks/output/91_thermal_T<T>_phi<φ>_g<g>_<region>.png
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
T, PHI, G, REGION = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
COLORS = ['#2166ac', '#67a9cf', '#ef8a62', '#e3e2dd']
LABELS = ['V', 'I$_j$ (b = 0)', 'B$_n$ (спираль)', 'T (M* = 0)']


def load(path):
    d = json.load(open(path))
    ph = np.array(d['phase'])
    lnM = np.array([[np.nan if x is None else x for x in r] for r in d['lnM']], float)
    return np.array(d['mu']), np.array(d['L']), ph, lnM


def code(ph):
    c = np.full(ph.shape, 3.0)
    c[ph == 'V'] = 0
    c[np.char.startswith(ph, 'I')] = 1
    c[np.char.startswith(ph, 'B')] = 2
    return c


def main():
    new = os.path.join(ROOT, 'web', 'lab', 'data', 'inhT', f'maps_T{T:g}_phi{PHI:g}_g{G:g}_{REGION}.json')
    old = os.path.join(ROOT, 'web', 'lab', 'data', 'inh', f'maps_phi{PHI:g}_g{G:g}_{REGION}.json')
    mu, L, ph, lnM = load(new)
    mu0, L0, ph0, _ = load(old)
    cm = ListedColormap(COLORS)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].pcolormesh(L0, mu0, code(ph0).T, cmap=cm, vmin=-0.5, vmax=3.5, shading='nearest', rasterized=True)
    ax[0].set_title('прежняя: T = 0 + порог выживания щелей $T_c=0{,}567\\,(S_n/\\mu)M^*$, T = 10$^{-3}$', fontsize=9)
    ax[1].pcolormesh(L, mu, code(ph).T, cmap=cm, vmin=-0.5, vmax=3.5, shading='nearest', rasterized=True)
    ax[1].set_title(f'новая: минимум $\\Omega_T(M,b)$ при T = {T:g} (формула A)', fontsize=9)
    lg = np.where(np.isfinite(lnM), lnM / np.log(10), np.nan)
    im = ax[2].pcolormesh(L, mu, lg.T, cmap='viridis', vmin=-3, vmax=0.8, shading='nearest', rasterized=True)
    ax[2].set_facecolor(COLORS[3])
    fig.colorbar(im, ax=ax[2], label='$\\log_{10} M^*$')
    ax[2].set_title(f'$\\log_{{10}}M^*$ при T = {T:g} (серое — M* = 0)', fontsize=9)
    for a in ax:
        a.set_xlabel('$L$'); a.set_ylabel('$\\mu$')
    fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)], loc='upper center',
               ncol=4, fontsize=9)
    n = lambda p, k: int((code(p) == k).sum())
    fig.suptitle(f'φ = {PHI:g}, g = {G:g}, {REGION}; точек B: было {n(ph0, 2)}, стало {n(ph, 2)}; '
                 f'точек T: было {n(ph0, 3)}, стало {n(ph, 3)}', y=0.005, fontsize=9, va='bottom')
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = os.path.join(ROOT, 'notebooks', 'output', f'91_thermal_T{T:g}_phi{PHI:g}_g{G:g}_{REGION}.png')
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main()

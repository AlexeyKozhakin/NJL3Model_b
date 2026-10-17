"""
Статья 2, шаг 3: структура Omega(M, b) в точках с b* > 0 (g = -1, phi = 0).

Для нескольких (mu, L): карта Omega(M, b) − Omega(0, 0) на плотной сетке, глобальный
минимум на сетке, профиль Omega_eff(b) = min_M Omega(M, b) и увеличение около
порогов b = S_n = sqrt(mu² − (2πn/L)²).

Рисунки: notebooks/output/51_maps.png, 51_profiles.png
Запуск:  python notebooks/51_explore_mb.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full import formulas as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
POINTS = [(3.0, 1.0), (3.5, 1.0), (4.0, 2.0), (3.3, 0.5)]


def omega_grid(Mg, bg, mu, L):
    OL = F.omega_L_vec(Mg, bg, L) - F.omega_L_vec(np.zeros_like(bg), bg, L)
    dU = np.vectorize(F.dU)(Mg, bg)
    OmuL = np.vectorize(lambda M, b: F.omega_muL(M, b, mu, L) - F.omega_muL(0.0, b, mu, L))(Mg, bg)
    out = Mg**2 / (2 * G) + dU + OL + OmuL
    return np.where(Mg == 0, 0.0, out)


def thresholds(mu, L):
    nmax = int(math.floor(L * mu / (2 * math.pi)))
    return [math.sqrt(mu * mu - (2 * math.pi * n / L)**2) for n in range(nmax + 1)]


def main():
    figm, axm = plt.subplots(1, len(POINTS), figsize=(4.8 * len(POINTS), 4.2))
    figp, axp = plt.subplots(2, len(POINTS), figsize=(4.8 * len(POINTS), 7.5))
    for c, (mu, L) in enumerate(POINTS):
        t = time.time()
        M = np.linspace(0, 4.0, 161)
        b = np.linspace(0, mu + 1.0, 241)
        Mg, bg = np.meshgrid(M, b)
        Om = omega_grid(Mg, bg, mu, L)
        i, j = np.unravel_index(np.argmin(Om), Om.shape)
        S = thresholds(mu, L)
        print(f"mu={mu}, L={L}: {time.time() - t:.1f} с; минимум на сетке b*={b[i]:.3f}, M*={M[j]:.3f}, "
              f"ΔΩ={Om[i, j]:+.5f}; пороги S_n = {[round(x, 4) for x in S]}")

        ax = axm[c]
        lim = np.percentile(np.abs(Om), 50)
        im = ax.pcolormesh(M, b, Om, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
        ax.plot(M[j], b[i], 'k*', ms=12)
        for s in S:
            ax.axhline(s, color='k', lw=0.6, ls=':')
        ax.set_title(f'μ={mu}, L={L}'); ax.set_xlabel('M'); ax.set_ylabel('b')
        figm.colorbar(im, ax=ax)

        # профиль по b и увеличение около ближайшего к b* порога
        prof = Om.min(axis=1)
        axp[0, c].plot(b, prof, 'C0-')
        for s in S:
            axp[0, c].axvline(s, color='k', lw=0.6, ls=':')
        axp[0, c].set_title(f'min_M Ω(M, b), μ={mu}, L={L}'); axp[0, c].set_xlabel('b'); axp[0, c].grid(alpha=0.3)

        s0 = min(S, key=lambda x: abs(x - b[i]))
        bz = np.linspace(s0 - 0.3, s0 + 0.15, 181)
        Mz = np.linspace(max(M[j] - 0.6, 0.0), M[j] + 0.6, 121)
        Mzg, bzg = np.meshgrid(Mz, bz)
        Oz = omega_grid(Mzg, bzg, mu, L)
        iz = np.argmin(Oz, axis=1)
        axp[1, c].plot(bz, Oz.min(axis=1), 'C0-', label='min_M Ω')
        axp[1, c].axvline(s0, color='k', lw=0.8, ls=':', label=f'S = {s0:.4f}')
        ax2 = axp[1, c].twinx()
        ax2.plot(bz, Mz[iz], 'C1-', lw=1, label='M*(b)')
        ax2.set_ylabel('M*(b)', color='C1')
        axp[1, c].set_xlabel('b'); axp[1, c].grid(alpha=0.3); axp[1, c].legend(fontsize=7, loc='lower left')
        ib = int(np.argmin(Oz.min(axis=1)))
        print(f"    увеличение около S={s0:.4f}: b*={bz[ib]:.4f} (b* − S = {bz[ib] - s0:+.4f}), M*={Mz[iz[ib]]:.4f}")

    figm.tight_layout(); figm.savefig(os.path.join(OUT, '51_maps.png'), dpi=110)
    figp.tight_layout(); figp.savefig(os.path.join(OUT, '51_profiles.png'), dpi=110)


if __name__ == '__main__':
    main()

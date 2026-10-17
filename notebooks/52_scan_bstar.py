"""
Статья 2, шаг 3б: где лежит глобальный минимум по (M, b) на всей диаграмме (g = -1).

Для L из набора и mu ∈ [0.5, 7]: минимум на сетке (M, b) (шаг 0.05) + локальное
уточнение (L-BFGS-B по непрерывной Omega), сравнение с точным решением при b = 0
(njl3.b0). Для b* > 0 — ближайший порог S_n = sqrt(mu² − (2πn/L)²) и b* − S_n.

Проверяемые гипотезы:
    H1: каждый минимум с b* > 0 лежит у одного из порогов S_n (|b* − S_n| мало);
    H2: какой именно порог — старший (n = n_max) или любой.

Выход: таблица в stdout, notebooks/output/52_bstar.png, 52_data.npz
Запуск:  python notebooks/52_scan_bstar.py
"""

import math
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full import formulas as F
from njl3.b0 import find_minimum as b0_min

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
L_SET = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
MU = np.round(np.arange(0.5, 7.0001, 0.1), 4)
M_GRID = np.linspace(0.0, 4.5, 91)
B_GRID = np.linspace(0.0, 8.0, 161)


def thresholds(mu, L):
    nmax = int(math.floor(L * mu / (2 * math.pi)))
    return [(n, math.sqrt(mu * mu - (2 * math.pi * n / L)**2)) for n in range(nmax + 1)]


def main():
    Mg, bg = np.meshgrid(M_GRID, B_GRID)
    rows = []
    t0 = time.time()
    for L in L_SET:
        OL0 = F.omega_L_vec(Mg, bg, L) - F.omega_L_vec(np.zeros_like(bg), bg, L)
        dUg = np.vectorize(F.dU)(Mg, bg)
        for mu in MU:
            Om = Mg**2 / (2 * G) + dUg + OL0 + F.omega_muL_vec(Mg, bg, mu, L) - F.omega_muL_vec(np.zeros_like(bg), bg, mu, L)
            Om = np.where(Mg == 0, 0.0, Om)
            i, j = np.unravel_index(np.argmin(Om), Om.shape)
            Mb, Ob, _ = b0_min(mu, L, G)                       # точное решение при b = 0
            rec = dict(L=L, mu=mu, M=0.0, b=np.nan, Om=0.0, kind='T', n=-1, dS=np.nan, nmax=len(thresholds(mu, L)) - 1)
            if Ob < rec['Om']:
                rec.update(M=Mb, b=0.0, Om=Ob, kind='b0')
            if Om[i, j] < -1e-10 and B_GRID[i] > 0:
                f = lambda x: F.omega_rel(x[0], x[1], mu, L, G)
                r = minimize(f, [M_GRID[j], B_GRID[i]], method='L-BFGS-B',
                             bounds=[(1e-6, 6), (0, 10)], options=dict(ftol=1e-15, gtol=1e-11))
                if r.fun < rec['Om'] - 1e-12 and r.x[1] > 1e-6:
                    S = thresholds(mu, L)
                    n, s = min(S, key=lambda q: abs(q[1] - r.x[1]))
                    rec.update(M=r.x[0], b=r.x[1], Om=r.fun, kind='b>0', n=n, dS=r.x[1] - s)
            rows.append(rec)
        print(f"L={L}: {time.time() - t0:.0f} с")

    kinds = [r['kind'] for r in rows]
    print(f"\nВсего точек: {len(rows)}; T: {kinds.count('T')}, b=0 (M>0): {kinds.count('b0')}, b>0: {kinds.count('b>0')}")
    bp = [r for r in rows if r['kind'] == 'b>0']
    if bp:
        dS = np.array([r['dS'] for r in bp])
        top = sum(r['n'] == r['nmax'] for r in bp)
        print(f"b* − S_n: min {dS.min():+.4f}, max {dS.max():+.4f}, медиана {np.median(dS):+.4f}")
        print(f"порог старший (n = n_max): {top} из {len(bp)}")
        print("Примеры (L, mu, M*, b*, n, b*−S_n, ΔΩ):")
        for r in bp[:: max(1, len(bp) // 15)]:
            print(f"   {r['L']:4.1f} {r['mu']:4.1f}  M*={r['M']:.5f}  b*={r['b']:.5f}  n={r['n']}/{r['nmax']}  "
                  f"{r['dS']:+.4f}  {r['Om']:+.5f}")

    np.savez(os.path.join(OUT, '52_data.npz'), rows=np.array([[r['L'], r['mu'], r['M'], r['b'], r['Om'], r['n'], r['dS']]
                                                            for r in rows]))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, L in zip(axes.ravel(), L_SET):
        rr = [r for r in rows if r['L'] == L]
        mu = np.array([r['mu'] for r in rr])
        ax.plot(mu, [r['M'] for r in rr], 'C1.-', ms=3, label='M*')
        ax.plot(mu, [r['b'] if r['kind'] == 'b>0' else np.nan for r in rr], 'C0.-', ms=3, label='b*')
        for n in range(0, 12):
            k = 2 * math.pi * n / L
            m = mu[mu > k]
            ax.plot(m, np.sqrt(m**2 - k**2), color='gray', lw=0.6, ls=':')
        ax.set_title(f'L = {L}'); ax.set_ylim(0, 5); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('μ')
    fig.suptitle('Глобальный минимум по (M, b), g = −1; пунктир — пороги S_n(μ)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '52_bstar.png'), dpi=110)


if __name__ == '__main__':
    main()

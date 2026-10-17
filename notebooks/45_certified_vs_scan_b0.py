"""
Задача B1 (статья 1): гарантированный поиск корней G против сканирования.

Сравнивает на сетке (mu, L):
    scan      — find_minimum_b0: скан G по сетке M + Брент (эвристика)
    certified — find_minimum_b0_certified: branch-and-bound по монотонной вилке
                1/g + p/pi + D(q) ≤ G ≤ 1/g + q/pi + D(p)  (гарантия полноты)

Считает: где различается число локальных минимумов, где различается глобальный
минимум, время. Рисунок: notebooks/output/45_scan_misses.png

Запуск:  python notebooks/45_certified_vs_scan_b0.py [--quick]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import analytic_b0 as an

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
QUICK = '--quick' in sys.argv
G = -1.0

MU = np.linspace(0.05, 7.0, 70 if QUICK else 279)
L  = np.linspace(0.1, 10.0, 50 if QUICK else 250)


def main():
    shape = (len(L), len(MU))
    n_scan = np.zeros(shape, dtype=int)
    n_cert = np.zeros(shape, dtype=int)
    dM = np.zeros(shape)
    dOm = np.zeros(shape)
    t_scan = t_cert = 0.0
    for i, l in enumerate(L):
        for j, mu in enumerate(MU):
            t = time.perf_counter()
            Ms, Os, ms = an.find_minimum_b0(mu, l, G)
            t_scan += time.perf_counter() - t
            t = time.perf_counter()
            Mc, Oc, mc = an.find_minimum_b0_certified(mu, l, G)
            t_cert += time.perf_counter() - t
            n_scan[i, j], n_cert[i, j] = len(ms), len(mc)
            dM[i, j], dOm[i, j] = Ms - Mc, Os - Oc

    N = n_scan.size
    miss = n_cert > n_scan
    extra = n_scan > n_cert
    wrong = np.abs(dOm) > 1e-12
    print(f"Сетка {len(MU)}×{len(L)} = {N} точек")
    print(f"Время: scan {t_scan:.1f}s ({t_scan / N * 1e3:.2f} мс/точка), "
          f"certified {t_cert:.1f}s ({t_cert / N * 1e3:.2f} мс/точка)")
    print(f"Скан пропустил локальные минимумы: {miss.sum()} точек ({100 * miss.mean():.2f}%)")
    print(f"Скан нашёл лишние минимумы:        {extra.sum()} точек")
    print(f"Глобальный минимум различается (|ΔΩ| > 1e-12): {wrong.sum()} точек; "
          f"max |ΔΩ| = {np.abs(dOm).max():.2e}, max |ΔM*| = {np.abs(dM).max():.2e}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.pcolormesh(L, MU, (n_cert - n_scan).T, cmap='Reds', vmin=0, vmax=2, shading='auto')
    ax.contour(L, MU, wrong.T.astype(float), levels=[0.5], colors='b')
    ax.set_xlabel('L'); ax.set_ylabel('μ')
    ax.set_title('Красное — скан пропустил локальный минимум;\nсиний контур — неверный глобальный минимум')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '45_scan_misses.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()

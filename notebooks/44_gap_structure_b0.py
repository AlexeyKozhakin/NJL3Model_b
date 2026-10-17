"""
Задача A5 (статья 1): структура G(M) при b = 0 между порогами мод.

Пороги S_n = sqrt(mu² - k_n²) делят (0, mu) на интервалы с фиксированным набором
заполненных мод. При M > mu мод нет и G строго возрастает (доказано: все слагаемые
возрастают) — не более одного корня (вакуум).

Гипотезы для интервалов внутри (0, mu):
    H1: у G не больше одного экстремума на интервале  (G' меняет знак ≤ 1 раза)
    H2: G вогнута на интервале                        (G'' < 0)
Из H1 следует: не больше двух корней G на интервале — основа доказуемо полного
поиска минимума (B1). G' не зависит от g, поэтому H1, H2 верны для любого g.

Проверка: плотная выборка по каждому интервалу (сгущение к концам), сетка
mu ∈ [0.1, 7], L ∈ [0.05, 10].

Вывод: таблицы в stdout, рисунки notebooks/output/44_*.png

Запуск:  python notebooks/44_gap_structure_b0.py [--quick]
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

MU = np.linspace(0.1, 7.0, 40 if QUICK else 140)
L  = np.geomspace(0.05, 10.0, 30 if QUICK else 120)
N_S = 2000 if QUICK else 6000     # точек на интервал
G_COUPLING = -1.0


def interval_samples(a, b, n):
    """Точки внутри (a, b), сгущённые к обоим концам (чебышёвские + логарифмические)."""
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    cheb = a + (b - a) * (1.0 - np.cos(np.pi * t)) / 2.0
    w = b - a
    edge = np.geomspace(1e-12 * w, 1e-3 * w, 60)
    pts = np.concatenate([cheb, a + edge, b - edge])
    pts = pts[(pts > a) & (pts < b)]
    if a == 0.0:
        pts = pts[pts > 1e-12]
    return np.unique(pts)


def sign_changes(y):
    s = np.sign(y)
    s = s[s != 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))


def main():
    t0 = time.time()
    worst_crit, worst_conc = (0, None), (0, None)
    n_intervals = 0
    hist_crit = {}
    hist_roots = {}
    conc_fail = 0
    max_crit_map = np.zeros((len(L), len(MU)), dtype=int)

    for i, l in enumerate(L):
        for j, mu in enumerate(MU):
            edges = [0.0] + an.thresholds(mu, l)       # 0 < S_nmax < ... < S_0 = mu
            for a, b in zip(edges[:-1], edges[1:]):
                if b - a < 1e-9:
                    continue
                M = interval_samples(a, b, N_S)
                dG = an.gap_dG_vec(M, mu, l)
                G  = an.gap_G_vec(M, mu, l, G_COUPLING)
                nc = sign_changes(dG)
                nr = sign_changes(G)
                hist_crit[nc] = hist_crit.get(nc, 0) + 1
                hist_roots[nr] = hist_roots.get(nr, 0) + 1
                max_crit_map[i, j] = max(max_crit_map[i, j], nc)
                n_intervals += 1
                if nc > worst_crit[0]:
                    worst_crit = (nc, (mu, l, a, b))
                # H2: G' монотонно убывает (вогнутость G), допуск на округление
                d2 = np.diff(dG)
                bad = np.count_nonzero(d2 > 1e-12 * (1 + np.abs(dG[:-1])))
                if bad:
                    conc_fail += 1
                    if bad > worst_conc[0]:
                        worst_conc = (bad, (mu, l, a, b))

    print(f"Проверено {n_intervals} интервалов на сетке {len(MU)}×{len(L)} за {time.time() - t0:.1f}s")
    print(f"H1: число экстремумов G на интервале -> число интервалов: {dict(sorted(hist_crit.items()))}")
    print(f"    худший случай: {worst_crit}")
    print(f"    число корней G (g={G_COUPLING:+.0f}) на интервале -> число интервалов: "
          f"{dict(sorted(hist_roots.items()))}")
    print(f"H2: интервалов, где G не вогнута: {conc_fail}; худший: {worst_conc}")

    np.savez(os.path.join(OUT_DIR, '44_data.npz'), mu=MU, L=L, max_crit=max_crit_map)

    # Иллюстрация: G(M) с порогами для нескольких (mu, L)
    cases = [(3.2, 1.0), (3.5, 2.0), (3.3, 3.73), (3.2, 7.8), (6.0, 4.0)]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.8))
    for ax, (mu, l) in zip(axes, cases):
        M = np.linspace(1e-4, 1.4 * max(mu, 3.3), 4000)
        ax.plot(M, an.gap_G_vec(M, mu, l, G_COUPLING), 'k-', lw=1.5)
        for s in an.thresholds(mu, l):
            ax.axvline(s, color='C1', lw=0.8, ls='--')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f'μ={mu}, L={l}')
        ax.set_xlabel('M')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('G(M),  g = −1')
    fig.suptitle('G(M) при b = 0; пунктир — пороги мод S_n (последний — μ)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '44_G_examples.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()

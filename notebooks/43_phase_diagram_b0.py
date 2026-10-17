"""
Фазовая диаграмма при b = 0 (g = -1, phi = 0) по замкнутым формулам.

Для каждой точки (mu, L) глобальный минимум Omega(M) ищется без численных
интегралов: кандидаты M = 0 и корни уравнения щели G(M) = 0
(functions/analytic_b0.py: find_minimum_b0).

Рисунки (notebooks/output/), <tag> — суффикс диапазона (пусто по умолчанию):
    43_phase_diagram_b0<tag>.png      — M*(mu, L) с линиями переходов (mu по x)
    43_phase_diagram_b0<tag>_Lmu.png  — то же, L по x, mu по y
    43_b0_vs_full.png                 — только для диапазона по умолчанию: срезы
                                        L = 0.5, 1, 2, b = 0 против минимума по (b, M)
                                        из разведки 41
Данные: notebooks/output/43_data<tag>.npz

Запуск:
    python notebooks/43_phase_diagram_b0.py                       # mu ∈ [0.05, 7], L ∈ [0.1, 4]
    python notebooks/43_phase_diagram_b0.py --mu 2.5 4 --L 0.05 4 --n 300 300 --tag _zoom1
    python notebooks/43_phase_diagram_b0.py --plot-only [--tag _zoom1] [--vmax 4]
L = 0 — особая точка (Omega_L ~ 1/L³, M_vac ~ 1/L), поэтому L > 0.
"""

import argparse
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
os.makedirs(OUT_DIR, exist_ok=True)

G = -1.0


def compute(MU, L):
    shape = (len(L), len(MU))
    M_star  = np.zeros(shape)          # глобальный минимум
    Om_star = np.zeros(shape)
    Om_nt   = np.full(shape, np.nan)   # лучший нетривиальный минимум (M > 0)
    n_min   = np.zeros(shape, dtype=int)
    G0      = np.zeros(shape)
    for i, l in enumerate(L):
        for j, mu in enumerate(MU):
            M, Om, minima = an.find_minimum_b0(mu, l, G)
            M_star[i, j], Om_star[i, j] = M, Om
            n_min[i, j] = len(minima)
            if minima:
                Om_nt[i, j] = min(om for _, om in minima)
            G0[i, j] = an.gap_G0(mu, l, G)
    return dict(mu=MU, L=L, M_star=M_star, Om_star=Om_star, Om_nt=Om_nt, n_min=n_min, G0=G0)


def plot_diagram(d, swap, tag, vmax=None):
    """swap=False: mu по x, L по y; swap=True: L по x, mu по y. vmax — верх шкалы M*."""
    MU, L = d['mu'], d['L']
    X, Y = (L, MU) if swap else (MU, L)
    T = (lambda A: A.T) if swap else (lambda A: A)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.pcolormesh(X, Y, T(d['M_star']), cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label='M*', extend='max' if vmax else 'neither')
    # Метастабильность: M = 0 локально устойчив (G0 > 0) и есть нетривиальный минимум
    meta = (d['G0'] > 0) & (d['n_min'] > 0)
    ax.contourf(X, Y, T(meta.astype(float)), levels=[0.5, 1.5], colors='none', hatches=['//'])
    ax.contour(X, Y, T(np.nan_to_num(d['Om_nt'], nan=1.0)), levels=[0], colors='w', linewidths=2)
    ax.contour(X, Y, T(d['G0']), levels=[0], colors='r', linewidths=1.5, linestyles='--')
    # Вакуумная ветвь M* > mu (ферми-шар пуст, Omega_muL = 0) против ветви M* < mu
    MUg = MU[None, :] * np.ones((len(L), 1))
    vac_branch = ((d['M_star'] > 0) & (d['M_star'] >= MUg)).astype(float)
    ax.contour(X, Y, T(vac_branch), levels=[0.5], colors='orange', linewidths=2, linestyles=':')
    ax.plot([], [], ':', color='orange', lw=2, label='скачок вакуум (M* > μ) → M* < μ')
    ax.plot([], [], 'w-', lw=2, label='переход: Ω(M*) = Ω(0)')
    ax.plot([], [], 'r--', label='G(0) = 0: граница устойчивости M = 0')
    ax.fill([], [], hatch='//', fill=False, label='метастабильность (два минимума)')
    ax.set_xlabel('L' if swap else 'μ')
    ax.set_ylabel('μ' if swap else 'L')
    ax.set_title(f'Фазовая диаграмма при b = 0, g = −1:  '
                 f'μ ∈ [{MU[0]:g}, {MU[-1]:g}],  L ∈ [{L[0]:g}, {L[-1]:g}]')
    ax.legend(loc='best', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    name = f'43_phase_diagram_b0{tag}' + ('_Lmu' if swap else '') + '.png'
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130)
    plt.close(fig)
    print(f"    {name}")


def plot_vs_full():
    """b = 0 против минимума по (b, M) из разведки 41 (срезы L = 0.5, 1, 2)."""
    path = os.path.join(OUT_DIR, '41_data.npz')
    if not os.path.exists(path):
        print("Нет 41_data.npz — рисунок сравнения пропущен")
        return
    d = np.load(path)
    mu41 = d['mu']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for c, l in enumerate([0.5, 1.0, 2.0]):
        res = [an.find_minimum_b0(mu, l, G) for mu in mu41]
        M_b0  = np.array([r[0] for r in res])
        Om_b0 = np.array([r[1] for r in res])
        M_full, b_full, Om_full = (d[f'min_{k}_g-1_L{l}'] for k in ('M', 'b', 'Om'))

        ax = axes[0, c]
        ax.plot(mu41, M_b0, 'k-', lw=2, label='M*, только b = 0 (формулы)')
        ax.plot(mu41, M_full, 'C1.', ms=4, label='M*, минимум по (b, M) (сетка)')
        ax.plot(mu41, b_full, 'C0.', ms=4, label='b*, минимум по (b, M) (сетка)')
        ax.set_title(f'L = {l}')
        ax.grid(alpha=0.3)
        ax = axes[1, c]
        ax.plot(mu41, Om_b0 - Om_full, 'k-')
        ax.fill_between(mu41, 0, Om_b0 - Om_full, where=(Om_b0 - Om_full) > 1e-3, color='C0', alpha=0.3,
                        label='фаза с b > 0 глубже')
        ax.set_xlabel('μ')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_ylabel(r'$\Omega_{b=0}^{min}-\Omega_{(b,M)}^{min}$')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('b = 0 против минимизации по (b, M), g = −1')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '43_b0_vs_full.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', nargs=2, type=float, default=[0.05, 7.0])
    ap.add_argument('--L', nargs=2, type=float, default=[0.1, 4.0])
    ap.add_argument('--n', nargs=2, type=int, default=[279, 196], metavar=('N_MU', 'N_L'))
    ap.add_argument('--tag', default='')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--vmax', type=float, default=None, help='верх цветовой шкалы M*')
    args = ap.parse_args()
    if args.L[0] <= 0:
        ap.error('L должно быть > 0 (L = 0 — особая точка)')

    data_path = os.path.join(OUT_DIR, f'43_data{args.tag}.npz')
    if args.plot_only:
        d = dict(np.load(data_path))
    else:
        MU = np.linspace(*args.mu, args.n[0])
        L  = np.linspace(*args.L, args.n[1])
        t0 = time.time()
        d = compute(MU, L)
        print(f"Диаграмма {len(MU)}×{len(L)} точек: {time.time() - t0:.1f}s")
        np.savez(data_path, **d)

    for swap in (False, True):
        plot_diagram(d, swap, args.tag, args.vmax)
    if args.tag == '' and not args.plot_only:
        plot_vs_full()


if __name__ == '__main__':
    main()

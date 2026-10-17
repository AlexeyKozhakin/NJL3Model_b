"""
Задача 1.9 (часть b = 0): проверка замкнутых формул functions/analytic_b0.py
против численного кода (functions/*_fast.py, functions/Omega_mu_L.py).

Проверки:
    V1. Каждый вклад dU, Omega_L, Omega_muL при b = 0 — формула vs код.
    V2. Коэффициент a = G(0)/2 vs численная карта a(mu, b=0) из эксперимента 41.
    V3. Граница a = 0 при b = 0: mu_c(L) из формулы vs контур на карте R3.
    V4. Вакуумная масса M_vac(L) (mu = 0) vs разведка 41: 3.8 / 3.2 / 3.1.
    V5. G(M) при малых M: формула vs конечные разности по коду (эксп. 40).

Рисунок: notebooks/output/42_V5_gap_analytic_vs_numeric.png

Запуск:  python notebooks/42_verify_analytic_b0.py
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import analytic_b0 as an
from functions.dU_fast import dU_scalar
from functions.Omega_L_fast import Omega_L_scalar
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
L_VALS = [0.5, 1.0, 2.0]


def v1_components():
    print("V1. Вклады при b = 0: формула vs код (N2d = 640, N1d = 25600)")
    print(f"    {'L':>4} {'mu':>4} {'M':>5} | {'dU: an':>11} {'код':>11} | "
          f"{'Omega_L: an':>12} {'код':>12} | {'Omega_muL: an':>13} {'код':>13}")
    worst = 0.0
    for L in L_VALS:
        for mu in [1.5, 3.2, 5.0]:
            for M in [0.0, 0.3, 1.0, 2.5]:
                c_dU = dU_scalar(0.0, M, 640, 640) - dU_scalar(0.0, 0.0, 640, 640)
                c_L  = Omega_L_scalar(L, 0.0, M, 0.0, 640, 640)
                c_mu = fun_Omega_L_mu_int(mu, L, 0.0, M, 0, 25600)
                a_dU, a_L, a_mu = an.dU(M), an.omega_L(M, L), an.omega_muL(M, mu, L)
                worst = max(worst, abs(c_dU - a_dU), abs(c_L - a_L), abs(c_mu - a_mu))
                if mu == 3.2:
                    print(f"    {L:4.1f} {mu:4.1f} {M:5.2f} | {a_dU:11.6f} {c_dU:11.6f} | "
                          f"{a_L:12.6f} {c_L:12.6f} | {a_mu:13.6f} {c_mu:13.6f}")
    print(f"    Максимальное |формула - код| по всем 36 точкам: {worst:.1e}\n")


def v2_v3_coefficient():
    path = os.path.join(OUT_DIR, '41_data.npz')
    if not os.path.exists(path):
        print("V2. Пропущено: нет 41_data.npz (запустите notebooks/41_*.py)\n")
        return
    d = np.load(path)
    mu_a = d['a_mu']
    print("V2. a(mu, b=0) = G(0)/2: формула vs численная карта (M_A = 0.03)")
    for L in L_VALS:
        a_num = d[f'a_g-1_L{L}'][:, 0]
        a_an = np.array([an.gap_G0(mu, L, G) / 2.0 for mu in mu_a])
        print(f"    L={L}: max |Δa| = {np.abs(a_num - a_an).max():.1e}  "
              f"(диапазон a: [{a_an.min():+.3f}, {a_an.max():+.3f}])")

    print("\nV3. Граница a = 0 при b = 0 (M = 0 перестаёт быть «горкой»)")
    for L in L_VALS:
        f = lambda mu: an.gap_G0(mu, L, G)
        mus = np.linspace(0.3, 6.0, 2000)
        vals = np.array([f(m) for m in mus])
        roots = [brentq(f, mus[i], mus[i + 1]) for i in range(len(mus) - 1)
                 if vals[i] * vals[i + 1] < 0]
        hint = "e^{πL/2}/(2L) = %.3f" % (np.exp(np.pi * L / 2) / (2 * L))
        print(f"    L={L}: mu_c = {', '.join(f'{r:.3f}' for r in roots)}   (только n=0: {hint})")
    print()


def v4_vacuum():
    print("V4. Вакуумная масса M_vac(L) при mu = 0, b = 0")
    for L in L_VALS:
        r = minimize_scalar(lambda M: an.omega_ren_rel(M, 0.0, L, G), bounds=(0.01, 6.0),
                            method='bounded', options={'xatol': 1e-10})
        print(f"    L={L}: M_vac = {r.x:.4f},  Omega - Omega(0) = {r.fun:+.4f}")
    print(f"    L→∞: M_vac → π = {np.pi:.4f}\n")


def v5_gap_plot():
    mu, L = 3.2, 1.0
    M = np.logspace(-4, 0, 41)
    G_an = np.array([an.gap_G(m, mu, L, G) for m in M])
    print(f"V5. G(0⁺) = {an.gap_G0(mu, L, G):.6f};  G(1e-4) = {G_an[0]:.6f}  (μ=3.2, L=1)")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    path = os.path.join(OUT_DIR, '40_data.npz')
    if os.path.exists(path):
        d = np.load(path)
        for lev, ls in [(2, ':'), (3, '-')]:
            c = d[f'comps_b0.0_lev{lev}']
            delta = M**2 / (2 * G) + c.sum(axis=1)
            ax.semilogx(M, np.gradient(delta, M) / M, ls, color='gray',
                        label=f'код, N2d={d["N_levels"][lev][0]}')
    ax.semilogx(M, G_an, 'k-', lw=2, label='формула')
    ax.axhline(an.gap_G0(mu, L, G), color='C3', ls='--', lw=1, label='G(0⁺)')
    ax.set_xlabel('M'); ax.set_ylabel(r'$G(M)=\Omega^\prime(M)/M$')
    ax.set_title(f'V5: уравнение щели при b=0, μ={mu}, L={L}, g={G:+.0f}')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '42_V5_gap_analytic_vs_numeric.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    v1_components()
    v2_v3_coefficient()
    v4_vacuum()
    v5_gap_plot()

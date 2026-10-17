"""
Эксперимент 1.8: поведение Omega_ren при малых M (phi = 0).

Дизайн и вопросы E1–E5: publication/notes/2026-10-16_exp-1.8-small-M.md

При фиксированных (mu, L, b) считается разность по каждой компоненте
    Delta_X(M) = X(b, M) - X(b, 0),   X ∈ {dU, Omega_L, Omega_muL}
(вычитаемое X(0,0) из перенормировки сокращается) и
    Delta(M) = M²/(2g) + Delta_dU + Delta_L + Delta_muL = Omega_ren(M) - Omega_ren(0).

Графики (notebooks/output/):
    40_E1_components.png   — Delta_X / M² vs M (лог-шкала): форма у нуля, вклад компонент
    40_E3_gap.png          — G(M) = (dDelta/dM) / M для нескольких b
    40_E4_transition.png   — Omega(M) - Omega(0) через переход по mu (b = 0 и min по b)
    40_E5_convergence.png  — сходимость Delta / M² по числу узлов

Сырые данные: notebooks/output/40_data.npz

Запуск:  python notebooks/40_small_M_asymptotics.py [--quick]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.dU_fast import dU_scalar
from functions.Omega_L_fast import Omega_L_scalar
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

QUICK = '--quick' in sys.argv

# ==============================================================================
# Параметры эксперимента
# ==============================================================================

G_COUPLING = -1.0
L_VAL      = 1.0
MU_REF     = 3.2                  # точка около перехода (docs/asymptotics.html)
MU_SCAN    = [2.8, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6]
B_VALS     = [0.0, 0.5, 1.0]

M_LOG = np.logspace(-4, 0, 21 if QUICK else 41)

# Число узлов: (N для 2D-интегралов dU и Omega_L, N для 1D-интеграла Omega_muL)
N_LEVELS = [(80, 400), (160, 1600)] if QUICK else \
           [(80, 400), (160, 1600), (320, 6400), (640, 25600)]


# ==============================================================================
# Компоненты
# ==============================================================================

def components(M, b, mu, L, n2d, n1d):
    """Delta_X(M) = X(b,M) - X(b,0) для dU, Omega_L, Omega_muL."""
    d_dU = dU_scalar(b, M, n2d, n2d) - dU_scalar(b, 0.0, n2d, n2d)
    d_L  = Omega_L_scalar(L, b, M, 0.0, n2d, n2d) - Omega_L_scalar(L, b, 0.0, 0.0, n2d, n2d)
    d_mu = fun_Omega_L_mu_int(mu, L, b, M, 0, n1d) - fun_Omega_L_mu_int(mu, L, b, 0.0, 0, n1d)
    return d_dU, d_L, d_mu


def scan_components(M_vals, b, mu, L, n2d, n1d):
    """Массив (len(M_vals), 3): компоненты по M."""
    # X(b, 0) не зависит от M — считаем один раз
    dU0 = dU_scalar(b, 0.0, n2d, n2d)
    L0  = Omega_L_scalar(L, b, 0.0, 0.0, n2d, n2d)
    mu0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, n1d)
    out = np.empty((len(M_vals), 3))
    for k, M in enumerate(M_vals):
        out[k, 0] = dU_scalar(b, M, n2d, n2d) - dU0
        out[k, 1] = Omega_L_scalar(L, b, M, 0.0, n2d, n2d) - L0
        out[k, 2] = fun_Omega_L_mu_int(mu, L, b, M, 0, n1d) - mu0
    return out


def total(M_vals, comps, g):
    return M_vals**2 / (2.0 * g) + comps.sum(axis=1)


def gap_G(M_vals, delta):
    """G(M) = (dDelta/dM) / M; производная по неравномерной сетке."""
    return np.gradient(delta, M_vals) / M_vals


def _fmt(s):
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


# ==============================================================================
# Основной расчёт
# ==============================================================================

def main():
    t_start = time.time()
    data = {'M_log': M_LOG, 'N_levels': np.array(N_LEVELS),
            'mu_ref': MU_REF, 'L': L_VAL, 'g': G_COUPLING, 'b_vals': np.array(B_VALS)}

    # JIT-прогрев
    components(0.1, 0.0, MU_REF, L_VAL, 8, 8)

    # --- E1, E2, E5: компоненты при разных N и b ------------------------------
    print("[E1/E2/E5] Компоненты на лог-сетке M")
    comps = {}   # (b, level) -> (N_M, 3)
    for b in B_VALS:
        for lev, (n2d, n1d) in enumerate(N_LEVELS):
            t0 = time.time()
            comps[(b, lev)] = scan_components(M_LOG, b, MU_REF, L_VAL, n2d, n1d)
            data[f'comps_b{b}_lev{lev}'] = comps[(b, lev)]
            print(f"    b={b:.1f}  N2d={n2d:4d}  N1d={n1d:6d}  {_fmt(time.time() - t0)}")

    best = len(N_LEVELS) - 1

    # --- Рисунок E1/E2: Delta_X / M² ------------------------------------------
    fig, axes = plt.subplots(1, len(B_VALS), figsize=(5.2 * len(B_VALS), 4.4), sharey=False)
    names = [r'$\Delta U$', r'$\Omega_L$', r'$\Omega_{\mu L}$']
    for ax, b in zip(axes, B_VALS):
        c = comps[(b, best)]
        for k in range(3):
            ax.semilogx(M_LOG, c[:, k] / M_LOG**2, label=names[k])
        ax.semilogx(M_LOG, c[:, 1:].sum(axis=1) / M_LOG**2, 'k:', label=r'$\Omega_L+\Omega_{\mu L}$')
        for g, st in [(-1.0, 'k-'), (1.0, 'k--')]:
            ax.semilogx(M_LOG, total(M_LOG, c, g) / M_LOG**2, st, lw=2, label=f'total, g={g:+.0f}')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'b={b}, μ={MU_REF}, L={L_VAL}')
        ax.set_xlabel('M')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$[X(M)-X(0)]\,/\,M^2$')
    axes[0].legend(fontsize=8)
    fig.suptitle('E1/E2: константа → $M^2$;  прямая по ln M → $M^2\\ln M$;  наклон ~M → $M^3$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '40_E1_components.png'), dpi=130)
    plt.close(fig)

    # --- Рисунок E5: сходимость по N ------------------------------------------
    fig, axes = plt.subplots(1, len(B_VALS), figsize=(5.2 * len(B_VALS), 4.4))
    for ax, b in zip(axes, B_VALS):
        for lev, (n2d, n1d) in enumerate(N_LEVELS):
            d = total(M_LOG, comps[(b, lev)], G_COUPLING)
            ax.semilogx(M_LOG, d / M_LOG**2, label=f'N2d={n2d}, N1d={n1d}')
        ax.set_title(f'b={b}, g={G_COUPLING:+.0f}')
        ax.set_xlabel('M')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$[\Omega(M)-\Omega(0)]\,/\,M^2$')
    axes[0].legend(fontsize=8)
    fig.suptitle('E5: сходимость по числу узлов (midpoint)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '40_E5_convergence.png'), dpi=130)
    plt.close(fig)

    # --- Рисунок E3: G(M) ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for b in B_VALS:
        for lev in [best - 1, best] if best > 0 else [best]:
            G = gap_G(M_LOG, total(M_LOG, comps[(b, lev)], G_COUPLING))
            ls = '-' if lev == best else ':'
            ax.semilogx(M_LOG, G, ls, label=f'b={b}, N2d={N_LEVELS[lev][0]}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('M')
    ax.set_ylabel(r'$G(M) = \Omega^\prime(M)/M$')
    ax.set_title(f'E3: уравнение щели, μ={MU_REF}, L={L_VAL}, g={G_COUPLING:+.0f}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '40_E3_gap.png'), dpi=130)
    plt.close(fig)

    # --- E4: проход через переход по mu ---------------------------------------
    print("[E4] Карта Omega(b, M) для mu из MU_SCAN")
    n2d, n1d = N_LEVELS[1]
    M_lin = np.linspace(0.0, 5.0, 26 if QUICK else 51)
    b_lin = np.linspace(0.0, 3.0, 16 if QUICK else 31)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap('viridis')
    for i, mu in enumerate(MU_SCAN):
        t0 = time.time()
        Om = np.empty((len(b_lin), len(M_lin)))
        for jb, b in enumerate(b_lin):
            # По схеме перенормировки Omega_ren(b, 0) = Omega_ren(0, 0) для всех b,
            # поэтому разности при разных b отсчитываются от общего нуля.
            Om[jb] = total(M_lin, scan_components(M_lin, b, mu, L_VAL, n2d, n1d), G_COUPLING)
        data[f'E4_Omega_mu{mu}'] = Om
        col = cmap(i / max(len(MU_SCAN) - 1, 1))
        axes[0].plot(M_lin, Om[0], color=col, label=f'μ={mu}')
        prof = Om.min(axis=0)
        axes[1].plot(M_lin, prof, color=col, label=f'μ={mu}')
        print(f"    mu={mu:.2f}  {_fmt(time.time() - t0)}")
    data['E4_M'] = M_lin
    data['E4_b'] = b_lin
    for ax, title in zip(axes, ['b = 0', r'профиль $\min_b\,\Omega(b,M)$']):
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('M')
        ax.set_ylabel(r'$\Omega(M)-\Omega(0)$')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'E4: проход через переход, L={L_VAL}, g={G_COUPLING:+.0f}')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '40_E4_transition.png'), dpi=130)
    plt.close(fig)

    np.savez(os.path.join(OUT_DIR, '40_data.npz'), **data)
    print(f"\nГотово за {_fmt(time.time() - t_start)}. Результаты в {OUT_DIR}")


if __name__ == '__main__':
    main()

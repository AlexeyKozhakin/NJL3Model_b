"""
Эксперимент 1.8b: расширенная разведка (phi = 0).

Продолжение notebooks/40_small_M_asymptotics.py.
Результаты и выводы: publication/notes/2026-10-16_exp-1.8-small-M.md

Вопросы:
    R1. Где лежит глобальный минимум (M*, b*) при mu ∈ [0.5, 6], L ∈ {0.5, 1, 2}?
        Не упирается ли он в край сетки?
    R2. Где фазовые переходы (скачки M*, b* по mu)?
    R3. Бывает ли a(mu, L, b) < 0 — «горка» в M = 0 (возможен переход 2-го рода)?
        a — коэффициент в Omega(M) - Omega(0) ≈ a M² у нуля.
    R4. Коэффициент логарифма в Omega_L и Omega_muL при b = 0 равен ±1/(pi L)?
    R5. Всё то же для g = +1 (меняется только член M²/(2g)).

Ускорение: dU(b, M) и Omega_L(L, b, M) не зависят от mu и считаются один раз
на каждое L; от mu зависит только одномерный Omega_muL.

Перенормировка: Omega_ren(b, M) - Omega_ren(0, 0) = M²/(2g) + sum_X [X(b,M) - X(b,0)],
так как Omega_ren(b, 0) = Omega_ren(0, 0) при всех b.

Графики (notebooks/output/):
    41_R1R2_minimum_vs_mu.png
    41_R1_maps.png
    41_R3_a_coefficient.png
    41_R4_log_coefficient.png
Данные: notebooks/output/41_data.npz

Запуск:  python notebooks/41_scouting_global_minimum.py [--quick]
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
# Параметры
# ==============================================================================

G_VALS  = [-1.0, 1.0]
L_VALS  = [0.5, 1.0, 2.0]
MU_VALS = np.round(np.arange(0.5, 6.0 + 1e-9, 0.25 if QUICK else 0.1), 4)

B_GRID = np.linspace(0.0, 7.0, 36 if QUICK else 71)    # шаг 0.2 / 0.1
M_GRID = np.linspace(0.0, 6.0, 31 if QUICK else 61)    # шаг 0.2 / 0.1

N2D, N1D = 160, 1600            # узлы для карт (сигнал большой, шум не важен)

# R3: a ≈ [Omega(M_A) - Omega(0)] / M_A² при малом M_A (из эксп. 40: шум ~3% при 1e-2)
M_A      = 0.03
N2D_A    = 320
N1D_A    = 6400
B_A_GRID = np.linspace(0.0, 7.0, 15 if QUICK else 29)  # шаг 0.5 / 0.25
MU_A     = MU_VALS[::2] if not QUICK else MU_VALS

# R4: наклон X/M² по ln M
M_LOG4 = np.logspace(-2, np.log10(0.2), 12)
N2D_4, N1D_4 = 640, 25600
MU_4 = 3.2


def _fmt(s):
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


# ==============================================================================
# Слагаемые
# ==============================================================================

def delta_mu_independent(L, b_grid, M_grid, n2d):
    """[dU + Omega_L](b, M) - [dU + Omega_L](b, 0); форма (N_b, N_M)."""
    out = np.empty((len(b_grid), len(M_grid)))
    for i, b in enumerate(b_grid):
        x0 = dU_scalar(b, 0.0, n2d, n2d) + Omega_L_scalar(L, b, 0.0, 0.0, n2d, n2d)
        for j, M in enumerate(M_grid):
            out[i, j] = dU_scalar(b, M, n2d, n2d) + Omega_L_scalar(L, b, M, 0.0, n2d, n2d) - x0
    return out


def delta_muL(mu, L, b_grid, M_grid, n1d):
    """Omega_muL(b, M) - Omega_muL(b, 0); форма (N_b, N_M)."""
    out = np.empty((len(b_grid), len(M_grid)))
    for i, b in enumerate(b_grid):
        x0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, n1d)
        for j, M in enumerate(M_grid):
            out[i, j] = fun_Omega_L_mu_int(mu, L, b, M, 0, n1d) - x0
    return out


def vac(M_grid, g):
    return (M_grid**2 / (2.0 * g))[None, :]


def on_edge(i_b, i_M):
    """Минимум на внешней границе сетки (b_max или M_max)."""
    return i_b == len(B_GRID) - 1 or i_M == len(M_GRID) - 1


# ==============================================================================
# Основной расчёт
# ==============================================================================

def main():
    t_start = time.time()
    data = {'mu': MU_VALS, 'L': np.array(L_VALS), 'g': np.array(G_VALS),
            'b_grid': B_GRID, 'M_grid': M_GRID}

    # JIT-прогрев
    dU_scalar(0.1, 0.1, 4, 4); Omega_L_scalar(1.0, 0.1, 0.1, 0.0, 4, 4)
    fun_Omega_L_mu_int(1.0, 1.0, 0.1, 0.1, 0, 4)

    # ---------- R1/R2/R5: глобальный минимум ----------------------------------
    print("[R1/R2] Глобальный минимум по сетке (b, M)")
    res = {}    # (g, L) -> dict массивов по mu
    maps_keep = {}
    for L in L_VALS:
        t0 = time.time()
        base = delta_mu_independent(L, B_GRID, M_GRID, N2D)
        print(f"    L={L}: dU + Omega_L  {_fmt(time.time() - t0)}")

        # Вакуумный минимум (mu = 0: Omega_muL = 0) при b = 0
        for g in G_VALS:
            prof = vac(M_GRID, g)[0] + base[0]
            print(f"        g={g:+.0f}: вакуумный минимум при b=0: M_vac ≈ {M_GRID[prof.argmin()]:.2f}, "
                  f"Omega-Omega(0) = {prof.min():+.4f}")

        t0 = time.time()
        rec = {g: {k: np.empty(len(MU_VALS)) for k in ('M', 'b', 'Om', 'edge')} for g in G_VALS}
        for k, mu in enumerate(MU_VALS):
            Om_nog = base + delta_muL(mu, L, B_GRID, M_GRID, N1D)
            for g in G_VALS:
                Om = Om_nog + vac(M_GRID, g)
                i_b, i_M = np.unravel_index(np.argmin(Om), Om.shape)
                rec[g]['M'][k]    = M_GRID[i_M]
                rec[g]['b'][k]    = B_GRID[i_b]
                rec[g]['Om'][k]   = Om[i_b, i_M]
                rec[g]['edge'][k] = on_edge(i_b, i_M)
                if L == 1.0 and g == -1.0 and np.isclose(mu, [1.5, 2.5, 3.5, 5.0]).any():
                    maps_keep[float(mu)] = Om
        print(f"    L={L}: Omega_muL по {len(MU_VALS)} mu  {_fmt(time.time() - t0)}")
        for g in G_VALS:
            res[(g, L)] = rec[g]
            for key, arr in rec[g].items():
                data[f'min_{key}_g{g:+.0f}_L{L}'] = arr

    # Печать скачков (кандидаты в переходы 1-го рода)
    print("\n    Скачки (|ΔM*| > 0.3 или |Δb*| > 0.3 между соседними mu):")
    for (g, L), rec in res.items():
        jumps = [f"{MU_VALS[k]:.2f}→{MU_VALS[k+1]:.2f}: (b,M) "
                 f"({rec['b'][k]:.1f},{rec['M'][k]:.1f})→({rec['b'][k+1]:.1f},{rec['M'][k+1]:.1f})"
                 for k in range(len(MU_VALS) - 1)
                 if abs(rec['M'][k+1] - rec['M'][k]) > 0.3 or abs(rec['b'][k+1] - rec['b'][k]) > 0.3]
        n_edge = int(rec['edge'].sum())
        print(f"    g={g:+.0f} L={L}: на краю сетки {n_edge}/{len(MU_VALS)}")
        for s in jumps:
            print(f"        {s}")

    # Рисунок R1/R2
    fig, axes = plt.subplots(len(G_VALS), 3, figsize=(15, 4.2 * len(G_VALS)), squeeze=False)
    for r, g in enumerate(G_VALS):
        for L in L_VALS:
            rec = res[(g, L)]
            for c, key in enumerate(['M', 'b', 'Om']):
                ln, = axes[r, c].plot(MU_VALS, rec[key], '.-', ms=3, label=f'L={L}')
                e = rec['edge'].astype(bool)
                axes[r, c].plot(MU_VALS[e], rec[key][e], 'x', color=ln.get_color(), ms=7)
        for c, lab in enumerate(['M*', 'b*', r'$\Omega_{min}-\Omega(0,0)$']):
            axes[r, c].set_title(f'{lab},  g={g:+.0f}')
            axes[r, c].set_xlabel('μ')
            axes[r, c].grid(alpha=0.3)
        axes[r, 0].legend(fontsize=8)
    fig.suptitle('R1/R2: глобальный минимум на сетке b∈[0,7], M∈[0,6]  (× — минимум на краю сетки)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '41_R1R2_minimum_vs_mu.png'), dpi=120)
    plt.close(fig)

    # Карты Omega(b, M) при L = 1, g = -1
    mus = sorted(maps_keep)
    fig, axes = plt.subplots(1, len(mus), figsize=(4.6 * len(mus), 4.2), squeeze=False)
    for ax, mu in zip(axes[0], mus):
        Om = maps_keep[mu]
        lim = np.percentile(np.abs(Om), 60)
        im = ax.pcolormesh(M_GRID, B_GRID, Om, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
        i_b, i_M = np.unravel_index(np.argmin(Om), Om.shape)
        ax.plot(M_GRID[i_M], B_GRID[i_b], 'k*', ms=12)
        ax.set_title(f'μ={mu}, L=1, g=-1')
        ax.set_xlabel('M'); ax.set_ylabel('b')
        fig.colorbar(im, ax=ax)
    fig.suptitle(r'R1: $\Omega(b,M)-\Omega(0,0)$, ★ — глобальный минимум на сетке')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '41_R1_maps.png'), dpi=120)
    plt.close(fig)

    # ---------- R3: коэффициент a(mu, b) --------------------------------------
    print("\n[R3] Коэффициент a(mu, b) у M = 0")
    M_pair = np.array([M_A])
    fig, axes = plt.subplots(len(G_VALS), len(L_VALS), figsize=(5 * len(L_VALS), 4.2 * len(G_VALS)),
                             squeeze=False)
    for c, L in enumerate(L_VALS):
        t0 = time.time()
        base_a = delta_mu_independent(L, B_A_GRID, M_pair, N2D_A)[:, 0]
        a_nog = np.empty((len(MU_A), len(B_A_GRID)))
        for k, mu in enumerate(MU_A):
            a_nog[k] = (base_a + delta_muL(mu, L, B_A_GRID, M_pair, N1D_A)[:, 0]) / M_A**2
        print(f"    L={L}: {_fmt(time.time() - t0)}")
        for r, g in enumerate(G_VALS):
            a = a_nog + 1.0 / (2.0 * g)
            data[f'a_g{g:+.0f}_L{L}'] = a
            neg = (a < 0).sum()
            print(f"        g={g:+.0f}: a<0 в {neg}/{a.size} точек; min a = {a.min():+.3f}")
            ax = axes[r, c]
            lim = np.abs(a).max()
            im = ax.pcolormesh(B_A_GRID, MU_A, a, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
            if a.min() < 0 < a.max():
                ax.contour(B_A_GRID, MU_A, a, levels=[0], colors='k', linewidths=1.5)
            ax.set_title(f'a(μ, b),  L={L}, g={g:+.0f}')
            ax.set_xlabel('b'); ax.set_ylabel('μ')
            fig.colorbar(im, ax=ax)
    data['a_mu'] = MU_A
    data['a_b'] = B_A_GRID
    fig.suptitle('R3: a > 0 (красный) — M = 0 устойчив; a < 0 (синий) — «горка», минимум при M > 0')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '41_R3_a_coefficient.png'), dpi=120)
    plt.close(fig)

    # ---------- R4: коэффициент логарифма --------------------------------------
    print("\n[R4] Коэффициент M² ln M при b = 0, mu = 3.2")
    lm = np.log(M_LOG4)
    slopes = []
    for L in L_VALS:
        dL = np.array([Omega_L_scalar(L, 0.0, M, 0.0, N2D_4, N2D_4) for M in M_LOG4]) \
            - Omega_L_scalar(L, 0.0, 0.0, 0.0, N2D_4, N2D_4)
        dmu = np.array([fun_Omega_L_mu_int(MU_4, L, 0.0, M, 0, N1D_4) for M in M_LOG4]) \
            - fun_Omega_L_mu_int(MU_4, L, 0.0, 0.0, 0, N1D_4)
        sL  = np.polyfit(lm, dL / M_LOG4**2, 1)[0]
        smu = np.polyfit(lm, dmu / M_LOG4**2, 1)[0]
        slopes.append((sL, smu))
        print(f"    L={L}: Omega_L {sL:+.4f}, Omega_muL {smu:+.4f}, 1/(pi L) = {1 / (np.pi * L):.4f}")
    slopes = np.array(slopes)
    data['R4_slopes'] = slopes

    fig, ax = plt.subplots(figsize=(6, 4.4))
    Ls = np.array(L_VALS)
    ax.plot(Ls, slopes[:, 0], 'o', label=r'$\Omega_L$')
    ax.plot(Ls, -slopes[:, 1], 's', label=r'$-\Omega_{\mu L}$')
    Lf = np.linspace(0.4, 2.1, 100)
    ax.plot(Lf, 1 / (np.pi * Lf), 'k-', label=r'$1/(\pi L)$')
    ax.set_xlabel('L'); ax.set_ylabel(r'наклон $X/M^2$ по $\ln M$')
    ax.set_title('R4: коэффициент логарифма, b = 0')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '41_R4_log_coefficient.png'), dpi=120)
    plt.close(fig)

    np.savez(os.path.join(OUT_DIR, '41_data.npz'), **data)
    print(f"\nГотово за {_fmt(time.time() - t_start)}. Результаты в {OUT_DIR}")


if __name__ == '__main__':
    main()

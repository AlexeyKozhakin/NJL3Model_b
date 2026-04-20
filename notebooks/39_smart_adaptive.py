"""
39_smart_adaptive.py

Грубая сетка 8×8 (grid + L-BFGS-B) → интерполяция M_min, b_min на 128×128
→ уточнение ВСЕХ точек L-BFGS-B с warm start из интерполяции.
Параллельно на n_workers ядрах.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import time

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int


def Omega_ren_scalar(M, b, mu, L, g, N_h=100):
    """Вычисляет Omega_ren(M, b, mu, L, g) для скалярных значений."""
    L_arr = np.array([L])
    b_arr = np.array([b])
    M_arr = np.array([M])
    M0_arr = np.array([0.0])
    b0_arr = np.array([0.0])

    Omega_L_val = fun_Omega_L(L_arr, b_arr, M_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_b0 = fun_Omega_L(L_arr, b_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_00 = fun_Omega_L(L_arr, b0_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_phys = Omega_L_val - Omega_L_b0 + Omega_L_00

    dU = fun_dU_phys(b_arr, M_arr, N_h_p=N_h, N_h_phi=N_h)[0, 0]

    Omega_mu_L = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=N_h)
    Omega_mu_L_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_phys = Omega_mu_L - Omega_mu_L_b0 + Omega_mu_L_00

    tree = M**2 / (2 * g)
    return tree + dU + Omega_L_phys + Omega_mu_L_phys


def find_minimum_grid_lbfgsb(mu, L, g, grid_n=10, N_h=100):
    """Grid search grid_n×grid_n + L-BFGS-B refinement."""
    M_vals = np.linspace(0, 10, grid_n)
    b_vals = np.linspace(0, 10, grid_n)

    best_val = np.inf
    best_M, best_b = 0.0, 0.0

    for M in M_vals:
        for b in b_vals:
            val = Omega_ren_scalar(M, b, mu, L, g, N_h)
            if val < best_val:
                best_val = val
                best_M, best_b = M, b

    M_win = max(0.5, best_M * 0.5)
    b_win = max(0.5, best_b * 0.5)

    result = minimize(
        lambda x: Omega_ren_scalar(x[0], x[1], mu, L, g, N_h),
        [best_M, best_b],
        bounds=[(max(0, best_M - M_win), min(10, best_M + M_win)),
                (max(0, best_b - b_win), min(10, best_b + b_win))],
        method='L-BFGS-B'
    )

    M_opt, b_opt = result.x
    if M_opt < 0.05:
        M_opt = 0.0
        b_opt = 0.0

    return M_opt, b_opt, result.fun


def find_minimum_warm(mu, L, g, M_start, b_start, N_h=100):
    """L-BFGS-B с warm start из интерполяции."""
    M_start = max(0, float(M_start))
    b_start = max(0, float(b_start))

    bounds = [
        (max(0, M_start * 0.3), min(10, M_start * 3.0 + 0.1)),
        (max(0, b_start * 0.3), min(10, b_start * 3.0 + 0.1))
    ]

    result = minimize(
        lambda x: Omega_ren_scalar(x[0], x[1], mu, L, g, N_h),
        [M_start, b_start],
        bounds=bounds,
        method='L-BFGS-B'
    )

    M_opt, b_opt = result.x
    if M_opt < 0.05:
        M_opt = 0.0
        b_opt = 0.0

    return M_opt, b_opt, result.fun


def worker_grid(args):
    mu, L, g = args
    return find_minimum_grid_lbfgsb(mu, L, g)


def worker_warm(args):
    mu, L, g, M_start, b_start = args
    return find_minimum_warm(mu, L, g, M_start, b_start)


def main():
    g = -1
    N = 256
    n_coarse = 32
    n_workers = 10  # Параллельно на 10 ядрах

    mu_min, mu_max = 2.0, 7.0
    L_min, L_max = 1.5, 3.0

    mu_vals = np.linspace(mu_min, mu_max, N)
    L_vals = np.linspace(L_min, L_max, N)
    mu_coarse = np.linspace(mu_min, mu_max, n_coarse)
    L_coarse = np.linspace(L_min, L_max, n_coarse)

    print("=" * 60)
    print("ГРУБАЯ СЕТКА + ИНТЕРПОЛЯЦИЯ + УТОЧНЕНИЕ ВСЕХ ТОЧЕК")
    print("=" * 60)
    print(f"Грубая сетка: {n_coarse}×{n_coarse}")
    print(f"Финальное разрешение: {N}×{N} = {N*N} точек")
    print(f"Параллельно на: {n_workers} ядрах")
    print()

    # ========================================================================
    # Шаг 1: Грубая сетка 8×8
    # ========================================================================
    print("Шаг 1: Грубая сетка 8×8 (grid + L-BFGS-B)...")
    points_coarse = [(mu, L, g) for mu in mu_coarse for L in L_coarse]

    start = time.time()
    with Pool(n_workers) as pool:
        results_coarse = list(tqdm(
            pool.imap(worker_grid, points_coarse),
            total=len(points_coarse)
        ))
    t_coarse = time.time() - start
    print(f"Время: {t_coarse:.1f} сек")

    M_coarse = np.zeros((n_coarse, n_coarse))
    b_coarse = np.zeros((n_coarse, n_coarse))

    for idx, (M, b, val) in enumerate(results_coarse):
        i = idx // n_coarse
        j = idx % n_coarse
        M_coarse[i, j] = M
        b_coarse[i, j] = b

    # ========================================================================
    # Шаг 2: Интерполяция на 128×128
    # ========================================================================
    print("\nШаг 2: Интерполяция на 128×128...")

    interp_M = RegularGridInterpolator(
        (mu_coarse, L_coarse), M_coarse,
        method='cubic', bounds_error=False, fill_value=None
    )
    interp_b = RegularGridInterpolator(
        (mu_coarse, L_coarse), b_coarse,
        method='cubic', bounds_error=False, fill_value=None
    )

    MU, L_grid = np.meshgrid(mu_vals, L_vals, indexing='ij')
    points_interp = np.array([MU.ravel(), L_grid.ravel()]).T

    M_interp = interp_M(points_interp).reshape(N, N)
    b_interp = interp_b(points_interp).reshape(N, N)
    M_interp = np.clip(M_interp, 0, None)
    b_interp = np.clip(b_interp, 0, None)

    # ========================================================================
    # Шаг 3: Выбор точек для уточнения (по градиенту интерполяции)
    # ========================================================================
    print("\nШаг 3: Анализ градиента интерполяции...")

    # Градиент M и b
    grad_M = np.abs(np.gradient(M_interp)[0]) + np.abs(np.gradient(M_interp)[1])
    grad_b = np.abs(np.gradient(b_interp)[0]) + np.abs(np.gradient(b_interp)[1])

    # Нормализуем
    grad_M_norm = grad_M / (np.max(grad_M) + 1e-10)
    grad_b_norm = grad_b / (np.max(grad_b) + 1e-10)

    # Уточняем точки где градиент большой (верхние ~15-20%)
    threshold = 0.15
    mask = (grad_M_norm > threshold) | (grad_b_norm > threshold)

    n_refine = np.sum(mask)
    print(f"Точек для уточнения: {n_refine} из {N*N} ({100*n_refine/(N*N):.1f}%)")

    # ========================================================================
    # Шаг 4: Уточнение выбранных точек L-BFGS-B warm start
    # ========================================================================
    print(f"\nШаг 4: Уточнение {n_refine} точек L-BFGS-B (warm start)...")

    points_warm = []
    indices_warm = []
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                points_warm.append((mu_vals[i], L_vals[j], g, M_interp[i, j], b_interp[i, j]))
                indices_warm.append((i, j))

    start = time.time()
    with Pool(n_workers) as pool:
        results_warm = list(tqdm(
            pool.imap(worker_warm, points_warm),
            total=len(points_warm)
        ))
    t_warm = time.time() - start
    print(f"Время: {t_warm:.1f} сек")

    M_map = M_interp.copy()
    b_map = b_interp.copy()

    for (i, j), (M, b, val) in zip(indices_warm, results_warm):
        M_map[i, j] = M
        b_map[i, j] = b

    # ========================================================================
    # Сохранение
    # ========================================================================
    np.savez('notebooks/output/39_smart_adaptive.npz',
             mu_vals=mu_vals, L_vals=L_vals,
             M_map=M_map, b_map=b_map,
             M_interp=M_interp, b_interp=b_interp,
             mask=mask, threshold=threshold,
             mu_coarse=mu_coarse, L_coarse=L_coarse,
             M_coarse=M_coarse, b_coarse=b_coarse)
    print(f"\nСохранено: notebooks/output/39_smart_adaptive.npz")
    print(f"  - M_map, b_map: уточнённые значения")
    print(f"  - M_interp, b_interp: интерполированные значения")
    print(f"  - mask: True где уточнялось L-BFGS-B")
    print(f"Общее время: {t_coarse + t_warm:.1f} сек")

    # ========================================================================
    # Визуализация
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    L_grid_plot, mu_grid_plot = np.meshgrid(L_vals, mu_vals)

    ax = axes[0]
    im = ax.pcolormesh(L_grid_plot, mu_grid_plot, M_map, shading='auto', cmap='hot')
    ax.set_title(r'$M_{\min}$ (уточнённые)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.pcolormesh(L_grid_plot, mu_grid_plot, b_map, shading='auto', cmap='hot')
    ax.set_title(r'$b_{\min}$ (уточнённые)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('notebooks/output/39_smart_adaptive_map.png', dpi=150, bbox_inches='tight')
    print("Картинка: notebooks/output/39_smart_adaptive_map.png")
    plt.show()


if __name__ == "__main__":
    main()

"""
37_fast_three_point.py

Ультрабыстрая классификация фазы по 3 точкам:
  1. (0, 0)      — базовое значение
  2. (M_test, 0) — проверяет, падает ли Ω при росте M
  3. (M_test, b_test) — проверяет, падает ли Ω при росте b (при M>0)

При M_test=0.5 сигнал надёжно превышает численную ошибку (~1e-4).
Для граничных точек (|delta| < tol) — fallback к точному grid+L-BFGS-B.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize_scalar, minimize

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int


# =============================================================================
# Вычисление Omega_ren для скалярных значений
# =============================================================================

def Omega_ren_scalar(M, b, mu, L, g, N_h=100):
    """
    Вычисляет Omega_ren(M, b, mu, L, g) для скалярных значений.
    """
    L_arr = np.array([L])
    b_arr = np.array([b])
    M_arr = np.array([M])
    M0_arr = np.array([0.0])
    b0_arr = np.array([0.0])
    
    # Omega_L_phys = Omega_L(L,b,M) - Omega_L(L,b,0) + Omega_L(L,0,0)
    Omega_L_val = fun_Omega_L(L_arr, b_arr, M_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_b0 = fun_Omega_L(L_arr, b_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_00 = fun_Omega_L(L_arr, b0_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_phys = Omega_L_val - Omega_L_b0 + Omega_L_00
    
    # dU_phys
    dU_phys = fun_dU_phys(b_arr, M_arr, N_h_p=N_h, N_h_phi=N_h)[0, 0]
    
    # Omega_mu_L_phys = Omega_mu_L(mu,L,b,M) - Omega_mu_L(mu,L,b,0) + Omega_mu_L(mu,L,0,0)
    Omega_mu_L = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=N_h)
    Omega_mu_L_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_phys = Omega_mu_L - Omega_mu_L_b0 + Omega_mu_L_00
    
    tree = M**2 / (2 * g)
    
    return tree + dU_phys + Omega_L_phys + Omega_mu_L_phys


# =============================================================================
# Быстрая классификация по 3 точкам
# =============================================================================

def classify_three_point(mu, L, g, M_test=0.5, b_test=0.5,
                         tol_M=1e-7, tol_b=1e-7):
    """
    Классифицирует фазу по 3 точкам.
    
    Returns:
        phase: 0=HIGH, 1=LOW_B0, 2=LOW_BPOS
        M_est, b_est: оценки M, b (не точные, только для визуализации)
        val: лучшее значение Omega
        is_gray: True если попали в граничную зону
    """
    val_00 = Omega_ren_scalar(0.0, 0.0, mu, L, g)
    val_M0 = Omega_ren_scalar(M_test, 0.0, mu, L, g)
    
    delta_M = val_M0 - val_00
    
    # Граничная зона по M: разница меньше tol → нужен fallback
    if abs(delta_M) < tol_M:
        return -1, 0.0, 0.0, val_00, True  # gray zone
    
    if delta_M > 0:
        # M = 0 минимально
        return 0, 0.0, 0.0, val_00, False  # HIGH
    
    # M > 0: проверяем b
    val_Mb = Omega_ren_scalar(M_test, b_test, mu, L, g)
    delta_b = val_Mb - val_M0
    
    if abs(delta_b) < tol_b:
        return -1, M_test, 0.0, val_M0, True  # gray zone
    
    if delta_b > 0:
        return 1, M_test, 0.0, val_M0, False  # LOW_B0
    else:
        return 2, M_test, b_test, val_Mb, False  # LOW_BPOS


# =============================================================================
# Точный fallback: grid + L-BFGS-B
# =============================================================================

def find_minimum_exact(mu, L, g, N_h=100):
    """
    Точный поиск минимума через grid search + L-BFGS-B.
    """
    # Адаптивные границы на основе эвристик
    if mu < 2.5 or L > 3.5:
        M_max, b_max = 2.0, 0.0  # b точно 0
    else:
        M_max = min(10.0, max(2.0, mu + 1.0))
        b_max = min(10.0, max(2.0, mu * 0.8))
    
    # Grid search
    M_grid = np.linspace(0, M_max, 15)
    b_grid = np.linspace(0, b_max, 15)
    
    best_val = np.inf
    best_M, best_b = 0.0, 0.0
    
    for M in M_grid:
        for b in b_grid:
            val = Omega_ren_scalar(M, b, mu, L, g, N_h=N_h)
            if val < best_val:
                best_val = val
                best_M, best_b = M, b
    
    # L-BFGS-B refinement
    if best_M > 0.01 and best_b > 0.01:
        # 2D refinement
        M_win = max(0.5, best_M * 0.5)
        b_win = max(0.5, best_b * 0.5)
        M_lo = max(0, best_M - M_win)
        M_hi = best_M + M_win
        b_lo = max(0, best_b - b_win)
        b_hi = best_b + b_win
        
        result = minimize(
            lambda x: Omega_ren_scalar(x[0], x[1], mu, L, g, N_h=N_h),
            [best_M, best_b],
            bounds=[(M_lo, M_hi), (b_lo, b_hi)],
            method='L-BFGS-B'
        )
        M_opt, b_opt = result.x
        val_opt = result.fun
    elif best_M > 0.01:
        # 1D refinement по M при b=0
        result = minimize_scalar(
            lambda M: Omega_ren_scalar(M, 0.0, mu, L, g, N_h=N_h),
            bounds=(0, M_max),
            method='bounded'
        )
        M_opt, b_opt = result.x, 0.0
        val_opt = result.fun
    else:
        M_opt, b_opt = 0.0, 0.0
        val_opt = best_val
    
    # Канонизация
    M_canon_tol = 0.05
    if M_opt < M_canon_tol:
        M_opt = 0.0
        b_opt = 0.0
    elif b_opt < 1e-4:
        b_opt = 0.0
    
    # Определение фазы
    if M_opt < M_canon_tol:
        phase = 0  # HIGH
    elif b_opt < 1e-4:
        phase = 1  # LOW_B0
    else:
        phase = 2  # LOW_BPOS
    
    return phase, M_opt, b_opt, val_opt


# =============================================================================
# Параллельная обработка одной точки
# =============================================================================

def process_point(args):
    mu, L, g = args
    
    # Пробуем быструю классификацию
    phase, M_est, b_est, val, is_gray = classify_three_point(mu, L, g)
    
    n_evals = 3  # (0,0), (M_test,0), (M_test,b_test)
    
    if is_gray:
        # Fallback к точному методу
        phase, M_est, b_est, val = find_minimum_exact(mu, L, g)
        n_evals = 120  # примерно столько нужно grid+L-BFGS-B
    
    return {
        'mu': mu,
        'L': L,
        'phase': phase,
        'M': M_est,
        'b': b_est,
        'val': val,
        'n_evals': n_evals,
        'is_gray': is_gray
    }


# =============================================================================
# Основной блок
# =============================================================================

if __name__ == "__main__":
    g = -1
    
    # Сетка 50x50
    n_mu, n_L = 100, 100
    mu_vals = np.linspace(0.1, 7.0, n_mu)
    L_vals = np.linspace(0.1, 4.0, n_L)
    
    points = [(mu, L, g) for mu in mu_vals for L in L_vals]
    
    print(f"Классификация {len(points)} точек сетки...")
    print(f"M_test = 0.5, b_test = 0.5")
    print()
    
    start = time.time()
    
    n_workers = 10
    results = []
    
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_point, points),
            total=len(points),
            desc="Классификация"
        ):
            results.append(result)
    
    elapsed = time.time() - start
    
    # Сборка массивов
    phase_map = np.zeros((n_mu, n_L))
    M_map = np.zeros((n_mu, n_L))
    b_map = np.zeros((n_mu, n_L))
    n_evals_map = np.zeros((n_mu, n_L))
    
    for r in results:
        i = np.argmin(np.abs(mu_vals - r['mu']))
        j = np.argmin(np.abs(L_vals - r['L']))
        phase_map[i, j] = r['phase']
        M_map[i, j] = r['M']
        b_map[i, j] = r['b']
        n_evals_map[i, j] = r['n_evals']
    
    # Статистика
    n_high = np.sum(phase_map == 0)
    n_low_b0 = np.sum(phase_map == 1)
    n_low_bpos = np.sum(phase_map == 2)
    n_gray = np.sum(n_evals_map > 10)
    avg_evals = np.mean(n_evals_map)
    total_evals = np.sum(n_evals_map)
    
    print(f"\n{'='*50}")
    print(f"Результаты классификации (время: {elapsed:.1f} сек)")
    print(f"{'='*50}")
    print(f"HIGH    (M=0, b=0):  {n_high:5d} точек ({100*n_high/len(points):.1f}%)")
    print(f"LOW_B0  (M>0, b=0):  {n_low_b0:5d} точек ({100*n_low_b0/len(points):.1f}%)")
    print(f"LOW_BPOS(M>0, b>0):  {n_low_bpos:5d} точек ({100*n_low_bpos/len(points):.1f}%)")
    print(f"Fallback (gray zone): {n_gray:5d} точек ({100*n_gray/len(points):.1f}%)")
    print(f"Среднее число вызовов Omega_ren: {avg_evals:.1f}")
    print(f"Всего вызовов Omega_ren: {int(total_evals)}")
    print(f"{'='*50}\n")
    
    # Сохранение
    np.savez(
        'notebooks/output/37_three_point_ckpt.npz',
        mu_vals=mu_vals,
        L_vals=L_vals,
        phase_map=phase_map,
        M_map=M_map,
        b_map=b_map,
        n_evals_map=n_evals_map
    )
    print("Сохранено в notebooks/output/37_three_point_ckpt.npz")
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    L_grid, mu_grid = np.meshgrid(L_vals, mu_vals)
    
    # Phase map
    ax = axes[0, 0]
    im = ax.pcolormesh(L_grid, mu_grid, phase_map, shading='auto', cmap='viridis')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Phase (0=HIGH, 1=LOW_B0, 2=LOW_BPOS)')
    plt.colorbar(im, ax=ax)
    
    # M map
    ax = axes[0, 1]
    im = ax.pcolormesh(L_grid, mu_grid, M_map, shading='auto', cmap='hot')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    ax.set_title(r'$M_{\min}$ (0.5 — только классификация)')
    plt.colorbar(im, ax=ax)
    
    # b map
    ax = axes[1, 0]
    im = ax.pcolormesh(L_grid, mu_grid, b_map, shading='auto', cmap='hot')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    ax.set_title(r'$b_{\min}$ (0.5 — только классификация)')
    plt.colorbar(im, ax=ax)
    
    # N evals map
    ax = axes[1, 1]
    im = ax.pcolormesh(L_grid, mu_grid, n_evals_map, shading='auto', cmap='coolwarm')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    ax.set_title('Число вызовов Omega_ren (3=быстро, 120=fallback)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('notebooks/output/37_three_point_map.png', dpi=150)
    print("График сохранён в notebooks/output/37_three_point_map.png")
    plt.show()

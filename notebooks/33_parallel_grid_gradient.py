"""
================================================================================
Параллельная равномерная сетка (mu,L) + grid + градиентный поиск (M,b)
================================================================================
Этап 1: grid search 10×10 по (M,b) — находим область минимума
Этап 2: Проверка: если Omega(M,b) ≈ Omega(M,0) → b=0, ищем min только по M
         Иначе → L-BFGS-B по (M,b) в окне

Запуск:  python 33_parallel_grid_gradient.py
================================================================================
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from scipy.optimize import minimize_scalar, minimize

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from functions.Omega_L import fun_Omega_L
from functions.Omega_mu_L import fun_Omega_L_mu_int
from functions.dU import fun_dU_phys
from functools import lru_cache


# ==============================================================================
# Omega_ren
# ==============================================================================
@lru_cache(maxsize=None)
def _cached_omega_L(L, b, M, N_h_p1, N_h_p2):
    return fun_Omega_L(L, b, M, N_h_p1, N_h_p2).item()

@lru_cache(maxsize=None)
def _cached_dU(b, M, N_h_p, N_h_phi):
    return fun_dU_phys(np.array([b]), np.array([M]), N_h_p, N_h_phi).item()

def Omega_ren(M, b, mu, L, g, N_h_p1=80, N_h_p2=80, N_h_phi=80, N_h_mu=80):
    if M < 0.0: M = 0.0
    if b < 0.0: b = 0.0

    vac = M * M / (2.0 * g)

    dU_M  = _cached_dU(float(b), float(M), N_h_phi, N_h_phi)
    dU_M0 = _cached_dU(float(b), 0.0, N_h_phi, N_h_phi)
    dU_b0 = _cached_dU(0.0, 0.0, N_h_phi, N_h_phi)
    dU_phys = dU_M - dU_M0 + dU_b0

    oL_M  = _cached_omega_L(float(L), float(b), float(M), N_h_p1, N_h_p2)
    oL_M0 = _cached_omega_L(float(L), float(b), 0.0, N_h_p1, N_h_p2)
    oL_b0 = _cached_omega_L(float(L), 0.0, 0.0, N_h_p1, N_h_p2)
    Omega_L_phys = oL_M - oL_M0 + oL_b0

    o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h_mu)
    o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h_mu)
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h_mu)
    Omega_mu_L_phys = o_mu_M - o_mu_M0 + o_mu_b0

    return vac + dU_phys + Omega_L_phys + Omega_mu_L_phys


# ==============================================================================
# Двухэтапный поиск: grid + градиент
# ==============================================================================
def find_minimum_grid_gradient(mu, L, g, M_max=1.0, b_max=3.0,
                                grid_M=10, grid_b=10, b_flat_tol=1e-6):
    """
    Этап 1: grid search grid_M × grid_b
    Этап 2: проверка плоскости по b → либо min по M, либо L-BFGS-B по (M,b)
    """
    # --- ЭТАП 1: грубый grid ---
    M_vals = np.linspace(0.01, M_max, grid_M)
    b_vals = np.linspace(0, b_max, grid_b)
    
    best_val = float('inf')
    best_M = 0.0
    best_b = 0.0
    
    for M in M_vals:
        for b in b_vals:
            val = Omega_ren(M, b, mu, L, g)
            if val < best_val:
                best_val = val
                best_M = M
                best_b = b
    
    # Шаг сетки
    dM = M_max / (grid_M - 1) if grid_M > 1 else M_max
    db = b_max / (grid_b - 1) if grid_b > 1 else b_max
    
    # --- ЭТАП 2: проверка плоскости по b ---
    val_best = Omega_ren(best_M, best_b, mu, L, g)
    val_b0 = Omega_ren(best_M, 0.0, mu, L, g)
    
    b_is_flat = abs(val_best - val_b0) < b_flat_tol
    
    if b_is_flat or best_M < 0.05:
        # b не влияет (или M≈0) → канонизируем b=0, ищем min только по M
        M_low = max(0.0, best_M - dM)
        M_high = min(M_max, best_M + dM)
        
        result = minimize_scalar(
            lambda M: Omega_ren(M, 0.0, mu, L, g),
            bounds=(M_low, M_high),
            method='bounded'
        )
        M_opt = result.x
        b_opt = 0.0
        val_opt = result.fun
    else:
        # b влияет → L-BFGS-B по (M, b)
        M_low = max(0.0, best_M - dM)
        M_high = min(M_max, best_M + dM)
        b_low = max(0.0, best_b - db)
        b_high = min(b_max, best_b + db)
        
        result = minimize(
            lambda x: Omega_ren(x[0], x[1], mu, L, g),
            x0=[best_M, best_b],
            bounds=[(M_low, M_high), (b_low, b_high)],
            method='L-BFGS-B',
            options={'ftol': 1e-7, 'gtol': 1e-5, 'maxiter': 20}
        )
        M_opt, b_opt = result.x
        val_opt = result.fun
    
    # Финальная канонизация
    if M_opt < 0.001:
        M_opt = 0.0
        b_opt = 0.0
    elif b_opt < 1e-4:
        b_opt = 0.0
    
    return float(M_opt), float(b_opt), float(val_opt)


# ==============================================================================
# Параллельный воркер
# ==============================================================================
_worker_cache = {}

def _init_worker(g):
    _worker_cache['g'] = g
    find_minimum_grid_gradient(4.0, 1.0, g)  # прогрев JIT

def _compute_one(args):
    mu, L = args
    g = _worker_cache.get('g', -1.0)
    M_min, b_min, val = find_minimum_grid_gradient(mu, L, g)
    return (mu, L, M_min, b_min, val)


# ==============================================================================
# Главный скрипт
# ==============================================================================
if __name__ == '__main__':
    # ======================== НАСТРОЙКИ ========================
    g = -1.0
    mu_min, mu_max = 3.5, 5.0
    L_min, L_max = 1.5, 4.0
    n_mu = 100
    n_L = 100
    n_workers = 10
    checkpoint_every = 100
    out_dir = 'output'
    # ===========================================================

    os.makedirs(out_dir, exist_ok=True)

    mu_vals = np.linspace(mu_min, mu_max, n_mu)
    L_vals = np.linspace(L_min, L_max, n_L)
    points = [(float(mu), float(L)) for mu in mu_vals for L in L_vals]
    total = len(points)

    print(f"{'='*60}")
    print(f"Grid+Gradient parallel: {n_mu}×{n_L} = {total} точек")
    print(f"Workers: {n_workers} | ETA: ~{total * 0.08 / n_workers / 60:.1f} мин")
    print(f"Этап 1: grid 10×10 | Этап 2: L-BFGS-B или minimize_scalar")
    print(f"{'='*60}\n")

    t0 = time.time()
    results_M = np.zeros((n_mu, n_L))
    results_b = np.zeros((n_mu, n_L))
    results_v = np.zeros((n_mu, n_L))

    with Pool(n_workers, initializer=_init_worker, initargs=(g,)) as pool:
        for i, (mu, L, M_min, b_min, val) in enumerate(tqdm(
            pool.imap_unordered(_compute_one, points),
            total=total, desc="Points", unit="pt", ncols=80
        )):
            i_mu = np.argmin(np.abs(mu_vals - mu))
            i_L = np.argmin(np.abs(L_vals - L))
            results_M[i_mu, i_L] = M_min
            results_b[i_mu, i_L] = b_min
            results_v[i_mu, i_L] = val

            if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"\n  [CHECKPOINT] {i+1}/{total} | {rate:.1f} pt/s | ETA {eta/60:.1f} min")
                np.savez(os.path.join(out_dir, '33_grid_gradient_ckpt.npz'),
                         mu=mu_vals, L=L_vals, M=results_M, b=results_b, val=results_v)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Готово! {total} точек за {elapsed/60:.1f} мин ({total/elapsed:.1f} pt/s)")
    print(f"{'='*60}\n")

    # ======================== ВИЗУАЛИЗАЦИЯ ========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(results_M, origin='lower', aspect='auto', cmap='viridis',
                   extent=[L_min, L_max, mu_min, mu_max])
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$M_{{min}}(\mu, L)$, g={g} (grid+gradient, {n_mu}×{n_L})')
    fig.colorbar(im, ax=ax, label=r'$M_{min}$')

    ax = axes[1]
    im = ax.imshow(results_b, origin='lower', aspect='auto', cmap='plasma',
                   extent=[L_min, L_max, mu_min, mu_max])
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$b_{{min}}(\mu, L)$, g={g} (grid+gradient, {n_mu}×{n_L})')
    fig.colorbar(im, ax=ax, label=r'$b_{min}$')

    plt.tight_layout()
    png_path = os.path.join(out_dir, '33_grid_gradient_final.png')
    plt.savefig(png_path, dpi=200)
    print(f"Сохранено: {png_path}")
    plt.show()

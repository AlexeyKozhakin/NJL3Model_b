"""
================================================================================
Бинарная классификация фазовой диаграммы (M=0 vs M>0, b=0 vs b>0)
================================================================================
Вместо точного минимума — быстрая классификация за 2-4 вызова Omega_ren:

1. Проверка M:
   - Omega(0,0) vs Omega(0.5,0) vs Omega(3.0,0)
   - Если хоть одна дает выигрыш > tol → M>0
   
2. Проверка b (если M>0):
   - Omega(best_M, 0) vs Omega(best_M, 1.0)
   - Если выигрыш > tol → b>0

3. Fallback: если в "серой зоне" (разница < tol) → полный grid+L-BFGS-B

Запуск:  python 36_binary_classification.py
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
# Fallback: полный поиск (grid + L-BFGS-B)
# ==============================================================================
def _full_search(mu, L, g):
    """Полный grid + L-BFGS-B для fallback на границе фаз."""
    M_vals = np.linspace(0, 10, 10)
    b_vals = np.linspace(0, 10, 10)
    
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
    
    result = minimize(
        lambda x: Omega_ren(x[0], x[1], mu, L, g),
        x0=[best_M, best_b],
        bounds=[(max(0, best_M - 1.1), min(10, best_M + 1.1)),
                (max(0, best_b - 1.1), min(10, best_b + 1.1))],
        method='L-BFGS-B',
        options={'maxiter': 15}
    )
    
    M_opt, b_opt = result.x
    if M_opt < 0.05:
        M_opt, b_opt = 0.0, 0.0
    elif b_opt < 1e-4:
        b_opt = 0.0
    
    return float(M_opt), float(b_opt)


# ==============================================================================
# Быстрая бинарная классификация
# ==============================================================================
def classify_point(mu, L, g, tol_M=1e-3, tol_b=1e-3):
    """
    Возвращает (M_class, b_class, M_approx, b_approx)
    M_class: '=0' или '>0'
    b_class: '=0' или '>0'
    """
    # --- Проверка M ---
    val_00 = Omega_ren(0.0, 0.0, mu, L, g)
    
    # Тест M = 0.5
    val_05 = Omega_ren(0.5, 0.0, mu, L, g)
    if val_05 < val_00 - tol_M:
        M_class = ">0"
        best_M = 0.5
        best_val = val_05
    else:
        # Тест M = 3.0
        val_30 = Omega_ren(3.0, 0.0, mu, L, g)
        if val_30 < val_00 - tol_M:
            M_class = ">0"
            best_M = 3.0
            best_val = val_30
        else:
            M_class = "=0"
            best_M = 0.0
            best_val = val_00
    
    # --- Проверка b (если M>0) ---
    if M_class == "=0":
        b_class = "=0"
        best_b = 0.0
    else:
        # Тест b = 1.0 при найденном best_M
        val_Mb = Omega_ren(best_M, 1.0, mu, L, g)
        if val_Mb < best_val - tol_b:
            b_class = ">0"
            best_b = 1.0
        else:
            b_class = "=0"
            best_b = 0.0
    
    # --- Fallback: серая зона ---
    # Если M>0 но b=0, и разница с M=0 малая — возможно M=0
    if M_class == ">0" and b_class == "=0":
        if abs(best_val - val_00) < tol_M:
            # Серая зона — запускаем полный поиск
            M_opt, b_opt = _full_search(mu, L, g)
            M_class = "=0" if M_opt < 0.05 else ">0"
            b_class = "=0" if b_opt < 0.05 else ">0"
            best_M = M_opt
            best_b = b_opt
    
    return M_class, b_class, best_M, best_b


# ==============================================================================
# Параллельный воркер
# ==============================================================================
_worker_cache = {}

def _init_worker(g):
    _worker_cache['g'] = g
    classify_point(4.0, 1.0, g)

def _compute_one(args):
    mu, L = args
    g = _worker_cache.get('g', -1.0)
    M_cls, b_cls, M_val, b_val = classify_point(mu, L, g)
    return (mu, L, M_cls, b_cls, M_val, b_val)


# ==============================================================================
# Главный скрипт
# ==============================================================================
if __name__ == '__main__':
    g = -1.0
    mu_min, mu_max = 0.0, 7.0
    L_min, L_max = 0.1, 10.0
    n_mu = 100
    n_L = 100
    n_workers = 10
    checkpoint_every = 200
    out_dir = 'output'

    os.makedirs(out_dir, exist_ok=True)

    mu_vals = np.linspace(mu_min, mu_max, n_mu)
    L_vals = np.linspace(L_min, L_max, n_L)
    points = [(float(mu), float(L)) for mu in mu_vals for L in L_vals]
    total = len(points)

    print(f"{'='*60}")
    print(f"Бинарная классификация: {n_mu}×{n_L} = {total} точек")
    print(f"Workers: {n_workers}")
    print(f"Тесты: M∈{{0, 0.5, 3.0}}, b∈{{0, 1.0}}")
    print(f"Ожидаемое время: ~{total * 0.003 / n_workers:.0f} сек")
    print(f"{'='*60}\n")

    t0 = time.time()
    
    # Бинарные маски: 0 = =0, 1 = >0
    mask_M = np.zeros((n_mu, n_L), dtype=np.int8)
    mask_b = np.zeros((n_mu, n_L), dtype=np.int8)
    
    # Приближенные значения (для визуализации)
    approx_M = np.zeros((n_mu, n_L))
    approx_b = np.zeros((n_mu, n_L))
    
    stats = {'=0=0': 0, '>0=0': 0, '>0>0': 0, 'fallback': 0}

    with Pool(n_workers, initializer=_init_worker, initargs=(g,)) as pool:
        for i, (mu, L, M_cls, b_cls, M_val, b_val) in enumerate(tqdm(
            pool.imap_unordered(_compute_one, points),
            total=total, desc="Classify", unit="pt", ncols=80
        )):
            i_mu = np.argmin(np.abs(mu_vals - mu))
            i_L = np.argmin(np.abs(L_vals - L))
            
            mask_M[i_mu, i_L] = 1 if M_cls == ">0" else 0
            mask_b[i_mu, i_L] = 1 if b_cls == ">0" else 0
            approx_M[i_mu, i_L] = M_val
            approx_b[i_mu, i_L] = b_val
            
            key = M_cls + b_cls
            if key in stats:
                stats[key] += 1
            else:
                stats['fallback'] += 1

            if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"\n  [CHECKPOINT] {i+1}/{total} | {rate:.1f} pt/s | ETA {eta:.1f}s")
                print(f"               M=0,b=0: {stats['=0=0']} | M>0,b=0: {stats['>0=0']} | M>0,b>0: {stats['>0>0']}")
                np.savez(os.path.join(out_dir, '36_binary_ckpt.npz'),
                         mu=mu_vals, L=L_vals, mask_M=mask_M, mask_b=mask_b,
                         approx_M=approx_M, approx_b=approx_b)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Готово! {total} точек за {elapsed:.1f} сек ({total/elapsed:.1f} pt/s)")
    print(f"M=0, b=0:  {stats['=0=0']} ({100*stats['=0=0']/total:.1f}%)")
    print(f"M>0, b=0:  {stats['>0=0']} ({100*stats['>0=0']/total:.1f}%)")
    print(f"M>0, b>0:  {stats['>0>0']} ({100*stats['>0>0']/total:.1f}%)")
    print(f"Fallback:  {stats['fallback']} ({100*stats['fallback']/total:.1f}%)")
    print(f"{'='*60}\n")

    # ======================== ВИЗУАЛИЗАЦИЯ ========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.imshow(mask_M, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[L_min, L_max, mu_min, mu_max], vmin=0, vmax=1)
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$M_{{min}} > 0$ ? (binary, {n_mu}×{n_L})')
    fig.colorbar(im, ax=ax, ticks=[0, 1], label='0 = M=0, 1 = M>0')

    ax = axes[1]
    im = ax.imshow(mask_b, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[L_min, L_max, mu_min, mu_max], vmin=0, vmax=1)
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$b_{{min}} > 0$ ? (binary, {n_mu}×{n_L})')
    fig.colorbar(im, ax=ax, ticks=[0, 1], label='0 = b=0, 1 = b>0')

    plt.tight_layout()
    png_path = os.path.join(out_dir, '36_binary_classification.png')
    plt.savefig(png_path, dpi=200)
    print(f"Сохранено: {png_path}")
    plt.show()

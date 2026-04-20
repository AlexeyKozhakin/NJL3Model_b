"""
36_micro_three_point.py

Классификация по 3 точкам с M_test = b_test = 1e-3:
  1. (0, 0)
  2. (1e-3, 0)
  3. (1e-3, 1e-3)

Логика:
  - Если Omega(0,0) < Omega(1e-3,0): M_min = 0, b_min = 0  → HIGH
  - Иначе M_min > 0:
    - Если Omega(0,0) < Omega(1e-3,1e-3): b_min = 0  → LOW_B0
    - Иначе: b_min > 0  → LOW_BPOS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int


def Omega_ren_scalar(M, b, mu, L, g, N_h=100):
    L_arr = np.array([L])
    b_arr = np.array([b])
    M_arr = np.array([M])
    M0_arr = np.array([0.0])
    b0_arr = np.array([0.0])
    
    Omega_L_val = fun_Omega_L(L_arr, b_arr, M_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_b0 = fun_Omega_L(L_arr, b_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_00 = fun_Omega_L(L_arr, b0_arr, M0_arr, N_h_p1=N_h, N_h_p2=N_h)[0, 0, 0]
    Omega_L_phys = Omega_L_val - Omega_L_b0 + Omega_L_00
    
    dU_phys = fun_dU_phys(b_arr, M_arr, N_h_p=N_h, N_h_phi=N_h)[0, 0]
    
    Omega_mu_L = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=N_h)
    Omega_mu_L_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi=0, N_h=N_h)
    Omega_mu_L_phys = Omega_mu_L - Omega_mu_L_b0 + Omega_mu_L_00
    
    tree = M**2 / (2 * g)
    return tree + dU_phys + Omega_L_phys + Omega_mu_L_phys


def classify_micro(mu, L, g, M_test=1e-3, b_test=1e-3):
    """
    Классификация по 3 точкам:
      1. (0, 0)
      2. (M_test, 0)
      3. (M_test, b_test)
    
    Логика:
      M: Omega(0,0) vs Omega(M_test,0)
      b: Omega(M_test,0) vs Omega(M_test,b_test)  (только если M>0)
    """
    val_00 = Omega_ren_scalar(0.0, 0.0, mu, L, g)
    val_M0 = Omega_ren_scalar(M_test, 0.0, mu, L, g)
    val_Mb = Omega_ren_scalar(M_test, b_test, mu, L, g)
    
    if val_00 < val_M0:
        # M = 0 → автоматически b = 0
        return 0, 0.0, 0.0, val_00, 3   # HIGH
    else:
        # M > 0: проверяем b
        if val_Mb < val_M0:
            return 2, M_test, b_test, val_Mb, 3   # LOW_BPOS (b>0)
        else:
            return 1, M_test, 0.0, val_M0, 3   # LOW_B0 (b=0)


def process_point(args):
    mu, L, g = args
    phase, M_est, b_est, val, n_evals = classify_micro(mu, L, g)
    return {
        'mu': mu, 'L': L, 'phase': phase,
        'M': M_est, 'b': b_est, 'val': val, 'n_evals': n_evals
    }


if __name__ == "__main__":
    g = -1
    n_mu, n_L = 50, 50
    mu_vals = np.linspace(0.0, 7.0, n_mu)
    L_vals = np.linspace(0.1, 4.0, n_L)
    points = [(mu, L, g) for mu in mu_vals for L in L_vals]
    
    print(f"Микро-классификация {len(points)} точек (M_test=1e-3)...")
    print(f"Предупреждение: delta_M ~ M_test^2 * mu ~ 1e-6-1e-7")
    print(f"Численная ошибка интегрирования ~ 1e-4")
    print()
    
    start = time.time()
    
    results = []
    with Pool(processes=4) as pool:
        for r in tqdm(pool.imap_unordered(process_point, points), total=len(points)):
            results.append(r)
    
    elapsed = time.time() - start
    
    # Сборка массивов
    phase_map = np.zeros((n_mu, n_L))
    M_map = np.zeros((n_mu, n_L))
    b_map = np.zeros((n_mu, n_L))
    delta_M_map = np.zeros((n_mu, n_L))
    delta_b_map = np.zeros((n_mu, n_L))
    
    for r in results:
        i = np.argmin(np.abs(mu_vals - r['mu']))
        j = np.argmin(np.abs(L_vals - r['L']))
        phase_map[i, j] = r['phase']
        M_map[i, j] = r['M']
        b_map[i, j] = r['b']
    
    # Вычислим delta для анализа
    for idx, (mu, L, _) in enumerate(points):
        i = np.argmin(np.abs(mu_vals - mu))
        j = np.argmin(np.abs(L_vals - L))
        val_00 = Omega_ren_scalar(0.0, 0.0, mu, L, g)
        val_M0 = Omega_ren_scalar(1e-3, 0.0, mu, L, g)
        val_Mb = Omega_ren_scalar(1e-3, 1e-3, mu, L, g)
        delta_M_map[i, j] = val_M0 - val_00
        delta_b_map[i, j] = val_Mb - val_M0
    
    n_high = np.sum(phase_map == 0)
    n_low_b0 = np.sum(phase_map == 1)
    n_low_bpos = np.sum(phase_map == 2)
    
    print(f"\n{'='*60}")
    print(f"Результаты (время: {elapsed:.1f} сек)")
    print(f"{'='*60}")
    print(f"HIGH:     {n_high:5d} ({100*n_high/len(points):.1f}%)")
    print(f"LOW_B0:   {n_low_b0:5d} ({100*n_low_b0/len(points):.1f}%)")
    print(f"LOW_BPOS: {n_low_bpos:5d} ({100*n_low_bpos/len(points):.1f}%)")
    print(f"\nСтатистика delta_M = Omega(1e-3,0) - Omega(0,0):")
    print(f"  min={np.min(delta_M_map):.2e}, max={np.max(delta_M_map):.2e}")
    print(f"  mean={np.mean(delta_M_map):.2e}, std={np.std(delta_M_map):.2e}")
    print(f"\nСтатистика delta_b = Omega(1e-3,1e-3) - Omega(1e-3,0):")
    print(f"  min={np.min(delta_b_map):.2e}, max={np.max(delta_b_map):.2e}")
    print(f"  mean={np.mean(delta_b_map):.2e}, std={np.std(delta_b_map):.2e}")
    print(f"{'='*60}\n")
    
    np.savez('notebooks/output/36_micro_ckpt.npz',
             mu_vals=mu_vals, L_vals=L_vals,
             phase_map=phase_map, M_map=M_map, b_map=b_map,
             delta_M_map=delta_M_map, delta_b_map=delta_b_map)
    
    # Бинарные карты классификации
    M_class = np.where(phase_map == 0, 0, 1)  # 0=M=0, 1=M>0
    b_class = np.where(phase_map == 2, 1, 0)  # 0=b=0 (вкл. M=0), 1=b>0
    
    # Визуализация: две диаграммы
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    L_grid, mu_grid = np.meshgrid(L_vals, mu_vals)
    
    # M классификация
    ax = axes[0]
    im = ax.pcolormesh(L_grid, mu_grid, M_class, shading='auto', cmap='coolwarm',
                       vmin=-0.5, vmax=1.5)
    ax.set_title(r'$M$ классификация (синий=$M=0$, красный=$M>0$)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['$M=0$', '$M>0$'])
    
    # b классификация
    ax = axes[1]
    im = ax.pcolormesh(L_grid, mu_grid, b_class, shading='auto', cmap='PiYG',
                       vmin=-0.5, vmax=1.5)
    ax.set_title(r'$b$ классификация (зелёный=$b=0$, пурпурный=$b>0$)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['$b=0$', '$b>0$'])
    
    plt.tight_layout()
    plt.savefig('notebooks/output/36_micro_classification.png', dpi=150, bbox_inches='tight')
    print("Сохранено: notebooks/output/36_micro_classification.png")
    plt.show()

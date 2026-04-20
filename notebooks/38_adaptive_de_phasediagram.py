"""
38_adaptive_de_phasediagram.py

Адаптивная фазовая диаграмма 128×128 с DE.
Алгоритм:
  1. Начальный уровень: 8×8 (шаг 16)
  2. Для каждой ячейки: если 4 угла одной фазы → заполняем всю ячейку
  3. Иначе → переходим к следующему уровень (шаг в 2 раза меньше)
  4. Уровни: 16 → 8 → 4 → 2 → 1

DE параметры: popsize=5, maxiter=30 (быстро-грубо).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from multiprocessing import Pool
import time
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def find_minimum_de(mu, L, g, M_max=10.0, b_max=10.0):
    """
    DE для поиска (M_min, b_min) при фиксированных mu, L, g.
    popsize=5, maxiter=30 — быстро-грубо.
    """
    bounds = [(0, M_max), (0, b_max)]

    result = differential_evolution(
        lambda x: Omega_ren_scalar(x[0], x[1], mu, L, g),
        bounds=bounds,
        maxiter=30,
        popsize=5,
        tol=0.01,
        polish=True,
        workers=1
    )

    M_opt, b_opt = result.x
    val_opt = result.fun

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


def worker_de(args):
    """Глобальная worker-функция для multiprocessing."""
    mu, L, g = args
    return find_minimum_de(mu, L, g)


def process_points_parallel(points, n_workers=4):
    """Параллельное вычисление DE для списка точек (mu, L, g)."""
    results = []
    with Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_de, points), total=len(points)):
            results.append(res)

    return results


def adaptive_phase_diagram(g=-1, n_workers=4):
    """
    Адаптивная фазовая диаграмма 128×128.
    Уровни: step = 16, 8, 4, 2, 1
    """
    N = 128
    phase_map = np.full((N, N), -1, dtype=np.int8)
    M_map = np.full((N, N), np.nan)
    b_map = np.full((N, N), np.nan)
    val_map = np.full((N, N), np.nan)

    # mu и L диапазоны
    mu_min, mu_max = 0.1, 7.0
    L_min, L_max = 0.1, 4.0

    mu_vals = np.linspace(mu_min, mu_max, N)
    L_vals = np.linspace(L_min, L_max, N)

    n_levels = 5  # step = 16, 8, 4, 2, 1

    for level in range(n_levels):
        n_cells = 2 ** (3 + level)  # 8, 16, 32, 64, 128
        step = N // n_cells          # 16, 8, 4, 2, 1

        print(f"\n{'='*50}")
        print(f"Уровень {level}: сетка {n_cells}×{n_cells}, шаг {step}")
        print(f"{'='*50}")

        # Собираем точки для вычисления
        points_to_compute = []
        indices = []

        for i in range(n_cells + 1):
            for j in range(n_cells + 1):
                ii = i * step
                jj = j * step
                if ii >= N or jj >= N:
                    continue
                if phase_map[ii, jj] == -1:
                    mu = mu_vals[ii]
                    L = L_vals[jj]
                    points_to_compute.append((mu, L, g))
                    indices.append((ii, jj))

        print(f"Новых точек для DE: {len(points_to_compute)}")

        if len(points_to_compute) > 0:
            # Параллельное вычисление
            start = time.time()
            results = process_points_parallel(points_to_compute, n_workers)
            elapsed = time.time() - start
            print(f"Время уровня {level}: {elapsed:.1f} сек")

            # Сохранение результатов
            for (ii, jj), (phase, M_opt, b_opt, val_opt) in zip(indices, results):
                phase_map[ii, jj] = phase
                M_map[ii, jj] = M_opt
                b_map[ii, jj] = b_opt
                val_map[ii, jj] = val_opt

        # Заполнение однородных ячеек
        n_filled = 0
        for i in range(n_cells):
            for j in range(n_cells):
                i0 = i * step
                i1 = min((i + 1) * step, N - 1)
                j0 = j * step
                j1 = min((j + 1) * step, N - 1)

                c00 = phase_map[i0, j0]
                c10 = phase_map[i1, j0]
                c01 = phase_map[i0, j1]
                c11 = phase_map[i1, j1]

                # Если все 4 угла вычислены и одинаковы — заполняем ячейку
                if c00 != -1 and c00 == c10 == c01 == c11:
                    for ii in range(i0, i1 + 1):
                        for jj in range(j0, j1 + 1):
                            if phase_map[ii, jj] == -1:
                                phase_map[ii, jj] = c00
                                n_filled += 1

        print(f"Заполнено интерполяцией: {n_filled} точек")
        print(f"Всего вычислено+заполнено: {np.sum(phase_map != -1)} / {N*N}")

    return mu_vals, L_vals, phase_map, M_map, b_map, val_map


if __name__ == "__main__":
    g = -1
    n_workers = 10

    print(f"Адаптивная фазовая диаграмма {128}×{128} с DE")
    print(f"Параметры DE: popsize=5, maxiter=30, tol=0.01")
    print(f"Уровни: 8×8 → 16×16 → 32×32 → 64×64 → 128×128")
    print()

    start_total = time.time()
    mu_vals, L_vals, phase_map, M_map, b_map, val_map = adaptive_phase_diagram(g, n_workers)
    elapsed_total = time.time() - start_total

    # Статистика
    n_high = np.sum(phase_map == 0)
    n_low_b0 = np.sum(phase_map == 1)
    n_low_bpos = np.sum(phase_map == 2)

    print(f"\n{'='*50}")
    print(f"ИТОГО: время {elapsed_total:.1f} сек")
    print(f"HIGH:     {n_high:5d} ({100*n_high/16384:.1f}%)")
    print(f"LOW_B0:   {n_low_b0:5d} ({100*n_low_b0/16384:.1f}%)")
    print(f"LOW_BPOS: {n_low_bpos:5d} ({100*n_low_bpos/16384:.1f}%)")
    print(f"{'='*50}\n")

    # Сохранение
    np.savez('notebooks/output/38_adaptive_de_128x128.npz',
             mu_vals=mu_vals, L_vals=L_vals,
             phase_map=phase_map, M_map=M_map, b_map=b_map, val_map=val_map)
    print("Данные сохранены: notebooks/output/38_adaptive_de_128x128.npz")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    L_grid, mu_grid = np.meshgrid(L_vals, mu_vals)

    # M-классификация
    M_class = np.where(phase_map == 0, 0, 1)
    ax = axes[0]
    im = ax.pcolormesh(L_grid, mu_grid, M_class, shading='auto', cmap='coolwarm',
                       vmin=-0.5, vmax=1.5)
    ax.set_title(r'$M$ классификация (синий=$M=0$, красный=$M>0$)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['$M=0$', '$M>0$'])

    # b-классификация
    b_class = np.where(phase_map == 2, 1, 0)
    ax = axes[1]
    im = ax.pcolormesh(L_grid, mu_grid, b_class, shading='auto', cmap='PiYG',
                       vmin=-0.5, vmax=1.5)
    ax.set_title(r'$b$ классификация (зелёный=$b=0$, пурпурный=$b>0$)')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel(r'$\mu$')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['$b=0$', '$b>0$'])

    plt.tight_layout()
    plt.savefig('notebooks/output/38_adaptive_de_map.png', dpi=150, bbox_inches='tight')
    print("Картинка сохранена: notebooks/output/38_adaptive_de_map.png")
    plt.show()

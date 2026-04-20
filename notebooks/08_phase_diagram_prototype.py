"""
Отладочный скрипт 08: Прототип фазовой диаграммы

Цель: для сетки (mu, L) найти минимум Omega_ren(M, b) и построить
тепловые карты M_min(mu, L) и b_min(mu, L).

Диапазоны:
- mu ∈ [0, 7]
- L  ∈ [0.1, 10]  (избегаем L=0 из-за особенности 1/L)
- M  ∈ [0, 10]
- b  ∈ [0, 10]    (используем симметрию b -> -b)

g = -1 для начала.

Алгоритм поиска минимума:
1. Грубая сетка (например 20×20 по M, b) с лог-распределением.
2. Проверка на "плоскость" (разница max-min < epsilon -> симметричная фаза M=0, b=0).
3. Локальная оптимизация Nelder-Mead из лучшей точки грубой сетки.
4. Проверка: если оптимум близок к (0,0), принудительно возвращаем (0,0).
"""

import numpy as np
import sys
import os
from functools import lru_cache
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# Кэширование функций, не зависящих от mu
# ==============================================================================
@lru_cache(maxsize=None)
def _cached_omega_L(L, b, M, N_h_p1, N_h_p2):
    return fun_Omega_L(np.array([L]), np.array([b]), np.array([M]), N_h_p1, N_h_p2).item()


@lru_cache(maxsize=None)
def _cached_dU(b, M, N_h_p, N_h_phi):
    return fun_dU_phys(np.array([b]), np.array([M]), N_h_p, N_h_phi).item()


# ==============================================================================
# Полный ренормализованный потенциал
# ==============================================================================
def Omega_ren(M, b, mu, L, g, N_h_p1=80, N_h_p2=80, N_h_phi=80, N_h_mu=80):
    """
    Omega_ren(M, b) = M^2/(2g) + dU_phys + Omega_L_phys + Omega_mu_L_phys
    """
    if M < 0:
        M = 0.0
    if b < 0:
        b = 0.0

    # Vacuum term
    vac = M**2 / (2.0 * g)

    # dU_phys
    dU_M  = _cached_dU(float(b), float(M), N_h_phi, N_h_phi)
    dU_M0 = _cached_dU(float(b), 0.0, N_h_phi, N_h_phi)
    dU_b0 = _cached_dU(0.0, 0.0, N_h_phi, N_h_phi)
    dU_phys = dU_M - dU_M0 + dU_b0

    # Omega_L_phys
    oL_M  = _cached_omega_L(float(L), float(b), float(M), N_h_p1, N_h_p2)
    oL_M0 = _cached_omega_L(float(L), float(b), 0.0, N_h_p1, N_h_p2)
    oL_b0 = _cached_omega_L(float(L), 0.0, 0.0, N_h_p1, N_h_p2)
    Omega_L_phys = oL_M - oL_M0 + oL_b0

    # Omega_mu_L_phys
    o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h_mu)
    o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h_mu)
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h_mu)
    Omega_mu_L_phys = o_mu_M - o_mu_M0 + o_mu_b0

    return vac + dU_phys + Omega_L_phys + Omega_mu_L_phys


# ==============================================================================
# Поиск минимума
# ==============================================================================
def find_minimum(mu, L, g, M_max=10.0, b_max=10.0,
                 N_grid=15, eps_flat=1e-8, eps_sym=1e-6):
    """
    Двухэтапный поиск минимума Omega_ren(M, b).
    Возвращает (M_min, b_min, Omega_min).
    """
    # --- Этап 1: грубая сетка ---
    # Логарифмическое распределение по M (больше точек у 0)
    M_grid = np.concatenate(([0.0], np.logspace(-2, np.log10(M_max), N_grid - 1)))
    b_grid = np.concatenate(([0.0], np.logspace(-2, np.log10(b_max), N_grid - 1)))

    best_val = np.inf
    best_M, best_b = 0.0, 0.0
    values = []

    for M in M_grid:
        for b in b_grid:
            val = Omega_ren(M, b, mu, L, g)
            values.append(val)
            if val < best_val:
                best_val = val
                best_M = M
                best_b = b

    values = np.array(values)
    val_range = values.max() - values.min()

    # Если поверхность почти плоская — симметричная фаза
    if val_range < eps_flat:
        return 0.0, 0.0, values.min()

    # --- Этап 2: локальная оптимизация ---
    def obj(x):
        return Omega_ren(x[0], x[1], mu, L, g)

    result = minimize(
        obj,
        x0=[best_M, best_b],
        method='Nelder-Mead',
        bounds=[(0.0, M_max), (0.0, b_max)],
        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
    )

    M_opt, b_opt = result.x
    val_opt = result.fun

    # Если оптимум незначительно отличается от (0,0) — принудительно симметричная фаза
    val_origin = Omega_ren(0.0, 0.0, mu, L, g)
    if abs(val_opt - val_origin) < eps_sym:
        return 0.0, 0.0, val_origin

    # Округление очень малых значений до 0
    if M_opt < 1e-4:
        M_opt = 0.0
    if b_opt < 1e-4:
        b_opt = 0.0

    return float(M_opt), float(b_opt), float(val_opt)


# ==============================================================================
# Визуализация
# ==============================================================================
def plot_phase_diagrams(L_vals, mu_vals, M_map, b_map, g, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Теперь L по X, mu по Y
    L_grid, MU_grid = np.meshgrid(L_vals, mu_vals, indexing='ij')

    ax = axes[0]
    pcm = ax.pcolormesh(L_grid, MU_grid, M_map.T, shading='auto', cmap='viridis')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$M_{{min}}(\mu, L)$, g={g}')
    fig.colorbar(pcm, ax=ax, label=r'$M_{min}$')

    ax = axes[1]
    pcm = ax.pcolormesh(L_grid, MU_grid, b_map.T, shading='auto', cmap='plasma')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$b_{{min}}(\mu, L)$, g={g}')
    fig.colorbar(pcm, ax=ax, label=r'$b_{min}$')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_sections(L_fixed_list, mu_vals, section_data, g, out_path):
    """
    section_data: dict {L_value: M_min_array}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # M_min(mu)
    ax = axes[0]
    for L_val in L_fixed_list:
        ax.plot(mu_vals, section_data[L_val]['M'], '-o', markersize=3, label=f'L={L_val:.2f}')
    ax.set_xlabel(r'$\mu$', fontsize=12)
    ax.set_ylabel(r'$M_{min}$', fontsize=12)
    ax.set_title(f'$M_{{min}}(\mu)$ при фиксированных $L$, g={g}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # b_min(mu)
    ax = axes[1]
    for L_val in L_fixed_list:
        ax.plot(mu_vals, section_data[L_val]['b'], '-s', markersize=3, label=f'L={L_val:.2f}')
    ax.set_xlabel(r'$\mu$', fontsize=12)
    ax.set_ylabel(r'$b_{min}$', fontsize=12)
    ax.set_title(f'$b_{{min}}(\mu)$ при фиксированных $L$, g={g}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    g = -1.0
    n_mu = 50
    n_L = 50
    mu_vals = np.linspace(0, 7, n_mu)
    # Логарифмическая сетка по L: больше точек у L -> 0
    L_vals = np.logspace(-1, 1, n_L)  # 0.1 .. 10

    # Карты индексируются как (i_mu, j_L), но график будет L по X, mu по Y
    M_map = np.zeros((n_mu, n_L))
    b_map = np.zeros((n_mu, n_L))

    print(f"\nРасчет фазовой диаграммы: g={g}")
    print(f"Сетка: {n_mu} x {n_L} = {n_mu*n_L} точек")
    print(f"L сетка (log): [{L_vals.min():.3f}, {L_vals.max():.3f}]")
    print("-" * 50)

    total = n_mu * n_L
    with tqdm(total=total, desc="Phase diagram") as pbar:
        for i, mu in enumerate(mu_vals):
            for j, L in enumerate(L_vals):
                M_min, b_min, _ = find_minimum(mu, L, g)
                M_map[i, j] = M_min
                b_map[i, j] = b_min
                pbar.update(1)

    plot_phase_diagrams(L_vals, mu_vals, M_map, b_map, g,
                        os.path.join(out_dir, '08_phase_diagram_g-1.png'))

    # Сечения при фиксированных L
    L_fixed_list = [0.15, 0.3, 0.5, 1.0, 2.0, 5.0]
    section_data = {}
    actual_L_list = []
    print("\nРасчет сечений при фиксированных L...")
    for L_val in L_fixed_list:
        # Найдем ближайший индекс j в L_vals
        j = np.argmin(np.abs(L_vals - L_val))
        actual_L = float(L_vals[j])
        actual_L_list.append(actual_L)
        section_data[actual_L] = {'M': M_map[:, j], 'b': b_map[:, j]}

    plot_sections(actual_L_list, mu_vals, section_data, g,
                  os.path.join(out_dir, '08_sections_g-1.png'))

    print("\n>>> Готово!")
    print(f"    Уникальных M_min: {len(np.unique(np.round(M_map, 4)))}")
    print(f"    Уникальных b_min: {len(np.unique(np.round(b_map, 4)))}")
    print(f"    M_max на диаграмме: {M_map.max():.3f}")
    print(f"    b_max на диаграмме: {b_map.max():.3f}")


if __name__ == "__main__":
    main()

"""
Скрипт 25: 4D-сетка (10x10x10x10) + кубическая интерполяция + поиск минимумов.
Область: mu=[2,4], L=[0.1,4], b=[0,10], M=[0,10]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

# ==============================================================================
# Параметры сеток
# ==============================================================================
N_mu = 10
N_L = 10
N_b = 10
N_M = 10

mu_vals = np.linspace(2.0, 4.0, N_mu)
L_vals = np.linspace(0.1, 4.0, N_L)
b_vals = np.linspace(0.0, 10.0, N_b)
M_vals = np.linspace(0.0, 10.0, N_M)

g = -1.0
N_h = 80

print("Построение 4D-сетки Omega(mu, L, b, M)...")
print(f"Размер сетки: {N_mu}x{N_L}x{N_b}x{N_M} = {N_mu*N_L*N_b*N_M} точек")

# ==============================================================================
# 1. Предвычисление Omega_L(b,M) и dU(b,M) для каждой пары (L)
# ==============================================================================
Omega_L_00 = fun_Omega_L(
    np.array([L_vals[0]]), np.array([0.0]), np.array([0.0]), N_h, N_h
)[0, 0, 0]

# Кэш Omega_L и dU для каждого L
Omega_L_cache = {}
dU_cache = {}
for L in L_vals:
    Omega_L_full = fun_Omega_L(np.array([L]), b_vals, M_vals, N_h, N_h)[0, :, :]
    Omega_L_0M = fun_Omega_L(np.array([L]), b_vals, np.array([0.0]), N_h, N_h)[0, :, 0][:, None]
    Omega_L_cache[L] = Omega_L_full - Omega_L_0M + Omega_L_00

    dU_full = fun_dU_phys(b_vals, M_vals, N_h, N_h)
    dU_0M = fun_dU_phys(b_vals, np.array([0.0]), N_h, N_h)[:, 0][:, None]
    dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0]
    dU_cache[L] = dU_full - dU_0M + dU_b0

# ==============================================================================
# 2. Построение полного 4D-массива Omega
# ==============================================================================
Omega_4D = np.empty((N_mu, N_L, N_b, N_M), dtype=np.float64)

count = 0
total_points = N_mu * N_L
for i, mu in enumerate(mu_vals):
    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L_vals[0], 0.0, 0.0, 0, N_h)
    for j, L in enumerate(L_vals):
        Omega_L_grid = Omega_L_cache[L]
        dU_grid = dU_cache[L]
        vac_grid = (M_vals[None, :]**2) / (2.0 * g)

        # Omega_mu_L для всех (b, M) при данном (mu, L)
        Omega_mu_grid = np.empty((N_b, N_M), dtype=np.float64)
        for k, b in enumerate(b_vals):
            o_mu_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h)
            for l, M in enumerate(M_vals):
                o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h)
                Omega_mu_grid[k, l] = o_mu_M - o_mu_b0 + Omega_mu_L_00

        Omega_4D[i, j, :, :] = vac_grid + dU_grid + Omega_L_grid + Omega_mu_grid
        count += 1
        if count % 10 == 0:
            print(f"  Готово {count}/{total_points} (mu,L)-срезов...")

print("\n4D-сетка построена!")

# ==============================================================================
# 3. Кубическая интерполяция
# ==============================================================================
print("Построение интерполятора...")
interp = RegularGridInterpolator(
    (mu_vals, L_vals, b_vals, M_vals),
    Omega_4D,
    method='cubic',
    bounds_error=False,
    fill_value=None
)
print("Интерполятор готов.")

# ==============================================================================
# 4. Поиск минимумов на плотной сетке mu, L
# ==============================================================================
N_mu_fine = 60
N_L_fine = 60
mu_fine = np.linspace(2.0, 4.0, N_mu_fine)
L_fine = np.linspace(0.1, 4.0, N_L_fine)

M_min_grid = np.empty((N_mu_fine, N_L_fine), dtype=np.float64)
b_min_grid = np.empty((N_mu_fine, N_L_fine), dtype=np.float64)

print(f"\nПоиск минимумов на сетке {N_mu_fine}x{N_L_fine}...")

for i, mu in enumerate(mu_fine):
    for j, L in enumerate(L_fine):
        # Грубый поиск на сетке 10x10 через интерполятор
        b_grid_search = np.linspace(0, 10, 20)
        M_grid_search = np.linspace(0, 10, 20)
        best_val = np.inf
        best_b, best_M = 0.0, 0.0
        for b in b_grid_search:
            for M in M_grid_search:
                val = float(interp([[mu, L, b, M]])[0])
                if val < best_val:
                    best_val = val
                    best_b, best_M = b, M

        # Уточнение через minimize на интерполяторе
        res = minimize(
            lambda x: float(interp([[mu, L, x[0], x[1]]])[0]),
            x0=[best_M, best_b],
            method='Nelder-Mead',
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
        )
        M_opt, b_opt = res.x
        if M_opt < 1e-4:
            M_opt = 0.0
        if b_opt < 1e-4:
            b_opt = 0.0
        M_min_grid[i, j] = M_opt
        b_min_grid[i, j] = b_opt

    if (i + 1) % 10 == 0:
        print(f"  Готово {i+1}/{N_mu_fine} строк...")

print("\nМинимумы найдены!")

# ==============================================================================
# 5. Визуализация
# ==============================================================================
out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
im = ax.imshow(M_min_grid.T, origin='lower', aspect='auto',
               extent=[L_fine[0], L_fine[-1], mu_fine[0], mu_fine[-1]],
               cmap='viridis')
ax.set_xlabel(r'$L$', fontsize=12)
ax.set_ylabel(r'$\mu$', fontsize=12)
ax.set_title(f'$M_{{min}}(\mu, L)$, g={g} (spline interp)')
fig.colorbar(im, ax=ax, label=r'$M_{min}$')

ax = axes[1]
im = ax.imshow(b_min_grid.T, origin='lower', aspect='auto',
               extent=[L_fine[0], L_fine[-1], mu_fine[0], mu_fine[-1]],
               cmap='plasma')
ax.set_xlabel(r'$L$', fontsize=12)
ax.set_ylabel(r'$\mu$', fontsize=12)
ax.set_title(f'$b_{{min}}(\mu, L)$, g={g} (spline interp)')
fig.colorbar(im, ax=ax, label=r'$b_{min}$')

plt.tight_layout()
png_path = os.path.join(out_dir, '25_spline_interp_diagram.png')
plt.savefig(png_path, dpi=200)
print(f"\nСохранено: {png_path}")

# Статистика
print(f"\nСтатистика b_min:")
print(f"  b>0.1: {(b_min_grid > 0.1).sum()} / {N_mu_fine*N_L_fine} точек")
print(f"  b_max: {b_min_grid.max():.3f}, b_min: {b_min_grid.min():.3f}")

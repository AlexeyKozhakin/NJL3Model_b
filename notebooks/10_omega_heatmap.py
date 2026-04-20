"""
notebooks/10_omega_heatmap.py
==============================
Тепловая карта полного термодинамического потенциала Ω(b, M)
при фиксированных μ и L.

Пример использования:
    mu = 5.0
    L  = 0.25
    b_range = (0, 10)
    M_range = (0, 10)
    N_b = 200
    N_M = 200
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

# =============================================================================
# ПАРАМЕТРЫ ПОЛЬЗОВАТЕЛЯ
# =============================================================================
mu = 5.0          # химический потенциал
L = 0.25          # размер системы (длина)
g = -1.0          # константа взаимодействия NJL
phi = 0           # фаза (в данной работе φ = 0)

b_range = (4, 5.5)   # диапазон по b
M_range = (1, 2)   # диапазон по M

N_b = 200         # число точек по b
N_M = 200         # число точек по M

# параметры интегрирования (совпадают с main.py / adaptive diagram)
N_h_p1 = 100
N_h_p2 = 100
N_h_p = 100
N_h_phi = 100
N_h_mu = 100      # для fun_Omega_L_mu_int

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 1. Сетки
# =============================================================================
b_vals = np.linspace(b_range[0], b_range[1], N_b)
M_vals = np.linspace(M_range[0], M_range[1], N_M)

print(f"Параметры: mu={mu}, L={L}, g={g}, phi={phi}")
print(f"Сетка: {N_b} x {N_M}  (b: {b_range}, M: {M_range})")

# =============================================================================
# 2. Предвычисление Omega_L_00 (константа, не зависит от b, M)
# =============================================================================
Omega_L_00 = fun_Omega_L(
    np.array([L]), np.array([0.0]), np.array([0.0]),
    N_h_p1=N_h_p1, N_h_p2=N_h_p2
)[0, 0, 0]

Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi=phi, N_h=N_h_mu)

# =============================================================================
# 3. Вычисление полного потенциала на сетке (построчно по b, векторно по M)
# =============================================================================
Omega_total = np.zeros((N_b, N_M))
Omega_mu_L_grid = np.zeros((N_b, N_M))
Omega_L_phys_grid = np.zeros((N_b, N_M))
dU_phys_grid = np.zeros((N_b, N_M))

print("Вычисление Omega на сетке ...")
for i, b in enumerate(tqdm(b_vals, desc="b-loop")):
    # --- Omega_L для данного b и всего M_vals ---
    Omega_L_full = fun_Omega_L(
        np.array([L]), np.array([b]), M_vals,
        N_h_p1=N_h_p1, N_h_p2=N_h_p2
    )[0, 0, :]  # shape (N_M,)
    Omega_L_0M = fun_Omega_L(
        np.array([L]), np.array([b]), np.array([0.0]),
        N_h_p1=N_h_p1, N_h_p2=N_h_p2
    )[0, 0, 0]
    Omega_L_phys_grid[i, :] = Omega_L_full - Omega_L_0M + Omega_L_00

    # --- dU для данного b и всего M_vals ---
    dU_row = fun_dU_phys(
        np.array([b]), M_vals,
        N_h_p=N_h_p, N_h_phi=N_h_phi
    )[0, :]  # shape (N_M,)
    dU_phys_grid[i, :] = dU_row

    # --- Omega_mu_L (скалярная numba-функция) ---
    Omega_mu_L_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi=phi, N_h=N_h_mu)
    for j, M in enumerate(M_vals):
        o_mu_L = (
            fun_Omega_L_mu_int(mu, L, b, M, phi=phi, N_h=N_h_mu)
            - Omega_mu_L_b0
            + Omega_mu_L_00
        )
        Omega_mu_L_grid[i, j] = o_mu_L

        Omega_total[i, j] = (
            M**2 / (2.0 * g)
            + dU_row[j]
            + Omega_L_phys_grid[i, j]
            + o_mu_L
        )

# =============================================================================
# 4. Поиск минимума
# =============================================================================
idx_min = np.unravel_index(np.argmin(Omega_total), Omega_total.shape)
b_min = b_vals[idx_min[0]]
M_min = M_vals[idx_min[1]]
Omega_min = Omega_total[idx_min]

print(f"\nМинимум потенциала:")
print(f"  b_min = {b_min:.4f}")
print(f"  M_min = {M_min:.4f}")
print(f"  Omega_min = {Omega_min:.6f}")

# =============================================================================
# 5. Визуализация heatmaps
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.ravel()

# --- общая функция для отрисовки heatmap ---
def plot_heatmap(ax, data, title, cmap="viridis", mark_min=True):
    b_edges = np.linspace(b_range[0], b_range[1], N_b + 1)
    M_edges = np.linspace(M_range[0], M_range[1], N_M + 1)
    mesh = ax.pcolormesh(b_edges, M_edges, data.T, shading="auto", cmap=cmap)
    if mark_min:
        ax.scatter(b_min, M_min, color="white", marker="x", s=120, linewidths=2.5, label="min")
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(r"$M$")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=r"$\Omega$")
    ax.legend(loc="upper right")

plot_heatmap(axes[0], Omega_total,
             rf"$\Omega_{{\rm total}}$, $\mu={mu}$, $L={L}$, $g={g}$",
             cmap="viridis")

plot_heatmap(axes[1], Omega_mu_L_grid,
             rf"$\Omega_{{\mu L}}$, $\mu={mu}$, $L={L}$",
             cmap="plasma")

plot_heatmap(axes[2], dU_phys_grid,
             rf"$dU$, $g={g}$",
             cmap="cividis")

plot_heatmap(axes[3], Omega_L_phys_grid,
             rf"$\Omega_L^{{\rm phys}}$, $L={L}$",
             cmap="inferno")

plt.tight_layout()
out_path = os.path.join(output_dir, f"10_heatmap_mu{mu}_L{L}_g{g}.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nСохранено: {out_path}")

# =============================================================================
# 6. Срезы: b=0 и M=M_min
# =============================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

idx_b0 = np.argmin(np.abs(b_vals))
axes2[0].plot(M_vals, Omega_total[idx_b0, :], "k-", lw=1.5, label=r"$\Omega_{\rm total}(b=0, M)$")
axes2[0].plot(M_vals, (M_vals**2)/(2*g) + dU_phys_grid[idx_b0, :] + Omega_L_phys_grid[idx_b0, :],
              "--", label=r"$M^2/(2g)+dU+\Omega_L$")
axes2[0].plot(M_vals, Omega_mu_L_grid[idx_b0, :], ":", label=r"$\Omega_{\mu L}$")
axes2[0].axvline(M_min, color="gray", ls="-.", alpha=0.5)
axes2[0].set_xlabel(r"$M$")
axes2[0].set_ylabel(r"$\Omega$")
axes2[0].set_title(rf"Срез $b \approx {b_vals[idx_b0]:.3f}$")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(b_vals, Omega_total[:, idx_min[1]], "k-", lw=1.5, label=r"$\Omega_{\rm total}(b, M_{\rm min})$")
axes2[1].plot(b_vals, (M_vals[idx_min[1]]**2)/(2*g) + dU_phys_grid[:, idx_min[1]] + Omega_L_phys_grid[:, idx_min[1]],
              "--", label=r"$M^2/(2g)+dU+\Omega_L$")
axes2[1].plot(b_vals, Omega_mu_L_grid[:, idx_min[1]], ":", label=r"$\Omega_{\mu L}$")
axes2[1].axvline(b_min, color="gray", ls="-.", alpha=0.5)
axes2[1].set_xlabel(r"$b$")
axes2[1].set_ylabel(r"$\Omega$")
axes2[1].set_title(rf"Срез $M = {M_min:.3f}$")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = os.path.join(output_dir, f"10_slices_mu{mu}_L{L}_g{g}.png")
plt.savefig(out_path2, dpi=200, bbox_inches="tight")
print(f"Сохранено: {out_path2}")

plt.show()

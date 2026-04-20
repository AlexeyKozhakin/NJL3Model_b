"""
Тест dense grid с параллельным Omega_mu_L через ProcessPoolExecutor.
"""
import sys
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

mu, L, g = 4.0, 1.0, -1.0
N_h = 80
N = 100

b_vals = np.linspace(0, 10, N)
M_vals = np.linspace(0, 10, N)

# --- Omega_L и dU (векторизованно, быстро) ---
Omega_L_00 = fun_Omega_L(np.array([L]), np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0, 0]
Omega_L_full = fun_Omega_L(np.array([L]), b_vals, M_vals, N_h, N_h)[0, :, :]
Omega_L_0M = fun_Omega_L(np.array([L]), b_vals, np.array([0.0]), N_h, N_h)[0, :, 0][:, None]
Omega_L_grid = Omega_L_full - Omega_L_0M + Omega_L_00

dU_full = fun_dU_phys(b_vals, M_vals, N_h, N_h)
dU_0M = fun_dU_phys(b_vals, np.array([0.0]), N_h, N_h)[:, 0][:, None]
dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0]
dU_grid = dU_full - dU_0M + dU_b0

Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h)

# --- Параллельное построение Omega_mu_grid ---
def compute_row(args):
    b, M_vals, mu, L, N_h, Omega_mu_L_00 = args
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h)
    row = np.empty(len(M_vals), dtype=np.float64)
    for j, M in enumerate(M_vals):
        o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h)
        row[j] = o_mu_M - o_mu_b0 + Omega_mu_L_00
    return row

print("=== Parallel Omega_mu_L ===")
t0 = time.time()
args_list = [(b, M_vals, mu, L, N_h, Omega_mu_L_00) for b in b_vals]

with ProcessPoolExecutor() as executor:
    rows = list(executor.map(compute_row, args_list))

Omega_mu_grid = np.stack(rows, axis=0)
t_parallel = time.time() - t0
print(f"Parallel build time: {t_parallel:.3f}s")

# --- Полный потенциал ---
vac_grid = (M_vals[None, :]**2) / (2.0 * g)
total = vac_grid + dU_grid + Omega_L_grid + Omega_mu_grid
idx = np.unravel_index(np.argmin(total), total.shape)
print(f"Grid min: b={b_vals[idx[0]]:.3f}, M={M_vals[idx[1]]:.3f}, Omega={total[idx]:.6f}")
print(f"Omega at (0,0): {total[0,0]:.6f}")
print(f"\nEstimated for 6000 points (no refinement): {t_parallel * 6000 / 3600:.1f} hours")

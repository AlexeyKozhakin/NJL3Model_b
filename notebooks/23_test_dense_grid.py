"""
Тест: строим dense grid 100x100 для mu=4, L=1.
Omega_L и dU — векторизованно.
Omega_mu_L — скалярно через numba (как сейчас).
Смотрим время и попадает ли минимум в яму b=4, M=0.15.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

mu, L, g = 4.0, 1.0, -1.0
N_h = 80
N = 100

b_vals = np.linspace(0, 10, N)
M_vals = np.linspace(0, 10, N)

print("=== Building dense grid ===")
t0 = time.time()

# Omega_L (векторизованно)
Omega_L_00 = fun_Omega_L(np.array([L]), np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0, 0]
Omega_L_full = fun_Omega_L(np.array([L]), b_vals, M_vals, N_h, N_h)[0, :, :]
Omega_L_0M = fun_Omega_L(np.array([L]), b_vals, np.array([0.0]), N_h, N_h)[0, :, 0][:, None]
Omega_L_grid = Omega_L_full - Omega_L_0M + Omega_L_00

# dU (векторизованно)
dU_full = fun_dU_phys(b_vals, M_vals, N_h, N_h)
dU_0M = fun_dU_phys(b_vals, np.array([0.0]), N_h, N_h)[:, 0][:, None]
dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0]
dU_grid = dU_full - dU_0M + dU_b0

# Omega_mu_L (скалярно) — основное бутылочное горлышко
Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h)
Omega_mu_grid = np.empty((N, N), dtype=np.float64)
for i, b in enumerate(b_vals):
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h)
    for j, M in enumerate(M_vals):
        o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h)
        Omega_mu_grid[i, j] = o_mu_M - o_mu_b0 + Omega_mu_L_00

t_build = time.time() - t0
print(f"Build time: {t_build:.3f}s")

# Полный потенциал
vac_grid = (M_vals[None, :]**2) / (2.0 * g)
total = vac_grid + dU_grid + Omega_L_grid + Omega_mu_grid

idx = np.unravel_index(np.argmin(total), total.shape)
print(f"\nGrid min: b={b_vals[idx[0]]:.3f}, M={M_vals[idx[1]]:.3f}, Omega={total[idx]:.6f}")
print(f"Omega at (0,0): {total[0,0]:.6f}")

# Многоуровневое уточнение: берем окрестность ±2 шага и строим 50x50
b_idx, m_idx = idx
db = 2
if b_idx < db: db = b_idx
dm = 2
if m_idx < dm: dm = m_idx

b_lo = b_vals[b_idx - db]
b_hi = b_vals[min(b_idx + db, N-1)]
m_lo = M_vals[m_idx - dm]
m_hi = M_vals[min(m_idx + dm, N-1)]

print(f"\nRefinement region: b=[{b_lo:.3f}, {b_hi:.3f}], M=[{m_lo:.3f}, {m_hi:.3f}]")

N_ref = 50
b_ref = np.linspace(b_lo, b_hi, N_ref)
M_ref = np.linspace(m_lo, m_hi, N_ref)

t0 = time.time()
Omega_L_ref = fun_Omega_L(np.array([L]), b_ref, M_ref, N_h, N_h)[0, :, :]
Omega_L_0M_ref = fun_Omega_L(np.array([L]), b_ref, np.array([0.0]), N_h, N_h)[0, :, 0][:, None]
Omega_L_ref_grid = Omega_L_ref - Omega_L_0M_ref + Omega_L_00

dU_ref = fun_dU_phys(b_ref, M_ref, N_h, N_h)
dU_0M_ref = fun_dU_phys(b_ref, np.array([0.0]), N_h, N_h)[:, 0][:, None]
dU_ref_grid = dU_ref - dU_0M_ref + dU_b0

Omega_mu_ref = np.empty((N_ref, N_ref), dtype=np.float64)
for i, b in enumerate(b_ref):
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h)
    for j, M in enumerate(M_ref):
        o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h)
        Omega_mu_ref[i, j] = o_mu_M - o_mu_b0 + Omega_mu_L_00

t_ref = time.time() - t0
vac_ref = (M_ref[None, :]**2) / (2.0 * g)
total_ref = vac_ref + dU_ref_grid + Omega_L_ref_grid + Omega_mu_ref
idx_ref = np.unravel_index(np.argmin(total_ref), total_ref.shape)
print(f"Refined min: b={b_ref[idx_ref[0]]:.4f}, M={M_ref[idx_ref[1]]:.4f}, Omega={total_ref[idx_ref]:.6f}")
print(f"Refinement time: {t_ref:.3f}s")
print(f"\nTotal per point: {t_build + t_ref:.3f}s")

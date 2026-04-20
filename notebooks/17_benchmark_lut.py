"""
Benchmark: старая Omega_ren vs LUT-версия.
Проверяем correctness и скорость.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Старая версия
import importlib.util
_spec = importlib.util.spec_from_file_location('ad09', os.path.join(os.path.dirname(__file__), '09_adaptive_phase_diagram.py'))
ad09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad09)
Omega_ren = ad09.Omega_ren

# Новая LUT-версия
from functions.Omega_ren_lut import OmegaRenLUT

mu, L, g = 4.0, 1.0, -1.0
N_h = 80

print("=== Build LUT ===")
t0 = time.time()
lut = OmegaRenLUT(mu, L, g, N_M=200, N_b=200,
                    N_h_p1=N_h, N_h_p2=N_h,
                    N_h_p=N_h, N_h_phi=N_h, N_h_mu=N_h)
t_lut = time.time() - t0
print(f"LUT build time: {t_lut:.3f}s")

# === Точность ===
test_points = [
    (0.0, 0.0),
    (0.1, 4.0),
    (1.5, 2.0),
    (3.0, 5.0),
    (5.0, 0.5),
    (0.05, 3.99),
    (0.15, 4.01),
]

print("\n=== Correctness check ===")
max_diff = 0.0
for M, b in test_points:
    old = Omega_ren(M, b, mu, L, g, N_h, N_h, N_h, N_h)
    new = lut(M, b)
    diff = abs(old - new)
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 5e-4 else "FAIL"
    print(f"M={M:.3f}, b={b:.3f}: old={old:.8f}, new={new:.8f}, diff={diff:.2e} [{status}]")

print(f"\nMax diff: {max_diff:.2e}")

# === Скорость: мини-heatmap ===
print("\n=== Mini-heatmap benchmark (40x40) ===")
N_b = 40
N_M = 40
b_vals = np.linspace(0, 10, N_b)
M_vals = np.linspace(0, 10, N_M)

n_calls = N_b * N_M

t0 = time.time()
for b in b_vals:
    for M in M_vals:
        _ = Omega_ren(M, b, mu, L, g, N_h, N_h, N_h, N_h)
t_old = time.time() - t0

t0 = time.time()
for b in b_vals:
    for M in M_vals:
        _ = lut(M, b)
t_new = time.time() - t0

print(f"Old 40x40 grid: {t_old:.3f}s  ({t_old/n_calls*1000:.3f} ms/call)")
print(f"New 40x40 grid: {t_new:.3f}s  ({t_new/n_calls*1000:.3f} ms/call)")
print(f"Grid speedup: {t_old/t_new:.1f}x")
print(f"Including LUT build: {t_old/(t_new + t_lut):.1f}x")

# === DE benchmark ===
print("\n=== DE optimization benchmark ===")
from scipy.optimize import differential_evolution

obj_old = lambda x: Omega_ren(x[0], x[1], mu, L, g, N_h, N_h, N_h, N_h)
obj_new = lambda x: lut(x[0], x[1])

t0 = time.time()
res_old = differential_evolution(obj_old, bounds=[(0,10),(0,10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
t_de_old = time.time() - t0

t0 = time.time()
res_new = differential_evolution(obj_new, bounds=[(0,10),(0,10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
t_de_new = time.time() - t0

print(f"Old DE: {t_de_old:.2f}s -> M={res_old.x[0]:.4f}, b={res_old.x[1]:.4f}")
print(f"New DE: {t_de_new:.2f}s -> M={res_new.x[0]:.4f}, b={res_new.x[1]:.4f}")
print(f"DE speedup: {t_de_old/t_de_new:.1f}x")

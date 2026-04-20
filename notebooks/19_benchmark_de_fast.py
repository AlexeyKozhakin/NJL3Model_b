"""
Быстрый бенчмарк DE на Omega_ren_fast.
Сравниваем old exact vs new numba-скалярный.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.optimize import differential_evolution
import importlib.util

# Старая версия
_spec = importlib.util.spec_from_file_location('ad09', os.path.join(os.path.dirname(__file__), '09_adaptive_phase_diagram.py'))
ad09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad09)
Omega_ren_old = ad09.Omega_ren

# Новая numba-скалярная
from functions.Omega_ren_fast import Omega_ren_fast

mu, L, g = 4.0, 1.0, -1.0
N_h = 80

# === Warm-up ===
print("Warming up numba...")
_ = Omega_ren_fast(0.1, 4.0, mu, L, g, N_h, N_h, N_h, N_h)
_ = Omega_ren_old(0.1, 4.0, mu, L, g, N_h, N_h, N_h, N_h)

# === Single DE benchmark ===
print("\n=== DE benchmark (mu=4, L=1) ===")

obj_old = lambda x: Omega_ren_old(x[0], x[1], mu, L, g, N_h, N_h, N_h, N_h)
obj_new = lambda x: Omega_ren_fast(x[0], x[1], mu, L, g, N_h, N_h, N_h, N_h)

t0 = time.time()
res_old = differential_evolution(obj_old, bounds=[(0, 10), (0, 10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
t_old = time.time() - t0

t0 = time.time()
res_new = differential_evolution(obj_new, bounds=[(0, 10), (0, 10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
t_new = time.time() - t0

print(f"Old DE: {t_old:.3f}s -> M={res_old.x[0]:.4f}, b={res_old.x[1]:.4f}, Omega={res_old.fun:.6f}")
print(f"New DE: {t_new:.3f}s -> M={res_new.x[0]:.4f}, b={res_new.x[1]:.4f}, Omega={res_new.fun:.6f}")
print(f"Speedup: {t_old/t_new:.1f}x")

# === Multiple points benchmark ===
test_points = [
    (2.0, 0.5),
    (3.0, 1.0),
    (4.0, 1.0),
    (5.0, 0.25),
    (6.0, 2.0),
]

print("\n=== Multiple points DE benchmark ===")
t_old_sum = 0.0
t_new_sum = 0.0
for mu_i, L_i in test_points:
    t0 = time.time()
    differential_evolution(lambda x: Omega_ren_old(x[0], x[1], mu_i, L_i, g, N_h, N_h, N_h, N_h),
                           bounds=[(0, 10), (0, 10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
    t_old_sum += time.time() - t0

    t0 = time.time()
    differential_evolution(lambda x: Omega_ren_fast(x[0], x[1], mu_i, L_i, g, N_h, N_h, N_h, N_h),
                           bounds=[(0, 10), (0, 10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
    t_new_sum += time.time() - t0

print(f"Old 5xDE: {t_old_sum:.2f}s")
print(f"New 5xDE: {t_new_sum:.2f}s")
print(f"Speedup: {t_old_sum/t_new_sum:.1f}x")
print(f"\nEstimated time for 6000 points (old): {t_old_sum/5*6000/60:.0f} min")
print(f"Estimated time for 6000 points (new): {t_new_sum/5*6000/60:.0f} min")

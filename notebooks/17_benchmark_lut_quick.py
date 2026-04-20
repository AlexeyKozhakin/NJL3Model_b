"""
Быстрый бенчмарк LUT: только точность и скорость LUT grid.
Старый DE пропущен, т.к. известно что он медленный.
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
    (2.5, 7.5),
    (7.5, 2.5),
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

# === Скорость: один вызов LUT ===
print("\n=== Single LUT call speed ===")
n = 5000
t0 = time.time()
for _ in range(n):
    _ = lut(0.15, 4.0)
t_single = time.time() - t0
print(f"{n} LUT calls: {t_single:.3f}s  ({t_single/n*1000:.4f} ms/call)")

# === Скорость: DE с LUT ===
print("\n=== DE with LUT ===")
from scipy.optimize import differential_evolution
t0 = time.time()
res = differential_evolution(lambda x: lut(x[0], x[1]), bounds=[(0,10),(0,10)], maxiter=10, popsize=4, polish=True, tol=1e-6)
t_de = time.time() - t0
print(f"DE with LUT: {t_de:.3f}s -> M={res.x[0]:.4f}, b={res.x[1]:.4f}, Omega={res.fun:.6f}")

print("\nDone.")

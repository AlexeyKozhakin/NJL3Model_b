"""
Benchmark: старая Omega_ren vs новая Omega_ren_fast.
Проверяем correctness и замеряем скорость.
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

# Новая версия
from functions.Omega_ren_fast import Omega_ren_fast

mu, L, g = 4.0, 1.0, -1.0
N_h = 80

# === Точность ===
test_points = [
    (0.0, 0.0),
    (0.1, 4.0),
    (1.5, 2.0),
    (3.0, 5.0),
    (5.0, 0.5),
]

print("=== Correctness check ===")
for M, b in test_points:
    old = Omega_ren(M, b, mu, L, g, N_h, N_h, N_h, N_h)
    new = Omega_ren_fast(M, b, mu, L, g, N_h, N_h, N_h, N_h)
    diff = abs(old - new)
    status = "OK" if diff < 1e-6 else "FAIL"
    print(f"M={M:.2f}, b={b:.2f}: old={old:.8f}, new={new:.8f}, diff={diff:.2e} [{status}]")

# === Скорость: один вызов ===
M, b = 0.1, 4.0
print("\n=== Warm-up / compile ===")
_ = Omega_ren_fast(M, b, mu, L, g, N_h, N_h, N_h, N_h)

print("=== Single-call benchmark ===")
n_calls = 50

t0 = time.time()
for _ in range(n_calls):
    _ = Omega_ren(M, b, mu, L, g, N_h, N_h, N_h, N_h)
t_old = time.time() - t0

t0 = time.time()
for _ in range(n_calls):
    _ = Omega_ren_fast(M, b, mu, L, g, N_h, N_h, N_h, N_h)
t_new = time.time() - t0

print(f"Old: {t_old:.3f}s for {n_calls} calls  ({t_old/n_calls*1000:.2f} ms/call)")
print(f"New: {t_new:.3f}s for {n_calls} calls  ({t_new/n_calls*1000:.2f} ms/call)")
print(f"Speedup: {t_old/t_new:.1f}x")

# === Скорость: мини-heatmap ===
print("\n=== Mini-heatmap benchmark (40x40) ===")
N_b = 40
N_M = 40
b_vals = np.linspace(0, 10, N_b)
M_vals = np.linspace(0, 10, N_M)

t0 = time.time()
for b in b_vals:
    for M in M_vals:
        _ = Omega_ren(M, b, mu, L, g, N_h, N_h, N_h, N_h)
t_old_grid = time.time() - t0

t0 = time.time()
for b in b_vals:
    for M in M_vals:
        _ = Omega_ren_fast(M, b, mu, L, g, N_h, N_h, N_h, N_h)
t_new_grid = time.time() - t0

print(f"Old 40x40 grid: {t_old_grid:.2f}s")
print(f"New 40x40 grid: {t_new_grid:.2f}s")
print(f"Grid speedup: {t_old_grid/t_new_grid:.1f}x")

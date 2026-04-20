"""
Бенчмарк grid-search + multistart NM на одной точке.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
_spec = importlib.util.spec_from_file_location('gs', os.path.join(os.path.dirname(__file__), '20_gridsearch_final.py'))
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)
find_minimum_gridsearch = gs.find_minimum_gridsearch

mu, L, g = 4.0, 1.0, -1.0
N_h = 80

print("=== Grid-search + NM benchmark ===")
t0 = time.time()
M_opt, b_opt, val_opt = find_minimum_gridsearch(mu, L, g, N_grid=40, n_starts=3,
                                                N_h_p1=N_h, N_h_p2=N_h,
                                                N_h_p=N_h, N_h_phi=N_h, N_h_mu=N_h)
t = time.time() - t0
print(f"Result: M={M_opt:.4f}, b={b_opt:.4f}, Omega={val_opt:.6f}")
print(f"Time: {t:.3f}s")

# Compare with different grid sizes
for N in [30, 40, 50]:
    t0 = time.time()
    M_opt, b_opt, val_opt = find_minimum_gridsearch(mu, L, g, N_grid=N, n_starts=3,
                                                    N_h_p1=N_h, N_h_p2=N_h,
                                                    N_h_p=N_h, N_h_phi=N_h, N_h_mu=N_h)
    t = time.time() - t0
    print(f"N={N:2d}: M={M_opt:.4f}, b={b_opt:.4f}, Omega={val_opt:.6f}, time={t:.3f}s")

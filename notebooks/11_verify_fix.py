"""
Быстрая проверка: при mu=4, L=1 истинный минимум лежит при b>0,
но find_minimum с N_grid=12 его не ловит.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
spec = importlib.util.spec_from_file_location('ad09', os.path.join(os.path.dirname(__file__), '09_adaptive_phase_diagram.py'))
ad09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad09)
find_minimum = ad09.find_minimum
Omega_ren = ad09.Omega_ren

mu = 4.0
L = 1.0
g = -1.0

# ============================================================================
# 1. Быстрый heatmap (80x80, N_h=50 для скорости)
# ============================================================================
N = 80
b_vals = np.linspace(0, 10, N)
M_vals = np.linspace(0, 10, N)

print("Считаем быстрый heatmap 80x80 (N_h=50)...")
total = np.zeros((N, N))
for i, b in enumerate(b_vals):
    for j, M in enumerate(M_vals):
        total[i, j] = Omega_ren(M, b, mu, L, g,
                                N_h_p1=50, N_h_p2=50,
                                N_h_phi=50, N_h_mu=50)

idx = np.unravel_index(np.argmin(total), total.shape)
b_grid_min = b_vals[idx[0]]
M_grid_min = M_vals[idx[1]]
Omega_grid_min = total[idx]

print(f"\n>>> Heatmap minimum: b={b_grid_min:.3f}, M={M_grid_min:.3f}, Omega={Omega_grid_min:.6f}")
print(f"    Omega at (0,0):  {Omega_ren(0.0, 0.0, mu, L, g, 50, 50, 50, 50):.6f}")

# ============================================================================
# 2. find_minimum с разными N_grid
# ============================================================================
print("\nПроверка find_minimum с разной густотой сетки:")
for N_grid in [12, 20, 30, 50]:
    M_opt, b_opt, val_opt = find_minimum(mu, L, g, N_grid=N_grid)
    print(f"  N_grid={N_grid:2d}:  M={M_opt:.4f}, b={b_opt:.4f}, Omega={val_opt:.6f}")

# ============================================================================
# 3. Визуализация
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
im = ax.imshow(total.T, origin='lower', aspect='auto',
                extent=[b_vals[0], b_vals[-1], M_vals[0], M_vals[-1]],
                cmap='viridis')
ax.scatter(b_grid_min, M_grid_min, color='white', marker='x', s=120, lw=2, label='grid min')
ax.set_xlabel('b')
ax.set_ylabel('M')
ax.set_title(f'Ω(b,M) при μ={mu}, L={L}, g={g}\n(heatmap min: b={b_grid_min:.2f}, M={M_grid_min:.2f})')
fig.colorbar(im, ax=ax)
ax.legend()

ax = axes[1]
# срез b=0
ax.plot(M_vals, total[0, :], 'k-', lw=2, label='b=0')
# срез через найденный b_min
b_idx = np.argmin(np.abs(b_vals - b_grid_min))
ax.plot(M_vals, total[b_idx, :], 'r--', lw=2, label=f'b≈{b_vals[b_idx]:.2f} (grid min)')
ax.axvline(M_grid_min, color='gray', ls=':')
ax.set_xlabel('M')
ax.set_ylabel('Ω')
ax.set_title('Срезы Ω(M)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'output', '11_verify_mu4_L1.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200)
print(f"\nSaved: {out_path}")

"""
Быстрый proof-of-concept: uniform grid 20x20, mu=[2,5], L=[0.1,3].
Каждая точка - differential_evolution vs Nelder-Mead.
"""
import numpy as np
import sys, os
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib.util
_spec = importlib.util.spec_from_file_location('ad09', os.path.join(os.path.dirname(__file__), '09_adaptive_phase_diagram.py'))
ad09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad09)
Omega_ren = ad09.Omega_ren

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mu_vals = np.linspace(2.0, 5.0, 20)
L_vals = np.linspace(0.1, 3.0, 20)

M_nm = np.zeros((20, 20))
b_nm = np.zeros((20, 20))
M_de = np.zeros((20, 20))
b_de = np.zeros((20, 20))

print("Считаем 20x20 = 400 точек...")
for i, mu in enumerate(mu_vals):
    for j, L in enumerate(L_vals):
        # Nelder-Mead (текущий find_minimum)
        M_opt1, b_opt1, _ = ad09.find_minimum(mu, L, -1.0)
        M_nm[i, j] = M_opt1
        b_nm[i, j] = b_opt1

        # Differential Evolution
        obj = lambda x: Omega_ren(x[0], x[1], mu, L, -1.0)
        res = differential_evolution(obj, bounds=[(0, 10), (0, 10)], maxiter=15, popsize=4, polish=True, tol=1e-6)
        M_de[i, j] = res.x[0] if res.x[0] > 1e-4 else 0.0
        b_de[i, j] = res.x[1] if res.x[1] > 1e-4 else 0.0

    if (i+1) % 5 == 0:
        print(f"  Готово {i+1}/20")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

for ax, data, title in [
    (axes[0,0], b_nm, 'b_min Nelder-Mead'),
    (axes[0,1], b_de, 'b_min Diff. Evolution'),
    (axes[1,0], M_nm, 'M_min Nelder-Mead'),
    (axes[1,1], M_de, 'M_min Diff. Evolution')
]:
    im = ax.imshow(data.T, origin='lower', aspect='auto',
                   extent=[L_vals[0], L_vals[-1], mu_vals[0], mu_vals[-1]],
                   cmap='plasma' if 'b_min' in title else 'viridis')
    ax.set_xlabel('L')
    ax.set_ylabel('mu')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'output', '13_quick_de_proof.png')
plt.savefig(out_path, dpi=200)
print(f"\nSaved: {out_path}")

# key comparison
print("\nРазличия b>0:")
print(f"  NM: {(b_nm > 0.1).sum()} точек из 400")
print(f"  DE: {(b_de > 0.1).sum()} точек из 400")

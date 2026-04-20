"""
Быстрый тест: адаптивная диаграмма с глобальным оптимизатором
(differential_evolution) на области mu=[2,5], L=[0.1,3].
Цель: убедиться, что узкие ямы (например, при mu=4, L=1) теперь ловятся.
"""
import numpy as np
import sys
import os
from scipy.optimize import differential_evolution
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib.util
_spec = importlib.util.spec_from_file_location('ad09', os.path.join(os.path.dirname(__file__), '09_adaptive_phase_diagram.py'))
ad09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ad09)
Omega_ren = ad09.Omega_ren

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def find_minimum_de(mu, L, g, M_max=10.0, b_max=10.0):
    """Глобальный поиск через differential_evolution (быстрый режим)."""
    obj = lambda x: Omega_ren(x[0], x[1], mu, L, g)
    result = differential_evolution(
        obj,
        bounds=[(0.0, M_max), (0.0, b_max)],
        maxiter=20,
        popsize=5,
        polish=True,
        tol=1e-6
    )
    M_opt, b_opt = result.x
    if M_opt < 1e-4:
        M_opt = 0.0
    if b_opt < 1e-4:
        b_opt = 0.0
    return float(M_opt), float(b_opt), float(result.fun)


class TestDiagram:
    def __init__(self, g, mu_range, L_range, max_depth=2,
                 eps_M=0.3, eps_b=0.2, M_max=10.0, b_max=10.0,
                 init_mu_div=25, init_L_div=25):
        self.g = g
        self.mu_min, self.mu_max = mu_range
        self.L_min, self.L_max = L_range
        self.max_depth = max_depth
        self.eps_M = eps_M
        self.eps_b = eps_b
        self.M_max = M_max
        self.b_max = b_max
        self.init_mu_div = init_mu_div
        self.init_L_div = init_L_div

        self.cache = {}
        self.points_mu = []
        self.points_L = []
        self.points_M = []
        self.points_b = []
        self.computed_count = 0

    def _get_point(self, mu, L):
        key = (round(float(mu), 10), round(float(L), 10))
        if key not in self.cache:
            M_min, b_min, _ = find_minimum_de(mu, L, self.g,
                                              M_max=self.M_max, b_max=self.b_max)
            self.cache[key] = (M_min, b_min)
            self.computed_count += 1
            if self.computed_count % 10 == 0:
                print(f"  Computed {self.computed_count} points...")
        return self.cache[key]

    def _cell_variation(self, mu1, mu2, L1, L2):
        coords = [
            (mu1, L1), (mu1, L2), (mu2, L1), (mu2, L2),
            ((mu1 + mu2) / 2, (L1 + L2) / 2)
        ]
        M_vals = []
        b_vals = []
        for mu, L in coords:
            M_min, b_min = self._get_point(mu, L)
            M_vals.append(M_min)
            b_vals.append(b_min)
        return max(M_vals) - min(M_vals), max(b_vals) - min(b_vals)

    def _refine(self, mu1, mu2, L1, L2, depth):
        var_M, var_b = self._cell_variation(mu1, mu2, L1, L2)
        if depth >= self.max_depth or (var_M < self.eps_M and var_b < self.eps_b):
            mu_c = (mu1 + mu2) / 2
            L_c = (L1 + L2) / 2
            M_c, b_c = self._get_point(mu_c, L_c)
            self.points_mu.append(mu_c)
            self.points_L.append(L_c)
            self.points_M.append(M_c)
            self.points_b.append(b_c)
            return
        mu_mid = (mu1 + mu2) / 2
        L_mid = (L1 + L2) / 2
        self._refine(mu1, mu_mid, L1, L_mid, depth + 1)
        self._refine(mu1, mu_mid, L_mid, L2, depth + 1)
        self._refine(mu_mid, mu2, L1, L_mid, depth + 1)
        self._refine(mu_mid, mu2, L_mid, L2, depth + 1)

    def compute(self):
        print(f"Test diagram: g={self.g}, max_depth={self.max_depth}")
        print(f"Area: mu=[{self.mu_min},{self.mu_max}], L=[{self.L_min},{self.L_max}]")
        print("-" * 50)
        mu_edges = np.linspace(self.mu_min, self.mu_max, self.init_mu_div + 1)
        L_edges = np.linspace(self.L_min, self.L_max, self.init_L_div + 1)
        for i in range(self.init_mu_div):
            for j in range(self.init_L_div):
                self._refine(mu_edges[i], mu_edges[i + 1],
                             L_edges[j], L_edges[j + 1], depth=0)
        print(f"  DONE. Unique computed points: {len(self.points_mu)}")
        return (np.array(self.points_mu), np.array(self.points_L),
                np.array(self.points_M), np.array(self.points_b))


def plot_diagram(points_mu, points_L, points_M, points_b, g, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tri = Triangulation(points_L, points_mu)

    ax = axes[0]
    tpc = ax.tripcolor(tri, points_M, shading='gouraud', cmap='viridis')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$M_{{min}}(\mu, L)$, g={g} (DE fixed)')
    fig.colorbar(tpc, ax=ax, label=r'$M_{min}$')

    ax = axes[1]
    tpc = ax.tripcolor(tri, points_b, shading='gouraud', cmap='plasma')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$b_{{min}}(\mu, L)$, g={g} (DE fixed)')
    fig.colorbar(tpc, ax=ax, label=r'$b_{min}$')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    diagram = TestDiagram(
        g=-1.0,
        mu_range=(2.0, 5.0),
        L_range=(0.1, 3.0),
        max_depth=2,
        eps_M=0.3,
        eps_b=0.2,
        init_mu_div=25,
        init_L_div=25
    )
    mu_pts, L_pts, M_pts, b_pts = diagram.compute()
    print(f"M_max: {M_pts.max():.3f}, M_min: {M_pts.min():.3f}")
    print(f"b_max: {b_pts.max():.3f}, b_min: {b_pts.min():.3f}")

    plot_diagram(mu_pts, L_pts, M_pts, b_pts, -1.0,
                 os.path.join(out_dir, '12_test_fixed_DE.png'))


if __name__ == "__main__":
    main()

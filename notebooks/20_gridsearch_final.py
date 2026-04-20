"""
Финальная адаптивная диаграмма с grid-search + multistart Nelder-Mead.
Grid-search использует векторизованный Omega_ren (быстрый на NumPy).
Топ-3 точки сетки запускают NM для точной доводки.
"""
import os
import sys
import numpy as np
from functools import lru_cache
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ==============================================================================
# Кэширование
# ==============================================================================
@lru_cache(maxsize=None)
def _cached_omega_L(L, b, M, N_h_p1, N_h_p2):
    return fun_Omega_L(np.array([L]), np.array([b]), np.array([M]), N_h_p1, N_h_p2).item()


@lru_cache(maxsize=None)
def _cached_dU(b, M, N_h_p, N_h_phi):
    return fun_dU_phys(np.array([b]), np.array([M]), N_h_p, N_h_phi).item()


def Omega_ren(M, b, mu, L, g, N_h_p1=80, N_h_p2=80, N_h_phi=80, N_h_mu=80):
    if M < 0.0:
        M = 0.0
    if b < 0.0:
        b = 0.0
    vac = M * M / (2.0 * g)

    dU_M  = _cached_dU(float(b), float(M), N_h_phi, N_h_phi)
    dU_M0 = _cached_dU(float(b), 0.0, N_h_phi, N_h_phi)
    dU_b0 = _cached_dU(0.0, 0.0, N_h_phi, N_h_phi)
    dU_phys = dU_M - dU_M0 + dU_b0

    oL_M  = _cached_omega_L(float(L), float(b), float(M), N_h_p1, N_h_p2)
    oL_M0 = _cached_omega_L(float(L), float(b), 0.0, N_h_p1, N_h_p2)
    oL_b0 = _cached_omega_L(float(L), 0.0, 0.0, N_h_p1, N_h_p2)
    Omega_L_phys = oL_M - oL_M0 + oL_b0

    o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h_mu)
    o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h_mu)
    o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h_mu)
    Omega_mu_L_phys = o_mu_M - o_mu_M0 + o_mu_b0

    return vac + dU_phys + Omega_L_phys + Omega_mu_L_phys


# ==============================================================================
# Grid-search + multistart NM
# ==============================================================================
def find_minimum_gridsearch(mu, L, g, M_max=10.0, b_max=10.0,
                            N_grid=40, n_starts=3,
                            N_h_p1=80, N_h_p2=80, N_h_p=80, N_h_phi=80, N_h_mu=80):
    """
    1. Быстрый grid-search на векторизованном Omega_ren.
    2. Топ-n_starts точек запускают Nelder-Mead.
    3. Возвращается лучший.
    """
    M_vals = np.linspace(0.0, M_max, N_grid)
    b_vals = np.linspace(0.0, b_max, N_grid)

    # Предвычисление Omega_L и dU на всей сетке (векторизованно)
    Omega_L_00 = fun_Omega_L(np.array([L]), np.array([0.0]), np.array([0.0]), N_h_p1, N_h_p2)[0, 0, 0]
    Omega_L_full = fun_Omega_L(np.array([L]), b_vals, M_vals, N_h_p1, N_h_p2)[0, :, :]  # (N_b, N_M)
    Omega_L_0M = fun_Omega_L(np.array([L]), b_vals, np.array([0.0]), N_h_p1, N_h_p2)[0, :, 0][:, None]  # (N_b, 1)
    Omega_L_grid = Omega_L_full - Omega_L_0M + Omega_L_00

    dU_full = fun_dU_phys(b_vals, M_vals, N_h_p, N_h_phi)  # (N_b, N_M)
    dU_0M = fun_dU_phys(b_vals, np.array([0.0]), N_h_p, N_h_phi)[:, 0][:, None]  # (N_b, 1)
    dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h_p, N_h_phi)[0, 0]
    dU_grid = dU_full - dU_0M + dU_b0

    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h_mu)

    # Omega_mu_L — скалярно для каждой точки сетки (numba быстрое)
    grid = np.empty((N_grid, N_grid), dtype=np.float64)
    for i, b in enumerate(b_vals):
        omega_mu_L_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h_mu)
        for j, M in enumerate(M_vals):
            o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h_mu)
            omega_mu_L = o_mu_M - omega_mu_L_b0 + Omega_mu_L_00
            vac = M * M / (2.0 * g)
            grid[i, j] = vac + dU_grid[i, j] + Omega_L_grid[i, j] + omega_mu_L

    # Топ-n_starts точек
    flat = grid.ravel()
    best_indices = np.argsort(flat)[:n_starts]
    candidates = []
    for idx in best_indices:
        i = idx // N_grid
        j = idx % N_grid
        candidates.append((grid[i, j], M_vals[j], b_vals[i]))

    # Multistart Nelder-Mead
    best_val = np.inf
    best_x = (candidates[0][1], candidates[0][2])
    for val, M0, b0 in candidates:
        res = minimize(
            lambda x: Omega_ren(x[0], x[1], mu, L, g, N_h_p1, N_h_p2, N_h_phi, N_h_mu),
            x0=[M0, b0],
            method='Nelder-Mead',
            bounds=[(0.0, M_max), (0.0, b_max)],
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    M_opt, b_opt = best_x
    if M_opt < 1e-4:
        M_opt = 0.0
    if b_opt < 1e-4:
        b_opt = 0.0
    return float(M_opt), float(b_opt), float(best_val)


# ==============================================================================
# Адаптивная диаграмма
# ==============================================================================
class AdaptivePhaseDiagram:
    def __init__(self, g, mu_range, L_range, max_depth=2,
                 eps_M=0.3, eps_b=0.2, M_max=10.0, b_max=10.0,
                 init_mu_div=50, init_L_div=50,
                 checkpoint_every=500, out_dir='output'):
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
        self.checkpoint_every = checkpoint_every
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.cache = {}
        self.points_mu = []
        self.points_L = []
        self.points_M = []
        self.points_b = []
        self.computed_count = 0
        self.last_printed = 0
        self.last_checkpoint = 0

    def _get_point(self, mu, L):
        key = (round(float(mu), 10), round(float(L), 10))
        if key not in self.cache:
            M_min, b_min, _ = find_minimum_gridsearch(mu, L, self.g,
                                                       M_max=self.M_max, b_max=self.b_max)
            self.cache[key] = (M_min, b_min)
            self.computed_count += 1
            if self.computed_count - self.last_printed >= 50:
                print(f"  Computed {self.computed_count} points...")
                self.last_printed = self.computed_count
                self._maybe_checkpoint()
        return self.cache[key]

    def _maybe_checkpoint(self):
        if self.computed_count - self.last_checkpoint >= self.checkpoint_every:
            self.save_checkpoint()

    def save_checkpoint(self, final=False):
        if len(self.points_mu) == 0:
            return
        tag = 'final' if final else f"ckpt_{self.computed_count}"
        npz_path = os.path.join(self.out_dir, f'20_adaptive_{tag}_g_m1_0.npz')
        np.savez(npz_path,
                 mu=np.array(self.points_mu),
                 L=np.array(self.points_L),
                 M=np.array(self.points_M),
                 b=np.array(self.points_b))
        print(f"  [CHECKPOINT] Saved {len(self.points_mu)} grid points -> {npz_path}")

        png_path = os.path.join(self.out_dir, f'20_adaptive_{tag}_g_m1_0.png')
        self._plot(png_path)
        print(f"  [CHECKPOINT] Saved plot -> {png_path}")
        self.last_checkpoint = self.computed_count

    def _plot(self, out_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tri = Triangulation(np.array(self.points_L), np.array(self.points_mu))

        ax = axes[0]
        tpc = ax.tripcolor(tri, np.array(self.points_M), shading='gouraud', cmap='viridis')
        ax.set_xlabel(r'$L$', fontsize=12)
        ax.set_ylabel(r'$\mu$', fontsize=12)
        ax.set_title(f'$M_{{min}}(\mu, L)$, g={self.g} (grid+NM, {len(self.points_mu)} pts)')
        fig.colorbar(tpc, ax=ax, label=r'$M_{min}$')

        ax = axes[1]
        tpc = ax.tripcolor(tri, np.array(self.points_b), shading='gouraud', cmap='plasma')
        ax.set_xlabel(r'$L$', fontsize=12)
        ax.set_ylabel(r'$\mu$', fontsize=12)
        ax.set_title(f'$b_{{min}}(\mu, L)$, g={self.g} (grid+NM, {len(self.points_mu)} pts)')
        fig.colorbar(tpc, ax=ax, label=r'$b_{min}$')

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

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
        max_est = self.init_mu_div * self.init_L_div * (4 ** self.max_depth)
        print(f"Адаптивная диаграмма (grid+NM): g={self.g}, max_depth={self.max_depth}")
        print(f"Начальная сетка: {self.init_mu_div}x{self.init_L_div}")
        print(f"Оценочное макс число точек: ~{max_est}")
        print("-" * 50)

        mu_edges = np.linspace(self.mu_min, self.mu_max, self.init_mu_div + 1)
        L_edges = np.linspace(self.L_min, self.L_max, self.init_L_div + 1)

        for i in range(self.init_mu_div):
            for j in range(self.init_L_div):
                self._refine(mu_edges[i], mu_edges[i + 1],
                             L_edges[j], L_edges[j + 1], depth=0)

        if self.computed_count > self.last_printed:
            print(f"  Computed {self.computed_count} points... DONE")

        self.save_checkpoint(final=True)
        return (np.array(self.points_mu), np.array(self.points_L),
                np.array(self.points_M), np.array(self.points_b))


# ==============================================================================
# Main
# ==============================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')

    diagram = AdaptivePhaseDiagram(
        g=-1.0,
        mu_range=(0.0, 7.0),
        L_range=(0.1, 10.0),
        max_depth=2,
        eps_M=0.3,
        eps_b=0.2,
        M_max=10.0,
        b_max=10.0,
        init_mu_div=50,
        init_L_div=50,
        checkpoint_every=500,
        out_dir=out_dir
    )

    mu_pts, L_pts, M_pts, b_pts = diagram.compute()

    print(f"\n>>> Готово!")
    print(f"    Вычислено уникальных точек: {len(mu_pts)}")
    print(f"    M_max: {M_pts.max():.3f}, M_min: {M_pts.min():.3f}")
    print(f"    b_max: {b_pts.max():.3f}, b_min: {b_pts.min():.3f}")


if __name__ == "__main__":
    main()

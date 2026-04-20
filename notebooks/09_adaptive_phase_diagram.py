"""
Отладочный скрипт 09: Адаптивная фазовая диаграмма (quadtree)

Цель: построить фазовую диаграмму M_min(mu, L) и b_min(mu, L)
с адаптивным измельчением сетки. Точки сгущаются там, где M или b
сильно меняются (фазовые границы), и разрежены в гладких областях.

Алгоритм:
1. Начальная грубая сетка (равномерная по mu, логарифмическая по L).
2. Для каждой ячейки оцениваем max-min разброс M_min и b_min.
3. Если разброс превышает порог — делим ячейку на 4 и рекурсируем.
4. Кэшируем все вычисленные точки, чтобы не считать дважды.
5. Визуализация через matplotlib.tripcolor (неструктурированная сетка).
"""

import numpy as np
import sys
import os
from functools import lru_cache
from scipy.optimize import differential_evolution
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ==============================================================================
# Кэширование базовых функций
# ==============================================================================
@lru_cache(maxsize=None)
def _cached_omega_L(L, b, M, N_h_p1, N_h_p2):
    return fun_Omega_L(np.array([L]), np.array([b]), np.array([M]), N_h_p1, N_h_p2).item()


@lru_cache(maxsize=None)
def _cached_dU(b, M, N_h_p, N_h_phi):
    return fun_dU_phys(np.array([b]), np.array([M]), N_h_p, N_h_phi).item()


# ==============================================================================
# Полный потенциал
# ==============================================================================
def Omega_ren(M, b, mu, L, g, N_h_p1=80, N_h_p2=80, N_h_phi=80, N_h_mu=80):
    if M < 0:
        M = 0.0
    if b < 0:
        b = 0.0

    vac = M**2 / (2.0 * g)

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
# Поиск минимума — глобальный (differential_evolution)
# ==============================================================================
class _DEObjective:
    """Picklable wrapper for Omega_ren (required for workers=-1 on macOS)."""
    __slots__ = ('mu', 'L', 'g')
    def __init__(self, mu, L, g):
        self.mu = mu
        self.L = L
        self.g = g
    def __call__(self, x):
        return Omega_ren(x[0], x[1], self.mu, self.L, self.g)


def find_minimum(mu, L, g, M_max=10.0, b_max=10.0,
                 maxiter=10, popsize=4):
    """Используем differential_evolution для надёжного поиска глобального минимума."""
    obj = _DEObjective(mu, L, g)
    result = differential_evolution(
        obj,
        bounds=[(0.0, M_max), (0.0, b_max)],
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        tol=1e-6,
        workers=-1
    )
    M_opt, b_opt = result.x
    if M_opt < 1e-4:
        M_opt = 0.0
    if b_opt < 1e-4:
        b_opt = 0.0
    return float(M_opt), float(b_opt), float(result.fun)


# ==============================================================================
# Адаптивное измельчение с кэшированием
# ==============================================================================
class AdaptivePhaseDiagram:
    def __init__(self, g, mu_range, L_range, max_depth=3,
                 eps_M=0.5, eps_b=0.3, M_max=10.0, b_max=10.0,
                 init_mu_div=3, init_L_div=3):
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

        # Кэш точек: ключ (mu, L) -> (M_min, b_min)
        self.cache = {}
        self.points_mu = []
        self.points_L = []
        self.points_M = []
        self.points_b = []
        self.computed_count = 0
        self.last_printed = 0

    def _get_point(self, mu, L):
        key = (round(float(mu), 10), round(float(L), 10))
        if key not in self.cache:
            M_min, b_min, _ = find_minimum(mu, L, self.g,
                                           M_max=self.M_max, b_max=self.b_max)
            self.cache[key] = (M_min, b_min)
            self.computed_count += 1
            if self.computed_count - self.last_printed >= 50:
                print(f"  Computed {self.computed_count} points...")
                self.last_printed = self.computed_count
        return self.cache[key]

    def _cell_variation(self, mu1, mu2, L1, L2):
        """Возвращает (var_M, var_b) для ячейки [mu1,mu2] x [L1,L2]."""
        # Только 4 угла + центр = 5 точек (для скорости)
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

        # Если ячейка достаточно гладкая или достигнута макс глубина — сохраняем центр
        if depth >= self.max_depth or (var_M < self.eps_M and var_b < self.eps_b):
            mu_c = (mu1 + mu2) / 2
            L_c = (L1 + L2) / 2
            M_c, b_c = self._get_point(mu_c, L_c)
            self.points_mu.append(mu_c)
            self.points_L.append(L_c)
            self.points_M.append(M_c)
            self.points_b.append(b_c)
            return

        # Делим на 4 подъячейки
        mu_mid = (mu1 + mu2) / 2
        L_mid = (L1 + L2) / 2
        self._refine(mu1, mu_mid, L1, L_mid, depth + 1)
        self._refine(mu1, mu_mid, L_mid, L2, depth + 1)
        self._refine(mu_mid, mu2, L1, L_mid, depth + 1)
        self._refine(mu_mid, mu2, L_mid, L2, depth + 1)

    def compute(self):
        max_est = self.init_mu_div * self.init_L_div * (4 ** self.max_depth)
        print(f"Адаптивная диаграмма: g={self.g}, max_depth={self.max_depth}")
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

        return (np.array(self.points_mu), np.array(self.points_L),
                np.array(self.points_M), np.array(self.points_b))


# ==============================================================================
# Визуализация
# ==============================================================================
def plot_adaptive_diagram(points_mu, points_L, points_M, points_b, g, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Треугольная сетка Делоне
    tri = Triangulation(points_L, points_mu)

    ax = axes[0]
    tpc = ax.tripcolor(tri, points_M, shading='gouraud', cmap='viridis')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$M_{{min}}(\mu, L)$, g={g} (адаптивная сетка)')
    fig.colorbar(tpc, ax=ax, label=r'$M_{min}$')

    ax = axes[1]
    tpc = ax.tripcolor(tri, points_b, shading='gouraud', cmap='plasma')
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'$b_{{min}}(\mu, L)$, g={g} (адаптивная сетка)')
    fig.colorbar(tpc, ax=ax, label=r'$b_{min}$')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mesh(points_mu, points_L, g, out_path):
    """Визуализация самой адаптивной сетки (где лежат точки)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points_L, points_mu, c='black', s=3, alpha=0.6)
    ax.set_xlabel(r'$L$', fontsize=12)
    ax.set_ylabel(r'$\mu$', fontsize=12)
    ax.set_title(f'Адаптивная сетка точек, g={g}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    g = -1.0
    mu_range = (0.0, 7.0)
    L_range = (0.1, 10.0)

    diagram = AdaptivePhaseDiagram(
        g=g,
        mu_range=mu_range,
        L_range=L_range,
        max_depth=2,
        eps_M=0.3,
        eps_b=0.2,
        M_max=10.0,
        b_max=10.0,
        init_mu_div=50,
        init_L_div=50
    )

    mu_pts, L_pts, M_pts, b_pts = diagram.compute()

    print(f"\n>>> Готово!")
    print(f"    Вычислено уникальных точек: {len(mu_pts)}")
    print(f"    M_max: {M_pts.max():.3f}, M_min: {M_pts.min():.3f}")
    print(f"    b_max: {b_pts.max():.3f}, b_min: {b_pts.min():.3f}")

    g_str = str(g).replace('.', '_').replace('-', 'm')
    plot_adaptive_diagram(mu_pts, L_pts, M_pts, b_pts, g,
                          os.path.join(out_dir, f'09_adaptive_diagram_{g_str}.png'))
    plot_mesh(mu_pts, L_pts, g,
              os.path.join(out_dir, f'09_adaptive_mesh_{g_str}.png'))


if __name__ == "__main__":
    main()

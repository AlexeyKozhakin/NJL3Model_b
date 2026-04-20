"""
================================================================================
Параллельная адаптивная фазовая диаграмма (multiprocessing + DE)
================================================================================
Запуск:  python 30_parallel_adaptive.py

Особенности:
- Параллелизация на уровне точек (mu, L) через multiprocessing.Pool
- DE внутри каждой точки однопоточный (избегаем overhead)
- Адаптивная логика (refinement) последовательная — она мгновенная
- Прогресс-бар tqdm с ETA
- Checkpoint каждые N точек
- По умолчанию: 10x10 base, max_depth=2, 4 workers
- Поддерживает 100x100 base при увеличении init_mu_div/init_L_div

Память: ~200-400 MB на процесс (4 процесса = <2 GB)
================================================================================
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from multiprocessing import Pool
from tqdm import tqdm

# Добавляем корень проекта в путь для импорта functions/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scipy.optimize import differential_evolution
from functions.Omega_L import fun_Omega_L
from functions.Omega_mu_L import fun_Omega_L_mu_int
from functions.dU import fun_dU_phys


# ==============================================================================
# DE-оптимизатор (копия из 27_adaptive_de_10x10_rounded.py с канонизацией)
# ==============================================================================
from functools import lru_cache

@lru_cache(maxsize=None)
def _cached_omega_L(L, b, M, N_h_p1, N_h_p2):
    return fun_Omega_L(L, b, M, N_h_p1, N_h_p2).item()

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


def find_minimum(mu, L, g, M_max=10.0, b_max=10.0,
                 maxiter=15, popsize=5):
    """DE для одной точки (mu, L). Однопоточный — вызывается из воркера."""
    obj = lambda x: Omega_ren(x[0], x[1], mu, L, g)
    result = differential_evolution(
        obj,
        bounds=[(0.0, M_max), (0.0, b_max)],
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        tol=1e-6
    )
    M_opt, b_opt = result.x
    val_opt = result.fun

    # Канонизация высокой фазы (M≈0)
    M_canon_tol = 0.1
    if M_opt < M_canon_tol:
        M_opt = 0.0
        b_opt = 0.0
    else:
        if b_opt < 1e-4:
            b_opt = 0.0

    return float(M_opt), float(b_opt), float(val_opt)


# ==============================================================================
# Параллельный воркер
# ==============================================================================
_worker_cache = {}

def _init_worker(g):
    """Инициализация воркера — прогрев Numba JIT одним холостым вызовом."""
    _worker_cache['g'] = g
    # Прогрев
    find_minimum(4.0, 1.0, g)


def _compute_one(args):
    """Функция-воркер: считает одну точку (mu, L)."""
    mu, L = args
    g = _worker_cache.get('g', -1.0)
    M_min, b_min, val = find_minimum(mu, L, g)
    return (mu, L, M_min, b_min, val)


# ==============================================================================
# Параллельное вычисление списка точек с прогресс-баром
# ==============================================================================
def parallel_compute(points, g, n_workers=4, checkpoint_every=100,
                     out_dir='output', tag='parallel'):
    """
    Параллельно считает список точек [(mu, L), ...].
    Возвращает словарь {(mu, L): (M_min, b_min)}.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    total = len(points)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Параллельное вычисление: {total} точек, {n_workers} workers")
    print(f"Оценочное время: ~{total * 0.13 / n_workers:.0f} сек (~{total * 0.13 / n_workers / 60:.1f} мин)")
    print(f"{'='*60}\n")

    with Pool(n_workers, initializer=_init_worker, initargs=(g,)) as pool:
        for i, result in enumerate(tqdm(
            pool.imap_unordered(_compute_one, points),
            total=total,
            desc="Points",
            unit="pt",
            ncols=80
        )):
            mu, L, M_min, b_min, val = result
            results[(mu, L)] = (M_min, b_min)

            # Checkpoint
            if (i + 1) % checkpoint_every == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(f"\n  [CHECKPOINT] {i+1}/{total} | rate={rate:.1f} pt/s | ETA={remaining/60:.1f} min")
                _save_checkpoint(results, out_dir, tag, i + 1)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Готово! {total} точек за {elapsed:.1f} сек ({total/elapsed:.1f} pt/s)")
    print(f"{'='*60}\n")
    return results


def _save_checkpoint(results, out_dir, tag, count):
    """Сохраняет промежуточные результаты в NPZ."""
    mu_arr = np.array([k[0] for k in results.keys()])
    L_arr = np.array([k[1] for k in results.keys()])
    M_arr = np.array([v[0] for v in results.values()])
    b_arr = np.array([v[1] for v in results.values()])
    npz_path = os.path.join(out_dir, f'{tag}_ckpt_{count}.npz')
    np.savez(npz_path, mu=mu_arr, L=L_arr, M=M_arr, b=b_arr)


# ==============================================================================
# Адаптивное измельчение (последовательная логика + параллельные точки)
# ==============================================================================
class ParallelAdaptivePhaseDiagram:
    def __init__(self, g=-1.0, mu_range=(0, 7), L_range=(0.1, 10),
                 max_depth=2, eps_M=0.3, eps_b=0.2,
                 init_mu_div=10, init_L_div=10,
                 n_workers=4, checkpoint_every=100,
                 out_dir='output'):
        self.g = g
        self.mu_min, self.mu_max = mu_range
        self.L_min, self.L_max = L_range
        self.max_depth = max_depth
        self.eps_M = eps_M
        self.eps_b = eps_b
        self.init_mu_div = init_mu_div
        self.init_L_div = init_L_div
        self.n_workers = n_workers
        self.checkpoint_every = checkpoint_every
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.cache = {}
        self.points_mu = []
        self.points_L = []
        self.points_M = []
        self.points_b = []

    def _get_or_compute(self, mu, L):
        """Берёт из кэша или вычисляет (в контексте уже посчитанных точек)."""
        key = (round(float(mu), 10), round(float(L), 10))
        return self.cache.get(key, (None, None))

    def _collect_points_to_compute(self, mu1, mu2, L1, L2, depth):
        """
        Рекурсивно собирает ВСЕ точки, которые нужно посчитать.
        Возвращает список [(mu, L), ...] + список финальных центров.
        """
        # Сначала нужны углы + центр ячейки для проверки variation
        coords = [
            (mu1, L1), (mu1, L2), (mu2, L1), (mu2, L2),
            ((mu1 + mu2) / 2, (L1 + L2) / 2)
        ]

        # Собираем недостающие точки для проверки variation
        needed = []
        for mu, L in coords:
            key = (round(float(mu), 10), round(float(L), 10))
            if key not in self.cache:
                needed.append((mu, L))

        return needed, coords, (mu1, mu2, L1, L2, depth)

    def _variation(self, coords):
        """Вычисляет variation M и b по 5 точкам ячейки."""
        M_vals = []
        b_vals = []
        for mu, L in coords:
            M, b = self._get_or_compute(mu, L)
            if M is None:
                return None, None  # Ещё не посчитано
            M_vals.append(M)
            b_vals.append(b)
        return max(M_vals) - min(M_vals), max(b_vals) - min(b_vals)

    def compute(self):
        """
        Главный метод: поэтапный parallel compute + sequential refine.
        """
        t_start = time.time()

        # --- Этап 0: базовая сетка ---
        print(f"\n[ЭТАП 0] Базовая сетка {self.init_mu_div}×{self.init_L_div}")
        base_points = []
        for i in range(self.init_mu_div + 1):
            for j in range(self.init_L_div + 1):
                mu = self.mu_min + i * (self.mu_max - self.mu_min) / self.init_mu_div
                L = self.L_min + j * (self.L_max - self.L_min) / self.init_L_div
                base_points.append((mu, L))

        # Убираем дубликаты
        base_points = list({(round(m, 10), round(l, 10)): (m, l) for m, l in base_points}.values())
        self.cache.update(parallel_compute(base_points, self.g, self.n_workers,
                                           self.checkpoint_every, self.out_dir, '30_base'))

        # --- Этапы рефайнмента ---
        for depth in range(self.max_depth):
            print(f"\n[ЭТАП {depth+1}] Адаптивное измельчение (depth={depth+1})")
            cells_to_refine = self._find_cells_to_refine(depth)
            if not cells_to_refine:
                print("  Нет ячеек для измельчения — переход к следующему этапу.")
                continue

            # Собираем центры ячеек для вычисления
            refine_points = []
            for cell in cells_to_refine:
                mu_c = (cell[0] + cell[1]) / 2
                L_c = (cell[2] + cell[3]) / 2
                refine_points.append((mu_c, L_c))

            print(f"  Ячеек для измельчения: {len(cells_to_refine)}")
            self.cache.update(parallel_compute(refine_points, self.g, self.n_workers,
                                               self.checkpoint_every, self.out_dir,
                                               f'30_refine_d{depth+1}'))

            # Рекурсивно обрабатываем под-ячейки
            self._refine_cells(cells_to_refine, depth + 1)

        # --- Сбор финальных точек для триангуляции ---
        self._collect_final_points()

        # --- Финальный checkpoint + plot ---
        self._save_final()
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"ИТОГО: {len(self.points_mu)} точек, {elapsed/60:.1f} мин")
        print(f"M_max={max(self.points_M):.3f}, M_min={min(self.points_M):.3f}")
        print(f"b_max={max(self.points_b):.3f}, b_min={min(self.points_b):.3f}")
        print(f"{'='*60}\n")

    def _find_cells_to_refine(self, depth):
        """Находит ячейки, требующие измельчения на данном depth."""
        cells = []
        n_mu = self.init_mu_div * (2 ** depth)
        n_L = self.init_L_div * (2 ** depth)

        for i in range(n_mu):
            for j in range(n_L):
                mu1 = self.mu_min + i * (self.mu_max - self.mu_min) / n_mu
                mu2 = self.mu_min + (i + 1) * (self.mu_max - self.mu_min) / n_mu
                L1 = self.L_min + j * (self.L_max - self.L_min) / n_L
                L2 = self.L_min + (j + 1) * (self.L_max - self.L_min) / n_L

                coords = [
                    (mu1, L1), (mu1, L2), (mu2, L1), (mu2, L2),
                    ((mu1 + mu2) / 2, (L1 + L2) / 2)
                ]
                var_M, var_b = self._variation(coords)
                if var_M is None:
                    continue
                if var_M >= self.eps_M or var_b >= self.eps_b:
                    cells.append((mu1, mu2, L1, L2))
        return cells

    def _refine_cells(self, cells, depth):
        """Рекурсивно измельчает ячейки, если не достигнут max_depth."""
        if depth >= self.max_depth:
            # Записываем центры финальных ячеек
            for mu1, mu2, L1, L2 in cells:
                mu_c = (mu1 + mu2) / 2
                L_c = (L1 + L2) / 2
                self.points_mu.append(mu_c)
                self.points_L.append(L_c)
                M_c, b_c = self._get_or_compute(mu_c, L_c)
                self.points_M.append(M_c)
                self.points_b.append(b_c)
            return

        # Измельчаем каждую ячейку
        sub_cells = []
        for mu1, mu2, L1, L2 in cells:
            mu_mid = (mu1 + mu2) / 2
            L_mid = (L1 + L2) / 2

            # 4 под-ячейки
            sub_cells.extend([
                (mu1, mu_mid, L1, L_mid),
                (mu1, mu_mid, L_mid, L2),
                (mu_mid, mu2, L1, L_mid),
                (mu_mid, mu2, L_mid, L2),
            ])

        # Проверяем variation для каждой под-ячейки
        refined = []
        for sc in sub_cells:
            mu1, mu2, L1, L2 = sc
            coords = [
                (mu1, L1), (mu1, L2), (mu2, L1), (mu2, L2),
                ((mu1 + mu2) / 2, (L1 + L2) / 2)
            ]
            var_M, var_b = self._variation(coords)
            if var_M is None:
                continue
            if var_M >= self.eps_M or var_b >= self.eps_b:
                refined.append(sc)
            else:
                # Записываем центр как финальную точку
                mu_c = (mu1 + mu2) / 2
                L_c = (L1 + L2) / 2
                self.points_mu.append(mu_c)
                self.points_L.append(L_c)
                M_c, b_c = self._get_or_compute(mu_c, L_c)
                self.points_M.append(M_c)
                self.points_b.append(b_c)

        if refined:
            # Считаем центры под-ячеек параллельно
            refine_points = []
            for mu1, mu2, L1, L2 in refined:
                refine_points.append(((mu1 + mu2) / 2, (L1 + L2) / 2))
            self.cache.update(parallel_compute(refine_points, self.g, self.n_workers,
                                               self.checkpoint_every, self.out_dir,
                                               f'30_refine_d{depth+1}'))
            self._refine_cells(refined, depth + 1)

    def _collect_final_points(self):
        """Собирает все финальные точки (если _refine_cells ещё не всё записал)."""
        if self.points_mu:
            return
        # Fallback: собираем все уникальные точки из кэша
        for (mu, L), (M, b) in self.cache.items():
            self.points_mu.append(mu)
            self.points_L.append(L)
            self.points_M.append(M)
            self.points_b.append(b)

    def _save_final(self):
        """Сохраняет финальные результаты."""
        npz_path = os.path.join(self.out_dir, '30_parallel_final_g_m1_0.npz')
        np.savez(npz_path,
                 mu=np.array(self.points_mu),
                 L=np.array(self.points_L),
                 M=np.array(self.points_M),
                 b=np.array(self.points_b))
        print(f"  [FINAL] Saved {len(self.points_mu)} points -> {npz_path}")

        png_path = os.path.join(self.out_dir, '30_parallel_final_g_m1_0.png')
        self._plot(png_path)
        print(f"  [FINAL] Saved plot -> {png_path}")

    def _plot(self, out_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tri = Triangulation(np.array(self.points_L), np.array(self.points_mu))

        ax = axes[0]
        tpc = ax.tripcolor(tri, np.array(self.points_M), shading='gouraud', cmap='viridis')
        ax.set_xlabel(r'$L$', fontsize=12)
        ax.set_ylabel(r'$\mu$', fontsize=12)
        ax.set_title(f'$M_{{min}}(\mu, L)$, g={self.g} (Parallel DE, {len(self.points_mu)} pts)')
        fig.colorbar(tpc, ax=ax, label=r'$M_{min}$')

        ax = axes[1]
        tpc = ax.tripcolor(tri, np.array(self.points_b), shading='gouraud', cmap='plasma')
        ax.set_xlabel(r'$L$', fontsize=12)
        ax.set_ylabel(r'$\mu$', fontsize=12)
        ax.set_title(f'$b_{{min}}(\mu, L)$, g={self.g} (Parallel DE, {len(self.points_mu)} pts)')
        fig.colorbar(tpc, ax=ax, label=r'$b_{min}$')

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


# ==============================================================================
# Запуск
# ==============================================================================
if __name__ == '__main__':
    # Параметры (по умолчанию 10x10 для теста; для 100x100 изменить init_mu_div/L_div)
    adp = ParallelAdaptivePhaseDiagram(
        g=-1.0,
        mu_range=(2, 7),
        L_range=(0.1, 4),
        max_depth=8,
        eps_M=0.01,
        eps_b=0.01,
        init_mu_div=50,      # <-- ПОМЕНЯЙТЕ НА 100 для полной сетки
        init_L_div=50,       # <-- ПОМЕНЯЙТЕ НА 100 для полной сетки
        n_workers=12,         # <-- Количество CPU ядер
        checkpoint_every=100,
        out_dir='output'
    )
    adp.compute()

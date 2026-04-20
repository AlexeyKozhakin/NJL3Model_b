"""
Быстрая диагностика поведения Omega(b) в плоскости (mu, L).
Фиксируем M=0, считаем Omega в трех точках по b: 0, 5, 10.
Классификация:
  0 = Omega(b) ≈ const  (все равны в пределах 1e-6)
  1 = min в b=0
  2 = min в b!=0 (хотя бы одна точка меньше Omega(0))
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

# ==============================================================================
# Параметры
# ==============================================================================
N_mu = 80
N_L = 80
mu_vals = np.linspace(0.0, 7.0, N_mu)
L_vals = np.linspace(0.1, 10.0, N_L)
g = -1.0
N_h = 80
b_test = [0.0, 5.0, 10.0]
tol = 1e-6

M = 3.0

print(f"Быстрая диагностика: {N_mu}x{N_L} точек, M={M}, b={b_test}")

# ==============================================================================
# Предвычисление Omega_L и dU для M=0
# ==============================================================================
Omega_L_00 = fun_Omega_L(np.array([L_vals[0]]), np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0, 0]

# Для каждого L и b: Omega_L(L, b, M=0) = Omega_L(L,b,0) - Omega_L(L,b,0) + Omega_L_00 = Omega_L_00
# То есть Omega_L_phys при M=0 — это просто константа Omega_L_00 для любого b!
# Аналогично dU_phys(b, 0) = dU(b,0) - dU(b,0) + dU(0,0) = dU(0,0)
# И классический член M^2/(2g) = 0 при M=0.

dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h, N_h)[0, 0]

# ==============================================================================
# Основной цикл
# ==============================================================================
class_map = np.empty((N_mu, N_L), dtype=np.int8)
# Для отладки сохраним значения
Omega_b0 = np.empty((N_mu, N_L), dtype=np.float64)
Omega_b5 = np.empty((N_mu, N_L), dtype=np.float64)
Omega_b10 = np.empty((N_mu, N_L), dtype=np.float64)

for i, mu in enumerate(tqdm(mu_vals, desc="mu")):
    Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L_vals[0], 0.0, 0.0, 0, N_h)
    for j, L in enumerate(L_vals):
        vals = np.empty(3, dtype=np.float64)
        for k, b in enumerate(b_test):
            # Omega_ren(M=0, b, mu, L, g)
            # Только Omega_mu_L зависит от b, остальное константа при M=0
            o_mu_b0 = fun_Omega_L_mu_int(mu, L, b, 0.0, 0, N_h)
            o_mu_M = fun_Omega_L_mu_int(mu, L, b, M, 0, N_h)
            Omega_mu_L_phys = o_mu_M - o_mu_b0 + Omega_mu_L_00
            vals[k] = Omega_mu_L_phys + dU_b0 + Omega_L_00  # M=0 => vac=0, dU=dU_b0, Omega_L=Omega_L_00

        Omega_b0[i, j] = vals[0]
        Omega_b5[i, j] = vals[1]
        Omega_b10[i, j] = vals[2]

        # Классификация
        if np.max(np.abs(vals - vals[0])) < tol:
            class_map[i, j] = 0  # const
        elif vals[0] <= vals[1] and vals[0] <= vals[2]:
            class_map[i, j] = 1  # min в b=0
        else:
            class_map[i, j] = 2  # min в b!=0

# ==============================================================================
# Визуализация
# ==============================================================================
out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Карта классов
cmap_classes = plt.cm.colors.ListedColormap(['#1f77b4', '#2ca02c', '#d62728'])
# Синий = const, Зеленый = b=0, Красный = b!=0
im0 = axes[0].imshow(class_map.T, origin='lower', aspect='auto',
                     extent=[L_vals[0], L_vals[-1], mu_vals[0], mu_vals[-1]],
                     cmap=cmap_classes, vmin=-0.5, vmax=2.5)
axes[0].set_xlabel('L')
axes[0].set_ylabel('mu')
axes[0].set_title('Классификация Omega(b) при M=0\n(синий=const, зеленый=min@b=0, красный=min@b≠0)')

# Omega(b=0)
im1 = axes[1].imshow(Omega_b0.T, origin='lower', aspect='auto',
                     extent=[L_vals[0], L_vals[-1], mu_vals[0], mu_vals[-1]],
                     cmap='viridis')
axes[1].set_xlabel('L')
axes[1].set_ylabel('mu')
axes[1].set_title('Omega(b=0, M=0)')
fig.colorbar(im1, ax=axes[1])

# Разница Omega(b=5) - Omega(b=0)
im2 = axes[2].imshow((Omega_b5 - Omega_b0).T, origin='lower', aspect='auto',
                     extent=[L_vals[0], L_vals[-1], mu_vals[0], mu_vals[-1]],
                     cmap='RdBu_r')
axes[2].set_xlabel('L')
axes[2].set_ylabel('mu')
axes[2].set_title('Omega(b=5) - Omega(b=0)')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
png_path = os.path.join(out_dir, '28_quick_b_classification.png')
plt.savefig(png_path, dpi=200)
print(f"\nСохранено: {png_path}")

# Статистика
n_const = (class_map == 0).sum()
_n_b0 = (class_map == 1).sum()
n_bneq0 = (class_map == 2).sum()
print(f"\nСтатистика:")
print(f"  const:     {n_const} / {N_mu*N_L} ({100*n_const/(N_mu*N_L):.1f}%)")
print(f"  min@b=0:   {_n_b0} / {N_mu*N_L} ({100*_n_b0/(N_mu*N_L):.1f}%)")
print(f"  min@b≠0:   {n_bneq0} / {N_mu*N_L} ({100*n_bneq0/(N_mu*N_L):.1f}%)")

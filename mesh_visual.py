import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functions.minimum_Omega_b_M import find_minimum_2d
from functions.Omega_ren import Omega_ren


# ---- Ваша функция F(mu,L) ----
def F(mu, L):
    g=-1.0
    bm, Mm = find_minimum_2d(
    Omega_ren,
    bmin=0.0, bmax=10.0,
    Mmin=0.0, Mmax=10.0,
    mu=mu, L=L, g=g,
    eps=1e-3
)
    return bm  # пример
# ---- Кэш для ускорения ----
cache = {}
def evalF(mu, L):
    key = (mu, L)
    if key not in cache:
        cache[key] = F(mu, L)
    return cache[key]
# ---- Основной блок адаптивного разбиения ----
def refine_cell(mu0, mu1, L0, L1, G_threshold, min_size):
    # Вычисление углов
    f00 = evalF(mu0, L0)
    f10 = evalF(mu1, L0)
    f01 = evalF(mu0, L1)
    f11 = evalF(mu1, L1)
    # Градиенты
    dmu = mu1 - mu0
    dL = L1 - L0
    G_mu = max(abs(f10 - f00), abs(f11 - f01)) / dmu
    G_L = max(abs(f01 - f00), abs(f11 - f10)) / dL
    G = max(G_mu, G_L)
    # Решение: делить или нет
    if G < G_threshold or (dmu < min_size and dL < min_size):
        return [(mu0, mu1, L0, L1)]   # финальная ячейка
    # Иначе делим
    mu_mid = 0.5 * (mu0 + mu1)
    L_mid = 0.5 * (L0 + L1)
    subcells = [
        (mu0, mu_mid, L0, L_mid),
        (mu_mid, mu1, L0, L_mid),
        (mu0, mu_mid, L_mid, L1),
        (mu_mid, mu1, L_mid, L1)
    ]
    results = []
    for c in subcells:
        results.extend(refine_cell(*c, G_threshold, min_size))
    return results
# ---- Главная функция ----
def adaptive_mesh(mu_min, mu_max, L_min, L_max,
                  G_threshold=1.0, min_size=0.01):
    top_cell = (mu_min, mu_max, L_min, L_max)
    cells = refine_cell(*top_cell, G_threshold, min_size)
    # собираем уникальные точки для визуализации
    points = []
    for (mu0, mu1, L0, L1) in cells:
        points.extend([
            (mu0, L0), (mu1, L0),
            (mu0, L1), (mu1, L1)
        ])
    points = list(set(points))
    # создаём массив данных
    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    Z = np.array([evalF(p[0], p[1]) for p in points])
    return X, Y, Z, cells

import matplotlib.pyplot as plt

mu_min = 0.1
mu_max = 7.0
L_min = 0.1
L_max = 5.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Ваши данные
Y, X, Z, cells = adaptive_mesh(mu_min, mu_max, L_min, L_max,
                               G_threshold=0.1, min_size=0.1)

# Создаём регулярную сетку для интерполяции
num = 300   # плотность сетки, можно увеличить
Xi = np.linspace(X.min(), X.max(), num)
Yi = np.linspace(Y.min(), Y.max(), num)
Xi, Yi = np.meshgrid(Xi, Yi)

# Интерполяция
Zi = griddata(points=(X, Y), values=Z, xi=(Xi, Yi), method='linear')

# Можно заменить на 'cubic' для более гладкого результата:
# Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

# Визуализация
plt.figure(figsize=(7, 5))
plt.pcolormesh(Xi, Yi, Zi, shading='auto')
plt.colorbar(label="F")
plt.xlabel("L")
plt.ylabel("mu")
plt.title("Интерполяция по всей области")
plt.show()

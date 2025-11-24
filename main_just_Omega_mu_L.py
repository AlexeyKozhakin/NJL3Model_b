
import numpy as np
from multiprocessing import Pool, cpu_count
import time
# import cProfile
# import pstats
from functions_opt.Omega_mu_L import fun_Omega_L_mu_int

# Ваши функции: fun_Omega_L_mu_int, n_max_plus, и другие остаются без изменений


# --- Настройка точности ---
USE_FLOAT32 = False  # переключатель точности: True — float32, False — float64

# Задание значений параметров
DTYPE = np.float32 if USE_FLOAT32 else np.float64
MU_SPLIT = 100      # количество разбиений для mu
L_SPLIT = 100       # количество разбиений для L
M_SPLIT = 100       # количество разбиений для M
B_SPLIT = 11       # количество разбиений для b

mu_vals = np.linspace(2.5, 4, MU_SPLIT, dtype=DTYPE)
L_vals = np.linspace(0.1, 3.0, L_SPLIT, dtype=DTYPE)
M_vals = np.linspace(0, 5, M_SPLIT, dtype=DTYPE)
b_vals = np.linspace(-1, 1, B_SPLIT, dtype=DTYPE)
phi = DTYPE(0)
g = DTYPE(-1)





# Инициализация массивов

Omega_mu_L_phys = np.zeros((len(mu_vals), len(L_vals), len(b_vals), len(M_vals)), dtype=DTYPE)


# Функция для параллельных вычислений

def calculate(params):
    mu_ind, L_ind, b_ind, M_ind = params
    mu = mu_vals[mu_ind]
    L = L_vals[L_ind]
    b = b_vals[b_ind]
    M = M_vals[M_ind]

    # start_prof = time.time()
    Omega_mu_L = (fun_Omega_L_mu_int(mu, L, b, M, phi=0)
                  - fun_Omega_L_mu_int(mu, L, b, 0, phi=0)
                  + fun_Omega_L_mu_int(mu, L, 0, 0, phi=0))


    return mu_ind, L_ind, b_ind, M_ind, Omega_mu_L


if __name__ == "__main__":
    # Собираем параметры для каждой комбинации индексов
    params = [(mu_ind, L_ind, b_ind, M_ind)
              for mu_ind in range(len(mu_vals))
              for L_ind in range(len(L_vals))
              for b_ind in range(len(b_vals))
              for M_ind in range(len(M_vals))]

    # Параллельные вычисления
    start = time.time()

    with Pool(processes=10) as pool:
        results = pool.map(calculate, params)

    # profiler.disable()
    end = time.time()

    print(f"Total computation time: {end - start:.4f} seconds")


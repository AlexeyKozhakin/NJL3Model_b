import numpy as np
from multiprocessing import Pool
import time
from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

# Ваши функции: fun_Omega_L_mu_int, n_max_plus, и другие остаются без изменений

# Задание значений параметров
mu_vals = np.linspace(2.5, 4, 10)
L_vals = np.linspace(0.1, 3.0, 10)
M_vals = np.linspace(0, 5, 10)
b_vals = np.linspace(-1, 1, 11)
phi = 0
g = -1

# Предварительные расчеты
Omega_L_phys = (fun_Omega_L(L_vals, b_vals, M_vals, N_h_p1=100, N_h_p2=100) -
                fun_Omega_L(L_vals, b_vals, 0, N_h_p1=100, N_h_p2=100) +
                fun_Omega_L(L_vals, 0, 0, N_h_p1=100, N_h_p2=100))
dU_phys = fun_dU_phys(b_vals, M_vals, N_h_p=100, N_h_phi=100)

# Инициализация массивов
Omega_ren_phys = np.zeros((len(mu_vals), len(L_vals), len(b_vals), len(M_vals)))
Omega_mu_L_phys = np.zeros((len(mu_vals), len(L_vals), len(b_vals), len(M_vals)))


# Функция для параллельных вычислений
def calculate(params):
    mu_ind, L_ind, b_ind, M_ind = params
    mu = mu_vals[mu_ind]
    L = L_vals[L_ind]
    b = b_vals[b_ind]
    M = M_vals[M_ind]

    Omega_mu_L = (fun_Omega_L_mu_int(mu, L, b, M, phi=0)
                  - fun_Omega_L_mu_int(mu, L, b, 0, phi=0)
                  + fun_Omega_L_mu_int(mu, L, 0, 0, phi=0))
    Omega_ren = (M**2 / (2 * g) +
                 dU_phys[b_ind, M_ind] +
                 Omega_L_phys[L_ind, b_ind, M_ind] +
                 Omega_mu_L)

    return mu_ind, L_ind, b_ind, M_ind, Omega_mu_L, Omega_ren

if __name__ == "__main__":
    # Собираем параметры для каждой комбинации индексов
    params = [(mu_ind, L_ind, b_ind, M_ind)
              for mu_ind in range(len(mu_vals))
              for L_ind in range(len(L_vals))
              for b_ind in range(len(b_vals))
              for M_ind in range(len(M_vals))]

    # Параллельные вычисления
    start = time.time()

    with Pool(processes=1) as pool:
        results = pool.map(calculate, params)

    # Заполнение массивов результатами
    for mu_ind, L_ind, b_ind, M_ind, Omega_mu_L, Omega_ren in results:
        Omega_ren_phys[mu_ind, L_ind, b_ind, M_ind] = Omega_ren

    end = time.time()

    print("Total time:", end - start)

    # Сохранение тензора в файл .npy
    np.save('Omega_ren_phys.npy', Omega_ren_phys)
    print("Тензор Omega_ren_phys сохранен в файл Omega_ren_phys.npy")
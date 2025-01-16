import numpy as np
from functions.utils import midpoints

def E_1(p1, M):
    return np.sqrt(M**2 + p1**2)

def B_pm(p1, p3, b, M, sign):
    E1 = E_1(p1, M)
    return np.sqrt(p3**2 + (E1 + sign * b)**2)

# Define the integrand

def Omega_L_int(u1, u3, L, b, M, phi=0):
    p1=u1/(1-u1)
    p3=u3/(1-u3)
    B_plus = B_pm(p1, p3, b, M, 1)
    B_minus = B_pm(p1, p3, b, M, -1)

    term1 = 1 - 2 * np.cos(2 * np.pi * phi) * np.exp(-L * B_plus) + np.exp(-2 * L * B_plus)
    term2 = 1 - 2 * np.cos(2 * np.pi * phi) * np.exp(-L * B_minus) + np.exp(-2 * L * B_minus)

    return -4*np.log(term1 * term2) / (2 * np.pi)**2/(1-u1)**2/(1-u3)**2/L


def fun_Omega_L(L_vals, b_vals, M_vals, N_h_p1=200, N_h_p2=200):

    p1_vals = midpoints(0, 1, N_h_p1)  # Сетка для p
    p2_vals = midpoints(0, 1, N_h_p2)  # Сетка для p

    # Создаём 4D-сетку
    p1_grid, p2_grid, L_grid, b_grid, M_grid = np.meshgrid(p1_vals, p2_vals, L_vals, b_vals, M_vals, indexing='ij')

    # Вычисляем значения функции F на сетке
    F_values = Omega_L_int(p1_grid, p2_grid, L_grid, b_grid, M_grid)

    # Вычисляем численное интегрирование по p и phi (объёмная сумма)
    dp1 = (p1_vals[1] - p1_vals[0])  # Шаг сетки по p
    dp2 = (p2_vals[1] - p2_vals[0])  # Шаг сетки по phi
    return np.sum(F_values, axis=(0, 1)) * dp1 * dp2

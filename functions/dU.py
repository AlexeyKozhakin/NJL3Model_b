import numpy as np
from functions.utils import midpoints


def dU_int(u, phi, b, M):
    p=u/(1-u)
    term1 = M**2 / p
    term2 = np.sqrt(b**2 + p**2 + 2 * b * p * np.cos(phi))
    term3 = np.sqrt(b**2 + p**2 - 2 * b * p * np.cos(phi))
    term4 = np.sqrt(M**2 + b**2 + p**2 + 2 * b * np.sqrt(M**2 + p**2 * np.cos(phi)**2))
    term5 = np.sqrt(M**2 + b**2 + p**2 - 2 * b * np.sqrt(M**2 + p**2 * np.cos(phi)**2))
    return -(-term1 - term2 - term3 + term4 + term5)*p / (np.pi**2)/(1-u)**2

def fun_dU_phys(b_vals, M_vals, N_h_p=100, N_h_phi=100):

    p_vals = midpoints(0, 1, N_h_p)  # Сетка для p
    phi_vals = midpoints(0, np.pi/2, N_h_phi)  # Сетка для phi


    # Создаём 4D-сетку
    p_grid, phi_grid, b_grid, M_grid = np.meshgrid(p_vals, phi_vals, b_vals, M_vals, indexing='ij')

    # Вычисляем значения функции F на сетке
    F_values = dU_int(p_grid, phi_grid, b_grid, M_grid)

    # Вычисляем численное интегрирование по p и phi (объёмная сумма)
    dp = (p_vals[1] - p_vals[0])  # Шаг сетки по p
    dphi = (phi_vals[1] - phi_vals[0])  # Шаг сетки по phi
    return np.sum(F_values, axis=(0, 1)) * dp * dphi

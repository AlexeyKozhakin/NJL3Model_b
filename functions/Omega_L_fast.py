"""
Numba-ускоренная версия Omega_L.
Скалярные циклы вместо NumPy meshgrid.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def _E_1(p1, M):
    return np.sqrt(M**2 + p1**2)


@jit(nopython=True)
def _B_pm(p1, p3, b, M, sign):
    E1 = _E_1(p1, M)
    return np.sqrt(p3**2 + (E1 + sign * b)**2)


@jit(nopython=True)
def _Omega_L_int(u1, u3, L, b, M, phi):
    p1 = u1 / (1.0 - u1)
    p3 = u3 / (1.0 - u3)
    B_plus = _B_pm(p1, p3, b, M, 1)
    B_minus = _B_pm(p1, p3, b, M, -1)

    term1 = 1.0 - 2.0 * np.cos(2.0 * np.pi * phi) * np.exp(-L * B_plus) + np.exp(-2.0 * L * B_plus)
    term2 = 1.0 - 2.0 * np.cos(2.0 * np.pi * phi) * np.exp(-L * B_minus) + np.exp(-2.0 * L * B_minus)

    return -4.0 * np.log(term1 * term2) / (4.0 * np.pi * np.pi) / ((1.0 - u1)**2) / ((1.0 - u3)**2) / L


@jit(nopython=True)
def Omega_L_scalar(L, b, M, phi=0.0, N_h_p1=100, N_h_p2=100):
    """
    Скалярное вычисление Omega_L методом средних точек.
    """
    dp1 = 1.0 / N_h_p1
    dp2 = 1.0 / N_h_p2
    start1 = dp1 * 0.5
    start2 = dp2 * 0.5

    total = 0.0
    for i in range(N_h_p1):
        u1 = start1 + i * dp1
        for j in range(N_h_p2):
            u3 = start2 + j * dp2
            total += _Omega_L_int(u1, u3, L, b, M, phi)

    return total * dp1 * dp2

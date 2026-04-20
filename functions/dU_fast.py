"""
Numba-ускоренная версия контрчлена Уайтинга dU.
Скалярные циклы вместо NumPy meshgrid.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def _dU_int(u, phi, b, M):
    p = u / (1.0 - u)
    term1 = M**2 / p
    c = np.cos(phi)
    term2 = np.sqrt(b**2 + p**2 + 2.0 * b * p * c)
    term3 = np.sqrt(b**2 + p**2 - 2.0 * b * p * c)
    inner = np.sqrt(M**2 + p**2 * c**2)
    term4 = np.sqrt(M**2 + b**2 + p**2 + 2.0 * b * inner)
    term5 = np.sqrt(M**2 + b**2 + p**2 - 2.0 * b * inner)
    return -(-term1 - term2 - term3 + term4 + term5) * p / (np.pi**2) / ((1.0 - u)**2)


@jit(nopython=True)
def dU_scalar(b, M, N_h_p=100, N_h_phi=100):
    """
    Скалярное вычисление dU методом средних точек.
    """
    dp = 1.0 / N_h_p
    dphi = (np.pi / 2.0) / N_h_phi
    start_p = dp * 0.5
    start_phi = dphi * 0.5

    total = 0.0
    for i in range(N_h_p):
        u = start_p + i * dp
        for j in range(N_h_phi):
            phi = start_phi + j * dphi
            total += _dU_int(u, phi, b, M)

    return total * dp * dphi

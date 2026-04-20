"""
Numba-ускоренная версия полного перенормированного потенциала Omega_ren.
Использует скалярные Omega_L_fast, dU_fast и напрямую вызывает numba-ядро Omega_mu_L.
"""
import numpy as np
from numba import jit

from functions.Omega_L_fast import Omega_L_scalar
from functions.dU_fast import dU_scalar

# Импортируем внутренние numba-функции из Omega_mu_L
from functions.Omega_mu_L import _compute_Unp, _compute_Unm


@jit(nopython=True)
def fun_Omega_L_mu_int_njit(mu, L, b, M, phi=0, N_h=100):
    """
    Numba-совместимая обертка над _compute_Unp + _compute_Unm.
    """
    Unp = _compute_Unp(M, b, L, mu, phi, N_h)
    Unm = _compute_Unm(M, b, L, mu, phi, N_h)
    return -(2.0 / L) * (Unp + Unm)


@jit(nopython=True)
def Omega_ren_fast(M, b, mu, L, g,
                   N_h_p1=80, N_h_p2=80,
                   N_h_p=80, N_h_phi=80,
                   N_h_mu=80):
    """
    Полное перенормированное потенциала Omega(b, M).
    Чисто numba-реализация для максимальной скорости.
    """
    if M < 0.0:
        M = 0.0
    if b < 0.0:
        b = 0.0

    vac = M * M / (2.0 * g)

    dU_M = dU_scalar(b, M, N_h_p, N_h_phi)
    dU_M0 = dU_scalar(b, 0.0, N_h_p, N_h_phi)
    dU_b0 = dU_scalar(0.0, 0.0, N_h_p, N_h_phi)
    dU_phys = dU_M - dU_M0 + dU_b0

    oL_M = Omega_L_scalar(L, b, M, 0.0, N_h_p1, N_h_p2)
    oL_M0 = Omega_L_scalar(L, b, 0.0, 0.0, N_h_p1, N_h_p2)
    oL_b0 = Omega_L_scalar(L, 0.0, 0.0, 0.0, N_h_p1, N_h_p2)
    Omega_L_phys = oL_M - oL_M0 + oL_b0

    o_mu_M = fun_Omega_L_mu_int_njit(mu, L, b, M, 0, N_h_mu)
    o_mu_M0 = fun_Omega_L_mu_int_njit(mu, L, b, 0.0, 0, N_h_mu)
    o_mu_b0 = fun_Omega_L_mu_int_njit(mu, L, 0.0, 0.0, 0, N_h_mu)
    Omega_mu_L_phys = o_mu_M - o_mu_M0 + o_mu_b0

    return vac + dU_phys + Omega_L_phys + Omega_mu_L_phys

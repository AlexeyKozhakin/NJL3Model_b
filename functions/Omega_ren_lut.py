"""
LUT-ускоренная версия Omega_ren.
Строит dense grid Omega_L(M,b) и dU(M,b) построчно,
чтобы избежать переполнения памяти от 5D meshgrid в оригинальных функциях.
Затем использует билинейную интерполяцию во время оптимизации.
Omega_mu_L считается напрямую через numba.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int


class OmegaRenLUT:
    """
    Для фиксированных (mu, L, g) строит LUT по (M, b) и предоставляет
    быструю интерполяцию Omega_ren(M, b).
    """
    def __init__(self, mu, L, g,
                 M_max=10.0, b_max=10.0,
                 N_M=200, N_b=200,
                 N_h_p1=80, N_h_p2=80,
                 N_h_p=80, N_h_phi=80,
                 N_h_mu=80):
        self.mu = mu
        self.L = L
        self.g = g
        self.M_max = M_max
        self.b_max = b_max
        self.N_h_p1 = N_h_p1
        self.N_h_p2 = N_h_p2
        self.N_h_p = N_h_p
        self.N_h_phi = N_h_phi
        self.N_h_mu = N_h_mu

        # Сетки
        self.M_vals = np.linspace(0.0, M_max, N_M)
        self.b_vals = np.linspace(0.0, b_max, N_b)

        # === Omega_L LUT (построчно по b, чтобы избежать 5D meshgrid) ===
        self.Omega_L_00 = fun_Omega_L(
            np.array([L]), np.array([0.0]), np.array([0.0]), N_h_p1, N_h_p2
        )[0, 0, 0]

        Omega_L_0M = fun_Omega_L(
            np.array([L]), self.b_vals, np.array([0.0]), N_h_p1, N_h_p2
        )[0, :, 0]  # shape (N_b,)

        self.Omega_L_grid = np.empty((N_b, N_M), dtype=np.float64)
        for i, b in enumerate(self.b_vals):
            row = fun_Omega_L(
                np.array([L]), np.array([b]), self.M_vals, N_h_p1, N_h_p2
            )[0, 0, :]  # shape (N_M,)
            self.Omega_L_grid[i, :] = row - Omega_L_0M[i] + self.Omega_L_00

        # === dU LUT (построчно по b) ===
        dU_0M = fun_dU_phys(self.b_vals, np.array([0.0]), N_h_p, N_h_phi)[:, 0]  # shape (N_b,)
        dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h_p, N_h_phi)[0, 0]

        self.dU_grid = np.empty((N_b, N_M), dtype=np.float64)
        for i, b in enumerate(self.b_vals):
            row = fun_dU_phys(
                np.array([b]), self.M_vals, N_h_p, N_h_phi
            )[0, :]  # shape (N_M,)
            self.dU_grid[i, :] = row - dU_0M[i] + dU_b0

        # === Интерполяторы ===
        self.interp_Omega_L = RegularGridInterpolator(
            (self.b_vals, self.M_vals), self.Omega_L_grid,
            method='linear', bounds_error=False, fill_value=None
        )
        self.interp_dU = RegularGridInterpolator(
            (self.b_vals, self.M_vals), self.dU_grid,
            method='linear', bounds_error=False, fill_value=None
        )

        self.Omega_mu_L_00 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, 0, N_h_mu)

    def __call__(self, M, b):
        if M < 0.0:
            M = 0.0
        if b < 0.0:
            b = 0.0

        vac = M * M / (2.0 * self.g)
        dU_phys = float(self.interp_dU([[b, M]])[0])
        Omega_L_phys = float(self.interp_Omega_L([[b, M]])[0])

        o_mu_M = fun_Omega_L_mu_int(self.mu, self.L, b, M, 0, self.N_h_mu)
        o_mu_M0 = fun_Omega_L_mu_int(self.mu, self.L, b, 0.0, 0, self.N_h_mu)
        Omega_mu_L_phys = o_mu_M - o_mu_M0 + self.Omega_mu_L_00

        return vac + dU_phys + Omega_L_phys + Omega_mu_L_phys

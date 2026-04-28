"""
GPU-реализация тепловой поправки Omega_L(L, b, M) на PyTorch.

Формула (theory.md, раздел 5):
    Omega_L = integral dp1 dp3  f(p1, p3, L, b, M)
    f = -4*log(T+ * T-) / (4*pi^2) / L
    T± = 1 - 2*cos(2*pi*phi)*exp(-L*B±) + exp(-2*L*B±)
    B± = sqrt(p3^2 + (E1 ± b)^2),  E1 = sqrt(M^2 + p1^2)

phi=0 (наш случай) упрощает: T± = (1 - exp(-L*B±))^2.

Подстановка p = u/(1-u): оба импульса p1, p3 in [0,inf) <-> u1,u3 in [0,1).

Стратегия памяти (для каждого L):
    Цикл по u1 (N_p итераций), каждый шаг векторизует (N_B, N_M, N_p3).
    Пиковая память ≈ N_B * N_M * N_p3 * 6 * 8 байт ≈ 48 MB при 100^3.
"""

import math
import torch


def compute(
    L_grid: torch.Tensor,
    b_grid: torch.Tensor,
    M_grid: torch.Tensor,
    N_p: int = 100,
    phi: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Вычисляет Omega_L(L, b, M) для полной сетки (L, b, M).

    Параметры
    ---------
    L_grid : Tensor (N_L,)
    b_grid : Tensor (N_B,)
    M_grid : Tensor (N_M,)
    N_p    : int — число точек по каждому из u1, u3.
    phi    : float — twist-фаза (0 = антипериодические г.у.).
    device : torch.device
    dtype  : torch.dtype

    Возвращает
    ----------
    Tensor (N_L, N_B, N_M)
    """
    if device is None:
        device = b_grid.device

    N_L = len(L_grid)
    N_B = len(b_grid)
    N_M = len(M_grid)

    b = b_grid.to(device=device, dtype=dtype).view(N_B, 1, 1)   # (N_B, 1, 1)
    M = M_grid.to(device=device, dtype=dtype).view(1, N_M, 1)   # (1, N_M, 1)

    du = 1.0 / N_p

    # u3-сетка: midpoints, форма (1, 1, N_p)
    u3 = (torch.arange(N_p, device=device, dtype=dtype) + 0.5) * du
    p3 = u3 / (1.0 - u3)                      # (N_p,)
    jac3 = 1.0 / (1.0 - u3) ** 2              # (N_p,)
    p3   = p3.view(1, 1, N_p)                 # (1, 1, N_p)
    jac3 = jac3.view(1, 1, N_p)               # (1, 1, N_p)

    cos2phi = math.cos(2.0 * math.pi * phi)   # scalar
    pi2_4 = 4.0 * math.pi ** 2

    result = torch.zeros(N_L, N_B, N_M, device=device, dtype=dtype)

    for i_L in range(N_L):
        L = L_grid[i_L].item()
        acc = torch.zeros(N_B, N_M, device=device, dtype=dtype)

        for i_p1 in range(N_p):
            u1  = (i_p1 + 0.5) * du
            p1  = u1 / (1.0 - u1)
            jac1 = 1.0 / (1.0 - u1) ** 2

            # E1: (1, N_M, 1)
            E1 = torch.sqrt(M * M + p1 * p1)

            # B±: (N_B, N_M, N_p3)
            B_plus  = torch.sqrt(p3 * p3 + (E1 + b) ** 2)
            B_minus = torch.sqrt(p3 * p3 + (E1 - b) ** 2)

            # T± = 1 - 2*cos(2pi*phi)*exp(-L*B) + exp(-2*L*B)
            lB_p = L * B_plus
            lB_m = L * B_minus
            e_p  = torch.exp(-lB_p)
            e_m  = torch.exp(-lB_m)

            log_T_plus  = torch.log(1.0 - 2.0 * cos2phi * e_p + e_p * e_p)
            log_T_minus = torch.log(1.0 - 2.0 * cos2phi * e_m + e_m * e_m)

            # integrand = -4*(log_T+ + log_T-) / (4*pi^2) / L * jac1 * jac3
            integrand = (
                -4.0 * (log_T_plus + log_T_minus) / pi2_4 / L * jac1 * jac3
            )  # (N_B, N_M, N_p3)

            # sum over u3 dimension
            acc += integrand.sum(dim=2)   # (N_B, N_M)

        result[i_L] = acc * (du * du)

    return result

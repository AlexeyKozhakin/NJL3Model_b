"""
GPU-реализация вакуумного контрчлена dU(b, M) на PyTorch.

Формула (theory.md, раздел 4):
    dU(b,M) = integral_{0}^{pi/2} dphi integral_{0}^{inf} dp  f(p,phi,b,M)
    f = -(-M^2/p - sqrt(b^2+p^2+2bp*c) - sqrt(b^2+p^2-2bp*c)
           + sqrt(M^2+b^2+p^2+2b*E_c) + sqrt(M^2+b^2+p^2-2b*E_c)) * p/pi^2
    E_c = sqrt(M^2 + p^2*cos^2(phi))

Подстановка p = u/(1-u): p in [0,inf) <-> u in [0,1), dp = du/(1-u)^2.

Замечание (theory.md, раздел 9.1):
    dU(b, 0) = 0 для любого b  →  dU_phys(b, M) = dU(b, M)  (ренормализация тривиальна).

Стратегия памяти:
    Цикл по u (N_p итераций), каждый шаг векторизует (N_B, N_M, N_phi).
    Пиковая память ≈ N_B * N_M * N_phi * 8 * 8 байт ≈ 64 MB при 100^3.
"""

import math
import torch


def compute(
    b_grid: torch.Tensor,
    M_grid: torch.Tensor,
    N_p: int = 100,
    N_phi: int = 100,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Вычисляет dU(b, M) для всей сетки (b, M).

    Параметры
    ---------
    b_grid : Tensor (N_B,)
        Значения внешнего поля.
    M_grid : Tensor (N_M,)
        Значения динамической массы.
    N_p : int
        Число точек по u (= по p).
    N_phi : int
        Число точек по phi.
    device : torch.device
        Устройство вычислений.
    dtype : torch.dtype
        Тип данных.

    Возвращает
    ----------
    Tensor (N_B, N_M)
        dU[i_b, i_M] = dU(b_grid[i_b], M_grid[i_M]).
    """
    if device is None:
        device = b_grid.device

    b = b_grid.to(device=device, dtype=dtype).view(-1, 1, 1)   # (N_B, 1, 1)
    M = M_grid.to(device=device, dtype=dtype).view(1, -1, 1)   # (1, N_M, 1)

    du    = 1.0 / N_p
    dphi  = (math.pi / 2.0) / N_phi

    # phi-сетка: midpoints в [0, pi/2] — форма (1, 1, N_phi)
    phi = (torch.arange(N_phi, device=device, dtype=dtype) + 0.5) * dphi
    phi = phi.view(1, 1, N_phi)
    cos_phi = torch.cos(phi)    # (1, 1, N_phi)

    result = torch.zeros(b_grid.shape[0], M_grid.shape[0], device=device, dtype=dtype)

    pi2 = math.pi ** 2

    for i in range(N_p):
        u = (i + 0.5) * du
        p = u / (1.0 - u)
        jac = 1.0 / (1.0 - u) ** 2   # dp/du

        p2   = p * p
        c    = cos_phi                 # (1, 1, N_phi)
        b2   = b * b                   # (N_B, 1, 1)
        M2   = M * M                   # (1, N_M, 1)
        bc   = b * c                   # (N_B, 1, N_phi) after broadcast

        # term1 = M^2/p  —  скаляр по phi
        term1 = M2 / p                 # (1, N_M, 1)

        # term2, term3 = sqrt(b^2 + p^2 ± 2bp*cos(phi))
        base  = b2 + p2                # (N_B, 1, 1)
        cross = 2.0 * p * bc           # (N_B, 1, N_phi)
        term2 = torch.sqrt(base + cross)
        term3 = torch.sqrt(base - cross)

        # E_c = sqrt(M^2 + p^2 * cos^2(phi))
        E_c   = torch.sqrt(M2 + p2 * c * c)    # (1, N_M, N_phi)

        # term4, term5 = sqrt(M^2 + b^2 + p^2 ± 2b*E_c)
        base2 = M2 + b2 + p2           # (N_B, N_M, 1) after broadcast
        bEc   = 2.0 * b * E_c          # (N_B, N_M, N_phi)
        term4 = torch.sqrt(base2 + bEc)
        term5 = torch.sqrt(base2 - bEc)

        # integrand = -(-t1 - t2 - t3 + t4 + t5) * p / pi^2 * jac
        integrand = (term1 + term2 + term3 - term4 - term5) * (p * jac / pi2)

        # суммируем по phi (dim=2) и добавляем в результат
        result += integrand.sum(dim=2)

    return result * (du * dphi)

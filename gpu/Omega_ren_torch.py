"""
Сборка ренормализованного потенциала Omega_ren(mu, L, b, M) на PyTorch.

Схема ренормализации (theory.md, раздел 9):
    dU_phys       = dU(b,M)                              # dU(b,0)=0 → тривиально
    OmegaL_phys   = OmegaL(L,b,M) - OmegaL(L,b,0) + OmegaL(L,0,0)
    OmegamuL_phys = OmegamuL(b,M) - OmegamuL_b0(b)  + OmegamuL_00
    Omega_ren      = M²/(2g) + dU_phys + OmegaL_phys + OmegamuL_phys

Форматы тензоров:
    dU_phys        : (N_B, N_M)
    OmegaL_phys    : (N_L, N_B, N_M)
    OmegamuL_phys  : (N_MU, N_L, N_B, N_M)
    Omega_ren      : (N_MU, N_L, N_B, N_M)
"""

import math
import torch

from gpu.functions.Omega_muL_00_torch import compute as _omega_00_scalar
from gpu.functions.Omega_muL_b0_torch import compute as _omega_b0_scalar
from gpu.functions.dU_torch import compute as _dU
from gpu.functions.Omega_L_torch import compute as _omega_L
from gpu.functions.Omega_muL_torch import compute as _omega_muL


# ==============================================================================
# Вспомогательные функции: пакетное вычисление аналитических слагаемых
# ==============================================================================

def _omega_muL_00_batch(
    mu_grid: torch.Tensor,
    L_grid: torch.Tensor,
    n_max_global: int = 20,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    OmegamuL(mu, L, 0, 0) для всех пар (mu, L). Форма: (N_MU, N_L).
    Аналитическая формула, без численного интегрирования.
    """
    if device is None:
        device = mu_grid.device
    N_MU = len(mu_grid)
    N_L  = len(L_grid)
    two_pi = 2.0 * math.pi

    mu = mu_grid.to(device=device, dtype=dtype).view(N_MU, 1, 1)  # (N_MU, 1, 1)
    L  = L_grid.to(device=device,  dtype=dtype).view(1, N_L, 1)   # (1, N_L, 1)

    # n=0: вклад mu^2
    acc = (mu * mu).expand(N_MU, N_L, 1).squeeze(-1)  # (N_MU, N_L)

    if n_max_global > 0:
        n   = torch.arange(1, n_max_global + 1, device=device, dtype=dtype).view(1, 1, -1)
        kn  = two_pi * n / L                           # (1, N_L, N)
        kn2 = kn * kn
        Sn2 = mu * mu - kn2                            # (N_MU, N_L, N)

        # Маска: n <= n_max = floor(L*mu/(2*pi)) И Sn2 > 0
        n_max_real = torch.floor(L * mu / two_pi)      # (N_MU, N_L, 1)
        mask = ((n <= n_max_real) & (Sn2 > 0)).to(dtype)

        Sn      = torch.sqrt(Sn2.clamp(min=0.0))
        log_arg = (mu + Sn) / kn.clamp(min=1e-300)
        contrib = 2.0 * (mu * Sn - kn2 * torch.log(log_arg.clamp(min=1e-300)))

        acc = acc + (contrib * mask).sum(dim=2)        # (N_MU, N_L)

    return -acc / (math.pi * L.view(1, N_L))           # (N_MU, N_L)


def _omega_muL_b0_batch(
    mu_grid: torch.Tensor,
    L_grid: torch.Tensor,
    b_grid: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    OmegamuL(mu, L, b, 0) для всех троек (mu, L, b). Форма: (N_MU, N_L, N_B).
    Аналитическая сумма по KK-модам (n_max <= 4) — вызывает скалярную функцию.
    """
    if device is None:
        device = mu_grid.device
    N_MU = len(mu_grid)
    N_L  = len(L_grid)
    N_B  = len(b_grid)

    result = torch.zeros(N_MU, N_L, N_B, device=device, dtype=dtype)

    for i in range(N_MU):
        mu = mu_grid[i].item()
        for j in range(N_L):
            L = L_grid[j].item()
            for k in range(N_B):
                b = b_grid[k].item()
                result[i, j, k] = _omega_b0_scalar(mu, L, b)

    return result


# ==============================================================================
# Сборка Omega_ren из готовых тензоров
# ==============================================================================

def assemble(
    dU_bM: torch.Tensor,           # (N_B, N_M)
    OmegaL_bM: torch.Tensor,       # (N_L, N_B, N_M)
    OmegaL_b0: torch.Tensor,       # (N_L, N_B)
    OmegaL_00: torch.Tensor,       # (N_L,)
    OmegamuL_bM: torch.Tensor,     # (N_MU, N_L, N_B, N_M)
    OmegamuL_b0: torch.Tensor,     # (N_MU, N_L, N_B)
    OmegamuL_00: torch.Tensor,     # (N_MU, N_L)
    M_grid: torch.Tensor,          # (N_M,)
    g: float,
) -> torch.Tensor:
    """
    Собирает Omega_ren(mu, L, b, M) из предвычисленных слагаемых.

    Возвращает тензор (N_MU, N_L, N_B, N_M).
    """
    N_MU = OmegamuL_bM.shape[0]
    N_L  = OmegamuL_bM.shape[1]
    N_B  = OmegamuL_bM.shape[2]
    N_M  = OmegamuL_bM.shape[3]

    # dU(b,0)=0 для всех b → dU_phys = dU(b,M)
    dU_phys = dU_bM.view(1, 1, N_B, N_M)

    # OmegaL_phys = OmegaL(L,b,M) - OmegaL(L,b,0) + OmegaL(L,0,0)
    OmegaL_phys = (
        OmegaL_bM                           # (N_L, N_B, N_M)
        - OmegaL_b0.unsqueeze(-1)           # (N_L, N_B, 1)
        + OmegaL_00.view(N_L, 1, 1)         # (N_L, 1, 1)
    ).view(1, N_L, N_B, N_M)

    # OmegamuL_phys = OmegamuL(mu,L,b,M) - OmegamuL(mu,L,b,0) + OmegamuL(mu,L,0,0)
    OmegamuL_phys = (
        OmegamuL_bM                                  # (N_MU, N_L, N_B, N_M)
        - OmegamuL_b0.unsqueeze(-1)                  # (N_MU, N_L, N_B, 1)
        + OmegamuL_00.view(N_MU, N_L, 1, 1)          # (N_MU, N_L, 1, 1)
    )

    # Вакуумный член M²/(2g)
    vac = (M_grid ** 2 / (2.0 * g)).view(1, 1, 1, N_M)

    return vac + dU_phys + OmegaL_phys + OmegamuL_phys


# ==============================================================================
# Полный расчёт с нуля
# ==============================================================================

def compute(
    mu_grid: torch.Tensor,
    L_grid: torch.Tensor,
    b_grid: torch.Tensor,
    M_grid: torch.Tensor,
    g: float,
    N_p: int = 100,
    N_phi: int = 100,
    n_modes_max: int = 20,
    chunk_size: int = 50,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Полный расчёт Omega_ren(mu, L, b, M) с нуля.

    Возвращает тензор (N_MU, N_L, N_B, N_M).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    M0 = torch.tensor([0.0], device=device, dtype=dtype)
    b0 = torch.tensor([0.0], device=device, dtype=dtype)

    # dU(b, M) — форма (N_B, N_M)
    dU_bM = _dU(b_grid, M_grid, N_p=N_p, N_phi=N_phi, device=device, dtype=dtype)

    # OmegaL(L, b, M), OmegaL(L, b, 0), OmegaL(L, 0, 0)
    OmegaL_bM = _omega_L(L_grid, b_grid, M_grid, N_p=N_p, device=device, dtype=dtype)
    OmegaL_b0 = _omega_L(L_grid, b_grid, M0,     N_p=N_p, device=device, dtype=dtype).squeeze(-1)
    OmegaL_00 = _omega_L(L_grid, b0,     M0,     N_p=N_p, device=device, dtype=dtype).squeeze(-1).squeeze(-1)

    # OmegamuL(mu, L, b, M) — форма (N_MU, N_L, N_B, N_M)
    OmegamuL_bM = _omega_muL(
        mu_grid, L_grid, b_grid, M_grid,
        N_p=N_p, n_modes_max=n_modes_max, chunk_size=chunk_size,
        device=device, dtype=dtype, show_progress=show_progress,
    )

    # OmegamuL(mu, L, b, 0) и (mu, L, 0, 0) — аналитически
    OmegamuL_b0 = _omega_muL_b0_batch(mu_grid, L_grid, b_grid, device=device, dtype=dtype)
    OmegamuL_00 = _omega_muL_00_batch(mu_grid, L_grid, n_max_global=n_modes_max, device=device, dtype=dtype)

    return assemble(
        dU_bM, OmegaL_bM, OmegaL_b0, OmegaL_00,
        OmegamuL_bM, OmegamuL_b0, OmegamuL_00,
        M_grid, g,
    )

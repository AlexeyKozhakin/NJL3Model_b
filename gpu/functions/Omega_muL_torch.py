"""
GPU-реализация Omega_muL(mu, L, b, M) на PyTorch.

Формула (theory.md, раздел 6):
    Omega_muL = -(2/L) * sum_n C_n * integral F±(p,n) dp
    F±(p,n) = ReLU( (mu - sqrt((E1±b)^2 + (2*pi*n/L)^2)) / (2*pi) )
    E1 = sqrt(M^2 + p^2),  p in [0, mu],  C_n = 1 при n=0, 2 при n>0

ReLU-подход (GPU_IMPLEMENTATION_PLAN.md):
    clamp(F, min=0) вместо явного определения границ интегрирования.
    Автоматически обрабатывает одно- и двухкорневые случаи без ветвлений.

p_max для каждого чанка (b, M):
    - Плюс-ветвь F+: достигает нуля при p = sqrt((mu-b)^2 - M^2) < mu для b>0.
    - Минус-ветвь F-: достигает нуля при p = sqrt((mu+b)^2 - M^2) > mu для b>0!
    Поэтому p_max = sqrt((mu + b_max_chunk)^2 - M_min_chunk^2) — с запасом берётся
    max b и min M в текущем чанке. ReLU зануляет всё лишнее автоматически.

Стратегия памяти (двухуровневый батчинг):
    Внешний цикл по mu (N_MU итераций).
    Внутренний цикл по чанкам (b, M) размером C × C.
    За один шаг: (C, C, N_L, N_modes, N_p) — размер контролируется C.
    При C=50, N_L=50, N_modes=5, N_p=100: 50^2*50*5*100*8B ≈ 500 MB.
    Уменьшайте CHUNK_SIZE в config.py при нехватке памяти.

Входные/выходные данные:
    Входы: mu_grid (N_MU,), L_grid (N_L,), b_grid (N_B,), M_grid (N_M,)
    Выход: (N_MU, N_L, N_B, N_M) — полный тензор Omega_muL
"""

import math
import torch


def compute(
    mu_grid: torch.Tensor,
    L_grid: torch.Tensor,
    b_grid: torch.Tensor,
    M_grid: torch.Tensor,
    N_p: int = 100,
    n_modes_max: int = 20,
    chunk_size: int = 50,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Вычисляет Omega_muL(mu, L, b, M) для полной сетки.

    Параметры
    ---------
    mu_grid     : (N_MU,)
    L_grid      : (N_L,)
    b_grid      : (N_B,)
    M_grid      : (N_M,)
    N_p         : число точек по p в [0, mu] для каждого mu
    n_modes_max : максимальное число KK-мод (с запасом; реально n_max <= 4)
    chunk_size  : C — размер чанка по b и M
    device      : torch.device
    dtype       : torch.dtype

    Возвращает
    ----------
    Tensor (N_MU, N_L, N_B, N_M)
    """
    if device is None:
        device = mu_grid.device

    N_MU = len(mu_grid)
    N_L  = len(L_grid)
    N_B  = len(b_grid)
    N_M  = len(M_grid)
    C    = chunk_size

    result = torch.zeros(N_MU, N_L, N_B, N_M, device=device, dtype=dtype)

    L_t = L_grid.to(device=device, dtype=dtype)

    for i_mu in range(N_MU):
        mu = mu_grid[i_mu].item()

        if mu <= 0.0:
            continue   # Omega_muL = 0 при mu=0

        # p_max вычисляется позже для каждого чанка; здесь только плейсхолдер

        # n-индексы: 0, 1, 2, ..., n_modes_max  — форма (N_modes,)
        n_vals = torch.arange(n_modes_max + 1, device=device, dtype=dtype)   # (N_modes,)
        C_n    = torch.where(n_vals == 0,
                             torch.ones_like(n_vals),
                             2.0 * torch.ones_like(n_vals))                   # (N_modes,)

        # --- Батчинг по (b, M) ---
        for i_b0 in range(0, N_B, C):
            i_b1 = min(i_b0 + C, N_B)
            b_chunk = b_grid[i_b0:i_b1].to(device=device, dtype=dtype)   # (Cb,)
            Cb = len(b_chunk)

            for i_M0 in range(0, N_M, C):
                i_M1 = min(i_M0 + C, N_M)
                M_chunk = M_grid[i_M0:i_M1].to(device=device, dtype=dtype)   # (Cm,)
                Cm = len(M_chunk)

                # p_max: консервативная граница, накрывающая минус-ветвь при больших b.
                # Минус-ветвь обращается в 0 при p = sqrt((mu+b)^2 - M^2).
                # Берём max по b и min по M в текущем чанке.
                b_max_chunk = b_chunk.max().item()
                M_min_chunk = M_chunk.min().item()
                p_max = math.sqrt(max((mu + b_max_chunk) ** 2 - M_min_chunk ** 2, 0.0))
                if p_max <= 0.0:
                    continue

                dp     = p_max / N_p
                p_vals = (torch.arange(N_p, device=device, dtype=dtype) + 0.5) * dp

                # acc: (N_L, Cb, Cm) — накапливаем сумму по n и p
                acc = torch.zeros(N_L, Cb, Cm, device=device, dtype=dtype)

                # --- Цикл по KK-модам ---
                for i_n in range(n_modes_max + 1):
                    n  = float(i_n)
                    cn = 1.0 if i_n == 0 else 2.0

                    # kn: (N_L,)  — зависит от L
                    kn = (2.0 * math.pi * n) / L_t    # (N_L,)

                    # Маска: пропускаем моду если kn > mu (F± < 0 для всех p,b,M)
                    if i_n > 0:
                        kn_min = kn.min().item()
                        if kn_min >= mu:
                            break   # все последующие моды тоже не вносят вклад

                    # --- Цикл по p ---
                    # Shapes:
                    #   p:  (1, 1, 1, N_p)
                    #   b:  (1, Cb, 1, 1)
                    #   M:  (1, 1, Cm, 1)
                    #   kn: (N_L, 1, 1, 1)
                    p  = p_vals.view(1, 1, 1, N_p)
                    b  = b_chunk.view(1, Cb, 1, 1)
                    M  = M_chunk.view(1, 1, Cm, 1)
                    kn_ = kn.view(N_L, 1, 1, 1)

                    E1 = torch.sqrt(M * M + p * p)              # (1, 1, Cm, N_p)

                    # F+ = (mu - sqrt((E1+b)^2 + kn^2)) / (2*pi)
                    # F- = (mu - sqrt((E1-b)^2 + kn^2)) / (2*pi)
                    two_pi = 2.0 * math.pi
                    F_plus  = (mu - torch.sqrt((E1 + b) ** 2 + kn_ ** 2)) / two_pi
                    F_minus = (mu - torch.sqrt((E1 - b) ** 2 + kn_ ** 2)) / two_pi

                    # ReLU — автоматически обрабатывает все случаи
                    F_plus  = F_plus.clamp(min=0.0)    # (N_L, Cb, Cm, N_p)
                    F_minus = F_minus.clamp(min=0.0)

                    # Интегрирование по p (midpoint rule): sum over N_p, multiply by dp
                    contrib = cn * (F_plus + F_minus).sum(dim=3) * dp   # (N_L, Cb, Cm)

                    acc += contrib

                # Omega_muL = -(2/L) * acc
                # L: (N_L,) → (N_L, 1, 1)
                result[i_mu, :, i_b0:i_b1, i_M0:i_M1] = -(2.0 / L_t.view(N_L, 1, 1)) * acc

    return result

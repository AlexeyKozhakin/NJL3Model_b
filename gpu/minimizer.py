"""
Поиск минимума Omega_ren по (b, M) для каждой точки (mu, L).

Два шага:
  1. Грубый argmin на сетке — O(N_B * N_M) на точку.
  2. Уточнение сплайном в окне `window × window` вокруг грубого минимума.

Вход:  omega_ren (N_MU, N_L, N_B, N_M), b_grid (N_B,), M_grid (N_M,)
Выход: b_min (N_MU, N_L), M_min (N_MU, N_L), omega_min (N_MU, N_L)
"""

import json
import datetime
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize as scipy_minimize


def find_minimum(
    omega_ren: torch.Tensor,
    b_grid: torch.Tensor,
    M_grid: torch.Tensor,
    window: int = 5,
    spline_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Находит минимум Omega_ren по (b, M) для каждой пары (mu, L).

    Параметры
    ---------
    omega_ren    : Tensor (N_MU, N_L, N_B, N_M)
    b_grid       : Tensor (N_B,)
    M_grid       : Tensor (N_M,)
    window       : ширина окна для сплайна (нечётное число, >= 4)
    spline_order : порядок сплайна по каждой оси (3 = кубический)

    Возвращает
    ----------
    b_min     : ndarray (N_MU, N_L)
    M_min     : ndarray (N_MU, N_L)
    omega_min : ndarray (N_MU, N_L)
    """
    N_MU, N_L, N_B, N_M = omega_ren.shape
    half = window // 2

    b_np     = b_grid.cpu().numpy()
    M_np     = M_grid.cpu().numpy()
    omega_np = omega_ren.cpu().numpy()   # (N_MU, N_L, N_B, N_M)

    b_min_arr     = np.zeros((N_MU, N_L))
    M_min_arr     = np.zeros((N_MU, N_L))
    omega_min_arr = np.zeros((N_MU, N_L))

    for i in range(N_MU):
        for j in range(N_L):
            slab = omega_np[i, j]          # (N_B, N_M)

            # --- Шаг 1: грубый argmin ---
            flat_idx = int(np.argmin(slab))
            i_b = flat_idx // N_M
            i_M = flat_idx % N_M

            # --- Шаг 2: сплайн-уточнение ---
            b_lo = max(i_b - half, 0)
            b_hi = min(i_b + half + 1, N_B)
            M_lo = max(i_M - half, 0)
            M_hi = min(i_M + half + 1, N_M)

            b_win     = b_np[b_lo:b_hi]
            M_win     = M_np[M_lo:M_hi]
            omega_win = slab[b_lo:b_hi, M_lo:M_hi]

            refined = False
            if len(b_win) >= spline_order + 1 and len(M_win) >= spline_order + 1:
                try:
                    spline = RectBivariateSpline(
                        b_win, M_win, omega_win,
                        kx=spline_order, ky=spline_order,
                    )
                    res = scipy_minimize(
                        lambda x: float(spline(x[0], x[1]).item()),
                        x0=[b_np[i_b], M_np[i_M]],
                        bounds=[(b_win[0], b_win[-1]), (M_win[0], M_win[-1])],
                        method='L-BFGS-B',
                        options={'ftol': 1e-12, 'gtol': 1e-10},
                    )
                    if res.success or res.fun < slab[i_b, i_M]:
                        b_min_arr[i, j]     = float(res.x[0])
                        M_min_arr[i, j]     = float(res.x[1])
                        omega_min_arr[i, j] = float(res.fun)
                        refined = True
                except Exception:
                    pass

            if not refined:
                b_min_arr[i, j]     = b_np[i_b]
                M_min_arr[i, j]     = M_np[i_M]
                omega_min_arr[i, j] = slab[i_b, i_M]

    return b_min_arr, M_min_arr, omega_min_arr


def save_phase_diagram(
    mu_grid: torch.Tensor,
    L_grid: torch.Tensor,
    b_min: np.ndarray,
    M_min: np.ndarray,
    omega_min: np.ndarray,
    output_path: str,
    meta: dict | None = None,
) -> None:
    """
    Сохраняет результаты фазовой диаграммы в JSON.

    Формат файла:
    {
        "meta": { "date": ..., "N_MU": ..., "N_L": ..., ... },
        "mu":        [float, ...],          # (N_MU,)
        "L":         [float, ...],          # (N_L,)
        "M_min":     [[float, ...], ...],   # (N_MU, N_L)
        "b_min":     [[float, ...], ...],
        "Omega_min": [[float, ...], ...]
    }
    """
    payload: dict = {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "N_MU": int(len(mu_grid)),
            "N_L":  int(len(L_grid)),
            **(meta or {}),
        },
        "mu":        mu_grid.cpu().tolist(),
        "L":         L_grid.cpu().tolist(),
        "M_min":     M_min.tolist(),
        "b_min":     b_min.tolist(),
        "Omega_min": omega_min.tolist(),
    }

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved: {output_path}  ({size_mb:.2f} MB)")

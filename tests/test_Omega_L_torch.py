"""
Верификация gpu/functions/Omega_L_torch.py против Numba-реализации Omega_L_fast.py.

Этап 1.4: совпадение на 5–10 точках, погрешность < 1e-4.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L_fast import Omega_L_scalar
from gpu.functions.Omega_L_torch import compute as Omega_L_torch

DEVICE = torch.device('cpu')
DTYPE  = torch.float64
ATOL   = 1e-4
RTOL   = 1e-4

N_P = 200


def _close(val, ref, label):
    if abs(ref) > 1e-4:
        err = abs(val - ref) / abs(ref)
        assert err < RTOL, f"{label}: rel err={err:.3e}  val={val:.8g}  ref={ref:.8g}"
    else:
        err = abs(val - ref)
        assert err < ATOL, f"{label}: abs err={err:.3e}  val={val:.8g}  ref={ref:.8g}"


# Точки (L, b, M)
POINTS = [
    (1.0,  0.0,  0.0),
    (1.0,  0.0,  1.0),
    (1.0,  0.0,  3.0),
    (1.0,  1.0,  1.0),
    (1.0,  2.0,  1.0),
    (2.0,  0.5,  0.5),
    (2.0,  1.0,  2.0),
    (0.5,  0.0,  1.0),
    (3.0,  2.0,  2.0),
    (4.0,  1.0,  0.5),
]


@pytest.mark.parametrize("L,b,M", POINTS)
def test_omega_L_single_point(L, b, M):
    ref = Omega_L_scalar(L, b, M, phi=0.0, N_h_p1=N_P, N_h_p2=N_P)
    L_t = torch.tensor([L], dtype=DTYPE)
    b_t = torch.tensor([b], dtype=DTYPE)
    M_t = torch.tensor([M], dtype=DTYPE)
    val = Omega_L_torch(L_t, b_t, M_t, N_p=N_P, phi=0.0, device=DEVICE, dtype=DTYPE).item()
    _close(val, ref, f"L={L} b={b} M={M}")


def test_omega_L_batch_matches_scalar():
    """Батчевый расчёт совпадает со скалярным Numba поточечно."""
    L_vals = [0.5, 1.0, 2.0]
    b_vals = [0.0, 1.0]
    M_vals = [0.0, 1.0, 2.0]

    L_t = torch.tensor(L_vals, dtype=DTYPE)
    b_t = torch.tensor(b_vals, dtype=DTYPE)
    M_t = torch.tensor(M_vals, dtype=DTYPE)
    result = Omega_L_torch(L_t, b_t, M_t, N_p=N_P, phi=0.0, device=DEVICE, dtype=DTYPE)
    # result: (3, 2, 3)

    for i, L in enumerate(L_vals):
        for j, b in enumerate(b_vals):
            for k, M in enumerate(M_vals):
                ref = Omega_L_scalar(L, b, M, phi=0.0, N_h_p1=N_P, N_h_p2=N_P)
                val = result[i, j, k].item()
                _close(val, ref, f"batch L={L} b={b} M={M}")

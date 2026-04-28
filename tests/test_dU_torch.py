"""
Верификация gpu/functions/dU_torch.py против Numba-реализации dU_fast.py.

Этап 1.3: совпадение на 5–10 точках, погрешность < 1e-4.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.dU_fast import dU_scalar
from gpu.functions.dU_torch import compute as dU_torch

DEVICE = torch.device('cpu')
DTYPE  = torch.float64
ATOL   = 1e-4
RTOL   = 1e-4


def _close(val, ref, label):
    if abs(ref) > 1.0:
        err = abs(val - ref) / abs(ref)
        assert err < RTOL, f"{label}: rel err={err:.3e}  val={val:.8g}  ref={ref:.8g}"
    else:
        err = abs(val - ref)
        assert err < ATOL, f"{label}: abs err={err:.3e}  val={val:.8g}  ref={ref:.8g}"


# Точки (b, M) для проверки
POINTS = [
    (0.0,  0.0),   # dU=0 по теореме
    (0.0,  1.0),
    (0.0,  3.0),
    (1.0,  1.0),
    (1.0,  3.0),
    (2.0,  0.5),
    (2.0,  2.0),
    (3.0,  1.0),
    (0.5,  4.0),
    (4.0,  4.0),
]

N_P   = 200
N_PHI = 200


@pytest.mark.parametrize("b,M", POINTS)
def test_dU_single_point(b, M):
    ref = dU_scalar(b, M, N_P, N_PHI)
    b_t = torch.tensor([b], dtype=DTYPE)
    M_t = torch.tensor([M], dtype=DTYPE)
    val = dU_torch(b_t, M_t, N_p=N_P, N_phi=N_PHI, device=DEVICE, dtype=DTYPE).item()
    _close(val, ref, f"b={b} M={M}")


def test_dU_zero_at_M0():
    """dU(b,0) = 0 для любого b (theory.md раздел 9.1)."""
    b_vals = torch.linspace(0.0, 4.0, 9, dtype=DTYPE)
    M_zero = torch.tensor([0.0], dtype=DTYPE)
    result = dU_torch(b_vals, M_zero, N_p=100, N_phi=100, device=DEVICE, dtype=DTYPE)
    # result: (9, 1)
    assert result.abs().max().item() < 1e-8, f"dU(b,0) != 0: max={result.abs().max():.3e}"


def test_dU_batch_matches_scalar():
    """Батчевый расчёт совпадает со скалярным Numba поточечно."""
    b_vals = [0.5, 1.5, 3.0]
    M_vals = [0.5, 1.0, 2.0]
    b_t = torch.tensor(b_vals, dtype=DTYPE)
    M_t = torch.tensor(M_vals, dtype=DTYPE)
    result = dU_torch(b_t, M_t, N_p=N_P, N_phi=N_PHI, device=DEVICE, dtype=DTYPE)
    # result: (3, 3)
    for i, b in enumerate(b_vals):
        for j, M in enumerate(M_vals):
            ref = dU_scalar(b, M, N_P, N_PHI)
            val = result[i, j].item()
            _close(val, ref, f"batch b={b} M={M}")

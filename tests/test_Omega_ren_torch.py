"""
Верификация gpu/Omega_ren_torch.py против Numba Omega_ren_fast.py.

Этап 1.6: совпадение Omega_ren для 5+ точек (mu, L, b, M), погрешность < 1e-3.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_ren_fast import Omega_ren_fast
from gpu.Omega_ren_torch import compute as omega_ren_torch

DEVICE = torch.device('cpu')
DTYPE  = torch.float64
RTOL   = 1e-2    # Omega_ren — разность больших чисел, допускаем 1% погрешности
ATOL   = 1e-3
N_P    = 200

G = -1.0


def _ref(mu, L, b, M):
    return Omega_ren_fast(M, b, mu, L, G, N_h_p1=N_P, N_h_p2=N_P,
                          N_h_p=N_P, N_h_phi=N_P, N_h_mu=N_P)


def _gpu(mu, L, b, M):
    result = omega_ren_torch(
        torch.tensor([mu], dtype=DTYPE),
        torch.tensor([L],  dtype=DTYPE),
        torch.tensor([b],  dtype=DTYPE),
        torch.tensor([M],  dtype=DTYPE),
        g=G, N_p=N_P, N_phi=N_P,
        device=DEVICE, dtype=DTYPE,
    )
    return result[0, 0, 0, 0].item()


def _close(val, ref, label):
    if abs(ref) > 1e-2:
        err = abs(val - ref) / abs(ref)
        assert err < RTOL, f"{label}: rel err={err:.3e}  val={val:.6g}  ref={ref:.6g}"
    else:
        err = abs(val - ref)
        assert err < ATOL, f"{label}: abs err={err:.3e}  val={val:.6g}  ref={ref:.6g}"


# Точки (mu, L, b, M)
POINTS = [
    (3.0, 1.0, 0.0, 1.0),   # b=0, стандартный случай
    (3.0, 1.0, 1.0, 1.0),   # b=M
    (3.0, 1.0, 2.0, 0.5),   # b > M (два корня в minus-ветви)
    (5.0, 2.0, 1.0, 2.0),
    (2.0, 0.5, 0.5, 0.5),
    (4.0, 3.0, 2.0, 1.5),
    (1.5, 1.0, 0.0, 0.0),   # M=0: vacuum term = 0
]


@pytest.mark.parametrize("mu,L,b,M", POINTS)
def test_omega_ren_point(mu, L, b, M):
    ref = _ref(mu, L, b, M)
    val = _gpu(mu, L, b, M)
    _close(val, ref, f"mu={mu} L={L} b={b} M={M}")


def test_omega_ren_batch():
    """Батчевый расчёт: (N_MU, N_L, N_B, N_M) совпадает со скалярным Numba."""
    mu_vals = [2.0, 4.0]
    L_vals  = [1.0, 2.0]
    b_vals  = [0.0, 1.0]
    M_vals  = [0.5, 2.0]

    result = omega_ren_torch(
        torch.tensor(mu_vals, dtype=DTYPE),
        torch.tensor(L_vals,  dtype=DTYPE),
        torch.tensor(b_vals,  dtype=DTYPE),
        torch.tensor(M_vals,  dtype=DTYPE),
        g=G, N_p=N_P, N_phi=N_P,
        device=DEVICE, dtype=DTYPE,
    )

    for i, mu in enumerate(mu_vals):
        for j, L in enumerate(L_vals):
            for k, b in enumerate(b_vals):
                for m, M in enumerate(M_vals):
                    ref = _ref(mu, L, b, M)
                    val = result[i, j, k, m].item()
                    _close(val, ref, f"batch mu={mu} L={L} b={b} M={M}")


def test_vacuum_term():
    """При M=0: вакуумный член M²/(2g)=0, dU=0 → Omega_ren = OmegaL_phys + OmegamuL_phys."""
    mu, L, b = 3.0, 1.0, 0.5
    ref = _ref(mu, L, b, 0.0)
    val = _gpu(mu, L, b, 0.0)
    _close(val, ref, f"M=0 mu={mu} L={L} b={b}")

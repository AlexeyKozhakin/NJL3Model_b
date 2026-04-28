"""
Верификация gpu/functions/Omega_muL_torch.py против Numba fun_Omega_L_mu_int.

Этап 1.5: совпадение на 10+ точках, включая двухкорневые случаи и граничные.
Порог: погрешность < 1e-3.
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import fun_Omega_L_mu_int
from gpu.functions.Omega_muL_torch import compute as Omega_muL_torch

DEVICE = torch.device('cpu')
DTYPE  = torch.float64
ATOL   = 1e-3
RTOL   = 1e-3

N_P   = 500   # большое N для точности (ReLU-метод менее точен у границ)


def _close(val, ref, label):
    if abs(ref) > 1e-3:
        err = abs(val - ref) / abs(ref)
        assert err < RTOL, f"{label}: rel err={err:.3e}  val={val:.8g}  ref={ref:.8g}"
    else:
        err = abs(val - ref)
        assert err < ATOL, f"{label}: abs err={err:.3e}  val={val:.8g}  ref={ref:.8g}"


# Точки (mu, L, b, M)
NORMAL_POINTS = [
    (3.0,  1.0,  0.0,  0.0),
    (3.0,  1.0,  0.5,  1.0),
    (3.0,  1.0,  1.0,  1.0),
    (5.0,  2.0,  1.0,  2.0),
    (2.0,  1.0,  0.0,  1.0),
    (4.0,  3.0,  2.0,  1.5),
    (7.0,  4.0,  1.0,  2.0),
    (1.5,  0.5,  0.0,  0.5),
]

# Двухкорневые случаи: |b| > M, F(0) < 0
TWO_ROOT_POINTS = [
    (3.0,  1.0,  2.0,  0.5),   # b > M
    (5.0,  2.0,  3.0,  1.0),   # b > M, несколько мод
    (4.0,  1.0,  3.0,  0.5),
]

# Граничные случаи: mu маленький — n_max = -1 (нет вклада)
EDGE_POINTS = [
    (0.5,  1.0,  0.0,  1.0),   # mu мало, E1 > mu → вклад = 0
    (0.1,  1.0,  0.0,  0.0),
    (0.0,  1.0,  1.0,  1.0),   # mu=0 → 0
]


@pytest.mark.parametrize("mu,L,b,M", NORMAL_POINTS)
def test_omega_muL_normal(mu, L, b, M):
    ref = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=500)
    val = _single(mu, L, b, M)
    _close(val, ref, f"normal mu={mu} L={L} b={b} M={M}")


@pytest.mark.parametrize("mu,L,b,M", TWO_ROOT_POINTS)
def test_omega_muL_two_root(mu, L, b, M):
    ref = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=500)
    val = _single(mu, L, b, M)
    _close(val, ref, f"two-root mu={mu} L={L} b={b} M={M}")


@pytest.mark.parametrize("mu,L,b,M", EDGE_POINTS)
def test_omega_muL_edge(mu, L, b, M):
    ref = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=500)
    val = _single(mu, L, b, M)
    _close(val, ref, f"edge mu={mu} L={L} b={b} M={M}")


def test_omega_muL_batch():
    """Батчевый расчёт (N_MU, N_L, N_B, N_M) совпадает со скалярным Numba."""
    mu_vals = [2.0, 4.0]
    L_vals  = [1.0, 2.0]
    b_vals  = [0.0, 1.0]
    M_vals  = [0.5, 1.5]

    result = Omega_muL_torch(
        torch.tensor(mu_vals, dtype=DTYPE),
        torch.tensor(L_vals,  dtype=DTYPE),
        torch.tensor(b_vals,  dtype=DTYPE),
        torch.tensor(M_vals,  dtype=DTYPE),
        N_p=N_P, device=DEVICE, dtype=DTYPE,
    )

    for i, mu in enumerate(mu_vals):
        for j, L in enumerate(L_vals):
            for k, b in enumerate(b_vals):
                for m, M in enumerate(M_vals):
                    ref = fun_Omega_L_mu_int(mu, L, b, M, phi=0, N_h=500)
                    val = result[i, j, k, m].item()
                    _close(val, ref, f"batch mu={mu} L={L} b={b} M={M}")


# ---- Вспомогательная функция -----------------------------------------------

def _single(mu, L, b, M):
    result = Omega_muL_torch(
        torch.tensor([mu], dtype=DTYPE),
        torch.tensor([L],  dtype=DTYPE),
        torch.tensor([b],  dtype=DTYPE),
        torch.tensor([M],  dtype=DTYPE),
        N_p=N_P, device=DEVICE, dtype=DTYPE,
    )
    return result[0, 0, 0, 0].item()

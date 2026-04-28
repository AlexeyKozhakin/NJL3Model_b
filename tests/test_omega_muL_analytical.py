"""
Верификация аналитических функций Omega_muL против Numba-реализации.

Этап 1.2: совпадение на 10+ точках, погрешность < 1e-6.
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import fun_Omega_L_mu_int
from gpu.functions.Omega_muL_00_torch import compute as omega_00
from gpu.functions.Omega_muL_b0_torch import compute as omega_b0

ATOL = 1e-6    # абсолютный порог
RTOL = 1e-5    # относительный порог


def _close(val, ref, label):
    if abs(ref) > 1.0:
        err = abs(val - ref) / abs(ref)
        assert err < RTOL, f"{label}: rel err={err:.3e}  val={val:.8g}  ref={ref:.8g}"
    else:
        err = abs(val - ref)
        assert err < ATOL, f"{label}: abs err={err:.3e}  val={val:.8g}  ref={ref:.8g}"


# ==============================================================================
# Omega_muL(mu, L, 0, 0)
# ==============================================================================

POINTS_00 = [
    (0.0,  1.0),
    (0.5,  1.0),
    (1.0,  1.0),
    (2.0,  1.0),
    (3.0,  1.0),
    (5.0,  1.0),
    (2.0,  0.5),
    (4.0,  2.0),
    (6.0,  2.0),
    (7.0,  4.0),
    (1.0,  4.0),
    (3.5,  0.3),
]


@pytest.mark.parametrize("mu,L", POINTS_00)
def test_omega_muL_00(mu, L):
    ref = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi=0, N_h=500)
    val = omega_00(mu, L)
    _close(val, ref, f"00 mu={mu} L={L}")


# ==============================================================================
# Omega_muL(mu, L, b, 0)
# ==============================================================================

POINTS_B0 = [
    # (mu, L, b)
    (3.0, 1.0, 0.0),    # b=0 должно совпадать с omega_00
    (3.0, 1.0, 0.5),
    (3.0, 1.0, 1.0),
    (3.0, 1.0, 2.0),    # b > некоторых Sn → два корня
    (3.0, 1.0, 3.5),    # b > mu → n_max^+ = -1
    (5.0, 2.0, 1.0),
    (5.0, 2.0, 3.0),
    (2.0, 0.5, 0.5),
    (7.0, 4.0, 2.0),
    (1.0, 1.0, 0.8),
    (4.0, 2.0, 4.5),    # b > mu
    (6.0, 3.0, 5.9),    # b ≈ mu
]


@pytest.mark.parametrize("mu,L,b", POINTS_B0)
def test_omega_muL_b0(mu, L, b):
    ref = fun_Omega_L_mu_int(mu, L, b, 0.0, phi=0, N_h=500)
    val = omega_b0(mu, L, b)
    _close(val, ref, f"b0 mu={mu} L={L} b={b}")


def test_omega_b0_equals_00_at_b0():
    """omega_b0(b=0) должно совпадать с omega_00 для нескольких точек."""
    for mu, L in [(2.0, 1.0), (4.0, 2.0), (7.0, 4.0)]:
        v00 = omega_00(mu, L)
        vb0 = omega_b0(mu, L, 0.0)
        assert abs(v00 - vb0) < 1e-12, \
            f"Расхождение при b=0: mu={mu} L={L}  00={v00:.8g}  b0={vb0:.8g}"

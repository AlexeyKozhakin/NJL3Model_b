"""
Проверка симметрии b -> -b для Omega_mu_L.

Физический потенциал должен быть симметричен относительно замены b на -b,
так как спектральные ветви (E1+b) и (E1-b) просто меняются местами.
"""

import numpy as np
import pytest
from functions.Omega_mu_L import fun_Omega_L_mu_int


MU = 3.0
L = 2.0
PHI = 0
N_H = 100


@pytest.mark.parametrize("M", [0.5, 1.5, 3.0])
@pytest.mark.parametrize("b", [0.0, 0.5, 1.0, 2.0, 2.5, 5.0])
def test_symmetry_b_to_minus_b(M, b):
    """Omega_mu_L(mu, L, b, M) == Omega_mu_L(mu, L, -b, M)"""
    val_pos = fun_Omega_L_mu_int(MU, L, b, M, PHI, N_H)
    val_neg = fun_Omega_L_mu_int(MU, L, -b, M, PHI, N_H)
    assert val_pos == pytest.approx(val_neg, abs=1e-6), (
        f"Asymmetry detected: M={M}, b={b}: {val_pos} != {val_neg}"
    )

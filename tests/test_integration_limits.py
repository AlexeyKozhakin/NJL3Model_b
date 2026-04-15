"""
Проверка пределов интегрирования из кода против аналитических формул.

Тестируем integration_limits_plus и integration_limits_minus
на соответствие выводу из docs/theory.md (раздел 6.3).
"""

import numpy as np
import pytest
from functions.Omega_mu_L import integration_limits_plus, integration_limits_minus, n_max_plus, n_max_minus


MU = 3.0
L = 2.0
PHI = 0
M = 1.5


def _manual_p_right_plus(b, M_val, S_n):
    """Theory.md eq (6.8): p_right^2 = (S_n - b)^2 - M^2"""
    val = (S_n - b) ** 2 - M_val ** 2
    return np.sqrt(np.maximum(val, 0.0)) if val >= 0 else np.nan


def _manual_p_right_minus(b, M_val, S_n):
    """Theory.md eq (6.9): p_right^2 = (S_n + b)^2 - M^2 (primary branch)"""
    val = (S_n + b) ** 2 - M_val ** 2
    return np.sqrt(np.maximum(val, 0.0)) if val >= 0 else np.nan


@pytest.mark.parametrize("b", [0.5, -0.5, 2.0, -2.0, 2.5, -2.5])
def test_integration_limits_plus(b):
    Nmax = n_max_plus(M, b, L, MU, PHI)
    if Nmax < 0:
        pytest.skip("No contributing modes")

    p_left_code, p_right_code = integration_limits_plus(M, b, MU, L, Nmax, PHI)
    n_vals = np.arange(Nmax + 1)

    for n in n_vals:
        S_n = np.sqrt(MU**2 - (2 * np.pi * n / L) ** 2)
        expected = _manual_p_right_plus(b, M, S_n)
        actual = p_right_code[n]
        assert actual == pytest.approx(expected, abs=1e-10), (
            f"Mismatch in integration_limits_plus: b={b}, n={n}, "
            f"expected={expected}, actual={actual}"
        )
        assert p_left_code[n] == 0.0


@pytest.mark.parametrize("b", [0.5, -0.5, 2.0, -2.0, 2.5, -2.5])
def test_integration_limits_minus(b):
    Nmax = n_max_minus(M, b, L, MU, PHI)
    if Nmax < 0:
        pytest.skip("No contributing modes")

    p_left_code, p_right_code = integration_limits_minus(M, b, MU, L, Nmax, PHI)
    n_vals = np.arange(Nmax + 1)

    for n in n_vals:
        S_n = np.sqrt(MU**2 - (2 * np.pi * n / L) ** 2)
        expected = _manual_p_right_minus(b, M, S_n)
        actual = p_right_code[n]
        assert actual == pytest.approx(expected, abs=1e-10), (
            f"Mismatch in integration_limits_minus: b={b}, n={n}, "
            f"expected={expected}, actual={actual}"
        )
        assert p_left_code[n] == 0.0

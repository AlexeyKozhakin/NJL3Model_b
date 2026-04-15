"""
Регрессионные тесты для двухкорневых случаев (|b| > M, F(0) < 0).

При |b| > M функция F_n^±(p1) может иметь максимум в p_ext = sqrt(b^2-M^2).
Если F(0) < 0, но максимум положителен, существует ДВА корня:
- левый корень p_left > 0
- правый корень p_right > p_left

Старый код ошибочно всегда брал p_left = 0. Новая numba-реализация
должна давать результат, совпадающий с brute-force интегрированием
по правильной области [p_left, p_right].
"""

import numpy as np
import pytest
from functions.Omega_mu_L import fun_Omega_L_mu_int, Fpn_plus_numpy, Fpn_minus_numpy


MU = 3.0
L = 2.0
PHI = 0
N_H_FINE = 20000


def _brute_force_reference(mu, L_val, b, M, phi, branch):
    """
    Brute-force интегрирование по p1 с очень мелкой сеткой.
    Используем np.trapz с Heaviside-обрезкой.
    """
    p_max = 15.0
    p1_vals = np.linspace(0, p_max, N_H_FINE)
    n_vals = np.arange(0, 100)  # достаточно большой диапазон

    total = 0.0
    for n in n_vals:
        if branch == 'plus':
            F = Fpn_plus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L_val, phi, mu).ravel()
        else:
            F = Fpn_minus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L_val, phi, mu).ravel()

        # Учитываем только положительную область
        F_positive = np.where(F > 0, F, 0.0)
        integral = np.trapezoid(F_positive, p1_vals)

        if n == 0:
            total += integral
        else:
            # Проверяем, существует ли ещё какая-то мода — если интеграл упал до нуля, дальше не идём
            # (для phi=0 симметрия n <-> -n даёт множитель 2)
            total += 2 * integral
            if integral == 0.0:
                break

    return -(2.0 / L_val) * total


@pytest.mark.parametrize("b, M", [
    (5.0, 1.5),
    (-5.0, 1.5),
    (4.0, 1.0),
    (-4.0, 1.0),
])
def test_two_roots_against_brute_force(b, M):
    """Сравнение numba-реализации с brute-force интегрированием."""
    numba_val = fun_Omega_L_mu_int(MU, L, b, M, PHI, N_h=400)
    # Brute-force берёт сумму обеих ветвей
    brute_plus = _brute_force_reference(MU, L, b, M, PHI, 'plus')
    brute_minus = _brute_force_reference(MU, L, b, M, PHI, 'minus')
    brute_total = brute_plus + brute_minus

    assert numba_val == pytest.approx(brute_total, abs=1e-4), (
        f"Two-root case mismatch: b={b}, M={M}: numba={numba_val}, brute={brute_total}"
    )




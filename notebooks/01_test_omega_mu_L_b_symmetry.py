"""
Отладочный скрипт 01: Проверка симметрии b -> -b для Omega_mu_L

Цель: проверить, совпадают ли:
1. Unp(mu, L,  b, M)  vs  Unm(mu, L, -b, M)   (ветви должны меняться местами)
2. Total(mu, L, b, M)  vs  Total(mu, L, -b, M)  (полный результат должен быть симметричен)

Если (1) нарушается, значит есть баг в пределах интегрирования или в n_max для одного из знаков b.
Если (2) нарушается при нарушении (1), значит ошибки в plus/minus не компенсируют друг друга.
"""

import numpy as np
import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import (
    Fpn_plus_numpy,
    Fpn_minus_numpy,
    divide_intervals,
    integration_limits_plus,
    integration_limits_minus,
    n_max_plus,
    n_max_minus,
)


def fun_Omega_L_mu_int_detailed(mu, L, b, M, phi=0, N_h=100):
    """
    Детальная версия fun_Omega_L_mu_int.
    Возвращает кортеж (Unp, Unm, total).
    """
    # =========================== PLUS BRANCH ===========================
    Nmax_plus = n_max_plus(M, b, L, mu, phi)
    if Nmax_plus < 0:
        Unp = 0.0
    else:
        p_left, p_right = integration_limits_plus(M, b, mu, L, Nmax_plus, phi)
        p_tens = np.stack((p_left, p_right))
        bounds = p_tens.swapaxes(0, 1)
        midpoints, steps = divide_intervals(bounds, N_h)
        n_plus = np.linspace(0, Nmax_plus, Nmax_plus + 1)
        F_tp = Fpn_plus_numpy(midpoints, n_plus, M, b, L, phi, mu) * steps[:, np.newaxis]
        Unp = 2 * np.sum(F_tp[1:, :]) + np.sum(F_tp[0, :])

    # =========================== MINUS BRANCH ==========================
    Nmax_minus = n_max_minus(M, b, L, mu, phi)
    if Nmax_minus < 0:
        Unm = 0.0
    else:
        p_left, p_right = integration_limits_minus(M, b, mu, L, Nmax_minus, phi)
        p_tens = np.stack((p_left, p_right))
        bounds = p_tens.swapaxes(0, 1)
        midpoints, steps = divide_intervals(bounds, N_h)
        n_minus = np.linspace(0, Nmax_minus, Nmax_minus + 1)
        F_tp = Fpn_minus_numpy(midpoints, n_minus, M, b, L, phi, mu) * steps[:, np.newaxis]
        Unm = 2 * np.sum(F_tp[1:, :]) + np.sum(F_tp[0, :])

    total = -(2.0 / L) * (Unp + Unm)
    return float(Unp), float(Unm), float(total)


def main():
    print("=" * 70)
    print("Тест симметрии b -> -b для Omega_mu_L")
    print("=" * 70)

    # Параметры для теста
    mu = 3.0
    L = 2.0
    M = 1.5
    phi = 0
    N_h = 100

    # Набор значений b для проверки
    b_values = np.array([0.0, 0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 3.0])

    print(f"\nПараметры: mu={mu}, L={L}, M={M}, phi={phi}, N_h={N_h}\n")
    print(f"{'b':>8} | {'Unp(b)':>14} | {'Unm(-b)':>14} | {'diff_plus':>12} | {'Total(b)':>14} | {'Total(-b)':>14} | {'diff_tot':>12}")
    print("-" * 100)

    all_ok = True
    for b in b_values:
        unp_b, unm_b, total_b = fun_Omega_L_mu_int_detailed(mu, L, b, M, phi, N_h)
        unp_mb, unm_mb, total_mb = fun_Omega_L_mu_int_detailed(mu, L, -b, M, phi, N_h)

        # Ожидаем: Unp(b) == Unm(-b) и Unm(b) == Unp(-b)
        diff_plus = abs(unp_b - unm_mb)
        diff_minus = abs(unm_b - unp_mb)
        diff_total = abs(total_b - total_mb)

        status = "OK" if (diff_plus < 1e-10 and diff_total < 1e-10) else "FAIL"
        if status == "FAIL":
            all_ok = False

        print(
            f"{b:8.2f} | {unp_b:14.6e} | {unm_mb:14.6e} | {diff_plus:12.3e} | "
            f"{total_b:14.6e} | {total_mb:14.6e} | {diff_total:12.3e} | {status}"
        )

        # Дополнительно выведем n_max для диагностики
        n_plus_b = n_max_plus(M, b, L, mu, phi)
        n_minus_mb = n_max_minus(M, -b, L, mu, phi)
        if n_plus_b != n_minus_mb:
            print(f"  [DIAG] n_max_plus(M,{b})={n_plus_b} != n_max_minus(M,{-b})={n_minus_mb}")

    print("-" * 100)
    if all_ok:
        print("\n>>> РЕЗУЛЬТАТ: Симметрия b -> -b СОБЛЮДАЕТСЯ на уровне машинной точности.")
        print("    Это означает, что ошибки в integration_limits (если есть) компенсируются")
        print("    суммой plus + minus ветвей.")
    else:
        print("\n>>> РЕЗУЛЬТАТ: Симметрия НАРУШЕНА! Нужна детальная диагностика.")
    print("=" * 70)


if __name__ == "__main__":
    main()

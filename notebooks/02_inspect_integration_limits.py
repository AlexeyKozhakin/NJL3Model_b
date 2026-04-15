"""
Отладочный скрипт 02: Инспекция пределов интегрирования по p1 для Omega_mu_L

Цель: сравнить пределы интегрирования, которые выдает код
(integration_limits_plus / integration_limits_minus), с аналитическими
формулами из theory.md (раздел 6.3).

Для каждой моды n выводим:
- Nmax_plus, Nmax_minus
- p_left, p_right из кода
- p_right, вычисленное вручную по формулам:
    plus:  p^2 = (S_n - b)^2 - M^2
    minus: p^2 = (S_n + b)^2 - M^2   [или (b - S_n)^2 - M^2]
- где S_n = sqrt(mu^2 - (2*pi*n/L)^2)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import (
    integration_limits_plus,
    integration_limits_minus,
    n_max_plus,
    n_max_minus,
)


def manual_p_right_plus(b, M, S_n):
    """Theory.md eq (6.8): p_right^2 = (S_n - b)^2 - M^2"""
    val = (S_n - b) ** 2 - M ** 2
    return np.sqrt(np.maximum(val, 0.0)) if val >= 0 else np.nan


def manual_p_right_minus(b, M, S_n):
    """Theory.md eq (6.9): p_right^2 = (S_n + b)^2 - M^2  (primary branch)"""
    val1 = (S_n + b) ** 2 - M ** 2
    val2 = (b - S_n) ** 2 - M ** 2
    # Primary branch is val1 for b > 0, val2 when it makes sense
    return np.sqrt(np.maximum(val1, 0.0)) if val1 >= 0 else np.nan, \
           np.sqrt(np.maximum(val2, 0.0)) if val2 >= 0 else np.nan


def inspect_branch(branch_name, b, M, mu, L, phi, n_max_fn, limits_fn, manual_fn):
    print(f"\n{'='*70}")
    print(f"Branch: {branch_name} | b={b}, M={M}, mu={mu}, L={L}")
    print(f"{'='*70}")
    Nmax = n_max_fn(M, b, L, mu, phi)
    print(f"Nmax = {Nmax}")
    if Nmax < 0:
        print("No modes contribute.")
        return

    p_left_arr, p_right_arr = limits_fn(M, b, mu, L, Nmax, phi)
    n_vals = np.arange(Nmax + 1)

    print(f"{'n':>4} | {'S_n':>10} | {'code p_left':>12} | {'code p_right':>13} | {'manual p_right':>15} | {'match?':>6}")
    print("-" * 80)
    for n in n_vals:
        S_n = np.sqrt(max(mu**2 - (2 * np.pi * n / L) ** 2, 0.0))
        p_left = p_left_arr[n]
        p_right_code = p_right_arr[n]
        p_right_manual = manual_fn(b, M, S_n)

        match = "OK" if np.isclose(p_right_code, p_right_manual, atol=1e-12) else "FAIL"
        if np.isnan(p_right_code) or np.isnan(p_right_manual):
            match = "N/A"

        print(f"{n:4d} | {S_n:10.4f} | {p_left:12.6f} | {p_right_code:13.6f} | {p_right_manual:15.6f} | {match:>6}")


def main():
    mu = 3.0
    L = 2.0
    M = 1.5
    phi = 0

    # Случай b > 0
    b_pos = 2.0
    inspect_branch("PLUS", b_pos, M, mu, L, phi, n_max_plus, integration_limits_plus,
                   lambda b, M, S: manual_p_right_plus(b, M, S))
    inspect_branch("MINUS", b_pos, M, mu, L, phi, n_max_minus, integration_limits_minus,
                   lambda b, M, S: manual_p_right_minus(b, M, S)[0])

    # Случай b < 0
    b_neg = -2.0
    inspect_branch("PLUS", b_neg, M, mu, L, phi, n_max_plus, integration_limits_plus,
                   lambda b, M, S: manual_p_right_plus(b, M, S))
    inspect_branch("MINUS", b_neg, M, mu, L, phi, n_max_minus, integration_limits_minus,
                   lambda b, M, S: manual_p_right_minus(b, M, S)[0])

    # Случай b=0.5 (обе ветви активны)
    b_mid = 0.5
    print(f"\n{'#'*70}")
    print(f"# Case: b = {b_mid} (both branches expected active)")
    print(f"{'#'*70}")
    inspect_branch("PLUS", b_mid, M, mu, L, phi, n_max_plus, integration_limits_plus,
                   lambda b, M, S: manual_p_right_plus(b, M, S))
    inspect_branch("MINUS", b_mid, M, mu, L, phi, n_max_minus, integration_limits_minus,
                   lambda b, M, S: manual_p_right_minus(b, M, S)[0])

    # Особый случай b > M (b=2.5, M=1.5)
    b_special = 2.5
    print(f"\n{'#'*70}")
    print(f"# Special case: b > M ({b_special} > {M})")
    print(f"{'#'*70}")
    inspect_branch("PLUS", b_special, M, mu, L, phi, n_max_plus, integration_limits_plus,
                   lambda b, M, S: manual_p_right_plus(b, M, S))
    inspect_branch("MINUS", b_special, M, mu, L, phi, n_max_minus, integration_limits_minus,
                   lambda b, M, S: manual_p_right_minus(b, M, S)[0])


if __name__ == "__main__":
    main()

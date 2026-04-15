"""
Отладочный скрипт 06: Прототип numba-версии fun_Omega_L_mu_int

Цель: реализовать корректную обработку двухкорневых случаев
через явный цикл по модам n с JIT-компиляцией.

Подход:
- Для каждой моды n вычисляем S_n = sqrt(mu^2 - (2*pi*n/L)^2).
- Анализируем число корней (1 или 2) в зависимости от ветви, знака b, |b|/M, F(0).
- Интегрируем по нужному числу интервалов методом средних точек.
- Умножаем на 2 для n > 0 (симметрия при phi=0).
- Сравниваем со старой версией для случаев с одним корнем.
- Показываем различие для случаев с двумя корнями.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import fun_Omega_L_mu_int
from numba import jit


@jit(nopython=True)
def _n_max_plus(M, b, L, mu, phi):
    """Адаптированная из Omega_mu_L.py n_max_plus для numba."""
    if abs(b) > M and b < 0:
        # В экстремуме p1 = sqrt(b^2-M^2), E1 = |b| = -b, E1+b = 0
        val = mu - np.sqrt(0.0**2 + (2.0 * np.pi / L * (0.0 + phi))**2)
        if val <= 0.0:
            return -1
        else:
            return int(np.floor(L * mu / (2.0 * np.pi) - phi))
    elif mu**2 <= (M + b)**2:
        return -1
    else:
        return int(np.floor(L / (2.0 * np.pi) * np.sqrt(mu**2 - (M + b)**2) - phi))


@jit(nopython=True)
def _n_max_minus(M, b, L, mu, phi):
    """Адаптированная из Omega_mu_L.py n_max_minus для numba."""
    if abs(b) > M and b > 0:
        # В экстремуме p1 = sqrt(b^2-M^2), E1 = |b| = b, E1-b = 0
        val = mu - np.sqrt(0.0**2 + (2.0 * np.pi / L * (0.0 + phi))**2)
        if val <= 0.0:
            return -1
        else:
            return int(np.floor(L * mu / (2.0 * np.pi) - phi))
    elif mu**2 <= (M - b)**2:
        return -1
    else:
        return int(np.floor(L / (2.0 * np.pi) * np.sqrt(mu**2 - (M - b)**2) - phi))


@jit(nopython=True)
def _Fpn_plus(p1, n, M, b, L, phi, mu):
    """Адаптированная Fpn_plus для скаляров (numba)."""
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 + b)**2 + (2.0 * np.pi / L * (n + phi))**2)) / (2.0 * np.pi)


@jit(nopython=True)
def _Fpn_minus(p1, n, M, b, L, phi, mu):
    """Адаптированная Fpn_minus для скаляров (numba)."""
    E1 = np.sqrt(M**2 + p1**2)
    return (mu - np.sqrt((E1 - b)**2 + (2.0 * np.pi / L * (n + phi))**2)) / (2.0 * np.pi)


@jit(nopython=True)
def _integrate_mode(F_func, n, M, b, L, phi, mu, N_h, p_left, p_right):
    """
    Интегрирует F_func(p1, n, ...) по [p_left, p_right] методом средних точек.
    F_func передается как целое (0=plus, 1=minus) для обхода ограничений numba на first-class functions.
    """
    if p_right <= p_left:
        return 0.0
    dp = (p_right - p_left) / N_h
    total = 0.0
    for i in range(N_h):
        p = p_left + (i + 0.5) * dp
        if F_func == 0:
            val = _Fpn_plus(p, n, M, b, L, phi, mu)
        else:
            val = _Fpn_minus(p, n, M, b, L, phi, mu)
        if val > 0.0:
            total += val
    return total * dp


@jit(nopython=True)
def _compute_Unp_numba(M, b, L, mu, phi, N_h):
    """PLUS ветвь с корректной обработкой 1 или 2 корней."""
    Nmax = _n_max_plus(M, b, L, mu, phi)
    if Nmax < 0:
        return 0.0

    total = 0.0
    for n in range(Nmax + 1):
        Sn = np.sqrt(mu**2 - (2.0 * np.pi * n / L)**2)
        # Определяем F(0)
        F0 = _Fpn_plus(0.0, n, M, b, L, phi, mu)

        two_roots = False
        if abs(b) > M and b < 0 and F0 < 0:
            # возможно два корня
            # p^2 = (|b| - Sn)^2 - M^2  и  (|b| + Sn)^2 - M^2
            val_left = (abs(b) - Sn)**2 - M**2
            val_right = (abs(b) + Sn)**2 - M**2
            if val_left >= 0.0 and val_right >= 0.0:
                two_roots = True
                p_left = np.sqrt(val_left)
                p_right = np.sqrt(val_right)

        if not two_roots:
            # один корень от 0 до p_right
            if b >= 0:
                pr_sq = (Sn - b)**2 - M**2
            else:
                pr_sq = (Sn + abs(b))**2 - M**2
            if pr_sq < 0.0:
                continue
            p_left = 0.0
            p_right = np.sqrt(pr_sq)

        integral = _integrate_mode(0, n, M, b, L, phi, mu, N_h, p_left, p_right)
        if n == 0:
            total += integral
        else:
            total += 2.0 * integral
    return total


@jit(nopython=True)
def _compute_Unm_numba(M, b, L, mu, phi, N_h):
    """MINUS ветвь с корректной обработкой 1 или 2 корней."""
    Nmax = _n_max_minus(M, b, L, mu, phi)
    if Nmax < 0:
        return 0.0

    total = 0.0
    for n in range(Nmax + 1):
        Sn = np.sqrt(mu**2 - (2.0 * np.pi * n / L)**2)
        F0 = _Fpn_minus(0.0, n, M, b, L, phi, mu)

        two_roots = False
        if abs(b) > M and b > 0 and F0 < 0:
            val_left = (b - Sn)**2 - M**2
            val_right = (b + Sn)**2 - M**2
            if val_left >= 0.0 and val_right >= 0.0:
                two_roots = True
                p_left = np.sqrt(val_left)
                p_right = np.sqrt(val_right)

        if not two_roots:
            if b >= 0:
                pr_sq = (Sn + b)**2 - M**2
            else:
                pr_sq = (Sn + abs(b))**2 - M**2
            if pr_sq < 0.0:
                continue
            p_left = 0.0
            p_right = np.sqrt(pr_sq)

        integral = _integrate_mode(1, n, M, b, L, phi, mu, N_h, p_left, p_right)
        if n == 0:
            total += integral
        else:
            total += 2.0 * integral
    return total


@jit(nopython=True)
def fun_Omega_L_mu_int_numba(mu, L, b, M, phi=0.0, N_h=100):
    """
    Numba-версия fun_Omega_L_mu_int с поддержкой двухкорневых случаев.
    """
    Unp = _compute_Unp_numba(M, b, L, mu, phi, N_h)
    Unm = _compute_Unm_numba(M, b, L, mu, phi, N_h)
    return -(2.0 / L) * (Unp + Unm)


def main():
    print("=" * 70)
    print("Тест numba-прототипа fun_Omega_L_mu_int")
    print("=" * 70)

    mu = 3.0
    L = 2.0
    M = 1.5
    phi = 0
    N_h = 100

    cases_one_root = [
        (0.5, "PLUS/MINUS, b=0.5, один корень"),
        (-0.5, "PLUS/MINUS, b=-0.5, один корень"),
        (2.0, "PLUS/MINUS, b=2.0, один корень"),
        (-2.0, "PLUS/MINUS, b=-2.0, один корень"),
    ]

    cases_two_roots = [
        (5.0, "PLUS/MINUS, b=5.0, два корня (баг-зона старого кода)"),
        (-5.0, "PLUS/MINUS, b=-5.0, два корня (баг-зона старого кода)"),
    ]

    # Warm-up numba
    _ = fun_Omega_L_mu_int_numba(mu, L, 0.5, M, phi, N_h)

    print("\n--- РЕГРЕССИОННЫЕ ТЕСТЫ: случаи с одним корнем ---")
    print(f"{'b':>6} | {'OLD':>16} | {'NUMBA':>16} | {'DIFF':>12} | {'TIME OLD':>10} | {'TIME NUMBA':>10}")
    print("-" * 85)
    for b, desc in cases_one_root:
        t0 = time.time()
        old = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h)
        t_old = time.time() - t0

        t0 = time.time()
        new = fun_Omega_L_mu_int_numba(mu, L, b, M, phi, N_h)
        t_new = time.time() - t0

        diff = abs(old - new)
        status = "OK" if diff < 1e-6 else "FAIL"
        print(f"{b:6.1f} | {old:16.8e} | {new:16.8e} | {diff:12.3e} | {t_old:10.4f} | {t_new:10.4f} | {status}")

    print("\n--- ТЕСТЫ НА ДВА КОРНЯ: старая версия должна отличаться ---")
    print(f"{'b':>6} | {'OLD (buggy)':>16} | {'NUMBA (fixed)':>16} | {'DIFF':>12} | {'COMMENT':>20}")
    print("-" * 75)
    for b, desc in cases_two_roots:
        old = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h)
        new = fun_Omega_L_mu_int_numba(mu, L, b, M, phi, N_h)
        diff = abs(old - new)
        comment = "старое > новое" if old > new else "старое < новое"
        print(f"{b:6.1f} | {old:16.8e} | {new:16.8e} | {diff:12.3e} | {comment:>20}")

    print("=" * 70)


if __name__ == "__main__":
    main()

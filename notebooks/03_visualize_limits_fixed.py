"""
Отладочный скрипт 03 (FIXED): Визуальная проверка исправленных пределов интегрирования

Цель: показать, что после исправления functions/Omega_mu_L.py
пределы интегрирования из КОДА (красный пунктир) совпадают
с АНАЛИТИЧЕСКИМ ВЫВОДОМ (зелёная линия) для разных параметров.

Строится сетка из 6 графиков:
- разные знаки b (+ и -)
- разные ветви (PLUS / MINUS)
- случай b > M
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import (
    integration_limits_plus,
    integration_limits_minus,
    Fpn_plus_numpy,
    Fpn_minus_numpy,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_and_plot(ax, branch, b, M, mu, L, phi, n):
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)

    if branch == 'plus':
        _, p_right_arr = integration_limits_plus(M, b, mu, L, n, phi)
        p_code = p_right_arr[n]
        p_manual = np.sqrt((S_n - b) ** 2 - M ** 2)
        p1_vals = np.linspace(0, max(p_code, p_manual, 1e-9) * 1.3, 400)
        F_vals = Fpn_plus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
    else:
        _, p_right_arr = integration_limits_minus(M, b, mu, L, n, phi)
        p_code = p_right_arr[n]
        p_manual = np.sqrt((S_n + b) ** 2 - M ** 2)
        p1_vals = np.linspace(0, max(p_code, p_manual, 1e-9) * 1.3, 400)
        F_vals = Fpn_minus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()

    # Рисуем
    ax.plot(p1_vals, F_vals, 'k-', lw=2, label=r'$F_n^{\pm}(p_1)$')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(p_code, color='red', ls='--', lw=2, label=f'CODE = {p_code:.3f}')
    ax.axvline(p_manual, color='green', ls='-', lw=2, label=f'MANUAL = {p_manual:.3f}')

    # Заливка правильной области
    mask = (p1_vals >= 0) & (p1_vals <= p_manual) & (F_vals >= 0)
    ax.fill_between(p1_vals, 0, F_vals, where=mask, color='green', alpha=0.15)

    match = "OK" if np.isclose(p_code, p_manual, atol=1e-12) else "FAIL"
    ax.set_title(f"{branch.upper()} | b={b}, M={M}, μ={mu}, L={L}, n={n}\n[{match}]", fontsize=10)
    ax.set_xlabel(r'$p_1$', fontsize=9)
    ax.set_ylabel(r'$F_n^{\pm}$', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(F_vals) * 1.1 if min(F_vals) < 0 else -0.1, top=max(F_vals) * 1.1)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    mu = 3.0
    L = 2.0
    phi = 0
    n = 0

    cases = [
        ('plus',  0.5, 1.5),
        ('minus', 0.5, 1.5),
        ('plus', -0.5, 1.5),
        ('minus', -0.5, 1.5),
        ('plus',  1.0, 1.5),
        ('minus', 1.0, 1.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    all_ok = True
    for ax, (branch, b, M) in zip(axes, cases):
        compute_and_plot(ax, branch, b, M, mu, L, phi, n)
        S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)
        if branch == 'plus':
            p_code = integration_limits_plus(M, b, mu, L, n, phi)[1][n]
            p_manual = np.sqrt((S_n - b) ** 2 - M ** 2)
        else:
            p_code = integration_limits_minus(M, b, mu, L, n, phi)[1][n]
            p_manual = np.sqrt((S_n + b) ** 2 - M ** 2)
        if not np.isclose(p_code, p_manual, atol=1e-12):
            all_ok = False

    fig.suptitle("Проверка исправленных пределов интегрирования", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, '03_limits_all_cases_fixed.png')
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")
    if all_ok:
        print(">>> Все 6 случаев: CODE limit == MANUAL limit. Исправление подтверждено.")
    else:
        print(">>> ВНИМАНИЕ: в одном из случаев есть расхождение!")


if __name__ == "__main__":
    main()

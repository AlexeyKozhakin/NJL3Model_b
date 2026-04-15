"""
Отладочный скрипт 04: Проверка случаев с ДВУМЯ корнями

Цель: показать, что при |b| > M и F(0) < 0 функция F_n^±(p1) имеет два корня,
но текущий код всегда возвращает p_left = 0, что является ошибкой.

Скрипт строит графики и:
- вычисляет оба корня аналитически
- сравнивает с p_left/p_right из кода
- визуализирует недостающую область интегрирования
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


def solve_roots_plus(b, M, mu, L, n):
    """
    Решаем (E1 + b)^2 + (2*pi*n/L)^2 = mu^2
    E1 = sqrt(M^2 + p^2)
    E1 + b = ± S_n
    где S_n = sqrt(mu^2 - (2*pi*n/L)^2)
    """
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)
    roots = []
    # E1 = S_n - b
    val = (S_n - b) ** 2 - M ** 2
    if val >= 0:
        roots.append(np.sqrt(val))
    # E1 = -S_n - b
    val2 = (-S_n - b) ** 2 - M ** 2
    if val2 >= 0:
        roots.append(np.sqrt(val2))
    return sorted(set(np.round(roots, 12)))


def solve_roots_minus(b, M, mu, L, n):
    """
    Решаем (E1 - b)^2 + (2*pi*n/L)^2 = mu^2
    E1 - b = ± S_n
    E1 = S_n + b  или  E1 = -S_n + b = b - S_n
    """
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)
    roots = []
    # E1 = S_n + b
    val = (S_n + b) ** 2 - M ** 2
    if val >= 0:
        roots.append(np.sqrt(val))
    # E1 = b - S_n
    val2 = (b - S_n) ** 2 - M ** 2
    if val2 >= 0:
        roots.append(np.sqrt(val2))
    return sorted(set(np.round(roots, 12)))


def plot_two_roots_case(ax, branch, b, M, mu, L, phi, n):
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)
    p_extremum = np.sqrt(max(b**2 - M**2, 0.0))

    if branch == 'plus':
        roots = solve_roots_plus(b, M, mu, L, n)
        p1_vals = np.linspace(0, max(roots + [p_extremum, 1e-9]) * 1.3, 600)
        F_vals = Fpn_plus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
        p_code_left, p_code_right = integration_limits_plus(M, b, mu, L, n, phi)
        p_code_left = p_code_left[n]
        p_code_right = p_code_right[n]
    else:
        roots = solve_roots_minus(b, M, mu, L, n)
        p1_vals = np.linspace(0, max(roots + [p_extremum, 1e-9]) * 1.3, 600)
        F_vals = Fpn_minus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
        p_code_left, p_code_right = integration_limits_minus(M, b, mu, L, n, phi)
        p_code_left = p_code_left[n]
        p_code_right = p_code_right[n]

    ax.plot(p1_vals, F_vals, 'k-', lw=2, label=r'$F_n^{\pm}(p_1)$')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(p_extremum, color='blue', ls=':', lw=1.5, alpha=0.7, label=f'$p_{{ext}}$={p_extremum:.2f}')

    # Корни из кода
    ax.axvline(p_code_left, color='red', ls='--', lw=2, label=f'CODE left={p_code_left:.3f}')
    ax.axvline(p_code_right, color='red', ls='--', lw=2, label=f'CODE right={p_code_right:.3f}')

    # Правильные корни
    colors = ['green', 'orange']
    for i, r in enumerate(roots):
        ax.axvline(r, color=colors[i % 2], ls='-', lw=2, label=f'TRUE root={r:.3f}')

    # Заливка: что считает код (если p_left=0)
    mask_code = (p1_vals >= p_code_left) & (p1_vals <= p_code_right) & (F_vals >= 0)
    ax.fill_between(p1_vals, 0, F_vals, where=mask_code, color='red', alpha=0.15,
                    label='Area counted by code')

    # Заливка: недостающая область (между истинным p_left и p_code_left=0)
    if len(roots) == 2:
        mask_missing = (p1_vals >= roots[0]) & (p1_vals <= roots[1]) & (F_vals >= 0)
        ax.fill_between(p1_vals, 0, F_vals, where=mask_missing, color='green', alpha=0.25,
                        hatch='//', label='MISSING area!')

    has_bug = (len(roots) == 2) and not np.isclose(p_code_left, roots[0], atol=1e-10)
    status = "BUG!" if has_bug else "OK"

    ax.set_title(f"{branch.upper()} | b={b}, M={M}, μ={mu}, n={n}\n[{status}]", fontsize=10)
    ax.set_xlabel(r'$p_1$', fontsize=9)
    ax.set_ylabel(r'$F_n^{\pm}$', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(F_vals) * 1.1 if min(F_vals) < 0 else -0.1, top=max(F_vals) * 1.1)

    return has_bug


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    mu = 3.0
    L = 2.0
    phi = 0
    n = 0

    cases = [
        ('plus', -5.0, 1.5),   # b < -M, F(0) < 0
        ('minus', 5.0, 1.5),   # b > M, F(0) < 0
        ('plus', -2.5, 1.5),   # b < -M, F(0) > 0  (один корень для контраста)
        ('minus', 2.5, 1.5),   # b > M, F(0) > 0   (один корень для контраста)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    any_bug = False
    for ax, (branch, b, M) in zip(axes, cases):
        has_bug = plot_two_roots_case(ax, branch, b, M, mu, L, phi, n)
        if has_bug:
            any_bug = True

    fig.suptitle("Проверка случаев с двумя корнями (|b| > M)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, '04_two_roots_check.png')
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")
    if any_bug:
        print(">>> ОБНАРУЖЕН БАГ: при |b|>M и F(0)<0 код возвращает p_left=0 вместо ненулевого левого корня!")
    else:
        print(">>> Во всех показанных случаях корней либо один, либо код обрабатывает оба.")


if __name__ == "__main__":
    main()

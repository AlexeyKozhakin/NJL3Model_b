"""
Отладочный скрипт 03: Визуализация ошибки в пределах интегрирования

Цель: наглядно показать, что при b>0 ветвь PLUS использует завышенный
правый предел интегрирования, а при b<0 — аналогичная ошибка в MINUS.

Скрипт строит:
- Функцию F_n^+(p1) или F_n^-(p1).
- Вертикальные линии для предела из КОДА (красный пунктир) и
  предела из АНАЛИТИЧЕСКОГО ВЫВОДА (зелёная сплошная).
- Заливку: правильная область интегрирования (зелёная) и
  лишняя область, которую неправильно включает код (красная штриховка).

Графики сохраняются в notebooks/output/
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
    Fpn_plus_numpy,
    Fpn_minus_numpy,
)

import matplotlib
matplotlib.use('Agg')  # без GUI backend
import matplotlib.pyplot as plt


def compute_p_limits_case(branch, b, M, mu, L, phi, n):
    """
    Возвращает словарь с параметрами для построения графика.
    """
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)

    if branch == 'plus':
        p_code = np.sqrt((-b - S_n) ** 2 - M ** 2)  # текущий код
        p_manual = np.sqrt((S_n - b) ** 2 - M ** 2)  # правильный предел
        p1_vals = np.linspace(0, max(p_code, p_manual) * 1.3, 500)
        F_vals = Fpn_plus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
        title = f"PLUS branch: b={b}, M={M}, μ={mu}, L={L}, n={n}"
    else:
        p_code = np.sqrt((b - S_n) ** 2 - M ** 2)  # текущий код
        p_manual = np.sqrt((S_n + b) ** 2 - M ** 2)  # правильный предел (primary)
        p1_vals = np.linspace(0, max(p_code, p_manual) * 1.3, 500)
        F_vals = Fpn_minus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
        title = f"MINUS branch: b={b}, M={M}, μ={mu}, L={L}, n={n}"

    return {
        'p1_vals': p1_vals,
        'F_vals': F_vals,
        'p_code': p_code,
        'p_manual': p_manual,
        'S_n': S_n,
        'title': title,
    }


def plot_case(data, filename):
    p1 = data['p1_vals']
    F = data['F_vals']
    p_code = data['p_code']
    p_manual = data['p_manual']

    fig, ax = plt.subplots(figsize=(8, 5))

    # Основная кривая
    ax.plot(p1, F, 'k-', lw=2, label=r'$F_n^{\pm}(p_1)$')
    ax.axhline(0, color='gray', ls='--', lw=0.8)

    # Вертикальные линии пределов
    ax.axvline(p_code, color='red', ls='--', lw=2, label=f'CODE limit = {p_code:.3f}')
    ax.axvline(p_manual, color='green', ls='-', lw=2, label=f'CORRECT limit = {p_manual:.3f}')

    # Заливка правильной области (от 0 до p_manual, где F>0)
    mask_correct = (p1 >= 0) & (p1 <= p_manual) & (F >= 0)
    ax.fill_between(p1, 0, F, where=mask_correct, color='green', alpha=0.2,
                    label='Correct integration area')

    # Заливка лишней области (от p_manual до p_code, если p_code > p_manual)
    if p_code > p_manual:
        mask_bug = (p1 >= p_manual) & (p1 <= p_code)
        ax.fill_between(p1, 0, F, where=mask_bug, color='red', alpha=0.2,
                        hatch='//', label='BUG: extra area included by code')
    elif p_code < p_manual:
        mask_missing = (p1 >= p_code) & (p1 <= p_manual) & (F >= 0)
        ax.fill_between(p1, 0, F, where=mask_missing, color='orange', alpha=0.3,
                        hatch='\\', label='BUG: missing area')

    ax.set_xlabel(r'$p_1$', fontsize=12)
    ax.set_ylabel(r'$F_n^{\pm}(p_1)$', fontsize=12)
    ax.set_title(data['title'])
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(F) * 1.1 if min(F) < 0 else -0.1, top=max(F) * 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Общие параметры
    mu = 3.0
    L = 2.0
    M = 1.5
    phi = 0
    n = 0

    # Случай 1: b>0, PLUS-ветвь (здесь найден баг)
    b1 = 0.5
    d1 = compute_p_limits_case('plus', b1, M, mu, L, phi, n)
    plot_case(d1, os.path.join(out_dir, '03_bug_plus_b_pos.png'))

    # Случай 2: b<0, MINUS-ветвь (зеркальный баг)
    b2 = -0.5
    d2 = compute_p_limits_case('minus', b2, M, mu, L, phi, n)
    plot_case(d2, os.path.join(out_dir, '03_bug_minus_b_neg.png'))

    print("\nГрафики сохранены в notebooks/output/")
    print("- 03_bug_plus_b_pos.png   : PLUS branch, b=+0.5")
    print("- 03_bug_minus_b_neg.png  : MINUS branch, b=-0.5")
    print("\nЗелёная заливка = правильная область интегрирования.")
    print("Красная штриховка = лишняя область, которую неправильно включает код.")


if __name__ == "__main__":
    main()

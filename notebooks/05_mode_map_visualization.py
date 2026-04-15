"""
Отладочный скрипт 05: Карта режимов функции F_n^±(p)

Цель: визуально продемонстрировать все возможные типы поведения
функции F_n^±(p) для различных комбинаций знака b, соотношения |b|/M
и знака F(0). Это подтверждает теоретический анализ из docs/theory.md (раздел 6.3).

Режимы:
- b > 0, PLUS: всегда монотонное убывание (1 корень)
- b < 0, PLUS, |b| <= M: монотонное убывание (1 корень)
- b < 0, PLUS, |b| > M, F(0)>0: максимум (1 корень)
- b < 0, PLUS, |b| > M, F(0)<0: максимум (2 корня)  [BUG-zone]
- b < 0, MINUS: всегда монотонное убывание (1 корень)
- b > 0, MINUS, b <= M: монотонное убывание (1 корень)
- b > 0, MINUS, b > M, F(0)>0: максимум (1 корень)
- b > 0, MINUS, b > M, F(0)<0: максимум (2 корня)  [BUG-zone]
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_mu_L import Fpn_plus_numpy, Fpn_minus_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_mode(ax, branch, b, M, mu, L, phi, n):
    S_n = np.sqrt(mu**2 - (2 * np.pi * n / L) ** 2)
    p_ext = np.sqrt(max(b**2 - M**2, 0.0))
    p_max = max(p_ext * 1.5, 5.0)

    if branch == 'plus':
        p1_vals = np.linspace(0, p_max, 400)
        F_vals = Fpn_plus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()
    else:
        p1_vals = np.linspace(0, p_max, 400)
        F_vals = Fpn_minus_numpy(p1_vals.reshape(1, -1), np.array([n]), M, b, L, phi, mu).ravel()

    F0 = F_vals[0]
    has_max = (abs(b) > M) and (
        (branch == 'plus' and b < 0) or (branch == 'minus' and b > 0)
    )
    num_roots = 2 if (has_max and F0 < 0) else (1 if (F0 >= 0 or np.max(F_vals) > 0) else 0)

    ax.plot(p1_vals, F_vals, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    if has_max and p_ext > 0:
        ax.axvline(p_ext, color='blue', ls=':', lw=1.5, alpha=0.7)

    # Заливка положительной области
    mask = (F_vals >= 0) & (p1_vals >= 0)
    ax.fill_between(p1_vals, 0, F_vals, where=mask, color='green', alpha=0.2)

    status = f"{num_roots} root{'s' if num_roots != 1 else ''}"
    if num_roots == 2:
        status += " [BUG-zone]"

    title = (
        f"{branch.upper()} | b={b:.1f}, M={M:.1f}\n"
        f"|b|/M={abs(b)/M:.2f}, F(0)={F0:+.3f}\n"
        f"{status}"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r'$p_1$', fontsize=9)
    ax.set_ylabel(r'$F_n^{\pm}$', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(F_vals) * 1.1 if min(F_vals) < 0 else -0.1, top=max(F_vals) * 1.1)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    mu = 3.0
    L = 2.0
    phi = 0
    n = 0
    M = 1.5

    cases = [
        # (row, col, branch, b, description)
        ('plus',  2.0,  "b>0, |b|>M"),
        ('plus',  0.5,  "b>0, |b|<M"),
        ('plus', -0.5,  "b<0, |b|<M"),
        ('plus', -2.0,  "b<0, |b|>M, F(0)>0"),
        ('plus', -5.0,  "b<0, |b|>M, F(0)<0"),
        ('minus', -2.0, "b<0, |b|>M"),
        ('minus', -0.5, "b<0, |b|<M"),
        ('minus',  0.5, "b>0, |b|<M"),
        ('minus',  2.0, "b>0, |b|>M, F(0)>0"),
        ('minus',  5.0, "b>0, |b|>M, F(0)<0"),
    ]

    # Сделаем 2 ряда по 5 графиков
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (branch, b, desc) in zip(axes.flatten(), cases):
        plot_mode(ax, branch, b, M, mu, L, phi, n)

    fig.suptitle(
        f"Карта режимов F_n^{'{'}\\pm{'}'}(p_1)$ при μ={mu}, L={L}, M={M}, n={n}\n"
        r"Синяя точка — $p_{ext}$ (только при $|b|>M$). Зелёная заливка — положительная область.",
        fontsize=13
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_path = os.path.join(out_dir, '05_mode_map.png')
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")
    print("\nИтоговая классификация по theory.md, раздел 6.3:")
    print("- Столбцы 1,2,3,4,7,8: один корень (код работает корректно).")
    print("- Столбцы 5, 10: ДВА корня (код дает p_left=0, это баг).")


if __name__ == "__main__":
    main()

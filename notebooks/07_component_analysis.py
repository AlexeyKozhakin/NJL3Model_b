"""
Отладочный скрипт 07: Анализ отдельных вкладов в Omega_ren

Цель:
1. При фиксированном M и b ∈ [0, 5] вычислить каждый вклад:
   - Delta U_phys(M, b)
   - Omega_L_phys(M, b)
   - Omega_mu_L_phys(M, b)
2. При фиксированном b и M ∈ [0, 10] вычислить те же вклады.
3. Проверить сходимость Omega_mu_L по N_h.
4. Измерить время расчёта.

Графики сохраняются в notebooks/output/
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.Omega_L import fun_Omega_L
from functions.dU import fun_dU_phys
from functions.Omega_mu_L import fun_Omega_L_mu_int

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_components_at_fixed_M(M, b_vals, L, mu, phi, N_h_p1, N_h_p2, N_h_phi, N_h_mu):
    """
    Вычисляет все физические вклады для массива b_vals при фиксированном M.
    """
    Omega_L_phys_vals = np.zeros(len(b_vals))
    dU_phys_vals = np.zeros(len(b_vals))
    Omega_mu_L_phys_vals = np.zeros(len(b_vals))

    # Omega_L и dU векторизованы по b_vals
    Omega_L_full = fun_Omega_L(np.array([L]), b_vals, np.array([M]), N_h_p1=N_h_p1, N_h_p2=N_h_p2).squeeze()
    Omega_L_M0   = fun_Omega_L(np.array([L]), b_vals, np.array([0.0]), N_h_p1=N_h_p1, N_h_p2=N_h_p2).squeeze()
    Omega_L_b0   = fun_Omega_L(np.array([L]), np.array([0.0]), np.array([0.0]), N_h_p1=N_h_p1, N_h_p2=N_h_p2).squeeze()
    Omega_L_phys_vals = Omega_L_full - Omega_L_M0 + Omega_L_b0

    dU_full = fun_dU_phys(b_vals, np.array([M]), N_h_p=N_h_phi, N_h_phi=N_h_phi).squeeze()
    dU_M0   = fun_dU_phys(b_vals, np.array([0.0]), N_h_p=N_h_phi, N_h_phi=N_h_phi).squeeze()
    dU_b0   = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h_p=N_h_phi, N_h_phi=N_h_phi).squeeze()
    dU_phys_vals = dU_full - dU_M0 + dU_b0

    # Omega_mu_L приходится считать в цикле (скалярная функция)
    t0 = time.time()
    for i, b in enumerate(b_vals):
        o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h_mu)
        o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi, N_h_mu)
        o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi, N_h_mu)
        Omega_mu_L_phys_vals[i] = o_mu_M - o_mu_M0 + o_mu_b0
    t_mu = time.time() - t0

    return Omega_L_phys_vals, dU_phys_vals, Omega_mu_L_phys_vals, t_mu


def compute_components_at_fixed_b(b, M_vals, L, mu, phi, N_h_p1, N_h_p2, N_h_phi, N_h_mu):
    """
    Вычисляет все физические вклады для массива M_vals при фиксированном b.
    """
    n_b = len(M_vals)
    Omega_L_phys_vals = np.zeros(n_b)
    dU_phys_vals = np.zeros(n_b)
    Omega_mu_L_phys_vals = np.zeros(n_b)

    for i, M in enumerate(M_vals):
        # Omega_L_phys
        oL_M  = fun_Omega_L(L, [b], M, N_h_p1=N_h_p1, N_h_p2=N_h_p2)[0]
        oL_M0 = fun_Omega_L(L, [b], 0.0, N_h_p1=N_h_p1, N_h_p2=N_h_p2)[0]
        oL_b0 = fun_Omega_L(L, [0.0], 0.0, N_h_p1=N_h_p1, N_h_p2=N_h_p2)[0]
        Omega_L_phys_vals[i] = (oL_M - oL_M0 + oL_b0).item()

        # dU_phys
        dU_M  = fun_dU_phys(np.array([b]), np.array([M]), N_h_p=N_h_phi, N_h_phi=N_h_phi).item()
        dU_M0 = fun_dU_phys(np.array([b]), np.array([0.0]), N_h_p=N_h_phi, N_h_phi=N_h_phi).item()
        dU_b0 = fun_dU_phys(np.array([0.0]), np.array([0.0]), N_h_p=N_h_phi, N_h_phi=N_h_phi).item()
        dU_phys_vals[i] = float(dU_M - dU_M0 + dU_b0)

        # Omega_mu_L_phys
        o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h_mu)
        o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi, N_h_mu)
        o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi, N_h_mu)
        Omega_mu_L_phys_vals[i] = o_mu_M - o_mu_M0 + o_mu_b0

    return Omega_L_phys_vals, dU_phys_vals, Omega_mu_L_phys_vals


def convergence_omega_mu_L(mu, L, b, M, phi, N_h_list):
    """Сходимость Omega_mu_L_phys по N_h."""
    vals = []
    times = []
    for N_h in N_h_list:
        t0 = time.time()
        o_mu_M  = fun_Omega_L_mu_int(mu, L, b, M, phi, N_h)
        o_mu_M0 = fun_Omega_L_mu_int(mu, L, b, 0.0, phi, N_h)
        o_mu_b0 = fun_Omega_L_mu_int(mu, L, 0.0, 0.0, phi, N_h)
        val = o_mu_M - o_mu_M0 + o_mu_b0
        t = time.time() - t0
        vals.append(val)
        times.append(t)
    return np.array(vals), np.array(times)


def plot_fixed_M(b_vals, Omega_L, dU, Omega_mu_L, M_fix, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(b_vals, Omega_L, 'b-', lw=2, label=r'$\Omega_{L,phys}$')
    ax.set_xlabel(r'$b$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Omega_L_phys vs b (M={M_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(b_vals, dU, 'g-', lw=2, label=r'$\Delta U_{phys}$')
    ax.set_xlabel(r'$b$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'dU_phys vs b (M={M_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(b_vals, Omega_mu_L, 'r-', lw=2, label=r'$\Omega_{\mu L,phys}$')
    ax.set_xlabel(r'$b$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Omega_mu_L_phys vs b (M={M_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(b_vals, Omega_L + dU + Omega_mu_L, 'k-', lw=2, label='Sum of 3 terms')
    ax.set_xlabel(r'$b$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Sum of contributions vs b (M={M_fix}, no M^2/(2g))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_fixed_b(M_vals, Omega_L, dU, Omega_mu_L, b_fix, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(M_vals, Omega_L, 'b-', lw=2, label=r'$\Omega_{L,phys}$')
    ax.set_xlabel(r'$M$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Omega_L_phys vs M (b={b_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(M_vals, dU, 'g-', lw=2, label=r'$\Delta U_{phys}$')
    ax.set_xlabel(r'$M$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'dU_phys vs M (b={b_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(M_vals, Omega_mu_L, 'r-', lw=2, label=r'$\Omega_{\mu L,phys}$')
    ax.set_xlabel(r'$M$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Omega_mu_L_phys vs M (b={b_fix})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(M_vals, Omega_L + dU + Omega_mu_L, 'k-', lw=2, label='Sum of 3 terms')
    ax.set_xlabel(r'$M$', fontsize=11)
    ax.set_ylabel(r'Value', fontsize=11)
    ax.set_title(f'Sum of contributions vs M (b={b_fix}, no M^2/(2g))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_convergence(N_h_list, vals, times, mu, L, b, M, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(N_h_list, vals, 'bo-', lw=2, markersize=6)
    ax.set_xlabel(r'$N_h$', fontsize=11)
    ax.set_ylabel(r'$\Omega_{\mu L,phys}$', fontsize=11)
    ax.set_title(f'Convergence of Omega_mu_L_phys (b={b}, M={M})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    ax.plot(N_h_list, times, 'rs-', lw=2, markersize=6)
    ax.set_xlabel(r'$N_h$', fontsize=11)
    ax.set_ylabel(r'Time (s)', fontsize=11)
    ax.set_title(f'Compute time vs N_h (b={b}, M={M})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # Общие параметры
    mu = 3.0
    L = 2.0
    phi = 0
    N_h_p1 = 100
    N_h_p2 = 100
    N_h_phi = 100
    N_h_mu = 100

    # ==================== Фиксированное M, меняем b ====================
    M_fix = 1.5
    b_vals = np.linspace(0, 5, 51)
    print(f"\n[1/3] Расчет при фиксированном M={M_fix}, b=[0, 5] ...")
    t0 = time.time()
    oL_b, dU_b, omu_b, t_mu = compute_components_at_fixed_M(
        M_fix, b_vals, L, mu, phi, N_h_p1, N_h_p2, N_h_phi, N_h_mu
    )
    t_total = time.time() - t0
    print(f"  Omega_mu_L: {t_mu:.3f}s, Total: {t_total:.3f}s")

    plot_fixed_M(b_vals, oL_b, dU_b, omu_b, M_fix,
                 os.path.join(out_dir, '07_fixed_M_b_scan.png'))

    # ==================== Фиксированное b, меняем M ====================
    b_fix = 2.0
    M_vals = np.linspace(0, 10, 101)
    print(f"\n[2/3] Расчет при фиксированном b={b_fix}, M=[0, 10] ...")
    t0 = time.time()
    oL_M, dU_M, omu_M = compute_components_at_fixed_b(
        b_fix, M_vals, L, mu, phi, N_h_p1, N_h_p2, N_h_phi, N_h_mu
    )
    t_total = time.time() - t0
    print(f"  Total: {t_total:.3f}s")

    plot_fixed_b(M_vals, oL_M, dU_M, omu_M, b_fix,
                 os.path.join(out_dir, '07_fixed_b_M_scan.png'))

    # ==================== Сходимость Omega_mu_L по N_h ====================
    b_conv = 2.0
    M_conv = 1.5
    N_h_list = [10, 20, 50, 100, 200, 500, 1000]
    print(f"\n[3/3] Сходимость Omega_mu_L_phys по N_h (b={b_conv}, M={M_conv}) ...")
    vals_conv, times_conv = convergence_omega_mu_L(mu, L, b_conv, M_conv, phi, N_h_list)
    for nh, val, t in zip(N_h_list, vals_conv, times_conv):
        print(f"  N_h={nh:4d} -> {val:16.8e} | {t:.4f}s")

    plot_convergence(N_h_list, vals_conv, times_conv, mu, L, b_conv, M_conv,
                     os.path.join(out_dir, '07_convergence_Nh.png'))

    print("\n>>> Все графики сохранены в notebooks/output/")
    print("    - 07_fixed_M_b_scan.png")
    print("    - 07_fixed_b_M_scan.png")
    print("    - 07_convergence_Nh.png")


if __name__ == "__main__":
    main()

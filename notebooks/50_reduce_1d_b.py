"""
Статья 2, шаг 1: сведение Omega(M, b) к одномерным интегралам по p1 (phi = 0).

A_± = E1 ± b,  E1 = sqrt(M² + p1²),  A0_± = p1 ± b  (то же при M = 0).

ΔU(M, b) = -(1/pi²) ∫_0^∞ dp1 [ M²/2 + M² ln p1 - ½ Σ± A±² ln|A±| + ½ Σ± A0±² ln|A0±| ]
    (внутренний интеграл по p2 взят аналитически; ln Λ сокращается с контрчленом -M²/p).

Omega_L(M, b) = (2/(pi² L)) ∫_0^∞ dp1 Σ± S(|A±|),
    S(a) = Σ_k (a/k) K1(k L a) = -∫_0^∞ ln(1 - e^{-L sqrt(q² + a²)}) dq.

Omega_muL(M, b) = -(1/(pi L)) Σ_n C_n Σ± ∫_{F>0} (mu - sqrt(A±² + k_n²)) dp1
    (одномерный интеграл с явными пределами, docs/theory.md §6.3).

Проверка: против прямого 2D-интегрирования (functions/*_fast) и при b = 0 против
замкнутых формул njl3.b0.

Запуск:  python notebooks/50_reduce_1d_b.py
"""

import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import k1e

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.dU_fast import dU_scalar
from functions.Omega_L_fast import Omega_L_scalar
from functions.Omega_mu_L import fun_Omega_L_mu_int
from njl3.b0 import formulas as B0

warnings.simplefilter('ignore', IntegrationWarning)
PI = math.pi


def _xlogx2(A):
    """A² ln|A| (0 при A = 0)."""
    return 0.0 if A == 0.0 else A * A * math.log(abs(A))


def dU_1d(M, b):
    def h(p):
        E = math.sqrt(M * M + p * p)
        lp = math.log(p) if p > 0 else 0.0
        return (M * M / 2 + M * M * lp
                - 0.5 * (_xlogx2(E + b) + _xlogx2(E - b))
                + 0.5 * (_xlogx2(p + b) + _xlogx2(p - b)))
    pts = sorted({x for x in (b, math.sqrt(max(b * b - M * M, 0.0))) if x > 0})
    edges = [0.0] + pts + [max(pts + [1.0]) * 4]
    total = sum(quad(h, a, c, limit=200, epsabs=1e-14, epsrel=1e-13)[0] for a, c in zip(edges[:-1], edges[1:]))
    total += quad(h, edges[-1], np.inf, limit=200, epsabs=1e-14, epsrel=1e-13)[0]
    return -total / PI**2


def S_L(a, L):
    """S(a) = sum_k (a/k) K1(kLa); при малом La — квадратурой."""
    x = L * a
    if x > 0.2:
        s, k = 0.0, 1
        while True:
            t = (a / k) * k1e(k * x) * math.exp(-k * x)
            s += t
            if t < 1e-18 * max(s, 1e-300) or k > 2000:
                return s
            k += 1
    f = lambda q: -math.log(-math.expm1(-L * math.sqrt(q * q + a * a)))
    v1 = quad(f, 0, 1.0, limit=200, epsabs=1e-15, epsrel=1e-13, points=[a] if 0 < a < 1 else None)[0]
    v2 = quad(f, 1.0, np.inf, limit=200, epsabs=1e-15, epsrel=1e-13)[0]
    return v1 + v2


def omega_L_1d(M, b, L):
    f = lambda p: S_L(abs(math.sqrt(M * M + p * p) + b), L) + S_L(abs(math.sqrt(M * M + p * p) - b), L)
    pts = [math.sqrt(b * b - M * M)] if b > M else []
    edges = [0.0] + pts + [max(pts + [1.0]) * 3]
    total = sum(quad(f, a, c, limit=200, epsabs=1e-14, epsrel=1e-12)[0] for a, c in zip(edges[:-1], edges[1:]))
    total += quad(f, edges[-1], np.inf, limit=200, epsabs=1e-14, epsrel=1e-12)[0]
    return 2.0 / (PI**2 * L) * total


def omega_muL_1d(mu, L, b, M):
    """b ≥ 0. Интеграл по p1 ≥ 0 там, где F > 0 (явные пределы)."""
    total = 0.0
    n_max = int(math.floor(L * mu / (2 * PI)))
    for n in range(n_max + 1):
        k = 2 * PI * n / L
        S = math.sqrt(mu * mu - k * k)
        C = 1.0 if n == 0 else 2.0
        # плюс-ветвь: E1 + b ≤ S  ->  p ∈ [0, sqrt((S-b)² - M²)]
        if S - b > M:
            pr = math.sqrt((S - b)**2 - M * M)
            f = lambda p: mu - math.sqrt((math.sqrt(M * M + p * p) + b)**2 + k * k)
            total += C * quad(f, 0, pr, limit=200, epsabs=1e-14, epsrel=1e-13)[0]
        # минус-ветвь: |E1 - b| ≤ S  ->  E1 ∈ [max(M, b - S), b + S]
        lo_E, hi_E = max(M, b - S), b + S
        if hi_E > lo_E:
            pl = math.sqrt(max(lo_E**2 - M * M, 0.0))
            pr = math.sqrt(hi_E**2 - M * M)
            f = lambda p: mu - math.sqrt((math.sqrt(M * M + p * p) - b)**2 + k * k)
            kink = [math.sqrt(b * b - M * M)] if (n == 0 and b > M and pl < math.sqrt(b * b - M * M) < pr) else None
            total += C * quad(f, pl, pr, points=kink, limit=200, epsabs=1e-14, epsrel=1e-13)[0]
    return -2.0 / L * total / (2 * PI)


def main():
    print("1) b = 0 против замкнутых формул")
    for M, L, mu in [(0.3, 1.0, 3.2), (1.5, 0.5, 2.0), (3.0, 2.0, 5.0)]:
        print(f"   M={M}, L={L}, mu={mu}:  ΔU {dU_1d(M, 0.0) - B0.dU(M):+.1e}   "
              f"Ω_L {omega_L_1d(M, 0.0, L) - B0.omega_L(M, L):+.1e}   "
              f"Ω_μL {omega_muL_1d(mu, L, 0.0, M) - B0.omega_muL(M, mu, L):+.1e}")

    print("\n2) b > 0 против прямого 2D-интегрирования (N2d = 800, N1d = 40000)")
    print(f"   {'M':>4} {'b':>4} {'L':>4} {'mu':>4} | {'ΔU 1D':>12} {'Δ':>9} | {'Ω_L 1D':>12} {'Δ':>9} | {'Ω_μL 1D':>12} {'Δ':>9}")
    t1 = t2 = 0.0
    for M, b, L, mu in [(0.5, 0.3, 1.0, 3.2), (0.8, 2.5, 1.0, 3.0), (0.2, 3.0, 2.0, 3.5),
                        (2.0, 1.0, 0.5, 4.0), (0.05, 4.0, 3.5, 4.2), (1.0, 1.0, 1.5, 2.0)]:
        t = time.perf_counter()
        u, l, m = dU_1d(M, b), omega_L_1d(M, b, L), omega_muL_1d(mu, L, b, M)
        t1 += time.perf_counter() - t
        t = time.perf_counter()
        u2 = dU_scalar(b, M, 800, 800)
        l2 = Omega_L_scalar(L, b, M, 0.0, 800, 800)
        m2 = fun_Omega_L_mu_int(mu, L, b, M, 0, 40000)
        t2 += time.perf_counter() - t
        print(f"   {M:4.2f} {b:4.1f} {L:4.1f} {mu:4.1f} | {u:12.8f} {u - u2:+9.1e} | {l:12.8f} {l - l2:+9.1e} | "
              f"{m:12.8f} {m - m2:+9.1e}")
    print(f"\n   время: 1D (quad) {t1:.2f} с, 2D-код {t2:.2f} с (6 точек)")


if __name__ == '__main__':
    main()

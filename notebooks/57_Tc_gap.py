"""
Критическая температура T_c(mu, L) из функции щели при M → 0 (быстро, без вычитаний).

При T > 0 функция щели G = (1/M) ∂Ω/∂M имеет конечный предел при M → 0:
    G(0, b; T) = 1/g + b/2 + G_L(0, b) + G_μL(0, b; T),
    G_μL(0, b; T) = (1/(πL)) Σ_n C_n ∫_0^∞ dp/p Σ± [f(ε± − mu) + f(ε± + mu)] A±/ε±,
    A± = p ± b, ε± = sqrt(A±² + k_n²), f(x) = 1/(e^{x/T} + 1).
1/p сокращается между ветвями ± (при T = 0 — нет, если b = S_n: отсюда ln M).
M = 0 неустойчиво, если min_b G(0, b; T) < 0; T_c — где этот минимум проходит через 0
(порог переход 2-го рода; ниже него щель заведомо есть).

Сравнение: M*(T = 0) из find_minimum и асимптотика T_c,n = (e^γ/π) M*_n,
M*_n = exp(−A_n/c_n − 1/2) — как в модели Гросса–Невё (1+1), к которой задача сводится у порога.

Запуск:  python notebooks/57_Tc_gap.py
"""

import importlib.util
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full.minimize import channel_asymptotics, find_minimum, thresholds

_spec = importlib.util.spec_from_file_location(
    'gap55', os.path.join(os.path.dirname(__file__), '55_gap_log_check.py'))
gap55 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gap55)

PI = math.pi
G = -1.0
GL_X, GL_W = np.polynomial.legendre.leggauss(16)
BCS = math.exp(np.euler_gamma) / PI                      # 0.5669...


def _fermi(x, T):
    return 0.5 * (1.0 - np.tanh(x / (2.0 * T)))


def G_muL_T(b, mu, L, T):
    tot = 0.0
    n_max = int((mu + 40 * T) * L / (2 * PI))
    for n in range(n_max + 1):
        k, C = 2 * PI * n / L, (1 if n == 0 else 2)
        S = math.sqrt(max(mu * mu - k * k, 0.0))
        pmax = b + math.sqrt(max((mu + 40 * T)**2 - k * k, 0.0)) + 40 * T
        brk = {0.0, pmax, b} | {q for q in (S - b, b - S, b + S) if S > 0 and 0 < q < pmax}
        pts = set(brk)
        for q in brk:
            for d in 0.25 * 2.0 ** np.arange(0, 60):     # геометрически: ~1/x на многих порядках
                if d * T > pmax:
                    break
                for s in (-1, 1):
                    if 0 < q + s * d * T < pmax:
                        pts.add(q + s * d * T)
        pts = np.array(sorted(pts))
        a, c = pts[:-1, None], pts[1:, None]
        p = (0.5 * (c - a) * GL_X + 0.5 * (a + c)).ravel()
        w = (0.5 * (c - a) * GL_W).ravel()
        f = 0.0
        for sg in (+1.0, -1.0):
            A = p + sg * b
            eps = np.sqrt(A * A + k * k)
            r = np.where(eps > 0, A / np.where(eps > 0, eps, 1.0), 0.0)
            f = f + (_fermi(eps - mu, T) + _fermi(eps + mu, T)) * r
        tot += C * float(np.dot(w, f / p))
    return tot / (PI * L)


def G0(b, mu, L, T):
    return 1.0 / G + b / 2.0 + gap55.G_L(0.0, b, L) + G_muL_T(b, mu, L, T)


def g_min(mu, L, T, ret_b=False):
    bs = np.concatenate([np.linspace(0.0, mu + 1.0, 121)]
                        + [S + T * np.linspace(-8, 8, 33) for _, S, _ in thresholds(mu, L)])
    bs = np.unique(bs[bs >= 0])
    v = np.array([G0(b, mu, L, T) for b in bs])
    i = int(np.argmin(v))
    lo, hi = max(bs[max(i - 1, 0)], 0.0), bs[min(i + 1, len(bs) - 1)]
    r = minimize_scalar(lambda b: G0(b, mu, L, T), bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-4 * T})
    return (r.fun, r.x) if ret_b else r.fun


def Tc(mu, L, M0):
    T_lo, T_hi = 0.2 * M0, 1.5 * M0
    while g_min(mu, L, T_lo) >= 0:
        T_lo /= 2
    while g_min(mu, L, T_hi) < 0:
        T_hi *= 2
    while T_hi / T_lo > 1 + 1e-5:
        Tm = math.sqrt(T_lo * T_hi)
        if g_min(mu, L, Tm) < 0:
            T_lo = Tm
        else:
            T_hi = Tm
    return math.sqrt(T_lo * T_hi)


def run_point(args):
    mu, L = args
    t0 = time.time()
    r0 = find_minimum(mu, L, G)
    tc = Tc(mu, L, r0.M_star)
    _, bc = g_min(mu, L, tc, ret_b=True)
    n, S, C = min(thresholds(mu, L), key=lambda q: abs(q[1] - bc))
    a = channel_asymptotics(S, C, mu, L, G)
    M_asy = math.exp(-a['A'] / a['c'] - 0.5)
    return dict(mu=mu, L=L, phase=r0.phase, M0=r0.M_star, b0=r0.b_star, Tc=tc, bc=bc, n=n, S=S,
                M_asy=M_asy, sec=time.time() - t0)


def check():
    """Контроль против свёртки из 56 (μ=5, L=3, b=S_1): там Ω/M² при M = 0.05T ≈ G/2."""
    mu, L = 5.0, 3.0
    S = thresholds(mu, L)[1][1]
    for T, a56 in ((1e-2, 0.02932), (1e-4, -0.41432)):
        print(f"   контроль: T={T:g}, b=S_1: G/2 = {G0(S, mu, L, T) / 2:+.5f}; свёртка (56): {a56:+.5f}")
    b = S - 0.3                                           # вдали от порога: T → 0 к G(M → 0) из 55
    print(f"   контроль: b=S_1−0.3: G(0; T=1e-4) = {G0(b, mu, L, 1e-4):+.8f}, "
          f"G(M=1e-8; T=0) = {gap55.G_total(1e-8, b, mu, L, G):+.8f}")


POINTS = [(4.291, 0.939), (5.469, 1.107), (5.351, 2.785), (6.058, 3.456), (5.233, 6.308), (6.647, 5.134)]


def main():
    check()
    with Pool(min(6, os.cpu_count() or 1)) as pool:
        res = pool.map(run_point, POINTS)
    print(f"\n(e^γ/π = {BCS:.4f})")
    print(" μ      L      фаза  M*(T=0)     T_c         T_c/M*   (b(T_c)−S_n)/T_c   M*_асимпт/M*   [с]")
    for r in res:
        print(f" {r['mu']:.3f}  {r['L']:.3f}  {r['phase']:4s}  {r['M0']:.4e}  {r['Tc']:.4e}  {r['Tc'] / r['M0']:.4f}"
              f"   {(r['bc'] - r['S']) / r['Tc']:+.4f} (n={r['n']})     {r['M_asy'] / r['M0']:.4f}      {r['sec']:.0f}")


if __name__ == '__main__':
    main()

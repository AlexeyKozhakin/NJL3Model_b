"""
Задача B4 (статья 1): оценка точности реализации при b = 0 (g = -1, phi = 0).

Эталон — те же формулы в mpmath с 50 значащими цифрами (прямые, без
«устойчивых» преобразований: при 50 цифрах сокращение логарифмов безопасно).

Проверки:
    P1. G(M), G(0), Omega(M) - Omega(0): double против mpmath на случайной выборке,
        включая M -> 0, окрестности порогов мод, M > mu, малые и большие L.
    P2. Корни G (isolate_roots_b0) против mp.findroot.
    P3. Линии: mu_s(L), mu_VT(L) и переходы 1-го рода из трассировки против
        решения тех же уравнений в mpmath.

Вывод: таблицы в stdout (для заметки и статьи).
Запуск:  python notebooks/47_accuracy_b0.py [--quick]
"""

import json
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import analytic_b0 as an
from functions import phase_lines_b0 as pl

mp.mp.dps = 50
G = -1.0
QUICK = '--quick' in sys.argv
RNG = np.random.default_rng(12345)
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


# ==============================================================================
# Эталон mpmath
# ==============================================================================

def mp_modes(mu, L):
    mu, L = mp.mpf(mu), mp.mpf(L)
    nmax = int(mp.floor(L * mu / (2 * mp.pi)))
    return [(2 * mp.pi * n / L, 1 if n == 0 else 2) for n in range(nmax + 1)]


def mp_G(M, mu, L, g=G):
    M, mu, L = mp.mpf(M), mp.mpf(mu), mp.mpf(L)
    s = mp.log(1 - mp.exp(-L * M))
    for k, C in mp_modes(mu, L):
        m = mp.sqrt(M**2 + k**2)
        if m < mu:
            s += C * mp.acosh(mu / m)
    return 1 / mp.mpf(g) + M / mp.pi + 2 / (mp.pi * L) * s


def mp_G0(mu, L, g=G):
    mu, L = mp.mpf(mu), mp.mpf(L)
    s = mp.log(2 * mu * L)
    for k, C in mp_modes(mu, L)[1:]:
        s += C * mp.acosh(mu / k)
    return 1 / mp.mpf(g) + 2 / (mp.pi * L) * s


def mp_I(m, mu):
    if m >= mu:
        return mp.mpf(0)
    P = mp.sqrt(mu**2 - m**2)
    if m == 0:
        return mu * P / 2
    return mu * P / 2 - m**2 / 2 * mp.log((mu + P) / m)


def mp_omega_rel(M, mu, L, g=G):
    M, mu, L = mp.mpf(M), mp.mpf(mu), mp.mpf(L)
    x = mp.exp(-L * M)
    OL = lambda MM, xx: 2 / (mp.pi * L**3) * (L * MM * mp.polylog(2, xx) + mp.polylog(3, xx))
    dL = OL(M, x) - OL(0, mp.mpf(1))
    OmuL = lambda MM: -2 / (mp.pi * L) * mp.fsum(C * mp_I(mp.sqrt(MM**2 + k**2), mu) for k, C in mp_modes(mu, L))
    return M**2 / (2 * g) + M**3 / (3 * mp.pi) + dL + OmuL(M) - OmuL(0)


# ==============================================================================
# P1
# ==============================================================================

def sample_points(n):
    pts = []
    for _ in range(n):
        mu = float(RNG.uniform(0.1, 7.0))
        L = float(np.exp(RNG.uniform(np.log(0.05), np.log(10.0))))
        kind = RNG.integers(4)
        if kind == 0:                                      # M -> 0
            M = float(np.exp(RNG.uniform(np.log(1e-12), np.log(1e-2))))
        elif kind == 1:                                    # общий случай 0 < M < mu
            M = float(RNG.uniform(0.0, mu))
        elif kind == 2:                                    # около порога моды
            S = an.thresholds(mu, L)
            s = S[int(RNG.integers(len(S)))]
            M = s * (1.0 + float(RNG.choice([-1, 1])) * 10.0 ** RNG.uniform(-12, -2))
        else:                                              # M > mu
            M = float(RNG.uniform(mu, mu + 5.0))
        if M > 0:
            pts.append((M, mu, L, int(kind)))
    return pts


def p1():
    n = 300 if QUICK else 3000
    pts = sample_points(n)
    names = ['M → 0', '0 < M < μ', 'у порога', 'M > μ']
    errG = {k: [] for k in range(4)}
    errO = {k: [] for k in range(4)}
    errO_closed = {k: [] for k in range(4)}
    errO_abs = {k: [] for k in range(4)}
    for M, mu, L, kind in pts:
        gr = mp_G(M, mu, L)
        errG[kind].append(abs(an.gap_G(M, mu, L, G) - float(gr)))
        orf = mp_omega_rel(M, mu, L)
        scale = max(abs(float(orf)), 1e-300)
        errO[kind].append(abs(an.omega_rel_stable(M, mu, L, G) - float(orf)) / scale)
        errO_closed[kind].append(abs(an.omega_ren_rel(M, mu, L, G) - float(orf)) / scale)
        errO_abs[kind].append(abs(an.omega_rel_stable(M, mu, L, G) - float(orf)))
    g0 = [abs(an.gap_G0(mu, L, G) - float(mp_G0(mu, L))) for _, mu, L, _ in pts]

    print(f"P1. double против mpmath (50 цифр), {len(pts)} точек")
    print(f"    {'область':12s} | {'N':>5s} | {'|ΔG| max':>9s} | {'|Δ(Ω−Ω0)| устойч.':>18s} | "
          f"{'отн., устойч.':>13s} | {'отн., замкн.':>12s}")
    for k in range(4):
        if not errG[k]:
            continue
        print(f"    {names[k]:12s} | {len(errG[k]):5d} | {max(errG[k]):9.1e} | {max(errO_abs[k]):18.1e} | "
              f"{max(errO[k]):13.1e} | {max(errO_closed[k]):12.1e}")
    print(f"    G(0): max |ΔG0| = {max(g0):.1e}\n")


# ==============================================================================
# P2
# ==============================================================================

def p2():
    n = 60 if QUICK else 400
    worst_abs, worst_rel, total = 0.0, 0.0, 0
    for _ in range(n):
        mu = float(RNG.uniform(0.5, 7.0))
        L = float(np.exp(RNG.uniform(np.log(0.1), np.log(10.0))))
        for r, kind in an.isolate_roots_b0(mu, L, G):
            if kind == 'touch':
                continue
            try:
                rr = mp.findroot(lambda x: mp_G(x, mu, L), mp.mpf(r))
            except Exception:
                continue
            e = abs(r - float(rr))
            worst_abs = max(worst_abs, e)
            worst_rel = max(worst_rel, e / float(rr))
            total += 1
    print(f"P2. Корни G: {total} корней в {n} точках (μ, L)")
    print(f"    max |ΔM*| = {worst_abs:.1e},  max |ΔM*|/M* = {worst_rel:.1e}\n")


# ==============================================================================
# P3
# ==============================================================================

def mp_mu_s(L):
    f = lambda mu: mp_G0(mu, L)
    return mp.findroot(f, mp.mpf(pl.mu_stability(L, G)))


def mp_mu_VT(L):
    Mv = mp.findroot(lambda M: mp_G(M, 0.0 + 1e-30, L), mp.mpf(pl.M_vacuum(L, G)))
    f = lambda mu: mp_omega_rel(Mv, mu, L)
    return mp.findroot(f, mp.mpf(pl.mu_VT(L, G)))


def mp_first_order(L, mu0, M_left, M_right):
    """Переход 1-го рода: Omega_rel(M_A(mu)) = Omega_rel(M_B(mu)); M = 0 — фаза T."""
    def branch(Mguess, mu):
        """Корень своей ветви: только внутри вилки со сменой знака G вокруг Mguess
        (Ньютон без вилки может перескочить на соседнюю ветвь)."""
        if Mguess == 0.0:
            return mp.mpf(0)
        for w in ('1e-6', '1e-4', '1e-3'):
            a, b = mp.mpf(Mguess) * (1 - mp.mpf(w)), mp.mpf(Mguess) * (1 + mp.mpf(w))
            if mp_G(a, mu, L) * mp_G(b, mu, L) < 0:
                return mp.findroot(lambda x: mp_G(x, mu, L), (a, b), solver='anderson')
        raise ValueError('нет смены знака G около ветви')

    def F(mu):
        A, B = branch(M_left, mu), branch(M_right, mu)
        oa = mp_omega_rel(A, mu, L) if A != 0 else mp.mpf(0)
        ob = mp_omega_rel(B, mu, L) if B != 0 else mp.mpf(0)
        return oa - ob
    for w in ('1e-10', '1e-8'):
        a, b = mp.mpf(mu0) - mp.mpf(w), mp.mpf(mu0) + mp.mpf(w)
        if F(a) * F(b) < 0:
            return mp.findroot(F, (a, b), solver='anderson')
    raise ValueError('нет смены знака F около перехода')


def p3():
    Ls = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]
    if QUICK:
        Ls = Ls[::3]
    ds = [abs(pl.mu_stability(L, G) - float(mp_mu_s(L))) for L in Ls]
    print(f"P3. Линии против mpmath")
    print(f"    mu_s(L), {len(Ls)} значений L:  max |Δμ| = {max(ds):.1e}")
    dv = []
    for L in [0.2, 0.3, 0.5, 0.8, 1.0, 1.1]:
        dv.append(abs(pl.mu_VT(L, G) - float(mp_mu_VT(L))))
    print(f"    mu_VT(L), {len(dv)} значений L:   max |Δμ| = {max(dv):.1e}")

    path = os.path.join(OUT_DIR, '46_phase_lines.json')
    if not os.path.exists(path):
        print("    (нет 46_phase_lines.json — переходы 1-го рода пропущены)")
        return
    with open(path) as f:
        lines = json.load(f)['lines']
    res = {}
    for key, pts in lines.items():
        sel = [p for p in pts if p['order'] == 1]
        if not sel:
            continue
        for p in sel[:: max(1, len(sel) // (2 if QUICK else 4))][: (2 if QUICK else 4)]:
            try:
                ref = mp_first_order(p['L'], p['mu'], p['M_left'], p['M_right'])
                res.setdefault(key, []).append(abs(p['mu'] - float(ref)))
            except ValueError:
                res.setdefault(key, []).append(float('nan'))
    print(f"    переходы 1-го рода (трассировка против mpmath):")
    for key, v in res.items():
        v = [x for x in v if math.isfinite(x)]
        if v:
            print(f"      {key:7s}: {len(v)} точек, max |Δμ| = {max(v):.1e}")


if __name__ == '__main__':
    p1()
    p2()
    p3()

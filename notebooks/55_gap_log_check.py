"""
Проверка: логарифмическая неустойчивость у порога — не шум.

Вместо разности Omega(M) − Omega(0) (шум ~1e-14 абсолютный) считается функция щели
    G(M, b) = (1/M) ∂Omega/∂M
напрямую, без вычитания больших чисел. Если у порога b = S_n
    Omega(M, S_n) − Omega(0) = M² [A_n − c_n ln(1/M)] + o(M²),
то
    G(M, S_n) = 2A_n + c_n + 2c_n ln M + o(1),
т.е. G − 2c_n ln M → const = 2A_n + c_n при M → 0 — проверяется до M = 1e-30.

Вклады в G (b ≥ 0, b ≥ M):
    1/g + b/2                                           (вакуум и ΔU = M² b / 4)
    (2/(π² L)) ∫_0^∞ dp Σ± s'(L|A±|) sgn(A±) / E        (Ω_L; s'(x) — производная таблицы)
    (1/(πL)) Σ_n C_n Σ± ∫_{F>0} A± / (E sqrt(A±² + k_n²)) dp   (Ω_μL)
Для Ω_μL замена p = M sinh u (dp/E = du) убирает особенность масштаба M при p → 0.

Перепроверка: G_μL в mpmath (40 цифр) в нескольких точках; G против конечных
разностей Omega при M = 1e-2 (где разность Omega надёжна).

Запуск:  python notebooks/55_gap_log_check.py
"""

import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full import formulas as F
from njl3.full.minimize import channel_asymptotics, thresholds

PI = math.pi
G = -1.0
GLX, GLW = np.polynomial.legendre.leggauss(64)

# ---------------------------------------------------------------------------
# s'(x) из таблицы Чебышёва (кусок 0 хранит s + (x²/4) ln x)
# ---------------------------------------------------------------------------
_DER = [(a, b, np.polynomial.chebyshev.chebder(c)) for a, b, c in F._PIECES]


def s_prime(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i, (a, b, dc) in enumerate(_DER):
        m = (x >= a) & (x < b)
        if not np.any(m):
            continue
        xm = x[m]
        v = np.polynomial.chebyshev.chebval((2 * xm - (a + b)) / (b - a), dc) * 2.0 / (b - a)
        if i == 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = v - np.where(xm > 0, xm / 2 * np.log(xm) + xm / 4, 0.0)
        out[m] = v
    return out


def _gl(a, b):
    h = 0.5 * (b - a)
    return h * GLX + 0.5 * (a + b), h * GLW


def G_L(M, b, L):
    """(1/M) ∂Ω_L/∂M = (2/(π² L)) ∫ dp Σ± s'(L|A±|) sgn(A±)/E."""
    p_b = math.sqrt(max(b * b - M * M, 0.0))
    p_hi = math.sqrt((b + 64.0 / L)**2)
    edges = sorted({0.0, p_b, p_hi} | {p_b + d for d in (0.01, 0.1, 1.0) if p_b + d < p_hi}
                   | {max(p_b - d, 0.0) for d in (0.01, 0.1, 1.0)})
    tot = 0.0
    for a, c in zip(edges[:-1], edges[1:]):
        # дополнительное деление длинных кусков
        n = max(1, int(math.ceil((c - a) / (2.0 / L))))
        for k in range(n):
            p, w = _gl(a + (c - a) * k / n, a + (c - a) * (k + 1) / n)
            E = np.sqrt(M * M + p * p)
            f = s_prime(L * (E + b)) / E + s_prime(L * np.abs(E - b)) * np.sign(E - b) / E
            tot += float(np.dot(w, f))
    return 2.0 / (PI * PI * L) * tot


def _asinh_ratio(p2, p1, M):
    """∫_{p1}^{p2} dp / sqrt(p² + M²) = ln((p2 + E2)/(p1 + E1))."""
    E1, E2 = math.sqrt(p1 * p1 + M * M), math.sqrt(p2 * p2 + M * M)
    return math.log((p2 + E2) / (p1 + E1))


def G_muL(M, b, mu, L):
    """(1/M) ∂Ω_μL/∂M = (1/(πL)) Σ C_n Σ± ∫ A/(E sqrt(A² + k²)) dp."""
    pE = lambda E: math.sqrt(max(E * E - M * M, 0.0))
    tot = 0.0
    for n, S, C in thresholds(mu, L):
        k = 2 * PI * n / L
        if n == 0:                                        # sgn(A±)/E — точно
            I = 0.0
            if mu - b > M:
                I += _asinh_ratio(pE(mu - b), 0.0, M)
            lo, hi = max(M, b - mu), b + mu
            e2 = min(b, hi)
            if e2 > lo:
                I -= _asinh_ratio(pE(e2), pE(lo), M)
            e3 = max(b, lo)
            if hi > e3:
                I += _asinh_ratio(pE(hi), pE(e3), M)
            tot += C * I
            continue
        I = 0.0
        segs = []
        if S - b > M:
            segs.append((0.0, pE(S - b), +1.0))
        lo, hi = max(M, b - S), b + S
        if hi > lo:
            pl, pr = pE(lo), pE(hi)
            pb = pE(b) if b > M else None
            if pb is not None and pl < pb < pr:
                segs += [(pl, pb, -1.0), (pb, pr, -1.0)]
            else:
                segs.append((pl, pr, -1.0))
        for a, c, sg in segs:
            # p = M sinh u: dp / E = du
            ua, uc = math.asinh(a / M), math.asinh(c / M)
            nn = max(1, int(math.ceil((uc - ua) / 1.0)))
            for j in range(nn):
                u, w = _gl(ua + (uc - ua) * j / nn, ua + (uc - ua) * (j + 1) / nn)
                A = M * np.cosh(u) + sg * b
                I += float(np.dot(w, A / np.sqrt(A * A + k * k)))
        tot += C * I
    return tot / (PI * L)


def G_total(M, b, mu, L, g):
    return 1.0 / g + b / 2.0 + G_L(M, b, L) + G_muL(M, b, mu, L)


def G_muL_mp(M, b, mu, L, dps=40):
    """G_μL в mpmath (то же разбиение, adaptive quad)."""
    with mp.workdps(dps):
        M, b, mu, L = mp.mpf(M), mp.mpf(b), mp.mpf(mu), mp.mpf(L)
        pE = lambda E: mp.sqrt(max(E * E - M * M, 0))
        tot = mp.mpf(0)
        nmax = int(mp.floor(L * mu / (2 * mp.pi)))
        for n in range(nmax + 1):
            k = 2 * mp.pi * n / L
            S = mp.sqrt(mu * mu - k * k)
            C = 1 if n == 0 else 2
            f = lambda u, sg: (M * mp.cosh(u) + sg * b) / mp.sqrt((M * mp.cosh(u) + sg * b)**2 + k * k)
            I = mp.mpf(0)
            if S - b > M:
                I += mp.quad(lambda u: f(u, 1), [0, mp.asinh(pE(S - b) / M)])
            lo, hi = max(M, b - S), b + S
            if hi > lo:
                ua, uc = mp.asinh(pE(lo) / M), mp.asinh(pE(hi) / M)
                pts = [ua, uc]
                if b > M:
                    ub = mp.asinh(pE(b) / M)
                    if ua < ub < uc:
                        pts = [ua, ub, uc]
                I += mp.quad(lambda u: f(u, -1), pts)
            tot += C * I
        return tot / (mp.pi * L)


def main():
    cases = [(6.0, 1.0), (5.0, 3.0)]
    Ms = [1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 1e-16, 1e-20, 1e-30]
    for mu, L in cases:
        for n, S, C in thresholds(mu, L):
            c = C * S / (2 * PI * L * mu)
            A_dbl = channel_asymptotics(S, C, mu, L, G)['A']
            print(f"\nμ={mu}, L={L}, n={n}: S={S:.6f}, c_n={c:.6f}; A_n (из Ω при M=1e-4) = {A_dbl:.6f} → ожидаем 2A+c = {2 * A_dbl + c:.6f}")
            for M in Ms:
                g_ = G_total(M, S, mu, L, G)
                print(f"   M={M:7.0e}: G = {g_:+12.6f}   G − 2c·ln M = {g_ - 2 * c * math.log(M):+.8f}")
            # контроль 1: G против конечных разностей Ω при M = 1e-2
            M0, h = 1e-2, 1e-6
            fd = (F.omega_rel(M0 + h, S, mu, L, G) - F.omega_rel(M0 - h, S, mu, L, G)) / (2 * h) / M0
            print(f"   контроль: G(1e-2) = {G_total(M0, S, mu, L, G):+.8f}, конечные разности Ω: {fd:+.8f}")
            # контроль 2: G_μL в mpmath
            for M in (1e-6, 1e-20):
                d = G_muL(M, S, mu, L)
                m = float(G_muL_mp(M, S, mu, L))
                print(f"   контроль mpmath: G_μL(M={M:.0e}) double {d:+.12f}, mpmath {m:+.12f}, Δ = {d - m:.1e}")


if __name__ == '__main__':
    main()

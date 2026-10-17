"""
Линии фазовых переходов и особые точки при b = 0 (phi = 0).

Вывод и обоснование: publication/notes/2026-10-17_phase-lines-b0.md

Фазы (глобальный минимум Omega по M):
    T — тривиальная,     M* = 0
    I_j — промежуточная, 0 < M* < mu   (заполнены KK-моды n = 0..j при массе M*)
    V — вакуумная,       M* ≥ mu       (ферми-шар пуст, M* = M_vac(L) не зависит от mu)

Аналитические элементы:
    G0(mu, L) = G(0⁺) строго возрастает по mu  ->  единственная граница
        устойчивости M = 0:  mu_s(L),  G0(mu_s, L) = 0.
    G(M) = G0 + G2 M² + O(M⁴),
        G2 = (2/(pi L)) [ L²/24 - 1/(4 mu²) - sum_{n≥1} C_n mu / (2 k_n² S_n) ],
        Omega - Omega(0) = G0 M²/2 + G2 M⁴/4 + ...
    На линии mu_s(L): G2 > 0 — переход 2-го рода, G2 < 0 — 1-го рода,
    G2 = 0 — трикритическая точка.

Численная трассировка: при фиксированном L сканируем mu гарантированным
минимизатором (analytic_b0.find_minimum_b0_certified), каждую смену фазы
уточняем бисекцией по mu, род перехода определяем по скачку M*.
"""

import math

import numpy as np
from scipy.optimize import brentq

from functions import analytic_b0 as an


# ==============================================================================
# Аналитические элементы
# ==============================================================================

def G2(mu, L):
    """Коэффициент при M² в разложении G(M) = G0 + G2 M² + ... (не зависит от g)."""
    s = L * L / 24.0 - 1.0 / (4.0 * mu * mu)
    for k, C in an._modes(mu, L)[1:]:
        S2 = mu * mu - k * k
        if S2 <= 0.0:
            return -math.inf          # ровно на пороге моды: G2 -> -inf
        s -= C * mu / (2.0 * k * k * math.sqrt(S2))
    return 2.0 / (math.pi * L) * s


def mu_stability(L, g, mu_max=50.0):
    """
    mu_s(L): единственный корень G0(mu, L) = 0 (G0 строго возрастает по mu).
    При mu > mu_s точка M = 0 локально устойчива. None, если корня нет до mu_max.
    """
    f = lambda mu: an.gap_G0(mu, L, g)
    lo, hi = 1e-12, 1.0
    while f(hi) < 0.0:
        lo, hi = hi, 2.0 * hi
        if hi > mu_max:
            return None
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15)


def M_vacuum(L, g):
    """Вакуумная масса: корень G на [0, ∞) без мод (mu = 0)."""
    f = lambda M: 1.0 / g + M / math.pi + 2.0 / (math.pi * L) * math.log(-math.expm1(-L * M))
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 1e-300, hi, xtol=1e-15, rtol=1e-15)


def mu_VT(L, g):
    """
    Переход V–T: Omega(M_vac) = Omega(0).  Omega(M_vac) от mu не зависит
    (Omega_muL = 0 при M ≥ mu), поэтому условие —  Omega_muL(0; mu, L) = W(L),
        W(L) = M_v²/(2g) + M_v³/(3 pi) + Omega_L(M_v) - Omega_L(0),
    а Omega_muL(0; mu, L) строго убывает по mu -> единственный корень.
    Это граница фаз, только если фаза I не глубже обеих (проверяется трассировкой).
    """
    Mv = M_vacuum(L, g)
    W = Mv**2 / (2 * g) + an.dU(Mv) + an.omega_L(Mv, L) - an.omega_L(0.0, L)
    f = lambda mu: an.omega_muL(0.0, mu, L) - W
    hi = 1.0
    while f(hi) > 0.0:
        hi *= 2.0
    return brentq(f, 1e-12, hi, xtol=1e-15, rtol=1e-15)


def peak(n, g):
    """
    Пик n: пересечение линии G0 = 0 с порогом моды mu = 2 pi n / L — явно.
    На пороге mu/k_m = n/m, поэтому G0 = 0 даёт
        L_n = -(2 g / pi) [ ln(4 pi n) + 2 sum_{m=1}^{n-1} arccosh(n/m) ],  mu_n = 2 pi n / L_n.
    При n -> ∞ (g = -1): mu_n = pi + 2√2 |zeta(-1/2)| n^{-3/2} + O(n^{-5/2})
    (численно, publication/notes/2026-10-17_phase-lines-b0.md).
    """
    s = math.log(4.0 * math.pi * n) + 2.0 * sum(math.acosh(n / m) for m in range(1, n))
    L = -2.0 * g / math.pi * s
    return L, 2.0 * math.pi * n / L


def tricritical_points(g, L_lo, L_hi, n_scan=4000):
    """
    Точки mu = mu_s(L), где G2(mu_s(L), L) = 0 (непрерывная смена знака).
    Смены знака на порогах мод (G2 скачком -> -inf, это пики, а не трикритические
    точки) отбрасываются: на концах шага должно совпадать число заполненных мод.
    """
    Ls = np.linspace(L_lo, L_hi, n_scan)
    mus = [mu_stability(L, g) for L in Ls]
    vals = np.array([G2(mu, L) for mu, L in zip(mus, Ls)])
    nmax = [len(an._modes(mu, L)) for mu, L in zip(mus, Ls)]
    h = lambda L: G2(mu_stability(L, g), L)
    pts = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        if nmax[i] != nmax[i + 1] or not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        Lt = brentq(h, Ls[i], Ls[i + 1], xtol=1e-14)
        pts.append((Lt, mu_stability(Lt, g)))
    return pts


# ==============================================================================
# Численная трассировка
# ==============================================================================

def phase_at(mu, L, g):
    """
    (метка фазы, M*). Метки: 'T' (M* = 0), 'V' (M* ≥ mu), 'I<j>' (0 < M* < mu),
    где j = floor(L sqrt(mu² - M*²) / (2 pi)) — старшая заполненная KK-мода
    при массе M* (подфазы промежуточной фазы по числу заполненных мод).
    """
    M, _, _ = an.find_minimum_b0_certified(mu, L, g)
    if M == 0.0:
        return 'T', 0.0
    if M >= mu:
        return 'V', M
    return f'I{int(math.floor(L * math.sqrt(mu * mu - M * M) / (2.0 * math.pi)))}', M


def _bisect_switch(a, la, b, L, g, tol):
    """Первая смена фазы на (a, b] при метке la слева: бисекция по mu."""
    while b - a > tol:
        c = 0.5 * (a + b)
        if phase_at(c, L, g)[0] == la:
            a = c
        else:
            b = c
    return 0.5 * (a + b)


def _classify(L, g, mu_c, frm, to, Ml, Mr):
    """
    Род перехода. Для I–T — по теории: 2-й род, если mu_c совпадает с mu_s(L)
    и G2(mu_s) > 0 (минимум рождается из M = 0 непрерывно); иначе 1-й.
    Для переходов с участием V — всегда 1-й (M* скачет от M_vac ≥ mu).
    """
    if 'T' in (frm, to) and (frm.startswith('I') or to.startswith('I')):
        mu_s = mu_stability(L, g)
        if mu_s is not None and abs(mu_c - mu_s) < 1e-7 and G2(mu_s, L) > 0.0:
            return 2, mu_c
        return 1, mu_c
    return 1, mu_c


def trace_L(L, g, mu_lo=0.05, mu_hi=7.0, n_mu=400, tol=1e-12, eps=1e-9):
    """
    Все смены фазы по mu при фиксированном L.

    Скан по mu с шагом (mu_hi - mu_lo)/(n_mu - 1); на каждом шаге, где метка
    меняется, бисекцией находим первую смену, определяем фактическую метку справа
    и, если она отличается от метки сетки, ищем следующую смену в том же шаге
    (несколько переходов внутри одного шага). Переходы туда-обратно внутри одного
    шага (A→B→A) не видны — ограничение шага сканирования.

    Возвращает список словарей: L, mu, frm, to, order (1|2), M_left, M_right.
    """
    mus = np.linspace(mu_lo, mu_hi, n_mu)
    labels = [phase_at(mu, L, g)[0] for mu in mus]
    out = []
    for k in range(n_mu - 1):
        a, la, b, lb = mus[k], labels[k], mus[k + 1], labels[k + 1]
        while la != lb:
            mu_c = _bisect_switch(a, la, b, L, g, tol)
            to, Mr = phase_at(mu_c + eps, L, g)
            _, Ml = phase_at(mu_c - eps, L, g)
            order, mu_c = _classify(L, g, mu_c, la, to, Ml, Mr)
            out.append(dict(L=L, mu=mu_c, frm=la, to=to, order=order, M_left=Ml, M_right=Mr))
            a, la = mu_c + eps, to
    return out

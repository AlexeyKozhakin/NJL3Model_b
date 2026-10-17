"""
Замкнутые формулы для потенциала NJL3 на R² × S¹ при b = 0, phi = 0.

Обозначения: k_n = 2 pi n / L, m_n = sqrt(M² + k_n²), P_n = sqrt(mu² - m_n²),
S_n = sqrt(mu² - k_n²) (порог моды n по M), C_0 = 1, C_{n≥1} = 2,
n_max = floor(L mu / (2 pi)).

    Delta U(M)   = M³ / (3 pi)
    Omega_L(M)   = 2/(pi L³) [ L M Li₂(e^{-LM}) + Li₃(e^{-LM}) ]
    Omega_muL(M) = -(2/(pi L)) sum_{m_n<mu} C_n I(m_n),
                   I(m) = mu P/2 - (m²/2) ln((mu + P)/m)

Уравнение щели dOmega/dM = M G(M):

    G(M) = 1/g + M/pi + (2/(pi L)) [ ln(1 - e^{-LM}) + sum_{m_n<mu} C_n arccosh(mu/m_n) ]
    G(0⁺) = 1/g + (2/(pi L)) [ ln(2 mu L) + 2 sum_{n=1}^{n_max} arccosh(mu/k_n) ]
    G(M) = G0 + G2 M² + O(M⁴),
    G2 = (2/(pi L)) [ L²/24 - 1/(4 mu²) - sum_{n≥1} C_n mu / (2 k_n² S_n) ]

Логарифмические особенности Omega_L и моды n = 0 при M -> 0 сокращаются; функции
ниже вычисляют их в объединённой (устойчивой) форме.

Вывод: publication/notes/2026-10-16_analytic-b0.md, 2026-10-17_root-isolation-b0.md.
"""

import math

import mpmath
import numpy as np


# ==============================================================================
# Моды
# ==============================================================================

def modes(mu, L):
    """Заполненные при M = 0 моды: список (k_n, C_n) для n = 0..n_max."""
    if mu <= 0.0:
        return []
    n_max = int(math.floor(L * mu / (2.0 * math.pi)))
    return [(2.0 * math.pi * n / L, 1.0 if n == 0 else 2.0) for n in range(n_max + 1)]


def thresholds(mu, L):
    """Пороги S_n = sqrt(mu² - k_n²) по возрастанию (последний — S_0 = mu)."""
    return sorted(math.sqrt(mu * mu - k * k) for k, _ in modes(mu, L))


def filled_mode(M, mu, L):
    """Старшая заполненная мода при массе M < mu: floor(L sqrt(mu² - M²) / (2 pi))."""
    return int(math.floor(L * math.sqrt(mu * mu - M * M) / (2.0 * math.pi)))


# ==============================================================================
# Потенциал
# ==============================================================================

def dU(M):
    """Вакуумный вклад Delta U(M) = M³ / (3 pi)."""
    return M**3 / (3.0 * math.pi)


def omega_L(M, L):
    """Компактификационный вклад Omega_L(M); Omega_L(0) = 2 zeta(3) / (pi L³)."""
    x = math.exp(-L * M)
    return 2.0 / (math.pi * L**3) * float(L * M * mpmath.polylog(2, x) + mpmath.polylog(3, x))


def _I(m, mu):
    if m >= mu:
        return 0.0
    P = math.sqrt(mu * mu - m * m)
    if m == 0.0:
        return mu * P / 2.0
    return mu * P / 2.0 - m * m / 2.0 * math.log((mu + P) / m)


def omega_muL(M, mu, L):
    """Термический вклад Omega_muL(M)."""
    return -2.0 / (math.pi * L) * sum(C * _I(math.sqrt(M * M + k * k), mu) for k, C in modes(mu, L))


def omega_rel_closed(M, mu, L, g):
    """
    Omega(M) - Omega(0) по замкнутым формулам.
    Внимание: при малых M вычитает числа порядка единицы (см. omega_rel).
    """
    return (M**2 / (2.0 * g) + dU(M)
            + omega_L(M, L) - omega_L(0.0, L)
            + omega_muL(M, mu, L) - omega_muL(0.0, mu, L))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def omega_rel(M, mu, L, g, M_switch=0.5):
    """
    Omega(M) - Omega(0) без потери точности.

    При M < M_switch: ∫_0^M m G(m) dm квадратурой Гаусса–Лежандра (48 узлов на отрезок
    между порогами мод). У G корневая особенность sqrt(S - m) слева от порога S,
    поэтому на отрезке [a, b], если ближайший порог S ≥ b ближе длины отрезка,
    делается замена m = S - u² (подынтегральное выражение становится гладким). Относительная
    точность ~1e-14 даже при разности ~1e-16 (около переходов 2-го рода).
    При M ≥ M_switch — замкнутая формула.
    """
    if M >= M_switch:
        return omega_rel_closed(M, mu, L, g)
    if M == 0.0:
        return 0.0
    S_all = thresholds(mu, L)
    edges = [0.0] + [S for S in S_all if 0.0 < S < M] + [M]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        S = next((x for x in S_all if x >= b), None)
        if S is None or S - b >= b - a:
            # особенность далеко (или её нет): обычная квадратура по m
            m = 0.5 * (b - a) * _GL_X + 0.5 * (a + b)
            total += 0.5 * (b - a) * np.dot(_GL_W, m * gap_G_vec(m, mu, L, g))
        else:
            # порог близко: m = S - u², u ∈ [sqrt(S - b), sqrt(S - a)]
            ua, ub = math.sqrt(S - b), math.sqrt(S - a)
            u = 0.5 * (ub - ua) * _GL_X + 0.5 * (ua + ub)
            m = np.maximum(S - u * u, 1e-300)
            total += 0.5 * (ub - ua) * np.dot(_GL_W, 2.0 * u * m * gap_G_vec(m, mu, L, g))
    return float(total)


# ==============================================================================
# Уравнение щели
# ==============================================================================

def _inv_expm1_minus_inv(x):
    """1/(e^x - 1) - 1/x без потери точности при малых x."""
    x = np.asarray(x, dtype=float)
    small = x < 0.1
    xs = np.where(small, x, 0.1)
    series = -0.5 + xs / 12.0 - xs**3 / 720.0 + xs**5 / 30240.0
    xl = np.where(small, 1.0, x)
    return np.where(small, series, 1.0 / np.expm1(xl) - 1.0 / xl)


def gap_G_vec(M, mu, L, g):
    """
    G(M) для массива M > 0.

    При M < mu особые слагаемые объединены:
        ln(1 - e^{-LM}) + ln((mu + P_0)/M) = ln[(1 - e^{-x})/x] + ln[L (mu + P_0)],  x = LM.
    """
    M = np.asarray(M, dtype=float)
    x = L * M
    P0 = np.sqrt(np.clip(mu * mu - M * M, 0.0, None))
    n0 = M < mu
    s = np.where(n0,
                 np.log(-np.expm1(-x) / np.where(x > 0, x, 1.0)) + np.log(L * (mu + P0)),
                 np.log(-np.expm1(-x)))
    for k, C in modes(mu, L)[1:]:
        m = np.sqrt(M * M + k * k)
        P = np.sqrt(np.clip(mu * mu - m * m, 0.0, None))
        s = s + C * np.where(m < mu, np.log((mu + P) / m), 0.0)
    return 1.0 / g + M / math.pi + 2.0 / (math.pi * L) * s


def gap_G(M, mu, L, g):
    """G(M) при M > 0 (скаляр)."""
    return float(gap_G_vec(np.array([M]), mu, L, g)[0])


def gap_G0(mu, L, g):
    """G(0⁺) с явно сокращёнными логарифмами (-inf при mu ≤ 0)."""
    if mu <= 0.0:
        return -math.inf
    s = math.log(2.0 * mu * L)
    for k, C in modes(mu, L)[1:]:
        s += C * math.acosh(mu / k)
    return 1.0 / g + 2.0 / (math.pi * L) * s


def gap_dG_vec(M, mu, L):
    """
    G'(M) для массива M > 0 (не зависит от g):
        G' = (1/pi) coth(LM/2) - (2 mu/(pi L)) sum_{m_n<mu} C_n M / (m_n² P_n),
    особенности ~1/M объединены: (2/pi)[1/(e^x - 1) - 1/x] - (2/pi) M / (L P_0 (mu + P_0)).
    """
    M = np.asarray(M, dtype=float)
    x = L * M
    P0 = np.sqrt(np.clip(mu * mu - M * M, 0.0, None))
    n0 = M < mu
    out = 1.0 / math.pi + (2.0 / math.pi) * np.where(
        n0,
        _inv_expm1_minus_inv(x) - M / (L * np.where(n0, P0, 1.0) * (mu + P0)),
        1.0 / np.expm1(np.where(x > 0, x, 1.0)))
    for k, C in modes(mu, L)[1:]:
        m2 = M * M + k * k
        P = np.sqrt(np.clip(mu * mu - m2, 0.0, None))
        active = m2 < mu * mu
        out = out - np.where(active, 2.0 / (math.pi * L) * C * mu * M / (m2 * np.where(active, P, 1.0)), 0.0)
    return out


def G2(mu, L):
    """Коэффициент при M² в G(M) = G0 + G2 M² + ... (не зависит от g; -inf ровно на пороге моды)."""
    s = L * L / 24.0 - 1.0 / (4.0 * mu * mu)
    for k, C in modes(mu, L)[1:]:
        S2 = mu * mu - k * k
        if S2 <= 0.0:
            return -math.inf
        s -= C * mu / (2.0 * k * k * math.sqrt(S2))
    return 2.0 / (math.pi * L) * s


# ==============================================================================
# Монотонное разложение G = 1/g + A(M) + B(M) на [0, mu]
# ==============================================================================
#
#   A(M) = M/pi + (2/(pi L)) ln[(1 - e^{-LM})/(LM)]  — возрастает:
#          A'(M) = (1/pi) [coth(LM/2) - 2/(LM)] > 0;
#   B(M) = (2/(pi L)) [ln(L (mu + P_0)) + sum_{n≥1, m_n<mu} C_n arccosh(mu/m_n)] — убывает.
#   Поэтому на [p, q] ⊂ [0, mu]:  1/g + A(p) + B(q) ≤ G ≤ 1/g + A(q) + B(p).

def part_A(M, L):
    """A(M), A(0) = 0. Возрастает."""
    if M == 0.0:
        return 0.0
    x = L * M
    return M / math.pi + 2.0 / (math.pi * L) * math.log(-math.expm1(-x) / x)


def part_B(M, mu, L):
    """B(M) на [0, mu]. Убывает."""
    s = math.log(L * (mu + math.sqrt(max(mu * mu - M * M, 0.0))))
    for k, C in modes(mu, L)[1:]:
        m = math.sqrt(M * M + k * k)
        if m < mu:
            s += C * math.acosh(mu / m)
    return 2.0 / (math.pi * L) * s

"""
Замкнутые формулы для Omega_ren при b = 0, phi = 0.

Вывод: publication/notes/2026-10-16_analytic-b0.md

При b = 0 все три вклада берутся в элементарных функциях и полилогарифмах:

    Delta U(M)       = M³ / (3 pi)
    Omega_L(M)       = 2 / (pi L³) · [ L M Li₂(e^{-LM}) + Li₃(e^{-LM}) ]
    Omega_muL(M)     = -(2 / (pi L)) · sum_n C_n I(m_n),
        m_n = sqrt(M² + k_n²),  k_n = 2 pi n / L,  C_0 = 1, C_{n>0} = 2,
        I(m) = mu P / 2 - (m² / 2) ln((mu + P) / m),  P = sqrt(mu² - m²)   (m < mu)

Уравнение щели dOmega/dM = M · G(M):

    G(M) = 1/g + M/pi + (2 / (pi L)) · [ ln(1 - e^{-LM})
                                         + sum_{n: m_n < mu} C_n ln((mu + P_n) / m_n) ]

Логарифмы ln(1 - e^{-LM}) ≈ ln(LM) и -ln(m_0) = -ln M сокращаются:

    G(0) = 1/g + (2 / (pi L)) · [ ln(2 mu L) + 2 sum_{n=1}^{n_max} ln((mu + S_n) / k_n) ],
    S_n = sqrt(mu² - k_n²),  n_max = floor(L mu / (2 pi)).

У нуля Omega(M) - Omega(0) = a M² + O(M⁴), a = G(0) / 2.
"""

import math

import mpmath


def _modes(mu, L):
    """k_n для n = 0..n_max (k_n < mu) и веса C_n."""
    n_max = int(math.floor(L * mu / (2.0 * math.pi))) if mu > 0 else -1
    return [(2.0 * math.pi * n / L, 1.0 if n == 0 else 2.0) for n in range(n_max + 1)]


def dU(M):
    return M**3 / (3.0 * math.pi)


def omega_L(M, L):
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
    acc = 0.0
    for k, C in _modes(mu, L):
        acc += C * _I(math.sqrt(M * M + k * k), mu)
    return -2.0 / (math.pi * L) * acc


def omega_ren_rel(M, mu, L, g):
    """Omega_ren(M) - Omega_ren(0) при b = 0."""
    return (M**2 / (2.0 * g) + dU(M)
            + omega_L(M, L) - omega_L(0.0, L)
            + omega_muL(M, mu, L) - omega_muL(0.0, mu, L))


def gap_G(M, mu, L, g):
    """G(M) = (dOmega/dM) / M при b = 0, M > 0 (устойчивая форма, см. gap_G_vec)."""
    import numpy as np
    return float(gap_G_vec(np.array([M]), mu, L, g)[0])


def gap_G0(mu, L, g):
    """G(0⁺) с явно сокращёнными логарифмами."""
    s = math.log(2.0 * mu * L)
    for k, C in _modes(mu, L)[1:]:
        s += C * math.log((mu + math.sqrt(mu * mu - k * k)) / k)
    return 1.0 / g + 2.0 / (math.pi * L) * s


# ==============================================================================
# Векторизованная G и поиск глобального минимума по M при b = 0
# ==============================================================================

def _inv_expm1_minus_inv(x):
    """1/(e^x - 1) - 1/x без потери точности при малых x."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    small = x < 0.1
    xs = np.where(small, x, 0.1)
    series = -0.5 + xs / 12.0 - xs**3 / 720.0 + xs**5 / 30240.0
    xl = np.where(small, 1.0, x)
    direct = 1.0 / np.expm1(xl) - 1.0 / xl
    return np.where(small, series, direct)


def gap_G_vec(M, mu, L, g):
    """
    G(M) для массива M > 0 (numpy).

    Устойчивая форма: при M < mu логарифмические особенности Omega_L и моды n = 0
    объединены аналитически,
        ln(1 - e^{-LM}) + ln((mu + P_0)/M) = ln[(1 - e^{-x})/x] + ln[L (mu + P_0)],  x = LM,
    поэтому нет вычитания больших почти равных чисел при M -> 0.
    """
    import numpy as np
    M = np.asarray(M, dtype=float)
    x = L * M
    P0 = np.sqrt(np.clip(mu * mu - M * M, 0.0, None))
    n0 = M < mu
    s = np.where(n0,
                 np.log(-np.expm1(-x) / np.where(x > 0, x, 1.0)) + np.log(L * (mu + P0)),
                 np.log(-np.expm1(-x)))
    for k, C in _modes(mu, L)[1:]:
        m = np.sqrt(M * M + k * k)
        P = np.sqrt(np.clip(mu * mu - m * m, 0.0, None))
        s = s + C * np.where(m < mu, np.log((mu + P) / m), 0.0)
    return 1.0 / g + M / math.pi + 2.0 / (math.pi * L) * s


def find_minimum_b0(mu, L, g, M_max=8.0, n_scan=600):
    """
    Глобальный минимум Omega(M) при b = 0.

    Кандидаты: M = 0 (всегда стационарная точка, т.к. dOmega/dM = M·G) и корни
    G(M) = 0, где G меняет знак с «−» на «+» (локальные минимумы). Сетка нужна
    только для отделения корней; сами корни уточняются методом Брента.
    M_max автоматически растёт, пока G(M_max) < 0 (при малых L M_vac ~ 1/L).

    Возвращает (M_star, Omega_star - Omega(0), список локальных минимумов).
    """
    import numpy as np
    from scipy.optimize import brentq

    # G(M) ~ M/pi при больших M: расширяем диапазон, пока все корни не внутри
    while gap_G(M_max, mu, L, g) < 0:
        M_max *= 2.0

    M_scan = np.unique(np.concatenate([np.logspace(-8, 0, n_scan // 3),
                                       np.linspace(1e-3, M_max, n_scan)]))
    G_scan = gap_G_vec(M_scan, mu, L, g)

    minima = []
    for i in np.nonzero((G_scan[:-1] < 0) & (G_scan[1:] >= 0))[0]:
        r = brentq(lambda x: gap_G(x, mu, L, g), M_scan[i], M_scan[i + 1], xtol=1e-14)
        minima.append((r, omega_ren_rel(r, mu, L, g)))

    best_M, best_Om = 0.0, 0.0
    for r, om in minima:
        if om < best_Om:
            best_M, best_Om = r, om
    return best_M, best_Om, minima


def gap_dG_vec(M, mu, L):
    """
    G'(M) для массива M > 0 (не зависит от g):
        G' = 1/pi + (2/pi) / (e^{LM} - 1) - (2/(pi L)) sum_{m_n<mu} C_n mu M / (m_n² P_n).
    Устойчивая форма: особенности ~1/M от Omega_L и моды n = 0 объединены,
        (2/pi)[1/(e^x - 1) - 1/x] - (2/pi) M / (L P_0 (mu + P_0)),  x = LM.
    """
    import numpy as np
    M = np.asarray(M, dtype=float)
    x = L * M
    P0 = np.sqrt(np.clip(mu * mu - M * M, 0.0, None))
    n0 = M < mu
    out = 1.0 / math.pi + (2.0 / math.pi) * np.where(
        n0,
        _inv_expm1_minus_inv(x) - M / (L * np.where(n0, P0, 1.0) * (mu + P0)),
        1.0 / np.expm1(np.where(x > 0, x, 1.0)))
    for k, C in _modes(mu, L)[1:]:
        m2 = M * M + k * k
        P = np.sqrt(np.clip(mu * mu - m2, 0.0, None))
        active = m2 < mu * mu
        out = out - np.where(active, 2.0 / (math.pi * L) * C * mu * M / (m2 * np.where(active, P, 1.0)), 0.0)
    return out


def thresholds(mu, L):
    """Пороги S_n = sqrt(mu² - k_n²), n = n_max..0 (по возрастанию); S_0 = mu."""
    return sorted(math.sqrt(mu * mu - k * k) for k, _ in _modes(mu, L))


# ==============================================================================
# Гарантированное отделение корней G при b = 0
# ==============================================================================
#
# Теорема (монотонное разложение).
#   1) На [mu, ∞) мод нет и G(M) = 1/g + M/pi + (2/(pi L)) ln(1 - e^{-LM}) строго
#      возрастает: не более одного корня (вакуумный минимум).
#   2) На [0, mu]  G(M) = 1/g + A(M) + B(M),
#        A(M) = M/pi + (2/(pi L)) ln[(1 - e^{-LM})/(LM)]                 — возрастает,
#               A'(M) = (1/pi) [coth(LM/2) - 2/(LM)] > 0  (coth y > 1/y);
#        B(M) = (2/(pi L)) [ ln(L (mu + P_0)) + sum_{n≥1, m_n<mu} C_n arccosh(mu/m_n) ]
#                                                                         — убывает
#      (P_0 убывает, m_n растёт; arccosh непрерывно обращается в 0 на пороге моды).
#      Поэтому на любом [p, q] ⊂ [0, mu]
#          1/g + A(p) + B(q)  ≤  G(M)  ≤  1/g + A(q) + B(p),
#      и если 0 вне этой вилки, корней на [p, q] нет. Деление пополам с таким
#      отсечением находит ВСЕ корни с заданным разрешением (branch-and-bound).
#      У A и B производные ~M при M -> 0, поэтому вилка узкая и около нуля
#      (более грубое разложение M/pi + D(M) давало там широкую вилку и медленный поиск).

def _A(M, L):
    """A(M) = M/pi + (2/(pi L)) ln[(1 - e^{-LM})/(LM)], A(0) = 0. Возрастает."""
    if M == 0.0:
        return 0.0
    x = L * M
    return M / math.pi + 2.0 / (math.pi * L) * math.log(-math.expm1(-x) / x)


def _B(M, mu, L):
    """B(M) = (2/(pi L)) [ln(L(mu + P0)) + sum_{n≥1} C_n arccosh(mu/m_n)] на [0, mu]. Убывает."""
    s = math.log(L * (mu + math.sqrt(max(mu * mu - M * M, 0.0))))
    for k, C in _modes(mu, L)[1:]:
        m = math.sqrt(M * M + k * k)
        if m < mu:
            s += C * math.acosh(mu / m)
    return 2.0 / (math.pi * L) * s


def isolate_roots_b0(mu, L, g, tol=1e-10):
    """
    Все корни G(M) = 0 при M > 0, найденные с гарантией полноты.

    Возвращает список (M_root, kind), kind ∈ {'min', 'max', 'touch'}:
        'min'   — G меняет знак − → + (локальный минимум Omega),
        'max'   — G меняет знак + → − (локальный максимум Omega),
        'touch' — касание нуля без смены знака в окне ширины tol (вырожденный
                  случай, граница существования минимума).
    """
    from scipy.optimize import brentq

    roots = []
    inv_g = 1.0 / g

    # --- [0, mu]: branch-and-bound с монотонной вилкой ------------------------
    if mu > 0:
        windows = []
        stack = [(0.0, mu, _A(0.0, L), _A(mu, L), _B(0.0, mu, L), _B(mu, mu, L))]
        while stack:
            p, q, Ap, Aq, Bp, Bq = stack.pop()
            lo = inv_g + Ap + Bq
            hi = inv_g + Aq + Bp
            if lo > 0.0 or hi < 0.0:
                continue
            if q - p <= tol:
                windows.append((p, q))
                continue
            c = 0.5 * (p + q)
            Ac, Bc = _A(c, L), _B(c, mu, L)
            stack.append((c, q, Ac, Aq, Bc, Bq))
            stack.append((p, c, Ap, Ac, Bp, Bc))
        # склеиваем соседние окна
        windows.sort()
        merged = []
        for p, q in windows:
            if merged and p <= merged[-1][1] + 1e-15:
                merged[-1] = (merged[-1][0], max(merged[-1][1], q))
            else:
                merged.append((p, q))
        for p, q in merged:
            Gp = gap_G0(mu, L, g) if p == 0.0 else gap_G(p, mu, L, g)
            Gq = gap_G(q, mu, L, g)
            if p == 0.0 and q <= tol:
                continue                     # окрестность M = 0: корень M = 0 учитывается отдельно
            if Gp < 0.0 < Gq:
                roots.append((brentq(lambda x: gap_G(x, mu, L, g), p, q, xtol=1e-15), 'min'))
            elif Gp > 0.0 > Gq:
                roots.append((brentq(lambda x: gap_G(x, mu, L, g), p, q, xtol=1e-15), 'max'))
            else:
                roots.append((0.5 * (p + q), 'touch'))

    # --- [mu, ∞): G строго возрастает, не более одного корня -----------------
    a = max(mu, 1e-300)
    if gap_G(a, mu, L, g) < 0.0:
        b = 2.0 * a + 1.0
        while gap_G(b, mu, L, g) < 0.0:
            b *= 2.0
        roots.append((brentq(lambda x: gap_G(x, mu, L, g), a, b, xtol=1e-15), 'min'))

    return sorted(roots)


def omega_rel_stable(M, mu, L, g, M_switch=0.5):
    """
    Omega(M) - Omega(0) без потери точности при малых M.

    При M < M_switch: Omega(M) - Omega(0) = ∫_0^M m G(m) dm (адаптивная квадратура,
    точки разбиения — пороги мод внутри [0, M]); относительная точность ~1e-13 даже
    когда сама разность ~1e-16 (около переходов 2-го рода). Замкнутая формула
    omega_ren_rel вычитает числа порядка единицы и там даёт шум округления.
    При M ≥ M_switch — замкнутая формула.
    """
    if M >= M_switch:
        return omega_ren_rel(M, mu, L, g)
    import warnings
    from scipy.integrate import quad, IntegrationWarning
    pts = [S for S in thresholds(mu, L) if 0.0 < S < M] if mu > 0 else []
    f = lambda m: m * (gap_G0(mu, L, g) if m == 0.0 else gap_G(m, mu, L, g))
    with warnings.catch_warnings():
        # quad сообщает о пределе округления при epsrel = 1e-13 — точность при этом ~1e-15
        warnings.simplefilter('ignore', IntegrationWarning)
        val, _ = quad(f, 0.0, M, points=pts or None, epsabs=0.0, epsrel=1e-13, limit=200)
    return val


def find_minimum_b0_certified(mu, L, g, tol=1e-10):
    """
    Глобальный минимум Omega(M) при b = 0 с гарантией полноты.

    Кандидаты: M = 0 и все корни G типа 'min' (и 'touch' — на всякий случай).
    Возвращает (M_star, Omega_star - Omega(0), список (M, Omega - Omega(0)) локальных минимумов).
    """
    minima = [(r, omega_rel_stable(r, mu, L, g)) for r, kind in isolate_roots_b0(mu, L, g, tol)
              if kind in ('min', 'touch')]
    best_M, best_Om = 0.0, 0.0
    for r, om in minima:
        if om < best_Om:
            best_M, best_Om = r, om
    return best_M, best_Om, minima

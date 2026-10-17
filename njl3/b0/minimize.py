"""
Гарантированный поиск глобального минимума Omega(M) при b = 0 и определение фазы.

Теорема (монотонное разложение):
  1) На [mu, ∞) мод нет и G строго возрастает — не более одного корня
     (вакуумный минимум M_vac(L), от mu не зависит).
  2) На [0, mu]  G = 1/g + A(M) + B(M), A возрастает, B убывает, поэтому
         1/g + A(p) + B(q) ≤ G(M) ≤ 1/g + A(q) + B(p)   на [p, q].
     Деление пополам с отсечением по этой вилке находит ВСЕ корни G.

Кандидаты глобального минимума: M = 0 (всегда стационарна, dOmega/dM = M G) и все
корни G со сменой знака − → +.

Фазы: 'T' (M* = 0), 'V' (M* ≥ mu), 'I<j>' (0 < M* < mu, заполнены моды 0..j).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from njl3.b0 import formulas as F


def isolate_roots(mu, L, g, tol=1e-10):
    """
    Все корни G(M) = 0 при M > 0 с гарантией полноты.

    Возвращает отсортированный список (M, kind), kind ∈ {'min', 'max', 'touch'}:
    'min' — G меняет знак − → + (локальный минимум Omega), 'max' — + → −,
    'touch' — касание нуля без смены знака в окне ширины tol (граница
    существования минимума).
    """
    G = lambda x: F.gap_G(x, mu, L, g)
    roots = []
    inv_g = 1.0 / g

    if mu > 0.0:
        windows = []
        stack = [(0.0, mu, F.part_A(0.0, L), F.part_A(mu, L), F.part_B(0.0, mu, L), F.part_B(mu, mu, L))]
        while stack:
            p, q, Ap, Aq, Bp, Bq = stack.pop()
            if inv_g + Ap + Bq > 0.0 or inv_g + Aq + Bp < 0.0:
                continue
            if q - p <= tol:
                windows.append((p, q))
                continue
            c = 0.5 * (p + q)
            Ac, Bc = F.part_A(c, L), F.part_B(c, mu, L)
            stack.append((c, q, Ac, Aq, Bc, Bq))
            stack.append((p, c, Ap, Ac, Bp, Bc))
        windows.sort()
        merged = []
        for p, q in windows:
            if merged and p <= merged[-1][1] + 1e-15:
                merged[-1] = (merged[-1][0], max(merged[-1][1], q))
            else:
                merged.append((p, q))
        for p, q in merged:
            if p == 0.0 and q <= tol:
                continue                      # окрестность M = 0: M = 0 учитывается отдельно
            Gp = F.gap_G0(mu, L, g) if p == 0.0 else G(p)
            Gq = G(q)
            if Gp < 0.0 < Gq:
                roots.append((brentq(G, p, q, xtol=1e-15), 'min'))
            elif Gp > 0.0 > Gq:
                roots.append((brentq(G, p, q, xtol=1e-15), 'max'))
            else:
                roots.append((0.5 * (p + q), 'touch'))

    a = max(mu, 1e-300)
    if G(a) < 0.0:
        b = 2.0 * a + 1.0
        while G(b) < 0.0:
            b *= 2.0
        roots.append((brentq(G, a, b, xtol=1e-15), 'min'))
    return sorted(roots)


def find_minimum(mu, L, g, tol=1e-10, roots=None):
    """
    Глобальный минимум Omega(M) при b = 0 (гарантированный).

    roots — уже найденные isolate_roots(mu, L, g) (чтобы не искать повторно).
    Возвращает (M*, Omega(M*) - Omega(0), [(M, Omega - Omega(0)) для всех локальных минимумов M > 0]).
    """
    if roots is None:
        roots = isolate_roots(mu, L, g, tol)
    minima = [(r, F.omega_rel(r, mu, L, g)) for r, kind in roots if kind in ('min', 'touch')]
    best_M, best_Om = 0.0, 0.0
    for r, om in minima:
        if om < best_Om:
            best_M, best_Om = r, om
    return best_M, best_Om, minima


def find_minimum_scan(mu, L, g, M_max=8.0, n_scan=600):
    """
    Базовый (негарантированный) метод для сравнения: скан G по сетке M + Брент.
    Пропускает пары корней уже шага сетки.
    """
    while F.gap_G(M_max, mu, L, g) < 0:
        M_max *= 2.0
    M_scan = np.unique(np.concatenate([np.logspace(-8, 0, n_scan // 3), np.linspace(1e-3, M_max, n_scan)]))
    G_scan = F.gap_G_vec(M_scan, mu, L, g)
    minima = []
    for i in np.nonzero((G_scan[:-1] < 0) & (G_scan[1:] >= 0))[0]:
        r = brentq(lambda x: F.gap_G(x, mu, L, g), M_scan[i], M_scan[i + 1], xtol=1e-14)
        minima.append((r, F.omega_rel_closed(r, mu, L, g)))
    best_M, best_Om = 0.0, 0.0
    for r, om in minima:
        if om < best_Om:
            best_M, best_Om = r, om
    return best_M, best_Om, minima


def M_vacuum(L, g):
    """Вакуумная масса: корень 1/g + M/pi + (2/(pi L)) ln(1 - e^{-LM}) = 0."""
    f = lambda M: 1.0 / g + M / math.pi + 2.0 / (math.pi * L) * math.log(-math.expm1(-L * M))
    hi = 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    return brentq(f, 1e-300, hi, xtol=1e-15, rtol=1e-15)


def phase_label(M_star, mu, L):
    """'T' (M* = 0), 'V' (M* ≥ mu) или 'I<j>' (заполнены моды 0..j)."""
    if M_star == 0.0:
        return 'T'
    if M_star >= mu:
        return 'V'
    return f'I{F.filled_mode(M_star, mu, L)}'


def phase_at(mu, L, g):
    """(метка фазы, M*)."""
    M, _, _ = find_minimum(mu, L, g)
    return phase_label(M, mu, L), M


@dataclass
class PointResult:
    """Полная информация о точке (mu, L, g)."""
    mu: float
    L: float
    g: float
    phase: str
    M_star: float
    omega_star: float                       # Omega(M*) - Omega(0)
    G0: float
    G2: float
    roots: list = field(default_factory=list)          # [(M, kind)]
    local_minima: list = field(default_factory=list)   # [(M, Omega - Omega(0))]

    def to_dict(self):
        return dict(mu=self.mu, L=self.L, g=self.g, phase=self.phase, M_star=self.M_star,
                    omega_star=self.omega_star, G0=self.G0, G2=self.G2,
                    roots=[dict(M=m, kind=k) for m, k in self.roots],
                    local_minima=[dict(M=m, omega=o) for m, o in self.local_minima])


def analyze_point(mu, L, g):
    """Фаза, глобальный и локальные минимумы, корни G, G0, G2 в одной точке."""
    roots = isolate_roots(mu, L, g)
    M, Om, minima = find_minimum(mu, L, g, roots=roots)
    return PointResult(mu=mu, L=L, g=g, phase=phase_label(M, mu, L), M_star=M, omega_star=Om,
                       G0=F.gap_G0(mu, L, g), G2=F.G2(mu, L), roots=roots, local_minima=minima)

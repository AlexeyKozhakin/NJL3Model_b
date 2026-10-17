"""
Глобальный минимум Omega(M, b) при phi = 0 (b ≥ 0).

Структура (численно установлено, notebooks/51–53):
  * минимумы с b > 0 лежат у порогов мод S_n = sqrt(mu² − (2πn/L)²) — «каналы» n;
  * у порога при малых M (логарифмическая неустойчивость):
        Omega(M, S_n + tM) − Omega(0) ≈ M² [A_n − c_n ln(1/M) + c_n h(t)],
        c_n = C_n S_n / (2πLμ),  h(t) = t² (|t| ≤ 1),  t² − |t| sqrt(t² − 1) + arccosh|t| (|t| ≥ 1),
    откуда b* → S_n, M*_n = exp(−A_n/c_n − 1/2), Omega*_n = −(c_n/2) M*_n²;
  * кандидат b = 0 — точное решение njl3.b0 (гарантированный поиск по M).

Алгоритм для точки (mu, L):
  1) кандидат b = 0 (njl3.b0.find_minimum);
  2) для каждого канала n: сетка в переменных (ln M, t = (b − S_n)/M) — узкая по b яма
     становится гладкой; лучший узел уточняется методом Нелдера–Мида; при малой щели
     (< M_ASYMP_VALID) асимптотика — дополнительный кандидат, ниже M_ASYMP — единственный;
  3) глобальный минимум — лучший из кандидатов.

Порог по температуре T (notebooks/57): при T > 0 щель канала n исчезает при
    T > T_c,n = (e^γ/π) (S_n/mu) M*_n
(модель Гросса–Невё у порога; скорость на пороге S_n/mu). find_minimum(..., T) отбрасывает
каналы с T_c,n ≤ T (по асимптотике — без численного поиска) и ищет M не ниже порога
M_min,n = T / ((e^γ/π) S_n/mu). Остальная картина — минимум при T = 0.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from njl3.full import formulas as F
from njl3.b0 import find_minimum as b0_find_minimum
from njl3.b0 import phase_label as b0_phase_label

PI = math.pi
M_ASYMP = 1e-6          # ниже — асимптотическая формула
M_PROBE = 1e-4          # где извлекается A_n
M_ASYMP_VALID = 1e-2    # асимптотика применима (сверено с численным минимумом до ~1e-3 по ln M)
OMEGA_NOISE = 1e-12     # численный минимум глубже этого — достоверен (точность Omega ~1e-14)
_LNM_LO, _LNM_HI = math.log(M_ASYMP), math.log(6.0)
PRUNE_FACTOR = 0.8      # T > 0: асимптотический канал мельче 0.8·(лучший Ω) не уточняется
SKIP_GRID_ASYMP = True  # где асимптотика применима, искать яму от неё, без сетки (сверено: notebooks/58)
TC_RATIO = math.exp(0.5772156649015329) / PI      # e^γ/π: T_c = TC_RATIO · (S_n/mu) · M*_n


def thresholds(mu, L):
    """[(n, S_n, C_n)] для n = 0..n_max."""
    if mu <= 0:
        return []
    n_max = int(math.floor(L * mu / (2 * PI)))
    return [(n, math.sqrt(mu * mu - (2 * PI * n / L)**2), 1.0 if n == 0 else 2.0) for n in range(n_max + 1)]


def h_universal(t):
    """Универсальная функция сдвига от порога (в единицах M)."""
    t = abs(t)
    if t <= 1.0:
        return t * t
    return t * t - t * math.sqrt(t * t - 1.0) + math.acosh(t)


def channel_asymptotics(S, C, mu, L, g):
    """c_n, A_n и асимптотические M*, Omega* канала с порогом S."""
    c = C * S / (2 * PI * L * mu)
    A = F.omega_rel(M_PROBE, S, mu, L, g) / M_PROBE**2 + c * math.log(1.0 / M_PROBE)
    lnM = -A / c - 0.5
    M_as = math.exp(lnM) if lnM > -700 else 0.0
    return dict(c=c, A=A, lnM=lnM, M=M_as, omega=-0.5 * c * M_as * M_as)


_LNM = np.linspace(math.log(M_ASYMP), math.log(4.5), 40)
_T = np.concatenate([np.linspace(-4.0, -1.0, 13), np.linspace(-0.9, 0.9, 19), np.linspace(1.0, 3.0, 5)])


def channel_m_min(S, mu, T):
    """Наименьшая щель канала, выживающая при температуре T."""
    return T / (TC_RATIO * S / mu) if T > 0 else 0.0


def channel_minimum(S, C, mu, L, g, T=0.0, asy=None):
    """Минимум канала с порогом S: dict(M, b, omega, asym). При T > 0 — только M ≥ M_min,n."""
    if asy is None:
        asy = channel_asymptotics(S, C, mu, L, g)
    m_min = channel_m_min(S, mu, T)
    lnm_lo = max(_LNM_LO, math.log(m_min)) if m_min > 0 else _LNM_LO
    if lnm_lo >= math.log(4.5):
        return dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=False, c=asy['c'], A=asy['A'])
    # тот же шаг по ln M, что и у полной сетки
    lnm_grid = _LNM if lnm_lo == _LNM_LO else np.linspace(
        lnm_lo, math.log(4.5), max(8, int(round(40 * (math.log(4.5) - lnm_lo) / (math.log(4.5) - _LNM_LO)))))
    if SKIP_GRID_ASYMP and asy['M'] < M_ASYMP_VALID and asy['lnM'] > lnm_lo:
        # асимптотика применима: яма у (ln M*_n, t = 0) известна — сетка не нужна
        lg, tg = np.array([[asy['lnM']]]), np.array([[0.0]])
    else:
        lg, tg = np.meshgrid(lnm_grid, _T)
    Mg = np.exp(lg)
    bg = S + tg * Mg
    ok = bg >= 0
    Om = np.full(Mg.shape, np.inf)
    Om[ok] = F.omega_rel_vec(Mg[ok], bg[ok], mu, L, g)
    i, j = np.unravel_index(np.argmin(Om), Om.shape)
    best = dict(M=float(Mg[i, j]), lnM=float(lg[i, j]), b=float(bg[i, j]), omega=float(Om[i, j]), asym=False)

    def f(x):
        lm = min(max(x[0], lnm_lo), _LNM_HI)
        M = math.exp(lm)
        b = S + x[1] * M
        return F.omega_rel(M, b, mu, L, g) if b >= 0 else 1e3
    # Нелдер–Мид: только сравнения значений (Omega ~1e-7 при абсолютной точности ~1e-14 —
    # конечно-разностный градиент шумит)
    x0 = np.array([lg[i, j], tg[i, j]])
    simplex = np.array([x0, x0 + [0.3, 0.0], x0 + [0.0, 0.3]])
    r = minimize(f, x0, method='Nelder-Mead',
                 # при T > 0 щели ниже M_min не нужны — хватает точности 1e-6 по ln M
                 options=dict(initial_simplex=simplex, xatol=1e-6 if T > 0 else 1e-9,
                              fatol=(1e-10 if T > 0 else 1e-13) * abs(best['omega']) + 1e-300,
                              maxiter=600, maxfev=1200))
    if r.fun < best['omega']:
        lm = min(max(float(r.x[0]), lnm_lo), _LNM_HI)
        M = math.exp(lm)
        best = dict(M=M, lnM=lm, b=S + r.x[1] * M, omega=float(r.fun), asym=False)
    # численный минимум мельче шума — недостоверен
    if best['omega'] > -OMEGA_NOISE or best['M'] < m_min * (1 + 1e-9):
        best = dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=False)
    # асимптотика — кандидат там, где она применима (малая щель)
    if asy['M'] < M_ASYMP_VALID and asy['M'] > m_min and asy['omega'] < best['omega']:
        best = dict(M=asy['M'], lnM=asy['lnM'], b=S, omega=asy['omega'], asym=True)
    best.update(c=asy['c'], A=asy['A'])
    return best


@dataclass
class FullResult:
    mu: float
    L: float
    g: float
    phase: str                 # 'V', 'I<j>' (b = 0, из статьи 1) или 'B<n>' (b ≈ S_n)
    M_star: float
    ln_M_star: float           # ln M* (для экспоненциально малых щелей M* может быть 0 в double)
    b_star: float
    omega_star: float
    asymptotic: bool           # M* из асимптотики (экспоненциально малая щель)
    channels: list = field(default_factory=list)
    b0: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(mu=self.mu, L=self.L, g=self.g, phase=self.phase, M_star=self.M_star, ln_M_star=self.ln_M_star,
                    b_star=self.b_star,
                    omega_star=self.omega_star, asymptotic=self.asymptotic, channels=self.channels, b0=self.b0)


def find_minimum(mu, L, g, T=0.0):
    """Глобальный минимум по (M, b) в точке (mu, L); T > 0 — порог выживания щелей (см. модуль)."""
    Mb, Ob, _ = b0_find_minimum(mu, L, g)
    b0 = dict(M=Mb, omega=Ob, phase=b0_phase_label(Mb, mu, L))
    # b = 0: порог T_c ≈ (e^γ/π) M (скорость 1) — приближённо; щели при b = 0 обычно ~1
    if T > 0 and 0 < Mb <= T / TC_RATIO:
        Mb, Ob = 0.0, 0.0
    cand = [dict(phase=b0_phase_label(Mb, mu, L), M=Mb, lnM=math.log(Mb) if Mb > 0 else -math.inf, b=0.0, omega=Ob,
                 asym=False)]
    thr = thresholds(mu, L)
    asys = [channel_asymptotics(S, C, mu, L, g) for _, S, C in thr]
    # при T > 0 — только победитель: каналы в порядке асимптотической глубины; канал в области
    # асимптотики (точность Ω ~1e-3), явно мельче уже найденного лучшего, не уточняется
    order = sorted(range(len(thr)), key=lambda k: asys[k]['omega'] if asys[k]['M'] < M_ASYMP_VALID else -math.inf)
    channels = [None] * len(thr)
    best_om = min(Ob, 0.0)
    for k in order:
        (n, S, C), asy = thr[k], asys[k]
        small = asy['M'] < M_ASYMP_VALID
        if T > 0 and small and asy['M'] <= channel_m_min(S, mu, T):
            # щель канала (по асимптотике) не переживает температуру T — численный поиск не нужен
            ch = dict(M=0.0, lnM=-math.inf, b=S, omega=0.0, asym=True, c=asy['c'], A=asy['A'], pruned=True)
        elif T > 0 and small and asy['omega'] > PRUNE_FACTOR * best_om:
            ch = dict(M=asy['M'], lnM=asy['lnM'], b=S, omega=asy['omega'], asym=True, c=asy['c'], A=asy['A'],
                      pruned=True)
        else:
            ch = channel_minimum(S, C, mu, L, g, T=T, asy=asy)
        ch.update(n=n, S=S)
        channels[k] = ch
        if ch['M'] > 0:
            best_om = min(best_om, ch['omega'])
            cand.append(dict(phase=f'B{n}', M=ch['M'], lnM=ch['lnM'], b=ch['b'], omega=ch['omega'], asym=ch['asym']))
    # канал, ушедший в b ≈ 0, — это решение при b = 0 (уже есть кандидатом); при равенстве — b = 0
    cand = [c for c in cand if c['phase'][0] != 'B' or c['b'] > 1e-8]
    # каналы сравниваются между собой без запаса; b = 0 — точное (гарантированное) решение:
    # численный канал побеждает его только с запасом больше шума, асимптотический — без запаса
    best = cand[0]
    for c in sorted(cand[1:], key=lambda c: c['omega']):
        margin = 0.0 if c['asym'] else OMEGA_NOISE
        if c['omega'] < best['omega'] - margin:
            best = c
            break
    if best['phase'][0] == 'B':
        # номер фазы — ближайший порог к найденному b* (поиск в канале n может прийти в яму соседнего порога)
        n_near, _, _ = min(thresholds(mu, L), key=lambda q: abs(q[1] - best['b']))
        best = dict(best, phase=f'B{n_near}')
    if best['phase'] == 'T':                         # формально M = 0 при b = 0: b не определено
        best = dict(best, b=float('nan'))
    return FullResult(mu=mu, L=L, g=g, phase=best['phase'], M_star=best['M'], ln_M_star=best['lnM'], b_star=best['b'],
                      omega_star=best['omega'], asymptotic=best['asym'], channels=channels,
                      b0=b0)

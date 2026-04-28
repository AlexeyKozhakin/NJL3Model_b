"""
Аналитическая формула для Omega_muL(mu, L, b=0, M=0).

Без численного интегрирования — только сумма по KK-модам (n_max <= 4).

Формула (theory.md, раздел 9.2):
    Omega_muL(mu,L,0,0) = -(1/pi*L) * [mu^2 + 2*sum_{n=1}^{n_max}(mu*Sn - kn^2*ln((mu+Sn)/kn))]
где kn = 2*pi*n/L,  Sn = sqrt(mu^2 - kn^2),  n_max = floor(L*mu / 2*pi).
"""

import math


def compute(mu: float, L: float) -> float:
    """
    Omega_muL(mu, L, b=0, M=0) — аналитически.

    Параметры
    ---------
    mu : float
        Химический потенциал (>=0).
    L : float
        Размер компактного измерения (>0).

    Возвращает
    ----------
    float
        Значение термодинамического потенциала.
    """
    if mu <= 0.0:
        return 0.0

    two_pi = 2.0 * math.pi
    n_max = int(math.floor(L * mu / two_pi))

    # n=0: I_0 = mu^2 / (4*pi), вклад в сумму = mu^2
    acc = mu * mu

    for n in range(1, n_max + 1):
        kn = two_pi * n / L
        kn2 = kn * kn
        Sn2 = mu * mu - kn2
        if Sn2 <= 0.0:
            break
        Sn = math.sqrt(Sn2)
        acc += 2.0 * (mu * Sn - kn2 * math.log((mu + Sn) / kn))

    return -acc / (math.pi * L)

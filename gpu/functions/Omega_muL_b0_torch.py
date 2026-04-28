"""
Аналитическая формула для Omega_muL(mu, L, b, M=0) при произвольном b >= 0.

Без численного интегрирования — конечная сумма по KK-модам.

Формула (theory.md, раздел 9.3):
    Omega_muL(mu, L, b, 0) = -(2/L) * (U_np + U_nm)
    U_np = I_0^+ + 2*sum_{n=1}^{n_max^+} I_n^+
    U_nm = I_0^- + 2*sum_{n=1}^{n_max^-} I_n^-

Плюс-ветвь (n_max^+ = floor(L/(2*pi)*sqrt(mu^2-b^2)) при mu>b, иначе -1):
    I_0^+ = (mu-b)^2 / (4*pi)
    I_n^+ = (1/2*pi) * { mu*Sn/2 - mu*b + b*mu_b/2 + kn^2/2 * ln((b+mu_b)/(Sn+mu)) }
    mu_b = sqrt(b^2 + kn^2)

Минус-ветвь (n_max^- = floor(L*mu/(2*pi))):
    Если Sn >= b (один корень, p in [0, b+Sn]):
        I_0^- = (mu^2/2 + mu*b - b^2/2) / (2*pi)
        I_n^- = (1/2*pi) * { mu*Sn/2 + mu*b - b*mu_b/2 + kn^2/2 * ln((mu_b-b)/(Sn+mu)) }
    Если Sn < b (два корня, p in [b-Sn, b+Sn]):
        I_0^- = mu^2 / (2*pi)
        I_n^- = (1/2*pi) * { mu*Sn - kn^2 * ln((Sn+mu)/kn) }
"""

import math


def _I_n_plus(n: int, mu: float, b: float, L: float) -> float:
    """Вклад моды n в плюс-ветвь."""
    if n == 0:
        return (mu - b) ** 2 / (4.0 * math.pi)

    two_pi = 2.0 * math.pi
    kn = two_pi * n / L
    kn2 = kn * kn
    Sn = math.sqrt(mu * mu - kn2)          # гарантировано >= 0 (вызывается только для n <= n_max^+)
    mu_b = math.sqrt(b * b + kn2)
    return (1.0 / (2.0 * math.pi)) * (
        0.5 * mu * Sn
        - mu * b
        + 0.5 * b * mu_b
        + 0.5 * kn2 * math.log((b + mu_b) / (Sn + mu))
    )


def _I_n_minus(n: int, mu: float, b: float, Sn: float, L: float) -> float:
    """
    Вклад моды n в минус-ветвь.
    Sn = sqrt(mu^2 - kn^2) передаётся снаружи (уже вычислен).
    """
    if n == 0:
        if Sn >= b:
            return (0.5 * mu * mu + mu * b - 0.5 * b * b) / (2.0 * math.pi)
        else:
            return mu * mu / (2.0 * math.pi)

    two_pi = 2.0 * math.pi
    kn = two_pi * n / L
    kn2 = kn * kn

    if Sn >= b:
        mu_b = math.sqrt(b * b + kn2)
        # Защита от log(0): mu_b - b > 0 при kn > 0
        log_arg = (mu_b - b) / (Sn + mu)
        return (1.0 / (2.0 * math.pi)) * (
            0.5 * mu * Sn
            + mu * b
            - 0.5 * b * mu_b
            + 0.5 * kn2 * math.log(log_arg)
        )
    else:
        # Два корня
        log_arg = (Sn + mu) / kn
        return (1.0 / (2.0 * math.pi)) * (
            mu * Sn - kn2 * math.log(log_arg)
        )


def compute(mu: float, L: float, b: float) -> float:
    """
    Omega_muL(mu, L, b, M=0) — аналитически.

    Параметры
    ---------
    mu : float
        Химический потенциал (>=0).
    L : float
        Размер компактного измерения (>0).
    b : float
        Внешнее поле (>=0). Для b<0 используется |b| по симметрии.

    Возвращает
    ----------
    float
        Значение термодинамического потенциала.
    """
    if mu <= 0.0:
        return 0.0

    b = abs(b)           # потенциал чётный по b при M=0
    two_pi = 2.0 * math.pi

    # ---- Плюс-ветвь --------------------------------------------------------
    U_np = 0.0
    if mu > b:
        n_max_p = int(math.floor(L / two_pi * math.sqrt(mu * mu - b * b)))
        for n in range(n_max_p + 1):
            contrib = _I_n_plus(n, mu, b, L)
            U_np += contrib if n == 0 else 2.0 * contrib
    # else: mu <= b → n_max^+ = -1, U_np = 0

    # ---- Минус-ветвь -------------------------------------------------------
    U_nm = 0.0
    n_max_m = int(math.floor(L * mu / two_pi))
    for n in range(n_max_m + 1):
        kn2 = (two_pi * n / L) ** 2 if n > 0 else 0.0
        Sn = math.sqrt(mu * mu - kn2)
        contrib = _I_n_minus(n, mu, b, Sn, L)
        U_nm += contrib if n == 0 else 2.0 * contrib

    return -(2.0 / L) * (U_np + U_nm)

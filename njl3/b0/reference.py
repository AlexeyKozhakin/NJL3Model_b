"""
High-precision reference (mpmath) for the b = 0 formulas.

Direct formulas without the "stable" rewritings: with enough digits the cancellation of
logarithms near M = 0 is harmless. Used by the tests and for the accuracy study.
"""

import mpmath as mp


def _modes(mu, L):
    mu, L = mp.mpf(mu), mp.mpf(L)
    if mu <= 0:
        return []
    n_max = int(mp.floor(L * mu / (2 * mp.pi)))
    return [(2 * mp.pi * n / L, 1 if n == 0 else 2) for n in range(n_max + 1)]


def G(M, mu, L, g, dps=50):
    """G(M) with dps significant digits."""
    with mp.workdps(dps):
        M, mu, L = mp.mpf(M), mp.mpf(mu), mp.mpf(L)
        s = mp.log(1 - mp.exp(-L * M))
        for k, C in _modes(mu, L):
            m = mp.sqrt(M**2 + k**2)
            if m < mu:
                s += C * mp.acosh(mu / m)
        return 1 / mp.mpf(g) + M / mp.pi + 2 / (mp.pi * L) * s


def G0(mu, L, g, dps=50):
    """G(0⁺)."""
    with mp.workdps(dps):
        mu, L = mp.mpf(mu), mp.mpf(L)
        s = mp.log(2 * mu * L)
        for k, C in _modes(mu, L)[1:]:
            s += C * mp.acosh(mu / k)
        return 1 / mp.mpf(g) + 2 / (mp.pi * L) * s


def _I(m, mu):
    if m >= mu:
        return mp.mpf(0)
    P = mp.sqrt(mu**2 - m**2)
    if m == 0:
        return mu * P / 2
    return mu * P / 2 - m**2 / 2 * mp.log((mu + P) / m)


def omega_rel(M, mu, L, g, dps=50):
    """Omega(M) - Omega(0) from the closed forms with dps digits."""
    with mp.workdps(dps):
        M, mu, L = mp.mpf(M), mp.mpf(mu), mp.mpf(L)
        OL = lambda MM: 2 / (mp.pi * L**3) * (L * MM * mp.polylog(2, mp.exp(-L * MM)) + mp.polylog(3, mp.exp(-L * MM)))
        OmuL = lambda MM: -2 / (mp.pi * L) * mp.fsum(C * _I(mp.sqrt(MM**2 + k**2), mu) for k, C in _modes(mu, L))
        return (M**2 / (2 * mp.mpf(g)) + M**3 / (3 * mp.pi)
                + OL(M) - 2 * mp.zeta(3) / (mp.pi * L**3) + OmuL(M) - OmuL(mp.mpf(0)))


def root_near(M_guess, mu, L, g, rel_width=1e-6, dps=50):
    """Root of G near M_guess — only inside a sign-changing bracket (no jump to a neighbouring branch)."""
    with mp.workdps(dps):
        for w in (rel_width, 1e-4, 1e-3):
            a, b = mp.mpf(M_guess) * (1 - w), mp.mpf(M_guess) * (1 + w)
            if G(a, mu, L, g, dps) * G(b, mu, L, g, dps) < 0:
                return mp.findroot(lambda x: G(x, mu, L, g, dps), (a, b), solver='anderson')
    raise ValueError('G does not change sign near M_guess')

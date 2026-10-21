"""
Stability of the homogeneous solution of the GN model on R² × S¹ (phi = 0) against
inhomogeneous fluctuations eps_k(q1) e^{i q1 x1 + i 2 pi k x2 / L}.

Second variation (after the p0 integration and the gap equation, M = M*(mu, L) ≠ 0):

    Phi(k, q1) = ∫ dp1/(2 pi L) Σ_n  N_nk(p1) B(E_n, E_nk),
    N_nk = 2M² + q1 (p1 + q1) + k (2pi/L)² (n + k),
    B(E1, E2) = (θ(E1 - mu)/E1 - θ(E2 - mu)/E2) / (E2² - E1²),
    E_n = sqrt(M² + p1² + (2pi n/L)²),  E_nk = sqrt(M² + (p1 + q1)² + (2pi (n+k)/L)²).

The substitution p1 -> -p1 - q1, n -> -n - k swaps E_n and E_nk and leaves B unchanged, so the
numerator may be replaced by its symmetric part

    N = 2M² + q²/2,  q² = q1² + q2²,  q2 = 2 pi k / L,

and Phi(-k, q1) = Phi(k, q1). In vacuum (mu = 0) B = 1/(E1 E2 (E1 + E2)) > 0, so Phi > 0 there and
Phi < 0 signals an instability of the homogeneous phase.

Vacuum part (all θ = 1). With 1/(E1 E2 (E1+E2)) = (2/π)∫_0^∞ dω /((ω²+E1²)(ω²+E2²)), a Feynman
parameter x and the exact sum over n:

    Phi_vac = N/(4 pi) ∫_0^1 dx  sinh(L a) / (a (cosh(L a) - cos(2 pi k x))),  a² = M² + x(1-x) q².

Matter part. B - B_vac = (χ2/E2 - χ1/E1)/(E2² - E1²), χ = θ(mu - E); by the same symmetry both terms
contribute equally:

    Phi_mat = -(N/(pi L)) Σ_{m_n < mu} PV ∫_{-P_n}^{P_n} dp1 / ( sqrt(m_n² + p1²) (2 q1 p1 + c_nk) ),
    m_n² = M² + (2pi n/L)²,  P_n = sqrt(mu² - m_n²),  c_nk = q1² + (2pi/L)² ((n+k)² - n²).

With u = (p1 + sqrt(m² + p1²))/m the integral is elementary:
    ∫ dp1 / (E (a p1 + c)) = (1/D) ln|(u - u₊)/(u - u₋)|,  D = sqrt(c² + a² m²),  u± = (-c ± D)/(a m).

At q1 → 2 P_n (k = 0) both levels reach the Fermi surface together and Phi_mat has a logarithmic
singularity (1D nesting), Phi -> -∞.
"""

import math

from scipy.integrate import quad

PI = math.pi


def numerator(M, q1, k, L):
    """Symmetric numerator N = 2M² + (q1² + (2πk/L)²)/2."""
    q2 = 2.0 * PI * k / L
    return 2.0 * M * M + 0.5 * (q1 * q1 + q2 * q2)


def _vac_kernel(x, M, q, k, L):
    a = math.sqrt(M * M + x * (1.0 - x) * q * q)
    La = L * a
    c = math.cos(2.0 * PI * k * x)
    if La > 40.0:                                  # sinh/cosh -> 1 up to e^{-La}
        return 1.0 / (a * (1.0 - 2.0 * c * math.exp(-La)))
    return math.sinh(La) / (a * (math.cosh(La) - c))


def phi_vac(k, q1, M, L):
    """Vacuum part of Phi(k, q1)."""
    q2 = 2.0 * PI * k / L
    q = math.hypot(q1, q2)
    k = abs(k)
    pts = [j / (2.0 * k) for j in range(1, 2 * k)] if k else None
    I = quad(_vac_kernel, 0.0, 1.0, args=(M, q, k, L), points=pts, limit=200,
             epsabs=1e-13, epsrel=1e-11)[0]
    return numerator(M, q1, k, L) / (4.0 * PI) * I


def _antider(p, m, a, c):
    """(1/D) ln|(u - u₊)/(u - u₋)| — antiderivative of 1/(E (a p + c)), a > 0."""
    D = math.sqrt(c * c + a * a * m * m)
    u = (p + math.sqrt(m * m + p * p)) / m
    up, um = (-c + D) / (a * m), (-c - D) / (a * m)
    return math.log(abs((u - up) / (u - um))) / D


def channel_integral(m, P, q1, c):
    """PV ∫_{-P}^{P} dp / (sqrt(m² + p²) (2 q1 p + c))."""
    if q1 == 0.0:
        if c == 0.0:
            raise ZeroDivisionError('q1 = 0 and c = 0: use the q -> 0 limit')
        return 2.0 * math.asinh(P / m) / c
    a = 2.0 * q1
    return _antider(P, m, a, c) - _antider(-P, m, a, c)


def channels(M, mu, L):
    """Filled channels: list of (n, m_n, P_n) with m_n < mu, n ∈ Z."""
    out = []
    if mu <= M:
        return out
    n_max = int(math.floor(L * math.sqrt(mu * mu - M * M) / (2.0 * PI)))
    for n in range(-n_max, n_max + 1):
        m = math.hypot(M, 2.0 * PI * n / L)
        if m < mu:
            out.append((n, m, math.sqrt(mu * mu - m * m)))
    return out


def phi_mat(k, q1, M, mu, L):
    """Matter part of Phi(k, q1) (T = 0)."""
    w = (2.0 * PI / L) ** 2
    s = 0.0
    for n, m, P in channels(M, mu, L):
        c = q1 * q1 + w * ((n + k) ** 2 - n * n)
        s += channel_integral(m, P, q1, c)
    return -numerator(M, q1, k, L) / (PI * L) * s


def phi(k, q1, M, mu, L):
    """Phi(k, q1) = Phi_vac + Phi_mat (T = 0, M = M* ≠ 0)."""
    return phi_vac(k, q1, M, L) + phi_mat(k, q1, M, mu, L)


def singular_q1(M, mu, L, k=0):
    """Positions q1 of the logarithmic singularities: both levels on the Fermi surface."""
    w = (2.0 * PI / L) ** 2
    out = []
    for n, m, P in channels(M, mu, L):
        mk2 = M * M + w * (n + k) ** 2
        if mk2 < mu * mu:                          # p1 = ±P_n, p1 + q1 = ±P_{n+k}
            Pk = math.sqrt(mu * mu - mk2)
            for q in (P + Pk, abs(P - Pk)):
                if q > 0.0:
                    out.append(q)
    return sorted(set(round(q, 12) for q in out))


# ==============================================================================
# Reference: direct quadrature of the original (unsymmetrized) formula
# ==============================================================================

def _direct_integrand(k, q1, M, mu, L, matter):
    w = 2.0 * PI / L

    def B(E1, E2, th1, th2):
        t1 = th1 / E1
        t2 = th2 / E2
        d = E2 * E2 - E1 * E1
        if abs(d) < 1e-13 * E1 * E1:
            return None
        return (t1 - t2) / d

    def f(p, n):
        E1 = math.sqrt(M * M + p * p + (w * n) ** 2)
        E2 = math.sqrt(M * M + (p + q1) ** 2 + (w * (n + k)) ** 2)
        N = 2.0 * M * M + q1 * (p + q1) + k * w * w * (n + k)
        th1, th2 = float(E1 > mu), float(E2 > mu)
        b = B(E1, E2, th1, th2)
        if b is None:                               # E1 = E2: the limit -d/dE²(θ/E)
            b = 0.5 * th1 / E1**3
        if matter:
            b -= 1.0 / (E1 * E2 * (E1 + E2))
        return N * b
    return f


def phi_direct_mat(k, q1, M, mu, L, epsrel=1e-11):
    """Matter part straight from the original (unsymmetrized) formula: Σ_n ∫dp1 N (B - B_vac).
    The integrand has compact support (one of the levels below mu). For tests."""
    w = 2.0 * PI / L
    f = _direct_integrand(k, q1, M, mu, L, matter=True)
    ns = set()
    for n, _, _ in channels(M, mu, L):
        ns.update((n, n - k))
    tot = 0.0
    for n in sorted(ns):
        pts = []
        for nn, sh in ((n, 0.0), (n + k, q1)):
            m2 = M * M + (w * nn) ** 2
            if m2 < mu * mu:
                P = math.sqrt(mu * mu - m2)
                pts += [P - sh, -P - sh]
        pts = sorted(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if b > a:
                tot += quad(f, a, b, args=(n,), limit=400, epsrel=epsrel, epsabs=1e-15)[0]
    return tot / (2.0 * PI * L)


def phi_direct_vac(k, q1, M, L, n_cut=300):
    """Vacuum part by direct summation over |n| ≤ n_cut (p1 integral by quad) plus the continuum
    tail |n| > n_cut. For tests."""
    w = 2.0 * PI / L
    f = _direct_integrand(k, q1, M, -1.0, L, matter=False)
    tot = 0.0
    for n in range(-n_cut, n_cut + 1):
        c = -q1 / 2.0
        tot += quad(f, -math.inf, c, args=(n,), limit=200, epsrel=1e-12, epsabs=0)[0]
        tot += quad(f, c, math.inf, args=(n,), limit=200, epsrel=1e-12, epsabs=0)[0]
    # tail: Σ_{|n|>n_cut} ∫dp N/(2E³) ≈ 2 Σ_{n>n_cut} N/(w n)² -> 2N/(w² (n_cut + 1/2))
    N = numerator(M, q1, k, L)
    tot += 2.0 * N / (w * w * (n_cut + 0.5))
    return tot / (2.0 * PI * L)

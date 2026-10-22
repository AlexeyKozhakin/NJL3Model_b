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

Matter part. B - B_vac = (χ2/E2 - χ1/E1)/(E2² - E1²), χ = θ(mu - E). Shifting n -> n - k, p1 -> p1 - q1 turns
the χ2 term into the χ1 term with the level n - k instead of n + k (q1 -> -q1, which does not change the p1
integral), so

    Phi_mat = -(N/(2 pi L)) Σ_{m_n < mu} Σ_{±k} PV ∫_{-P_n}^{P_n} dp1 / ( sqrt(m_n² + p1²) (2 q1 p1 + c_n,±k) ),

(for φ ∈ {0, 1/2} the two terms are equal, n -> -n - k; for a general twist they are not),
    m_n² = M² + (2pi (n+φ)/L)²,  P_n = sqrt(mu² - m_n²),  c_n,±k = q1² + (2pi/L)² ((n+φ±k)² - (n+φ)²).

With u = (p1 + sqrt(m² + p1²))/m the integral is elementary:
    ∫ dp1 / (E (a p1 + c)) = (1/D) ln|(u - u₊)/(u - u₋)|,  D = sqrt(c² + a² m²),  u± = (-c ± D)/(a m).

At q1 → 2 P_n (k = 0) both levels reach the Fermi surface together and Phi_mat has a logarithmic
singularity (1D nesting), Phi -> -∞.

Twist φ of the boundary condition: p₂ₙ = 2π(n + φ)/L everywhere (m_n, c_nk, singular points) and
cos 2πkx → cos 2π(kx − φ) in Phi_vac. The symmetric numerator 2M² + q²/2 holds for any φ: Zhokhov's numerator
differs from it by (E_nk² − E_n²)/2, which turns the bracket into (θ_n/E_n − θ_nk/E_nk)/2, and the sums over
n of the two terms coincide (the sets n + φ and n + k + φ are the same); Phi(−k) = Phi(k).

Temperature. For a static fluctuation (q0 = 0) the partial fractions in z = iω + mu are exact, and
T Σ_ω 1/(z² - E²) = -(1 - n_F(E - mu) - n_F(E + mu))/(2E), so T > 0 amounts to
    θ(E - mu) -> 1 - n_F(E - mu) - n_F(E + mu),
and in the matter part χ(mu - E) -> w(E) = n_F(E - mu) + n_F(E + mu):
    Phi_mat(T) = -(N/(pi L)) Σ_n PV ∫ dp1 w(E_n) / (E_n (2 q1 p1 + c_nk)).
The singularity at 2 P_n is cut off at ~ ln(mu/T).
"""

import math

from scipy.integrate import quad

PI = math.pi


def numerator(M, q1, k, L):
    """Symmetric numerator N = 2M² + (q1² + (2πk/L)²)/2."""
    q2 = 2.0 * PI * k / L
    return 2.0 * M * M + 0.5 * (q1 * q1 + q2 * q2)


def _vac_kernel(x, M, q, k, L, phi=0.0):
    a = math.sqrt(M * M + x * (1.0 - x) * q * q)
    La = L * a
    c = math.cos(2.0 * PI * (k * x - phi))
    if La > 40.0:                                  # sinh/cosh -> 1 up to e^{-La}
        return 1.0 / (a * (1.0 - 2.0 * c * math.exp(-La)))
    return math.sinh(La) / (a * (math.cosh(La) - c))


def phi_vac(k, q1, M, L, phi=0.0):
    """Vacuum part of Phi(k, q1); twist phi: cos 2π(kx − φ) in the kernel."""
    q2 = 2.0 * PI * k / L
    q = math.hypot(q1, q2)
    k = abs(k)
    pts = [j / (2.0 * k) for j in range(1, 2 * k)] if k else None
    I = quad(_vac_kernel, 0.0, 1.0, args=(M, q, k, L, phi), points=pts, limit=200,
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


def channels(M, mu, L, phi=0.0):
    """Filled channels: list of (n, m_n, P_n) with m_n = sqrt(M² + (2π(n + φ)/L)²) < mu, n ∈ Z."""
    out = []
    if mu <= M:
        return out
    n_max = int(math.floor(L * math.sqrt(mu * mu - M * M) / (2.0 * PI))) + 1
    for n in range(-n_max - 1, n_max + 1):
        m = math.hypot(M, 2.0 * PI * (n + phi) / L)
        if m < mu:
            out.append((n, m, math.sqrt(mu * mu - m * m)))
    return out


def _couplings(k, phi):
    """(k', weight): both n → n ± k in general; one with weight 2 for k = 0 or φ ∈ {0, 1/2} (n → −n − k − 2φ)."""
    if k == 0 or abs(2 * phi - round(2 * phi)) < 1e-12:
        return ((k, 2.0),)
    return ((k, 1.0), (-k, 1.0))


def phi_mat(k, q1, M, mu, L, phi=0.0):
    """Matter part of Phi(k, q1) (T = 0)."""
    w = (2.0 * PI / L) ** 2
    s = 0.0
    for n, m, P in channels(M, mu, L, phi):
        nu = n + phi
        for kk, wt in _couplings(k, phi):
            c = q1 * q1 + w * ((nu + kk) ** 2 - nu * nu)
            s += channel_integral(m, P, q1, c) * wt
    return -numerator(M, q1, k, L) / (2.0 * PI * L) * s


def _n_F(x, T):
    if x > 40.0 * T:
        return math.exp(-x / T)
    if x < -40.0 * T:
        return 1.0 - math.exp(x / T)
    return 1.0 / (math.exp(x / T) + 1.0)


def _pv_line(f, a, b, p0, pts):
    """PV ∫_a^b f(p)/(p - p0) dp by subtracting the pole."""
    if not (a < p0 < b):
        br = [p for p in pts if a < p < b]
        return quad(lambda p: f(p) / (p - p0), a, b, points=br or None, limit=400,
                    epsabs=1e-14, epsrel=1e-11)[0]
    f0 = f(p0)
    br = [p for p in pts + [p0] if a < p < b]
    g = lambda p: (f(p) - f0) / (p - p0) if p != p0 else 0.0
    return (quad(g, a, b, points=br, limit=400, epsabs=1e-14, epsrel=1e-11)[0]
            + f0 * math.log((b - p0) / (p0 - a)))


def phi_mat_T(k, q1, M, mu, L, T, phi=0.0):
    """Matter part of Phi(k, q1) at temperature T (T = 0: closed form)."""
    if T == 0.0:
        return phi_mat(k, q1, M, mu, L, phi)
    w2 = (2.0 * PI / L) ** 2
    E_max = mu + 40.0 * T
    s = 0.0
    if E_max <= M:
        return 0.0
    n_max = int(math.floor(math.sqrt(E_max * E_max - M * M) * L / (2.0 * PI))) + 1
    for n in range(-n_max - 1, n_max + 1):
        nu = n + phi
        m = math.sqrt(M * M + w2 * nu * nu)
        if m >= E_max:
            continue
        pm = math.sqrt(E_max * E_max - m * m)
        pts = [-math.sqrt(mu * mu - m * m), math.sqrt(mu * mu - m * m)] if m < mu else []

        def f(p, m=m):
            E = math.sqrt(m * m + p * p)
            return (_n_F(E - mu, T) + _n_F(E + mu, T)) / E

        for kk, wt in _couplings(k, phi):
            c = q1 * q1 + w2 * ((nu + kk) ** 2 - nu * nu)
            if q1 == 0.0:
                s += wt * quad(f, -pm, pm, points=pts or None, limit=400, epsabs=1e-14, epsrel=1e-11)[0] / c
            else:
                s += wt * _pv_line(f, -pm, pm, -c / (2.0 * q1), pts) / (2.0 * q1)
    return -numerator(M, q1, k, L) / (2.0 * PI * L) * s


def phi(k, q1, M, mu, L, T=0.0, phi=0.0):
    """Phi(k, q1) = Phi_vac + Phi_mat (M = M* ≠ 0); twist phi of the boundary condition."""
    return phi_vac(k, q1, M, L, phi) + phi_mat_T(k, q1, M, mu, L, T, phi)


_phi_total = phi          # the function, for callers whose parameter `phi` (the twist) shadows it


def singular_q1(M, mu, L, k=0, phi=0.0):
    """Positions q1 of the logarithmic singularities: both levels on the Fermi surface."""
    w = (2.0 * PI / L) ** 2
    out = []
    for n, m, P in channels(M, mu, L, phi):
        for kk in {k, -k}:                         # levels n and n ± k (both couplings for φ ∉ {0, ½})
            mk2 = M * M + w * (n + phi + kk) ** 2
            if mk2 < mu * mu:                      # p1 = ±P_n, p1 + q1 = ±P_{n±k}
                Pk = math.sqrt(mu * mu - mk2)
                for q in (P + Pk, abs(P - Pk)):
                    if q > 0.0:
                        out.append(q)
    return sorted(set(round(q, 12) for q in out))


# ==============================================================================
# Reference: direct quadrature of the original (unsymmetrized) formula
# ==============================================================================

def _direct_integrand(k, q1, M, mu, L, matter, phi=0.0):
    w = 2.0 * PI / L

    def B(E1, E2, th1, th2):
        t1 = th1 / E1
        t2 = th2 / E2
        d = E2 * E2 - E1 * E1
        if abs(d) < 1e-13 * E1 * E1:
            return None
        return (t1 - t2) / d

    def f(p, n):
        E1 = math.sqrt(M * M + p * p + (w * (n + phi)) ** 2)
        E2 = math.sqrt(M * M + (p + q1) ** 2 + (w * (n + phi + k)) ** 2)
        N = 2.0 * M * M + q1 * (p + q1) + k * w * w * (n + phi + k)
        th1, th2 = float(E1 > mu), float(E2 > mu)
        b = B(E1, E2, th1, th2)
        if b is None:                               # E1 = E2: the limit -d/dE²(θ/E)
            b = 0.5 * th1 / E1**3
        if matter:
            b -= 1.0 / (E1 * E2 * (E1 + E2))
        return N * b
    return f


def phi_direct_mat(k, q1, M, mu, L, epsrel=1e-11, phi=0.0):
    """Matter part straight from the original (unsymmetrized) formula: Σ_n ∫dp1 N (B - B_vac).
    The integrand has compact support (one of the levels below mu). For tests."""
    w = 2.0 * PI / L
    f = _direct_integrand(k, q1, M, mu, L, matter=True, phi=phi)
    ns = set()
    for n, _, _ in channels(M, mu, L, phi):
        ns.update((n, n - k))
    tot = 0.0
    for n in sorted(ns):
        pts = []
        for nn, sh in ((n, 0.0), (n + k, q1)):
            m2 = M * M + (w * (nn + phi)) ** 2
            if m2 < mu * mu:
                P = math.sqrt(mu * mu - m2)
                pts += [P - sh, -P - sh]
        pts = sorted(pts)
        for a, b in zip(pts[:-1], pts[1:]):
            if b > a:
                tot += quad(f, a, b, args=(n,), limit=400, epsrel=epsrel, epsabs=1e-15)[0]
    return tot / (2.0 * PI * L)


def phi_direct_vac(k, q1, M, L, n_cut=300, phi=0.0):
    """Vacuum part by direct summation over |n| ≤ n_cut (p1 integral by quad) plus the continuum
    tail |n| > n_cut. For tests."""
    w = 2.0 * PI / L
    f = _direct_integrand(k, q1, M, -1.0, L, matter=False, phi=phi)
    tot = 0.0
    for n in range(-n_cut, n_cut + 1):
        c = -q1 / 2.0
        tot += quad(f, -math.inf, c, args=(n,), limit=200, epsrel=1e-12, epsabs=0)[0]
        tot += quad(f, c, math.inf, args=(n,), limit=200, epsrel=1e-12, epsabs=0)[0]
    # tail: Σ_{|n|>n_cut} ∫dp N/(2E³) ≈ 2 Σ_{n>n_cut} N/(w n)² -> 2N/(w² (n_cut + 1/2))
    N = numerator(M, q1, k, L)
    tot += 2.0 * N / (w * w * (n_cut + 0.5))
    return tot / (2.0 * PI * L)


# ==============================================================================
# Vectorized over q1 (T = 0) — for scans and maps
# ==============================================================================

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def phi_vac_vec(k, q1, M, L, n_sub=None, phi=0.0):
    """Phi_vac(k, q1) for an array q1: composite Gauss–Legendre in x (k half-periods resolved)."""
    q1 = np.asarray(q1, float)
    k = abs(int(k))
    n_sub = n_sub or max(4, 4 * k)
    edges = np.linspace(0.0, 1.0, n_sub + 1)
    if M < 1e-2:
        # small mass: peaks of width ~M²/q² at x = 0, 1 (massless modes) — geometric grading to the ends
        g = np.geomspace(1e-16, 0.5 / n_sub, 48)
        edges = np.unique(np.concatenate([edges, g, 1.0 - g]))
    h = 0.5 * (edges[1:] - edges[:-1])
    x = (0.5 * (edges[1:] + edges[:-1])[:, None] + h[:, None] * _GL_X[None, :]).ravel()
    wx = (h[:, None] * _GL_W[None, :]).ravel()
    q2 = (2.0 * PI * k / L) ** 2
    a = np.sqrt(M * M + np.outer(q1 * q1 + q2, x * (1.0 - x)))
    La = L * a
    c = np.cos(2.0 * PI * (k * x - phi))[None, :]
    ex = np.exp(-np.minimum(La, 700.0))
    ker = (1.0 - ex * ex) / (a * (1.0 + ex * ex - 2.0 * c * ex))      # sinh/(cosh - c), overflow-safe
    N = 2.0 * M * M + 0.5 * (q1 * q1 + q2)
    return N / (4.0 * PI) * (ker @ wx)


def phi_mat_vec(k, q1, M, mu, L, phi=0.0):
    """Phi_mat(k, q1) at T = 0 for an array q1 > 0 (closed form)."""
    q1 = np.asarray(q1, float)
    w2 = (2.0 * PI / L) ** 2
    s = np.zeros_like(q1)
    a = 2.0 * q1
    for n, m, P in channels(M, mu, L, phi):
        nu = n + phi
        uP, uM = (P + mu) / m, (mu - P) / m                    # u(±P), sqrt(m² + P²) = mu
        for kk, wt in _couplings(k, phi):
            c = q1 * q1 + w2 * ((nu + kk) ** 2 - nu * nu)
            D = np.sqrt(c * c + a * a * m * m)
            up, um = (-c + D) / (a * m), (-c - D) / (a * m)
            with np.errstate(divide='ignore'):
                s += wt * (np.log(np.abs((uP - up) / (uP - um)))
                                            - np.log(np.abs((uM - up) / (uM - um)))) / D
    N = 2.0 * M * M + 0.5 * (q1 * q1 + w2 * k * k)
    return -N / (2.0 * PI * L) * s


def phi_vec(k, q1, M, mu, L, phi=0.0):
    """Phi(k, q1) at T = 0 for an array q1 > 0."""
    return phi_vac_vec(k, q1, M, L, phi=phi) + phi_mat_vec(k, q1, M, mu, L, phi)


N_DIPS = 3        # finite-T evaluations at the deepest singular dips only


def scan_min(M, mu, L, T=1e-3, q_max=10.0, n_q=400, k_max=None, phi=0.0):
    """
    min over k ≥ 0 and q1 ∈ (0, q_max] of Phi at temperature T. The T = 0 closed form on a q-grid is
    accurate away from the singular points q_s; at every q_s ≤ q_max Phi is evaluated at T exactly
    (there the T = 0 value is -∞). Returns dict(phi_min, k, q1, singular: bool).
    """
    k_max = k_max if k_max is not None else int(q_max * L / (2.0 * PI))
    q = np.linspace(q_max / n_q, q_max, n_q)
    best = dict(phi_min=math.inf, k=None, q1=None, singular=False)
    dips = []
    for k in range(0, k_max + 1):
        v = phi_vec(k, q, M, mu, L, phi)
        qs = [x for x in singular_q1(M, mu, L, k, phi) if 0.0 < x <= q_max]
        if qs:   # drop grid points within the thermal width of a singularity
            near = np.min(np.abs(q[:, None] - np.array(qs)[None, :]), axis=1) < 50.0 * T
            v = np.where(near, np.inf, v)
        i = int(np.argmin(v))
        if v[i] < best['phi_min']:
            best = dict(phi_min=float(v[i]), k=k, q1=float(q[i]), singular=False)
        if T > 0.0:
            for x in qs:   # T = 0 value at a distance ~T from the singularity: ranks the dips (log cut-off)
                pr = float(np.min(phi_vec(k, np.array([x - 3 * T, x + 3 * T]), M, mu, L, phi)))
                dips.append((pr, k, x))
    if T > 0.0:
        for _, k, x in sorted(dips)[:N_DIPS]:
            val = _phi_total(k, x, M, mu, L, T, phi)
            if val < best['phi_min']:
                best = dict(phi_min=val, k=k, q1=x, singular=True)
    return best


# ==============================================================================
# The symmetric phase M* = 0
# ==============================================================================
#
# Without the gap equation (M = 0), the second variation of Zhokhov (regulator mass m -> 0) splits, with
# p·p' = (E_n² + E_nk² - q²)/2, α = ½ + q²/(2(E_n² - E_nk²)), into
#   - "tadpoles": 1/(2g) + (1/(πL)) ln(1 - e^{-Lm}) + (1/(πL)) Σ_n arccosh(mu/m_n) = ½ G(m) - m/(2π) -> ½ G(0⁺)
#     (the homogeneous gap function of paper 1 / njl3.hom: ln(Lm) and ln(2mu/m) cancel), and
#   - the loop with the numerator q²/2 = Phi_vac + Phi_mat of the M ≠ 0 formula at M -> 0 (2M² + q²/2 -> q²/2;
#     its ln(1/m) from the vacuum loop and from the zero mode of the matter part cancel),
#     Phi_{M=0}(k, q1) = ½ G(0⁺; mu, L) + [Phi_vac + Phi_mat](k, q1; M -> 0),   Phi -> ½ G(0⁺) as q -> 0.
# Checked against Zhokhov's formula taken literally with the regulator (with the matter term multiplied by ½,
# i.e. prefactor 1/L instead of 2/L — otherwise the q -> 0 limit is not ½ G(0⁺)): phi_zero_direct.

M_ZERO = 1e-4            # regulator mass: m-dependence ~1e-8 (1e-3 vs 1e-4); smaller m loses quadrature accuracy (peaks of width m²/q²)
                         # (much smaller m needs the Feynman-x peaks of width m²/q² at x = 0, 1 resolved)


def half_G0(mu, L, g, phi=0.0):
    from njl3.hom.formulas import gap_G0
    return 0.5 * gap_G0(mu, L, g, phi)


def phi_zero(k, q1, mu, L, g, T=0.0, phi=0.0, m_reg=M_ZERO):
    """Phi(k, q1) around the symmetric solution M = 0."""
    return half_G0(mu, L, g, phi) + _phi_total(k, q1, m_reg, mu, L, T, phi)


def scan_min_zero(mu, L, g, T=1e-3, phi=0.0, m_reg=M_ZERO, **kw):
    """scan_min for M = 0: the loop part at M = m_reg plus the constant ½ G(0⁺)."""
    r = scan_min(m_reg, mu, L, T=T, phi=phi, **kw)
    r = dict(r)
    r['phi_min'] += half_G0(mu, L, g, phi)
    return r


def phi_zero_direct(q1, k, mu, L, g, m, matter_factor=0.5):
    """Zhokhov's M = 0 formula taken literally (φ = 0, regulator mass m) — for tests. matter_factor = ½ is the
    correction found here (prefactor 1/L of the matter term); matter_factor = 1 is the formula as written."""
    w = 2.0 * PI / L
    q2 = q1 * q1 + (w * k) ** 2
    vac = phi_vac(k, q1, m, L) * (0.5 * q2) / numerator(m, q1, k, L)
    cas = math.log(-math.expm1(-L * m)) / (PI * L)
    nmax = int(L * mu / (2 * PI)) + 2
    mat = 0.0
    for n in range(-nmax - abs(k) - 1, nmax + abs(k) + 2):
        a2, b2 = w * n, w * (n + k)
        if math.hypot(m, a2) >= mu and math.hypot(m, b2) >= mu:
            continue

        def f(p, a2=a2, b2=b2):
            En = math.sqrt(p * p + a2 * a2 + m * m)
            Ek = math.sqrt((p + q1) ** 2 + b2 * b2 + m * m)
            pp = p * (p + q1) + a2 * b2
            d = En * En - Ek * Ek
            if abs(d) < 1e-12:
                return 0.0
            return (En * En - pp) / d * float(En < mu) / En + (Ek * Ek - pp) / (-d) * float(Ek < mu) / Ek
        pts = {0.0, -q1, -q1 / 2}
        for s in (1, -1):
            pts.add(s * math.sqrt(max(mu * mu - a2 * a2 - m * m, 0.0)))
            pts.add(s * math.sqrt(max(mu * mu - b2 * b2 - m * m, 0.0)) - q1)
        pts = sorted(pts)
        lo, hi = pts[0] - 1e-9, pts[-1] + 1e-9
        edges = [lo] + [x for x in pts if lo < x < hi] + [hi]
        for x0, x1 in zip(edges[:-1], edges[1:]):
            if x1 > x0:
                mat += quad(f, x0, x1, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    mat *= 2.0 / L / (2.0 * PI)
    return 1.0 / (2.0 * g) + vac + cas + matter_factor * mat

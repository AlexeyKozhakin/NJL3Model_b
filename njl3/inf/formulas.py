"""
Closed forms and 1D integrals for NJL3 at L -> infinity.

b = 0. The two branches coincide, E = sqrt(M² + p²) in the (p1, p2) plane, and
    W(M) = -(1/π) ∫_M^mu E (mu - E) dE = -(mu - M)² (mu + 2M) / (6π)   (M < mu),  0 (M ≥ mu),
    ΔV(M) = M³ / (3π).
For M ≤ mu the cubic terms cancel and the potential is purely quadratic:
    Omega(M) - Omega(0) = M² [1/(2g) + mu/(2π)],
and the gap function G = (1/M) dOmega/dM is constant there:
        G(M) = 1/g + mu/π   (M < mu),   1/g + M/π   (M ≥ mu).

b ≠ 0. The p2 integral is elementary,
    J(A) = ∫ (mu - sqrt(A² + q²))₊ dq = mu P - A² ln((mu + P)/|A|),  P = sqrt(mu² - A²)  (|A| < mu),
so with A± = E ± b and E = sqrt(M² + p²)
    W(M, b) = -(1/(2π²)) Σ± ∫_0^∞ dp J(A±(p)).
W(0, b) = W(0, 0) = -mu³/(6π) (at M = 0, b only shifts the momentum). The vacuum term ΔV is the same as
ΔU on R² × S¹ (closed form njl3.full.formulas.dU).
"""

import math

from scipy.integrate import quad

PI = math.pi


def W_b0(M, mu):
    """W_mu(M, b = 0)."""
    if mu <= 0.0 or M >= mu:
        return 0.0
    return -(mu - M)**2 * (mu + 2.0 * M) / (6.0 * PI)


def omega_rel_b0(M, mu, g):
    """Omega(M, 0) - Omega(0, 0)."""
    return M * M / (2.0 * g) + M**3 / (3.0 * PI) + W_b0(M, mu) - W_b0(0.0, mu)


def gap_G_b0(M, mu, g):
    """Gap function G(M) at b = 0 (dOmega/dM = M G)."""
    return 1.0 / g + (mu if M < mu else M) / PI


def M_vacuum(g):
    """Vacuum mass (mu = 0): the root of 1/g + M/π = 0."""
    return -PI / g


# ==============================================================================
# b ≠ 0
# ==============================================================================

def J(A, mu):
    """∫ (mu - sqrt(A² + q²))₊ dq over q ∈ R."""
    a = abs(A)
    if a >= mu:
        return 0.0
    P = math.sqrt(mu * mu - a * a)
    if a == 0.0:
        return mu * P
    return mu * P - a * a * math.log((mu + P) / a)


def _p_of_E(E, M):
    return math.sqrt(max(E * E - M * M, 0.0))


def W(M, b, mu, epsabs=1e-13, epsrel=1e-12):
    """W_mu(M, b) (b ≥ 0) by adaptive quadrature in p with the kinks as break points."""
    if mu <= 0.0:
        return 0.0
    tot = 0.0
    # plus branch: E + b < mu
    if mu - b > M:
        hi = _p_of_E(mu - b, M)
        tot += quad(lambda p: J(math.sqrt(M * M + p * p) + b, mu), 0.0, hi, epsabs=epsabs, epsrel=epsrel,
                    limit=200)[0]
    # minus branch: |E - b| < mu, kink of A² ln|A| at E = b
    lo, hi = _p_of_E(max(M, b - mu), M), _p_of_E(b + mu, M)
    if hi > lo:
        pts = [q for q in (_p_of_E(b, M),) if b > M and lo < q < hi]
        tot += quad(lambda p: J(math.sqrt(M * M + p * p) - b, mu), lo, hi, points=pts or None, epsabs=epsabs,
                    epsrel=epsrel, limit=200)[0]
    return -tot / (2.0 * PI * PI)


def omega_rel(M, b, mu, g):
    """Omega(M, b) - Omega(0, 0)."""
    from njl3.full.formulas import dU
    if M == 0.0:
        return 0.0
    return M * M / (2.0 * g) + dU(M, b) + W(M, b, mu) - W_b0(0.0, mu)


# ==============================================================================
# Exact representation through the Hilbert transform (proof of the b-independence)
# ==============================================================================
#
# With a = ±E + b the matter term is a convolution, W = -(1/2π²) ∫ w̃(a - b) J(a) da,
# w̃(x) = |x|/sqrt(x² - M²) θ(|x| > M). Writing w̃ - 1 = (1/π) PV∫_0^π M cos φ /(x - M cos φ) dφ gives
#     W(M, b) = -mu³/(6π) - (M/(2π³)) ∫_0^π cos φ U(b + M cos φ) dφ,
# where U(c) = PV∫ J(a)/(a - c) da is the Hilbert transform of J:
#     U(c) = -2π mu c + (π²/2) c|c|                                          (|c| ≤ mu),
#     U(c) = -2π mu c + π mu sgn(c) sqrt(c² - mu²) + π c|c| arcsin(mu/|c|)    (|c| > mu).
# If b + M ≤ mu every argument stays in [-mu, mu], and with (πM/2)∫cos φ c|c| dφ = 2π² ΔU(M, b):
#     Omega(M, b) - Omega(0, 0) = M² [1/(2g) + mu/(2π)]       (b + M ≤ mu)  — independent of b.
# In general Omega(M, b) - Omega(0, 0) = M² [1/(2g) + mu/(2π)] + C(M, b), C ≥ 0, C nondecreasing in b.

def hilbert_J(c, mu):
    """U(c) = PV ∫ J(a)/(a - c) da."""
    a = abs(c)
    if a <= mu:
        return -2.0 * PI * mu * c + 0.5 * PI * PI * c * a
    return (-2.0 * PI * mu * c + PI * mu * math.copysign(1.0, c) * math.sqrt(c * c - mu * mu)
            + PI * c * a * math.asin(mu / a))


def _phi_kinks(M, b, values):
    out = []
    for v in values:
        t = (v - b) / M
        if -1.0 < t < 1.0:
            out.append(math.acos(t))
    return sorted(out)


def W_hilbert(M, b, mu):
    """W_mu(M, b) from the Hilbert-transform representation (smooth integrand in φ)."""
    if M == 0.0:
        return W_b0(0.0, mu)
    pts = _phi_kinks(M, b, (0.0, mu, -mu))
    T = quad(lambda f: math.cos(f) * hilbert_J(b + M * math.cos(f), mu), 0.0, PI, points=pts or None,
             limit=400, epsabs=1e-14, epsrel=1e-13)[0] * M / PI
    return -mu**3 / (6.0 * PI) - T / (2.0 * PI * PI)


def excess_C(M, b, mu):
    """C(M, b) = Omega(M, b) - Omega(0, 0) - M² [1/(2g) + mu/(2π)] ≥ 0; zero iff b + M ≤ mu."""
    if M == 0.0 or b + M <= mu:
        return 0.0

    def delta(c):                                   # U(c) minus its |c| ≤ mu form
        a = abs(c)
        if a <= mu:
            return 0.0
        return (1.0 if c > 0 else -1.0) * PI * (mu * math.sqrt(c * c - mu * mu) - c * c * math.acos(mu / a))
    pts = _phi_kinks(M, b, (mu, -mu))
    I = quad(lambda f: math.cos(f) * delta(b + M * math.cos(f)), 0.0, PI, points=pts or None, limit=400,
             epsabs=1e-14, epsrel=1e-13)[0]
    return -M * I / (2.0 * PI**3)

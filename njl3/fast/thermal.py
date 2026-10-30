"""
Omega(M, b) on R² × S¹ at temperature T > 0 (formula (A), publication note on the T-formula):

    Omega_T = M²/(2g) + ΔV + Omega_L|_{T=0} + W_T,
    W_T(mu) = -(T/L) Σ_n ∫ dp1/(2π) Σ± Σ_{s=±1} ln(1 + exp(-(E_{n,±} - s mu)/T)).

The vacuum terms do not depend on T. W_T is the T = 0 matter term W_0 (omega_muL of njl3.inh / njl3.fast.nb)
smeared over the chemical potential (exact, level by level: T ln(1 + e^{(nu-E)/T}) = ∫ (nu'-E)₊ K_T(nu'-nu) dnu'):

    W_T(mu) = ∫_0^∞ [K_T(nu - mu) + K_T(nu + mu)] W_0(nu) dnu,     K_T(x) = 1/(4T cosh²(x/2T)),  ∫ K_T = 1.

Quadrature: double exponential in x = nu ∓ mu, x = T sinh(π/2 sinh t), t uniform on [-T_MAX, T_MAX] (N_DE nodes,
symmetric — the linear term of W_0 cancels exactly). Nodes and weights are dimensionless (x = T s_j); against direct
integration of the Fermi–Dirac logarithms: ~1e-11 at 81 nodes (Gauss–Legendre in u = tanh(x/2T): 3e-7 at 64 nodes,
because of the log endpoint singularity).
"""

import math

import numpy as np
from numba import njit

from njl3.fast import nb
from njl3.inh import formulas as F

PI = math.pi
N_DE, T_MAX = 81, 3.2
_t = np.linspace(-T_MAX, T_MAX, N_DE)
_s = np.sinh(0.5 * np.pi * np.sinh(_t))
_ds = 0.5 * np.pi * np.cosh(_t) * np.cosh(0.5 * np.pi * np.sinh(_t))
_w = _ds * (_t[1] - _t[0]) / (4.0 * np.cosh(np.minimum(np.abs(_s) / 2.0, 350.0)) ** 2)
_keep = np.abs(_s) < 45.0                       # the kernel ~ e^{-|s|}: weights below ~1e-19 are dropped
UX = np.ascontiguousarray(_s[_keep])                                                  # x = T * UX
UW = np.ascontiguousarray(_w[_keep])


@njit(cache=True)
def W0_nb(M, b, nu, L, levk, levC, TSX, TSW):
    """T = 0 matter term at chemical potential nu (levels with k < nu only); 0 for nu <= 0."""
    if nu <= 0.0:
        return 0.0
    tot = 0.0
    for i in range(levk.shape[0]):
        k = levk[i]
        if k >= nu:
            continue
        if k == 0.0:
            tot += levC[i] * nb.mode0_nb(nu, b, M)
        else:
            tot += levC[i] * nb.modeK_nb(M, b, nu, k, TSX, TSW)
    return -tot / (PI * L)


@njit(cache=True)
def WT_nb(M, b, mu, T, L, levk, levC, UX, UW, TSX, TSW):
    """Matter term at temperature T (smearing of W0 over the chemical potential)."""
    s = 0.0
    for j in range(UX.shape[0]):
        x = T * UX[j]
        s += UW[j] * (W0_nb(M, b, mu + x, L, levk, levC, TSX, TSW) + W0_nb(M, b, -mu + x, L, levk, levC, TSX, TSW))
    return s


@njit(cache=True)
def omegaT_nb(M, b, mu, L, g, T, zL0, wT0, levk, levC, edges, coef, mid, half, phi0, x_min, s0,
              UX, UW, TSX, TSW, GTX, GTW, CHX, CHW):
    """Omega_T(M, b) - Omega_T(0, 0); zL0 = Omega_L(0, 0), wT0 = W_T(0, 0)."""
    if M == 0.0:
        return 0.0
    if b < 0.0:
        return 1e3
    return (M * M / (2 * g) + nb.dU_nb(M, b)
            + nb.omega_L_nb(M, b, L, edges, coef, mid, half, phi0, x_min, s0, TSX, TSW, GTX, GTW, CHX, CHW) - zL0
            + WT_nb(M, b, mu, T, L, levk, levC, UX, UW, TSX, TSW) - wT0)


class CtxT:
    """Everything omegaT_nb needs at a point (mu, L, g, phi, T)."""

    def __init__(self, mu, L, g, phi, T):
        self.mu, self.L, self.g, self.phi, self.T = mu, L, g, phi, T
        numax = mu + T * float(UX[-1]) + 1e-12
        lv = F.levels(numax, L, phi)
        self.levk = np.array([k for k, _ in lv], dtype=float)
        self.levC = np.array([float(c) for _, c in lv], dtype=float)
        self.tab = nb.table(phi)
        e, c, m, h, p0, xm, s0 = self.tab
        self.zL0 = nb.omega_L_nb(0.0, 0.0, L, e, c, m, h, p0, xm, s0, nb.TSX, nb.TSW, nb.GTX, nb.GTW, nb.CHX, nb.CHW)
        self.wT0 = WT_nb(0.0, 0.0, mu, T, L, self.levk, self.levC, UX, UW, nb.TSX, nb.TSW)
        self.args = (mu, L, g, T, self.zL0, self.wT0, self.levk, self.levC, e, c, m, h, p0, xm, s0,
                     UX, UW, nb.TSX, nb.TSW, nb.GTX, nb.GTW, nb.CHX, nb.CHW)

    def omega(self, M, b):
        return omegaT_nb(M, b, *self.args)

    def W(self, M, b):
        return WT_nb(M, b, self.mu, self.T, self.L, self.levk, self.levC, UX, UW, nb.TSX, nb.TSW)

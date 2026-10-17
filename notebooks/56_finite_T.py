"""
Конечная температура: при какой T исчезает щель M* и как T_c связана с M*(T = 0).

При T > 0 в Ω_μL ступенька заполнения заменяется распределением Ферми:
    (mu − ε) θ(mu − ε)  →  T ln(1 + e^{(mu − ε)/T}) + T ln(1 + e^{−(mu + ε)/T}).
Оба слагаемых — свёртка нулевой температуры по mu' с ядром w_T(x) = 1/(4T ch²(x/2T))
(вторая производная по mu у обеих сторон — w_T и δ соответственно), поэтому
    Ω_μL(M, b; mu, T) = ∫ dmu' w_T(mu − mu') Ω_μL(M, b; mu', 0) + (то же для −mu),
и проверенный код T = 0 переиспользуется без изменений. ΔU и Ω_L от T не зависят.

Что считается для каждой точки (mu, L):
  1. контроль свёртки прямым тепловым интегралом (scipy.quad) и сходимости по шагу;
  2. T_c^(2) — где коэффициент a(T) = min_b Ω/M² (M ≪ T) меняет знак
     (порог неустойчивости M = 0; при переходе 2-го рода это и есть T_c);
  3. полный минимум M*(T) по (M, b) при нескольких T/T_c^(2): непрерывен ли переход.

Запуск:  python notebooks/56_finite_T.py
"""

import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full import formulas as F
from njl3.full.minimize import find_minimum, thresholds

PI = math.pi
G = -1.0
XMAX = 34.0                       # ядро ~ e^{−|x|}: хвост < 1e-14


def kernel_nodes(dx):
    x = np.arange(-XMAX, XMAX + 1e-12, dx)
    w = dx / (4.0 * np.cosh(x / 2) ** 2)
    w[0] *= 0.5; w[-1] *= 0.5                             # трапеция (ядро гладкое, хвосты ~0)
    return x, w


def d_muL(M, b, mu, L):
    return F.omega_muL_vec(M, b, mu, L) - F.omega_muL_vec(np.zeros_like(b), b, mu, L)


def omega_rel_T(M, b, mu, L, T, OL0=None, dx=0.1):
    """Ω(M, b; T) − Ω(0, 0; T) для массивов M, b."""
    M, b = np.broadcast_arrays(np.asarray(M, float), np.asarray(b, float))
    if OL0 is None:
        OL0 = F.omega_L_vec(M, b, L) - F.omega_L_vec(np.zeros_like(b), b, L)
    base = M * M / (2 * G) + np.vectorize(F.dU)(M, b) + OL0
    if T == 0:
        return base + d_muL(M, b, mu, L).reshape(M.shape)
    x, w = kernel_nodes(dx)
    th = np.zeros(M.shape)
    for s in (+1.0, -1.0):                                 # частицы и античастицы
        for xi, wi in zip(x, w):
            m = s * mu + T * xi
            if m > 0:
                th = th + wi * d_muL(M, b, m, L).reshape(M.shape)
    return base + th


# ------------------------------------------------------------------------------
# 1. Контроль: прямой тепловой интеграл
# ------------------------------------------------------------------------------

def _phiT(eps, mu, T):
    return T * (np.logaddexp(0.0, (mu - eps) / T) + np.logaddexp(0.0, -(mu + eps) / T))


def d_muL_T_direct(M, b, mu, L, T):
    """Ω_μL(M, b; T) − Ω_μL(0, b; T) прямым интегрированием по p (scipy.quad)."""
    tot = 0.0
    nmax = int((mu + 40 * T) * L / (2 * PI))
    for n in range(nmax + 1):
        k, C = 2 * PI * n / L, (1 if n == 0 else 2)
        for sg in (+1.0, -1.0):
            def f(p):
                A1 = math.sqrt(M * M + p * p) + sg * b
                A0 = abs(p) + sg * b
                return _phiT(math.hypot(A1, k), mu, T) - _phiT(math.hypot(A0, k), mu, T)
            pmax = b + mu + 50 * T + 5
            # изломы: A = 0 (E = b) и поверхности Ферми A = ±S_n (E = ±S − sg·b), для M и для 0
            S = math.sqrt(max(mu * mu - k * k, 0.0))
            Es = [b] + [e for e in (S - sg * b, -S - sg * b) if e > 0]
            pts = sorted({q for E in Es for q in (E, math.sqrt(max(E * E - M * M, 0.0))) if 0 < q < pmax})
            tot += C * quad(f, 0, pmax, points=pts, limit=400, epsabs=1e-14, epsrel=1e-11)[0]
    return -tot / (PI * L)


def check_convolution(mu, L):
    M, T = 0.1, 0.05
    S = [s for _, s, _ in thresholds(mu, L)]
    for b in (0.0, S[-1], S[0] - 0.3):
        conv = [float(omega_rel_T(M, b, mu, L, T, OL0=0.0, dx=dx) - (M * M / (2 * G) + F.dU(M, b)))
                for dx in (0.2, 0.1)]
        direct = d_muL_T_direct(M, b, mu, L, T)
        print(f"   контроль свёртки M={M}, T={T}, b={b:.4f}: dx=0.2 {conv[0]:+.10f}, dx=0.1 {conv[1]:+.10f}, "
              f"прямо {direct:+.10f}")


# ------------------------------------------------------------------------------
# 2. Порог неустойчивости M = 0: a(T) = min_b Ω/M²
# ------------------------------------------------------------------------------

DX_A = 0.05                       # шаг свёртки для коэффициента (особенности ширины M ≪ T)


def a_min(mu, L, T, ret_b=False):
    Mc = 0.05 * T
    bs = np.concatenate([np.linspace(0.0, mu + 1.0, 241)]
                        + [S + T * np.linspace(-12, 12, 49) for _, S, _ in thresholds(mu, L)])
    bs = bs[bs >= 0]
    a = omega_rel_T(np.full_like(bs, Mc), bs, mu, L, T, dx=DX_A) / Mc**2
    i = int(np.argmin(a))
    lo, hi = max(bs[i] - 1.5 * T - 0.02, 0.0), bs[i] + 1.5 * T + 0.02
    r = minimize_scalar(lambda bb: float(omega_rel_T(Mc, bb, mu, L, T, dx=DX_A)[()]) / Mc**2, bounds=(lo, hi),
                        method='bounded', options={'xatol': 1e-3 * T})
    return (r.fun, r.x) if ret_b else r.fun


def Tc2(mu, L, T_lo, T_hi):
    """Бисекция по ln T: a(T_lo) < 0 < a(T_hi)."""
    for _ in range(40):
        Tm = math.sqrt(T_lo * T_hi)
        if a_min(mu, L, Tm) < 0:
            T_lo = Tm
        else:
            T_hi = Tm
        if T_hi / T_lo < 1 + 1e-3:
            break
    return math.sqrt(T_lo * T_hi)


# ------------------------------------------------------------------------------
# 3. Полный минимум по (M, b) при температуре T
# ------------------------------------------------------------------------------

def full_min_T(args):
    mu, L, T, M_scale = args
    Ms = np.geomspace(1e-3 * M_scale, 5.0, 45)
    bs = np.concatenate([np.linspace(0.0, mu + 1.0, 121)]
                        + [S + np.linspace(-6, 6, 41) * max(T, M_scale) for _, S, _ in thresholds(mu, L)])
    bs = np.unique(bs[bs >= 0])
    Mg, bg = np.meshgrid(Ms, bs, indexing='ij')
    Om = omega_rel_T(Mg, bg, mu, L, T)
    best = (0.0, 0.0, 0.0)
    order = np.argsort(Om.ravel())[:2]
    for idx in order:
        i, j = np.unravel_index(idx, Om.shape)
        f = lambda z: float(omega_rel_T(math.exp(min(z[0], 2.0)), abs(z[1]), mu, L, T)[()])
        r = minimize(f, [math.log(Ms[i]), bs[j]], method='Nelder-Mead',
                     options={'xatol': 1e-7, 'fatol': 1e-16, 'maxiter': 300,
                              'initial_simplex': [[math.log(Ms[i]), bs[j]],
                                                  [math.log(Ms[i]) + 0.3, bs[j]],
                                                  [math.log(Ms[i]), bs[j] + 0.3 * max(T, Ms[i])]]})
        if r.fun < best[0]:
            best = (r.fun, math.exp(min(r.x[0], 2.0)), abs(r.x[1]))
    return T, best


# ------------------------------------------------------------------------------

POINTS = [  # (mu, L) — из карт; фаза и M* при T = 0 пересчитываются
    (4.291, 0.939),
    (5.469, 1.107),
    (5.351, 2.785),
    (6.058, 3.456),
    (5.233, 6.308),
    (6.647, 5.134),
]


def point_Tc(args):
    k, mu, L = args
    r0 = find_minimum(mu, L, G)
    M0 = r0.M_star
    T_lo, T_hi = 0.05 * M0, 2.0 * M0
    while a_min(mu, L, T_lo) >= 0:
        T_lo /= 3
    while a_min(mu, L, T_hi) < 0:
        T_hi *= 2
    tc = Tc2(mu, L, T_lo, T_hi)
    _, bc = a_min(mu, L, tc, ret_b=True)
    return dict(mu=mu, L=L, phase=r0.phase, M0=M0, b0=r0.b_star, asym=r0.asymptotic, Tc=tc, bc=bc)


FRACS = [0.0, 0.3, 0.6, 0.85, 0.95, 0.99, 1.02, 1.1]


def main():
    check_convolution(5.0, 3.0)
    pool = Pool(min(8, os.cpu_count() or 1))
    t0 = time.time()
    pts = pool.map(point_Tc, [(k, mu, L) for k, (mu, L) in enumerate(POINTS)])
    print(f"\n[порог 2-го рода: {time.time() - t0:.0f} с]")
    for p in pts:
        n_near, S_near, _ = min(thresholds(p['mu'], p['L']), key=lambda q: abs(q[1] - p['bc']))
        print(f"μ={p['mu']}, L={p['L']}: T=0 фаза {p['phase']}, M*={p['M0']:.4e}, b*={p['b0']:.6f}"
              f"{' (асимпт.)' if p['asym'] else ''} | T_c^(2)={p['Tc']:.5e}, T_c/M*={p['Tc'] / p['M0']:.4f}, "
              f"b(T_c)−S_{n_near} = {(p['bc'] - S_near) / p['Tc']:+.3f}·T_c")
    t0 = time.time()
    jobs = [(p['mu'], p['L'], f * p['Tc'], p['M0']) for p in pts for f in FRACS]
    res = pool.map(full_min_T, jobs)
    print(f"\n[полный минимум: {time.time() - t0:.0f} с]")
    for k, p in enumerate(pts):
        print(f"\nμ={p['mu']}, L={p['L']} (M*(0)={p['M0']:.4e}, T_c^(2)={p['Tc']:.4e})")
        for f, (T, (om, Ms, bs)) in zip(FRACS, res[k * len(FRACS):(k + 1) * len(FRACS)]):
            print(f"   T/T_c={f:4.2f}: M*/M*(0)={Ms / p['M0']:.4f}  b*={bs:.6f}  Ω={om:+.3e}")
        p['curve'] = [(f, T, om, Ms, bs) for f, (T, (om, Ms, bs)) in
                      zip(FRACS, res[k * len(FRACS):(k + 1) * len(FRACS)])]
    np.save(os.path.join(os.path.dirname(__file__), 'output', '56_finite_T.npy'),
            np.array(pts, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    main()

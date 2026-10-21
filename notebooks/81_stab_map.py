"""
Instability map of the homogeneous solution in the intermediate phases I_j (0 < M* < mu) — the peak
region of paper 1 (mu ∈ [3, 3.4], L ∈ [2, 10]). Phase V (M* ≥ mu) is always stable (Phi = Phi_vac > 0),
phase T (M* = 0) is not considered.

For each I-point:
  phi_min  — min over k ≥ 0, q1 ∈ (0, 10] of Phi at T = 1e-3 (M* taken at T = 0);
  T_inst   — temperature above which no singular dip (k, q_s) reaches Phi < 0 (max over the three
             deepest dips at T = 1e-3); M* is kept at its T = 0 value;
  k, q1    — position of phi_min.

    python notebooks/81_stab_map.py [--n MU_N L_N] [--jobs J]
→ output/81_stab_map.npz, 81_stab_map.png
"""
import argparse
import math
import os
import time
import warnings
from multiprocessing import Pool

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
T_MAP = 1e-3


def _T_inst(k, qs, M, mu, L):
    from njl3.stab import phi
    f = lambda lT: phi(k, qs, M, mu, L, 10.0 ** lT)
    lo, hi = -8.0, 0.5
    if f(hi) < 0.0:
        return 10.0 ** hi
    if f(lo) > 0.0:
        return 0.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


def point(args):
    warnings.simplefilter('ignore')
    from njl3.b0 import phase_at
    from njl3.stab.formulas import phi_vec, scan_min, singular_q1
    mu, L = args
    ph, M = phase_at(mu, L, G)
    res = dict(phase=ph, M=M, phi_min=np.nan, T_inst=np.nan, k=-1, q1=np.nan)
    if not ph.startswith('I'):
        return res
    b = scan_min(M, mu, L, T=T_MAP)
    res.update(phi_min=b['phi_min'], k=b['k'], q1=b['q1'])
    from njl3.stab import phi
    dips = []
    for k in range(int(10.0 * L / (2 * math.pi)) + 1):
        for x in singular_q1(M, mu, L, k):
            if 0.0 < x <= 10.0:
                dips.append((phi(k, x, M, mu, L, T_MAP), k, x))
    dips.sort()
    res['T_inst'] = max((_T_inst(k, x, M, mu, L) for v, k, x in dips[:3] if v < 0.0), default=0.0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs=2, default=(120, 160))
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args()
    mus = np.linspace(3.0, 3.4, a.n[0])
    Ls = np.linspace(2.0, 10.0, a.n[1])
    pts = [(mu, L) for mu in mus for L in Ls]
    t = time.time()
    with Pool(a.jobs) as pool:
        rows = pool.map(point, pts, chunksize=8)
    print(f'{len(pts)} points, {sum(r["phase"].startswith("I") for r in rows)} in I phases: '
          f'{time.time() - t:.0f} s')
    shape = (len(mus), len(Ls))
    arr = {key: np.array([r[key] for r in rows]).reshape(shape)
           for key in ('M', 'phi_min', 'T_inst', 'k', 'q1')}
    phase = np.array([r['phase'] for r in rows]).reshape(shape)
    np.savez(os.path.join(OUT, '81_stab_map.npz'), mu=mus, L=Ls, phase=phase, **arr)
    plot(mus, Ls, phase, arr)


def plot(mus, Ls, phase, arr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    isI = np.char.startswith(phase.astype(str), 'I')
    code = np.where(phase == 'V', 0, np.where(phase == 'T', 1, 2)).astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    unstable = isI & (arr['phi_min'] < 0)
    c2 = np.where(isI, np.where(unstable, 3, 2), code)
    ax.pcolormesh(Ls, mus, c2, cmap=ListedColormap(['#dce9f5', '#eeeeee', '#9bd39b', '#e0605a']),
                  vmin=-0.5, vmax=3.5, shading='auto', rasterized=True)
    ax.set_title('V (голуб.), T (сер.), I: устойчива (зел.) / неустойчива (красн.)\n'
                 f'при $T={T_MAP:g}$', fontsize=9)
    ax = axes[1]
    Ti = np.where(isI, arr['T_inst'], np.nan)
    im = ax.pcolormesh(Ls, mus, np.log10(Ti), cmap='magma', shading='auto', rasterized=True)
    fig.colorbar(im, ax=ax, label='$\\log_{10} T_{inst}$')
    ax.set_title('Температура, выше которой провал у $2k_F$ исчезает', fontsize=9)
    ax = axes[2]
    kk = np.where(isI, arr['k'], np.nan)
    im = ax.pcolormesh(Ls, mus, kk, cmap='viridis', shading='auto', rasterized=True)
    fig.colorbar(im, ax=ax, label='$k^*$')
    ax.set_title('Самая неустойчивая мода по окружности ($T=10^{-3}$)', fontsize=9)
    for ax in axes:
        ax.set_xlabel('$L$'); ax.set_ylabel('$\\mu$')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '81_stab_map.png'), dpi=150)


if __name__ == '__main__':
    main()

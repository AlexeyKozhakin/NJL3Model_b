"""
"Phase diagram" of the stability of the homogeneous solution: colour = min over q1 ∈ (0, 10] of the second
variation Phi(k, q1) at T = 1e-3 (M* = M*(mu, L) from njl3.hom; twist φ, coupling g). Blue Phi > 0 — stable, red Phi < 0 —
unstable against an inhomogeneous fluctuation. Phase T (M* = 0): Phi = ½G(0⁺) + loop at a regulator mass
(njl3.stab.formulas.scan_min_zero); --fill-zero computes only these points in an existing .npz.

Left: min over q1 and all k ≥ 0; right: k = 0 only (modulation along x1 only).

    python notebooks/82_stab_minphi.py [--phi PHI] [--g G] [--region MU0 MU1 L0 L1 NAME] [--n MU_N L_N] [--fill-zero]
→ output/82_stab_minphi_<NAME>.npz / .png (φ = 0, g = −1), otherwise 82_stab_minphi_<NAME>_phi<φ>_g<g>.*
"""
import argparse
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


def point(args):
    warnings.simplefilter('ignore')
    from njl3.hom import phase_at
    from njl3.stab.formulas import scan_min
    mu, L, phi, g = args
    ph, M = phase_at(mu, L, g, phi)
    if M == 0.0:
        return (ph, M) + point_zero(args)
    b = scan_min(M, mu, L, T=T_MAP, phi=phi)
    b0 = scan_min(M, mu, L, T=T_MAP, k_max=0, phi=phi)
    return ph, M, b['phi_min'], b0['phi_min'], b['k']


def point_zero(args):
    warnings.simplefilter('ignore')
    from njl3.stab.formulas import scan_min_zero
    mu, L, phi, g = args
    b = scan_min_zero(mu, L, g, T=T_MAP, phi=phi)
    b0 = scan_min_zero(mu, L, g, T=T_MAP, k_max=0, phi=phi)
    return b['phi_min'], b0['phi_min'], b['k']


def fill_zero(path, phi, g):
    d = dict(np.load(path))
    ii = np.argwhere((d['phase'] == 'T') & ~np.isfinite(d['phi_min']))    # only the points not yet done
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(point_zero, [(d['mu'][i], d['L'][j], phi, g) for i, j in ii], chunksize=16)
    for (i, j), (p, p0, k) in zip(ii, rows):
        d['phi_min'][i, j], d['phi_min_k0'][i, j], d['k'][i, j] = p, p0, k
    np.savez(path, **d)
    print(f'{os.path.basename(path)}: {len(ii)} T points, {time.time() - t:.0f} s', flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', nargs=5, default=['0.05', '7', '0.1', '10', 'full'])
    ap.add_argument('--n', type=int, nargs=2, default=(200, 200))
    ap.add_argument('--plot-only', action='store_true', help='redraw from the saved .npz')
    ap.add_argument('--phi', type=float, default=0.0)
    ap.add_argument('--g', type=float, default=-1.0)
    ap.add_argument('--fill-zero', action='store_true', help='compute the phase-T points of the saved .npz')
    a = ap.parse_args()
    tag = '' if (a.phi == 0.0 and a.g == -1.0) else f'_phi{a.phi:g}_g{a.g:g}'
    if a.fill_zero:
        d = fill_zero(os.path.join(OUT, f'82_stab_minphi_{a.region[4]}{tag}.npz'), a.phi, a.g)
        return plot(d['mu'], d['L'], d['phase'], d['phi_min'], d['phi_min_k0'], a.region[4] + tag)
    if a.plot_only:
        d = np.load(os.path.join(OUT, f'82_stab_minphi_{a.region[4]}{tag}.npz'))
        return plot(d['mu'], d['L'], d['phase'], d['phi_min'], d['phi_min_k0'], a.region[4] + tag)
    mu0, mu1, L0, L1 = map(float, a.region[:4])
    name = a.region[4]
    mus, Ls = np.linspace(mu0, mu1, a.n[0]), np.linspace(L0, L1, a.n[1])
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(point, [(mu, L, a.phi, a.g) for mu in mus for L in Ls], chunksize=16)
    print(f'{name}{tag}: {len(rows)} points, {time.time() - t:.0f} s', flush=True)
    sh = (len(mus), len(Ls))
    phase = np.array([r[0] for r in rows]).reshape(sh)
    M, pmin, pmin0, kst = (np.array([r[i] for r in rows], float).reshape(sh) for i in range(1, 5))
    np.savez(os.path.join(OUT, f'82_stab_minphi_{name}{tag}.npz'), mu=mus, L=Ls, phase=phase, M=M,
             phi_min=pmin, phi_min_k0=pmin0, k=kst, phi=a.phi, g=a.g)
    plot(mus, Ls, phase, pmin, pmin0, name + tag)


def plot(mus, Ls, phase, pmin, pmin0, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lo = min(-0.05, np.nanmin(pmin)); hi = min(1.0, max(0.05, np.nanmax(pmin)))
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    for ax, v, ttl in ((axes[0], pmin, 'по всем $k$'), (axes[1], pmin0, 'только $k=0$')):
        ax.set_facecolor('#e6e6e6')
        im = ax.pcolormesh(Ls, mus, v, cmap='RdBu', norm=norm, shading='auto', rasterized=True)
        # boundaries between phases V / I_j / T
        code = np.zeros(phase.shape)
        code[np.char.startswith(phase.astype(str), 'I')] = 1
        code[phase == 'T'] = 2
        ax.contour(Ls, mus, code, levels=[0.5, 1.5], colors='k', linewidths=0.7)
        ax.set_xlabel('$L$'); ax.set_ylabel('$\\mu$')
        ax.set_title(f'минимум $\\Phi$ по $q_1\\in(0,10]$ {ttl}, $T=10^{{-3}}$', fontsize=10)
        fig.colorbar(im, ax=ax, label='$\\min\\Phi$  (< 0 — неустойчиво)', extend='max' if np.nanmax(pmin) > hi else 'neither')
    fig.suptitle('Устойчивость однородного решения модели ГН (фаза $T$: $M^*=0$)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'82_stab_minphi_{name}.png'), dpi=150)


if __name__ == '__main__':
    main()

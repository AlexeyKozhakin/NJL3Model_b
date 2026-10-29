"""
Sign of the second derivative of the second variation at q1 = 0:  Phi''(k = 0, q1 = 0) over (mu, L).

Phi(k, q1) is even in q1 (Phi'(0) = 0 identically), so near zero Phi ≈ Phi(0) + ½ Phi''(0) q1²:
Phi''(0) is the gradient (Ginzburg–Landau) coefficient. Phi'' < 0 — the homogeneous solution "prefers" a modulation
with finite q1. Closed form njl3.stab.formulas.phi_dd0 (stored as d2); the finite difference
2 [Phi(2h) - Phi(h)] / (3 h²), h = 0.01 (phase T at the regulator mass M_ZERO, h >> M_ZERO) is stored as d2_fd for checking.

    python notebooks/83_stab_curv.py [--phi PHI] [--g G] [--region MU0 MU1 L0 L1 NAME] [--n MU_N L_N] [--plot-only]
→ output/83_stab_curv_<NAME>[_phi<φ>_g<g>].npz / .png
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
H = 0.01


def point(args):
    warnings.simplefilter('ignore')
    from njl3.hom import phase_at
    from njl3.stab.formulas import M_ZERO, phi_dd0, phi_vec
    mu, L, phi, g = args
    ph, M = phase_at(mu, L, g, phi)
    m = M if M > 0.0 else M_ZERO
    v = phi_vec(0, np.array([H, 2 * H]), m, mu, L, phi)
    return ph, M, phi_dd0(M, mu, L, phi), 2.0 * (v[1] - v[0]) / (3.0 * H * H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', nargs=5, default=['0.05', '7', '0.1', '10', 'full'])
    ap.add_argument('--n', type=int, nargs=2, default=(200, 200))
    ap.add_argument('--phi', type=float, default=0.0)
    ap.add_argument('--g', type=float, default=-1.0)
    ap.add_argument('--plot-only', action='store_true')
    a = ap.parse_args()
    tag = '' if (a.phi == 0.0 and a.g == -1.0) else f'_phi{a.phi:g}_g{a.g:g}'
    path = os.path.join(OUT, f'83_stab_curv_{a.region[4]}{tag}.npz')
    if a.plot_only:
        d = np.load(path)
        return plot(d['mu'], d['L'], d['phase'], d['d2'], a.region[4] + tag)
    mu0, mu1, L0, L1 = map(float, a.region[:4])
    mus, Ls = np.linspace(mu0, mu1, a.n[0]), np.linspace(L0, L1, a.n[1])
    t = time.time()
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(point, [(mu, L, a.phi, a.g) for mu in mus for L in Ls], chunksize=16)
    print(f'{a.region[4]}{tag}: {len(rows)} points, {time.time() - t:.0f} s', flush=True)
    sh = (len(mus), len(Ls))
    phase = np.array([r[0] for r in rows]).reshape(sh)
    M = np.array([r[1] for r in rows], float).reshape(sh)
    d2 = np.array([r[2] for r in rows], float).reshape(sh)
    d2_fd = np.array([r[3] for r in rows], float).reshape(sh)
    np.savez(path, mu=mus, L=Ls, phase=phase, M=M, d2=d2, d2_fd=d2_fd, h=H, phi=a.phi, g=a.g)
    plot(mus, Ls, phase, d2, a.region[4] + tag)


def plot(mus, Ls, phase, d2, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, SymLogNorm
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    code = np.zeros(phase.shape)
    code[np.char.startswith(phase.astype(str), 'I')] = 1
    code[phase == 'T'] = 2
    s = np.sign(d2)
    ax = axes[0]
    ax.pcolormesh(Ls, mus, s, cmap=ListedColormap(['#b2182b', '#f2f1ec', '#2166ac']), vmin=-1.5, vmax=1.5,
                  shading='auto', rasterized=True)
    ax.set_title("знак $\\Phi''(k=0,\\,q_1=0)$: синее $>0$, красное $<0$", fontsize=10)
    ax = axes[1]
    lim = np.nanmax(np.abs(d2))
    im = ax.pcolormesh(Ls, mus, d2, cmap='RdBu', norm=SymLogNorm(linthresh=1e-3, vmin=-lim, vmax=lim),
                       shading='auto', rasterized=True)
    fig.colorbar(im, ax=ax, label="$\\Phi''(0)$")
    ax.set_title("$\\Phi''(k=0,\\,q_1=0)$ (симметричная лог. шкала)", fontsize=10)
    for ax in axes:
        ax.contour(Ls, mus, code, levels=[0.5, 1.5], colors='k', linewidths=0.7)
        ax.set_xlabel('$L$'); ax.set_ylabel('$\\mu$')
    n_neg = int((d2 < 0).sum())
    fig.suptitle(f'Вторая производная второй вариации по $q_1$ в нуле, аналитически ({name}); '
                 f'отрицательных точек: {n_neg} из {d2.size}; линии — границы фаз V / I / T', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'83_stab_curv_{name}.png'), dpi=150)


if __name__ == '__main__':
    main()

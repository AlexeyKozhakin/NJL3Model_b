"""
Phi(k, q1) at one point (mu, L): curves for k = 0..3 at T = 0 and k = 0 at T = 1e-3; if there is a
2k_F singularity (k = 0), a second panel shows its neighbourhood on a log scale in |q1 - 2k_F|.

    python notebooks/84_phi_point.py --mu 3.15 --L 1.9
→ output/84_phi_mu<mu>_L<L>.png
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from njl3.b0 import analyze_point
from njl3.stab import channels, phi
from njl3.stab.formulas import phi_vec, singular_q1

warnings.simplefilter('ignore')
OUT = os.path.join(os.path.dirname(__file__), 'output')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, required=True)
    ap.add_argument('--L', type=float, required=True)
    ap.add_argument('--g', type=float, default=-1.0)
    a = ap.parse_args()
    mu, L = a.mu, a.L
    r = analyze_point(mu, L, a.g)
    M = r.M_star
    print(f'mu={mu} L={L}: phase {r.phase}, M*={M:.6f}, filled channels {channels(M, mu, L)}')
    if M == 0.0:
        raise SystemExit('M* = 0: Phi is not defined (formula uses the gap equation)')
    qs = [x for x in singular_q1(M, mu, L, 0) if x > 0]
    fig, axes = plt.subplots(1, 2 if qs else 1, figsize=(13 if qs else 7, 4.5), squeeze=False)
    a1 = axes[0, 0]
    q = np.linspace(0.002, 10.0, 5000)
    for k, c in zip(range(4), COLORS):
        v = phi_vec(k, q, M, mu, L)
        a1.plot(q, v, color=c, lw=1.4, label=f'$k={k}$, $T=0$')
        print(f'  k={k}: min Phi(T=0) = {v.min():.6f} at q1 = {q[v.argmin()]:.4f}')
    qT = np.linspace(0.002, 10.0, 1200)
    vT = np.array([phi(0, x, M, mu, L, 1e-3) for x in qT])
    for x in qs:
        vT = np.append(vT, phi(0, x, M, mu, L, 1e-3)); qT = np.append(qT, x)
    o = np.argsort(qT)
    a1.plot(qT[o], vT[o], 'k--', lw=1, label='$k=0$, $T=10^{-3}$')
    print(f'  k=0: min Phi(T=1e-3) = {vT.min():.6f} at q1 = {qT[vT.argmin()]:.4f}')
    for x in qs:
        a1.axvline(x, color='0.5', ls=':', lw=0.8)
    lo = min(-0.2, 1.2 * vT.min())
    a1.set_ylim(lo, max(1.3, 1.05 * phi_vec(0, np.array([10.0]), M, mu, L)[0]))
    a1.set_xlim(0, 10)
    a1.axhline(0, color='k', lw=0.7); a1.grid(alpha=0.3)
    a1.set_xlabel('$q_1$'); a1.set_ylabel('$\\Phi(k, q_1)$'); a1.legend(fontsize=8)
    a1.set_title(f'$\\mu={mu}$, $L={L}$: фаза {r.phase}, $M^*={M:.4f}$'
                 + ('' if qs else ' (нет ферми-моря: $\\Phi=\\Phi_{vac}>0$)'), fontsize=10)
    if qs:
        a2 = axes[0, 1]
        x0 = qs[-1]
        d = np.geomspace(1e-9, 0.5, 300)
        for sgn, ls in ((-1, '-'), (1, '--')):
            x = x0 + sgn * d
            s = '-' if sgn < 0 else '+'
            a2.semilogx(d, phi_vec(0, x, M, mu, L), ls, color=COLORS[0], label=f'$T=0$, $q_1=2k_F{s}\\delta$')
            a2.semilogx(d, [phi(0, y, M, mu, L, 1e-3) for y in x], ls, color='k', lw=1,
                        label=f'$T=10^{{-3}}$, $q_1=2k_F{s}\\delta$')
        a2.axhline(0, color='k', lw=0.7); a2.grid(alpha=0.3)
        a2.set_xlabel('$\\delta=|q_1-2k_F|$'); a2.legend(fontsize=8)
        a2.set_title(f'$k=0$ около $2k_F={x0:.4f}$: логарифм и обрезание при $T>0$', fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, f'84_phi_mu{mu:g}_L{L:g}.png')
    fig.savefig(path, dpi=150)
    print('  ->', path)


if __name__ == '__main__':
    main()

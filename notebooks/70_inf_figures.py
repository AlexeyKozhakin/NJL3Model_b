"""
Рисунки для обсуждения задачи без компактификации (L -> infinity), g = -1.

    python notebooks/70_inf_figures.py

70_inf_Mstar.png      — M*(mu), b*(mu) при L = ∞ и M*(mu) задачи на R²×S¹ при L = 5, 10, 20, 40
70_inf_omega_b0.png   — Omega(M) - Omega(0) при b = 0 для нескольких mu (плоский потенциал при mu = π)
70_inf_b_map.png      — Omega(M, b) - Omega(M, 0) при mu = 4: нуль при b ≤ mu - M, рост вне
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.b0.minimize import find_minimum as find_min_L
from njl3.inf.formulas import omega_rel, omega_rel_b0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
PI = math.pi


def fig_Mstar():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (lo, hi, n) in zip(axes[:2], ((0.05, 5.0, 1500), (3.0, 3.4, 1500))):
        mu = np.linspace(lo, hi, n)
        for L, c in ((5, '0.75'), (10, '0.55'), (20, '0.35'), (40, '0.15')):
            M = [find_min_L(m, L, G)[0] for m in mu]
            ax.plot(mu, M, '-', color=c, lw=1.0, label=f'$\\mathbb{{R}}^2\\times S^1$, $L={L}$')
        a, b = mu[mu < PI], mu[mu > PI]
        ax.plot(a, np.full_like(a, PI), '-', color='tab:blue', lw=2.5, label='$L=\\infty$')
        ax.plot(b, np.zeros_like(b), '-', color='tab:blue', lw=2.5)
        ax.plot([PI, PI], [0, PI], ':', color='tab:blue', lw=1.5)
        ax.set_xlabel('$\\mu$'); ax.set_ylabel('$M^*$'); ax.set_xlim(lo, hi); ax.set_ylim(-0.1, 4.2)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title('$M^*(\\mu)$, $g=-1$ (на $\\mathbb{R}^2\\times S^1$ — при $b=0$)', fontsize=10)
    axes[1].set_title('то же, $\\mu\\in[3,\\,3.4]$: при конечном $L$ — промежуточные фазы', fontsize=10)
    axes[1].annotate('при $\\mu=\\pi$: любое $M\\in[0,\\pi]$\n(потенциал плоский)', xy=(PI, 1.6),
                     xytext=(3.2, 2.4), fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax = axes[2]
    mu = np.linspace(0.05, 5.0, 1000)
    ax.plot(mu[mu < PI], 0 * mu[mu < PI], '-', color='tab:orange', lw=2.5, label='$b^*=0$ (единственный минимум)')
    ax.fill_between(mu[mu > PI], 0, 5, color='tab:orange', alpha=0.15, label='$M^*=0$: $b$ не определено')
    ax.axvline(PI, color='0.5', ls=':', lw=1)
    ax.set_xlabel('$\\mu$'); ax.set_ylabel('$b^*$'); ax.set_xlim(0, 5); ax.set_ylim(-0.2, 5)
    ax.legend(fontsize=8, loc='upper left'); ax.set_title('$b^*(\\mu)$ при $L=\\infty$', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '70_inf_Mstar.png'), dpi=150); plt.close(fig)


def fig_omega_b0():
    M = np.linspace(0, 4.2, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mu, c in ((2.5, 'tab:blue'), (3.0, 'tab:cyan'), (PI, 'k'), (3.3, 'tab:orange'), (4.0, 'tab:red')):
        ax.plot(M, [omega_rel_b0(m, mu, G) for m in M], '-', color=c, lw=1.8,
                label=f'$\\mu={mu:.4g}$' + (' $=\\pi$' if mu == PI else ''))
    ax.axhline(0, color='0.6', lw=0.7)
    ax.set_xlabel('$M$'); ax.set_ylabel('$\\Omega(M)-\\Omega(0)$'); ax.set_ylim(-1.0, 1.0)
    ax.set_title('$L=\\infty$, $b=0$: при $M\\leq\\mu$ потенциал $=M^2[1/(2g)+\\mu/(2\\pi)]$', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '70_inf_omega_b0.png'), dpi=150); plt.close(fig)


def fig_b_map():
    mu = 4.0
    M = np.linspace(0.02, 4.5, 140)
    b = np.linspace(0.0, 6.0, 160)
    D = np.array([[omega_rel(m, bb, mu, G) - omega_rel_b0(m, mu, G) for m in M] for bb in b])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.pcolormesh(M, b, D, cmap='Reds', shading='auto', vmin=0, vmax=1.0, rasterized=True)
    fig.colorbar(im, ax=ax, label='$\\Omega(M,b)-\\Omega(M,0)$')
    ax.plot(M[M <= mu], mu - M[M <= mu], 'k--', lw=1.2, label='$b=\\mu-M$')
    ax.text(0.9, 1.2, '$\\Omega(M,b)=\\Omega(M,0)$\n(до $10^{-14}$)', fontsize=9)
    ax.set_xlabel('$M$'); ax.set_ylabel('$b$'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'$L=\\infty$, $\\mu={mu}$: $b$ не понижает потенциал', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '70_inf_b_map.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    fig_Mstar(); fig_omega_b0(); fig_b_map()

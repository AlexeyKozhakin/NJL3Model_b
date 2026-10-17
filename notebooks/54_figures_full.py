"""
Рисунки для черновика статьи 2 (полная задача (M, b), g = -1).

    python notebooks/54_figures_full.py           # рисунки 1–3
    python notebooks/54_figures_full.py --maps    # + карты M*, b* из 53_full_maps.npz

Выход: notebooks/output/54_*.png
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.full import formulas as F
from njl3.full.minimize import channel_asymptotics, h_universal, thresholds

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output')
G = -1.0
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'


def fig_log_instability():
    mu, L = 5.0, 3.0
    M = np.geomspace(1e-7, 0.3, 80)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (n, S, C), col in zip(thresholds(mu, L), (C1, C2, C3)):
        a = channel_asymptotics(S, C, mu, L, G)
        y = [F.omega_rel(m, S, mu, L, G) / m**2 for m in M]
        ax.semilogx(M, y, '-', color=col, lw=2, label=f'n = {n}: численно')
        ax.semilogx(M, a['A'] + a['c'] * np.log(M), '--', color=col, lw=1.2, label=f'n = {n}: A − c ln(1/M)')
    ax.axhline(0, color='#898781', lw=0.8)
    ax.set_xlabel('M'); ax.set_ylabel('[Ω(M, S_n) − Ω(0)] / M²')
    ax.set_title(f'Логарифмическая неустойчивость у порогов (μ = {mu}, L = {L})')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '54_log_instability.png'), dpi=150); plt.close(fig)


def fig_collapse():
    t = np.linspace(-6, 6, 241)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t, [h_universal(x) for x in t], 'k-', lw=2, label='h(t)')
    marks = iter(['o', 's', '^', 'D', 'v', 'P'])
    for mu, L in [(6.0, 1.0), (5.0, 3.0)]:
        for n, S, C in thresholds(mu, L)[:2]:
            c = C * S / (2 * math.pi * L * mu)
            for Mv, col in [(1e-3, C2), (1e-5, C1)]:
                tt = np.linspace(-6, 6, 25)
                y0 = F.omega_rel(Mv, S, mu, L, G) / Mv**2
                h = [(F.omega_rel(Mv, S + x * Mv, mu, L, G) / Mv**2 - y0) / c for x in tt]
                ax.plot(tt, h, next(marks, 'x'), ms=4, mfc='none', color=col,
                        label=f'μ={mu}, L={L}, n={n}, M={Mv:g}')
            marks = iter(['o', 's', '^', 'D', 'v', 'P'])
    ax.set_xlabel('t = (b − S_n)/M'); ax.set_ylabel('h(t)')
    ax.set_title('Универсальная зависимость от сдвига b − S_n')
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '54_collapse.png'), dpi=150); plt.close(fig)


def fig_strong_well():
    mu, L = 3.0, 1.0
    M = np.linspace(0.005, 2.0, 160)
    b = np.linspace(0.0, 4.0, 200)
    Mg, bg = np.meshgrid(M, b)
    Om = F.omega_rel_vec(Mg, bg, mu, L, G)
    i, j = np.unravel_index(np.argmin(Om), Om.shape)
    lim = np.percentile(np.abs(Om), 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.pcolormesh(M, b, Om, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
    ax.plot(M[j], b[i], 'k*', ms=12)
    ax.axhline(mu, color='k', lw=0.8, ls=':')
    ax.set_xlabel('M'); ax.set_ylabel('b')
    ax.set_title(f'Ω(M, b) − Ω(0, 0), μ = {mu}, L = {L}: яма у порога S₀ = μ')
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '54_strong_well.png'), dpi=150); plt.close(fig)


def fig_maps():
    d = np.load(os.path.join(OUT, '53_full_maps.npz'), allow_pickle=True)
    L, mu, lnM, b, phase = d['L'], d['mu'], d['lnM'], d['b'], d['phase']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    logM = np.clip(lnM / math.log(10), -8, 1)
    im = axes[0].pcolormesh(L, mu, logM.T, cmap='Blues', shading='auto', vmin=-8, vmax=1)
    cb = fig.colorbar(im, ax=axes[0]); cb.set_label('log₁₀ M*')
    im = axes[1].pcolormesh(L, mu, np.nan_to_num(b).T, cmap='Oranges', shading='auto')
    cb = fig.colorbar(im, ax=axes[1]); cb.set_label('b*')
    # границы фаз
    for ax in axes:
        for i in range(len(L)):
            for j in range(len(mu)):
                if i + 1 < len(L) and phase[i, j] != phase[i + 1, j]:
                    x = 0.5 * (L[i] + L[i + 1]); y0 = mu[max(j - 1, 0)] if j else mu[0]
                    ax.plot([x, x], [0.5 * (mu[j] + mu[max(j - 1, 0)]), 0.5 * (mu[j] + mu[min(j + 1, len(mu) - 1)])], 'k-', lw=0.5)
                if j + 1 < len(mu) and phase[i, j] != phase[i, j + 1]:
                    y = 0.5 * (mu[j] + mu[j + 1])
                    ax.plot([0.5 * (L[i] + L[max(i - 1, 0)]), 0.5 * (L[i] + L[min(i + 1, len(L) - 1)])], [y, y], 'k-', lw=0.5)
        ax.set_xlabel('L'); ax.set_ylabel('μ')
    axes[0].set_title('M* (логарифмическая шкала)'); axes[1].set_title('b*')
    fig.suptitle('Глобальный минимум по (M, b), g = −1; линии — границы фаз')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, '54_maps.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    if '--maps' in sys.argv:
        fig_maps()
    else:
        fig_log_instability(); fig_collapse(); fig_strong_well()

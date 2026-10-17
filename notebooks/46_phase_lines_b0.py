"""
Задачи B2, B3 (статья 1): линии фазовых переходов и особые точки при b = 0, g = -1.

Линии считаются как решения уравнений (functions/phase_lines_b0.py):
трассировка по mu при фиксированном L гарантированным минимизатором
+ бисекция до 1e-11, род перехода — по скачку M*.

Особые точки:
    tricritical — на линии G0 = 0 коэффициент G2 = 0 (смена рода I–T)
    triple      — исчезает фаза I: [V→I, I→T] сменяется на [V→T]
    peak        — линия I–T / V–T пересекает порог моды mu = 2 pi n / L

Перекрёстные проверки: точки 2-го рода против mu_s(L); V→T против mu_VT(L).

Выход:
    notebooks/output/46_phase_lines.json   — линии и особые точки
    notebooks/output/46_phase_lines_b0.png — линии поверх диаграммы M*(L, mu) из 43

Запуск:  python notebooks/46_phase_lines_b0.py [--quick]
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import phase_lines_b0 as pl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
QUICK = '--quick' in sys.argv
G = -1.0
MU_LO, MU_HI = 0.05, 7.0
L_GRID = np.linspace(0.1, 10.0, 200 if QUICK else 1000)


def _trace(L):
    return pl.trace_L(L, G, MU_LO, MU_HI)


def pattern(L):
    return tuple((r['frm'], r['to']) for r in _trace(L))


def refine_L(La, Lb, pa, tol=1e-8):
    """Бисекция по L до смены последовательности переходов."""
    while Lb - La > tol:
        c = 0.5 * (La + Lb)
        if pattern(c) == pa:
            La = c
        else:
            Lb = c
    return 0.5 * (La + Lb)


def main():
    t0 = time.time()
    with Pool() as pool:
        traces = pool.map(_trace, L_GRID)
    print(f"Трассировка {len(L_GRID)} значений L: {time.time() - t0:.1f}s")

    # --- линии -------------------------------------------------------------------
    lines = {}
    for L, tr in zip(L_GRID, traces):
        for r in tr:
            key = f"{r['frm']}-{r['to']}"
            lines.setdefault(key, []).append(r)
    for key, pts in lines.items():
        o = [p['order'] for p in pts]
        print(f"  {key}: {len(pts)} точек, 1-й род {o.count(1)}, 2-й род {o.count(2)}")

    # --- перекрёстные проверки ---------------------------------------------------
    d2 = [abs(r['mu'] - pl.mu_stability(r['L'], G)) for k in lines for r in lines[k] if r['order'] == 2]
    # (mu здесь — результат бисекции по меткам фаз, независимый от формулы mu_s)
    dvt = [abs(r['mu'] - pl.mu_VT(r['L'], G)) for r in lines.get('V-T', [])]
    print(f"Проверка: 2-й род vs mu_s(L): max |Δmu| = {max(d2) if d2 else float('nan'):.1e}  ({len(d2)} точек)")
    print(f"Проверка: V→T vs mu_VT(L):   max |Δmu| = {max(dvt) if dvt else float('nan'):.1e}  ({len(dvt)} точек)")

    # --- особые точки ------------------------------------------------------------
    special = []
    # трикритические: G2 = 0 на mu_s(L), и там действительно граница I–T
    for Lt, mut in pl.tricritical_points(G, L_GRID[0], L_GRID[-1]):
        tr = _trace(Lt)
        on_line = any('T' in (r['frm'], r['to']) and abs(r['mu'] - mut) < 1e-6 for r in tr)
        special.append(dict(kind='tricritical', L=Lt, mu=mut, on_phase_boundary=bool(on_line)))
    # тройные: смена последовательности переходов по L
    pats = [tuple((r['frm'], r['to']) for r in tr) for tr in traces]
    jobs = [(L_GRID[i], L_GRID[i + 1], pats[i]) for i in range(len(L_GRID) - 1) if pats[i] != pats[i + 1]]
    with Pool() as pool:
        Lcs = pool.starmap(refine_L, jobs)
    for (La, Lb, pa), Lc in zip(jobs, Lcs):
        pb = pattern(Lb)
        special.append(dict(kind='pattern_change', L=Lc,
                            before=[f'{a}-{b}' for a, b in pa], after=[f'{a}-{b}' for a, b in pb]))
    # пики: линия перехода в фазу T пересекает mu = 2 pi n / L
    for key in ['*-T']:
        pts = sorted([r for k, v in lines.items() if k.endswith('-T') for r in v], key=lambda r: r['L'])
        for a, b in zip(pts[:-1], pts[1:]):
            if b['L'] - a['L'] > 2 * (L_GRID[1] - L_GRID[0]):
                continue
            na, nb = a['L'] * a['mu'] / (2 * np.pi), b['L'] * b['mu'] / (2 * np.pi)
            if np.floor(na) != np.floor(nb):
                n = int(max(np.floor(na), np.floor(nb)))
                special.append(dict(kind='peak', n=n, line=key,
                                    L=0.5 * (a['L'] + b['L']), mu=0.5 * (a['mu'] + b['mu'])))

    print("\nОсобые точки:")
    for s in sorted(special, key=lambda s: s['L']):
        extra = {k: v for k, v in s.items() if k not in ('kind', 'L', 'mu')}
        if s['kind'] == 'pattern_change':
            extra = f"{' '.join(extra['before'])}  =>  {' '.join(extra['after'])}"
        mu = f"{s['mu']:.8f}" if s.get('mu') is not None else '—'
        print(f"  {s['kind']:15s} L={s['L']:.8f}  mu={mu}  {extra}")

    out = dict(g=G, mu_range=[MU_LO, MU_HI], L_range=[float(L_GRID[0]), float(L_GRID[-1])],
               lines={k: [dict(L=r['L'], mu=r['mu'], order=r['order'], M_left=r['M_left'], M_right=r['M_right'])
                          for r in v] for k, v in lines.items()},
               special=special)
    with open(os.path.join(OUT_DIR, '46_phase_lines.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    # --- рисунок ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bg = os.path.join(OUT_DIR, '43_data.npz')
    if os.path.exists(bg):
        d = np.load(bg)
        im = ax.pcolormesh(d['L'], d['mu'], d['M_star'].T, cmap='viridis', shading='auto', vmax=4, alpha=0.85)
        fig.colorbar(im, ax=ax, label='M*', extend='max')
    style = {1: dict(color='w', lw=2.2), 2: dict(color='r', lw=2.2)}
    for key, pts in lines.items():
        for order in (1, 2):
            sel = sorted([p for p in pts if p['order'] == order], key=lambda p: p['L'])
            if not sel:
                continue
            Ls = np.array([p['L'] for p in sel]); ms = np.array([p['mu'] for p in sel])
            breaks = np.nonzero(np.diff(Ls) > 2.5 * (L_GRID[1] - L_GRID[0]))[0] + 1
            for seg in np.split(np.arange(len(Ls)), breaks):
                ax.plot(Ls[seg], ms[seg], **style[order])
    for n in range(1, 12):
        Lf = np.linspace(L_GRID[0], L_GRID[-1], 400)
        ax.plot(Lf, 2 * np.pi * n / Lf, color='gray', lw=0.5, ls=':')
    marks = {'tricritical': ('*', 'gold', 14), 'pattern_change': ('o', 'cyan', 7), 'peak': ('^', 'orange', 7)}
    for kind, (mk, col, ms_) in marks.items():
        pts = [s for s in special if s['kind'] == kind and s.get('mu') is not None]
        if pts:
            ax.plot([s['L'] for s in pts], [s['mu'] for s in pts], mk, color=col, ms=ms_,
                    mec='k', mew=0.6, ls='none', label=kind)
    ax.plot([], [], color='w', lw=2.2, label='1-й род')
    ax.plot([], [], color='r', lw=2.2, label='2-й род')
    ax.plot([], [], color='gray', lw=0.5, ls=':', label='μ = 2πn/L')
    ax.set_xlim(L_GRID[0], L_GRID[-1]); ax.set_ylim(MU_LO, MU_HI)
    ax.set_xlabel('L'); ax.set_ylabel('μ')
    ax.set_title('Линии переходов при b = 0, g = −1 (решения уравнений, точность 1e-11)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '46_phase_lines_b0.png'), dpi=130)
    plt.close(fig)
    print(f"\nГотово за {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()

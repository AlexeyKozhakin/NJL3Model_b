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

Запуск:  python notebooks/46_phase_lines_b0.py [--quick | --plot-only [--refresh-special]]
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


LINE_STYLE = {
    'V-T':  dict(color='k', lw=2.0),
    'V-I':  dict(color='tab:orange', lw=2.0),
    'I-I':  dict(color='tab:blue', lw=1.6),
    'I-T1': dict(color='k', lw=2.0),
    'I-T2': dict(color='r', lw=2.4),
}


def _style_key(key, order):
    a, b = key.split('-')
    if a == 'V':
        return 'V-T' if b == 'T' else 'V-I'
    if b == 'T':
        return f'I-T{order}'
    return 'I-I'


def plot_lines(out):
    """Линии переходов: вся область и увеличение полосы mu ∈ [2.5, 4]."""
    L_all = sorted({r['L'] for v in out['lines'].values() for r in v})
    dL = np.median(np.diff(L_all))
    fig, axes = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw=dict(height_ratios=[1, 1.15]))
    for ax, (ylo, yhi) in zip(axes, [out['mu_range'], (2.5, 4.0)]):
        for key, pts in out['lines'].items():
            for order in (1, 2):
                sel = sorted([p for p in pts if p['order'] == order], key=lambda p: p['L'])
                if not sel:
                    continue
                Ls = np.array([p['L'] for p in sel]); ms = np.array([p['mu'] for p in sel])
                breaks = np.nonzero(np.diff(Ls) > 2.5 * dL)[0] + 1
                for seg in np.split(np.arange(len(Ls)), breaks):
                    ax.plot(Ls[seg], ms[seg], **LINE_STYLE[_style_key(key, order)])
        Lf = np.linspace(*out['L_range'], 600)
        for n in range(1, 12):
            ax.plot(Lf, 2 * np.pi * n / Lf, color='gray', lw=0.5, ls=':')
        ax.axhline(np.pi, color='gray', lw=0.6, ls='--')
        sp = out['special']
        for kind, (mk, col, ms_, lab) in {
                'critical_end': ('o', 'cyan', 8, 'критическая концевая точка (рождение I_n)'),
                'peak': ('^', 'orange', 8, 'пик: μ = 2πn/L'),
                'tricritical': ('*', 'gold', 15, 'трикритическая точка')}.items():
            pts = [x for x in sp if x['kind'] == kind and x.get('mu') is not None
                   and x.get('on_phase_boundary', True)]
            if pts:
                ax.plot([x['L'] for x in pts], [x['mu'] for x in pts], mk, color=col, ms=ms_,
                        mec='k', mew=0.7, ls='none', label=lab, zorder=5)
        ax.set_xlim(*out['L_range']); ax.set_ylim(ylo, yhi)
        ax.set_xlabel('L'); ax.set_ylabel('μ')
        ax.grid(alpha=0.2)
    ax0, ax1 = axes
    for lab, (x, y) in {'V\n(вакуум, M* = M_vac(L))': (6.0, 1.5), 'T  (M* = 0)': (6.0, 5.5)}.items():
        ax0.text(x, y, lab, ha='center', fontsize=12)
    # подписи подфаз I_n в увеличении: между линией I_{n-1}–I_n (или V–I0) и границей с T
    ce = sorted([x for x in out['special'] if x['kind'] == 'critical_end'], key=lambda x: x['L'])
    pk = sorted([x for x in out['special'] if x['kind'] == 'peak'], key=lambda x: x['L'])

    def _line_mu(keys, L):
        pts = sorted([r for k in keys for r in out['lines'].get(k, [])], key=lambda r: r['L'])
        Ls = np.array([r['L'] for r in pts]); ms = np.array([r['mu'] for r in pts])
        return np.interp(L, Ls, ms) if len(Ls) else np.nan

    T_keys = [k for k in out['lines'] if k.endswith('-T')]
    for n, x in enumerate(ce):
        L_lab = x['L'] + 0.75 * (pk[n]['L'] - x['L']) if n < len(pk) else x['L'] + 0.3
        lower = 'V-I0' if n == 0 else f'I{n - 1}-I{n}'
        mu_lab = 0.5 * (_line_mu([lower], L_lab) + _line_mu(T_keys, L_lab))
        ax1.text(L_lab, mu_lab, f'I{n}', fontsize=11, color='tab:blue', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.7))
    ax1.text(0.5, 2.6, 'V', fontsize=13); ax1.text(6.0, 3.7, 'T', fontsize=13)
    handles = [plt.Line2D([], [], **LINE_STYLE[k]) for k in ('V-T', 'V-I', 'I-I', 'I-T1', 'I-T2')]
    labels = ['V–T (1-й род)', 'V–I₀ (1-й род)', 'I_n–I_{n+1} (1-й род)', 'I_n–T (1-й род)', 'I_n–T (2-й род)']
    h2, l2 = ax1.get_legend_handles_labels()
    ax1.legend(handles + h2 + [plt.Line2D([], [], color='gray', lw=0.5, ls=':')],
               labels + l2 + ['пороги мод μ = 2πn/L'], fontsize=8, loc='upper right', framealpha=0.95)
    ax0.set_title('Фазовая диаграмма при b = 0, g = −1: линии переходов (точность ~1e-12)')
    ax1.set_title('Увеличение: каскад подфаз I₀, I₁, … по числу заполненных KK-мод')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '46_phase_lines_b0.png'), dpi=130)
    plt.close(fig)


def refresh_special(out):
    """Трикритические точки (исправленный поиск) и пики по явной формуле."""
    keep = [x for x in out['special'] if x['kind'] not in ('tricritical', 'peak')]
    for Lt, mut in pl.tricritical_points(G, *out['L_range']):
        tr = _trace(Lt)
        on_line = any('T' in (r['frm'], r['to']) and abs(r['mu'] - mut) < 1e-6 for r in tr)
        keep.append(dict(kind='tricritical', L=Lt, mu=mut, on_phase_boundary=bool(on_line)))
    n = 1
    while True:
        Ln, mun = pl.peak(n, G)
        if Ln > out['L_range'][1]:
            break
        keep.append(dict(kind='peak', n=n, L=Ln, mu=mun, exact=True))
        n += 1
    out['special'] = sorted(keep, key=lambda x: x['L'])
    return out


def add_critical_end_points(out):
    """pattern_change -> critical_end с mu: точка, где рождается подфаза I_n
    (линия 2-го рода I_n–T упирается в линию 1-го рода)."""
    for x in out['special']:
        if x['kind'] == 'pattern_change' and x['before'] and x['after'] and len(x['after']) > len(x['before']):
            tr = _trace(x['L'] + 1e-6)
            muT = [r['mu'] for r in tr if r['to'] == 'T']
            x['kind'] = 'critical_end'
            x['mu'] = muT[0] if muT else None
    return out


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
    out = refresh_special(add_critical_end_points(out))
    with open(os.path.join(OUT_DIR, '46_phase_lines.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    plot_lines(out)
    print(f"\nГотово за {time.time() - t0:.1f}s")


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        path = os.path.join(OUT_DIR, '46_phase_lines.json')
        with open(path) as f:
            out = json.load(f)
        if any(x['kind'] == 'pattern_change' for x in out['special']):
            out = refresh_special(add_critical_end_points(out))
        if '--refresh-special' in sys.argv:
            out = refresh_special(out)
        with open(path, 'w') as f:
            json.dump(out, f, indent=1, default=float)
        plot_lines(out)
    else:
        main()

"""
Reproduce every figure and table of the paper on the b = 0 phase diagram (njl3.b0 only).

    python paper/reproduce.py              # everything (~15 min on 8 cores, mostly fig1 lines)
    python paper/reproduce.py --quick      # smaller grids and samples (~1-2 min), same outputs
    python paper/reproduce.py fig1 tab-accuracy   # selected targets
    python paper/reproduce.py --ru --bw    # Russian labels, greyscale figures 16 cm wide (journal version)
    python paper/reproduce.py --ru --print # same, in colour but readable in greyscale print

Targets:
    fig1          transition lines: whole range and the strip mu in [2.5, 4]
    fig2          map of M*(L, mu) in the peak region mu in [3, 3.4], L in [2, 10]
    fig3          G(M) from the closed form against finite differences of direct integration
    fig4          slices M*(mu) at fixed L and M*(L) at fixed mu: phase sequence and order of transitions
    tab-special   special points (critical end points, peaks)
    tab-accuracy  accuracy against mpmath (50 digits): G, Omega, roots, lines
    tab-scan      certified search against a grid scan: missed minima and timing
    tab-perf      performance: single operations, a map, the lines; JS core in Node.js

Output: paper/output/ (PNG and PDF figures; tables as Markdown, CSV and LaTeX; lines.json).
With --ru / --bw / --print the file names get the suffixes _ru / _bw / _print.
Coupling g = -1 throughout.
"""

import csv
import json
import math
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import mpmath as mp
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from njl3.b0 import formulas as F
from njl3.b0 import reference as R
from njl3.b0.lines import M_vacuum, compute_grid, mu_stability, mu_VT, peaks, trace_lines
from njl3.b0.minimize import find_minimum, find_minimum_scan, isolate_roots, phase_label

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = -1.0
QUICK = '--quick' in sys.argv
RU = '--ru' in sys.argv          # Russian labels and table texts
BW = '--bw' in sys.argv          # greyscale figures (line styles and markers instead of colours), 16 cm wide
PRINT = '--print' in sys.argv    # colour figures that stay readable in greyscale print (colour + line styles)
JR = BW or PRINT                 # journal layout: line styles and markers, 16 cm wide, no titles
SUF = ('_ru' if RU else '') + ('_bw' if BW else '_print' if PRINT else '')


def tx(en, ru):
    """Text in the chosen language."""
    return ru if RU else en


# colours (colour version) and their greyscale replacements
C = dict(first='k', second='tab:red', vi='tab:orange', ii='tab:blue', thr='0.6', peak='tab:orange',
         cep='tab:cyan', curve='tab:blue', points='tab:red') if not BW else \
    dict(first='k', second='k', vi='0.35', ii='0.35', thr='0.75', peak='white', cep='k', curve='k', points='k')
FIGW = 6.3 if JR else None       # text width of the journal page (A4, margins 2.5 cm)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
MU_RANGE, L_RANGE = (0.05, 7.0), (0.1, 10.0)
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8, 'savefig.bbox': 'tight'})


def save(fig, name):
    name += SUF
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200)
    plt.close(fig)
    print(f'  -> {name}.png, {name}.pdf')


def write_table(name, header, rows, fmt):
    name += SUF
    with open(os.path.join(OUT, f'{name}.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([header, *rows])
    with open(os.path.join(OUT, f'{name}.md'), 'w') as f:
        f.write('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header) + '\n')
        for r in rows:
            f.write('| ' + ' | '.join(fm.format(x) if isinstance(x, (int, float)) else str(x)
                                      for fm, x in zip(fmt, r)) + ' |\n')
    # LaTeX: body of a tabular (header + rows), for \\input in the manuscript
    def tex(x, fm):
        v = fm.format(x) if isinstance(x, (int, float)) else str(x)
        if v == 'nan':
            return '--'
        if '$' in v:                                       # already LaTeX
            return v.replace('%', r'\%')
        m = __import__('re').fullmatch(r'([-+]?\d+(?:\.\d+)?)e([-+]\d+)', v)
        if m:
            if float(v) == 0.0:
                return '$0$'
            return f'${m.group(1)}\\times10^{{{int(m.group(2))}}}$'
        return v.replace('_', r'\_').replace('%', r'\%').replace('µ', r'$\mu$').replace('×', r'$\times$')
    with open(os.path.join(OUT, f'{name}.tex'), 'w') as f:
        f.write(' & '.join(tex(h, '{}') for h in header) + r' \\ \hline' + '\n')
        for r in rows:
            f.write(' & '.join(tex(x, fm) for fm, x in zip(fmt, r)) + r' \\' + '\n')
    print(f'  -> {name}.md, {name}.csv, {name}.tex')


# ==============================================================================
# Transition lines (fig1, tab-special, tab-accuracy)
# ==============================================================================

def lines():
    """Transition lines on the whole range; cached in output/lines.json (rebuilt per mode)."""
    path = os.path.join(OUT, 'lines.json')
    nL, n_mu = (60, 200) if QUICK else (400, 400)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d.get('nL') == nL and d.get('n_mu') == n_mu:
            return d
    t = time.time()
    d = trace_lines(np.linspace(*L_RANGE, nL), G, *MU_RANGE, n_mu)
    d.update(nL=nL, n_mu=n_mu, seconds=time.time() - t, cores=os.cpu_count())
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)
    print(f'  transition lines: {nL} values of L, {time.time() - t:.0f} s')
    return d


def _draw_lines(ax, d, mu_lim, L_lim):
    style = {2: dict(color=C['second'], lw=1.8, ls='--' if JR else '-')}
    for key, pts in d['lines'].items():
        a, b = key.split('-')
        for order in (1, 2):
            sel = sorted((p for p in pts if p['order'] == order), key=lambda p: p['L'])
            if not sel:
                continue
            Ls = np.array([p['L'] for p in sel]); ms = np.array([p['mu'] for p in sel])
            step = (L_RANGE[1] - L_RANGE[0]) / (d['nL'] - 1)
            if style.get(order):
                st = style[order]
            elif b == 'T':
                st = dict(color=C['first'], lw=1.4)
            elif a == 'V':
                st = dict(color=C['vi'], lw=1.4, ls='-.' if JR else '-')
            else:
                st = dict(color=C['ii'], lw=1.4, ls=(0, (1, 1)) if JR else '-')
            for seg in np.split(np.arange(len(Ls)), np.nonzero(np.diff(Ls) > 2.5 * step)[0] + 1):
                ax.plot(Ls[seg], ms[seg], **st)
    for n in range(1, 40):                                  # mode thresholds mu = 2 pi n / L
        L = np.linspace(*L_lim, 400)
        ax.plot(L, 2 * np.pi * n / L, '-' if JR else ':', color=C['thr'], lw=0.5 if JR else 0.7)
    pk = [s for s in d['special'] if s['kind'] == 'peak']
    ce = [s for s in d['special'] if s['kind'] == 'critical_end' and s.get('mu') is not None]
    ax.plot([s['L'] for s in pk], [s['mu'] for s in pk], '^', ms=6, color=C['peak'], mec='k', mew=0.8,
            ls='none', label=tx('peak', 'пик'))
    ax.plot([s['L'] for s in ce], [s['mu'] for s in ce], 'o', ms=5, color=C['cep'], mec='k', mew=0.6,
            ls='none', label=tx('critical end point', 'критическая концевая точка'))
    ax.set_xlim(*L_lim); ax.set_ylim(*mu_lim)
    ax.set_xlabel('$L$'); ax.set_ylabel(r'$\mu$')


def fig1():
    print('fig1: transition lines')
    d = lines()
    fig, axes = plt.subplots(2, 1, figsize=(FIGW or 7.0, 8.0 * (FIGW or 7.0) / 7.0),
                             gridspec_kw=dict(height_ratios=[1.3, 1]))
    _draw_lines(axes[0], d, MU_RANGE, L_RANGE)
    _draw_lines(axes[1], d, (2.5, 4.0), L_RANGE)
    for x, y, t in [(8.5, 1.5, '$V$'), (8.5, 5.5, '$T$')]:
        axes[0].text(x, y, t, fontsize=13)
    axes[0].legend(loc='upper right', fontsize=8)
    if not JR:                       # in the journal version the legend of line types is in the caption
        axes[0].set_title(tx('Transition lines at $b=0$, $g=-1$: black — first order (to $T$), red — second order,\n'
                            'orange — $V$–$I_0$, blue — $I_n$–$I_{n+1}$; dotted — mode thresholds $\\mu=2\\pi n/L$',
                            'Линии переходов при $b=0$, $g=-1$: чёрные — 1-го рода (в $T$), красные — 2-го рода,\n'
                            'оранжевая — $V$–$I_0$, синие — $I_n$–$I_{n+1}$; пунктир — пороги мод $\\mu=2\\pi n/L$'),
                          fontsize=9)
    save(fig, 'fig1_transition_lines')


def tab_special():
    print('tab-special: special points')
    d = lines()
    rows = []
    for s in d['special']:
        if s['kind'] == 'tricritical':
            continue
        j = sum(1 for x in d['special'] if x['kind'] == 'critical_end' and x['L'] < s['L'])
        name = (tx(f"peak n = {s['n']}", f"пик $n={s['n']}$") if s['kind'] == 'peak'
                else tx(f'critical end point (birth of I_{j})', f'критическая концевая точка (рождение $I_{j}$)'))
        rows.append((name, s['L'], s['mu'] if s.get('mu') is not None else float('nan')))
    write_table('tab_special_points', [tx('point', 'Точка'), tx('L', '$L$'), tx('mu', r'$\mu$')], rows,
                ['{}', '{:.10f}', '{:.10f}'])


# ==============================================================================
# fig2: map in the peak region
# ==============================================================================

def fig2():
    print('fig2: M*(L, mu) in the peak region')
    n_mu, n_L = (100, 130) if QUICK else (300, 400)
    t = time.time()
    d = compute_grid(np.linspace(3.0, 3.4, n_mu), np.linspace(2.0, 10.0, n_L), G)
    print(f'  grid {n_mu}×{n_L}: {time.time() - t:.0f} s')
    fig, ax = plt.subplots(figsize=(FIGW or 7.5, 4.5 * (FIGW or 7.5) / 7.5))
    im = ax.pcolormesh(d['L'], d['mu'], d['M_star'].T, cmap='Greys' if BW else 'viridis', shading='auto',
                       vmin=0, vmax=3.5, rasterized=True)     # raster map inside the vector PDF (no seams)
    fig.colorbar(im, ax=ax, label='$M^*$', extend='max')
    pk = [(L, mu) for n, L, mu in peaks(G, 10.0) if 2 <= L <= 10]
    ax.plot(*zip(*pk), '^', ms=6, color=C['peak'], mec='k', mew=0.8, ls='none',
            label=tx('peak (explicit formula)', 'пик (явная формула)'))
    ax.set_xlabel('$L$'); ax.set_ylabel(r'$\mu$'); ax.legend(loc='upper right', fontsize=8)
    if not JR:
        ax.set_title(tx(r'Global minimum $M^*(L,\mu)$ at $b=0$, $g=-1$; yellow — vacuum phase',
                       r'Глобальный минимум $M^*(L,\mu)$ при $b=0$, $g=-1$; жёлтое — вакуумная фаза'), fontsize=9)
    save(fig, 'fig2_peak_region')


# ==============================================================================
# fig3: closed form against direct integration
# ==============================================================================

def _s(x):
    return -quad(lambda t: math.log(-math.expm1(-math.sqrt(t * t + x * x))), 0, math.inf, limit=200)[0]


def _omega_direct(M, mu, L):
    """Omega(M) from the defining integrals (adaptive quadrature), up to an M-independent constant."""
    oL = 4 / (math.pi**2 * L**2) * quad(lambda p: _s(L * math.sqrt(M * M + p * p)), 0, math.inf,
                                         limit=200, epsabs=1e-13, epsrel=1e-12)[0]
    om = 0.0
    for k, C in F.modes(mu, L):
        m = math.hypot(M, k)
        if m < mu:
            P = math.sqrt(mu * mu - m * m)
            om += C * 2 * quad(lambda p: mu - math.sqrt(p * p + m * m), 0, P, epsabs=1e-14)[0]
    return M * M / (2 * G) + F.dU(M) + oL - om / (math.pi * L)


def fig3():
    print('fig3: G(M), closed form against direct integration')
    mu, L = 3.2, 1.0
    Ms = np.geomspace(1e-4, 3.0, 14 if QUICK else 30)
    Gn = []
    for M in Ms:
        h = 1e-4 * M
        Gn.append((_omega_direct(M + h, mu, L) - _omega_direct(M - h, mu, L)) / (2 * h) / M)
    Gn = np.array(Gn)
    Mf = np.geomspace(1e-5, 3.0, 400)
    Ga = F.gap_G_vec(Mf, mu, L, G)
    fig, axes = plt.subplots(2, 1, figsize=(FIGW or 6.5, 6.0 * (FIGW or 6.5) / 6.5), sharex=True)
    axes[0].semilogx(Mf, Ga, '-', color=C['curve'], lw=1.8, label=tx('closed form', 'явная формула'))
    lo, hi = Ga.min() - 0.15, Ga.max() + 0.15
    inside = (Gn >= lo) & (Gn <= hi)
    axes[0].semilogx(Ms[inside], Gn[inside], 'o', ms=4, mfc='none', color=C['points'],
                     label=tx('finite differences of direct integration',
                             'конечные разности прямого численного интегрирования'))
    axes[0].semilogx(Ms[~inside], np.full((~inside).sum(), lo + 0.02), 'v', ms=5, color=C['points'],
                     label=tx('same, off scale (cancellation noise)', 'то же, вне шкалы (шум сокращения)'))
    axes[0].set_ylim(lo, hi)
    axes[0].axhline(0, color='0.5', lw=0.7); axes[0].set_ylabel('$G(M)$'); axes[0].legend(fontsize=8)
    if not JR:
        axes[0].set_title(rf'$b=0$, $\mu={mu}$, $L={L}$, $g=-1$', fontsize=9)
    axes[1].loglog(Ms, np.abs(Gn - F.gap_G_vec(Ms, mu, L, G)), 'o-', ms=4, color=C['points'])
    axes[1].set_xlabel('$M$'); axes[1].set_ylabel(tx('|difference|', '|разность|'))
    save(fig, 'fig3_gap_closed_vs_direct')


# ==============================================================================
# fig4: slices through a point
# ==============================================================================

def _slice(at, xs):
    """M* and phase along a slice; at(x) -> (mu, L)."""
    out = []
    for x in xs:
        mu, L = at(x)
        M = find_minimum(mu, L, G)[0]
        out.append((M, phase_label(M, mu, L)))
    return out


def _slice_transitions(at, xs, pts, tol=1e-12):
    """Phase changes along a slice: bisection to tol; order 2 if M* is continuous (M* -> 0)."""
    ph = lambda x: _slice(at, [x])[0]
    tr = []
    for k in range(len(xs) - 1):
        a, pa, b, pb = xs[k], pts[k][1], xs[k + 1], pts[k + 1][1]
        while pa != pb:
            lo, hi = a, b
            while hi - lo > tol * max(1.0, abs(hi)):
                c = 0.5 * (lo + hi)
                if ph(c)[1] == pa:
                    lo = c
                else:
                    hi = c
            eps = 1e-9 * max(1.0, abs(hi))
            Ml, Mr = ph(lo - eps)[0], ph(hi + eps)
            tr.append(dict(x=0.5 * (lo + hi), frm=pa, to=Mr[1], order=2 if abs(Ml - Mr[0]) < 1e-2 else 1))
            a, pa = hi + eps, Mr[1]
    return tr


def _draw_slice(ax, at, lo, hi, n, xlabel, mark):
    xs = np.linspace(lo, hi, n)
    pts = _slice(at, xs)
    tr = _slice_transitions(at, xs, pts)
    # линия M* с разрывами на переходах 1-го рода
    M = np.array([p[0] for p in pts])
    xx, MM = list(xs), list(M)
    for q in tr:
        if q['order'] == 1:
            i = int(np.searchsorted(xx, q['x']))
            xx.insert(i, q['x']); MM.insert(i, np.nan)
    ax.plot(xx, MM, '-', color=C['curve'], lw=1.8)
    for q in tr:
        ax.axvline(q['x'], color=C['second'] if q['order'] == 2 else C['thr'], lw=1.0,
                   ls='--' if q['order'] == 2 else ':')
    edges = [lo] + [q['x'] for q in tr] + [hi]
    top = np.nanmax(M) * 1.12
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > 0.035 * (hi - lo):
            m = 0.5 * (a + b)
            lab = _slice(at, [m])[0][1]
            lab = f'$I_{{{lab[1:]}}}$' if lab.startswith('I') else f'${lab}$'
            ax.text(m, top * 0.97, lab, ha='center', va='top', fontsize=10)
    ax.plot([mark[0]], [mark[1]], 'o', ms=8, mfc='none', mec='k', mew=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(0, top)
    ax.set_xlabel(xlabel); ax.set_ylabel('$M^*$')
    return tr


def fig4():
    print('fig4: slices M*(mu) at fixed L and M*(L) at fixed mu')
    mu0, L0 = 3.15, 3.5
    M0 = find_minimum(mu0, L0, G)[0]
    fig, axes = plt.subplots(1, 2, figsize=(FIGW or 9.0, 3.4 * (FIGW or 9.0) / 9.0 * 1.35))
    n = 300 if QUICK else 1200
    tr1 = _draw_slice(axes[0], lambda x: (x, L0), 2.9, 3.35, n, r'$\mu$', (mu0, M0))
    tr2 = _draw_slice(axes[1], lambda x: (mu0, x), 0.3, 10.0, n, '$L$', (L0, M0))
    axes[0].set_title(tx(f'$L={L0}$', f'$L={L0}$'), fontsize=10)
    axes[1].set_title(tx(rf'$\mu={mu0}$', rf'$\mu={mu0}$'), fontsize=10)
    fig.tight_layout()
    save(fig, 'fig4_slices')
    for name, trs in (('mu', tr1), ('L', tr2)):
        print(f'  transitions along {name}: ' + ', '.join(f"{q['frm']}->{q['to']} at {q['x']:.10f} ({q['order']})"
                                                          for q in trs))


# ==============================================================================
# tab-accuracy: double precision against mpmath (50 digits)
# ==============================================================================

def tab_accuracy():
    print('tab-accuracy: against mpmath (50 digits)')
    rng = np.random.default_rng(12345)
    n = 300 if QUICK else 3000
    names = [tx('M -> 0', r'$M\to0$'), tx('0 < M < mu', r'$0<M<\mu$'), tx('near a threshold', 'у порога моды'),
             tx('M > mu', r'$M>\mu$')]
    eG, eO, eOc = ([[] for _ in range(4)] for _ in range(3))
    eG0 = []
    for _ in range(n):
        mu = float(rng.uniform(0.1, 7.0))
        L = float(np.exp(rng.uniform(np.log(0.05), np.log(10.0))))
        kind = int(rng.integers(4))
        if kind == 0:
            M = float(np.exp(rng.uniform(np.log(1e-12), np.log(1e-2))))
        elif kind == 1:
            M = float(rng.uniform(0.0, mu))
        elif kind == 2:
            S = F.thresholds(mu, L)
            M = S[int(rng.integers(len(S)))] * (1.0 + float(rng.choice([-1, 1])) * 10.0 ** rng.uniform(-12, -2))
        else:
            M = float(rng.uniform(mu, mu + 5.0))
        if M <= 0:
            continue
        eG[kind].append(abs(F.gap_G(M, mu, L, G) - float(R.G(M, mu, L, G))))
        ref = float(R.omega_rel(M, mu, L, G))
        eO[kind].append(abs(F.omega_rel(M, mu, L, G) - ref))
        eOc[kind].append(abs(F.omega_rel_closed(M, mu, L, G) - ref) / max(abs(ref), 1e-300))
        eG0.append(abs(F.gap_G0(mu, L, G) - float(R.G0(mu, L, G))))
    rows = [(names[k], len(eG[k]), max(eG[k]), max(eO[k]), max(eOc[k])) for k in range(4) if eG[k]]
    rows.append((tx('G(0+)', '$G_0$'), len(eG0), max(eG0), float('nan'), float('nan')))

    # roots of G
    m = 60 if QUICK else 400
    er, nr = 0.0, 0
    for _ in range(m):
        mu = float(rng.uniform(0.5, 7.0))
        L = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
        for r, kind in isolate_roots(mu, L, G):
            if kind == 'touch':
                continue
            try:
                er = max(er, abs(r - float(R.root_near(r, mu, L, G))))
                nr += 1
            except ValueError:
                pass
    rows.append((tx(f'roots of G ({m} points)', f'корни $G$ ({m} точек)'), nr, er, float('nan'), float('nan')))

    # analytic lines
    Ls = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0][:: 3 if QUICK else 1]
    with mp.workdps(50):
        es = max(abs(mu_stability(L, G) - float(mp.findroot(lambda x: R.G0(x, L, G), mp.mpf(mu_stability(L, G)))))
                 for L in Ls)
        LV = [0.2, 0.5, 1.0]
        ev = []
        for L in LV:
            Mv = mp.findroot(lambda M: R.G(M, 1e-30, L, G), mp.mpf(M_vacuum(L, G)))
            ev.append(abs(mu_VT(L, G) - float(mp.findroot(lambda x: R.omega_rel(Mv, x, L, G), mp.mpf(mu_VT(L, G))))))
    rows.append((tx(f'mu_s(L) ({len(Ls)} values)', r'$\mu_s(L)$'), len(Ls), es, float('nan'), float('nan')))
    rows.append((tx(f'mu_VT(L) ({len(LV)} values)', r'$\mu_{VT}(L)$'), len(LV), max(ev), float('nan'), float('nan')))

    # first-order lines from the tracing
    d = lines()
    ef, nf = [], 0
    for key, pts in d['lines'].items():
        sel = [p for p in pts if p['order'] == 1]
        for p in sel[:: max(1, len(sel) // (2 if QUICK else 4))][: 2 if QUICK else 4]:
            try:
                ef.append(abs(p['mu'] - _mp_first_order(p['L'], p['mu'], p['M_left'], p['M_right'])))
                nf += 1
            except ValueError:
                pass
    rows.append((tx('first-order lines (traced)', 'линии 1-го рода (трассировка)'), nf,
                 max(ef) if ef else float('nan'), float('nan'), float('nan')))
    write_table('tab_accuracy', [tx('quantity', 'Величина'), tx('N', '$N$'),
                                 tx('max |error| of G or position', 'погрешность $G$ или положения'),
                                 tx('max |error| of Omega(M)-Omega(0), stable form',
                                   r'погрешность $\Omega(M)-\Omega(0)$, устойчивая форма'),
                                 tx('max rel. error of the closed-form difference',
                                   'отн. погрешность разности замкнутых формул')],
                rows, ['{}', '{:d}', '{:.1e}', '{:.1e}', '{:.1e}'])


def _mp_first_order(L, mu0, M_left, M_right):
    """First-order transition in mpmath: Omega on the two branches is equal (M = 0 is phase T)."""
    with mp.workdps(50):
        def branch(Mg, mu):
            return mp.mpf(0) if Mg == 0.0 else R.root_near(Mg, mu, L, G)

        def f(mu):
            a, b = branch(M_left, mu), branch(M_right, mu)
            return ((R.omega_rel(a, mu, L, G) if a != 0 else 0) - (R.omega_rel(b, mu, L, G) if b != 0 else 0))
        for w in ('1e-10', '1e-8'):
            a, b = mp.mpf(mu0) - mp.mpf(w), mp.mpf(mu0) + mp.mpf(w)
            if f(a) * f(b) < 0:
                return float(mp.findroot(f, (a, b), solver='anderson'))
    raise ValueError('no sign change near the transition')


# ==============================================================================
# tab-scan: certified search against a grid scan
# ==============================================================================

def tab_scan():
    print('tab-scan: certified search against a grid scan')
    MU = np.linspace(*MU_RANGE, 70 if QUICK else 279)
    L = np.linspace(*L_RANGE, 50 if QUICK else 250)
    n_s, n_c, d_om = (np.zeros((len(L), len(MU))) for _ in range(3))
    ts = tc = 0.0
    for i, l in enumerate(L):
        for j, mu in enumerate(MU):
            t = time.perf_counter(); Ms, Os, ms = find_minimum_scan(mu, l, G); ts += time.perf_counter() - t
            t = time.perf_counter(); Mc, Oc, mc = find_minimum(mu, l, G); tc += time.perf_counter() - t
            n_s[i, j], n_c[i, j], d_om[i, j] = len(ms), len(mc), Os - Oc
    N = n_s.size
    dmax = np.abs(d_om).max()
    rows = [(tx('grid points', 'точек сетки'), N, ''),
            (tx('scan misses a local minimum', 'скан пропускает локальный минимум'), int((n_c > n_s).sum()),
             f'{100 * (n_c > n_s).mean():.2f} %'),
            (tx('scan finds a spurious minimum', 'скан находит лишний минимум'), int((n_s > n_c).sum()), ''),
            (tx('wrong global minimum (|dOmega| > 1e-12)', r'неверный глобальный минимум ($|\Delta\Omega|>10^{-12}$)'),
             int((np.abs(d_om) > 1e-12).sum()),
             tx(f'max |dOmega| = {dmax:.1e}', r'$\max|\Delta\Omega|=' + _texnum(dmax) + '$')),
            (tx('time per point, scan (600-point grid + Brent)', 'время на точку, скан (600 узлов + Брент)'),
             tx(f'{ts / N * 1e3:.2f} ms', f'{ts / N * 1e3:.2f} мс'), ''),
            (tx('time per point, certified', 'время на точку, гарантированный метод'),
             tx(f'{tc / N * 1e3:.2f} ms', f'{tc / N * 1e3:.2f} мс'), '')]
    write_table('tab_scan_vs_certified', [tx('quantity', 'Величина'), tx('value', 'Значение'),
                                          tx('comment', 'Комментарий')], rows, ['{}', '{}', '{}'])


# ==============================================================================
# tab-perf: performance
# ==============================================================================

def _per_call(f, n):
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


def tab_perf():
    import platform
    import shutil
    import subprocess
    import scipy
    print('tab-perf: performance')
    rng = np.random.default_rng(7)
    n = 300 if QUICK else 2000
    pts = [(float(rng.uniform(*MU_RANGE)), float(rng.uniform(*L_RANGE))) for _ in range(n)]
    us = lambda s: tx(f'{s * 1e6:.1f} µs', f'{s * 1e6:.1f} мкс')
    ms = lambda s: tx(f'{s * 1e3:.2f} ms', f'{s * 1e3:.2f} мс')
    rows = []
    t_gap = np.mean([_per_call(lambda: F.gap_G(0.37 * mu, mu, L, G), 5) for mu, L in pts[:200]])
    rows.append((tx('G(M), one value (Python, double)', '$G(M)$, одно значение (Python)'), us(t_gap),
                 tx('closed form, stable', 'явная формула, устойчивая форма')))
    Mv = np.linspace(1e-3, 6.0, 1000)
    t_vec = _per_call(lambda: F.gap_G_vec(Mv, 4.0, 5.0, G), 20) / Mv.size
    rows.append((tx('G(M), vectorised over 1000 values of M', '$G(M)$, вектор из 1000 значений $M$'), us(t_vec),
                 tx('per value', 'на одно значение')))
    t_om = np.mean([_per_call(lambda: F.omega_rel(0.3 * min(mu, 1.5), mu, L, G), 3) for mu, L in pts[:200]])
    rows.append((tx('Omega(M) - Omega(0), stable form (M < 0.5)', r'$\Omega(M)-\Omega(0)$, устойчивая форма ($M<0.5$)'),
                 us(t_om), tx('Gauss-Legendre over m G(m)', r'Гаусс--Лежандр по $m\,G(m)$')))
    t_ref = np.mean([_per_call(lambda: R.G(0.37 * mu, mu, L, G), 1) for mu, L in pts[:50]])
    rows.append((tx('G(M), mpmath reference (50 digits)', '$G(M)$, эталон mpmath (50 цифр)'), ms(t_ref),
                 tx('for comparison', 'для сравнения')))
    t = time.perf_counter()
    for mu, L in pts:
        isolate_roots(mu, L, G)
    rows.append((tx('all roots of G, certified (one point)', 'все корни $G$, гарантированно (одна точка)'),
                 ms((time.perf_counter() - t) / n), tx(f'mean over {n} random points', f'среднее по {n} точкам')))
    t = time.perf_counter()
    for mu, L in pts:
        find_minimum(mu, L, G)
    rows.append((tx('global minimum, certified (one point)', 'глобальный минимум (одна точка)'),
                 ms((time.perf_counter() - t) / n),
                 tx('roots + Omega at every local minimum', r'корни и $\Omega$ во всех локальных минимумах')))
    nm = 150 if QUICK else 300
    t = time.time()
    compute_grid(np.linspace(*MU_RANGE, nm), np.linspace(*L_RANGE, nm), G)
    rows.append((tx(f'map {nm}×{nm} = {nm * nm} points', f'карта {nm}×{nm} = {nm * nm} точек'),
                 tx(f'{time.time() - t:.1f} s', f'{time.time() - t:.1f} с'),
                 tx(f'{os.cpu_count()} processes', f'{os.cpu_count()} процессов')))
    Ls = [0.5, 2.0, 5.0, 9.0]
    from njl3.b0.lines import trace_L
    t = time.time()
    for L in Ls:
        trace_L(L, G)
    rows.append((tx('transitions along mu at fixed L (400-point scan + bisection)',
                   r'переходы по $\mu$ при фиксированном $L$'),
                 tx(f'{(time.time() - t) / len(Ls):.2f} s', f'{(time.time() - t) / len(Ls):.2f} с'),
                 tx('one value of L', 'одно значение $L$')))
    d = lines()
    if 'seconds' in d:
        rows.append((tx(f"transition lines, {d['nL']} values of L, with special points",
                       f"линии переходов, {d['nL']} значений $L$, с особыми точками"),
                     tx(f"{d['seconds']:.0f} s", f"{d['seconds']:.0f} с"),
                     tx(f"{d.get('cores', '?')} processes", f"{d.get('cores', '?')} процессов")))
    node = shutil.which('node')
    if node:
        js = json.loads(subprocess.run([node, os.path.join(os.path.dirname(__file__), '..', 'web', 'b0', 'test',
                                                           'bench.js'), str(n)],
                                       capture_output=True, text=True, check=True).stdout)
        rows.append((tx('G(M), one value (JS, Node.js)', '$G(M)$, одно значение (JS, Node.js)'),
                     us(js['gapG_us'] * 1e-6), f"Node {js['node']}"))
        rows.append((tx('global minimum, certified (JS, one point)', 'глобальный минимум (JS, одна точка)'),
                     ms(js['findMinimum_us'] * 1e-6),
                     tx('the web app runs this in background workers', 'так считает веб-приложение')))
    env = (f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores; Python {platform.python_version()}, "
           f"NumPy {np.__version__}, SciPy {scipy.__version__}, mpmath {mp.__version__}")
    rows.append((tx('machine', 'компьютер'), '', env))
    write_table('tab_performance', [tx('operation', 'Операция'), tx('time', 'Время'), tx('comment', 'Комментарий')],
                rows, ['{}', '{}', '{}'])


def _texnum(x):
    m, e = f'{x:.1e}'.split('e')
    return f'{m}\\cdot10^{{{int(e)}}}'


# ==============================================================================

TARGETS = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'tab-special': tab_special,
           'tab-accuracy': tab_accuracy, 'tab-scan': tab_scan,
           'tab-perf': tab_perf}


def main():
    os.makedirs(OUT, exist_ok=True)
    chosen = [a for a in sys.argv[1:] if not a.startswith('--')] or list(TARGETS)
    for a in chosen:
        if a not in TARGETS:
            sys.exit(f'unknown target {a!r}; choose from {", ".join(TARGETS)}')
    t0 = time.time()
    for a in chosen:
        t = time.time()
        TARGETS[a]()
        print(f'  [{a}: {time.time() - t:.0f} s]')
    print(f'done in {time.time() - t0:.0f} s; output in {OUT}')


if __name__ == '__main__':
    main()

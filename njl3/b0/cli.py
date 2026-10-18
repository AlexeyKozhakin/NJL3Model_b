"""
Command line njl3-b0: phase diagram of NJL3 at b = 0.

    njl3-b0 point   --mu 3.5 --L 2                       # one point: phase, minima, roots of G
    njl3-b0 peaks   --L-max 10                           # peaks mu_n = 2 pi n / L_n (explicit)
    njl3-b0 diagram --mu 2.5 4 --L 0.1 10 --n 200 200 --out map.npz --plot map.png
    njl3-b0 lines   --mu 0.05 7 --L 0.1 10 --nL 400 --out lines.json --plot lines.png

Default g = -1. Output formats: diagram — .npz / .json / .csv, lines — .json.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np


def _save_grid(d, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        np.savez(path, **d)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}, f)
    elif ext == '.csv':
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['L', 'mu', 'M_star', 'omega_star', 'phase', 'n_minima'])
            for i, L in enumerate(d['L']):
                for j, mu in enumerate(d['mu']):
                    w.writerow([L, mu, d['M_star'][i, j], d['omega_star'][i, j], d['phase'][i, j],
                                d['n_minima'][i, j]])
    else:
        sys.exit(f'Unknown format: {path} (use .npz, .json or .csv)')


def _plot_grid(d, path, vmax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.pcolormesh(d['L'], d['mu'], d['M_star'].T, cmap='viridis', shading='auto', vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label='M*', extend='max' if vmax else 'neither')
    ax.set_xlabel('L'); ax.set_ylabel('μ')
    ax.set_title(f"M*(L, μ) at b = 0, g = {d['g']:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _plot_lines(out, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    dL = (out['L_range'][1] - out['L_range'][0])
    for key, pts in out['lines'].items():
        a, b = key.split('-')
        for order in (1, 2):
            sel = sorted([p for p in pts if p['order'] == order], key=lambda p: p['L'])
            if not sel:
                continue
            Ls = np.array([p['L'] for p in sel]); ms = np.array([p['mu'] for p in sel])
            step = np.median(np.diff(Ls)) if len(Ls) > 1 else dL
            color = 'r' if order == 2 else ('tab:orange' if a == 'V' and b != 'T'
                                           else 'tab:blue' if b != 'T' else 'k')
            for seg in np.split(np.arange(len(Ls)), np.nonzero(np.diff(Ls) > 2.5 * step)[0] + 1):
                ax.plot(Ls[seg], ms[seg], color=color, lw=1.8)
    marks = {'peak': ('^', 'orange'), 'critical_end': ('o', 'cyan')}
    for kind, (mk, col) in marks.items():
        p = [s for s in out['special'] if s['kind'] == kind and s.get('mu') is not None]
        if p:
            ax.plot([s['L'] for s in p], [s['mu'] for s in p], mk, color=col, mec='k', ls='none', label=kind)
    ax.set_xlim(*out['L_range']); ax.set_ylim(*out['mu_range'])
    ax.set_xlabel('L'); ax.set_ylabel('μ')
    ax.set_title(f"Transition lines at b = 0, g = {out['g']:g}  (red: second order; other colours: first order)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def cmd_point(a):
    from njl3.b0 import analyze_point
    r = analyze_point(a.mu, a.L, a.g)
    if a.json:
        print(json.dumps(r.to_dict(), indent=1))
        return
    print(f"μ = {a.mu}, L = {a.L}, g = {a.g}")
    print(f"  phase       : {r.phase}")
    print(f"  M*          : {r.M_star:.15g}")
    print(f"  Ω(M*) − Ω(0): {r.omega_star:.6e}")
    print(f"  G(0)        : {r.G0:+.10f}   ({'M = 0 locally stable' if r.G0 > 0 else 'M = 0 unstable'})")
    print(f"  G2          : {r.G2:+.10f}")
    print("  roots of G  : " + (', '.join(f'{m:.12g} ({k})' for m, k in r.roots) or '—'))
    print("  minima      : " + (', '.join(f'M={m:.10g}, ΔΩ={o:.4e}' for m, o in r.local_minima) or '—'))


def cmd_peaks(a):
    from njl3.b0 import peaks
    print(f"{'n':>3}  {'L_n':>16}  {'mu_n':>16}  {'mu_n - pi/|g|':>14}")
    for n, L, mu in peaks(a.g, a.L_max):
        print(f"{n:3d}  {L:16.12f}  {mu:16.12f}  {mu - np.pi / abs(a.g):14.6e}")


def cmd_diagram(a):
    from njl3.b0 import compute_grid
    t = time.time()
    d = compute_grid(np.linspace(*a.mu, a.n[0]), np.linspace(*a.L, a.n[1]), a.g, jobs=a.jobs)
    d['g'] = a.g
    print(f"Map {a.n[0]}×{a.n[1]}: {time.time() - t:.1f} s")
    if a.out:
        _save_grid(d, a.out)
        print(f"  data  : {a.out}")
    if a.plot:
        _plot_grid(d, a.plot, a.vmax)
        print(f"  figure: {a.plot}")


def cmd_lines(a):
    from njl3.b0 import trace_lines
    t = time.time()
    out = trace_lines(np.linspace(*a.L, a.nL), a.g, a.mu[0], a.mu[1], a.n_mu, jobs=a.jobs)
    print(f"Lines for {a.nL} values of L: {time.time() - t:.1f} s")
    for key, pts in out['lines'].items():
        o = [p['order'] for p in pts]
        print(f"  {key:7s}: {len(pts)} points (first order {o.count(1)}, second order {o.count(2)})")
    for s in out['special']:
        mu = f"{s['mu']:.10f}" if s.get('mu') is not None else '—'
        note = '  (not on a phase boundary)' if s['kind'] == 'tricritical' and not s.get('on_phase_boundary') else ''
        print(f"  {s['kind']:13s} L = {s['L']:.10f}  μ = {mu}{note}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"  data  : {a.out}")
    if a.plot:
        _plot_lines(out, a.plot)
        print(f"  figure: {a.plot}")


def main(argv=None):
    p = argparse.ArgumentParser(prog='njl3-b0', description='Phase diagram of the NJL3 model on R²×S¹ at b = 0.')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--g', type=float, default=-1.0, help='coupling constant (g < 0), default -1')
        return sp

    sp = common(sub.add_parser('point', help='one point (mu, L): phase, minima, roots of G'))
    sp.add_argument('--mu', type=float, required=True, help='chemical potential')
    sp.add_argument('--L', type=float, required=True, help='length of the compact dimension')
    sp.add_argument('--json', action='store_true', help='print the full result as JSON')
    sp.set_defaults(func=cmd_point)

    sp = common(sub.add_parser('peaks', help='peaks of the phase boundary (explicit formula)'))
    sp.add_argument('--L-max', type=float, default=10.0, help='largest L_n to list')
    sp.set_defaults(func=cmd_peaks)

    sp = common(sub.add_parser('diagram', help='map of M*(mu, L) on a grid'))
    sp.add_argument('--mu', nargs=2, type=float, default=[0.05, 7.0], metavar=('MU_MIN', 'MU_MAX'))
    sp.add_argument('--L', nargs=2, type=float, default=[0.1, 4.0], metavar=('L_MIN', 'L_MAX'))
    sp.add_argument('--n', nargs=2, type=int, default=[200, 200], metavar=('N_MU', 'N_L'), help='grid size')
    sp.add_argument('--jobs', type=int, default=None, help='worker processes (default: all cores)')
    sp.add_argument('--out', help='save the map (.npz, .json or .csv)')
    sp.add_argument('--plot', help='save a figure (needs matplotlib)')
    sp.add_argument('--vmax', type=float, default=None, help='upper limit of the colour scale')
    sp.set_defaults(func=cmd_diagram)

    sp = common(sub.add_parser('lines', help='transition lines and special points'))
    sp.add_argument('--mu', nargs=2, type=float, default=[0.05, 7.0], metavar=('MU_MIN', 'MU_MAX'))
    sp.add_argument('--L', nargs=2, type=float, default=[0.1, 10.0], metavar=('L_MIN', 'L_MAX'))
    sp.add_argument('--nL', type=int, default=200, help='number of L values')
    sp.add_argument('--n-mu', type=int, default=400, help='scan points in mu (the step limits the resolution)')
    sp.add_argument('--jobs', type=int, default=None, help='worker processes (default: all cores)')
    sp.add_argument('--out', help='save the lines (.json)')
    sp.add_argument('--plot', help='save a figure (needs matplotlib)')
    sp.set_defaults(func=cmd_lines)

    a = p.parse_args(argv)
    if getattr(a, 'L', None) is not None:
        L_lo = a.L if isinstance(a.L, float) else a.L[0]
        if L_lo <= 0:
            p.error('L must be > 0')
    if a.g >= 0:
        p.error('only g < 0 is implemented')
    a.func(a)


if __name__ == '__main__':
    main()

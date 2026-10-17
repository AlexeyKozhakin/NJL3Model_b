"""
Данные для веб-приложения: точные линии переходов при b = 0, g = -1.

    python web/b0/build_data.py                      # расчёт (несколько минут)
    python web/b0/build_data.py --from lines.json    # из готового вывода `njl3-b0 lines --out`

Пишет web/b0/data/phase_lines.json в компактном формате:
    {"g", "mu_range", "L_range", "nL",
     "lines": {"V-T": [[L, mu, order], ...], ...},
     "special": [{"kind", "L", "mu", ...}, ...]}
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'phase_lines.json')


def compact(full, nL):
    r = lambda x: float(f'{x:.12g}')
    lines = {k: [[r(p['L']), r(p['mu']), p['order']] for p in v] for k, v in full['lines'].items()}
    special = [{k: (r(v) if isinstance(v, float) else v) for k, v in s.items()} for s in full['special']]
    return dict(g=full['g'], mu_range=full['mu_range'], L_range=full['L_range'], nL=nL,
                lines=lines, special=special)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='src')
    ap.add_argument('--nL', type=int, default=500)
    a = ap.parse_args()
    if a.src:
        with open(a.src) as f:
            full = json.load(f)
        nL = a.nL
    else:
        from njl3.b0 import trace_lines
        full = trace_lines(np.linspace(0.1, 10.0, a.nL), -1.0, 0.05, 7.0, 400)
        nL = a.nL
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(compact(full, nL), f, separators=(',', ':'))
    print(f'{OUT}: {os.path.getsize(OUT) / 1024:.0f} КБ')


if __name__ == '__main__':
    main()

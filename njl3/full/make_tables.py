"""
Генерация таблицы универсальной функции s(x) для Omega_L при b ≠ 0.

    s(x) = -∫_0^∞ ln(1 - e^{-sqrt(t² + x²)}) dt = Σ_k (x/k) K1(k x),
    s(x) = π²/6 - πx/2 - (x²/4)[ln(x/4π) + γ - 1/2] + ζ(3) x⁴/(64π²) + O(x⁶)   (x -> 0).

Кусочно-чебышёвская аппроксимация на [0, X_MAX]; на первом отрезке аппроксимируется
гладкая часть r(x) = s(x) + (x²/4) ln x. Значения в узлах — mpmath (40 цифр).
Выход: njl3/full/data/s_table.json (тот же файл использует веб-приложение).

    python -m njl3.full.make_tables
"""

import json
import os

import mpmath as mp
import numpy as np

mp.mp.dps = 40
EDGES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
DEG = 34
OUT = os.path.join(os.path.dirname(__file__), 'data', 's_table.json')


def s_mp(x):
    x = mp.mpf(x)
    if x == 0:
        return mp.pi**2 / 6
    if x > mp.mpf('0.4'):
        tot, k = mp.mpf(0), 1
        while True:
            t = x / k * mp.besselk(1, k * x)
            tot += t
            if t < mp.mpf('1e-45') * tot:
                return tot
            k += 1
    f = lambda t: -mp.log(1 - mp.exp(-mp.sqrt(t * t + x * x)))
    return mp.quad(f, [0, x, 1, 5, 20, mp.inf])


def piece(a, b, first):
    k = np.arange(DEG + 1)
    nodes = np.cos(np.pi * (k + 0.5) / (DEG + 1))           # узлы Чебышёва на [-1, 1]
    xs = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    vals = []
    for x in xs:
        v = s_mp(x)
        if first:
            v = v + (mp.mpf(x)**2 / 4) * mp.log(mp.mpf(x))
        vals.append(float(v))
    coef = np.polynomial.chebyshev.chebfit(nodes, np.array(vals), DEG)
    return [float(c) for c in coef]


def main():
    pieces = [dict(a=a, b=b, coef=piece(a, b, i == 0)) for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:]))]
    table = dict(description='s(x) = -∫_0^∞ ln(1 - exp(-sqrt(t²+x²))) dt; piece 0 fits s(x) + (x²/4) ln x',
                 x_max=EDGES[-1], pieces=pieces)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(table, f)
    print(f'{OUT}: {len(pieces)} кусков, степень {DEG}')


if __name__ == '__main__':
    main()

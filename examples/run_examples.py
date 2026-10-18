"""
Test-run examples for njl3.b0 (the inputs a referee is expected to run).

    python examples/run_examples.py            # all examples, output in examples/output/
    python examples/run_examples.py --fast     # skip the slow one (transition lines, ~1-2 min)
    python examples/run_examples.py --update   # regenerate examples/expected/ (maintainers only)

Each example is a command of the `njl3-b0` command line (run as `python -m njl3.b0`) or
a short Python script; its output is compared with examples/expected/ number by number
(relative tolerance 1e-9, see compare()). The same check runs in tests/b0/test_examples.py.
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
EXPECTED = os.path.join(HERE, 'expected')

# name, arguments of `python -m njl3.b0` (or a script), output file (None: stdout), slow
EXAMPLES = [
    ('01_point', ['point', '--mu', '3.15', '--L', '3.5'], None, False),
    ('02_point_json', ['point', '--mu', '3.5', '--L', '2', '--json'], None, False),
    ('03_peaks', ['peaks', '--L-max', '10'], None, False),
    ('04_diagram', ['diagram', '--mu', '2.5', '4', '--L', '0.1', '4', '--n', '60', '60',
                    '--out', '{out}/04_diagram.csv'], '04_diagram.csv', False),
    ('05_api', [os.path.join(HERE, 'api_example.py')], None, False),
    ('06_lines', ['lines', '--mu', '0.05', '7', '--L', '0.1', '10', '--nL', '40',
                  '--out', '{out}/06_lines.json'], '06_lines.json', True),
]

_NUM = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def run(name, args, outfile, out_dir):
    """Run one example; returns the text to compare (stdout or the output file)."""
    args = [a.replace('{out}', out_dir) for a in args]
    cmd = [sys.executable, args[0]] if args[0].endswith('.py') else [sys.executable, '-m', 'njl3.b0', *args]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if outfile:
        with open(os.path.join(out_dir, outfile)) as f:
            return f.read()
    # timings vary between runs and machines
    return re.sub(r': [\d.]+ s$', ': <time> s', res.stdout, flags=re.M)


def compare(got, expected, rtol=1e-9, atol=1e-12):
    """Same text up to numbers; numbers agree within rtol/atol. Returns a list of mismatches."""
    if expected.lstrip().startswith(('{', '[')):
        got, expected = json.dumps(json.loads(got), sort_keys=True), json.dumps(json.loads(expected), sort_keys=True)
    gn, en = _NUM.findall(got), _NUM.findall(expected)
    bad = []
    if _NUM.sub('#', got) != _NUM.sub('#', expected):
        bad.append('text differs (besides numbers)')
    if len(gn) != len(en):
        return bad + [f'{len(gn)} numbers instead of {len(en)}']
    for g, e in zip(gn, en):
        g, e = float(g), float(e)
        if abs(g - e) > atol + rtol * abs(e):
            bad.append(f'{g!r} != {e!r}')
            if len(bad) > 10:
                break
    return bad


def main():
    fast, update = '--fast' in sys.argv, '--update' in sys.argv
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for name, args, outfile, slow in EXAMPLES:
        if slow and fast:
            print(f'{name}: skipped (slow)')
            continue
        got = run(name, args, outfile, OUT)
        ref = os.path.join(EXPECTED, outfile or f'{name}.txt')
        if update:
            with open(ref, 'w') as f:
                f.write(got)
            print(f'{name}: expected output written')
            continue
        if outfile is None:
            with open(os.path.join(OUT, f'{name}.txt'), 'w') as f:
                f.write(got)
        with open(ref) as f:
            bad = compare(got, f.read())
        ok &= not bad
        print(f"{name}: {'OK' if not bad else 'MISMATCH: ' + '; '.join(bad)}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

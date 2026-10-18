"""The test-run examples (examples/) reproduce their expected output."""

import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_spec = importlib.util.spec_from_file_location('run_examples', os.path.join(ROOT, 'examples', 'run_examples.py'))
EX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EX)


@pytest.mark.parametrize('name,args,outfile,slow', [
    pytest.param(*e, marks=pytest.mark.slow) if e[3] else e for e in EX.EXAMPLES], ids=[e[0] for e in EX.EXAMPLES])
def test_example(name, args, outfile, slow, tmp_path):
    got = EX.run(name, args, outfile, str(tmp_path))
    with open(os.path.join(EX.EXPECTED, outfile or f'{name}.txt')) as f:
        assert EX.compare(got, f.read()) == []


def test_compare_detects_changes():
    assert EX.compare('M* = 1.000000000001', 'M* = 1.0') == []
    assert EX.compare('M* = 1.0001', 'M* = 1.0') != []
    assert EX.compare('phase I1', 'phase T') != []

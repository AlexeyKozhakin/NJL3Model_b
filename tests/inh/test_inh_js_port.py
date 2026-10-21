"""JS port web/lab/js/njl3inh.js and the worker vs Python njl3.inh (web/lab/test/expected_inh.json)."""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node.js')
@pytest.mark.parametrize('script', ['compare_inh.js', 'worker_inh_smoke.mjs'])
def test_js(script):
    res = subprocess.run(['node', os.path.join(ROOT, 'web', 'lab', 'test', script)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

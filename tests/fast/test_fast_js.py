"""JS port web/fast/js/fastmin.js vs Python njl3.fast (web/fast/test/expected_fast.json)."""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node.js')
def test_fast_js_matches_python():
    res = subprocess.run(['node', os.path.join(ROOT, 'web', 'fast', 'test', 'compare_fast.mjs')], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

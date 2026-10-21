"""JS port web/lab/js/njl3hom.js vs Python njl3.hom (reference values in web/lab/test/expected_hom.json)."""

import os
import shutil
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node.js')
def test_js_port_matches_python():
    res = subprocess.run(['node', os.path.join(ROOT, 'web', 'lab', 'test', 'compare_hom.js')],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

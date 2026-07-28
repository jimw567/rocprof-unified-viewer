"""Syntax-check the frontend JS with `node --check`. The overlay's JavaScript used to live
as string fragments inside Python, so JS errors (a missing `+`, an out-of-scope var, a
`.toFixed` on undefined) were runtime-only and shipped silently. Now it is a real .js file
(js/overlay.js, inlined at build time by render_html); this test parses it so those bugs
fail the build instead of the browser.

Skips cleanly when node is absent (e.g. the dev VM). CI (ubuntu-latest) has node
preinstalled, so it runs there -- see .github/workflows/ci.yml.
"""
import glob
import os
import shutil
import subprocess
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS_DIR = os.path.join(ROOT, "js")

# Build-time placeholders the Python inliner fills (see render_html). Substitute valid JS
# stand-ins so `node --check` sees syntactically complete source.
_PLACEHOLDERS = {"__DATA__": "{}"}


def _js_files():
    return sorted(glob.glob(os.path.join(JS_DIR, "*.js")))


def test_js_dir_exists():
    assert _js_files(), "no js/*.js files found -- frontend JS should be extracted"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_files_parse():
    bad = []
    for path in _js_files():
        src = open(path, encoding="utf-8").read()
        for ph, val in _PLACEHOLDERS.items():
            src = src.replace(ph, val)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tf:
            tf.write(src)
            tmp = tf.name
        try:
            r = subprocess.run(["node", "--check", tmp],
                               capture_output=True, text=True)
            if r.returncode != 0:
                bad.append("%s:\n%s" % (os.path.basename(path), r.stderr.strip()))
        finally:
            os.unlink(tmp)
    assert not bad, "node --check failed:\n" + "\n".join(bad)

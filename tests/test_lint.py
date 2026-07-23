"""Lint gates enforced in CI: the modules compile, and every tracked Python source is
pure ASCII (this repo has a hard ASCII-only rule -- no en/em-dashes or smart quotes)."""
import os
import py_compile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODULES = ["rocprof_unified_viewer.py", "serve.py", "disasm_loadwidth.py"]
# isa_glossary.py is a large generated data table; include it in the ASCII sweep too.
ASCII_FILES = MODULES + ["isa_glossary.py", os.path.join("tests", "test_smoke.py"),
                         os.path.join("tests", "test_lint.py"),
                         os.path.join("tests", "make_fixtures.py")]


def test_modules_compile():
    for m in MODULES:
        py_compile.compile(os.path.join(ROOT, m), doraise=True)


def test_sources_are_ascii():
    bad = []
    for rel in ASCII_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            for lineno, line in enumerate(fh, 1):
                for col, byte in enumerate(line, 1):
                    if byte > 0x7F:
                        bad.append("%s:%d:%d byte 0x%02x" % (rel, lineno, col, byte))
                        break
    assert not bad, "non-ASCII bytes found:\n" + "\n".join(bad)

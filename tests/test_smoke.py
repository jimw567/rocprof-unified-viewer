"""Smoke test: the generator still produces a well-formed overlay from the committed
fixtures. Runs offline (no GPU / no rocprofv3) so it works on GitHub-hosted CI.

Also guards the provenance stamp (Part A) by asserting it lands in the embedded payload.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")
SCRIPT = os.path.join(ROOT, "rocprof_unified_viewer.py")


def _gen(tmp_path, *extra):
    """Run the generator on the decode fixtures into tmp_path/out.html; return the HTML."""
    out = os.path.join(str(tmp_path), "out.html")
    cmd = [
        sys.executable, SCRIPT, "--mode", "decode",
        "--kernel-csv", os.path.join(FIX, "decode_kernel_trace.csv"),
        "--hip-csv", os.path.join(FIX, "decode_hip_api_trace.csv"),
        "--pmc-csv", os.path.join(FIX, "decode_stall_counter_collection.csv"),
        "--fetch-csv", os.path.join(FIX, "decode_fetch_counter_collection.csv"),
        "--skip-tokens", "0", "--tokens", "2",
        "--out", out,
    ] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "generator failed:\n%s\n%s" % (r.stdout, r.stderr)
    assert os.path.exists(out) and os.path.getsize(out) > 0, "no/empty output HTML"
    with open(out) as fh:
        return fh.read()


def _raw(html):
    """Extract the embedded RAW payload JSON from the overlay."""
    m = re.search(r"const RAW = (\{.*?\});\n", html, re.S)
    assert m, "no embedded RAW payload in overlay"
    return json.loads(m.group(1))


def test_generates_wellformed_overlay(tmp_path):
    html = _gen(tmp_path)
    # End-to-end sentinels: the payload, the summary table, the arch roofline tag.
    for marker in ("const RAW =", 'id="tbl"', "230 GB/s", "DECODE"):
        assert marker in html, "missing marker: %s" % marker


def test_payload_has_provenance(tmp_path):
    raw = _raw(_gen(tmp_path))
    prov = raw.get("provenance")
    assert prov, "payload missing provenance stamp"
    for k in ("version", "git_sha", "generated_utc", "host", "python"):
        assert k in prov, "provenance missing key: %s" % k
    assert prov["version"], "provenance version is empty"


def test_pmc_coloring_present(tmp_path):
    # With --pmc-csv the family diagnosis should render a ladder verdict; decode matvec
    # is DRAM-bound, so the memory rung must appear somewhere in the overlay.
    html = _gen(tmp_path)
    assert "DRAM-BOUND" in html, "expected a DRAM-bound ladder verdict from PMC fixtures"


def test_model_name_field(tmp_path):
    # No --gguf fixture (real model is multi-GB), so we can only assert the key exists
    # and is empty here; the frontend guard must gate the sub-line on it either way.
    html = _gen(tmp_path)
    raw = _raw(html)
    assert "model_name" in raw, "payload missing model_name field"
    assert raw["model_name"] == "", "model_name should be empty without --gguf"
    assert "D.model_name ?" in html, "frontend sub-line must guard on D.model_name"


def test_unmapped_family_lists_dispatches(tmp_path):
    # A family that does not stream a GGUF weight (e.g. k_bin_bcast) has no order-map,
    # but the family panel must still list its dispatches instead of dead-ending. The
    # fixtures contain such families; assert the "unmapped" fallback branch is present
    # and a non-weight family (k_bin_bcast) actually appears in the payload.
    html = _gen(tmp_path)
    raw = _raw(html)
    fams = {s["fam"] for s in raw["gpu"]}
    assert "k_bin_bcast" in fams, "fixture should contain the unmapped k_bin_bcast family"
    assert ", unmapped)</" in html, "frontend must render an unmapped-dispatch table branch"

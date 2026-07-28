"""Golden-payload regression: the RAW payload build_payload emits for a fixed decode AND
prefill fixture must stay byte-identical (as sorted-key JSON) across refactors. This is the
safety net for the viewer re-architecture -- any structural move (regime split, module
extraction, JS relocation) must preserve behavior, and this test proves it.

Nondeterministic / environment-specific keys (provenance, absolute input paths) are
stripped before compare. To intentionally update the goldens after a real behavior change,
run:  UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")
GOLD = os.path.join(HERE, "golden")
SCRIPT = os.path.join(ROOT, "rocprof_unified_viewer.py")

# Keys whose values depend on the environment / run, not the logic under test.
# att_cmd embeds the (tmp) output path + absolute regen flags -- a UI convenience string,
# path-dependent, no logic worth pinning.
_VOLATILE = {"provenance", "kernel_csv", "hip_csv", "pmc_csv", "fetch_csv",
             "loadwidth_json", "gguf", "att_dir", "graph_json", "clean_tps_file",
             "att_cmd", "regen_cmd", "regen_flags"}


def _strip(obj):
    """Recursively drop volatile keys so the compare reflects logic, not paths/timestamps."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _gen_payload(tmp_path, name, extra):
    out = os.path.join(str(tmp_path), name + ".html")
    cmd = [sys.executable, SCRIPT] + extra + ["--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, "generator failed:\n%s\n%s" % (r.stdout, r.stderr)
    html = open(out).read()
    m = re.search(r"const RAW = (\{.*?\});\n", html, re.S)
    assert m, "no RAW payload in overlay"
    return _strip(json.loads(m.group(1)))


def _check(tmp_path, name, extra):
    payload = _gen_payload(tmp_path, name, extra)
    gold_path = os.path.join(GOLD, name + "_payload.json")
    text = json.dumps(payload, sort_keys=True, indent=1)
    if os.environ.get("UPDATE_GOLDEN"):
        os.makedirs(GOLD, exist_ok=True)
        with open(gold_path, "w") as fh:
            fh.write(text + "\n")
        return
    assert os.path.exists(gold_path), (
        "no golden for %s; capture with UPDATE_GOLDEN=1 pytest" % name)
    want = open(gold_path).read().rstrip("\n")
    assert text == want, (
        "%s payload changed vs golden. If intentional, re-capture with "
        "UPDATE_GOLDEN=1 pytest tests/test_golden.py" % name)


def test_decode_payload_golden(tmp_path):
    _check(tmp_path, "decode", [
        "--mode", "decode",
        "--kernel-csv", os.path.join(FIX, "decode_kernel_trace.csv"),
        "--hip-csv", os.path.join(FIX, "decode_hip_api_trace.csv"),
        "--pmc-csv", os.path.join(FIX, "decode_stall_counter_collection.csv"),
        "--fetch-csv", os.path.join(FIX, "decode_fetch_counter_collection.csv"),
        "--graph-json", os.path.join(FIX, "decode_graph.json"),
        "--skip-tokens", "0", "--tokens", "2",
    ])


def test_prefill_payload_golden(tmp_path):
    _check(tmp_path, "prefill", [
        "--mode", "prefill",
        "--kernel-csv", os.path.join(FIX, "prefill_kernel_trace.csv"),
        "--hip-csv", os.path.join(FIX, "prefill_hip_api_trace.csv"),
        "--pmc-csv", os.path.join(FIX, "prefill_stall_counter_collection.csv"),
        "--fetch-csv", os.path.join(FIX, "prefill_fetch_counter_collection.csv"),
        "--clean-tps-file", os.path.join(FIX, "prefill_clean_tps.json"),
        "--loadwidth-json", os.path.join(FIX, "prefill_loadwidth.json"),
    ])

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


def test_layer_graph_input(tmp_path):
    # --graph-json attaches the ggml compute graph. Assert the payload carries it,
    # the node's block index L is parsed from the blk.<N>. name prefix, and the
    # frontend layer-graph popup function is present in the overlay.
    html = _gen(tmp_path, "--graph-json", os.path.join(FIX, "decode_graph.json"))
    raw = _raw(html)
    assert raw.get("has_layer_graph"), "payload missing has_layer_graph"
    lg = raw["layer_graph"]
    assert lg["n_nodes"] >= 1, "layer_graph has no nodes"
    by_layer = lg["by_layer"]
    # blk.0.* -> L0, blk.1.* -> L1, non-block (output/inp_embd) -> L-1. JSON keys
    # are strings.
    assert "0" in by_layer and "1" in by_layer, "block layers not indexed"
    assert "-1" in by_layer, "non-block nodes should bucket under L-1"
    assert "function openLayerGraph" in html, "frontend missing openLayerGraph popup"
    assert "D.has_layer_graph" in html, "frontend must gate layer click on D.has_layer_graph"
    # A trace-derived graph flags its edges as inferred; the loader must pass that
    # through so the popup can honestly label edge trust.
    assert lg.get("edges_inferred") is True, "edges_inferred flag not passed through"
    assert "inferred edges" in html, "frontend missing inferred-edges honesty banner"
    # Graph-view fusion analysis: the popup must carry the analysis fn + its model inputs.
    assert "function analyzeFusion" in html, "popup missing fusion analysis"
    assert "fam_counters:D.fam_counters" in html, "popup missing fam_counters for fusion model"


def test_roofline_topology_reconstruction(tmp_path):
    # A ggml-roofline artifact with per-invocation storage-id topology (PR #66) must
    # reconstruct FAITHFUL edges via last-writer-wins -- no inferred flag. Build a tiny
    # 2-node artifact: op A writes storage 100; op B reads storage 100 -> edge B<-A.
    import json as _json
    art = {"step": "decode", "ops": [
        {"invocation": 1, "name": "blk.0.attn_norm", "ggml_op": "RMS_NORM",
         "out_storage_id": 100, "in_storage_ids": [],
         "kernels": [{"name": "rms", "gpu_time_us": 5.0}]},
        {"invocation": 2, "name": "blk.0.attn_q", "ggml_op": "MUL_MAT",
         "out_storage_id": 101, "in_storage_ids": [100],
         "kernels": [{"name": "mm", "gpu_time_us": 50.0}]},
    ]}
    fp = os.path.join(str(tmp_path), "roofline.json")
    with open(fp, "w") as fh:
        _json.dump(art, fh)
    html = _gen(tmp_path, "--graph-json", fp)
    lg = _raw(html)["layer_graph"]
    assert lg["edges_inferred"] is False, "roofline path must NOT be flagged inferred"
    nodes = {n["name"]: n for n in lg["nodes"]}
    q = nodes["blk.0.attn_q"]
    norm = nodes["blk.0.attn_norm"]
    assert q["src"] == [norm["i"]], "edge not reconstructed via last-writer-wins"
    assert "function layerGraphUnavailable" in html, "missing ask-maintainer fallback"


def test_checked_in_topologies_are_valid():
    # The checked-in arch-keyed topology skeletons must load, be model-name-free, and
    # expand across layers with faithful (non-inferred) edges. This guards the ~KB
    # data assets that give faithful graphs for all covered architectures with no dump.
    import importlib.util as _il
    spec = _il.spec_from_file_location("ruv", os.path.join(ROOT, "rocprof_unified_viewer.py"))
    ruv = _il.module_from_spec(spec)
    spec.loader.exec_module(ruv)
    topo_dir = os.path.join(ROOT, "topologies")
    files = [f for f in os.listdir(topo_dir) if f.endswith(".json")]
    assert files, "no checked-in topologies found"
    for fn in files:
        obj = json.load(open(os.path.join(topo_dir, fn)))
        # filename == topology_key == "<arch>-<8hex>", model-agnostic
        assert fn[:-5] == obj["topology_key"], "filename must equal topology_key"
        assert obj["nodes"], "%s has no nodes" % fn
        # expand across a few layers and check edges resolve to node indices, no inferred
        lg = ruv.expand_topology_to_layers(obj, 3)
        assert lg["edges_inferred"] is False
        assert lg["n_nodes"] == len(obj["nodes"]) * 3
        for nd in lg["nodes"]:
            for s in nd["src"]:
                assert isinstance(s, int) and 0 <= s < lg["n_nodes"], "bad edge index"


def test_model_name_field(tmp_path):
    # No --gguf and no --clean-tps-file: model_name is empty, but the field must exist
    # and the title JS must guard on it.
    html = _gen(tmp_path)
    raw = _raw(html)
    assert "model_name" in raw, "payload missing model_name field"
    assert raw["model_name"] == "", "model_name should be empty without --gguf/clean-tps"
    assert "D.model_name ?" in html, "frontend title must guard on D.model_name"
    assert "document.title" in html, "frontend must set the browser tab title"


def test_model_name_from_clean_tps(tmp_path):
    # Model name must appear in the title WITHOUT --gguf, recovered from clean_tps.txt's
    # model_filename (basename, minus .gguf). This is the "show the model in the title
    # regardless" behavior -- collect.sh always emits clean_tps.txt.
    html = _gen(tmp_path, "--clean-tps-file",
                os.path.join(FIX, "decode_clean_tps.json"))
    raw = _raw(html)
    assert raw["model_name"] == "Qwen3.6-35B-A3B-UD-Q4_K_XL", \
        "model_name should be recovered from clean_tps model_filename"


def test_unmapped_family_lists_dispatches(tmp_path):
    # A family that does not stream a GGUF weight (e.g. k_bin_bcast) has no order-map,
    # but the family panel must still list its dispatches instead of dead-ending. The
    # fixtures contain such families; assert the "unmapped" fallback branch is present
    # and a non-weight family (k_bin_bcast, now op-tagged e.g. k_bin_bcast[add]) appears.
    html = _gen(tmp_path)
    raw = _raw(html)
    fams = {s["fam"] for s in raw["gpu"]}
    assert any(f.startswith("k_bin_bcast") for f in fams), \
        "fixture should contain the unmapped k_bin_bcast family"
    assert ", unmapped)</" in html, "frontend must render an unmapped-dispatch table branch"

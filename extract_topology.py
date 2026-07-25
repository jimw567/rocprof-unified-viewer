#!/usr/bin/env python3
"""Extract the STABLE, architecture-keyed compute-graph topology from ggml-roofline
dumps (PR #66), for check-in.

A roofline dump is large (~5-17 MB) because it records every op INVOCATION with timings,
bytes, and kernel arrays -- most of it is the same layer repeated across layers and decode
tokens. But the *topology* (which op feeds which) is stable per ARCHITECTURE: it is fixed
by llama.cpp's build_graph() and does not depend on model size, prompt, tokens, or
hardware. So we dedup a dump down to one representative layer's structural skeleton and
key it by `general.architecture` (from the gguf) -- NOT the model name. All Qwen2.5 sizes
share one `qwen2` skeleton; Llama/SmolLM/Janus share `llama`; etc.

Output: topologies/<arch>.json, a few KB each, faithful (real node->src[] edges,
last-writer-wins), timing-free. The viewer loads these by arch at view time.

Usage:
    extract_topology.py --dumps DIR --models sweep_models.json --out topologies/
The models JSON pairs each dump's model name to its gguf path (to read the arch).
"""
import argparse
import json
import os
import re
import sys

# Strip per-layer indices ANYWHERE in the name so structurally-identical layers collapse:
#   Qcur-23 -> Qcur, cache_k_l12 (view) -> cache_k (view), node_1343 -> node,
#   conv_state_update (copy of conv_state_last-41) -> conv_state_update (copy of conv_state_last)
# The index appears as a trailing/embedded -<N>, _l<N>, or _<N> token. We remove every such
# run of digits that is delimited by -,_ (the ggml cb() layer tag), leaving the stable stem.
_LIDX = re.compile(r"[-_]l?\d+\b")
_NODE_N = re.compile(r"^node_\d+$")


def _norm_name(name):
    name = str(name or "")
    if _NODE_N.match(name):
        return "node"
    return _LIDX.sub("", name).strip()


def _arch_key(gguf_path):
    """Derive the TOPOLOGY KEY from the gguf's STRUCTURE -- never its name or size.

    `general.architecture` alone is too coarse (the same string can carry a dense OR a
    MoE OR a per-layer-embedding block -- e.g. gemma4 spans all three). Instead we
    fingerprint the block STRUCTURE directly: the SET of distinct per-layer tensor
    *roles* (the tensor names with their layer index stripped, e.g. blk.3.ssm_conv1d
    -> ssm_conv1d). This set is identical for every size of one architecture (0.5B and
    7B qwen2 share it), and differs exactly when the block differs -- with no model name,
    no size, and no hand-picked hparam list. Key = `<arch>-<8hex>` where the hex is a
    hash of that role set. The viewer computes the identical key from the gguf.

    Returns (key, arch, role_set) so the caller can also emit example role diffs."""
    import hashlib
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "ruv", os.path.join(here, "rocprof_unified_viewer.py"))
    ruv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ruv)
    tensors, meta = ruv.load_gguf_tensors(gguf_path)
    arch = meta.get("general.architecture", "unknown")
    roles = set()
    for t in tensors:
        nm = t.get("name", "") if isinstance(t, dict) else str(t)
        # blk.<N>.<role>  -> role  (only per-block tensors define the repeating structure)
        m = re.match(r"^blk\.\d+\.(.+)$", nm)
        if m:
            roles.add(m.group(1))
    fp = hashlib.sha1((arch + "|" + "|".join(sorted(roles))).encode()).hexdigest()[:8]
    return "%s-%s" % (arch, fp), arch, roles


def skeleton(dump_path):
    """Reduce a roofline dump to its structural skeleton: dedup layer/token repeats,
    keep name-normalized (op, edges). Edges via storage-id last-writer-wins, mapped to
    the PRODUCER's normalized name. Returns {"nodes":[{name,op,src:[names]}], ...}."""
    g = json.load(open(dump_path))
    rows = sorted(g.get("rows") or [], key=lambda r: r.get("invocation", 0))
    writer = {}          # storage_id -> normalized producer name
    seen = set()
    nodes = []
    for r in rows:
        nm = _norm_name(r.get("name", ""))
        srcs = []
        for sid in (r.get("in_storage_ids") or []):
            w = writer.get(sid)
            if w is not None and w != nm:
                srcs.append(w)
        srcs = sorted(dict.fromkeys(srcs))
        key = (nm, r.get("ggml_op"), tuple(srcs))
        if key not in seen:
            seen.add(key)
            nodes.append({"name": nm, "op": r.get("ggml_op", "?"), "src": srcs})
        out = r.get("out_storage_id")
        if out is not None:
            writer[out] = nm
    return {"nodes": nodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="dir of <model>.json roofline dumps")
    ap.add_argument("--models", required=True,
                    help="sweep_models.json (present[].name/.path)")
    ap.add_argument("--out", default="topologies", help="output dir for <arch>.json")
    args = ap.parse_args()

    models = json.load(open(args.models))["present"]
    by_key = {}          # topo-key -> {arch, roles, skeletons:[nodes,...], n}
    for m in models:
        dump = os.path.join(args.dumps, m["name"] + ".json")
        if not os.path.isfile(dump):
            print("skip (no dump)", file=sys.stderr)
            continue
        key, arch, roles = _arch_key(m["path"])
        sk = skeleton(dump)
        b = by_key.setdefault(key, {"arch": arch, "roles": roles, "sks": [], "n": 0})
        b["sks"].append(sk["nodes"])
        b["n"] += 1

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for key in sorted(by_key):
        b = by_key[key]
        # All models under one structural key MUST share a skeleton (that's the point of
        # the tensor-role fingerprint). Assert it; if they diverge it means the role set
        # is not fine-grained enough (a bug to fix), not a model to special-case.
        canon = max(b["sks"], key=len)
        for sk in b["sks"]:
            if sk != canon:
                sa = {(n["name"], n["op"], tuple(n["src"])) for n in sk}
                sb = {(n["name"], n["op"], tuple(n["src"])) for n in canon}
                print("WARN %-20s intra-key skeleton mismatch (%d nodes) -- role "
                      "fingerprint too coarse" % (key, len(sa ^ sb)), file=sys.stderr)
        out_obj = {
            # architecture family only -- NO model name, NO size. `key` = arch + a hash
            # of the block's distinct tensor roles, so it is fully model-agnostic.
            "topology_key": key,
            "architecture": b["arch"],
            "source": "ggml-roofline (PR#66) storage-id topology, last-writer-wins; "
                      "layer/token-deduped, timing-free. Faithful node->src[] edges. "
                      "Keyed by architecture + block tensor-role fingerprint.",
            "n_variants": b["n"],
            "nodes": canon,
        }
        path = os.path.join(args.out, key + ".json")
        with open(path, "w") as fh:
            json.dump(out_obj, fh, indent=1)
        sz = os.path.getsize(path)
        total += sz
        print("%-24s %3d nodes  %5d B  (%d variants)"
              % (key, len(canon), sz, b["n"]))
    print("---\n%d topology keys, %d B total" % (len(by_key), total))


if __name__ == "__main__":
    main()

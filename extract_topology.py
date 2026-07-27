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

# Reuse the viewer's OWN kernel-name -> family normalizer so the baked-in node families
# match exactly what the swim lane / order-map produce at view time.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rocprof_unified_viewer import family_of  # noqa: E402

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


_GGML_QUANT = {0: "f32", 1: "f16", 8: "q8_0", 10: "q2_K", 11: "q3_K", 12: "q4_K",
               13: "q5_K", 14: "q6_K", 15: "q8_K", 30: "bf16", 39: "mxfp4"}


def gguf_role_shape(gguf_path):
    """Read the gguf and return {(K, N, quant): [roles...]} plus a set of all roles.

    Each per-block weight blk.<L>.<role>.weight has ggml dims [K, N, ...] and a quant.
    We key by (K, N, quant) so a dump node's weight [K,N]+quant can be matched back to its
    ROLE with zero hand-authoring -- the shape+quant of a role is fixed by the architecture
    (a role's dims scale with model size, but within ONE model the mapping is exact). Ties
    (two roles sharing a shape, e.g. attn_k/attn_v, ffn_gate/ffn_up) are returned as a list
    for the caller to disambiguate by execution order."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "ruv", os.path.join(here, "rocprof_unified_viewer.py"))
    ruv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ruv)
    tensors, _ = ruv.load_gguf_tensors(gguf_path)
    by_shape = {}
    for t in tensors:
        nm = t.get("name", "")
        ne = t.get("ne") or []
        m = re.match(r"^blk\.\d+\.(.+)\.weight$", nm)
        if not m or len(ne) < 2:
            # the non-block output projection ("output.weight" / "token_embd.weight")
            m2 = re.match(r"^(output|token_embd)\.weight$", nm)
            if not (m2 and len(ne) >= 2):
                continue
            role = m2.group(1)
        else:
            role = m.group(1)
        q = _GGML_QUANT.get(t.get("gt"), str(t.get("gt")))
        key = (ne[0], ne[1], q)
        by_shape.setdefault(key, [])
        if role not in by_shape[key]:
            by_shape[key].append(role)
    return by_shape


def _row_fam(r):
    """The real kernel FAMILY a dump row's op ran as -- the matvec/gemm kernel, skipping
    the quantize_q8_1 prep subkernel. Returns "" if the row launched no informative kernel
    (a structural/elementwise op the viewer already labels from the trace). Stable per
    node across layers/tokens (a Q5_K qkv is always the same kernel), which is what lets us
    bake it into the checked-in skeleton and skip the dump at view time."""
    ks = r.get("kernels") or []
    mm = [k for k in ks if any(t in k.get("name", "")
                               for t in ("mul_mat", "wvsplitk", "gemm"))]
    pick = (mm[-1] if mm else (ks[-1] if ks else None))
    return family_of(pick["name"]) if pick and pick.get("name") else ""


def _row_kernels(r):
    """The GPU kernel families this op FOLDS, in launch order (e.g. a quantized matmul folds
    quantize_q8_1 then mul_mat_vec_q). Baked into the skeleton so the graph node shows a
    "+Nk" badge + hover breakdown for its prep kernels -- the quantize step is otherwise
    invisible (it is not a ggml op, just a sub-kernel). List of family strings; the kernel
    VARIANT is build-dependent but the fact that a matmul folds a quantize is arch-stable."""
    return [family_of(k.get("name", "")) for k in (r.get("kernels") or []) if k.get("name")]


# Node-name fragment -> role SUBSTRING. A shape-tie (roles sharing K,N,quant) is broken by
# the dump node's cb() name: Kcur -> a role containing "attn_k", ffn_moe_gate -> a role
# containing "ffn_gate" (matches both dense "ffn_gate" and MoE "ffn_gate_exps"). Ordered so
# the most specific fragment wins first (gate/up/down before the bare attn_q/k/v).
_ROLE_HINT = [
    ("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down"),
    ("alpha", "ssm_alpha"), ("beta", "ssm_beta"),
    ("kcur", "attn_k"), ("vcur", "attn_v"), ("qcur", "attn_q"),
]


def _match_role(rk, K, N, quant, tie_ctr, nm=""):
    """Match a dump matmul node to its GGUF weight ROLE by (K, N, quant). rk is the gguf's
    {(K,N,quant): [roles]}. On a shape TIE (>1 role shares the shape, e.g. attn_k/attn_v or
    the three MoE experts ffn_{gate,up,down}_exps), first try to break it with the node's own
    cb() NAME (Kcur -> attn_k, ffn_moe_gate -> ffn_gate_exps); if the name is uninformative,
    fall back to handing out the tied roles in execution order via tie_ctr. Returns "" if no
    gguf weight has this shape (a non-weight matmul, e.g. an attention score GEMM)."""
    roles = rk.get((K, N, quant)) or rk.get((N, K, quant))
    if not roles:
        return ""
    if len(roles) == 1:
        return roles[0]
    low = nm.lower()
    for frag, sub in _ROLE_HINT:
        if frag in low:
            hit = next((r for r in roles if sub in r), None)
            if hit:
                return hit
    key = (K, N, quant)
    i = tie_ctr.get(key, 0)
    tie_ctr[key] = i + 1
    return roles[i % len(roles)]


def _first_pass_rows(rows):
    """A decode dump replays the whole graph once per token; keep just the first pass so the
    skeleton is one forward pass (all layers once), cut at the terminal lm_head op."""
    terms = ("result_output", "result_embd", "result_norm")
    for i, r in enumerate(rows):
        if str(r.get("name", "")) in terms:
            return rows[:i + 1]
    return rows


def skeleton(dump_path, role_shape=None):
    """Reduce a roofline dump to a one-block structural skeleton with INDEX-based edges.

    Edges are the crux: the dump's storage-ids give an exact producer->consumer DAG. Rather
    than fold all layers together (a name-signature collapse creates cross-layer forward
    edges -> CYCLES, which break the layered layout), we extract ONE representative interior
    layer verbatim, in execution order: its edges only point backward within the block, so
    it is a clean DAG and connected. Intra-block edges become node indices; edges reaching
    OUTSIDE the block (the residual read from the previous layer) are dropped here and
    re-threaded per layer by the expander. An interior layer is used (not the first) so the
    block is fully formed -- the first layer reads the token embedding, not a residual.

    Each node also carries the kernel identity it ran as -- quant, family, and the WEIGHT
    ROLE (matched to the gguf by shape+quant) -- all stable per architecture, so the viewer
    names graph nodes and derives live per-size KxN (role -> loaded gguf weight) from a plain
    --gguf overlay, no roofline dump at view time. Returns
    {"nodes":[{name,op,src:[idx],fam,quant,role,roles}], ...}."""
    role_shape = role_shape or {}
    g = json.load(open(dump_path))
    rows = sorted(g.get("rows") or [], key=lambda r: r.get("invocation", 0))
    rows = _first_pass_rows(rows)

    # 1) full first-pass graph, edges = producing node index (last-writer-wins). This is a
    # strict DAG: a src is always an EARLIER row (its writer ran before this consumer).
    writer = {}          # storage_id -> node index that last wrote it
    full = []
    tie_ctr = {}
    for r in rows:
        nm = _norm_name(r.get("name", ""))
        srcs = []
        for sid in (r.get("in_storage_ids") or []):
            w = writer.get(sid)
            if w is not None:
                srcs.append(w)
        role = ""
        if r.get("ggml_op") in ("MUL_MAT", "MUL_MAT_ID"):
            sne = (r.get("src_ne") or [[None, None]])[0]
            if sne and sne[0] and sne[1]:
                role = _match_role(role_shape, sne[0], sne[1], (r.get("quant") or ""),
                                   tie_ctr, nm)
        idx = len(full)
        full.append({"name": nm, "op": r.get("ggml_op", "?"),
                     "src": [s for s in dict.fromkeys(srcs) if s < idx],
                     "fam": _row_fam(r), "quant": (r.get("quant") or ""), "role": role,
                     "kernels": _row_kernels(r)})
        out = r.get("out_storage_id")
        if out is not None:
            writer[out] = idx

    # 2) pick ONE real layer block by the raw name's layer suffix (blk index). The dump tags
    # per-layer ops as "<name>-<L>" (cb() layer tag), so a whole layer is the rows whose raw
    # name ends in "-<L>". We use the layer that has the MOST tagged ops (a fully-formed
    # interior layer -- some archs' layer 0 is special), avoiding any period guessing.
    span = _layer_span(rows)
    if span is None:
        # no per-layer tags: fall back to the whole first pass (still a valid DAG, just larger)
        lo, hi = 0, len(full)
    else:
        lo, hi = span

    # 3) re-index the block to 0-based; keep only intra-block edges (outside = the residual
    # read from the previous layer, re-threaded per layer by the expander).
    nodes = []
    for j in range(lo, hi):
        n = full[j]
        srcs = [s - lo for s in n["src"] if lo <= s < hi and (s - lo) != j]
        nd = {"name": n["name"], "op": n["op"], "src": list(dict.fromkeys(srcs)),
              "fam": n["fam"], "quant": n["quant"], "role": n["role"]}
        if n["role"]:
            nd["roles"] = [n["role"]]
        # folded sub-kernels (e.g. quantize_q8_1 prep) so the node shows a +Nk badge; drop
        # empties and keep only when there is a prep kernel beyond the main matvec.
        ker = [k for k in (n.get("kernels") or []) if k]
        if len(ker) > 1:
            nd["kernels"] = [{"fam": k} for k in ker]
        nodes.append(nd)
    return {"nodes": nodes}


def _layer_span(rows):
    """Return (lo, hi) row indices delimiting the single most-complete layer block, using the
    dump's raw "<name>-<L>" cb() layer tags. Returns None if the dump carries no layer tags.
    The chosen block is a contiguous run of rows sharing one layer index L (the L with the
    most tagged ops), extended to include the untagged ops interleaved within its span."""
    tag = re.compile(r"-(\d+)(?:\s|$|\))")
    layer_rows = {}      # L -> [row idx, ...]
    for i, r in enumerate(rows):
        m = tag.search(str(r.get("name", "")))
        if m:
            layer_rows.setdefault(int(m.group(1)), []).append(i)
    if not layer_rows:
        return None
    # the busiest layer = most fully captured; take its contiguous [min, max] row span so the
    # untagged ops (views, cache ops) that sit between its tagged ops are included in order.
    best_L = max(layer_rows, key=lambda L: len(layer_rows[L]))
    idxs = layer_rows[best_L]
    return (idxs[0], idxs[-1] + 1)


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
        role_shape = gguf_role_shape(m["path"])
        sk = skeleton(dump, role_shape)
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

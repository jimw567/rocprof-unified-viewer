#!/usr/bin/env python3
"""rocprof_unified_viewer.py -- fuse CPU overhead + GPU overhead + kernel stall +
achieved DRAM bandwidth into ONE self-contained HTML timeline from rocprofv3 CSVs.

No single existing tool overlays all these profiling layers. Perfetto can't tie a
PMC counter to the slice that produced it, chokes on large traces, and has no
aggregate summary beside the timeline. This does: a Canvas-rendered timeline with a
CPU (HIP-API) lane above and a GPU (kernel) lane below on a shared time axis, GPU
slices color-coded by dominant stall reason, a per-kernel-family summary panel, hover
detail, and a token stepper.

v1 is specialized for llama.cpp/ggml decode on gfx1151: decode is PERIODIC -- every
token replays the same kernel sequence -- so the default window is a tiny 2-token
slice (128 tokens is ~99% redundant and is exactly what chokes Perfetto). The tool
consumes generic rocprofv3 CSVs, so it has room to grow beyond this case.

INPUTS (rocprofv3 CSVs + one JSON; only --kernel-csv is required):
  --kernel-csv     *_kernel_trace.csv       GPU slices + timing     (from --sys-trace)
  --hip-csv        *_hip_api_trace.csv       CPU/host HIP-API lane   (from --sys-trace)
  --pmc-csv        *_counter_collection.csv  stall counters for coloring (from --pmc)
  --fetch-csv      *_counter_collection.csv  FETCH_SIZE bytes -> achieved BW (from --pmc)
  --loadwidth-json loadwidth.json            per-family load-width (from disasm_loadwidth.py)
  --gguf           model.gguf                order-map matvec dispatch -> weight tensor

The kernel + hip CSVs come from the SAME clean sys-trace run (shared clock, so they
overlay). The PMC/FETCH CSVs come from SEPARATE runs (PMC serializes/distorts timing),
so they are joined by kernel-name FAMILY -- per-family aggregate, never per-dispatch.

With --gguf, each mul_mat_vec decode dispatch is order-mapped to its GGUF weight
tensor: decode is strictly periodic, so the dispatch stream within a token matches the
weights' canonical execution order exactly. The join key is the launched output-row
count N (Grid_Size_X / Workgroup_Size_X) == the weight's true ne[1]; the kernel-name
(ggml_type) template arg is NOT a reliable weight-quant proxy (Q5_K weights dispatch
under Q4_K/Q6_K kernels), so shape (N), not type, is the join key. Each matvec slice
then carries its true [K x N] shape, quant, packed footprint, launch-vs-true padding,
and a measured (per-family+N FETCH_SIZE) over-fetch ratio in the detail panel.

Produce all inputs with the bundled collect.sh (see README), or run rocprofv3 by hand.

Example:
  rocprof-unified-viewer \\
      --kernel-csv run/xxx_kernel_trace.csv \\
      --hip-csv    run/xxx_hip_api_trace.csv \\
      --pmc-csv    run/yyy_counter_collection.csv \\
      --fetch-csv  run/zzz_counter_collection.csv \\
      --loadwidth-json run/loadwidth.json \\
      --gguf       model.gguf \\
      --out overlay.html --tokens 2
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict


# --- stall classification thresholds (tunable) --------------------------------
# Derived from gfx1151 4B decode PMC: mul_mat_vec_q = MemBusy 77 / L2 8 (memory);
# elementwise kernels sit low on everything (latency/occupancy bound); LDS bank
# conflicts are ~0 on this arch. See reference_gfx1151_intrakernel_profiling.
MEM_BUSY_HI = 25.0     # MemUnitBusy% at/above this + low L2 hit => memory-bound
L2_HIT_LO = 30.0       # L2CacheHit% at/below this => traffic misses to VRAM
LDS_CONFLICT_HI = 5.0  # LDSBankConflict above this => LDS-bound
OCC_LO = 20.0          # OccupancyPercent below this (and not busy) => under-occupied

STALL_COLORS = {
    "memory":    "#e6194b",  # red
    "compute":   "#4363d8",  # blue
    "occupancy": "#f58231",  # amber
    "lds":       "#911eb4",  # purple
    "copy":      "#9a9a9a",  # grey
    "unknown":   "#3cb44b",  # green (no PMC data)
}

PMC_COUNTERS = ["MemUnitBusy", "L2CacheHit", "OccupancyPercent",
                "Wavefronts", "LDSBankConflict", "WriteUnitStalled"]


# ggml_type enum -> quant name (ggml.h). Used to keep quant kernels distinct.
_GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    29: "IQ1_M", 30: "BF16",
}

# Per-arch peak DRAM bandwidth (GB/s) used as the roofline denominator. Built up
# gradually as boards are characterized. gfx1151 (Strix Halo) is 256-bit
# LPDDR5X-8000 = 256 GB/s theoretical, but ~230 GB/s is the realistic achievable
# ceiling, so we roofline against 230. achieved GB/s == bytes_per_token /
# kernel_time_ns_per_token exactly (1 B/ns == 1 GB/s).
PEAK_BW_GBS_BY_ARCH = {
    "gfx1151": 230.0,   # Strix Halo, LPDDR5X-8000 256-bit (~230 achievable of 256 theo)
}
DEFAULT_ARCH = "gfx1151"


def peak_bw_for(arch, override=None):
    if override:
        return float(override)
    return PEAK_BW_GBS_BY_ARCH.get(arch, PEAK_BW_GBS_BY_ARCH[DEFAULT_ARCH])


def family_of(kernel_name):
    """Normalize a mangled/templated kernel name to a family (the same
    aggregation used when collecting PMC, so PMC families join onto trace slices).
    For quantized kernels whose first template arg is a (ggml_type)N, keep the
    quant type so e.g. mul_mat_vec_q<(ggml_type)12,...> vs <(ggml_type)14,...>
    (Q4_K vs Q6_K) are distinct families instead of one blend."""
    short = re.sub(r"<.*", "", kernel_name).split("(")[0]
    short = short.split("void ")[-1].strip()
    m = re.search(r"<\s*\(ggml_type\)(\d+)", kernel_name)
    if m:
        n = int(m.group(1))
        short += "[" + _GGML_TYPES.get(n, "type%d" % n) + "]"
    return short


def dominant_stall(counters):
    """Classify a family's dominant stall from its mean counters."""
    mem = counters.get("MemUnitBusy", 0.0)
    l2 = counters.get("L2CacheHit", 0.0)
    lds = counters.get("LDSBankConflict", 0.0)
    occ = counters.get("OccupancyPercent", 0.0)
    if lds > LDS_CONFLICT_HI:
        return "lds"
    if mem >= MEM_BUSY_HI and l2 <= L2_HIT_LO:
        return "memory"
    if mem >= 40.0:
        return "compute"
    if occ < OCC_LO:
        return "occupancy"
    return "compute"


# --- CSV loaders (stdlib only; duplicated on purpose so this file is standalone) --

def load_kernel_slices(path):
    """Return {stream_id: [(start_ns, end_ns, kernel_name, N), ...] sorted by
    start}. N = Grid_Size_X / Workgroup_Size_X is the launched output-row count
    (one warp/workgroup-row per output row for mul_mat_vec), the join key onto the
    GGUF weight's true N (ne[1]); 0 when the grid dims are absent/degenerate."""
    by_stream = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                gx = int(r["Grid_Size_X"])
                wg = int(r["Workgroup_Size_X"])
                n = gx // wg if wg else 0
            except (KeyError, ValueError, TypeError):
                n = 0
            by_stream[r["Stream_Id"]].append(
                (int(r["Start_Timestamp"]), int(r["End_Timestamp"]),
                 r["Kernel_Name"], n))
    for evs in by_stream.values():
        evs.sort()
    return by_stream


_HIP_NAME_COLS = ("Function", "Api_Name", "Name", "Operation")


def load_hip_calls(path, t0, t1):
    """Return HIP-API calls overlapping [t0, t1] as (start, end, name), sorted."""
    out = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        name_col = next((c for c in _HIP_NAME_COLS if c in fields), None)
        if name_col is None:
            return out
        for r in reader:
            try:
                s = int(r["Start_Timestamp"])
                e = int(r["End_Timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            if e >= t0 and s <= t1:
                out.append((s, e, r[name_col]))
    out.sort()
    return out


def load_pmc_families(path):
    """Aggregate a PMC counter CSV into {family: {counter: mean, ...}} plus
    dispatch count and dominant stall."""
    agg = defaultdict(lambda: defaultdict(list))
    # Register counts are per-dispatch metadata columns (constant per kernel),
    # not PMC counters, so track them separately as a per-family max.
    regs = defaultdict(lambda: {"vgpr": 0, "accum_vgpr": 0, "sgpr": 0,
                                "scratch": 0})
    # Wavefront size = work-items / wavefronts dispatched, derived from
    # Grid_Size / Wavefronts. Both are summed over the SAME dispatches (gated on
    # the Wavefronts counter row, which occurs once per dispatch) so the ratio is
    # exact even when a family mixes dispatch sizes.
    wsz = defaultdict(lambda: {"grid": 0.0, "waves": 0.0})
    with open(path) as fh:
        for r in csv.DictReader(fh):
            fam = family_of(r["Kernel_Name"])
            try:
                agg[fam][r["Counter_Name"]].append(float(r["Counter_Value"]))
            except (KeyError, ValueError, TypeError):
                pass
            g = regs[fam]
            for key, col in (("vgpr", "VGPR_Count"),
                             ("accum_vgpr", "Accum_VGPR_Count"),
                             ("sgpr", "SGPR_Count"),
                             ("scratch", "Scratch_Size")):
                try:
                    g[key] = max(g[key], int(r[col]))
                except (KeyError, ValueError, TypeError):
                    pass
            if r.get("Counter_Name") == "Wavefronts":
                try:
                    wsz[fam]["grid"] += int(r["Grid_Size"])
                    wsz[fam]["waves"] += float(r["Counter_Value"])
                except (KeyError, ValueError, TypeError):
                    pass
    fams = {}
    for fam, cc in agg.items():
        means = {k: statistics.mean(v) for k, v in cc.items() if v}
        ndisp = max((len(v) for v in cc.values()), default=0)
        w = wsz[fam]
        fams[fam] = {
            "counters": means,
            "pmc_dispatches": ndisp,
            "stall": dominant_stall(means),
            "regs": regs[fam],
            "wave": int(round(w["grid"] / w["waves"])) if w["waves"] else 0,
        }
    return fams


def load_fetch_bytes(path):
    """Aggregate a rocprofv3 --pmc FETCH_SIZE CSV into {family: mean DRAM read
    bytes per dispatch}. FETCH_SIZE is post-L2 actual VRAM read traffic in KiB
    per dispatch ("all cache/memory effects taken into account"), so this is the
    MEASURED bytes each family streams from DRAM per dispatch -- the numerator of
    achieved bandwidth. Bytes/dispatch is token-count-independent (the same
    kernel does the same work each decode token), so a short -n 2 PMC run joins
    cleanly onto a longer clean timeline by family.

    Returns (by_fam, by_fam_n): by_fam is {family: mean bytes/dispatch}; by_fam_n
    is {(family, N): mean bytes/dispatch} where N = Grid_Size / Workgroup_Size, so
    a family that mixes output shapes (e.g. mul_mat_vec_q spanning N=9216/4096/...)
    can be compared per-shape against each dispatch's true weight footprint rather
    than a shape-blended family mean."""
    agg = defaultdict(list)
    agg_n = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("Counter_Name") != "FETCH_SIZE":
                continue
            try:
                v = float(r["Counter_Value"])
            except (KeyError, ValueError, TypeError):
                continue
            fam = family_of(r["Kernel_Name"])
            agg[fam].append(v)
            try:
                gs = int(r["Grid_Size"])
                ws = int(r["Workgroup_Size"])
                n = gs // ws if ws else 0
            except (KeyError, ValueError, TypeError):
                n = 0
            if n:
                agg_n[(fam, n)].append(v)
    by_fam = {fam: statistics.mean(v) * 1024.0 for fam, v in agg.items() if v}
    by_fam_n = {k: statistics.mean(v) * 1024.0 for k, v in agg_n.items() if v}
    return by_fam, by_fam_n


def load_loadwidth(path):
    """Load the disassembly load-width JSON ({family: {vector_loads, scalar_loads,
    lds_loads, dominant_lane_bytes, ...}}) produced from the gfx1151 device code
    objects. Keyed by the same family_of() names, so it joins onto slices."""
    with open(path) as fh:
        return json.load(fh)


# --- GGUF weight-tensor table (stdlib; per-dispatch true-shape mapping) --------
# GGUF value-type enum (gguf spec) used to walk metadata KV pairs.
_GGUF_SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9

# ggml_type -> (block_elems, block_bytes): on-disk packed size of one block.
# K-quants pack 256 elems/block; legacy quants 32; F32/F16/BF16 are dense.
_GGML_BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2),
}


def _gguf_packed_bytes(ne, gt):
    be, bb = _GGML_BLOCK.get(gt, (1, 4))
    n = 1
    for d in ne:
        n *= d
    return (n // be) * bb if be > 1 else n * bb


def load_gguf_tensors(path):
    """Parse a GGUF file's tensor-info table (stdlib, via mmap so the multi-MB
    tokenizer metadata is walked without reading the 2+GB of weight data). Returns
    (tensors, meta) where each tensor is {name, ne, gt, bytes}: ne is ggml dim
    order ([inner/K, rows/N, ...]) and bytes is the packed on-disk footprint."""
    import mmap
    import struct

    def rd(mm, o, fmt):
        v = struct.unpack_from(fmt, mm, o[0])
        o[0] += struct.calcsize(fmt)
        return v[0]

    def rstr(mm, o):
        n = rd(mm, o, "<Q")
        s = mm[o[0]:o[0] + n]
        o[0] += n
        return s.decode("utf-8", "replace")

    def rval(mm, o, t):
        if t == _GGUF_STRING:
            return rstr(mm, o)
        if t == _GGUF_ARRAY:
            at = rd(mm, o, "<I")
            n = rd(mm, o, "<Q")
            return [rval(mm, o, at) for _ in range(n)]
        return rd(mm, o, _GGUF_SIMPLE[t])

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    try:
        if mm[0:4] != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        o = [4]
        rd(mm, o, "<I")                 # version
        n_tensors = rd(mm, o, "<Q")
        n_kv = rd(mm, o, "<Q")
        meta = {}
        for _ in range(n_kv):
            k = rstr(mm, o)
            t = rd(mm, o, "<I")
            meta[k] = rval(mm, o, t)
        tensors = []
        for _ in range(n_tensors):
            nm = rstr(mm, o)
            nd = rd(mm, o, "<I")
            ne = [rd(mm, o, "<Q") for _ in range(nd)]
            gt = rd(mm, o, "<I")
            rd(mm, o, "<Q")             # data offset (unused)
            tensors.append({"name": nm, "ne": ne, "gt": gt,
                            "bytes": _gguf_packed_bytes(ne, gt)})
        return tensors, meta
    finally:
        mm.close()
        f.close()


# Per-layer matvec role order for llama.cpp decode, validated on the qwen35 hybrid
# (GDN/linear-attn layers interleaved with periodic full-attention layers). Only
# 2D projection weights become mul_mat_vec_{q,f} dispatches; norms/biases/1D
# tensors do not. One priority list covers both layer kinds because each layer
# owns only a subset of these roles.
_MATVEC_ROLE_ORDER = [
    "attn_qkv", "attn_q", "attn_v", "attn_k",
    "ssm_in", "ssm_alpha", "ssm_beta",
    "attn_gate", "ssm_out", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
]
# ne-dim quant/dense types dispatched as a matvec at decode (K-quants, legacy
# quants, and Q8_0 which carries the ssm alpha/beta scale projections).
_MATVEC_TYPES = {2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


def _seq_entry(t, layer, role):
    ne = t["ne"]
    return {"nm": t["name"], "L": layer, "role": role,
            "N": ne[1], "K": ne[0], "gt": t["gt"], "bytes": t["bytes"],
            "q": _GGML_TYPES.get(t["gt"], "type%d" % t["gt"])}


def build_expected_sequence(tensors, drop_ffn_up):
    """Ordered per-token matvec tensor sequence in decode execution order:
    layer-major (blk.0, blk.1, ...), roles within a layer by _MATVEC_ROLE_ORDER,
    then the output head. drop_ffn_up collapses the fused SwiGLU gate+up into the
    single gate dispatch llama.cpp emits at decode (the common case); the caller
    picks whichever of drop/keep best matches the trace. Each entry carries the
    weight's true (unpadded) N (rows/output), K (inner/input), quant + packed
    bytes."""
    bylayer = defaultdict(dict)
    nonblk = []
    for t in tensors:
        m = re.match(r"blk\.(\d+)\.(.*)\.weight$", t["name"])
        if m:
            bylayer[int(m.group(1))][m.group(2)] = t
        else:
            nonblk.append(t)
    seq = []
    for layer in sorted(bylayer):
        roles = bylayer[layer]
        for role in _MATVEC_ROLE_ORDER:
            if role == "ffn_up" and drop_ffn_up:
                continue
            t = roles.get(role)
            if t is None or len(t["ne"]) < 2 or t["gt"] not in _MATVEC_TYPES:
                continue
            seq.append(_seq_entry(t, layer, role))
    # Output head: a dedicated output.weight, else the tied token_embd.weight.
    head = (next((t for t in nonblk if t["name"] == "output.weight"), None)
            or next((t for t in nonblk if t["name"] == "token_embd.weight"), None))
    if head and len(head["ne"]) >= 2:
        seq.append(_seq_entry(head, -1, "output"))
    return seq


# --- token segmentation -------------------------------------------------------

def detect_boundaries(evs, gap_thr_ns):
    """Indices i where a gap > gap_thr_ns precedes evs[i] (candidate token
    boundaries), de-noised: drop boundaries closer than half the dominant period
    (spurious mid-token gaps), keeping the clean per-token cadence."""
    raw = [i for i in range(1, len(evs)) if evs[i][0] - evs[i - 1][1] > gap_thr_ns]
    if len(raw) < 3:
        return raw
    deltas = [raw[i] - raw[i - 1] for i in range(1, len(raw))]
    period = statistics.median([d for d in deltas if d > 10]) or 1
    min_sep = period * 0.5
    kept = [raw[0]]
    for i in raw[1:]:
        if i - kept[-1] >= min_sep:
            kept.append(i)
    return kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel-csv", required=True,
                    help="rocprofv3 *_kernel_trace.csv (GPU slices)")
    ap.add_argument("--hip-csv",
                    help="rocprofv3 *_hip_api_trace.csv (CPU lane; optional)")
    ap.add_argument("--pmc-csv",
                    help="rocprofv3 *_counter_collection.csv (stall coloring; "
                         "optional -- without it slices render uncolored)")
    ap.add_argument("--fetch-csv",
                    help="rocprofv3 --pmc FETCH_SIZE *_counter_collection.csv "
                         "(optional): MEASURED DRAM read bytes/dispatch per family "
                         "-> achieved DRAM bandwidth per family (bytes / kernel "
                         "time, vs the arch peak below). Measured attributes bytes "
                         "to the exact kernel that moved them.")
    ap.add_argument("--loadwidth-json",
                    help="JSON of per-family memory-load instruction widths from "
                         "device disassembly (optional): shows per-lane load width "
                         "(b32=4B, d16=2B, ...) in the selected-kernel detail panel")
    ap.add_argument("--gguf",
                    help="GGUF model file (optional): order-maps each mul_mat_vec "
                         "decode dispatch to its GGUF weight tensor by execution "
                         "order (join on launched N == weight ne[1]), attaching the "
                         "weight name, true [K x N] shape, and packed footprint to "
                         "the detail panel -- so launch-grid vs true shape reveals "
                         "any output-row/reduction padding waste and the packed "
                         "weight bytes give a theoretical-vs-measured over-fetch ratio")
    ap.add_argument("--arch", default=DEFAULT_ARCH,
                    help="GPU arch, selects peak DRAM BW for the roofline "
                         "(default %s = %g GB/s)"
                         % (DEFAULT_ARCH, PEAK_BW_GBS_BY_ARCH[DEFAULT_ARCH]))
    ap.add_argument("--peak-bw", type=float,
                    help="override peak DRAM bandwidth in GB/s (else from --arch)")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--tokens", type=int, default=2,
                    help="decode tokens to show in the viewport (default 2)")
    ap.add_argument("--skip-tokens", type=int, default=30,
                    help="tokens to skip before the window, to land in steady "
                         "state past warmup/prefill (default 30)")
    ap.add_argument("--context-tokens", type=int, default=0,
                    help="extra tokens baked on each side for the stepper (default 0)")
    ap.add_argument("--gap-threshold-us", type=float, default=150.0,
                    help="inter-dispatch gap (us) that marks a token boundary "
                         "(default 150)")
    ap.add_argument("--title", default="rocprof unified viewer (gfx1151)")
    args = ap.parse_args()

    peak_bw = peak_bw_for(args.arch, args.peak_bw)
    # Surface the peak next to the arch string in the title (e.g.
    # "... (gfx1151)" -> "... (gfx1151, 230 GB/s peak)").
    title = args.title
    tag = "%s, %g GB/s peak" % (args.arch, peak_bw)
    if args.arch in title:
        title = title.replace(args.arch, tag, 1)
    else:
        title = "%s (%s)" % (title, tag)

    by_stream = load_kernel_slices(args.kernel_csv)
    if not by_stream:
        sys.exit(f"error: no kernel rows in {args.kernel_csv}")
    # Compute stream = the one with the most dispatches (stream 1 is model load).
    sid = max(by_stream, key=lambda s: len(by_stream[s]))
    evs = by_stream[sid]

    bounds = detect_boundaries(evs, args.gap_threshold_us * 1000)
    if len(bounds) < args.skip_tokens + args.tokens + 2:
        sys.exit(f"error: only {len(bounds)} token boundaries detected; "
                 f"need > {args.skip_tokens + args.tokens}. Lower --skip-tokens "
                 f"or --gap-threshold-us.")

    # Bake a wider span (context on each side) so the stepper can pan.
    lo_tok = max(0, args.skip_tokens - args.context_tokens)
    hi_tok = min(len(bounds) - 1, args.skip_tokens + args.tokens + args.context_tokens)
    a = bounds[lo_tok]
    b = bounds[hi_tok]
    baked = evs[a:b]
    t0, t1 = baked[0][0], baked[-1][1]

    # Token boundary timestamps within the baked span (for stepper snapping).
    tok_starts = [evs[bounds[k]][0] for k in range(lo_tok, hi_tok + 1)]
    # Initial viewport = the first `tokens` tokens after the leading context.
    view_i0 = args.context_tokens if lo_tok > 0 else 0
    view_i1 = min(len(tok_starts) - 1, view_i0 + args.tokens)

    # PMC families -> color/stall lookup.
    fams = load_pmc_families(args.pmc_csv) if args.pmc_csv else {}

    # Measured roofline: DRAM read bytes/dispatch per family from the PMC
    # FETCH_SIZE run (post-L2 actual VRAM traffic). Replaces the old GGUF analytic
    # estimate -- measured attributes bytes to the exact kernel that moved them
    # (so it also fixes the Q5_K shared-quant-type over-attribution the analytic
    # method had), and covers every family, not just mul_mat_vec.
    fetch_bytes, fetch_bytes_n = (load_fetch_bytes(args.fetch_csv)
                                  if args.fetch_csv else ({}, {}))
    loadwidth = load_loadwidth(args.loadwidth_json) if args.loadwidth_json else {}

    # Optional GGUF order-map: build the expected per-token matvec tensor sequence
    # in decode execution order. llama.cpp fuses the SwiGLU gate+up into one
    # dispatch at decode, so try both dropping and keeping ffn_up and pick whichever
    # candidate's N-sequence best matches the actual matvec dispatches in one token.
    expected_seq = []
    if args.gguf:
        gguf_tensors, _ = load_gguf_tensors(args.gguf)
        # Reference token = the matvec N-stream between the first two boundaries in
        # the baked span (a full steady-state token).
        ref = [n for (s, e, nm, n) in evs[bounds[args.skip_tokens]:
                                          bounds[args.skip_tokens + 1]]
               if "mul_mat_vec" in nm and n]
        best = None
        for drop in (True, False):
            cand = build_expected_sequence(gguf_tensors, drop)
            m = min(len(cand), len(ref))
            hits = sum(1 for i in range(m) if cand[i]["N"] == ref[i])
            score = hits - abs(len(cand) - len(ref))
            if best is None or score > best[0]:
                best = (score, cand, drop, hits, m)
        if best:
            expected_seq = best[1]

    # Baked-relative indices where a new decode token starts (reset the order-map
    # pointer here so each token re-aligns to the expected sequence head).
    tok_boundary_idx = {bounds[k] - a for k in range(lo_tok, hi_tok + 1)}

    # GPU slices in the baked span (relative ns from t0). If a GGUF sequence was
    # built, order-map each mul_mat_vec dispatch to its expected weight tensor
    # (lockstep by execution order within a token), guarded on launched N == the
    # weight's true ne[1] so a shape mismatch is reported, not silently attached.
    gpu_slices = []
    busy_ns = 0
    fam_busy = defaultdict(float)
    fam_count = defaultdict(int)
    ei = 0
    mv_total = mv_mapped = 0
    for idx, (s, e, name, ncol) in enumerate(baked):
        if idx in tok_boundary_idx:
            ei = 0
        fam = family_of(name)
        finfo = fams.get(fam)
        stall = finfo["stall"] if finfo else "unknown"
        if "copy" in fam.lower() or "cpy" in fam.lower():
            stall = "copy"
        dur = e - s
        busy_ns += dur
        fam_busy[fam] += dur
        fam_count[fam] += 1
        sl = {"s": s - t0, "e": e - t0, "fam": fam, "stall": stall}
        if expected_seq and "mul_mat_vec" in name and ncol:
            mv_total += 1
            ent = expected_seq[ei] if ei < len(expected_seq) else None
            ei += 1
            if ent is not None:
                true_n = ent["N"]
                k = ent["K"]
                packed = ent["bytes"]
                measured = fetch_bytes_n.get((fam, ncol)) or fetch_bytes.get(fam, 0)
                sl["map"] = {
                    "nm": ent["nm"], "role": ent["role"], "L": ent["L"],
                    "q": ent["q"], "K": k, "trueN": true_n, "launchN": ncol,
                    # Output-row padding: launched rows beyond the true weight rows.
                    "padN": max(0, ncol - true_n),
                    # Reduction (K) padding to the quant block (256 for K-quants).
                    "padK": (((k + 255) // 256) * 256 - k) if k else 0,
                    "packed": packed,
                    "measured": round(measured) if measured else 0,
                    # Over-fetch: measured DRAM bytes / theoretical packed footprint.
                    "overfetch": (round(measured / packed, 2)
                                  if (measured and packed) else 0),
                    "nmatch": (true_n == ncol),
                }
                if true_n == ncol:
                    mv_mapped += 1
        gpu_slices.append(sl)

    map_stats = ({"total": mv_total, "mapped": mv_mapped,
                  "pct": round(100.0 * mv_mapped / mv_total, 1) if mv_total else 0,
                  "seq_len": len(expected_seq)}
                 if expected_seq else None)

    # CPU lane (HIP-API) in the baked span.
    cpu_slices = []
    cpu_busy_ns = 0
    if args.hip_csv:
        for s, e, name in load_hip_calls(args.hip_csv, t0, t1):
            cs, ce = max(s, t0), min(e, t1)
            cpu_busy_ns += (ce - cs)
            cpu_slices.append({"s": cs - t0, "e": ce - t0, "name": name})

    span_ns = t1 - t0
    ntok_baked = hi_tok - lo_tok

    # Per-family summary (over the baked span), enriched with PMC counters and,
    # if a FETCH_SIZE run was given, MEASURED achieved DRAM bandwidth. bytes/disp
    # is measured (FETCH_SIZE, token-independent); achieved GB/s = mean bytes/disp
    # divided by that family's mean kernel time/disp (1 byte/ns == 1 GB/s exactly).
    # bytes/token = bytes/disp * dispatches/token.
    summary = []
    for fam in sorted(fam_busy, key=lambda f: fam_busy[f], reverse=True):
        finfo = fams.get(fam, {})
        c = finfo.get("counters", {})
        b_disp = fetch_bytes.get(fam, 0.0)          # measured bytes/dispatch
        per_tok = fam_count[fam] / max(ntok_baked, 1)
        time_ns_disp = fam_busy[fam] / max(fam_count[fam], 1)
        bw_gbs = (b_disp / time_ns_disp) if (b_disp and time_ns_disp) else 0.0
        bytes_tok = b_disp * per_tok
        summary.append({
            "fam": fam,
            "count": fam_count[fam],
            "per_tok": round(per_tok, 1),
            "busy_pct": round(fam_busy[fam] / busy_ns * 100, 1) if busy_ns else 0,
            "stall": ("copy" if ("copy" in fam.lower() or "cpy" in fam.lower())
                      else finfo.get("stall", "unknown")),
            "mem": round(c.get("MemUnitBusy", 0), 1),
            "l2": round(c.get("L2CacheHit", 0), 1),
            "occ": round(c.get("OccupancyPercent", 0), 1),
            "lds": round(c.get("LDSBankConflict", 0), 2),
            "wr": round(c.get("WriteUnitStalled", 0), 2),
            "wav": round(c.get("Wavefronts", 0), 0),
            # Derived ratios (raw cycle counters): EA = DRAM-interface busy
            # fraction (the true BW bottleneck meter); ALU = VALU-active fraction
            # (can exceed 100% -- VALU cycles are counted across 4 SIMDs/CU).
            "ea": (round(c["GRBM_EA_BUSY"] / c["GRBM_GUI_ACTIVE"] * 100, 1)
                   if c.get("GRBM_GUI_ACTIVE") else 0),
            "alu": (round(c["SQ_INST_CYCLES_VALU"] / c["SQ_BUSY_CYCLES"] * 100, 1)
                    if c.get("SQ_BUSY_CYCLES") else 0),
            # Per-kernel register allocation (occupancy limiter): max over dispatches.
            "vgpr": finfo.get("regs", {}).get("vgpr", 0),
            "accum_vgpr": finfo.get("regs", {}).get("accum_vgpr", 0),
            "sgpr": finfo.get("regs", {}).get("sgpr", 0),
            "scratch": finfo.get("regs", {}).get("scratch", 0),
            # Wavefront size (Grid_Size / Wavefronts); computed in load_pmc_families.
            "wave": finfo.get("wave", 0),
            "kb_disp": round(b_disp / 1024.0, 1) if b_disp else 0,
            "mb_tok": round(bytes_tok / 1e6, 1) if bytes_tok else 0,
            "bw_gbs": round(bw_gbs, 1),
            "bw_pct": round(bw_gbs / peak_bw * 100, 1) if bw_gbs else 0,
            "loadw": loadwidth.get(fam),
        })

    # Per-family raw counters (+ measured BW + load width) for the hover/detail box.
    fam_counters = {fam: {"stall": summary_i["stall"],
                          "mem": summary_i["mem"], "l2": summary_i["l2"],
                          "occ": summary_i["occ"], "lds": summary_i["lds"],
                          "wr": summary_i["wr"], "wav": summary_i["wav"],
                          "ea": summary_i["ea"], "alu": summary_i["alu"],
                          "vgpr": summary_i["vgpr"],
                          "accum_vgpr": summary_i["accum_vgpr"],
                          "sgpr": summary_i["sgpr"],
                          "scratch": summary_i["scratch"],
                          "wave": summary_i["wave"],
                          "kb_disp": summary_i["kb_disp"],
                          "mb_tok": summary_i["mb_tok"],
                          "bw_gbs": summary_i["bw_gbs"], "bw_pct": summary_i["bw_pct"],
                          "loadw": summary_i["loadw"]}
                    for summary_i, fam in ((s, s["fam"]) for s in summary)}

    payload = {
        "title": title,
        "kernel_csv": args.kernel_csv,
        "pmc_csv": args.pmc_csv or "",
        "span_ns": span_ns,
        "busy_ns": busy_ns,
        "cpu_busy_ns": cpu_busy_ns,
        "n_tokens_baked": ntok_baked,
        "tokens_view": args.tokens,
        "tok_starts": [t - t0 for t in tok_starts],
        "view_i0": view_i0, "view_i1": view_i1,
        "gpu": gpu_slices,
        "cpu": cpu_slices,
        "summary": summary,
        "fam_counters": fam_counters,
        "colors": STALL_COLORS,
        "has_pmc": bool(fams),
        "has_cpu": bool(cpu_slices),
        "has_bw": bool(fetch_bytes),
        "has_loadw": bool(loadwidth),
        "has_map": bool(expected_seq),
        "map_stats": map_stats,
        "peak_bw_gbs": peak_bw,
    }

    html = render_html(payload)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  baked {ntok_baked} tokens ({len(gpu_slices)} GPU slices, "
          f"{len(cpu_slices)} CPU calls); viewport shows {args.tokens} tokens.",
          file=sys.stderr)
    print(f"  window busy {busy_ns/1e6:.3f} ms / span {span_ns/1e6:.3f} ms "
          f"({busy_ns/span_ns*100:.1f}% GPU-busy)", file=sys.stderr)
    if map_stats:
        print(f"  gguf order-map: {map_stats['mapped']}/{map_stats['total']} "
              f"matvec dispatches N-matched to weights "
              f"({map_stats['pct']:.1f}%); expected seq len "
              f"{map_stats['seq_len']}/token", file=sys.stderr)


def render_html(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return _HTML_TEMPLATE.replace("__DATA__", data)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>rocprof unified viewer</title>
<style>
  :root{--bg:#12141a;--panel:#1b1e27;--fg:#e6e6e6;--dim:#9aa0ad;--line:#2a2e3a;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;}
  header{padding:10px 16px;border-bottom:1px solid var(--line);}
  h1{font-size:15px;margin:0 0 2px;}
  .sub{color:var(--dim);font-size:11px;}
  .wrap{display:flex;gap:12px;padding:12px 16px;align-items:flex-start;}
  .left{flex:1 1 auto;min-width:0;}
  .right{flex:0 0 430px;}
  .bar{display:flex;align-items:center;gap:14px;margin:6px 0 10px;flex-wrap:wrap;}
  button{background:#2a2e3a;color:var(--fg);border:1px solid #3a3f4d;border-radius:6px;
         padding:5px 12px;cursor:pointer;font-size:12px;}
  button:hover{background:#343a49;}
  .legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--dim);}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;
      vertical-align:-1px;}
  canvas{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:8px;
         display:block;}
  .share{display:flex;height:22px;border-radius:6px;overflow:hidden;margin:10px 0;
          border:1px solid var(--line);font-size:10px;}
  .share div{display:flex;align-items:center;justify-content:center;color:#0d0f14;
             font-weight:600;white-space:nowrap;}
  table{width:100%;border-collapse:collapse;font-size:11px;}
  th,td{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line);}
  th:first-child,td:first-child{text-align:left;}
  th{color:var(--dim);font-weight:600;position:sticky;top:0;background:var(--panel);}
  /* Selected-kernel detail: keep label + value adjacent (not pushed to the two
     edges of the pane like the full-width family table). */
  #detail table{width:auto;}
  #detail td{text-align:left;}
  #detail td:first-child{color:var(--dim);padding-right:18px;white-space:nowrap;}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
         padding:10px 12px;max-height:72vh;overflow:auto;}
  /* .grow panels are not internally scrollable: they expand with their content
     and the page's own (browser) vertical scrollbar handles overflow. */
  .panel.grow{max-height:none;overflow:visible;}
  .panel h2{font-size:12px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;
            letter-spacing:.04em;}
  #hover{position:fixed;pointer-events:none;background:#0b0d12;border:1px solid #3a3f4d;
         border-radius:6px;padding:8px 10px;font-size:11px;max-width:340px;display:none;
         box-shadow:0 4px 16px rgba(0,0,0,.5);z-index:10;}
  #hover .k{color:#7fd1ff;word-break:break-all;}
  #hover .r{color:var(--dim);}
  .fam-dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
           vertical-align:-1px;}
  #tbl tbody tr{cursor:pointer;}
  #tbl tbody tr:hover{background:#1b2130;}
  #tbl tbody tr.sel{background:#2a3550;box-shadow:inset 3px 0 0 #ffffff;}
  #tbl tfoot td{position:sticky;bottom:0;background:var(--panel);color:#cfd6e4;
                font-weight:600;border-top:1px solid var(--line);}
  #tbl tfoot tr:first-child td{border-top:2px solid #4a5165;}
  .lane-label{color:var(--dim);font-size:10px;margin:8px 0 2px;}
</style></head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="sub" id="sub"></div>
</header>
<div class="wrap">
  <div class="left">
    <div class="bar">
      <button id="prev">&larr; prev</button>
      <button id="next">next &rarr;</button>
      <button id="zin">zoom +</button>
      <button id="zout">zoom &minus;</button>
      <button id="reset">reset</button>
      <button id="markhome">markers &rarr; view</button>
      <span id="viewinfo" class="sub"></span>
      <span class="legend" id="legend"></span>
    </div>
    <div class="lane-label">CPU / host (HIP API)</div>
    <div class="lane-label" style="margin-top:0" id="cpunote"></div>
    <canvas id="cv"></canvas>
    <div class="share" id="share"></div>
    <div class="sub">Time-share over the visible window. GPU-idle = wall not covered
      by any kernel (launch latency / host relaunch not hidden by GPU work).</div>
    <div id="detail" class="panel" style="margin-top:12px;display:none"></div>
  </div>
  <div class="right">
    <div class="panel grow">
      <h2>Per-kernel-family (baked span)</h2>
      <table id="tbl"><thead><tr>
        <th>family</th><th>count/tok</th><th>busy%</th><th title="measured achieved DRAM bandwidth: FETCH_SIZE bytes/dispatch / kernel time, vs arch peak">GB/s(%pk)</th><th>MemBusy</th>
        <th>L2hit</th><th>Occ</th><th>stall</th>
      </tr></thead><tbody></tbody><tfoot></tfoot></table>
      <div class="sub" id="bwnote" style="margin-top:8px"></div>
    </div>
  </div>
</div>
<div id="hover"></div>
<script>
const D = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CPU_H = 70, GPU_H = 150, PAD_T = 8, GAP = 26, AXIS_H = 22;
const H = PAD_T + CPU_H + GAP + GPU_H + AXIS_H;
let view0 = D.tok_starts[D.view_i0];
let view1 = D.tok_starts[D.view_i1];
let curI0 = D.view_i0, curI1 = D.view_i1;
let rects = [];  // hit-test: {x,y,w,h,type,payload}
let markA = view0 + (view1-view0)*0.33;   // measurement markers (ns, time coords)
let markB = view0 + (view1-view0)*0.66;
let markDrag = 0;                          // 0=none, 1=A, 2=B

function fmtus(ns){return (ns/1000).toFixed(1)+' \u00b5s';}
function fmtms(ns){return (ns/1e6).toFixed(3)+' ms';}
function fmtdur(ns){return Math.abs(ns)>=1e6 ? fmtms(ns) : fmtus(ns);}
// {"4":9,"2":5} -> "9x4B + 5x2B" (per-lane load widths, widest first)
function fmtLoads(m){
  const ks=Object.keys(m||{}).map(Number).sort((a,b)=>b-a);
  if(!ks.length) return '-';
  return ks.map(k=>`${m[k]}x${k}B`).join(' + ');
}

document.getElementById('title').textContent = D.title;
document.getElementById('sub').textContent =
  `baked ${D.n_tokens_baked} tokens | ${D.gpu.length} GPU slices | `+
  `${D.cpu.length} HIP calls | window GPU-busy ${fmtms(D.busy_ns)} / span `+
  `${fmtms(D.span_ns)}` + (D.has_pmc ? '' : ' | NO PMC (uncolored)') +
  (D.map_stats ? ` | GGUF map ${D.map_stats.mapped}/${D.map_stats.total} matvec `+
    `(${D.map_stats.pct}%)` : '');
document.getElementById('cpunote').textContent = D.has_cpu ? '' :
  '(no --hip-csv supplied: CPU lane empty)';
document.getElementById('bwnote').textContent = D.has_bw ?
  ('GB/s(%pk) = MEASURED achieved DRAM bandwidth: FETCH_SIZE bytes/dispatch (post-L2 '+
   'actual VRAM read traffic) / kernel time, vs '+D.peak_bw_gbs+' GB/s peak. Measured '+
   'attributes bytes to the exact kernel that moved them (no shared-quant-type '+
   'over-counting), so every family gets a real roofline number.')
  : '(no --fetch-csv supplied: achieved-bandwidth column omitted)';

// legend
const legOrder = ['memory','compute','occupancy','lds','copy','unknown'];
const legLabel = {memory:'memory-bound',compute:'compute',occupancy:'occupancy/latency',
                  lds:'LDS',copy:'copy/overhead',unknown:'no PMC'};
document.getElementById('legend').innerHTML = legOrder.map(k=>
  `<span><span class="sw" style="background:${D.colors[k]}"></span>${legLabel[k]}</span>`
).join('');

function resize(){
  const w = cv.clientWidth;
  cv.width = w * devicePixelRatio; cv.height = H * devicePixelRatio;
  cv.style.height = H + 'px';
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  draw();
}
window.addEventListener('resize', resize);

function xOf(ns, w){ return (ns - view0) / (view1 - view0) * w; }

function draw(){
  const w = cv.clientWidth;
  ctx.clearRect(0,0,w,H);
  rects = [];
  const cpuY = PAD_T, gpuY = PAD_T + CPU_H + GAP;

  // lane backgrounds
  ctx.fillStyle = '#161922';
  ctx.fillRect(0,cpuY,w,CPU_H); ctx.fillRect(0,gpuY,w,GPU_H);

  // token boundary markers + inter-token idle shading on GPU lane
  ctx.strokeStyle = '#3a4f6a'; ctx.lineWidth = 1;
  ctx.fillStyle = '#8fb0d8'; ctx.font = '10px sans-serif';
  for (let k=0;k<D.tok_starts.length;k++){
    const ts = D.tok_starts[k];
    if (ts < view0-1 || ts > view1+1) continue;
    const x = xOf(ts, w);
    ctx.beginPath(); ctx.moveTo(x,cpuY); ctx.lineTo(x,gpuY+GPU_H); ctx.stroke();
    ctx.fillText('tok '+k, x+3, gpuY+GPU_H+13);
  }

  // GPU slices
  for (const s of D.gpu){
    if (s.e < view0 || s.s > view1) continue;
    const x = xOf(s.s,w), x2 = xOf(s.e,w);
    const wpx = Math.max(1, x2-x);
    const isSel = selectedFam && s.fam===selectedFam;
    ctx.globalAlpha = (selectedFam && !isSel) ? 0.25 : 1.0;
    ctx.fillStyle = D.colors[s.stall] || D.colors.unknown;
    ctx.fillRect(x, gpuY+4, wpx, GPU_H-8);
    ctx.globalAlpha = 1.0;
    if(isSel){ ctx.strokeStyle='#ffffff'; ctx.lineWidth=1.5;
      ctx.strokeRect(x+0.5, gpuY+4.5, Math.max(1,wpx-1), GPU_H-9); ctx.lineWidth=1; }
    if(selectedSlice && s===selectedSlice){
      ctx.strokeStyle='#ffffff'; ctx.lineWidth=2;
      ctx.strokeRect(x-0.5, gpuY+2.5, Math.max(2,wpx+1), GPU_H-5); ctx.lineWidth=1;
      ctx.fillStyle='#ffffff'; const cx=x+wpx/2;   // caret so thin slices stay findable
      ctx.beginPath(); ctx.moveTo(cx-4,gpuY-5); ctx.lineTo(cx+4,gpuY-5);
      ctx.lineTo(cx,gpuY+1); ctx.closePath(); ctx.fill();
    }
    rects.push({x:x,y:gpuY+4,w:wpx,h:GPU_H-8,type:'gpu',p:s});
  }
  // CPU slices (host may nest; draw thin stacked)
  for (const c of D.cpu){
    if (c.e < view0 || c.s > view1) continue;
    const x = xOf(c.s,w), x2 = xOf(c.e,w);
    const wpx = Math.max(1, x2-x);
    ctx.fillStyle = c.name.indexOf('Graph')>=0 ? '#5fa8d3' :
                    c.name.indexOf('Memcpy')>=0 ? '#c9a227' :
                    c.name.indexOf('Synchronize')>=0 ? '#7a6f9b' : '#4a6070';
    ctx.fillRect(x, cpuY+4, wpx, CPU_H-8);
    rects.push({x:x,y:cpuY+4,w:wpx,h:CPU_H-8,type:'cpu',p:c});
  }

  // axis
  ctx.fillStyle = '#7a8090'; ctx.font = '10px sans-serif';
  const span = view1-view0;
  for (let i=0;i<=5;i++){
    const t = view0 + span*i/5, x = xOf(t,w);
    ctx.fillText(fmtus(t-view0), Math.min(x+2,w-40), H-6);
    ctx.strokeStyle='#5a6070'; ctx.beginPath();
    ctx.moveTo(x,gpuY+GPU_H); ctx.lineTo(x,gpuY+GPU_H+6); ctx.stroke();
  }

  // measurement markers A/B (draggable; full height so you can line up an edge)
  const gpuBot = gpuY+GPU_H;
  [[markA,'#00e5ff','A'],[markB,'#ffd400','B']].forEach(m=>{
    const t=m[0], col=m[1], lab=m[2];
    if (t<view0 || t>view1) return;
    const x=xOf(t,w);
    ctx.strokeStyle=col; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(x,PAD_T); ctx.lineTo(x,gpuBot); ctx.stroke();
    ctx.lineWidth=1;
    ctx.fillStyle=col; ctx.fillRect(x-6,PAD_T,12,12);
    ctx.fillStyle='#000'; ctx.font='bold 10px sans-serif';
    ctx.textAlign='center'; ctx.fillText(lab, x, PAD_T+9); ctx.textAlign='start';
  });
  // dt readout between the markers
  {
    const dt=Math.abs(markB-markA);
    const xa=xOf(markA,w), xb=xOf(markB,w), xm=(xa+xb)/2;
    if (Math.max(markA,markB)>=view0 && Math.min(markA,markB)<=view1){
      ctx.fillStyle='#12161f';
      ctx.fillRect(Math.min(Math.max(xm,44),w-44)-40, PAD_T+15, 80, 15);
      ctx.fillStyle='#e6e6e6'; ctx.font='bold 11px sans-serif';
      ctx.textAlign='center';
      ctx.fillText('dt '+fmtdur(dt), Math.min(Math.max(xm,44),w-44), PAD_T+26);
      ctx.textAlign='start';
    }
  }

  // time-share bar over the visible window
  let gpuBusy=0, cpuBusy=0;
  for (const s of D.gpu){ const a=Math.max(s.s,view0),b=Math.min(s.e,view1);
    if (b>a) gpuBusy += b-a; }
  for (const c of D.cpu){ const a=Math.max(c.s,view0),b=Math.min(c.e,view1);
    if (b>a) cpuBusy += b-a; }
  const idle = Math.max(0, span - gpuBusy);
  const sh = document.getElementById('share');
  const gp=(gpuBusy/span*100), ip=(idle/span*100), cp=(cpuBusy/span*100);
  sh.innerHTML =
    `<div style="width:${gp}%;background:#4363d8">GPU busy ${gp.toFixed(0)}%</div>`+
    `<div style="width:${ip}%;background:#9a9a9a">GPU idle ${ip.toFixed(0)}%</div>`;
  // which baked tokens are (partly) visible
  let tv=[]; for(let k=0;k<D.tok_starts.length;k++){const a=D.tok_starts[k],
    b=(k+1<D.tok_starts.length?D.tok_starts[k+1]:D.span_ns);
    if(b>view0&&a<view1) tv.push(k);}
  const tlabel = tv.length? (tv.length===1?`tok ${tv[0]}`:`tok ${tv[0]}-${tv[tv.length-1]}`):'-';
  document.getElementById('viewinfo').textContent =
    `${tlabel} | ${fmtms(span)} | GPU busy ${gp.toFixed(0)}% `+
    `idle ${ip.toFixed(0)}% | A-B dt ${fmtdur(Math.abs(markB-markA))} | `+
    `drag=zoom-box  shift+drag/arrows=pan  scroll/+-=zoom  drag A/B=measure  click=select kernel`;
}

// summary table
const tb = document.querySelector('#tbl tbody');
tb.innerHTML = D.summary.map(r=>{
  const col = D.colors[r.stall]||D.colors.unknown;
  const bw = r.bw_gbs
    ? `${r.bw_gbs} <span class="r">(${r.bw_pct}%)</span>`
    : '-';
  return `<tr data-fam="${r.fam}"><td><span class="fam-dot" style="background:${col}"></span>${r.fam}</td>`+
    `<td>${r.per_tok}</td><td>${r.busy_pct}</td><td>${bw}</td><td>${r.mem}</td>`+
    `<td>${r.l2}</td><td>${r.occ}</td><td>${r.stall}</td></tr>`;
}).join('');
// selection: a table row selects a FAMILY (dims other families); a click on the
// timeline selects a SINGLE kernel (bright outline) and shows its details below.
let selectedFam = null;
let selectedSlice = null;
function setSelection(fam){
  selectedFam = fam;
  selectedSlice = null;          // family mode clears single-kernel selection
  const rows = tb.querySelectorAll('tr');
  rows.forEach(tr=>tr.classList.toggle('sel', fam!==null && tr.dataset.fam===fam));
  if(fam){ const el=[...rows].find(tr=>tr.dataset.fam===fam);
           if(el) el.scrollIntoView({block:'nearest'}); }
  updateDetail(); draw();
}
function selectSlice(sl){        // sl is a slice object from D.gpu (or null)
  selectedSlice = sl; selectedFam = null;
  const rows = tb.querySelectorAll('tr');
  rows.forEach(tr=>tr.classList.toggle('sel', sl && tr.dataset.fam===sl.fam));
  if(sl){ const el=[...rows].find(tr=>tr.dataset.fam===sl.fam);
          if(el) el.scrollIntoView({block:'nearest'}); }
  updateDetail(); draw();
}
// Rule-based per-kernel bottleneck verdict. Thresholds are heuristics tuned on
// gfx1151 decode traces; the verdict is a HINT shown beside the raw numbers, not
// a replacement for them. Priority order matters: DRAM-bound is tested before
// VALU-bound because BW-bound matvec kernels also read ALU>100% (VALU cycles
// count across 4 SIMDs/CU), so ALU alone would mislabel them.
function diagnose(fc){
  if(!fc) return null;
  const ea=+fc.ea||0, alu=+fc.alu||0, occ=+fc.occ||0, l2=+fc.l2||0,
        bw=+fc.bw_pct||0, scr=+fc.scratch||0, vg=+fc.vgpr||0;
  const lw=fc.loadw||{}, lane=+lw.dominant_lane_bytes||0;
  const havePmc = (fc.ea!==undefined) && (ea||alu||occ||(+fc.mem||0));
  if(scr>0) return {c:'#ff6b6b',t:'REGISTER SPILLING',
    a:`spilling ${scr}B to scratch -- cut live VGPRs (split kernel / reduce unroll) `+
      `to stop spill/fill traffic.`};
  if(!havePmc){
    if(bw) return {c:'#7fd1ff',t:'DRAM TRAFFIC',
      a:`achieved ${bw}% of peak BW; no PMC counters for this family (not in the `+
        `--pmc set) so the bottleneck is unconfirmed.`};
    return null;
  }
  if(ea>=85 || bw>=80){
    // "widen loads" only helps when the DRAM interface is busy but NOT
    // delivering peak BW (bw<75) -- a sign of partial-line/uncoalesced access
    // that wider loads can fix. Near peak (e.g. Q6_K at 96%), narrow loads are
    // fine and the only lever is moving fewer bytes, so don't suggest it.
    const widen = (bw && bw<75 && lane && lane<16)
      ? ` Loads are ${lane}B/lane and BW is only ${bw}% of peak: try widening to `+
        `16B/b128 to coalesce fuller cache lines.` : ``;
    return {c:'#7fd1ff',t:'DRAM-BOUND',
      a:`EA (DRAM iface) ${ea}%${bw?`, ${bw}% of peak BW`:''}, L2 hit ${l2}% -- streaming `+
        `from DRAM. Occupancy/VGPR won't help; cut bytes moved (better quant, fusion).`+widen};
  }
  if(alu>=100) return {c:'#ffb454',t:'VALU / COMPUTE-BOUND',
    a:`ALU ${alu}% (VALU busy across 4 SIMDs) with EA only ${ea}% -- compute-bound. `+
      `More registers/ILP beats more waves; look for redundant math or low-throughput ops.`};
  if(occ<50) return {c:'#ffd54a',t:'LOW OCCUPANCY / LATENCY-BOUND',
    a:`occupancy ${occ}% with EA ${ea}% / ALU ${alu}% both moderate -- too few resident `+
      `waves to hide latency`+
      (vg>96?`; VGPR=${vg} caps it (>96/thread for wave32). Cut VGPRs to raise occupancy.`
            :`; likely a short or small-grid kernel dominated by ramp/drain tails.`)};
  return {c:'#9aa6b2',t:'BALANCED',
    a:`no single dominant bottleneck: EA ${ea}%, ALU ${alu}%, occupancy ${occ}%.`};
}
// details panel below the lanes for the single selected kernel
const dp = document.getElementById('detail');
function updateDetail(){
  if(!selectedSlice){ dp.style.display='none'; dp.innerHTML=''; return; }
  const s=selectedSlice, fc=D.fam_counters[s.fam]||{};
  const dg=diagnose(fc);
  let h=`<h2>Selected kernel</h2>`+
    `<div style="color:#7fd1ff;word-break:break-all;margin-bottom:6px">${s.fam}</div>`+
    (dg?`<div style="margin:0 0 8px;padding:6px 9px;border-left:3px solid ${dg.c};`+
        `background:rgba(255,255,255,.05);border-radius:3px;line-height:1.35">`+
        `<b style="color:${dg.c}">${dg.t}</b> `+
        `<span style="color:#c8d0da">${dg.a}</span></div>`:``)+
    `<table><tbody>`+
    `<tr><td>duration</td><td>${fmtus(s.e-s.s)}</td></tr>`;
  if(s.map){
    const m=s.map, KB=b=>b>=1048576?(b/1048576).toFixed(1)+' MB':(b/1024).toFixed(1)+' KB';
    const nOk=m.nmatch?'#8fe388':'#ff6b6b';
    h+=`<tr><td>weight tensor</td><td style="color:#ffd479;word-break:break-all">${m.nm}`+
       `<span class="r"> (L${m.L<0?'out':m.L} ${m.role})</span></td></tr>`+
       `<tr><td>true shape [K x N]</td><td>${m.K} x ${m.trueN} <span class="r">(${m.q})</span></td></tr>`+
       `<tr><td>launch N (rows)</td><td style="color:${nOk}">${m.launchN}`+
       (m.padN?` <span class="r">(+${m.padN} padding rows vs true ${m.trueN})</span>`
             :` <span class="r">(== true N, no output-row padding)</span>`)+`</td></tr>`;
    if(m.padK) h+=`<tr><td>K block padding</td><td>+${m.padK} <span class="r">(to 256-elem quant block)</span></td></tr>`;
    h+=`<tr><td>packed footprint</td><td>${KB(m.packed)} <span class="r">(theoretical, on-disk)</span></td></tr>`;
    if(m.measured) h+=`<tr><td>over-fetch</td><td>${m.overfetch}x <span class="r">(measured ${KB(m.measured)} / packed)</span></td></tr>`;
  }
  if(D.has_bw && fc.bw_gbs){
    h += `<tr><td>achieved BW</td><td>${fc.bw_gbs} GB/s (${fc.bw_pct}% of ${D.peak_bw_gbs}), ${fc.kb_disp} KB/disp</td></tr>`;
  }
  if(D.has_pmc){
    h+=`<tr><td>MemUnitBusy</td><td>${fc.mem}%</td></tr>`+
       `<tr><td>L2 hit</td><td>${fc.l2}%</td></tr>`+
       `<tr><td>Occupancy</td><td>${fc.occ}%</td></tr>`+
       `<tr><td>LDS bank conflict</td><td>${fc.lds}</td></tr>`+
       `<tr><td>WriteUnitStalled</td><td>${fc.wr}</td></tr>`+
       `<tr><td>Wavefronts</td><td>${fc.wav}</td></tr>`+
       `<tr><td>dominant stall</td><td style="color:${D.colors[s.stall]};font-weight:600">${s.stall}</td></tr>`;
    // Traffic amplification = load-unit bytes / DRAM bytes = 1/(1 - L2hit).
    // How many times each DRAM-fetched byte was re-served by L2. ~1.0 =
    // read-once/streaming (matvec/decode); >>1 = the kernel re-reads data L2
    // caught -- a tiling/reuse smell (stage it in LDS/registers). NOTE this only
    // sees re-reads L2 absorbed; refetch that MISSES L2 needs the unique
    // footprint (not known here) to detect -- watch low L2 hit + high DRAM bytes.
    const l2h=Math.min(fc.l2||0,99.9), amp=1/(1-l2h/100);
    // Byte breakdown per dispatch: FETCH_SIZE (DRAM, measured) + cache-hit
    // (L2 re-served, derived from L2 hit%) = load-unit bytes the loads requested.
    const fkb=fc.kb_disp||0, hitkb=fkb*l2h/(100-l2h), lukb=fkb+hitkb;
    h+=`<tr><td>traffic amplification</td><td>${amp.toFixed(2)}x`+
       (fkb?` <span class="r">(load-unit ${lukb.toFixed(1)} = FETCH_SIZE ${fkb.toFixed(1)}`+
            ` + cache-hit ${hitkb.toFixed(1)} KB/disp)</span>`
           :` <span class="r">(load bytes / DRAM bytes; L2-absorbed re-reads)</span>`)+
       `</td></tr>`;
    if(fc.ea) h+=`<tr><td>EA (DRAM iface) busy</td><td>${fc.ea}%</td></tr>`;
    if(fc.alu) h+=`<tr><td>ALU (VALU) busy</td><td>${fc.alu}%</td></tr>`;
    if(fc.vgpr) h+=`<tr><td>VGPR / SGPR</td><td>${fc.vgpr} / ${fc.sgpr}`+
       (fc.accum_vgpr?` (+${fc.accum_vgpr} accum)`:``)+`</td></tr>`;
    if(fc.wave) h+=`<tr><td>wavefront size</td><td>wave${fc.wave} `+
       `<span class="r">(grid / wavefronts)</span></td></tr>`;
    const sc=fc.scratch||0;
    h+=`<tr><td>scratch size</td><td>${sc>=1024?(sc/1024).toFixed(1)+' KB':sc+' B'}</td></tr>`;
  }
  if(D.has_loadw && fc.loadw){
    const lw=fc.loadw;
    h+=`<tr><td>mem load width (vector)</td><td>${fmtLoads(lw.vector_loads)} `+
       `<span class="r">(dominant ${lw.dominant_lane_bytes}B/lane)</span></td></tr>`;
    if(lw.scalar_loads && Object.keys(lw.scalar_loads).length)
      h+=`<tr><td>load width (scalar/uniform)</td><td>${fmtLoads(lw.scalar_loads)}</td></tr>`;
    if(lw.lds_loads && Object.keys(lw.lds_loads).length)
      h+=`<tr><td>load width (LDS)</td><td>${fmtLoads(lw.lds_loads)}</td></tr>`;
  }
  h+=`</tbody></table>`+
     `<div class="sub" style="margin-top:6px">duration is for this exact `+
     `dispatch; PMC, achieved BW + load width are per-family (PMC + FETCH_SIZE from `+
     `separate runs; load width from device disassembly). vector = per-lane global/`+
     `buffer loads (b32=4B, d16=2B, u8=1B); scalar = s_load uniform; LDS = ds_ reads.`+
     `<br>traffic amplification = load-unit / FETCH_SIZE = 1/(1-L2hit). `+
     `<b>FETCH_SIZE</b> = DRAM bytes read (GL2C-&gt;EA, L2 misses only, measured). `+
     `<b>cache-hit</b> = bytes L2 re-served (derived: FETCH_SIZE*hit/(1-hit)). `+
     `<b>theoretical input</b> = unique bytes touched once (weights read once) = the `+
     `algorithmic minimum; NOT a counter -- supply it from model/kernel shapes. `+
     `For a read-once kernel FETCH_SIZE ~ theoretical, so FETCH_SIZE well above the `+
     `known weight footprint = DRAM over-fetch (bad tiling missing L2).</div>`;
  dp.innerHTML=h; dp.style.display='block';
}
tb.querySelectorAll('tr').forEach(tr=>{
  tr.onclick=()=>setSelection(selectedFam===tr.dataset.fam ? null : tr.dataset.fam);
});
// totals footer: dispatches/token, kernel-busy/token, inter-kernel gap/token, mean gap
{
  const tf = document.querySelector('#tbl tfoot');
  const totCountTok = D.summary.reduce((a,r)=>a+r.per_tok,0);
  const totCount = D.summary.reduce((a,r)=>a+r.count,0);
  const timePerTok = D.n_tokens_baked ? D.busy_ns/D.n_tokens_baked : 0;
  const totGap = Math.max(0, D.span_ns - D.busy_ns);
  const gapPerTok = D.n_tokens_baked ? totGap/D.n_tokens_baked : 0;
  const nGaps = totCount>1 ? totCount-1 : 1;
  const avgGap = totGap/nGaps;
  let rows =
    `<tr><td colspan="7">total count / token</td><td>${totCountTok.toFixed(1)}</td></tr>`+
    `<tr><td colspan="7">total kernel time / token</td><td>${fmtdur(timePerTok)}</td></tr>`+
    `<tr><td colspan="7">total CP Transition Gap / token</td><td>${fmtdur(gapPerTok)}</td></tr>`+
    `<tr><td colspan="7">avg CP Transition Gap/kernel</td><td>${fmtdur(avgGap)}</td></tr>`;
  if (D.has_bw){
    const totMB = D.summary.reduce((a,r)=>a+(r.mb_tok||0),0);
    const bwAgg = timePerTok ? totMB*1e6/timePerTok : 0;  // bytes/ns == GB/s
    rows +=
      `<tr><td colspan="7">DRAM read / token (all kernels, measured)</td><td>${totMB.toFixed(0)} MB</td></tr>`+
      `<tr><td colspan="7">achieved BW (all bytes / kernel time)</td>`+
      `<td>${bwAgg.toFixed(0)} GB/s (${(bwAgg/D.peak_bw_gbs*100).toFixed(0)}%)</td></tr>`;
  }
  tf.innerHTML = rows;
}

// hover
const hv = document.getElementById('hover');
cv.addEventListener('mousemove', e=>{
  if(dragging||rbActive||markDrag){hv.style.display='none';return;}
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  // cursor hint when hovering a marker line
  const wC=cv.clientWidth;
  cv.style.cursor = Math.min(Math.abs(mx-xOf(markA,wC)),Math.abs(mx-xOf(markB,wC)))<=6 ? 'ew-resize' : '';
  let hit=null;
  for (let i=rects.length-1;i>=0;i--){const q=rects[i];
    if(mx>=q.x&&mx<=q.x+q.w&&my>=q.y&&my<=q.y+q.h){hit=q;break;}}
  if(!hit){hv.style.display='none';return;}
  let html='';
  if(hit.type==='gpu'){
    const s=hit.p, fc=D.fam_counters[s.fam]||{};
    html=`<div class="k">${s.fam}</div>`+
      `<div class="r">dur ${fmtus(s.e-s.s)}</div>`+
      (s.map?`<div style="color:#ffd479">${s.map.role} [${s.map.K}x${s.map.trueN}] ${s.map.q}`+
        (s.map.padN?` +${s.map.padN} pad`:` (no pad)`)+
        (s.map.overfetch?`, ${s.map.overfetch}x fetch`:``)+`</div>`:'')+
      ((D.has_bw && fc.bw_gbs)?
        `<div style="color:#7fd1ff">achieved BW ~ ${fc.bw_gbs} GB/s `+
        `(${fc.bw_pct}% of ${D.peak_bw_gbs} peak), ${fc.kb_disp} KB/disp</div>`:'')+
      ((D.has_loadw && fc.loadw)?
        `<div class="r">load ${fmtLoads(fc.loadw.vector_loads)} vec `+
        `(dom ${fc.loadw.dominant_lane_bytes}B/lane)</div>`:'')+
      (D.has_pmc?`<div class="r">MemUnitBusy ${fc.mem}%  L2hit ${fc.l2}%</div>`+
        `<div class="r">Occ ${fc.occ}%  LDSbc ${fc.lds}  WrStall ${fc.wr}  Wav ${fc.wav}</div>`+
        `<div style="color:${D.colors[s.stall]};font-weight:600">\u2192 dominant stall: ${s.stall}</div>`
        :`<div class="r">(no PMC data)</div>`);
  } else {
    const c=hit.p;
    html=`<div class="k">${c.name}</div><div class="r">host dur ${fmtus(c.e-c.s)}</div>`;
  }
  hv.innerHTML=html; hv.style.display='block';
  hv.style.left=Math.min(e.clientX+14, innerWidth-360)+'px';
  hv.style.top=(e.clientY+14)+'px';
});
cv.addEventListener('mouseleave', ()=>hv.style.display='none');

// --- view control: free zoom/pan within the baked span [0, span_ns] -----------
const BMIN=0, BMAX=D.span_ns, MINSPAN=2000;  // don't zoom below 2 us
function clampView(){
  let s=view1-view0;
  if(s<MINSPAN){const m=(view0+view1)/2; view0=m-MINSPAN/2; view1=m+MINSPAN/2; s=MINSPAN;}
  if(s>BMAX-BMIN){view0=BMIN; view1=BMAX; return;}
  if(view0<BMIN){view1+=BMIN-view0; view0=BMIN;}
  if(view1>BMAX){view0-=view1-BMAX; view1=BMAX;}
}
function setView(v0,v1){view0=v0; view1=v1; clampView(); draw();}
function zoomAt(frac, factor){       // frac = fixed point (0..1) across viewport
  const s=view1-view0, ns=s*factor, ft=view0+frac*s;
  setView(ft-frac*ns, ft+(1-frac)*ns);
}
// pan/zoom by one token width (nearest baked token to the viewport center)
function tokenStep(dir){
  const c=(view0+view1)/2; let k=0;
  for(let i=0;i<D.tok_starts.length;i++) if(D.tok_starts[i]<=c) k=i;
  const t0=D.tok_starts[k], t1=(k+1<D.tok_starts.length?D.tok_starts[k+1]:BMAX);
  setView(view0+dir*(t1-t0), view1+dir*(t1-t0));
}
document.getElementById('prev').onclick=()=>tokenStep(-1);
document.getElementById('next').onclick=()=>tokenStep(1);
document.getElementById('zin').onclick=()=>zoomAt(0.5,0.6);
document.getElementById('zout').onclick=()=>zoomAt(0.5,1/0.6);
document.getElementById('reset').onclick=()=>setView(D.tok_starts[D.view_i0],D.tok_starts[D.view_i1]);

cv.addEventListener('wheel', e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect();
  const frac=Math.min(1,Math.max(0,(e.clientX-r.left)/cv.clientWidth));
  zoomAt(frac, e.deltaY<0 ? 0.85 : 1/0.85);
}, {passive:false});

// arrow keys pan the view (Shift = a full page); +/- zoom
window.addEventListener('keydown', e=>{
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    e.preventDefault();
    const s=view1-view0, step=s*(e.shiftKey?1.0:0.25)*(e.key==='ArrowLeft'?-1:1);
    setView(view0+step, view1+step);
  } else if(e.key==='+'||e.key==='='){ e.preventDefault(); zoomAt(0.5,0.6); }
  else if(e.key==='-'||e.key==='_'){ e.preventDefault(); zoomAt(0.5,1/0.6); }
});

// left-drag = rubber-band zoom (draw a box, release to zoom to that X range);
// shift+left-drag = pan.
let dragging=false, dragX=0, dv0=0, dv1=0;   // pan state
let rbActive=false, rbX0=0, rbY0=0, rbX1=0;  // rubber-band state (+down y for click)
function tOf(px){ return view0 + px/cv.clientWidth*(view1-view0); }
function overlay(){                          // paint the selection box atop draw()
  if(!rbActive) return;
  const gpuBot=PAD_T+CPU_H+GAP+GPU_H;
  const x=Math.min(rbX0,rbX1), wpx=Math.abs(rbX1-rbX0);
  ctx.save();
  ctx.fillStyle='rgba(120,170,255,0.20)';
  ctx.strokeStyle='rgba(120,170,255,0.9)'; ctx.lineWidth=1;
  ctx.fillRect(x,PAD_T,wpx,gpuBot-PAD_T);
  ctx.strokeRect(x+0.5,PAD_T+0.5,wpx,gpuBot-PAD_T);
  ctx.restore();
}
cv.addEventListener('mousedown', e=>{
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, w=cv.clientWidth;
  hv.style.display='none';
  // grab a measurement marker if the click is within 6px of its line
  const dA=Math.abs(mx-xOf(markA,w)), dB=Math.abs(mx-xOf(markB,w));
  if(Math.min(dA,dB)<=6){ markDrag = (dA<=dB?1:2); cv.style.cursor='ew-resize'; return; }
  if(e.shiftKey){ dragging=true; dragX=e.clientX; dv0=view0; dv1=view1;
    cv.style.cursor='grabbing'; }
  else { rbActive=true; rbX0=mx; rbY0=e.clientY-r.top; rbX1=mx; cv.style.cursor='crosshair'; }
});
// hit-test a point against the drawn rects; select the GPU slice's family (or clear)
function clickSelect(mx,my){
  for(let i=rects.length-1;i>=0;i--){const q=rects[i];
    if(mx>=q.x&&mx<=q.x+q.w&&my>=q.y&&my<=q.y+q.h){
      if(q.type==='gpu'){ selectSlice(selectedSlice===q.p ? null : q.p); return; }
      break;
    }}
  selectSlice(null);
}
window.addEventListener('mousemove', e=>{
  if(markDrag){
    const r=cv.getBoundingClientRect();
    const mx=Math.min(cv.clientWidth, Math.max(0, e.clientX-r.left));
    if(markDrag===1) markA=tOf(mx); else markB=tOf(mx);
    draw(); return;
  }
  if(dragging){
    const dt=(e.clientX-dragX)/cv.clientWidth*(dv1-dv0);
    setView(dv0-dt, dv1-dt); return;
  }
  if(rbActive){
    const r=cv.getBoundingClientRect();
    rbX1=Math.min(cv.clientWidth, Math.max(0, e.clientX-r.left));
    draw(); overlay();
  }
});
window.addEventListener('mouseup', ()=>{
  if(markDrag){ markDrag=0; cv.style.cursor=''; return; }
  if(dragging){ dragging=false; cv.style.cursor=''; }
  if(rbActive){
    rbActive=false; cv.style.cursor='';
    if(Math.abs(rbX1-rbX0)>=4) setView(tOf(Math.min(rbX0,rbX1)), tOf(Math.max(rbX0,rbX1)));
    else clickSelect(rbX0, rbY0);   // no drag = a click: select the kernel under it
  }
});
// bring both markers back into the current viewport (button + double-click)
function markersToView(){
  markA=view0+(view1-view0)*0.33; markB=view0+(view1-view0)*0.66; draw();
}
document.getElementById('markhome').onclick=markersToView;
cv.addEventListener('dblclick', markersToView);

resize();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()

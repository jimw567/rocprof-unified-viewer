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
import base64
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

try:
    from isa_glossary import ISA_GLOSSARY, REG_GLOSSARY
except ImportError:
    ISA_GLOSSARY = {}
    REG_GLOSSARY = {}


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
            kname = r["Kernel_Name"]
            try:
                gx = int(r["Grid_Size_X"])
                wg = int(r["Workgroup_Size_X"])
                n = gx // wg if wg else 0
                # The wvsplitk decode kernel uses a 2D block (warp_size x WvPrGrp=16)
                # and grid.x = ceil(nrows/16), so Grid_Size_X/Workgroup_Size_X yields
                # ceil(nrows/16), not nrows. Recover true output rows (exact for the
                # 16-aligned decode shapes) so it still order-maps onto its weight.
                if "wvsplitk" in kname:
                    n *= 16
            except (KeyError, ValueError, TypeError):
                n = 0
            # Per-dispatch block (workgroup) count = product over grid dims of
            # (Grid_Size_d / Workgroup_Size_d). Grid_Size_* is in work-items, so the
            # per-dim ratio is that dim's block count. Unlike N above this is NOT
            # scaled for wvsplitk -- grid.x already IS the launched block count.
            try:
                nblk = 1
                for d in ("X", "Y", "Z"):
                    gd = int(r["Grid_Size_" + d]); wd = int(r["Workgroup_Size_" + d])
                    if wd:
                        nblk *= gd // wd
            except (KeyError, ValueError, TypeError):
                nblk = 0
            by_stream[r["Stream_Id"]].append(
                (int(r["Start_Timestamp"]), int(r["End_Timestamp"]),
                 kname, n, nblk))
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
                                "scratch": 0, "lds": 0})
    # Wavefront size = work-items / wavefronts dispatched, derived from
    # Grid_Size / Wavefronts. Both are summed over the SAME dispatches (gated on
    # the Wavefronts counter row, which occurs once per dispatch) so the ratio is
    # exact even when a family mixes dispatch sizes.
    # Tiling geometry: threads/block (Workgroup_Size, constant per family) and the
    # per-family mean block count (mean Grid_Size / Workgroup_Size). ndisp counts
    # dispatches (the Wavefronts counter row occurs once per dispatch).
    wsz = defaultdict(lambda: {"grid": 0.0, "waves": 0.0, "wg": 0, "ndisp": 0})
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
                             ("scratch", "Scratch_Size"),
                             ("lds", "LDS_Block_Size")):
                try:
                    g[key] = max(g[key], int(r[col]))
                except (KeyError, ValueError, TypeError):
                    pass
            if r.get("Counter_Name") == "Wavefronts":
                try:
                    wsz[fam]["grid"] += int(r["Grid_Size"])
                    wsz[fam]["waves"] += float(r["Counter_Value"])
                    wsz[fam]["wg"] = max(wsz[fam]["wg"], int(r["Workgroup_Size"]))
                    wsz[fam]["ndisp"] += 1
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
            "wg": int(w["wg"]),
            "blocks": (int(round((w["grid"] / w["ndisp"]) / w["wg"]))
                       if w["ndisp"] and w["wg"] else 0),
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
                if "wvsplitk" in r["Kernel_Name"]:  # 2D block: recover true rows
                    n *= 16
            except (KeyError, ValueError, TypeError):
                n = 0
            if n:
                agg_n[(fam, n)].append(v)
    by_fam = {fam: statistics.mean(v) * 1024.0 for fam, v in agg.items() if v}
    by_fam_n = {k: statistics.mean(v) * 1024.0 for k, v in agg_n.items() if v}
    return by_fam, by_fam_n


def load_fetch_bytes_mapped(path, expected_seq):
    """Order-map the FETCH_SIZE run to PER-WEIGHT measured DRAM bytes.

    The (family, N) bucket in load_fetch_bytes cannot separate two different
    weights that launch the same N -- e.g. ffn_down [9216 x 2560] and attn_output
    [2560 x 2560] both dispatch N=2560, so they share one blended measurement and
    over-fetch comes out physically impossible (< 1.0x for the bigger, > 1x for the
    smaller). But the FETCH run is also strictly-periodic decode, so each
    mul_mat_vec dispatch can be attached to its exact GGUF weight by execution
    order (the same heuristic the trace uses). Returns {weight_name: mean bytes/
    dispatch} averaged over the clean steady-state tokens in the run, giving an
    honest per-weight over-fetch. Falls back to {} (caller uses the blend) if the
    run cannot be cleanly segmented against expected_seq."""
    if not expected_seq:
        return {}
    L = len(expected_seq)
    vocab_n = expected_seq[-1]["N"]          # output head N delimits each token
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("Counter_Name") != "FETCH_SIZE":
                continue
            if "mul_mat_vec" not in r.get("Kernel_Name", ""):
                continue
            try:
                did = int(r["Dispatch_Id"])
                gs = int(r["Grid_Size"])
                ws = int(r["Workgroup_Size"])
                n = gs // ws if ws else 0
                if "wvsplitk" in r["Kernel_Name"]:  # 2D block: recover true rows
                    n *= 16
                v = float(r["Counter_Value"]) * 1024.0
            except (KeyError, ValueError, TypeError):
                continue
            if n:
                rows.append((did, n, v))
    rows.sort()
    # Segment into tokens at the output head (N == vocab), then keep only clean
    # tokens whose dispatch count matches the expected per-token sequence length.
    toks, cur = [], []
    for _did, n, v in rows:
        cur.append((n, v))
        if n == vocab_n:
            toks.append(cur)
            cur = []
    good = [t for t in toks if len(t) == L]
    if not good:
        return {}
    acc = defaultdict(list)
    for t in good:
        for i, (n, v) in enumerate(t):
            ent = expected_seq[i]
            if ent["N"] == n:                # attach only on shape match
                acc[ent["nm"]].append(v)
    return {nm: statistics.mean(vs) for nm, vs in acc.items() if vs}


def parse_clean_tps(path):
    """Parse the decode (tg) throughput from collect.sh's clean_tps.txt -- the
    untraced llama-bench markdown table. Returns {"test": "tg64", "tps": float,
    "sd": float or None} for the last tg row, or None if the file is missing or
    unparseable. This is the honest tok/s: rocprofv3 perturbs the traced runs,
    so this bare number is what to quote."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    best = None
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        test = next((c for c in cells if re.fullmatch(r"tg\d+", c)), None)
        if not test:
            continue
        m = re.search(r"([0-9]+\.?[0-9]*)\s*(?:\u00b1|\+/-)\s*([0-9]+\.?[0-9]*)",
                      cells[-1])
        if m:
            best = {"test": test, "tps": float(m.group(1)), "sd": float(m.group(2))}
        else:
            m2 = re.search(r"([0-9]+\.?[0-9]*)", cells[-1])
            if m2:
                best = {"test": test, "tps": float(m2.group(1)), "sd": None}
    return best


def load_hw_diagram():
    """Base64 data-URI of docs/rdna35-details.png (the RDNA 3.5 WGP diagram) so the
    overlay can show it inline WITHOUT breaking the self-contained-single-file
    property -- no relative path to resolve once the HTML is moved or web-shared.
    Returns "" if the file is absent (older checkout / stripped install)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "docs", "rdna35-details.png")
    try:
        with open(p, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return ""


def load_loadwidth(path):
    """Load the disassembly load-width JSON ({family: {vector_loads, scalar_loads,
    lds_loads, dominant_lane_bytes, ...}}) produced from the gfx1151 device code
    objects. Keyed by the same family_of() names, so it joins onto slices."""
    with open(path) as fh:
        return json.load(fh)


def load_att_stats(att_dir):
    """Aggregate decoded ATT (Advanced Thread Trace) instruction stats per kernel
    FAMILY so they join onto the same family_of() slices as PMC/loadwidth.

    Reads every `stats_ui_output_*_dispatch_*.csv` under `att_dir`. Each such CSV
    is one traced dispatch and lists per-instruction rows with columns
    CodeObj,Vaddr,Instruction,Hitcount,Latency,Stall,Idle,Source. The demangled
    kernel name only appears on the first row of each function block (the `Source`
    column is blank on the instruction rows that follow), so we carry the last
    non-blank Source forward. A dispatch's decoded output usually contains a few
    neighbouring kernels; every function block is attributed to its own family,
    and the one with the most Stall cycles is the traced target.

    Returns {family: {stall,lat,idle,hits, n_disp, top:[{i,st,idle,hits}],
    byclass:[[opcode,stall]]}} -- cycle totals summed across all traced dispatches
    of that family, top instructions ranked by stall cycles, and stall grouped by
    opcode. Empty dict if no populated stats CSVs are found (all cut off)."""
    import glob
    agg = {}
    for path in sorted(glob.glob(os.path.join(att_dir, "**",
                                              "stats_ui_output_*_dispatch_*.csv"),
                                 recursive=True)):
        # Per-file: which families appear, and their instruction rows.
        fam_rows = defaultdict(list)
        cur = None
        try:
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    instr = (r.get("Instruction") or "").strip()
                    # A family-header row's Instruction is the `;`-prefixed mangled
                    # symbol; its Source column carries the demangled name. Only
                    # these rows set the family. Instruction rows keep the current
                    # family even though a debug (`-g`) build now puts a source
                    # path in their Source column -- treating that as a new family
                    # would spawn one bogus family per source line.
                    if instr.startswith(";"):
                        src = (r.get("Source") or "").strip()
                        if src:
                            cur = family_of(src)
                        continue
                    if not cur or not instr:
                        continue

                    def _i(k):
                        try:
                            return int(r.get(k) or 0)
                        except ValueError:
                            return 0
                    st, idle, hits, lat = (_i("Stall"), _i("Idle"),
                                           _i("Hitcount"), _i("Latency"))
                    if instr:
                        fam_rows[cur].append((instr, st, idle, hits, lat))
        except OSError:
            continue
        for fam, rows in fam_rows.items():
            if not any(h for (_i, _s, _d, h, _l) in rows):
                continue                       # this dispatch was empty/cut off
            a = agg.setdefault(fam, {"stall": 0, "lat": 0, "idle": 0, "hits": 0,
                                     "n_disp": 0, "_instr": defaultdict(
                                         lambda: [0, 0, 0]),
                                     "_class": defaultdict(int)})
            a["n_disp"] += 1
            for instr, st, idle, hits, lat in rows:
                a["stall"] += st
                a["idle"] += idle
                a["hits"] += hits
                a["lat"] += lat
                d = a["_instr"][instr]
                d[0] += st
                d[1] += idle
                d[2] += hits
                a["_class"][instr.split()[0] if instr else "?"] += st
    out = {}
    for fam, a in agg.items():
        top = sorted(a["_instr"].items(), key=lambda kv: -kv[1][0])[:8]
        byclass = sorted(a["_class"].items(), key=lambda kv: -kv[1])[:10]
        out[fam] = {
            "stall": a["stall"], "lat": a["lat"], "idle": a["idle"],
            "hits": a["hits"], "n_disp": a["n_disp"],
            "top": [{"i": i, "st": v[0], "idle": v[1], "hits": v[2]}
                    for i, v in top],
            "byclass": [[op, st] for op, st in byclass],
        }
    return out


def _att_src_split(s):
    """Return (fullpath, line) for the deepest real source location in a decoded
    ATT Source chain, or (None, None) if it is blank / not a real file:line. The
    chain is inline-expanded, e.g.
    `hip_runtime.h:248 -> hip_runtime.h:272 -> mmvq.cu:1034`; the final `->` segment
    is the actual source file/line."""
    s = (s or "").strip()
    if not s:
        return None, None
    if "->" in s:
        s = s.rsplit("->", 1)[-1].strip()
    path, sep, line = s.rpartition(":")
    if not sep or not path:
        return None, None
    if not (line == "?" or line.isdigit()):
        return None, None                    # not a file:line (e.g. a C++ signature)
    return path, line


def _att_src_terminal(s):
    """Reduce a decoded ATT Source chain to the deepest real source location as
    `basename:line` (path stripped so the generated HTML never leaks absolute build
    paths). Returns "" when no line info is present, so callers can gate on it."""
    path, line = _att_src_split(s)
    if path is None:
        return ""
    return (os.path.basename(path) or path) + ":" + line


def _load_att_wave(dispatch_dir, ci2row):
    """Load one representative wave's stitched EXECUTED-instruction stream from a
    decoded ATT dispatch dir, for the debug view's Step mode.

    Each traced dispatch dir holds per-wave files `se*_sm*_sl*_wv*.json`. Each has
    `{duration, name, num_insts, num_stitched, wave{...}}`; `wave.instructions` is a
    list of 5-tuples, one per EXECUTED instruction in issue order (following real
    branches/loops), where col0 is a monotonic cycle timestamp and col4 is the
    0-based index into that dispatch's full code.json `code` array. We pick the wave
    with the most stitched instructions (richest trace) and remap each executed
    step's code-index onto the position of that instruction in the embedded `rows`
    list (via ci2row), so the client can highlight the ISA row + source line and show
    the per-step cycle delta. Steps whose code-index is not an embedded instruction
    row (e.g. a function-header row) are dropped.

    Returns {wave, nexec, t0, stream:[[rowpos, cycle], ...]} or None."""
    import glob
    if not dispatch_dir or not ci2row:
        return None
    best = None                  # (num_stitched, instructions)
    for p in glob.glob(os.path.join(dispatch_dir, "se*_sm*_sl*_wv*.json")):
        try:
            with open(p) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        wv = doc.get("wave") or {}
        insts = wv.get("instructions") or []
        if not insts:
            continue
        ns = doc.get("num_stitched") or len(insts)
        if best is None or ns > best[0]:
            best = (ns, insts, os.path.basename(p))
    if best is None:
        return None
    _ns, insts, name = best
    stream = []
    for it in insts:
        if not isinstance(it, list) or len(it) < 5:
            continue
        try:
            cyc, ci = int(it[0]), int(it[4])
        except (ValueError, TypeError):
            continue
        pos = ci2row.get(ci)
        if pos is None:
            continue                         # header/non-instruction row: skip
        stream.append([pos, cyc])
    if not stream:
        return None
    return {"wave": name, "nexec": len(stream), "t0": stream[0][1],
            "stream": stream}


_WAVE_NB = 900   # horizontal bucket budget for the Wave View global view


def load_att_waves(dispatch_dir):
    """Load ALL captured waves' state timelines from one decoded ATT dispatch dir,
    for the debug view's "Wave View" global view (the rocprof-compute-viewer-style
    occupancy panel: every wave is a lane, the shared X axis is cycles, each lane is
    colored by hardware state over time).

    Each per-wave file `se*_sm*_sl*_wv*.json` carries `wave.timeline`, a run-length
    list of `[state, cycles]` segments that sums exactly to the wave's duration
    (`end - begin`). States are 1=Idle, 2=Exec, 3=Wait, 4=Stall. We align every wave
    on a single global cycle span [t0, t1] = [min begin, max end] and downsample each
    timeline onto a fixed grid of `_WAVE_NB` buckets (dominant state per bucket), then
    run-length encode. This bounds the embedded payload regardless of wave count or
    dispatch length (~19 KB for ~70 waves) while staying pixel-faithful to a fixed-width
    canvas. Waves are sorted by (se, simd, slot, wave-id) so lanes group by SIMD.

    Returns {t0, t1, nb, states, waves:[{lab, cu, simd, slot, wv, begin, end,
             rle:[[state,count],...]}]} or None. Bucket state 0 means the wave was not
    resident there (drawn as background)."""
    import glob
    import re
    if not dispatch_dir:
        return None
    raw = []
    for p in glob.glob(os.path.join(dispatch_dir, "se*_sm*_sl*_wv*.json")):
        m = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)", os.path.basename(p))
        if not m:
            continue
        try:
            with open(p) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        w = doc.get("wave") or {}
        tl = w.get("timeline") or []
        begin, end = w.get("begin"), w.get("end")
        if not tl or begin is None or end is None:
            continue
        se, sm, sl, wv = (int(m.group(1)), int(m.group(2)),
                          int(m.group(3)), int(m.group(4)))
        raw.append((se, sm, int(w.get("cu", 0)), sl, wv, int(begin), int(end), tl))
    if not raw:
        return None
    t0 = min(r[5] for r in raw)
    t1 = max(r[6] for r in raw)
    span = max(1, t1 - t0)
    bw = span / float(_WAVE_NB)

    def _rle(begin, tl):
        perb = {}
        cur = begin
        for seg in tl:
            if not isinstance(seg, list) or len(seg) < 2:
                continue
            st, clen = seg[0], seg[1]
            s, e = cur, cur + clen
            cur = e
            if clen <= 0:
                continue
            b0 = int((s - t0) / bw)
            b1 = int((e - 1 - t0) / bw)
            if b1 < 0:
                continue
            b0 = max(0, b0)
            b1 = min(_WAVE_NB - 1, b1)
            for b in range(b0, b1 + 1):
                bs = max(s, t0 + b * bw)
                be = min(e, t0 + (b + 1) * bw)
                ov = be - bs
                if ov <= 0:
                    continue
                dd = perb.setdefault(b, {})
                dd[st] = dd.get(st, 0) + ov
        arr = [0] * _WAVE_NB
        for b, dd in perb.items():
            arr[b] = max(dd, key=dd.get)
        rle = []
        for v in arr:
            if rle and rle[-1][0] == v:
                rle[-1][1] += 1
            else:
                rle.append([v, 1])
        return rle

    waves = []
    for se, sm, cu, sl, wv, begin, end, tl in sorted(
            raw, key=lambda r: (r[0], r[1], r[3], r[4])):
        waves.append({"lab": "se%d sm%d sl%d wv%d" % (se, sm, sl, wv),
                      "se": se, "cu": cu, "simd": sm, "slot": sl, "wv": wv,
                      "begin": begin, "end": end, "rle": _rle(begin, tl)})
    return {"t0": t0, "t1": t1, "nb": _WAVE_NB,
            "states": ["", "Idle", "Exec", "Wait", "Stall"], "waves": waves}


def _demangle_short(sym):
    """Extract a readable short name from an Itanium-mangled kernel symbol as it
    appears in occupancy.json's `dispatches` map (e.g.
    `_ZL22mul_mat_vec_q_wvsplitkIL9ggml_type12E...` -> `mul_mat_vec_q_wvsplitk[Q4_K]`).
    Placeholder entries like `0 / 0x0` or a raw address are not kernels -> None.
    No c++filt dependency: parse the length-prefixed name directly and, when the
    signature encodes a `(ggml_type)N` first template arg, append the quant tag so
    labels line up with family_of() used everywhere else."""
    if not sym or not isinstance(sym, str):
        return None
    if sym[0].isdigit():                 # "0 / 0x0", "0 / 0x76e4..." placeholders
        return None
    m = re.match(r"_Z[NL]?(\d+)(.*)", sym)
    if not m:
        return sym
    n = int(m.group(1))
    name = m.group(2)[:n]
    if not name:
        return sym
    g = re.search(r"9ggml_type(\d+)", sym)
    if g:
        t = int(g.group(1))
        name += "[" + _GGML_TYPES.get(t, "type%d" % t) + "]"
    return name


def load_att_occupancy(dispatch_dir):
    """Reconstruct rocprof-compute-viewer's Global View from one decoded ATT
    dispatch dir's `occupancy.json`. Unlike the per-wave `se*_wv*.json` files (which
    exist only for the single thread-traced SIMD -> at most 64 lanes = 1 WGP), the
    occupancy table samples wave scheduling across EVERY CU the trace observed, so it
    is the source of the "more than 64 slots" global waterfall.

    Schema: `occupancy_fields` names 11 columns; key "0" is the event table. Each row
    is a wave alloc/free event: a lane is (cu, simd, wave_id); `start`=1 opens an
    occupied interval at `time`, `start`=0 closes it; `kernel_id` indexes the
    `dispatches` name map so each interval is colored by which kernel held the slot.

    We reconstruct per-lane RAW cycle intervals (no bucketing) so the client can
    render them directly on a cycle axis -- gaps between successive waves stay exact
    at any zoom level, and each interval is one wave residency (colored by run order).
    Returns {t0, t1, kernels:{shifted_id: name_or_None}, lanes:[{cu,simd,wv,
    iv:[[start_rel, end_rel, shifted_id], ...]}]} or None. start_rel/end_rel are
    cycles relative to t0. Real kernel ids are stored shifted by +1 so kernel_id 0
    (a valid placeholder) does not collide with any background sentinel."""
    if not dispatch_dir:
        return None
    path = os.path.join(dispatch_dir, "occupancy.json")
    try:
        with open(path) as fh:
            occ = json.load(fh)
    except (OSError, ValueError):
        return None
    fields = occ.get("occupancy_fields") or []
    rows = occ.get("0") or []
    if not fields or not rows:
        return None
    idx = {f: i for i, f in enumerate(fields)}
    need = ("time", "cu", "simd", "wave_id", "start", "kernel_id")
    if any(k not in idx for k in need):
        return None
    ti, ci, si, wi, sti, ki = (idx["time"], idx["cu"], idx["simd"],
                               idx["wave_id"], idx["start"], idx["kernel_id"])
    disp = occ.get("dispatches") or {}
    # shifted-id name map: real kernel_id N -> slot N+1; value None for placeholders.
    kernels = {}
    for k, sym in disp.items():
        try:
            kid = int(k)
        except (TypeError, ValueError):
            continue
        kernels[str(kid + 1)] = _demangle_short(sym)

    from collections import defaultdict
    evs = defaultdict(list)
    for r in rows:
        try:
            evs[(int(r[ci]), int(r[si]), int(r[wi]))].append(
                (int(r[ti]), int(r[sti]), int(r[ki])))
        except (TypeError, ValueError, IndexError):
            continue
    if not evs:
        return None
    t0 = min(r[ti] for r in rows)
    t1 = max(r[ti] for r in rows)

    def _intervals(lane_evs):
        """Reconstruct [start_rel, end_rel, shifted_kid] from alloc/free events."""
        out = []
        open_t = open_k = None
        for t, s, k in sorted(lane_evs):
            if s == 1:
                open_t, open_k = t, k
            elif open_t is not None:
                if t > open_t:
                    out.append([open_t - t0, t - t0, open_k + 1])
                open_t = None
        return out

    lanes = []
    for (cu, simd, wv) in sorted(evs):
        iv = _intervals(evs[(cu, simd, wv)])
        if not iv:
            continue                              # lane never resident in window
        lanes.append({"cu": cu, "simd": simd, "wv": wv, "iv": iv})
    if not lanes:
        return None
    return {"t0": t0, "t1": t1, "kernels": kernels, "lanes": lanes}


def load_att_code(att_dir):
    """Parse the full per-instruction ISA disassembly from decoded ATT
    `ui_output_*_dispatch_*/code.json` files, per kernel FAMILY, for the
    single-kernel debug view. Unlike load_att_stats (which reads the pre-aggregated
    top-N stats CSV), this keeps the COMPLETE program-order instruction listing with
    per-PC Vaddr/Hit/Latency/Stall/Idle.

    code.json `code` rows are 10-tuples:
    [ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle].
    Function-block header rows are the ones whose ISA (col 0) is a `;`-prefixed
    symbol comment; their Source (col 3) is the demangled kernel signature, which
    family_of() maps onto the same family slices as load_att_stats. The instruction
    rows that follow are not `;`-prefixed; when the device code object was built with
    DWARF line tables (`-gline-tables-only`/`-g`) their Source column carries the
    decoded inline source chain (e.g. `hip_runtime.h:272 -> mmvq.cu:1034`), otherwise
    it is blank. (LineNumber (col 2) is only an instruction ordinal, not a source
    line, and sqtt_funcmap stays empty even with line info -- neither is the gate.)

    To bound HTML payload size, only ONE representative dispatch per family is kept:
    the one with the most instruction rows that recorded a hit (the richest profile).

    Returns {family: {sym, n_disp, stall, lat, idle,
                      rows: [{a(vaddr), isa, hit, lat, st, idle, src}], has_src,
                      src_files}}.
    Each row's `src` is the deepest real source location (`basename:line`) resolved
    from the Source chain, or "" when no line info is present; has_src is True when
    any kept instruction row resolved a source location (the traced code object had
    DWARF line tables). src_files maps each referenced file's basename to its full
    text (as a list of lines), read at generation time, so the debug view can show
    ISA side-by-side with source. Only the basename is embedded (never the absolute
    build path); files that are missing/unreadable/oversized are simply omitted."""
    import glob
    best = {}                    # fam -> (n_hit_rows, dispatch_dict)
    ndisp = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(att_dir, "**",
                                              "ui_output_*_dispatch_*",
                                              "code.json"),
                                 recursive=True)):
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        code = doc.get("code") or []
        fam_data = {}
        cur = cur_sym = None
        for i, r in enumerate(code):
            if not isinstance(r, list) or len(r) < 10:
                continue
            isa = r[0].strip() if isinstance(r[0], str) else ""
            col3 = r[3].strip() if isinstance(r[3], str) else ""
            if isa.startswith(";"):
                # function-header row: col3 is the demangled kernel signature.
                if col3:
                    cur, cur_sym = family_of(col3), col3
                continue
            if not cur or not isa:
                continue
            # basename:line for display; full path (kept per-dispatch only) to read
            # the source file at generation time -- the path never reaches the HTML.
            spath, sline = _att_src_split(col3)
            src = (os.path.basename(spath) + ":" + sline) if spath else ""

            def _i(x):
                try:
                    return int(x or 0)
                except (ValueError, TypeError):
                    return 0
            vaddr, hit, lat, st, idle = (_i(r[5]), _i(r[6]), _i(r[7]),
                                         _i(r[8]), _i(r[9]))
            d = fam_data.setdefault(cur, {"sym": cur_sym, "rows": [], "ci": [],
                                          "stall": 0, "lat": 0, "idle": 0,
                                          "nhit": 0, "has_src": False,
                                          "srcpaths": set(), "_dir": ""})
            d["_dir"] = os.path.dirname(path)
            d["ci"].append(i)
            d["rows"].append({"a": vaddr, "isa": isa, "hit": hit,
                              "lat": lat, "st": st, "idle": idle, "src": src})
            d["stall"] += st
            d["lat"] += lat
            d["idle"] += idle
            if hit:
                d["nhit"] += 1
            if spath:
                d["has_src"] = True
                d["srcpaths"].add(spath)
        for fam, d in fam_data.items():
            if not d["nhit"]:
                continue                     # dispatch empty/cut off for this family
            ndisp[fam] += 1
            if fam not in best or d["nhit"] > best[fam][0]:
                best[fam] = (d["nhit"], d)
    out = {}
    max_src_bytes = 512 * 1024
    _occ_cache = {}    # dispatch dir -> occupancy (shared object; dedup at emit)
    for fam, (_n, d) in best.items():
        # Read each referenced source file once (keyed by basename, path discarded).
        src_files = {}
        for p in sorted(d.get("srcpaths") or ()):
            base = os.path.basename(p)
            if base in src_files:
                continue
            try:
                if os.path.getsize(p) > max_src_bytes:
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    src_files[base] = fh.read().split("\n")
            except OSError:
                continue
        # Executed-order stream (for the debug view's Step mode): remap the picked
        # dispatch's representative wave onto embedded row positions.
        ci2row = {ci: pos for pos, ci in enumerate(d.get("ci") or [])}
        exec_stream = _load_att_wave(d.get("_dir") or "", ci2row)
        waves = load_att_waves(d.get("_dir") or "")
        ddir = d.get("_dir") or ""
        if ddir not in _occ_cache:
            _occ_cache[ddir] = load_att_occupancy(ddir)
        occ = _occ_cache[ddir]
        out[fam] = {"sym": d["sym"], "n_disp": ndisp[fam], "stall": d["stall"],
                    "lat": d["lat"], "idle": d["idle"], "rows": d["rows"],
                    "has_src": d.get("has_src", False), "src_files": src_files,
                    "exec": exec_stream, "waves": waves, "occ": occ}
    return out


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
            ent = _seq_entry(t, layer, role)
            # Fused SwiGLU: the single ffn_gate dispatch streams BOTH the gate and
            # up weights from DRAM, so its true footprint is gate+up. Fold up's
            # bytes in, else the theoretical denominator is ~2x too small (the
            # dispatch would look like it over-fetches ~2x when it does not).
            if role == "ffn_gate" and drop_ffn_up:
                up = roles.get("ffn_up")
                if up is not None and up["gt"] in _MATVEC_TYPES:
                    ent["bytes"] += up["bytes"]
                    ent["fused"] = "gate+up"
            seq.append(ent)
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


def add_common_args(ap):
    """Input + rendering flags shared by the generator (main) and serve.py."""
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
    ap.add_argument("--att-dir",
                    help="directory of DECODED rocprofv3 --att output (the "
                         "stats_ui_output_*_dispatch_*.csv files, e.g. produced by "
                         "collect-att.sh): folds per-instruction thread-trace stall "
                         "cycles into the selected-kernel detail panel (total stall, "
                         "dominant stall instruction, top stalling instructions, and "
                         "stall grouped by opcode). ATT is a microscope -- one SIMD, "
                         "a few dispatches -- so it only enriches families it traced.")
    ap.add_argument("--gguf",
                    help="GGUF model file (optional): order-maps each mul_mat_vec "
                         "decode dispatch to its GGUF weight tensor by execution "
                         "order (join on launched N == weight ne[1]), attaching the "
                         "weight name, true [K x N] shape, and packed footprint to "
                         "the detail panel -- so launch-grid vs true shape reveals "
                         "any output-row/reduction padding waste and the packed "
                         "weight bytes give a theoretical-vs-measured over-fetch ratio")
    ap.add_argument("--build-dir",
                    help="llama.cpp build dir (optional): baked into the "
                         "copy-ready 'Trace this kernel with ATT' command in the "
                         "detail panel as a full path, so the command runs as-is "
                         "with no env vars to fill in. Falls back to a "
                         "/path/to/... placeholder if omitted.")
    ap.add_argument("--arch", default=DEFAULT_ARCH,
                    help="GPU arch, selects peak DRAM BW for the roofline "
                         "(default %s = %g GB/s)"
                         % (DEFAULT_ARCH, PEAK_BW_GBS_BY_ARCH[DEFAULT_ARCH]))
    ap.add_argument("--peak-bw", type=float,
                    help="override peak DRAM bandwidth in GB/s (else from --arch)")
    ap.add_argument("--clean-tps-file",
                    help="path to collect.sh's clean_tps.txt (the untraced "
                         "llama-bench run): parses the decode (tg) row's t/s and "
                         "shows it in the header as the honest throughput, since "
                         "rocprofv3 perturbs the traced runs' timing. Silently "
                         "ignored if the file is missing or unparseable.")
    ap.add_argument("--out", help="output HTML path (required for the generator)")
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
    ap.add_argument("--kv-context-tokens", type=int, default=-1,
                    help="context length (tokens) to size the KV-cache traffic in "
                         "the 'eff token BW%%' footer metric. KV bytes/token = "
                         "n_attn_layers * head_count_kv * (key_len+value_len) * 2 "
                         "* n_ctx. Default -1 infers n_ctx from the clean-tps "
                         "'tgNN' test name; 0 excludes KV (weights only).")
    ap.add_argument("--title", default="rocprof unified viewer (gfx1151)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    if not args.out:
        ap.error("--out is required")
    write_overlay(args)


def build_payload(args):
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
    att_by_fam = load_att_stats(args.att_dir) if args.att_dir else {}
    att_code_by_fam = load_att_code(args.att_dir) if args.att_dir else {}
    # Occupancy is dispatch-wide: many families share the SAME occ object (identical
    # 640-lane table). Pool distinct occ objects into att_occ_pool and replace each
    # family's inline occ with an index (occ_ref) so the HTML embeds it ONCE, not 18x.
    att_occ_pool = []
    _occ_seen = {}
    for _fam, _c in att_code_by_fam.items():
        _o = _c.pop("occ", None)
        if _o is None:
            _c["occ_ref"] = -1
            continue
        _key = id(_o)
        if _key not in _occ_seen:
            _occ_seen[_key] = len(att_occ_pool)
            att_occ_pool.append(_o)
        _c["occ_ref"] = _occ_seen[_key]
    clean_tps = parse_clean_tps(args.clean_tps_file) if args.clean_tps_file else None

    # Optional GGUF order-map: build the expected per-token matvec tensor sequence
    # in decode execution order. llama.cpp fuses the SwiGLU gate+up into one
    # dispatch at decode, so try both dropping and keeping ffn_up and pick whichever
    # candidate's N-sequence best matches the actual matvec dispatches in one token.
    expected_seq = []
    gguf_meta = {}
    if args.gguf:
        gguf_tensors, gguf_meta = load_gguf_tensors(args.gguf)
        # Reference token = the matvec N-stream between the first two boundaries in
        # the baked span (a full steady-state token).
        ref = [n for (s, e, nm, n, _nb) in evs[bounds[args.skip_tokens]:
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

    # KV-cache DRAM traffic per decode step, for the "eff token BW%" roofline: a
    # decode step re-reads the FULL K/V cache accumulated over the context. Sizes
    # analytically from GGUF meta -- KV bytes = n_attn_layers * head_count_kv *
    # (key_len + value_len) * 2 (f16) * n_ctx. n_attn_layers is the count of GGUF
    # blocks that own an attn_k projection (the hybrid GDN model has attention on
    # only a subset of blocks). Negligible vs weights at short context; the point
    # is it grows linearly with n_ctx and eventually rivals weight traffic.
    kv_bytes_per_tok = 0
    kv_ctx = 0
    if expected_seq:
        arch = gguf_meta.get("general.architecture", "")
        gk = lambda suffix, d=0: gguf_meta.get("%s.%s" % (arch, suffix), d)
        head_kv = gk("attention.head_count_kv", 0)
        key_len = gk("attention.key_length", 0)
        val_len = gk("attention.value_length", 0)
        n_attn = len({e["L"] for e in expected_seq if e["role"] == "attn_k"})
        kv_ctx = args.kv_context_tokens
        if kv_ctx < 0:
            m = re.search(r"tg(\d+)", (clean_tps or {}).get("test", "") or "")
            kv_ctx = int(m.group(1)) if m else 0
        if head_kv and key_len and val_len and n_attn and kv_ctx > 0:
            kv_bytes_per_tok = n_attn * head_kv * (key_len + val_len) * 2 * kv_ctx

    # Per-weight measured DRAM bytes: order-map the FETCH run itself so weights
    # sharing an N (ffn_down vs attn_output at N=2560) get exact measurements
    # instead of a shape-blended (family, N) average. Empty -> fall back to blend.
    fetch_by_name = (load_fetch_bytes_mapped(args.fetch_csv, expected_seq)
                     if (args.fetch_csv and expected_seq) else {})

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
    ti_ctr = 0
    mv_total = mv_mapped = 0
    for idx, (s, e, name, ncol, nblk) in enumerate(baked):
        if idx in tok_boundary_idx:
            ei = 0
            ti_ctr = 0
        fam = family_of(name)
        finfo = fams.get(fam)
        stall = finfo["stall"] if finfo else "unknown"
        if "copy" in fam.lower() or "cpy" in fam.lower():
            stall = "copy"
        dur = e - s
        busy_ns += dur
        fam_busy[fam] += dur
        fam_count[fam] += 1
        sl = {"s": s - t0, "e": e - t0, "fam": fam, "stall": stall,
              "blocks": nblk, "ti": ti_ctr}
        ti_ctr += 1
        if expected_seq and "mul_mat_vec" in name and ncol:
            mv_total += 1
            ent = expected_seq[ei] if ei < len(expected_seq) else None
            ei += 1
            if ent is not None:
                true_n = ent["N"]
                k = ent["K"]
                packed = ent["bytes"]
                # Prefer per-weight order-mapped bytes (over-fetch-honest even for
                # weights sharing an N); fall back to the (family, N) blend, then
                # the family mean.
                mexact = ent["nm"] in fetch_by_name
                measured = (fetch_by_name.get(ent["nm"])
                            or fetch_bytes_n.get((fam, ncol))
                            or fetch_bytes.get(fam, 0))
                sl["map"] = {
                    "nm": ent["nm"], "role": ent["role"], "L": ent["L"],
                    "q": ent["q"], "K": k, "trueN": true_n, "launchN": ncol,
                    # Output-row padding: launched rows beyond the true weight rows.
                    "padN": max(0, ncol - true_n),
                    # Reduction (K) padding to the quant block (256 for K-quants).
                    "padK": (((k + 255) // 256) * 256 - k) if k else 0,
                    "packed": packed,
                    "fused": ent.get("fused", ""),
                    "measured": round(measured) if measured else 0,
                    # True when `measured` is this weight's own order-mapped bytes
                    # (over-fetch-honest); False when it fell back to the (fam, N) blend.
                    "mexact": mexact,
                    # Over-fetch: measured DRAM bytes / theoretical packed footprint.
                    "overfetch": (round(measured / packed, 2)
                                  if (measured and packed) else 0),
                    # Effective (useful-work) bandwidth: the THEORETICAL bytes this
                    # matvec must move / its exact kernel time. Immune to over-fetch
                    # by construction (numerator is the algorithmic minimum, not
                    # measured traffic), and uses exact per-dispatch duration (no
                    # separate-run / family blend), so it is the honest roofline
                    # number: a kernel that over-fetches 100x keeps DRAM busy but
                    # its effective BW stays low. 1 byte/ns == 1 GB/s.
                    "effbw": round(packed / dur, 1) if dur else 0,
                    "effbw_pct": (round(packed / dur / peak_bw * 100, 1)
                                  if dur else 0),
                    "nmatch": (true_n == ncol),
                }
                if true_n == ncol:
                    mv_mapped += 1
        gpu_slices.append(sl)

    map_stats = ({"total": mv_total, "mapped": mv_mapped,
                  "pct": round(100.0 * mv_mapped / mv_total, 1) if mv_total else 0,
                  "seq_len": len(expected_seq)}
                 if expected_seq else None)

    # Per-kernel steady-state stats: decode tokens are structurally identical, so
    # the Nth kernel of every token is the same dispatch. Aggregate each within-token
    # position (ti) across ALL post-warmup tokens in the full stream (not just the
    # baked view) so the selected-kernel panel can show a stable mean +/- spread
    # instead of one noisy single-dispatch duration (per-token jitter includes the
    # once-per-token host-serialized GDN edge, launch bubbles, interrupt latency).
    # Keyed "ti|family" so a rare token with a different kernel count self-segregates
    # rather than blending mismatched positions. Durations are ns (JS renders us).
    kstats = {}
    kstats_ntok = max(0, len(bounds) - 1 - args.skip_tokens)
    if kstats_ntok > 0:
        agg = defaultdict(list)
        for k in range(args.skip_tokens, len(bounds) - 1):
            for ti, (s, e, nm, _n, _nb) in enumerate(evs[bounds[k]:bounds[k + 1]]):
                agg[(ti, family_of(nm))].append(e - s)
        for (ti, fam), durs in agg.items():
            cnt = len(durs)
            mean = sum(durs) / cnt
            std = (sum((d - mean) ** 2 for d in durs) / cnt) ** 0.5 if cnt > 1 else 0.0
            kstats["%d|%s" % (ti, fam)] = {
                "n": cnt, "mean": round(mean, 1), "std": round(std, 1),
                "min": round(min(durs), 1), "max": round(max(durs), 1),
            }

    # Layer swim-lane: segment the baked GPU slices into per-decode-layer spans
    # using the order-map's true GGUF block index (map.L). Leading input-norm /
    # conv slices that precede a layer's first matvec are folded into that layer
    # (backward fill within each token); each block's kind (GDN vs full-attention)
    # is inferred from tensor presence (ssm_* -> gated-delta-net).
    layers = []
    if expected_seq:
        block_kind = {}
        _roles = defaultdict(set)
        for t in gguf_tensors:
            m = re.match(r"blk\.(\d+)\.(.*)\.weight$", t["name"])
            if m:
                _roles[int(m.group(1))].add(m.group(2))
        for L, rs in _roles.items():
            block_kind[L] = "GDN" if any(r.startswith("ssm") for r in rs) else "ATTN"
        starts = sorted(tok_boundary_idx)
        n = len(gpu_slices)
        lay_L = [None] * n
        for wi in range(len(starts) - 1):
            st = starts[wi]
            en = min(starts[wi + 1], n)
            # backward fill: leading norms take the next matvec's layer index.
            cur = None
            for i in range(en - 1, st - 1, -1):
                mp = gpu_slices[i].get("map")
                if mp is not None:
                    cur = mp["L"]
                lay_L[i] = cur
            # forward fill: trailing slices past the last matvec keep the last layer.
            fill = None
            for i in range(st, en):
                if lay_L[i] is None:
                    lay_L[i] = fill
                else:
                    fill = lay_L[i]
            # coalesce consecutive equal-layer runs into one segment.
            j = st
            while j < en:
                L = lay_L[j]
                k = j
                while k < en and lay_L[k] == L:
                    k += 1
                kind = "head" if L == -1 else block_kind.get(L, "?")
                name = "head" if L == -1 else ("L%d %s" % (L, kind))
                layers.append({"s": gpu_slices[j]["s"], "e": gpu_slices[k - 1]["e"],
                               "kind": kind, "name": name})
                j = k

    # Phase sub-lane: one level below the layer lane -- group each layer's kernels
    # into functional sub-blocks (input-norm, q/k/v projections, ssm conv, l2-norm,
    # gated-delta-net / flash-attn, out proj, ffn, ...). Matvec slices are grouped by
    # their order-mapped weight role; the rest by kernel family. Phases are coalesced
    # within a (token, layer) so a sub-block never spans a layer boundary -- these
    # boundaries are exactly the fusion-candidate edges.
    phases = []
    if expected_seq:
        def _phase_of(sl):
            mp = sl.get("map")
            if mp is not None:
                role = mp.get("role", "")
                if role in ("attn_qkv", "attn_q", "attn_k", "attn_v"):
                    return "qkv"
                if role in ("ssm_in", "ssm_alpha", "ssm_beta"):
                    return "ssm_in"
                if role == "attn_gate":
                    return "gate"
                if role in ("ssm_out", "attn_output"):
                    return "o_proj"
                if role.startswith("ffn"):
                    return "ffn"
                if role == "output":
                    return "head"
            f = sl["fam"].lower()
            if "l2_norm" in f: return "l2norm"
            if "norm" in f: return "norm"
            if "quantize" in f: return "quant"
            if "conv" in f: return "conv"
            if "gated_delta" in f: return "gdn"
            if "flash_attn" in f or "fattn" in f: return "attn"
            if "rope" in f: return "rope"
            if "get_rows" in f: return "gather"
            if "set_rows" in f: return "scatter"
            if "bin_bcast" in f: return "binop"
            if "concat" in f: return "concat"
            if "unary" in f: return "act"
            if "copy" in f or "cpy" in f: return "copy"
            if "mul_mat_vec" in f: return "matvec"
            if "add" in f: return "add"
            return (f.split("_", 1)[0] or "op")[:6]
        ph = [_phase_of(sl) for sl in gpu_slices]
        for wi in range(len(starts) - 1):
            st = starts[wi]
            en = min(starts[wi + 1], n)
            j = st
            while j < en:
                key = (lay_L[j], ph[j])
                k = j
                while k < en and (lay_L[k], ph[k]) == key:
                    k += 1
                phases.append({"s": gpu_slices[j]["s"], "e": gpu_slices[k - 1]["e"],
                               "name": ph[j]})
                j = k

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
            # Tiling: static LDS/block (dynamic extern-shared not reported by the
            # profiler), threads/block, and per-family mean block count.
            "lds_static": finfo.get("regs", {}).get("lds", 0),
            "wg": finfo.get("wg", 0),
            "blocks": finfo.get("blocks", 0),
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
                          "lds_static": summary_i["lds_static"],
                          "wg": summary_i["wg"],
                          "blocks": summary_i["blocks"],
                          "wave": summary_i["wave"],
                          "kb_disp": summary_i["kb_disp"],
                          "mb_tok": summary_i["mb_tok"],
                          "bw_gbs": summary_i["bw_gbs"], "bw_pct": summary_i["bw_pct"],
                          "loadw": summary_i["loadw"]}
                    for summary_i, fam in ((s, s["fam"]) for s in summary)}

    # Bake a copy-ready ATT command for the detail panel using FULL PATHS and no
    # env vars, so it runs as-is. collect-att.sh lives next to this script; the
    # regen command reconstructs the exact flags used here (abspath'd) minus
    # --att-dir/--out, which the JS appends per selected kernel.
    _self_dir = os.path.dirname(os.path.abspath(__file__))
    _out = args.out or "overlay.html"
    regen_parts = ["--kernel-csv " + os.path.abspath(args.kernel_csv)]
    if args.hip_csv:
        regen_parts.append("--hip-csv " + os.path.abspath(args.hip_csv))
    if args.pmc_csv:
        regen_parts.append("--pmc-csv " + os.path.abspath(args.pmc_csv))
    if args.fetch_csv:
        regen_parts.append("--fetch-csv " + os.path.abspath(args.fetch_csv))
    if args.loadwidth_json:
        regen_parts.append("--loadwidth-json " + os.path.abspath(args.loadwidth_json))
    if args.gguf:
        regen_parts.append("--gguf " + os.path.abspath(args.gguf))
    if args.arch != DEFAULT_ARCH:
        regen_parts.append("--arch " + args.arch)
    if args.peak_bw:
        regen_parts.append("--peak-bw %g" % args.peak_bw)
    if args.tokens != 2:
        regen_parts.append("--tokens %d" % args.tokens)
    if args.skip_tokens != 30:
        regen_parts.append("--skip-tokens %d" % args.skip_tokens)
    if args.context_tokens:
        regen_parts.append("--context-tokens %d" % args.context_tokens)
    if args.gap_threshold_us != 150.0:
        regen_parts.append("--gap-threshold-us %g" % args.gap_threshold_us)
    att_cmd = {
        "script": os.path.join(_self_dir, "collect-att.sh"),
        "build_dir": os.path.abspath(args.build_dir) if args.build_dir
                     else "/path/to/llamacpp-build",
        "model": os.path.abspath(args.gguf) if args.gguf
                 else "/path/to/model.gguf",
        "out_base": os.path.dirname(os.path.abspath(_out)),
        "viewer": os.path.abspath(__file__),
        "regen_flags": " \\\n  ".join(regen_parts),
        "out_html": os.path.abspath(_out),
    }

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
        "att_by_fam": att_by_fam,
        "has_att": bool(att_by_fam),
        "att_code_by_fam": att_code_by_fam,
        "att_occ_pool": att_occ_pool,
        "has_att_code": bool(att_code_by_fam),
        # RDNA3.5 ISA one-line opcode glossary (mnemonic -> description), embedded
        # only when the debug view exists, so the view can show a hover tooltip
        # explaining each raw instruction. Keyed on the lowercased first token.
        "isa_gloss": ISA_GLOSSARY if att_code_by_fam else {},
        # Special-register / wait-counter glossary (operand token -> description),
        # so hovering vmcnt/lgkmcnt/SCC/EXEC/VCC/M0 etc. in an ISA line explains it.
        "reg_gloss": REG_GLOSSARY if att_code_by_fam else {},
        "att_cmd": att_cmd,
        # Live-tracing mode: false for the static export; serve.py flips this true
        # so the client shows a "Trace now" button instead of the copy command.
        "att_server": False,
        "has_map": bool(expected_seq),
        "map_stats": map_stats,
        "kv_bytes_per_tok": kv_bytes_per_tok,
        "kv_ctx": kv_ctx,
        "layers": layers,
        "has_layers": bool(layers),
        "phases": phases,
        "has_phases": bool(phases),
        "kstats": kstats,
        "has_kstats": bool(kstats),
        "kstats_ntok": kstats_ntok,
        "clean_tps": clean_tps,
        "peak_bw_gbs": peak_bw,
        # gfx1151 (RDNA3.5) scheduling constants for the modeled occupancy row.
        # 20 WGP; each WGP = 2 CU = 4 SIMD32; each SIMD32 holds 16 wave32 slots
        # and a 1536-entry VGPR file (wave32); 128 KB LDS shared per WGP.
        "hw": {"wgp": 20, "simd_per_wgp": 4, "slots_per_simd": 16,
               "vgpr_per_simd": 1536, "lds_per_wgp": 131072},
        # RDNA 3.5 WGP diagram (base64 PNG) shown by the "RDNA 3.5 HW" toolbar
        # button; embedded so the overlay stays a single self-contained file.
        "hw_diagram": load_hw_diagram(),
    }
    return payload


def write_overlay(args):
    payload = build_payload(args)
    html = render_html(payload)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  baked {payload['n_tokens_baked']} tokens ({len(payload['gpu'])} GPU "
          f"slices, {len(payload['cpu'])} CPU calls); viewport shows "
          f"{payload['tokens_view']} tokens.", file=sys.stderr)
    print(f"  window busy {payload['busy_ns']/1e6:.3f} ms / span "
          f"{payload['span_ns']/1e6:.3f} ms "
          f"({payload['busy_ns']/payload['span_ns']*100:.1f}% GPU-busy)",
          file=sys.stderr)
    ct = payload['clean_tps']
    if ct:
        sd = "" if ct['sd'] is None else f" +/- {ct['sd']:.2f}"
        print(f"  clean {ct['test']}: {ct['tps']:.2f}{sd} tok/s "
              f"(untraced baseline)", file=sys.stderr)
    ms = payload['map_stats']
    if ms:
        print(f"  gguf order-map: {ms['mapped']}/{ms['total']} "
              f"matvec dispatches N-matched to weights "
              f"({ms['pct']:.1f}%); expected seq len "
              f"{ms['seq_len']}/token", file=sys.stderr)
    ab = payload['att_by_fam']
    if ab:
        fams_str = ", ".join("%s (%d disp)" % (f, a["n_disp"])
                             for f, a in sorted(ab.items(),
                                                key=lambda kv: -kv[1]["stall"]))
        print(f"  att thread-trace folded into {len(ab)} "
              f"famil{'y' if len(ab)==1 else 'ies'}: {fams_str}",
              file=sys.stderr)


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
  #detail tr.shrow{cursor:pointer;}
  #detail tr.shrow:hover{background:#1b2130;}
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
      <select id="findWhat" title="what to find">
        <option value="maxgap">largest intra-token gap</option>
        <option value="mineffbw">lowest eff-BW matvec (mmvq/mmq)</option>
      </select>
      <button id="findGo" title="find (click again for next-largest)">find next</button>
      <button id="findPrev" title="previous (larger) match">find prev</button>
      <button id="hwbtn" title="RDNA 3.5 WGP hardware reference" style="display:none">RDNA 3.5 HW</button>
      <span id="findmsg" class="sub"></span>
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
      <h2>Per-kernel-family/token (baked span)</h2>
      <table id="tbl"><thead><tr>
        <th>family</th><th>cnt/tok</th><th>time%</th><th>stall</th>
      </tr></thead><tbody></tbody><tfoot></tfoot></table>
      <div class="sub" id="bwnote" style="margin-top:8px"></div>
    </div>
  </div>
</div>
<div id="hover"></div>
<div id="hwmodal" style="display:none;position:fixed;inset:0;z-index:9999;
  background:rgba(0,0,0,.72);align-items:center;justify-content:center;padding:24px">
  <div style="position:relative;max-width:96vw;max-height:92vh;background:#0d1017;
    border:1px solid #2a2f3a;border-radius:6px;padding:12px 12px 8px;overflow:auto">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <b style="color:#c8d0da">RDNA 3.5 WGP hardware reference</b>
      <button id="hwclose">close &times;</button>
    </div>
    <img id="hwimg" alt="RDNA 3.5 WGP: VGPR file, LDS banks, wave slots, gfx1151/gfx1150 constants"
      style="display:block;max-width:100%;height:auto;background:#fff;border-radius:3px">
    <div class="sub" style="margin-top:6px">The fusion-analysis panel models occupancy
      from these constants: 96 VGPR/wave = full 16-wave occupancy, 256 = scratch spill.</div>
  </div>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const CPU_H = 70, GPU_H = 70, PAD_T = 8, GAP = 26, AXIS_H = 22;
const LAYER_H = D.has_layers ? 20 : 0;      // per-decode-layer swim lane
const LGAP = D.has_layers ? 6 : 0;          // gap between GPU lane and layer lane
const PHASE_H = D.has_phases ? 16 : 0;      // functional sub-block lane (finer)
const PGAP = D.has_phases ? 3 : 0;
const H = PAD_T + CPU_H + GAP + GPU_H + LGAP + LAYER_H + PGAP + PHASE_H + AXIS_H;
const PHASE_COL = {norm:'#5a6b3d', l2norm:'#7a9b4d', quant:'#4a4f5e', qkv:'#3d6b8f',
  ssm_in:'#3d8f8a', conv:'#8f6b3d', gate:'#8f3d6b', gdn:'#6b3d8f', attn:'#8f3d3d',
  o_proj:'#3d5a8f', ffn:'#7a5a2d', rope:'#5a5a8f', gather:'#4a7a5a', scatter:'#4a7a6f',
  binop:'#6a6a4a', concat:'#5a7a7a', act:'#8a7a3d', copy:'#555a66', matvec:'#4a5a6a',
  add:'#666a55', head:'#777'};
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
// cycle counts: 199587 -> "199.6k", 1.2e6 -> "1.2M"
function fmtc(n){n=+n||0;return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':(''+n);}
function esc(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
// Copy text to clipboard with a file:// fallback (navigator.clipboard is often
// unavailable when the HTML is opened from disk, so fall back to a hidden
// textarea + execCommand). Flashes the sibling #attcopied on success.
function copyCmd(text){
  const done=()=>{const m=document.getElementById('attcopied');
    if(m){m.style.display='inline';setTimeout(()=>m.style.display='none',1500);}};
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done,()=>fallbackCopy(text,done));
  } else { fallbackCopy(text,done); }
}
function fallbackCopy(text,done){
  const ta=document.createElement('textarea');
  ta.value=text; ta.style.position='fixed'; ta.style.opacity='0';
  document.body.appendChild(ta); ta.select();
  try{ if(document.execCommand('copy')) done(); }catch(e){}
  document.body.removeChild(ta);
}

document.getElementById('title').textContent = D.title;
document.getElementById('sub').textContent =
  `baked ${D.n_tokens_baked} tokens | ${D.gpu.length} GPU slices | `+
  `${D.cpu.length} HIP calls | window GPU-busy ${fmtms(D.busy_ns)} / span `+
  `${fmtms(D.span_ns)}` + (D.has_pmc ? '' : ' | NO PMC (uncolored)') +
  (D.map_stats ? ` | GGUF map ${D.map_stats.mapped}/${D.map_stats.total} matvec `+
    `(${D.map_stats.pct}%)` : '') +
  (D.clean_tps ? ` | clean decode ${D.clean_tps.tps.toFixed(1)}`+
    (D.clean_tps.sd!=null ? ` +/- ${D.clean_tps.sd.toFixed(1)}` : '')+
    ` tok/s (untraced)` : '');
document.getElementById('cpunote').textContent = D.has_cpu ? '' :
  '(no --hip-csv supplied: CPU lane empty)';
document.getElementById('bwnote').textContent = D.has_bw ?
  ('time% = share of window GPU-busy time. Aggregate DRAM read/token and achieved BW '+
   '(all bytes / kernel time, vs '+D.peak_bw_gbs+' GB/s peak) are in the table footer. '+
   'Per-family roofline is in the click-through detail panel (real per-shape effective BW).')
  : '(no --fetch-csv supplied: achieved-bandwidth footer omitted)';

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

// Pick a crisp text color (dark or white) for legibility over a given box fill,
// by perceived luminance -- avoids a blurry outline while keeping contrast.
function textOn(bg){
  const m=/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(bg||'');
  if(!m) return '#fff';
  const r=parseInt(m[1],16), g=parseInt(m[2],16), b=parseInt(m[3],16);
  return (0.299*r+0.587*g+0.114*b) > 140 ? '#111' : '#fff';
}

// Draw a slice's name INSIDE its box, but only when the box is wide enough to
// hold the whole name (no clipping/ellipsis). Plain fillText (no outline) so it
// renders as crisply as the axis/token labels; inherits the box's globalAlpha.
// The label is pinned to the VISIBLE left edge of the box: when zoomed in so the
// box's real left edge is scrolled off-screen, the name sticks at the viewport
// edge (Perfetto-style) instead of vanishing with the off-screen edge. It is
// still never allowed to spill past the box's right edge.
function boxLabel(label, x, wpx, midY, bg){
  const W = cv.clientWidth;
  const vx0 = Math.max(x, 0), vx1 = Math.min(x + wpx, W);
  const vis = vx1 - vx0;                        // on-screen width of this box
  if (vis < 14) return;                         // too little visible to bother
  ctx.font = '10px sans-serif';
  const tw = ctx.measureText(label).width;
  if (tw + 8 > vis) return;                      // name would not fully fit in view
  let tx = Math.max(x, 0) + 4;                   // pin to the visible left edge
  tx = Math.min(tx, (x + wpx) - tw - 4);         // but keep it inside the box's right
  const prevBaseline = ctx.textBaseline;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = textOn(bg);
  ctx.fillText(label, tx, midY);
  ctx.textBaseline = prevBaseline;
}

function draw(){
  const w = cv.clientWidth;
  ctx.clearRect(0,0,w,H);
  rects = [];
  const cpuY = PAD_T, gpuY = PAD_T + CPU_H + GAP;
  const gpuBot = gpuY + GPU_H;
  // finer phase sub-lane sits directly under the GPU lane; the coarse GDN/ATTN
  // layer lane sits below it.
  const phaseY = gpuBot + LGAP;
  const layerY = D.has_phases ? (phaseY + PHASE_H + PGAP) : (gpuBot + LGAP);
  const axisTop = D.has_layers ? (layerY + LAYER_H)
                : D.has_phases ? (phaseY + PHASE_H) : gpuBot;

  // lane backgrounds
  ctx.fillStyle = '#161922';
  ctx.fillRect(0,cpuY,w,CPU_H); ctx.fillRect(0,gpuY,w,GPU_H);
  if (D.has_phases){ ctx.fillStyle = '#0f1118'; ctx.fillRect(0,phaseY,w,PHASE_H); }
  if (D.has_layers){ ctx.fillStyle = '#12141c'; ctx.fillRect(0,layerY,w,LAYER_H); }

  // token boundary markers + inter-token idle shading on GPU lane
  ctx.strokeStyle = '#3a4f6a'; ctx.lineWidth = 1;
  ctx.fillStyle = '#8fb0d8'; ctx.font = '10px sans-serif';
  for (let k=0;k<D.tok_starts.length;k++){
    const ts = D.tok_starts[k];
    if (ts < view0-1 || ts > view1+1) continue;
    const x = xOf(ts, w);
    ctx.beginPath(); ctx.moveTo(x,cpuY); ctx.lineTo(x,axisTop); ctx.stroke();
    ctx.fillText('tok '+k, x+3, axisTop+13);
  }

  // GPU slices
  for (const s of D.gpu){
    if (s.e < view0 || s.s > view1) continue;
    const x = xOf(s.s,w), x2 = xOf(s.e,w);
    const wpx = Math.max(1, x2-x);
    const isSel = selectedFam && s.fam===selectedFam;
    const inMulti = selectedSlices && selectedSlices.has(s);
    const dim = (selectedFam && !isSel) || (selectedSlices && !inMulti);
    ctx.globalAlpha = dim ? 0.25 : 1.0;
    const gcol = D.colors[s.stall] || D.colors.unknown;
    ctx.fillStyle = gcol;
    ctx.fillRect(x, gpuY+4, wpx, GPU_H-8);
    boxLabel(s.fam, x, wpx, gpuY + GPU_H/2, gcol);
    ctx.globalAlpha = 1.0;
    if(isSel){ ctx.strokeStyle='#ffffff'; ctx.lineWidth=1.5;
      ctx.strokeRect(x+0.5, gpuY+4.5, Math.max(1,wpx-1), GPU_H-9); ctx.lineWidth=1; }
    if(inMulti){ ctx.strokeStyle='#8fe388'; ctx.lineWidth=2;
      ctx.strokeRect(x+0.5, gpuY+3.5, Math.max(1,wpx-1), GPU_H-7); ctx.lineWidth=1; }
    if(selectedSlice && s===selectedSlice){
      ctx.strokeStyle='#ffffff'; ctx.lineWidth=2;
      ctx.strokeRect(x-0.5, gpuY+2.5, Math.max(2,wpx+1), GPU_H-5); ctx.lineWidth=1;
      ctx.fillStyle='#ffffff'; const cx=x+wpx/2;   // caret so thin slices stay findable
      ctx.beginPath(); ctx.moveTo(cx-4,gpuY-5); ctx.lineTo(cx+4,gpuY-5);
      ctx.lineTo(cx,gpuY+1); ctx.closePath(); ctx.fill();
    }
    rects.push({x:x,y:gpuY+4,w:wpx,h:GPU_H-8,type:'gpu',p:s});
  }
  // layer swim-lane: one colored segment per decode layer (GDN vs full-attn),
  // labeled with the true GGUF block index; hover shows the layer name + span.
  if (D.has_layers){
    for (const L of D.layers){
      if (L.e < view0 || L.s > view1) continue;
      const x = xOf(L.s,w), x2 = xOf(L.e,w);
      const wpx = Math.max(1, x2-x);
      const lcol = L.kind==='GDN' ? '#3d5a80' : L.kind==='ATTN' ? '#7a4f6d' : '#4a4a4a';
      ctx.fillStyle = lcol;
      ctx.fillRect(x, layerY+2, wpx, LAYER_H-4);
      ctx.strokeStyle = '#0b0d12'; ctx.lineWidth = 1;
      ctx.strokeRect(x+0.5, layerY+2.5, Math.max(1,wpx-1), LAYER_H-5);
      boxLabel(L.name, x, wpx, layerY + LAYER_H/2, lcol);
      rects.push({x:x,y:layerY+2,w:wpx,h:LAYER_H-4,type:'layer',p:L});
    }
  }
  // phase sub-lane: functional sub-blocks within each layer (finer than GDN/ATTN),
  // colored by phase; the boundaries mark where fusion would cross a functional edge.
  if (D.has_phases){
    for (const P of D.phases){
      if (P.e < view0 || P.s > view1) continue;
      const x = xOf(P.s,w), x2 = xOf(P.e,w);
      const wpx = Math.max(1, x2-x);
      const pcol = PHASE_COL[P.name] || '#556';
      ctx.fillStyle = pcol;
      ctx.fillRect(x, phaseY+1, wpx, PHASE_H-2);
      ctx.strokeStyle = '#0b0d12'; ctx.lineWidth = 1;
      ctx.strokeRect(x+0.5, phaseY+1.5, Math.max(1,wpx-1), PHASE_H-3);
      boxLabel(P.name, x, wpx, phaseY + PHASE_H/2, pcol);
      rects.push({x:x,y:phaseY+1,w:wpx,h:PHASE_H-2,type:'phase',p:P});
    }
  }
  // CPU slices (host may nest; draw thin stacked)
  for (const c of D.cpu){
    if (c.e < view0 || c.s > view1) continue;
    const x = xOf(c.s,w), x2 = xOf(c.e,w);
    const wpx = Math.max(1, x2-x);
    const ccol = c.name.indexOf('Graph')>=0 ? '#5fa8d3' :
                 c.name.indexOf('Memcpy')>=0 ? '#c9a227' :
                 c.name.indexOf('Synchronize')>=0 ? '#7a6f9b' : '#4a6070';
    ctx.fillStyle = ccol;
    ctx.fillRect(x, cpuY+4, wpx, CPU_H-8);
    boxLabel(c.name, x, wpx, cpuY + CPU_H/2, ccol);
    rects.push({x:x,y:cpuY+4,w:wpx,h:CPU_H-8,type:'cpu',p:c});
  }

  // axis
  ctx.fillStyle = '#7a8090'; ctx.font = '10px sans-serif';
  const span = view1-view0;
  for (let i=0;i<=5;i++){
    const t = view0 + span*i/5, x = xOf(t,w);
    ctx.fillText(fmtus(t-view0), Math.min(x+2,w-40), H-6);
    ctx.strokeStyle='#5a6070'; ctx.beginPath();
    ctx.moveTo(x,axisTop); ctx.lineTo(x,axisTop+6); ctx.stroke();
  }

  // measurement markers A/B (draggable; full height so you can line up an edge)
  [[markA,'#00e5ff','A'],[markB,'#ffd400','B']].forEach(m=>{
    const t=m[0], col=m[1], lab=m[2];
    if (t<view0 || t>view1) return;
    const x=xOf(t,w);
    ctx.strokeStyle=col; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(x,PAD_T); ctx.lineTo(x,axisTop); ctx.stroke();
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
    `drag=select kernels  ctrl/cmd+click=add/remove 1  ctrl/cmd+drag=add range  scroll/+-=zoom  shift+drag/arrows=pan  drag A/B=measure (snaps to edges; alt=free)  click=1 kernel  esc=clear`;
}

// summary table
const tb = document.querySelector('#tbl tbody');
tb.innerHTML = D.summary.map(r=>{
  const col = D.colors[r.stall]||D.colors.unknown;
  return `<tr data-fam="${r.fam}"><td><span class="fam-dot" style="background:${col}"></span>${r.fam}</td>`+
    `<td>${r.per_tok}</td><td>${r.busy_pct}</td><td>${r.stall}</td></tr>`;
}).join('');
// selection: a table row selects a FAMILY (dims other families); a click on the
// timeline selects a SINGLE kernel (bright outline) and shows its details below.
let selectedFam = null;
let selectedSlice = null;
let selectedSlices = null;       // Set of D.gpu slices (box multi-select), or null
function setSelection(fam){
  selectedFam = fam;
  selectedSlice = null;          // family mode clears single-kernel selection
  selectedSlices = null;         // ...and box multi-selection
  const rows = tb.querySelectorAll('tr');
  rows.forEach(tr=>tr.classList.toggle('sel', fam!==null && tr.dataset.fam===fam));
  if(fam){ const el=[...rows].find(tr=>tr.dataset.fam===fam);
           if(el) el.scrollIntoView({block:'nearest'}); }
  updateDetail(); draw();
}
function selectSlice(sl){        // sl is a slice object from D.gpu (or null)
  selectedSlice = sl; selectedFam = null; selectedSlices = null;
  const rows = tb.querySelectorAll('tr');
  rows.forEach(tr=>tr.classList.toggle('sel', sl && tr.dataset.fam===sl.fam));
  if(sl){ const el=[...rows].find(tr=>tr.dataset.fam===sl.fam);
          if(el) el.scrollIntoView({block:'nearest'}); }
  updateDetail(); draw();
}
// box multi-select: gather every GPU slice overlapping the dragged time range.
// add=true (Ctrl/Cmd+drag) unions into the current selection instead of replacing.
function selectBox(t0,t1,add){
  const set = (add && selectedSlices) ? new Set(selectedSlices) : new Set();
  if(add && !selectedSlices && selectedSlice) set.add(selectedSlice);
  for(const s of D.gpu){ if(s.e>=t0 && s.s<=t1) set.add(s); }
  selectedSlices = set.size ? set : null;
  selectedSlice = null; selectedFam = null;
  const fams = new Set([...(selectedSlices||[])].map(s=>s.fam));
  tb.querySelectorAll('tr').forEach(tr=>tr.classList.toggle('sel', fams.has(tr.dataset.fam)));
  updateDetail(); draw();
}
// Ctrl/Cmd+click: add or remove ONE slice from the multi-select set. Seeds a new
// set from the current single/box selection so you can refine right after a lasso;
// removing the last member clears back to no selection.
function toggleSlice(sl){
  const set = selectedSlices ? new Set(selectedSlices) : new Set();
  if(!selectedSlices && selectedSlice) set.add(selectedSlice);
  if(set.has(sl)) set.delete(sl); else set.add(sl);
  selectedSlices = set.size ? set : null;
  selectedSlice = null; selectedFam = null;
  const fams = new Set([...(selectedSlices||[])].map(s=>s.fam));
  tb.querySelectorAll('tr').forEach(tr=>tr.classList.toggle('sel', fams.has(tr.dataset.fam)));
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
  if(selectedSlices){ renderMultiSelect(); return; }
  if(selectedSlice){ renderSelectedKernel(); return; }
  if(selectedFam){ renderFamilyMembers(); return; }
  dp.style.display='none'; dp.innerHTML='';
}
// Box-selection view: aggregate the selected GPU slices by family (count, summed
// kernel time, share of the selection) -- a quick "what did I just lasso" summary.
function renderMultiSelect(){
  const sl=[...selectedSlices];
  const groups=new Map(); let totDur=0;
  for(const s of sl){
    const dur=s.e-s.s; totDur+=dur;
    let g=groups.get(s.fam);
    if(!g){ g={fam:s.fam,stall:s.stall,n:0,dur:0}; groups.set(s.fam,g); }
    g.n++; g.dur+=dur;
  }
  const gs=[...groups.values()].sort((a,b)=>b.dur-a.dur);
  let left=`<h2>Selection</h2>`+
    `<div class="sub" style="margin-bottom:6px">${sl.length} kernel${sl.length===1?'':'s'}, `+
    `${gs.length} famil${gs.length===1?'y':'ies'} &mdash; total kernel time ${fmtus(totDur)}</div>`;
  left+=`<table><thead><tr><th style="text-align:left">family</th>`+
     `<th style="text-align:left">count</th><th style="text-align:left">kernel time</th>`+
     `<th style="text-align:left">% of sel</th></tr></thead><tbody>`;
  for(const g of gs){
    const col=D.colors[g.stall]||D.colors.unknown;
    left+=`<tr><td><span class="fam-dot" style="background:${col}"></span>${g.fam}</td>`+
       `<td>${g.n}</td><td>${fmtus(g.dur)}</td>`+
       `<td>${(g.dur/totDur*100).toFixed(1)}%</td></tr>`;
  }
  left+=`</tbody></table>`;
  // Two columns: family breakdown on the left, fusion analysis in the wide space
  // on the right (falls back to stacking on narrow panes via flex-wrap).
  let h=`<div style="display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap">`+
     `<div style="flex:0 1 auto">${left}</div>`+
     `<div style="flex:1 1 340px;min-width:300px">${fusionSection(sl, gs, totDur)}</div>`+
     `</div>`+
     `<div class="sub" style="margin-top:6px">Drag again to reselect; click one kernel for its `+
     `full detail; Esc clears.</div>`;
  dp.innerHTML=h; dp.style.display='block';
}
// Fusion-opportunity analysis for a lasso selection of (usually small) kernels.
// Two questions: (1) how much END-TO-END time could fusing them reclaim -- the
// inter-kernel idle (launch + scheduling gaps between the selected dispatches);
// (2) would merging the distinct families' resources blow the VGPR file (spill)
// or the LDS budget (occupancy drop). Register/LDS totals are modeled two ways:
// max() = perfect reuse (kernels run as sequential phases, regs freed between),
// sum() = no reuse (everything live at once) -- the true fused cost is between.
function fusionSection(sl, gs, totDur){
  const KB=b=>b>=1024?(b/1024).toFixed(1)+' KB':b+' B';
  let minS=Infinity, maxE=-Infinity;
  for(const s of sl){ if(s.s<minS)minS=s.s; if(s.e>maxE)maxE=s.e; }
  const wall=maxE-minS, busy=totDur;
  const bySt=[...sl].sort((a,b)=>a.s-b.s);
  let gapIdle=0;
  for(let i=1;i<bySt.length;i++){ const g=bySt[i].s-bySt[i-1].e; if(g>0) gapIdle+=g; }
  const winPct=wall>0?gapIdle/wall*100:0;
  const rows=gs.map(g=>({g, fc:D.fam_counters[g.fam]||{}}));
  const vgprs=rows.map(r=>r.fc.vgpr||0);
  const ldss =rows.map(r=>r.fc.lds_static||0);
  const scr  =rows.map(r=>r.fc.scratch||0);
  const Ws   =rows.map(r=>r.fc.wave?Math.round(r.fc.wg/r.fc.wave):0);
  const vMax=Math.max(0,...vgprs), vSum=vgprs.reduce((a,b)=>a+b,0);
  const lMax=Math.max(0,...ldss),  lSum=ldss.reduce((a,b)=>a+b,0);
  const anyScratch=scr.some(x=>x>0);
  const Wf=Math.max(0,...Ws);           // most wave-heavy block sets the fused occupancy
  const hw=D.hw;
  // gfx1151 (RDNA3.5, wave32): each SIMD32 shares a 1536-VGPR file across up to 16
  // resident waves, so 1536/16 = 96 VGPR/wave is the most a wave can use and still
  // hit full 16-wave occupancy; every VGPR above that costs resident waves. v0-v255
  // is the architectural max -> above 256 the compiler spills to scratch.
  const VMAX_THREAD=256;                                              // architectural v0-v255 max -> spill above
  const VGPR_FULL_OCC = hw ? Math.floor(hw.vgpr_per_simd/hw.slots_per_simd) : 96;  // 1536/16 = 96/wave for 16 waves
  const occFor=(V,L)=>{
    if(!hw||!Wf) return null;
    const slotsWGP=hw.simd_per_wgp*hw.slots_per_simd;
    const vgprWGP =hw.simd_per_wgp*hw.vgpr_per_simd;
    const bSlots=Math.floor(slotsWGP/Wf);
    const bVgpr =V>0?Math.floor(vgprWGP/(Wf*V)):bSlots;
    const bLds  =L>0?Math.floor(hw.lds_per_wgp/L):Infinity;
    let lim='slots', resid=bSlots;
    if(bVgpr<resid){lim='VGPR';resid=bVgpr;}
    if(bLds<resid){lim='LDS';resid=bLds;}
    return {occ:resid>0?Math.round(resid*Wf/slotsWGP*100):0, lim, resid};
  };
  const occBest=occFor(vMax,lMax), occWorst=occFor(vSum,lSum);
  let h=`<h2>Fusion analysis</h2>`;
  h+=`<table><tbody>`+
     `<tr><td>span (wall)</td><td>${fmtus(wall)}</td></tr>`+
     `<tr><td>busy (sum kernels)</td><td>${fmtus(busy)}</td></tr>`+
     `<tr><td>inter-kernel idle</td><td style="color:${gapIdle>0?'#8fe388':'#9aa6b2'}">`+
       `${fmtus(gapIdle)} <span class="r">(reclaimable by fusion; ${winPct.toFixed(1)}% of span)</span></td></tr>`+
     `</tbody></table>`;
  h+=`<table style="margin-top:6px"><thead><tr><th style="text-align:left">family</th>`+
     `<th style="text-align:left">VGPR</th><th style="text-align:left">LDS/blk</th>`+
     `<th style="text-align:left">scratch</th><th style="text-align:left">occ</th></tr></thead><tbody>`;
  for(const r of rows){
    const fc=r.fc, col=D.colors[r.g.stall]||D.colors.unknown;
    h+=`<tr><td><span class="fam-dot" style="background:${col}"></span>${r.g.fam}</td>`+
       `<td>${fc.vgpr||'<span class="r">-</span>'}</td>`+
       `<td>${fc.lds_static?KB(fc.lds_static):'0'}</td>`+
       `<td style="color:${fc.scratch?'#ff6b6b':'inherit'}">${fc.scratch?KB(fc.scratch):'0'}</td>`+
       `<td>${fc.occ?fc.occ+'%':'<span class="r">-</span>'}</td></tr>`;
  }
  h+=`</tbody></table>`;
  if(hw && Wf){
    h+=`<table style="margin-top:6px"><tbody>`+
       `<tr><td>fused VGPR/wave</td><td>${vMax} .. ${vSum} `+
         `<span class="r">(reuse .. no-reuse; ${VGPR_FULL_OCC}=full occ, ${VMAX_THREAD}=spill)</span></td></tr>`+
       `<tr><td>fused LDS/block</td><td>${KB(lSum)} `+
         `<span class="r">/ ${KB(hw.lds_per_wgp)} per-WGP budget</span></td></tr>`;
    if(occBest && occWorst)
      h+=`<tr><td>fused occupancy (modeled)</td><td>~${occWorst.occ}% .. ${occBest.occ}% `+
         `<span class="r">(${occWorst.lim} .. ${occBest.lim}-bound)</span></td></tr>`;
    h+=`</tbody></table>`;
  }
  // A fused kernel allocates one register file for its whole body, so it inherits
  // AT LEAST the largest member's VGPR/wave even under perfect reuse -- which forces
  // that wave's (possibly low) occupancy onto every folded-in kernel. So vMax, not
  // just vSum, drives the verdict: past 96/wave you drop below 16 resident waves.
  const risks=[];
  if(anyScratch) risks.push('a selected kernel already spills to scratch');
  if(vSum>VMAX_THREAD) risks.push(`no-reuse VGPR ${vSum} exceeds the ${VMAX_THREAD}/wave architectural max (v0-v${VMAX_THREAD-1}) &rarr; scratch spill unless the compiler reuses registers`);
  else if(occBest && occBest.occ<50) risks.push(`the largest kernel uses ${vMax} VGPR/wave (> ${VGPR_FULL_OCC}/wave for full 16-wave occupancy); even perfect reuse caps the fused kernel at ~${occBest.occ}% occupancy and drags every folded-in kernel down to it`);
  if(hw && lSum>hw.lds_per_wgp) risks.push(`combined LDS ${KB(lSum)} exceeds the ${KB(hw.lds_per_wgp)} per-WGP budget`);
  let verdict, vc;
  if(gapIdle<=0 || winPct<2){
    verdict='Low payoff -- almost no inter-kernel idle to reclaim; fusion would only save launch bookkeeping.'; vc='#9aa6b2';
  } else if(risks.length){
    verdict='Fusible but risky -- '+risks.join('; ')+'. Weigh the '+fmtus(gapIdle)+' idle win against the spill / occupancy cost.'; vc='#ffb454';
  } else {
    verdict='Good candidate -- '+fmtus(gapIdle)+' reclaimable idle, and the combined VGPR/LDS stay within budget'+
      (occWorst?` (modeled occ >=${occWorst.occ}%)`:``)+'. Fusing removes the launch gaps without spilling.'; vc='#8fe388';
  }
  h+=`<div style="margin:8px 0 0;padding:6px 9px;border-left:3px solid ${vc};`+
     `background:rgba(255,255,255,.05);border-radius:3px;line-height:1.35">`+
     `<b style="color:${vc}">FUSION</b> <span style="color:#c8d0da">${verdict}</span></div>`;
  return h;
}
// 1D gap-clustering of a shape group's kernel times (ns) to surface multi-modal
// structure. items=[{d,L}]. A new cluster starts wherever the gap between two
// sorted durations exceeds max(3us, 6% of the median) -- wide enough to ignore
// the normal per-dispatch jitter but catch the attn_q ~57/70us fast/slow split
// and the lone ~+24us timestamp-artifact tail. Returns clusters low->high with
// center (median), count, span, and the distinct layers that landed in each.
function clusterDur(items){
  const v=items.slice().sort((a,b)=>a.d-b.d);
  const med=v[Math.floor(v.length/2)].d;
  const thr=Math.max(3000, med*0.06);
  const cl=[]; let cur=[v[0]];
  for(let i=1;i<v.length;i++){
    if(v[i].d-v[i-1].d>thr){ cl.push(cur); cur=[v[i]]; } else cur.push(v[i]);
  }
  cl.push(cur);
  return cl.map(c=>{
    const ds=c.map(x=>x.d);
    const layers=[...new Set(c.map(x=>x.L))].sort((a,b)=>a-b);
    return {c:ds[Math.floor(ds.length/2)], n:c.length, lo:ds[0], hi:ds[ds.length-1], layers};
  });
}
// A dispatch carries an effective-BW number only when it is an order-mapped
// weight-streaming matvec (mmvq / mmq / wvsplitk), which is inherently DRAM-BW
// bound. So "BW-bound" for the eff-BW < 80% red flag keys off the matvec identity,
// NOT the PMC dominant-stall bucket -- wvsplitk buckets as "lds" (it stages
// activations in LDS, hence bank conflicts) yet is still streaming weights from
// DRAM. Also accept memory/lds stall for any other mapped streaming kernel.
function isBwBound(fam){
  const fc=D.fam_counters[fam]||{};
  return /mul_mat_vec|mul_mat_q/.test(fam) || fc.stall==='memory' || fc.stall==='lds';
}
// Family view: when a per-kernel-family row is selected (no single slice), list
// every order-mapped dispatch in that family with its packed footprint + effective BW.
function renderFamilyMembers(){
  const fam=selectedFam;
  const KB=b=>b>=1048576?(b/1048576).toFixed(1)+' MB':(b/1024).toFixed(1)+' KB';
  // One row per dispatch -- NO aggregation. A per-shape mean would hide a bimodal
  // split (the same attn_q shape runs a stable ~57us at some layers and ~68us at
  // others -- structurally identical work, different by layer position), and would
  // also blend in the ~+24us interrupt-latency timestamp artifact that hits ~5% of
  // gfx1151 dispatches. Listing every dispatch keeps each measured time honest and
  // lets both effects be seen directly. Rows are ordered so dispatches with the
  // same shape [K x N] are adjacent (then role, layer, execution time) -- so
  // repeats of one shape sit together for eyeballing the spread.
  // allRows = every order-mapped dispatch of this family across ALL baked tokens;
  // used only to compute per-layer modes robustly (more samples per layer wash out
  // the ~5% +24us artifact). The DISPLAYED table is scoped to a single complete
  // token cycle (this panel is "Kernel family/Token"), so it stays ~one token's
  // worth of rows instead of n_tokens_baked copies.
  const allRows=[];
  for(const s of D.gpu){ if(s.fam===fam && s.map) allRows.push(s); }
  const win=secondTokenWin();
  const rows = win ? allRows.filter(s=> s.s>=win.t0 && s.s<win.t1) : allRows.slice();
  if(!rows.length && allRows.length) rows.push(...allRows);
  // Sort by SHAPE first (K, trueN, quant), then role, then layer, then exec time.
  // Shape-primary keeps dispatches of identical [K x N] adjacent even when they
  // belong to different roles that happen to share a shape (e.g. attn_k and attn_v
  // are both 2560 x 1024 Q4_K in this model) -- role-first would wedge an unrelated
  // shape between them.
  rows.sort((a,b)=>{
    const ma=a.map, mb=b.map;
    if(ma.K!==mb.K) return ma.K-mb.K;
    if(ma.trueN!==mb.trueN) return ma.trueN-mb.trueN;
    if(ma.q!==mb.q) return ma.q<mb.q?-1:1;
    if(ma.role!==mb.role) return ma.role<mb.role?-1:1;
    if(ma.L!==mb.L) return ma.L-mb.L;
    return a.s-b.s;
  });
  let h=`<h2>Kernel family/Token</h2>`+
    `<div style="color:#7fd1ff;word-break:break-all;margin-bottom:6px">${fam}`+
    `<span class="r"> (${rows.length} dispatch${rows.length===1?'':'es'})</span></div>`;
  if(!rows.length){
    h+=`<div class="sub">No order-mapped dispatches in this family`+
       (D.has_map?`.`:` -- run with --gguf to attach shape / packed footprint / effective BW.`)+`</div>`;
    dp.innerHTML=h; dp.style.display='block'; return;
  }
  // Per-shape modes -> per-LAYER lookup, rendered as a table column (below) so
  // each dispatch row shows which mode its layer belongs to. Within each
  // role+shape+quant group, reduce every LAYER to its median kernel time, then
  // cluster those per-layer medians. The fast/slow split (e.g. attn_q ~57us vs
  // ~70us) is layer-LOCKED -- each layer is individually tight but sits at a
  // different center -- so clustering the raw pooled dispatches would smear the
  // modes into one continuum (per-dispatch jitter + the ~5% +24us timestamp
  // artifact bridge the gap). Per-layer medians are robust to that artifact.
  const med=a=>{const v=a.slice().sort((x,y)=>x-y);return v[Math.floor(v.length/2)];};
  const shp=new Map();
  for(const s of allRows){ const m=s.map;
    const k=m.role+'|'+m.K+'x'+m.trueN+'|'+m.q;
    if(!shp.has(k)) shp.set(k,new Map());
    const bl=shp.get(k); if(!bl.has(m.L)) bl.set(m.L,[]);
    bl.get(m.L).push(s.e-s.s);
  }
  // layerMode: shapeKey|L -> {ci, n, c}  (mode index low->high, mode count, center ns)
  const layerMode=new Map();
  for(const [k,byL] of shp){
    const items=[...byL.entries()].map(([L,ds])=>({d:med(ds),L}));
    const cl = items.length>=2 ? clusterDur(items) : [{c:items[0].d,n:1,layers:[items[0].L]}];
    cl.forEach((c,ci)=>c.layers.forEach(L=>layerMode.set(k+'|'+L,{ci,n:cl.length,c:c.c})));
  }
  const modeName=['','unimodal','bimodal','trimodal'];
  const modeCol=(ci,n)=> n<=1 ? '#8aa0b4'
      : n===2 ? (ci===0?'#8fe388':'#ff9f6b')
      : ['#8fe388','#ffd479','#ff9f6b','#ff6b6b','#c58cff'][Math.min(ci,4)];
  h+=`<table><thead><tr><th style="text-align:left">role</th>`+
     `<th style="text-align:left">layer</th>`+
     `<th style="text-align:left">shape [K x N]</th><th style="text-align:left">packed</th>`+
     `<th style="text-align:left">kernel time</th>`+
     `<th style="text-align:left">eff BW</th><th style="text-align:left">eff BW %</th>`+
     `<th style="text-align:left">over-fetch</th>`+
     `<th style="text-align:left">modes</th></tr></thead><tbody>`;
  // Order-mapped matvec families whose achieved BW falls short of ~peak are the
  // ones worth flagging -- eff BW < 80% on a BW-bound kernel means it is leaving
  // DRAM bandwidth on the table (unlike a compute-bound kernel, where low eff BW
  // is expected and not a defect). See isBwBound(): matvec identity, not the PMC
  // stall bucket (wvsplitk buckets as "lds" but is still weight-streaming).
  const memBound=isBwBound(fam);
  // Shade alternating role+shape+layer groups so a weight's repeats cluster visually.
  let prevKey=null, band=0;
  rows.forEach((s,i)=>{
    const m=s.map, dur=s.e-s.s;
    const eb=dur?(m.packed/dur):0;
    const ep=D.peak_bw_gbs?(eb/D.peak_bw_gbs*100):0;
    const lowBW=memBound && ep>0 && ep<80;
    const bwCol=lowBW?'#ff6b6b':'#8fe388';
    const key=m.K+'x'+m.trueN+'|'+m.q+'|'+m.role+'|L'+m.L;
    if(key!==prevKey){ band^=1; prevKey=key; }
    h+=`<tr class="shrow" data-idx="${i}" title="frame this dispatch in the timeline"`+
       (band?` style="background:rgba(255,255,255,.04)"`:``)+`>`+
       `<td style="color:#ffd479">${m.role}</td>`+
       `<td>${m.L<0?'out':('L'+m.L)}</td>`+
       `<td>${m.K} x ${m.trueN} <span class="r">${m.q}</span></td>`+
       `<td>${KB(m.packed)}${m.fused?` <span class="r">(${m.fused})</span>`:``}</td>`+
       `<td>${fmtus(dur)}</td>`+
       `<td style="color:${bwCol}">${eb.toFixed(1)} GB/s</td>`+
       `<td style="color:${bwCol}">${ep.toFixed(1)}%${lowBW?' <span class="r">(BW-bound)</span>':''}</td>`+
       `<td>${m.overfetch?m.overfetch.toFixed(2)+'x':'<span class="r">-</span>'}</td>`+
       (()=>{const md=layerMode.get(m.role+'|'+m.K+'x'+m.trueN+'|'+m.q+'|'+m.L);
         if(!md||md.n<=1) return `<td><span class="r">unimodal</span></td>`;
         return `<td style="color:${modeCol(md.ci,md.n)};white-space:nowrap">`+
           `${modeName[md.n]||md.n+'-modal'} <b>${md.ci+1}/${md.n}</b> `+
           `<span class="r">@${fmtus(md.c)}</span></td>`;})()+`</tr>`;
  });
  h+=`</tbody></table>`+
     `<div class="sub" style="margin-top:6px">Every order-mapped dispatch of `+
     `<b>${fam}</b> in one complete decode token, one row per dispatch `+
     `(no averaging), ordered so same shape [K x N] sit together. <b>packed</b> = theoretical `+
     `on-disk weight bytes (gate+up folded when fused); <b>kernel time</b> = this dispatch's `+
     `measured Start->End (raw; ~5% of gfx1151 dispatches carry a ~+24us interrupt-latency `+
     `timestamp artifact -- visible as a lone inflated row); <b>modes</b> = this shape's `+
     `across-layer kernel-time clusters (per-layer medians, gap &gt; max(3&micro;s, 6% median)); `+
     `<b>k/N @center</b> marks which of N modes this layer falls in (fastest green -> slowest `+
     `orange), so a bimodal/trimodal fast/slow layer split is visible per row; <b>eff BW</b> = `+
     `packed / kernel time (over-fetch-immune, vs peak ${D.peak_bw_gbs} GB/s). `+
     `Click a row to frame that dispatch in the timeline.</div>`;
  dp.innerHTML=h; dp.style.display='block';
  // Row click frames + selects that exact dispatch, reusing find's framing.
  dp.querySelectorAll('tr.shrow').forEach(tr=>{
    tr.onclick=()=>{ const s=rows[+tr.dataset.idx];
      if(s) applyFindResult({t0:s.s, t1:s.e, select:s}); };
  });
}
function renderSelectedKernel(){
  const s=selectedSlice, fc=D.fam_counters[s.fam]||{};
  const dg=diagnose(fc);
  const hasCode = !!((D.att_code_by_fam||{})[s.fam]);
  const dbgBtn = hasCode
    ? `<button id="attdbg" style="cursor:pointer;background:#2a3a52;color:#dbe6f5;`+
      `border:1px solid #3a5578;border-radius:3px;padding:3px 12px;font-size:12px;`+
      `flex:0 0 auto;white-space:nowrap">Open Trace View</button>`
    : ``;
  let h=`<h2>Selected kernel</h2>`+
    `<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;flex-wrap:wrap">`+
    `<div style="color:#7fd1ff;word-break:break-all;flex:0 1 auto">${s.fam}</div>`+
    dbgBtn+`</div>`+
    (dg?`<div style="margin:0 0 8px;padding:6px 9px;border-left:3px solid ${dg.c};`+
        `background:rgba(255,255,255,.05);border-radius:3px;line-height:1.35">`+
        `<b style="color:${dg.c}">${dg.t}</b> `+
        `<span style="color:#c8d0da">${dg.a}</span></div>`:``)+
    `<table><tbody>`+
    `<tr><td>duration</td><td>${fmtus(s.e-s.s)} <span class="r">(this dispatch)</span></td></tr>`;
  // Steady-state average of this within-token kernel position across all post-warmup
  // tokens -- decode is periodic so the Nth kernel is the same dispatch every token;
  // the mean +/- spread is a far more stable cost signal than one noisy dispatch.
  if(D.has_kstats){
    const ks=D.kstats[s.ti+'|'+s.fam];
    if(ks && ks.n>1)
      h+=`<tr><td>duration (avg)</td><td style="color:#8fe388">${fmtus(ks.mean)} &plusmn; ${fmtus(ks.std)} `+
         `<span class="r">(${fmtus(ks.min)}..${fmtus(ks.max)}, n=${ks.n} of ${D.kstats_ntok} tokens)</span></td></tr>`;
  }
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
    h+=`<tr><td>packed weights</td><td>${KB(m.packed)} <span class="r">(theoretical, on-disk`+
       (m.fused?`, ${m.fused} fused`:``)+`)</span></td></tr>`;
    if(m.effbw){ const memBound=isBwBound(s.fam), lowBW=memBound && m.effbw_pct>0 && m.effbw_pct<80;
      h+=`<tr><td>effective BW</td><td style="color:${lowBW?'#ff6b6b':'#8fe388'}">${m.effbw} GB/s `+
       `(${m.effbw_pct}% of ${D.peak_bw_gbs})${lowBW?' <span class="r">(BW-bound, <80% peak)</span>':''} `+
       `<span class="r">(useful: packed / this dispatch time)</span></td></tr>`; }
    if(m.measured){ const src=m.mexact?'per-weight (order-mapped)':'family+N avg';
      h+=`<tr><td>FETCH_SIZE</td><td>${KB(m.measured)} <span class="r">(measured DRAM read, ${src})</span></td></tr>`+
         `<tr><td>over-fetch</td><td>${m.overfetch}x <span class="r">(FETCH_SIZE / packed; ${src})</span></td></tr>`; }
  }
  // Achieved BW is per-family (all shapes blended) and counts over-fetched bytes,
  // so it is misleading for a single dispatch. When this slice is order-mapped we
  // already show effective BW (per-dispatch, over-fetch-immune); only fall back to
  // achieved BW for unmapped kernels where it is the sole bandwidth signal.
  if(D.has_bw && fc.bw_gbs && !s.map){
    h += `<tr><td>achieved BW</td><td>${fc.bw_gbs} GB/s (${fc.bw_pct}% of ${D.peak_bw_gbs}), ${fc.kb_disp} KB/disp <span class="r">(per-family, raw traffic)</span></td></tr>`;
  }
  if(D.has_pmc){
    h+=`<tr><td>MemUnitBusy</td><td>${fc.mem}%</td></tr>`+
       `<tr><td>L2 hit</td><td>${fc.l2}%</td></tr>`+
       `<tr><td>Occupancy</td><td>${fc.occ}%</td></tr>`+
       `<tr><td>LDS bank conflict</td><td>${fc.lds}</td></tr>`+
       `<tr><td>WriteUnitStalled</td><td>${fc.wr}</td></tr>`;
    if(fc.ea) h+=`<tr><td>EA (DRAM iface) busy</td><td>${fc.ea}%</td></tr>`;
    if(fc.alu) h+=`<tr><td>ALU (VALU) busy</td><td>${fc.alu}%</td></tr>`;
    if(fc.vgpr) h+=`<tr><td>VGPR / SGPR</td><td>${fc.vgpr} / ${fc.sgpr}`+
       (fc.accum_vgpr?` (+${fc.accum_vgpr} accum)`:``)+`</td></tr>`;
    const sc=fc.scratch||0;
    h+=`<tr><td>scratch size</td><td>${sc>=1024?(sc/1024).toFixed(1)+' KB':sc+' B'}</td></tr>`;
    // ---- tiling / scheduling ----
    if(fc.wg){
      const W = fc.wave ? Math.round(fc.wg/fc.wave) : 0;   // waves per block
      const L = fc.lds_static||0;
      h+=`<tr><td colspan="2" style="color:var(--dim);padding-top:8px;`+
         `text-transform:uppercase;font-size:10px;letter-spacing:.5px">tiling / scheduling</td></tr>`;
      const nblk = s.blocks || 0;
      if(nblk)
        h+=`<tr><td>grid</td><td>${nblk} blocks <span class="r">(this dispatch)</span></td></tr>`;
      h+=`<tr><td>block (workgroup)</td><td>${fc.wg} threads`+
         (W?` = ${W} wave${fc.wave}`:``)+`</td></tr>`;
      // Wavefronts for THIS dispatch = grid blocks x waves-per-block (exact: the
      // hw counter is just grid_size/wave_size). Derived from this slice's grid so
      // it tracks the selected dispatch, not the useless per-family mean.
      if(nblk && W)
        h+=`<tr><td>wavefronts</td><td>${nblk*W} `+
           `<span class="r">(${nblk} blocks x ${W} wave${fc.wave})</span></td></tr>`;
      h+=`<tr><td>LDS / block</td><td>${L>=1024?(L/1024).toFixed(1)+' KB':L+' B'}`+
         (L===0?` <span class="r">(static; dynamic extern-shared not profiled)</span>`:``)+`</td></tr>`;
      // Tier 2: modeled occupancy limiter (gfx1151 wave32).
      const hw=D.hw;
      if(hw && W){
        const slotsWGP=hw.simd_per_wgp*hw.slots_per_simd;   // 64 wave32
        const vgprWGP =hw.simd_per_wgp*hw.vgpr_per_simd;    // 6144
        const V=fc.vgpr||1;
        const bSlots=Math.floor(slotsWGP/W);
        const bVgpr =Math.floor(vgprWGP/(W*V));
        const bLds  =L>0?Math.floor(hw.lds_per_wgp/L):Infinity;
        let lim='slots', resid=bSlots;
        if(bVgpr<resid){lim='VGPR';resid=bVgpr;}
        if(bLds<resid){lim='LDS';resid=bLds;}
        if(resid>0){
          const theo=Math.round(resid*W/slotsWGP*100);
          const chip=resid*hw.wgp;
          h+=`<tr><td>resident / WGP</td><td>${resid} block${resid!==1?'s':''} `+
             `<span class="r">(${lim}-bound; modeled gfx1151 wave32)</span></td></tr>`;
          h+=`<tr><td>occupancy (modeled)</td><td>theo ~${theo}%`+
             (fc.occ?` &middot; achieved ${fc.occ}%`:``)+`</td></tr>`;
          if(nblk)
            h+=`<tr><td>rounds to drain</td><td>~${Math.ceil(nblk/chip)} `+
               `<span class="r">(${nblk} blocks / ${chip} chip-resident)</span></td></tr>`;
        }
      }
    }
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
     `<br><b>packed footprint</b> = the weight tensor's on-disk quantized size (from GGUF) `+
     `= the DRAM bytes a read-once matvec MUST move (gate+up folded in when fused). `+
     `<b>effective BW</b> = packed / this dispatch's exact time = <i>useful</i> throughput. `+
     `Model assumption: <b>weights always come from DRAM</b> and <b>L2 hits are activation `+
     `reuse</b>. <b>FETCH_SIZE</b> = measured DRAM bytes read (L2 misses only). `+
     `<b>over-fetch</b> = FETCH_SIZE / packed: ~1.0x = streamed once, &gt;1 = weight `+
     `refetched (bad tiling). NOTE achieved BW (FETCH_SIZE/time) rewards over-fetch -- a `+
     `kernel that reads the same bytes 100x keeps DRAM busy but does no extra work; `+
     `<b>effective BW is the over-fetch-immune roofline number</b>. When the FETCH run is `+
     `order-mapped (GGUF present), over-fetch is <b>per-weight exact</b> even for weights `+
     `sharing an N; otherwise it falls back to a family+N average (separate PMC run) that `+
     `blends weights of the same N.</div>`;
  // ---- ATT thread-trace stall overlay (folded in via --att-dir) ----
  const at=(D.att_by_fam||{})[s.fam];
  if(at){
    const tot=at.stall||1;
    h+=`<div style="margin-top:12px;padding:8px 10px;border-left:3px solid #ff9d5c;`+
       `background:rgba(255,157,92,.07);border-radius:3px">`+
       `<b style="color:#ff9d5c">ATT thread-trace stalls</b> `+
       `<span class="r">(${at.n_disp} dispatch${at.n_disp!==1?'es':''}, ~1 SIMD; `+
       `${fmtc(at.stall)} stall / ${fmtc(at.lat)} latency / ${fmtc(at.idle)} idle cyc)</span>`;
    if(at.top && at.top.length){
      h+=`<table style="margin-top:6px"><thead><tr>`+
         `<th style="text-align:left">instruction</th>`+
         `<th style="text-align:right">stall</th><th style="text-align:right">% stall</th>`+
         `<th style="text-align:right">idle</th><th style="text-align:right">hits</th>`+
         `</tr></thead><tbody>`;
      for(const t of at.top){
        const pct=100*t.st/tot;
        h+=`<tr><td style="font-family:monospace;color:#d7dde5">${esc(t.i)}</td>`+
           `<td style="text-align:right">${fmtc(t.st)}</td>`+
           `<td style="text-align:right;color:${pct>=25?'#ff6b6b':'#c8d0da'}">${pct.toFixed(1)}%</td>`+
           `<td style="text-align:right">${fmtc(t.idle)}</td>`+
           `<td style="text-align:right">${fmtc(t.hits)}</td></tr>`;
      }
      h+=`</tbody></table>`;
    }
    if(at.byclass && at.byclass.length)
      h+=`<div class="r" style="margin-top:6px">stall by opcode: `+
         at.byclass.map(c=>`${esc(c[0])} ${fmtc(c[1])}`).join(' &middot; ')+`</div>`;
    h+=`<div class="sub" style="margin-top:4px">Per-instruction cycles from the decoded `+
       `ATT trace of this kernel family. <b>stall</b> = cycles the wave was blocked at that `+
       `PC (e.g. <code>s_waitcnt vmcnt</code> = waiting on a global-memory load, so a matvec `+
       `dominated by it is memory-bound). Sampled on ~1 SIMD across ${at.n_disp} `+
       `dispatch${at.n_disp!==1?'es':''} -- a representative profile, not a full-GPU count.`+
       `</div></div>`;
  }
  // ---- "Trace this kernel with ATT" -- copy-ready command (Option A round-trip) ----
  const sym=s.fam.replace(/\[.*$/,'');
  const ac=D.att_cmd||{};
  const outdir=(ac.out_base||'.')+'/att-'+sym;
  const cmd=
    `# on the gfx1151 board (local ROCm): trace just this kernel with ATT\n`+
    `${ac.script} --kernel '${sym}' \\\n`+
    `  --build-dir ${ac.build_dir} \\\n`+
    `  --model ${ac.model} \\\n`+
    `  --out-dir ${outdir}\n`+
    `# then regenerate this overlay with the decoded trace folded in:\n`+
    `python3 ${ac.viewer} \\\n`+
    `  ${ac.regen_flags} \\\n`+
    `  --att-dir ${outdir} --out ${ac.out_html}`;
  const liveBtn = D.att_server
    ? `<button id="atttrace" style="cursor:pointer;background:#1f5c34;color:#e6f5ea;`+
      `border:1px solid #2f7d48;border-radius:3px;padding:1px 8px;font-size:11px;`+
      `margin-left:6px">Trace now</button>`+
      `<span id="attstatus" class="r" style="margin-left:8px;color:#c8d0da;display:none"></span>`
    : ``;
  h+=`<div style="margin-top:12px;padding:8px 10px;border:1px dashed var(--dim);border-radius:3px">`+
     `<b>Trace this kernel with ATT</b> `+
     `<button id="attcopy" style="cursor:pointer;background:#2a3340;color:#d7dde5;`+
     `border:1px solid #3a4553;border-radius:3px;padding:1px 8px;font-size:11px">copy</button>`+
     liveBtn+
     `<span id="attcopied" class="r" style="margin-left:8px;color:#8fe388;display:none">copied</span>`+
     `<pre style="margin:6px 0 0;padding:7px 9px;background:#0d1117;border-radius:3px;`+
     `overflow:auto;white-space:pre;font-size:11px;color:#c8d0da">${esc(cmd)}</pre>`+
     (at?``:`<div class="sub">No ATT data loaded for this kernel yet -- `+
       (D.att_server
         ? `click <b>Trace now</b> to dispatch ATT to a free GPU board over ssh and fold `+
           `the result in live, or `
         : ``)+
       `run the command above on the board, then re-run the viewer with `+
       `<code>--att-dir</code> to see per-instruction stalls here. (ATT filters by kernel `+
       `<i>symbol</i>, so it captures every quant/shape variant of `+
       `<code>${esc(sym)}</code>.)</div>`)+
     `</div>`;
  dp.innerHTML=h; dp.style.display='block';
  const cp=document.getElementById('attcopy');
  if(cp) cp.onclick=()=>copyCmd(cmd);
  const tb2=document.getElementById('atttrace');
  if(tb2) tb2.onclick=()=>traceKernelLive(sym, s.fam);
  const db=document.getElementById('attdbg');
  if(db) db.onclick=()=>openDebugView(s.fam);
}

// Open a new browser tab with a self-contained debug view for one kernel family,
// built client-side from D.att_code_by_fam (no server round-trip). When the trace
// carries DWARF line info (has_src + embedded source), it is a SYNCHRONIZED two-pane
// view: source on the left (per-line stall heat), full program-order ISA on the
// right. Clicking a source line highlights + scrolls to its instructions; clicking
// an instruction highlights + scrolls to its source line. Without line info it falls
// back to the ISA-only table. Source text is embedded by basename only (no paths).
function openDebugView(fam){
  const c=(D.att_code_by_fam||{})[fam];
  if(!c){ alert('No decoded ISA for this kernel yet -- trace it first.'); return; }
  const w=window.open('','_blank');
  if(!w){ alert('Popup blocked -- allow popups for this page to open the debug view.'); return; }
  const maxStall=c.rows.reduce((m,r)=>Math.max(m,r.st||0),0)||1;
  const srcFiles=c.src_files||{};
  const split=!!(c.has_src && Object.keys(srcFiles).length);
  const payload={fam:fam, sym:c.sym, n_disp:c.n_disp, stall:c.stall, lat:c.lat,
                 idle:c.idle, has_src:!!c.has_src, rows:c.rows, maxStall:maxStall,
                 src_files:srcFiles, split:split, exec:c.exec||null,
                 waves:c.waves||null,
                 occ:((c.occ_ref!=null&&c.occ_ref>=0)?(D.att_occ_pool||[])[c.occ_ref]:(c.occ||null)),
                 gloss:(D.isa_gloss||{}), regGloss:(D.reg_gloss||{})};
  const fileOpts=Object.keys(srcFiles).map(f=>`<option value="`+esc(f)+`">`+esc(f)+`</option>`).join('');
  const doc=`<!doctype html><html><head><meta charset="utf-8">`+
    `<title>ISA debug -- `+esc(fam)+`</title><style>`+
    `*{box-sizing:border-box}`+
    `body{margin:0;background:#0d1117;color:#d7dde5;display:flex;flex-direction:column;`+
    `height:100vh;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}`+
    `header{flex:0 0 auto;background:#161b22;border-bottom:1px solid #2a3340;padding:10px 16px}`+
    `h1{margin:0 0 4px;font-size:20px;color:#dbe6f5}`+
    `.sym{font-family:monospace;font-size:14px;color:#8b98a8;word-break:break-all}`+
    `.tot{margin-top:6px;font-size:14px;color:#c8d0da}`+
    `.note{margin-top:6px;font-size:14px;color:#e0b341}`+
    `#f{margin-top:8px;width:340px;max-width:60%;padding:5px 9px;background:#0d1117;`+
    `color:#d7dde5;border:1px solid #3a4553;border-radius:3px;font-size:14px}`+
    `#main{flex:1 1 auto;display:flex;min-height:0}`+
    `section{display:flex;flex-direction:column;min-width:0;min-height:0}`+
    `#isapane{flex:1 1 56%;border-right:1px solid #2a3340}#srcpane{flex:1 1 44%}`+
    `.phdr{flex:0 0 auto;padding:6px 12px;background:#12161c;border-bottom:1px solid #2a3340;`+
    `color:#9fb0c4;font-size:13px}`+
    `.hint{color:#6f7d8f;font-size:12px;margin-left:6px}`+
    `.scroll{flex:1 1 auto;overflow:auto;min-height:0}`+
    `#src{font-family:monospace;font-size:14px;line-height:1.5;padding-bottom:40vh}`+
    `.sl{display:flex;align-items:baseline;white-space:pre;cursor:pointer;`+
    `border-left:3px solid transparent}`+
    `.sl:hover{background:#161b22}.sl.hotl{background:#2a1a17}`+
    `.sl.selline{background:#243044;border-left-color:#4d90fe}`+
    `.sln{flex:0 0 54px;text-align:right;padding-right:10px;color:#5a6675;user-select:none}`+
    `.shb{flex:0 0 54px}.sbar{display:inline-block;height:8px;background:#ff6b6b;`+
    `border-radius:2px;vertical-align:middle}`+
    `.sst{flex:0 0 46px;text-align:right;padding-right:8px;color:#8b98a8;font-size:11px}`+
    `.sc{flex:1 1 auto;color:#d7dde5}`+
    `table{border-collapse:collapse;width:100%;font-size:14px}`+
    `th,td{padding:3px 10px;text-align:right;white-space:nowrap}`+
    `th{position:sticky;top:0;background:#1a2029;color:#9fb0c4;text-align:right;`+
    `border-bottom:1px solid #2a3340}`+
    `td.isa,th.isa{text-align:left;font-family:monospace;color:#d7dde5;white-space:pre}`+
    `td.src,th.src{text-align:left;font-family:monospace;color:#7fa7d8;white-space:nowrap}`+
    `td.a{text-align:right;font-family:monospace;color:#6f7d8f}`+
    `tbody tr{cursor:pointer}tbody tr:hover td{background:#161b22}`+
    `tr.sel td{background:#243044}`+
    `.bar{display:inline-block;height:9px;background:#ff6b6b;border-radius:2px;`+
    `vertical-align:middle}`+
    `.hot td.isa{color:#ffd7b0}`+
    `tr.step td{background:#3a2f10 !important;box-shadow:inset 3px 0 #e0b341}`+
    `tr.step td.isa{color:#ffe0a0}`+
    `.sl.stepline{background:#3a2f10;border-left-color:#e0b341}`+
    `#stepbar{margin-top:8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}`+
    `#stepbar button{background:#1f2733;color:#d7dde5;border:1px solid #3a4553;`+
    `border-radius:3px;padding:4px 10px;font-size:13px;cursor:pointer}`+
    `#stepbar button:hover{background:#2a3340}`+
    `#stepinfo{font-family:monospace;font-size:13px;color:#e0b341;margin-left:6px}`+
    `#tip{position:fixed;z-index:20;max-width:520px;background:#1a2029;color:#d7dde5;`+
    `border:1px solid #3a4553;border-radius:4px;padding:6px 10px;`+
    `font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;`+
    `box-shadow:0 4px 14px rgba(0,0,0,.5);pointer-events:none;display:none}`+
    `#tip b{color:#ffd7b0;font-family:monospace}`+
    `.reg{color:#7fd0ff;cursor:help;text-decoration:underline dotted #4a6a80}`+
    `#wvbtn{margin-left:12px;background:#1f2733;color:#d7dde5;border:1px solid #3a4553;`+
    `border-radius:3px;padding:4px 12px;font-size:13px;cursor:pointer}`+
    `#wvbtn:hover{background:#2a3340}`+
    `#wavepane{position:fixed;inset:0;z-index:30;background:#0d1117;display:none;`+
    `flex-direction:column}#wavepane.show{display:flex}`+
    `#wvbar{flex:0 0 auto;display:flex;align-items:center;gap:14px;padding:10px 16px;`+
    `background:#161b22;border-bottom:1px solid #2a3340;flex-wrap:wrap}`+
    `#wvbar h2{margin:0;font-size:16px;color:#dbe6f5}`+
    `#wvbar .lg{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:#9fb0c4;margin-right:10px}`+
    `#wvbar .sw{display:inline-block;width:13px;height:13px;border-radius:2px}`+
    `#wvlegend{display:flex;flex-wrap:wrap;align-items:center}`+
    `#wvhint{font-size:12px;color:#6f7d8f}`+
    `#wvbar .wvzb{background:#1f2733;color:#d7dde5;border:1px solid #3a4553;border-radius:3px;`+
    `padding:3px 9px;font-size:12px;cursor:pointer}#wvbar .wvzb:hover{background:#2a3340}`+
    `#wvclose{margin-left:auto;background:#1f2733;color:#d7dde5;border:1px solid #3a4553;`+
    `border-radius:3px;padding:4px 14px;font-size:13px;cursor:pointer}`+
    `#wvclose:hover{background:#2a3340}`+
    `#wvwrap{flex:1 1 auto;overflow:auto;min-height:0}#wvcanvas{display:block}`+
    `</style></head><body>`+
    `<header><h1>ISA debug view`+(split?` -- source-linked`:``)+`</h1>`+
    `<div class="sym">`+esc(payload.sym||fam)+`</div>`+
    `<div class="tot">`+payload.rows.length+` instructions &middot; `+
    payload.n_disp+` dispatch(es), ~1 SIMD &middot; `+
    fmtc(payload.stall)+` stall / `+fmtc(payload.lat)+` latency / `+
    fmtc(payload.idle)+` idle cyc`+
    (payload.occ?`<button id="wvbtn">Occupancy View</button>`:``)+`</div>`+
    (payload.has_src?``:`<div class="note">Source lines unavailable: the traced `+
      `code object has no DWARF line tables (build ggml-hip with `+
      `-gline-tables-only/-g and re-trace to link ISA to source). Showing ISA only.`+
      `</div>`)+
    (payload.exec?`<div id="stepbar"><button id="sprev">&#9664; Prev</button>`+
      `<button id="snext">Next &#9654;</button>`+
      `<button id="slprev">&#9664; src line</button>`+
      `<button id="slnext">src line &#9654;</button>`+
      `<span id="stepinfo"></span>`+
      `<span class="hint">one sampled wave, executed order &middot; keys: `+
      `&larr;/&rarr; step, H/L source-line</span></div>`:``)+
    `<input id="f" placeholder="filter instructions (e.g. s_waitcnt, global_load)">`+
    `</header>`+
    `<div id="main">`+
    `<section id="isapane"><div class="phdr">ISA (program order)`+
    (split?` <span class="hint">click a row to jump to its source line</span>`:``)+
    `<span class="hint">hover an instruction for its ISA description</span>`+
    `</div><div class="scroll"><table><thead><tr>`+
    (split?`<th class="src">source</th>`:``)+
    `<th>addr</th><th class="isa">ISA</th><th>hits</th><th>latency</th>`+
    `<th>stall</th><th>stall%</th><th>idle</th></tr></thead>`+
    `<tbody id="b"></tbody></table></div></section>`+
    (split?`<section id="srcpane"><div class="phdr">source: <select id="file">`+
      fileOpts+`</select><span class="hint">click a line to jump to its instructions</span>`+
      `</div><div class="scroll"><div id="src"></div></div></section>`:``)+
    `</div>`+
    (payload.occ?`<div id="wavepane"><div id="wvbar"><h2>Occupancy View</h2>`+
      `<button class="wvzb" id="wvzout">time &minus;</button>`+
      `<button class="wvzb" id="wvzin">time +</button>`+
      `<button class="wvzb" id="wvrout">rows &minus;</button>`+
      `<button class="wvzb" id="wvrin">rows +</button>`+
      `<button class="wvzb" id="wvfit">fit</button>`+
      `<span id="wvhint"></span>`+
      `<span id="wvlegend"></span>`+
      `<button id="wvclose">Close (Esc)</button></div>`+
      `<div id="wvwrap"><canvas id="wvcanvas"></canvas></div></div>`:``)+
    `<div id="tip"></div>`+
    `<scr`+`ipt>`+
    `var D=`+JSON.stringify(payload).replace(/</g,'\\u003c')+`;`+
    `function esc(s){return(''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}`+
    `function fmtc(n){n=+n||0;return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':(''+n);}`+
    `var SPLIT=!!D.split,SEL=null,GLOSS=D.gloss||{};`+
    // RDNA3.5 opcode glossary lookup: match the ISA mnemonic (first token), then
    // retry after stripping encoding suffixes the disassembler adds (_e32/_e64/dpp).
    `var GSFX=['_e64_dpp','_e32','_e64','_dpp8','_dpp','_sdwa'];`+
    `function mnemOf(isa){var s=(''+isa).trim();var i=s.search(/\\s/);`+
    `return (i<0?s:s.slice(0,i)).toLowerCase();}`+
    `function lookGloss(m){if(!m)return '';if(GLOSS[m])return GLOSS[m];`+
    `for(var i=0;i<GSFX.length;i++){var x=GSFX[i];`+
    `if(m.length>x.length&&m.slice(-x.length)===x){var b=m.slice(0,-x.length);`+
    `if(GLOSS[b])return GLOSS[b];}}return '';}`+
    // Special-register / wait-counter glossary: wrap recognized operand tokens in the
    // ISA text so hovering vmcnt/lgkmcnt/scc/exec/vcc/m0 etc. shows their meaning.
    `var REGG=D.regGloss||{};`+
    `var REGRX=(function(){var k=Object.keys(REGG);if(!k.length)return null;`+
    `k.sort(function(a,b){return b.length-a.length;});`+
    `return new RegExp('\\\\b('+k.join('|')+')\\\\b','gi');})();`+
    `function annotIsa(isa){var s=esc(isa);if(!REGRX)return s;`+
    `return s.replace(REGRX,function(m){var d=REGG[m.toLowerCase()];`+
    `return d?'<span class=reg data-reg=\"'+m.toLowerCase()+'\">'+m+'</span>':m;});}`+
    `var STEP=!!D.exec,ES=D.exec?D.exec.stream:null,T0=D.exec?D.exec.t0:0,EX=null,POS2EX={};`+
    `if(ES){for(var ei=0;ei<ES.length;ei++){if(POS2EX[ES[ei][0]]==null)POS2EX[ES[ei][0]]=ei;}}`+
    `var LA={};for(var i=0;i<D.rows.length;i++){var r=D.rows[i];if(!r.src)continue;`+
    `var a=LA[r.src]||(LA[r.src]={st:0,hit:0,idle:0,lat:0});`+
    `a.st+=r.st||0;a.hit+=r.hit||0;a.idle+=r.idle||0;a.lat+=r.lat||0;}`+
    `var FILES=Object.keys(D.src_files||{});`+
    `function fstall(f){var s=0;for(var k in LA){if(k.indexOf(f+':')===0)s+=LA[k].st;}return s;}`+
    `FILES.sort(function(x,y){return fstall(y)-fstall(x);});`+
    `var curFile=FILES[0]||'';`+
    `function drawISA(q){q=(q||'').toLowerCase();var tot=D.stall||1,b=[];`+
    `for(var i=0;i<D.rows.length;i++){var r=D.rows[i];`+
    `if(q&&r.isa.toLowerCase().indexOf(q)<0)continue;`+
    `var pct=100*(r.st||0)/tot,bw=Math.round(60*(r.st||0)/(D.maxStall||1));`+
    `b.push('<tr data-idx='+i+' data-key=\"'+esc(r.src||'')+'\" class=\"'+(pct>=5?'hot':'')+'\">'+`+
    `(SPLIT?'<td class=src>'+esc(r.src||'')+'</td>':'')+`+
    `'<td class=a>0x'+(r.a||0).toString(16)+'</td>'+`+
    `'<td class=isa>'+annotIsa(r.isa)+'</td>'+`+
    `'<td>'+fmtc(r.hit)+'</td><td>'+fmtc(r.lat)+'</td>'+`+
    `'<td>'+fmtc(r.st)+'</td>'+`+
    `'<td>'+pct.toFixed(1)+'% <span class=bar style=\"width:'+bw+'px\"></span></td>'+`+
    `'<td>'+fmtc(r.idle)+'</td></tr>');}`+
    `document.getElementById('b').innerHTML=b.join('');bindISA();applyISASel(false);`+
    `if(STEP&&EX!=null)applyStepISA(false);}`+
    `function bindISA(){var rows=document.querySelectorAll('#b tr');`+
    `for(var i=0;i<rows.length;i++){rows[i].onclick=function(){`+
    `var idx=+this.getAttribute('data-idx');`+
    `if(ES&&POS2EX[idx]!=null){EX=POS2EX[idx];stepRender(true);}`+
    `if(SPLIT){var k=this.getAttribute('data-key');if(k)selectKey(k,true,false);}};}}`+
    `function renderSrc(){var lines=D.src_files[curFile]||[],mx=1;`+
    `for(var k in LA){if(k.indexOf(curFile+':')===0&&LA[k].st>mx)mx=LA[k].st;}`+
    `var tot=D.stall||1,h=[];`+
    `for(var L=1;L<=lines.length;L++){var key=curFile+':'+L,a=LA[key];`+
    `var pct=a?100*a.st/tot:0,bw=a?Math.round(48*a.st/mx):0;`+
    `h.push('<div class=\"sl'+(a&&pct>=3?' hotl':'')+'\" data-key=\"'+key+'\">'+`+
    `'<span class=sln>'+L+'</span>'+`+
    `'<span class=shb>'+(a?'<span class=sbar style=\"width:'+bw+'px\"></span>':'')+'</span>'+`+
    `'<span class=sst>'+(a?fmtc(a.st):'')+'</span>'+`+
    `'<span class=sc>'+esc(lines[L-1]||'')+'</span></div>');}`+
    `document.getElementById('src').innerHTML=h.join('');bindSrc();applySrcSel(false);`+
    `if(STEP&&EX!=null&&SPLIT)applyStepSrc(false);}`+
    `function bindSrc(){var ls=document.querySelectorAll('#src .sl');`+
    `for(var i=0;i<ls.length;i++){ls[i].onclick=function(){`+
    `selectKey(this.getAttribute('data-key'),false,true);};}}`+
    `function selectKey(key,scSrc,scIsa){SEL=key;`+
    `if(SPLIT){var f=key.split(':')[0];`+
    `if(f&&D.src_files[f]&&f!==curFile){curFile=f;var s=document.getElementById('file');`+
    `if(s)s.value=f;renderSrc();}applySrcSel(scSrc);}applyISASel(scIsa);}`+
    `function applySrcSel(scroll){var ls=document.querySelectorAll('#src .sl'),hit=null;`+
    `for(var i=0;i<ls.length;i++){if(SEL&&ls[i].getAttribute('data-key')===SEL){`+
    `ls[i].classList.add('selline');if(!hit)hit=ls[i];}else ls[i].classList.remove('selline');}`+
    `if(scroll&&hit)hit.scrollIntoView({block:'center'});}`+
    `function applyISASel(scroll){var rows=document.querySelectorAll('#b tr'),first=null;`+
    `for(var i=0;i<rows.length;i++){if(SEL&&rows[i].getAttribute('data-key')===SEL){`+
    `rows[i].classList.add('sel');if(!first)first=rows[i];}else rows[i].classList.remove('sel');}`+
    `if(scroll&&first)first.scrollIntoView({block:'center'});}`+
    // --- Step mode: walk one sampled wave's executed-instruction stream ---
    `function applyStepISA(scroll){var rows=document.querySelectorAll('#b tr'),hit=null;`+
    `var pos=(EX!=null&&ES)?ES[EX][0]:-1;`+
    `for(var i=0;i<rows.length;i++){if(+rows[i].getAttribute('data-idx')===pos){`+
    `rows[i].classList.add('step');if(!hit)hit=rows[i];}else rows[i].classList.remove('step');}`+
    `if(scroll&&hit)hit.scrollIntoView({block:'center'});}`+
    `function applyStepSrc(scroll){var ls=document.querySelectorAll('#src .sl'),hit=null;`+
    `var pos=(EX!=null&&ES)?ES[EX][0]:-1,key=(pos>=0&&D.rows[pos])?D.rows[pos].src:'';`+
    `for(var i=0;i<ls.length;i++){if(key&&ls[i].getAttribute('data-key')===key){`+
    `ls[i].classList.add('stepline');if(!hit)hit=ls[i];}else ls[i].classList.remove('stepline');}`+
    `if(scroll&&hit)hit.scrollIntoView({block:'center'});}`+
    `function stepReadout(){var el=document.getElementById('stepinfo');if(!el)return;`+
    `if(EX==null||!ES){el.textContent='';return;}`+
    `var cyc=ES[EX][1],dwell=(EX+1<ES.length)?ES[EX+1][1]-cyc:0,row=D.rows[ES[EX][0]]||{};`+
    `el.textContent='step '+(EX+1)+'/'+ES.length+'  @ +'+(cyc-T0)+' cyc  dwell '+dwell+' cyc'+`+
    `(row.src?'  '+row.src:'');}`+
    `function stepRender(scroll){if(EX==null||!ES)return;`+
    `var row=D.rows[ES[EX][0]]||{},key=row.src||'';`+
    `if(SPLIT&&key){var f=key.split(':')[0];`+
    `if(f&&D.src_files[f]&&f!==curFile){curFile=f;var s=document.getElementById('file');`+
    `if(s)s.value=f;renderSrc();}}`+
    `applyStepISA(scroll);if(SPLIT)applyStepSrc(scroll);stepReadout();}`+
    `function ensureUnfiltered(){var f=document.getElementById('f');`+
    `if(f&&f.value){f.value='';drawISA('');}}`+
    `function stepTo(k){if(!ES)return;EX=Math.max(0,Math.min(ES.length-1,k));`+
    `ensureUnfiltered();stepRender(true);}`+
    `function stepBy(d){if(!ES)return;stepTo(EX==null?0:EX+d);}`+
    `function stepLine(dir){if(EX==null||!ES)return;`+
    `var cur=D.rows[ES[EX][0]]?D.rows[ES[EX][0]].src:'';`+
    `for(var k=EX+dir;k>=0&&k<ES.length;k+=dir){`+
    `var s=D.rows[ES[k][0]]?D.rows[ES[k][0]].src:'';`+
    `if(s&&s!==cur){stepTo(k);return;}}}`+
    `drawISA('');`+
    // opcode hover tooltip: delegated on the (persistent) ISA tbody so it survives
    // re-render. pointer-events:none on #tip keeps row clicks/step working.
    `var TIP=document.getElementById('tip'),B=document.getElementById('b');`+
    `function hideTip(){if(TIP)TIP.style.display='none';}`+
    `function posTip(e){if(!TIP)return;var x=e.clientX+14,y=e.clientY+16;`+
    `var w=TIP.offsetWidth,h=TIP.offsetHeight;`+
    `if(x+w>innerWidth)x=Math.max(4,e.clientX-w-14);`+
    `if(y+h>innerHeight)y=Math.max(4,e.clientY-h-16);`+
    `TIP.style.left=x+'px';TIP.style.top=y+'px';}`+
    `if(B&&TIP){B.addEventListener('mouseover',function(e){`+
    `var rg=e.target&&e.target.closest?e.target.closest('span.reg'):null;`+
    `if(rg){var rk=rg.getAttribute('data-reg'),rd=REGG[rk];if(rd){`+
    `TIP.innerHTML='<b>'+esc(rk)+'</b> - '+esc(rd);TIP.style.display='block';posTip(e);return;}}`+
    `var td=e.target&&e.target.closest?e.target.closest('td.isa'):null;`+
    `if(!td){hideTip();return;}var m=mnemOf(td.textContent),g=lookGloss(m);`+
    `if(!g){hideTip();return;}`+
    `TIP.innerHTML='<b>'+esc(m)+'</b> - '+esc(g);TIP.style.display='block';posTip(e);});`+
    `B.addEventListener('mousemove',function(e){if(TIP.style.display==='block')posTip(e);});`+
    `B.addEventListener('mouseout',function(e){var to=e.relatedTarget;`+
    `if(!to||!to.closest||!to.closest('td.isa'))hideTip();});}`+
    `if(SPLIT){var fs=document.getElementById('file');`+
    `if(fs){fs.value=curFile;fs.onchange=function(){curFile=this.value;renderSrc();};}renderSrc();}`+
    `document.getElementById('f').addEventListener('input',function(e){drawISA(e.target.value);});`+
    `if(STEP){var _p=document.getElementById('sprev');if(_p)_p.onclick=function(){stepBy(-1);};`+
    `var _n=document.getElementById('snext');if(_n)_n.onclick=function(){stepBy(1);};`+
    `var _lp=document.getElementById('slprev');if(_lp)_lp.onclick=function(){stepLine(-1);};`+
    `var _ln=document.getElementById('slnext');if(_ln)_ln.onclick=function(){stepLine(1);};`+
    `document.addEventListener('keydown',function(e){var t=e.target;`+
    `if(t&&(t.id==='f'||t.tagName==='SELECT'||t.tagName==='INPUT'))return;var k=e.key;`+
    `if(k==='ArrowRight'||k==='ArrowDown'||k==='n'||k==='j'){stepBy(1);e.preventDefault();}`+
    `else if(k==='ArrowLeft'||k==='ArrowUp'||k==='p'||k==='k'){stepBy(-1);e.preventDefault();}`+
    `else if(k==='L'||k==='l'){stepLine(1);e.preventDefault();}`+
    `else if(k==='H'||k==='h'){stepLine(-1);e.preventDefault();}});`+
    `EX=0;stepRender(false);}`+
    // --- Occupancy View: RCV Global-View-style panel, FILTERED to the selected
    // kernel. occupancy.json samples wave scheduling across ALL CUs the trace saw;
    // each lane is (CU, SIMD, wave_id) and each interval is colored by which kernel
    // held the slot. Because one ATT capture spans several back-to-back dispatches,
    // we keep only the intervals whose kernel == the selected family and drop lanes
    // that never ran it -- so the view shows just this kernel's footprint over CUs.
    `if(D.occ){`+
    `var KN=D.occ.kernels||{};`+
    // shifted kernel ids whose name matches the selected family (same kernel may be
    // re-dispatched several times in the capture window -> several matching ids)
    `var TGT={};for(var kk in KN){if(KN[kk]&&KN[kk]===D.fam)TGT[kk]=1;}`+
    `var wpane=document.getElementById('wavepane'),wcan=document.getElementById('wvcanvas'),`+
    `wwrap=document.getElementById('wvwrap'),wbtn=document.getElementById('wvbtn'),`+
    `wclose=document.getElementById('wvclose'),whint=document.getElementById('wvhint');`+
    `var AX=26,ROWH=0,PLOTW=0,WOPEN=false,WROWS=[],ROWZ=1;`+
    // gutter columns mirror RCV's Global View, prefixed with a sequential group
    // ordinal (#) so the number of distinct SM groups shown is countable at a glance.
    // SA is not carried in ATT records (single-SA capture) so it renders 0.
    `var WCOLS=[['#','seq'],['SE','se'],['SA','sa'],['WG','cu'],['SM','sm']];`+
    `var CW=26,GUT=WCOLS.length*CW+10;`+
    `function AM(){return D.occ;}`+
    // Within a single lane (WG/SIMD/wave-slot) the kernel appears as one or more
    // contiguous resident RUNS separated by gaps (the slot drains, then another wave
    // of the same kernel is scheduled into it). We color each run by its ORDINAL in
    // that lane -- 1st run green, 2nd amber, 3rd green... -- so consecutive waves that
    // time-share a slot are visually distinct. Value codes: 0=background,1=even,2=odd.
    `var BURSTCOL=['#0d1117','#3fb950','#e8912a'];`+          // 0=bg,1=odd-run,2=even-run
    `function colorOf(v){return BURSTCOL[v]||'#30363d';}`+
    // Per lane, keep only THIS kernel's raw cycle intervals [s,e,run] where run is the
    // 1-based ordinal of the wave in that slot -> color alternates 1,2,1,2.. Because
    // intervals carry exact cycles (no bucketing), gaps of ANY size survive at any zoom.
    `var NBURST=0;`+
    `function laneWaves(iv){var out=[],run=0;`+
    `for(var r=0;r<iv.length;r++){if(TGT[iv[r][2]]){run++;`+
    `out.push([iv[r][0],iv[r][1],1+((run-1)%2)]);}}`+
    `if(run>NBURST)NBURST=run;return out;}`+
    `function regroup(){WROWS=[];var ls=D.occ.lanes.slice();`+
    `ls.sort(function(a,b){return (a.cu-b.cu)||(a.simd-b.simd)||(a.wv-b.wv);});`+
    `for(var i=0;i<ls.length;i++){var l=ls[i],w=laneWaves(l.iv);if(!w.length)continue;`+
    `WROWS.push({grp:l.cu+'/'+l.simd,coords:{se:0,sa:0,cu:l.cu,sm:l.simd,sl:-1,id:l.wv},`+
    `waves:w});}}`+
    `regroup();`+
    // data window in CYCLES (relative to occ t0): first wave start .. last wave end,
    // across all lanes -> default view fills the axis, axis reads 0 at the first wave.
    `var DC0=1e18,DC1=0;`+
    `for(var di=0;di<WROWS.length;di++){var ww=WROWS[di].waves;`+
    `if(ww.length){if(ww[0][0]<DC0)DC0=ww[0][0];if(ww[ww.length-1][1]>DC1)DC1=ww[ww.length-1][1];}}`+
    `if(DC1<=DC0){DC0=0;DC1=Math.max(1,D.occ.t1-D.occ.t0);}`+
    // zoom/pan window over CYCLES; V0,V1 are cycle offsets from DC0-anchored origin.
    `var WV0=DC0,WV1=DC1,WMINC=8,wpan=false,wpx=0,wpv0=0,wpv1=0;`+
    `function clampWV(){var s=WV1-WV0;if(s<WMINC){var m=(WV0+WV1)/2;WV0=m-WMINC/2;WV1=m+WMINC/2;}}`+
    `function cx(c){return GUT+PLOTW*(c-WV0)/(WV1-WV0);}`+
    `function zoomWV(frac,factor){var s=WV1-WV0,ns=s*factor,ft=WV0+frac*s;`+
    `WV0=ft-frac*ns;WV1=ft+(1-frac)*ns;clampWV();drawWaves();}`+
    `function fmtk(v){return v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':(''+Math.round(v));}`+
    `function drawWaves(){var dpr=window.devicePixelRatio||1,W=wwrap.clientWidth||900,`+
    `n=WROWS.length;`+
    `var baseH=Math.max(3,Math.min(26,Math.floor((wwrap.clientHeight-AX)/Math.max(1,n))));`+
    `ROWH=Math.max(2,Math.min(40,Math.round(baseH*ROWZ)));`+
    `var H=AX+n*ROWH;PLOTW=W-GUT-14;if(PLOTW<50)PLOTW=50;`+
    `wcan.style.width=W+'px';wcan.style.height=H+'px';`+
    `wcan.width=Math.floor(W*dpr);wcan.height=Math.floor(H*dpr);`+
    `var g=wcan.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);`+
    `g.clearRect(0,0,W,H);g.font='11px ui-monospace,Menlo,Consolas,monospace';`+
    `g.textBaseline='middle';`+
    // header band: column titles over the gutter, absolute-cycle ticks over the plot
    `g.fillStyle='#161b22';g.fillRect(0,0,W,AX);`+
    `g.fillStyle='#c8d0da';g.textAlign='center';`+
    `for(var c=0;c<WCOLS.length;c++)g.fillText(WCOLS[c][0],c*CW+CW/2,AX/2);`+
    `var ND=6;for(var t=0;t<=ND;t++){var fx=GUT+PLOTW*t/ND;`+
    `g.fillStyle='#3a4553';g.fillRect(Math.round(fx),AX,1,H-AX);`+
    `g.fillStyle='#9fb0c4';g.textAlign=(t===0?'left':(t===ND?'right':'center'));`+
    `g.fillText(fmtk(WV0+(WV1-WV0)*t/ND-DC0),fx,AX/2);}`+
    // gutter coords drawn ONCE per group (centered), bars per lane.
    `function gut(row,y0,y1,seq){if(!row.coords)return;var cc=row.coords,`+
    `cv=[seq,cc.se,cc.sa,cc.cu,cc.sm];g.textAlign='center';`+
    `var yc=(y0+y1)/2;if(y1-y0<9)return;`+
    `for(var c=0;c<WCOLS.length;c++){g.fillStyle=(c===0?'#7fd0ff':'#c8d0da');`+
    `g.fillText(''+cv[c],c*CW+CW/2,yc);}}`+
    `var gs=0,gseq=0;`+
    `for(var i=0;i<n;i++){var row=WROWS[i],y=AX+i*ROWH,wl=row.waves;`+
    `if(i>0&&row.grp!==WROWS[i-1].grp){gut(WROWS[gs],AX+gs*ROWH,y,gseq);gseq++;`+
    `g.strokeStyle='#2a3340';g.beginPath();g.moveTo(0,y+0.5);g.lineTo(W,y+0.5);g.stroke();gs=i;}`+
    `for(var wi=0;wi<wl.length;wi++){var s=wl[wi][0],e=wl[wi][1],col=wl[wi][2];`+
    `var x0=cx(s),x1=cx(e);`+
    `if(x1>GUT&&x0<GUT+PLOTW){if(x0<GUT)x0=GUT;if(x1>GUT+PLOTW)x1=GUT+PLOTW;`+
    `g.fillStyle=colorOf(col);g.fillRect(x0,y+1,Math.max(1,x1-x0),Math.max(1,ROWH-1));}}}`+
    `gut(WROWS[gs],AX+gs*ROWH,AX+n*ROWH,gseq);`+
    `g.strokeStyle='#3a4553';g.beginPath();g.moveTo(GUT+0.5,0);g.lineTo(GUT+0.5,H);g.stroke();}`+
    `function renderLegend(){var el=document.getElementById('wvlegend');if(!el)return;`+
    `var h='<span class="lg"><span class="sw" style="background:'+BURSTCOL[1]+`+
    `'"></span>wave 1/3/5..</span>';`+
    `if(NBURST>1)h+='<span class="lg"><span class="sw" style="background:'+BURSTCOL[2]+`+
    `'"></span>wave 2/4/6..</span>';el.innerHTML=h;}`+
    `function syncChrome(){`+
    `whint.textContent='up to '+NBURST+' wave'+(NBURST!==1?'s':'')+'/slot; '+`+
    `WROWS.length+' lanes (WG x SIMD x wave-slot); scroll=time zoom, alt+scroll=row zoom, `+
    `shift+scroll=pan rows, drag=pan';renderLegend();}`+
    `function openWaves(){wpane.classList.add('show');WOPEN=true;WV0=DC0;WV1=DC1;`+
    `syncChrome();drawWaves();}`+
    `function closeWaves(){wpane.classList.remove('show');WOPEN=false;hideTip();}`+
    `if(wbtn)wbtn.onclick=openWaves;if(wclose)wclose.onclick=closeWaves;`+
    `window.addEventListener('resize',function(){if(WOPEN)drawWaves();});`+
    `var wzi=document.getElementById('wvzin'),wzo=document.getElementById('wvzout'),`+
    `wri=document.getElementById('wvrin'),wro=document.getElementById('wvrout'),`+
    `wfit=document.getElementById('wvfit');`+
    `function rowZoom(f){ROWZ=Math.max(0.25,Math.min(8,ROWZ*f));drawWaves();}`+
    `if(wzi)wzi.onclick=function(){zoomWV(0.5,0.6);};`+
    `if(wzo)wzo.onclick=function(){zoomWV(0.5,1/0.6);};`+
    `if(wri)wri.onclick=function(){rowZoom(1.4);};`+
    `if(wro)wro.onclick=function(){rowZoom(1/1.4);};`+
    `if(wfit)wfit.onclick=function(){WV0=DC0;WV1=DC1;ROWZ=1;drawWaves();};`+
    `if(wcan){wcan.addEventListener('wheel',function(e){`+
    `if(e.shiftKey){wwrap.scrollTop+=e.deltaY;e.preventDefault();return;}`+
    `if(e.altKey){rowZoom(e.deltaY<0?1.15:1/1.15);e.preventDefault();return;}`+
    `var rect=wcan.getBoundingClientRect(),mx=e.clientX-rect.left;`+
    `var frac=(mx-GUT)/PLOTW;if(frac<0)frac=0;if(frac>1)frac=1;`+
    `zoomWV(frac,e.deltaY<0?0.85:1/0.85);e.preventDefault();},{passive:false});`+
    `wcan.addEventListener('mousedown',function(e){var rect=wcan.getBoundingClientRect();`+
    `if(e.clientX-rect.left<GUT)return;wpan=true;wpx=e.clientX;wpv0=WV0;wpv1=WV1;`+
    `wcan.style.cursor='grabbing';hideTip();e.preventDefault();});`+
    `window.addEventListener('mousemove',function(e){if(!wpan)return;`+
    `var dc=-(e.clientX-wpx)/PLOTW*(wpv1-wpv0);WV0=wpv0+dc;WV1=wpv1+dc;clampWV();drawWaves();});`+
    `window.addEventListener('mouseup',function(){if(wpan){wpan=false;wcan.style.cursor='grab';}});`+
    `wcan.style.cursor='grab';`+
    `wcan.addEventListener('mousemove',function(e){`+
    `if(wpan){hideTip();return;}`+
    `var rect=wcan.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;`+
    `if(my<AX||mx<GUT){hideTip();return;}`+
    `var i=Math.floor((my-AX)/ROWH);if(i<0||i>=WROWS.length){hideTip();return;}`+
    `var c=WV0+(mx-GUT)/PLOTW*(WV1-WV0);`+
    `var row=WROWS[i],cc=row.coords,nm='(not resident)',run=0;`+
    `for(var wi=0;wi<row.waves.length;wi++){run++;`+
    `if(c>=row.waves[wi][0]&&c<row.waves[wi][1]){`+
    `nm='wave #'+run+' of '+esc(D.fam)+' ('+(row.waves[wi][1]-row.waves[wi][0])+' cyc)';break;}}`+
    `var h='<b>WG:'+cc.cu+' SIMD:'+cc.sm+' wave_id:'+cc.id+'</b>'+`+
    `'<br>'+nm+' @ '+fmtk(c-DC0)+' cyc';`+
    `TIP.innerHTML=h;TIP.style.display='block';posTip(e);});`+
    `wcan.addEventListener('mouseout',function(){hideTip();});}`+
    `document.addEventListener('keydown',function(e){if(e.key==='Escape'&&WOPEN)closeWaves();});`+
    `}`+
    `<\/scr`+`ipt></body></html>`;
  w.document.open(); w.document.write(doc); w.document.close();
}

// POST to the companion server to run ATT on a free GPU board and fold the result in
async function traceKernelLive(sym, fam){
  const btn=document.getElementById('atttrace');
  const st=document.getElementById('attstatus');
  if(!btn||!st) return;
  btn.disabled=true; st.style.display='inline';
  st.style.color='#c8d0da'; st.textContent='dispatching to a free GPU board... (~30-90s)';
  try{
    const r=await fetch('/api/trace',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({kernel:sym})});
    const j=await r.json().catch(()=>({}));
    if(r.status===409){
      st.style.color='#e0b341';
      st.textContent=j.error||'a trace is already running -- try again shortly';
      btn.disabled=false; return;
    }
    if(!r.ok||!j.ok){
      st.style.color='#ff6b6b';
      st.textContent='trace failed: '+(j.error||('HTTP '+r.status));
      btn.disabled=false; return;
    }
    // fold every returned family into the cache (ATT captures all variants of the symbol)
    D.att_by_fam=D.att_by_fam||{};
    let n=0;
    for(const k in j.fam_stats){ D.att_by_fam[k]=j.fam_stats[k]; n++; }
    // fold the full per-instruction ISA so "Open Trace View" lights up too
    D.att_code_by_fam=D.att_code_by_fam||{};
    if(j.fam_code) for(const k in j.fam_code){ D.att_code_by_fam[k]=j.fam_code[k]; }
    st.style.color='#8fe388';
    st.textContent='traced on '+(j.host||'board')+' -- '+n+' famil'+(n===1?'y':'ies')+' folded in';
    renderSelectedKernel();
  }catch(e){
    st.style.color='#ff6b6b'; st.textContent='trace error: '+e; btn.disabled=false;
  }
}
tb.querySelectorAll('tr').forEach(tr=>{
  tr.onclick=()=>setSelection(selectedFam===tr.dataset.fam ? null : tr.dataset.fam);
});
// totals footer: dispatches/token, kernel-busy/token, inter-kernel gap/token, mean gap
{
  const tf = document.querySelector('#tbl tfoot');
  const totCountTok = D.summary.reduce((a,r)=>a+r.per_tok,0);
  const timePerTok = D.n_tokens_baked ? D.busy_ns/D.n_tokens_baked : 0;
  // CP Transition Gap/token = the non-kernel (host/launch/idle) time in a token.
  // Per-kernel GPU time is accurate under trace, but the trace inflates the SPAN
  // (host serialization + completion-signal latency), so span-busy OVERSTATES the
  // real gap by the profiling overhead. When a clean (untraced) tg throughput is
  // available, prefer [clean per-token wall] - [kernel time/token]: the clean wall
  // carries no profiling overhead and kernel time is trustworthy, so the difference
  // isolates the true gap. Fall back to the traced span-busy gap otherwise.
  const tracedGapPerTok = D.n_tokens_baked ? Math.max(0, D.span_ns - D.busy_ns)/D.n_tokens_baked : 0;
  const cleanTokNs = (D.clean_tps && D.clean_tps.tps) ? 1e9/D.clean_tps.tps : 0;
  const useClean = cleanTokNs>0;
  const gapPerTok = useClean ? Math.max(0, cleanTokNs - timePerTok) : tracedGapPerTok;
  const gapLabel = useClean ? 'clean total CP Transition Gap'
                            : 'total CP Transition Gap <span class="r">(traced span - busy)</span>';
  const gapsPerTok = totCountTok>1 ? totCountTok-1 : totCountTok;
  const avgGap = gapsPerTok>0 ? gapPerTok/gapsPerTok : 0;
  let rows =
    `<tr><td colspan="3">total count</td><td>${totCountTok.toFixed(1)}</td></tr>`+
    `<tr><td colspan="3">total kernel time</td><td>${fmtdur(timePerTok)}</td></tr>`+
    (useClean?`<tr><td colspan="3">clean token time (untraced)</td><td>${fmtdur(cleanTokNs)}</td></tr>`:``)+
    `<tr><td colspan="3">${gapLabel}</td><td>${fmtdur(gapPerTok)}</td></tr>`+
    `<tr><td colspan="3">avg CP Transition Gap/kernel</td><td>${fmtdur(avgGap)}</td></tr>`;
  if (D.has_bw){
    const totMB = D.summary.reduce((a,r)=>a+(r.mb_tok||0),0);
    rows +=
      `<tr><td colspan="3">DRAM read (all kernels, measured)</td><td>${totMB.toFixed(0)} MB</td></tr>`;
  }
  if (D.has_map){
    // Two effective-BW rooflines (over-fetch-immune; only order-mapped matvec
    // dispatches carry packed bytes):
    //  1) eff KERNEL BW% = packed weights / kernel time -- how well the matvecs
    //     use DRAM while the GPU is actually running them (excludes idle gaps).
    //  2) eff TOKEN BW% = (packed weights + KV cache) / token time -- the useful
    //     DRAM throughput per generated token end-to-end, so it charges the
    //     inter-kernel gaps too. Uses the clean (untraced) token time when known,
    //     else the traced span/token. KV cache re-read grows with context.
    let totPacked = 0;
    for(const s of D.gpu){ if(s.map) totPacked += s.map.packed||0; }
    const packedTok = D.n_tokens_baked ? totPacked/D.n_tokens_baked : 0;
    const kernBw = timePerTok ? packedTok/timePerTok : 0;  // bytes/ns == GB/s
    const tokTime = useClean ? cleanTokNs
                    : (D.n_tokens_baked ? D.span_ns/D.n_tokens_baked : 0);
    const kvTok = D.kv_bytes_per_tok||0;
    const tokBw = tokTime ? (packedTok+kvTok)/tokTime : 0;
    rows +=
      `<tr><td colspan="3">eff kernel BW% (packed weights / kernel time)</td>`+
      `<td>${kernBw.toFixed(0)} GB/s (${(kernBw/D.peak_bw_gbs*100).toFixed(0)}%)</td></tr>`+
      `<tr><td colspan="3">eff token BW%</td>`+
      `<td>${tokBw.toFixed(0)} GB/s (${(tokBw/D.peak_bw_gbs*100).toFixed(0)}%)</td></tr>`;
  }
  tf.innerHTML = rows;
}

// hover
const hv = document.getElementById('hover');
// Hit-test the drawn rects at (mx,my): exact containment first (topmost/last-drawn
// slice wins), then a nearest-rect fallback within HIT_SLOP px in the SAME lane.
// Sub-pixel-narrow slices draw only ~1px wide, so an exact-only test makes them
// nearly impossible to point at; the slop gives every slice a usable catch radius
// without shifting which slice wins where cells are wide enough to hit exactly.
const HIT_SLOP=4;
function hitTest(mx,my){
  for(let i=rects.length-1;i>=0;i--){const q=rects[i];
    if(mx>=q.x&&mx<=q.x+q.w&&my>=q.y&&my<=q.y+q.h) return q;}
  let best=null, bd=HIT_SLOP;
  for(let i=rects.length-1;i>=0;i--){const q=rects[i];
    if(my<q.y||my>q.y+q.h) continue;               // only rects in the pointed-at lane
    const d = mx<q.x ? q.x-mx : (mx>q.x+q.w ? mx-(q.x+q.w) : 0);
    if(d<bd){bd=d; best=q;}}
  return best;
}
cv.addEventListener('mousemove', e=>{
  if(dragging||rbActive||markDrag){hv.style.display='none';return;}
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  // cursor hint when hovering a marker line
  const wC=cv.clientWidth;
  cv.style.cursor = Math.min(Math.abs(mx-xOf(markA,wC)),Math.abs(mx-xOf(markB,wC)))<=6 ? 'ew-resize' : '';
  let hit=hitTest(mx,my);
  if(!hit){hv.style.display='none';return;}
  let html='';
  if(hit.type==='gpu'){
    const s=hit.p, fc=D.fam_counters[s.fam]||{};
    html=`<div class="k">${s.fam}</div>`+
      `<div class="r">dur ${fmtus(s.e-s.s)}</div>`+
      (s.map?`<div style="color:#ffd479">${s.map.role} [${s.map.K}x${s.map.trueN}] ${s.map.q}`+
        (s.map.padN?` +${s.map.padN} pad`:` (no pad)`)+
        (s.map.overfetch?`, ${s.map.overfetch}x fetch`:``)+`</div>`+
        (s.map.effbw?(m=>`<div style="color:${m}">effective BW ~ ${s.map.effbw} GB/s `+
          `(${s.map.effbw_pct}% of ${D.peak_bw_gbs} peak)</div>`)(isBwBound(s.fam)&&s.map.effbw_pct>0&&s.map.effbw_pct<80?'#ff6b6b':'#8fe388'):''):'')+
      ((D.has_bw && fc.bw_gbs && !s.map)?
        `<div style="color:#7fd1ff">achieved BW ~ ${fc.bw_gbs} GB/s `+
        `(${fc.bw_pct}% of ${D.peak_bw_gbs} peak), ${fc.kb_disp} KB/disp</div>`:'')+
      ((D.has_loadw && fc.loadw)?
        `<div class="r">load ${fmtLoads(fc.loadw.vector_loads)} vec `+
        `(dom ${fc.loadw.dominant_lane_bytes}B/lane)</div>`:'')+
      (D.has_pmc?`<div class="r">MemUnitBusy ${fc.mem}%  L2hit ${fc.l2}%</div>`+
        `<div class="r">Occ ${fc.occ}%  LDSbc ${fc.lds}  WrStall ${fc.wr}  Wav ${fc.wav}</div>`+
        `<div style="color:${D.colors[s.stall]};font-weight:600">\u2192 dominant stall: ${s.stall}</div>`
        :`<div class="r">(no PMC data)</div>`);
  } else if(hit.type==='layer'){
    const L=hit.p;
    html=`<div class="k">${L.name}</div><div class="r">layer span ${fmtus(L.e-L.s)}</div>`;
  } else if(hit.type==='phase'){
    const P=hit.p;
    html=`<div class="k">${P.name}</div><div class="r">phase span ${fmtus(P.e-P.s)}</div>`;
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
  else if(e.key==='Escape'){ if(selectedSlices||selectedSlice||selectedFam){ selectSlice(null); } }
});

// left-drag = rubber-band multi-select (lasso the GPU slices in the X range);
// shift+left-drag = pan; scroll / +- / arrows = zoom+pan.
let dragging=false, dragX=0, dv0=0, dv1=0;   // pan state
let rbActive=false, rbX0=0, rbY0=0, rbX1=0, rbCtrl=false;  // rubber-band state (+down y for click, +ctrl/cmd for add/toggle)
function tOf(px){ return view0 + px/cv.clientWidth*(view1-view0); }
// sorted list of every GPU + CPU slice boundary (ns) for marker edge-snapping
const SNAP_EDGES = (()=>{
  const s=new Set();
  for(const g of D.gpu){ s.add(g.s); s.add(g.e); }
  for(const c of D.cpu){ s.add(c.s); s.add(c.e); }
  return [...s].sort((a,b)=>a-b);
})();
const SNAP_PX = 8;                           // snap when a slice edge is within this many px
// snap a time (ns) to the nearest slice edge if it's within SNAP_PX on screen; the
// pixel test means snapping is coarse when zoomed out and precise when zoomed in.
function snapT(t){
  if(!SNAP_EDGES.length) return t;
  let lo=0, hi=SNAP_EDGES.length-1;
  while(lo<hi){ const mid=(lo+hi)>>1; if(SNAP_EDGES[mid]<t) lo=mid+1; else hi=mid; }
  let best=SNAP_EDGES[lo], bd=Math.abs(best-t);
  if(lo>0){ const d=Math.abs(SNAP_EDGES[lo-1]-t); if(d<bd){ best=SNAP_EDGES[lo-1]; bd=d; } }
  return (bd/(view1-view0)*cv.clientWidth <= SNAP_PX) ? best : t;
}
function overlay(){                          // paint the selection box atop draw()
  if(!rbActive) return;
  const gpuBot=PAD_T+CPU_H+GAP+GPU_H;
  const x=Math.min(rbX0,rbX1), wpx=Math.abs(rbX1-rbX0);
  ctx.save();
  ctx.fillStyle='rgba(143,227,136,0.18)';
  ctx.strokeStyle='rgba(143,227,136,0.9)'; ctx.lineWidth=1;
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
  else { rbActive=true; rbCtrl=e.ctrlKey||e.metaKey; rbX0=mx; rbY0=e.clientY-r.top; rbX1=mx; cv.style.cursor='cell'; }
});
// hit-test a point against the drawn rects; select the GPU slice's family (or clear).
// toggle=true (Ctrl/Cmd+click) adds/removes the hit slice from the multi-select set.
function clickSelect(mx,my,toggle){
  const q=hitTest(mx,my);
  if(q && q.type==='gpu'){
    if(toggle) toggleSlice(q.p);
    else selectSlice(selectedSlice===q.p ? null : q.p);
    return;
  }
  if(!toggle) selectSlice(null);   // plain click on empty clears; modifier-click keeps selection
}
window.addEventListener('mousemove', e=>{
  if(markDrag){
    const r=cv.getBoundingClientRect();
    const mx=Math.min(cv.clientWidth, Math.max(0, e.clientX-r.left));
    const t = e.altKey ? tOf(mx) : snapT(tOf(mx));  // Alt = free placement (no snap)
    if(markDrag===1) markA=t; else markB=t;
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
    if(Math.abs(rbX1-rbX0)>=4){
      selectBox(tOf(Math.min(rbX0,rbX1)), tOf(Math.max(rbX0,rbX1)), rbCtrl);  // drag = lasso (ctrl/cmd = add to selection)
    } else clickSelect(rbX0, rbY0, rbCtrl);   // no drag = click: select one kernel (ctrl/cmd = toggle in/out)
  }
});
// bring both markers back into the current viewport (button + double-click)
function markersToView(){
  markA=view0+(view1-view0)*0.33; markB=view0+(view1-view0)*0.66; draw();
}
document.getElementById('markhome').onclick=markersToView;
cv.addEventListener('dblclick', markersToView);

// --- Find: extensible "jump to X" registry ---------------------------------
// Each finder returns {t0,t1,label} (a time span to frame + a status string) or
// null. Add entries here + an <option> in #findWhat to grow the menu.
// The second of the two baked/displayed decode tokens: [tok_starts[view_i1-1],
// tok_starts[view_i1]). Finders scope to it because it is a complete token cycle
// (the first displayed token can be entered mid-relaunch), so a match maps to a
// real, self-contained token. Returns null if there aren't two token boundaries.
function secondTokenWin(){
  const ts=D.tok_starts||[];
  const hi=D.view_i1, lo=hi-1;
  if(lo<0 || hi>=ts.length) return null;
  return {t0:ts[lo], t1:ts[hi]};
}
// Identity of a slice as kernel + weight shape (role, [K x N], quant), so the
// same kernel/shape recurring across the model's ~40 layers collapses to one
// entry in the find list. Falls back to the family name for unmapped slices.
function shapeKey(s){
  const m=s.map;
  return m ? (s.fam+'#'+m.role+'#'+m.K+'x'+m.trueN+'#'+m.q) : s.fam;
}
function findMaxIntraTokenGap(){
  // Largest gap between consecutive GPU slices within the second decode token.
  // Scoped to one full token, so no token boundary can fall inside the window.
  const win=secondTokenWin();
  const g=[...D.gpu].filter(x=> !win || (x.s>=win.t0 && x.s<win.t1))
                    .sort((a,b)=>a.s-b.s);
  const dot=s=>`<span class="fam-dot" style="background:${D.colors[s.stall]||D.colors.unknown}"></span>`;
  // Dedup by the (bracketing kernel+shape) pair, keeping the largest instance of
  // each recurring gap type so the same edge does not repeat once per layer.
  const byKey=new Map();
  for(let i=0;i+1<g.length;i++){
    const gap=g[i+1].s-g[i].e;
    if(gap<=0) continue;
    const a=g[i], b=g[i+1];
    const key=shapeKey(a)+' -> '+shapeKey(b);
    const cur=byKey.get(key);
    if(cur && (cur.t1-cur.t0)>=gap) continue;   // already have a bigger instance
    const detail=`<h2>Find: intra-token gap</h2>`+
      `<div class="sub" style="margin-bottom:8px">GPU idle between two consecutive `+
      `kernels within the second (full) decode token.</div>`+
      `<table><tbody>`+
      `<tr><td>gap (GPU idle)</td><td><b>${fmtdur(gap)}</b></td></tr>`+
      `<tr><td>ending kernel (before gap)</td><td>${dot(a)}${a.fam} `+
        `<span class="sub">(${fmtus(a.e-a.s)})</span></td></tr>`+
      `<tr><td>starting kernel (after gap)</td><td>${dot(b)}${b.fam} `+
        `<span class="sub">(${fmtus(b.e-b.s)})</span></td></tr>`+
      `</tbody></table>`;
    byKey.set(key,{t0:a.e, t1:b.s, detail,
      label:`intra-token gap: ${fmtdur(gap)} (${a.fam} &rarr; ${b.fam})`});
  }
  // Ranked largest gap first; "next" walks toward smaller gaps.
  return [...byKey.values()].sort((x,y)=>(y.t1-y.t0)-(x.t1-x.t0));
}
function findMinEffBw(){
  // matvec/matmul dispatch with the lowest effective (useful-work) bandwidth =
  // theoretical packed weight bytes / kernel time. Low eff-BW = the kernel is
  // launch/latency-dominated relative to the data it moves, or is slow for its
  // shape -- the honest optimization target (over-fetch-immune by construction).
  // Ranked by the steady-state MEAN duration (kstats) when available so a single
  // once-per-token bubble does not skew the pick; falls back to this dispatch's
  // own time. Only order-mapped mmvq/mmq slices carry packed bytes.
  const isMM=f=>/mul_mat_vec|mul_mat_q|mmvq|mmq/i.test(f);
  const win=secondTokenWin();
  // Dedup by kernel+shape, keeping the lowest-eff-BW instance so a matvec that
  // recurs identically across layers shows once (at its worst case) in the list.
  const byKey=new Map();
  for(const s of D.gpu){
    if(win && (s.s<win.t0 || s.s>=win.t1)) continue;   // scope to the 2nd token
    if(!isMM(s.fam) || !s.map || !s.map.packed) continue;
    const ks=(D.has_kstats && D.kstats[s.ti+'|'+s.fam])||null;
    const durns=(ks&&ks.n>1)?ks.mean:(s.e-s.s);
    if(durns<=0) continue;
    const effbw=s.map.packed/durns;   // bytes/ns == GB/s
    const key=shapeKey(s);
    const cur=byKey.get(key);
    if(cur && cur.effbw<=effbw) continue;   // already have a lower-eff-BW instance
    const m=s.map, pct=effbw/D.peak_bw_gbs*100;
    const avg=(ks&&ks.n>1)?` (mean of ${ks.n})`:` (single dispatch)`;
    byKey.set(key,{effbw, t0:s.s, t1:s.e, select:s,
      label:`eff-BW matvec: ${effbw.toFixed(1)} GB/s `+
            `(${pct.toFixed(1)}% of peak)${avg} - ${m.role} L${m.L<0?'out':m.L}`});
  }
  // Ranked lowest eff-BW first; "next" walks toward higher eff-BW.
  return [...byKey.values()].sort((x,y)=>x.effbw-y.effbw);
}
const FINDERS={maxgap:findMaxIntraTokenGap, mineffbw:findMinEffBw};
// Each finder returns a list ranked most-significant first (largest gap / lowest
// eff-BW). findState walks that ranked list: "find" (or re-clicking it) steps to
// the next result, "prev" steps back; both wrap. Changing the finder resets it.
let findState={what:null, list:[], idx:0};
function applyFindResult(r){
  // Frame the span with ~1.5x padding on each side so the region is centered and
  // its bracketing kernels are visible; markers sit on the exact edges so the
  // measurement readout shows the width.
  const w=Math.max(r.t1-r.t0,1), pad=w*1.5;
  setView(r.t0-pad, r.t1+pad);
  markA=r.t0; markB=r.t1;
  if(r.select){
    // Reuse the full click-through detail panel (weight, shape, eff-BW, avg
    // duration, PMC, ...) for a found kernel; selectSlice() redraws.
    selectSlice(r.select);
  } else if(r.detail){
    // Show the bracketing kernels in the detail pane; clear any prior selection
    // so updateDetail() won't overwrite it until the user clicks a kernel next.
    selectedSlice=null; selectedSlices=null; selectedFam=null;
    tb.querySelectorAll('tr').forEach(tr=>tr.classList.remove('sel'));
    dp.innerHTML=r.detail; dp.style.display='block';
    draw();
  } else {
    draw();
  }
}
function runFind(dir){
  const msg=document.getElementById('findmsg');
  const what=document.getElementById('findWhat').value;
  if(findState.what!==what || !findState.list.length){
    const fn=FINDERS[what];
    findState.list = fn?fn():[];
    findState.what = what;
    findState.idx = 0;                 // fresh find starts at the top rank
  } else {
    const n=findState.list.length;
    findState.idx = ((findState.idx + dir) % n + n) % n;   // step + wrap
  }
  const list=findState.list;
  if(!list.length){ msg.textContent='nothing found'; return; }
  const r=list[findState.idx];
  applyFindResult(r);   // populates the detail pane (via selectSlice or r.detail)
  msg.textContent='';   // status lives in the detail pane now, not the toolbar
  const banner=`<div class="sub" style="margin-bottom:8px">`+
    `<b>find</b> #${findState.idx+1}/${list.length} &middot; ${r.label}</div>`;
  dp.insertAdjacentHTML('afterbegin', banner);
  dp.style.display='block';
}
document.getElementById('findGo').onclick=()=>runFind(1);
document.getElementById('findPrev').onclick=()=>runFind(-1);
document.getElementById('findWhat').onchange=()=>{
  findState.what=null; findState.list=[]; findState.idx=0;
  document.getElementById('findmsg').textContent='';};

// RDNA 3.5 hardware-reference modal (embedded PNG data-URI; button hidden if the
// diagram was not baked into the payload).
(function(){
  const modal=document.getElementById('hwmodal');
  if(D.hw_diagram){
    document.getElementById('hwimg').src=D.hw_diagram;
    const btn=document.getElementById('hwbtn');
    btn.style.display='';
    const open=()=>{ modal.style.display='flex'; };
    const close=()=>{ modal.style.display='none'; };
    btn.onclick=open;
    document.getElementById('hwclose').onclick=close;
    modal.addEventListener('click', e=>{ if(e.target===modal) close(); });
    window.addEventListener('keydown', e=>{ if(e.key==='Escape' && modal.style.display!=='none'){ e.stopPropagation(); close(); } }, true);
  }
})();

resize();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()

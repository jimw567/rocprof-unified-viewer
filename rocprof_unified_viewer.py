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
import copy
import csv
import datetime
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
from collections import defaultdict

from regimes import regime_for

# Generator version. Bump alongside pyproject.toml's [project] version when the
# output format or a user-visible behavior changes -- it is stamped into every
# overlay (payload["provenance"]) so a shared HTML self-identifies what produced it.
__version__ = "0.1.0"

try:
    from isa_glossary import ISA_GLOSSARY, REG_GLOSSARY, CONCEPT_GLOSSARY
except ImportError:
    ISA_GLOSSARY = {}
    REG_GLOSSARY = {}
    CONCEPT_GLOSSARY = {}


def _provenance():
    """Build-provenance stamp embedded in every overlay so a shared HTML says exactly
    what produced it (version, git commit, when, which machine). All best-effort: a
    loose script outside a git checkout still generates, just with git_sha 'unknown'."""
    sha = "unknown"
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip() or "unknown"
        dirty = subprocess.check_output(
            ["git", "-C", here, "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip()
        if dirty:
            sha += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        host = socket.gethostname()
    except OSError:
        host = "unknown"
    return {
        "version": __version__,
        "git_sha": sha,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%d %H:%M UTC"),
        "host": host,
        "python": platform.python_version(),
    }


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
    29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}

# Roofline peaks, stall thresholds, ggml type tables, and the family_of/dominant_stall
# hot helpers now live in common.py (the shared dependency hub). Re-exported here so
# existing importers (serve.py, tests) keep working at this module path.
from common import (  # noqa: E402
    _GGML_TYPES, _GGML_BLOCK, PEAK_BW_GBS_BY_ARCH, PEAK_TOPS_BY_ARCH, MMQ_Y_BY_ARCH,
    DEFAULT_ARCH, MEM_BUSY_HI, L2_HIT_LO, LDS_CONFLICT_HI, OCC_LO, STALL_COLORS,
    PMC_COUNTERS, peak_bw_for, mmq_y_for, peak_tops_for, dominant_stall, family_of,
    gguf_packed_bytes as _gguf_packed_bytes)


# Per-MODEL-architecture token-boundary segmentation profile. NOTE: keyed by the GGUF
# `general.architecture` (qwen35moe, gpt-oss, ...), NOT the GPU arch (gfx1151) used by the
# dicts above -- decode cadence is a property of the model's graph, not the board. Model
# architectures move far slower than this viewer, so hardcoding per-arch here is fine and
# is exactly what keeps a tweak for one arch from perturbing another.
#   method: "head" = output-head anchor only (robust for eager MoE, where the gap
#                    detector finds no clean cadence); "gap" = inter-dispatch gap detector
#                    at gap_us; "auto" = head first, gap fallback (the historical default).
#   gap_us: threshold (us) for the gap detector; None = use --gap-threshold-us / its default.
BOUNDARY_PROFILES = {
    "qwen35moe": {"method": "head", "gap_us": None},   # hybrid GDN/MoE, runs eager
    "qwen3moe":  {"method": "head", "gap_us": None},    # MoE, runs eager
    "gpt-oss":   {"method": "gap",  "gap_us": 10.0},    # head-anchor under-segments; needs a low gap
    "_default":  {"method": "auto", "gap_us": None},    # unknown/dense: head then gap fallback
}


def boundary_profile_for(arch):
    return BOUNDARY_PROFILES.get(arch or "", BOUNDARY_PROFILES["_default"])


    if occ < OCC_LO:
        return "occupancy"
    return "compute"


# --- CSV loaders (stdlib only; duplicated on purpose so this file is standalone) --

def load_kernel_slices(path, mmq_y=64):
    """Return {stream_id: [(start_ns, end_ns, kernel_name, N, nblk, gy), ...] sorted
    by start}. N = Grid_Size_X / Workgroup_Size_X is the launched output-row count
    (one warp/workgroup-row per output row for mul_mat_vec), the join key onto the
    GGUF weight's true N (ne[1]); 0 when the grid dims are absent/degenerate.
    mmq_y is the MMQ prefill row-tile height used to recover N for mul_mat_q.
    gy = grid.y block count: for a MoE mul_mat_id dispatch this is n_experts_used
    (the experts run in one grouped launch over grid.y), so gy>1 flags MoE and lets
    the order-map expect ONE grouped expert dispatch instead of desyncing on it."""
    by_stream = defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            kname = r["Kernel_Name"]
            try:
                gx = int(r["Grid_Size_X"])
                wg = int(r["Workgroup_Size_X"])
                n = gx // wg if wg else 0
                # The wvsplitk_q8_0_longk kernel launches grid.x = nrows_x WORKGROUPS (one
                # output row each) with block = (warp_size, NW); rocprofv3 reports Grid_Size_X
                # in WORK-ITEMS, so Grid_Size_X/Workgroup_Size_X = nrows_x exactly. N is that.
                if "wvsplitk_q8_0_longk" in kname:
                    pass  # n = gx//wg already equals nrows_x (one workgroup per row)
                # The other wvsplitk kernels use a 2D block (warp_size x WvPrGrp) and
                # grid.x = ceil(nrows / (YTILE*WvPrGrp)) workgroups, so
                # Grid_Size_X/Workgroup_Size_X = grid.x, and the true output-row count is
                # grid.x * WvPrGrp (YTILE=1 for the decode kernels). WvPrGrp is the block's
                # Y dimension == Workgroup_Size_Y, so read it PER DISPATCH rather than
                # hardcoding it: the Q4_K kernel uses WvPrGrp=16 but the Q8_0 kernel uses
                # WvPrGrp=8, and a fixed *16 tagged every Q8_0 dispatch's N at 2x its true
                # value (e.g. N=4096 attn_gate read as 8192), cascading a wrong weight ->
                # dispatch order-map, effbw, and over-fetch. Grid_Size_X is in WORK-ITEMS
                # so grid.x = gx/warp_size(=Workgroup_Size_X); multiply by WvPrGrp.
                elif "wvsplitk" in kname:
                    try:
                        wvpg = int(r["Workgroup_Size_Y"])
                    except (KeyError, ValueError, TypeError):
                        wvpg = 16   # legacy fallback (old Q4_K-only assumption)
                    n *= wvpg if wvpg else 16
                    # k512_fast uses YTILE=2 (2 output rows per wave), so grid.x =
                    # ceil(N/(YTILE*WvPrGrp)); the *WvPrGrp above recovers only N/YTILE.
                    # No YTILE column in the trace -> key off the kernel name.
                    if "k512_fast" in kname:
                        n *= 2
                # The mul_mat_q PREFILL GEMM tiles the N output rows in blocks of
                # mmq_y along grid.x (nty = ceil(nrows_x/mmq_y)); recover launched N
                # as (grid.x/wg.x)*mmq_y so it order-maps onto its GGUF weight just
                # like decode's mmvq. (Small-N weights round up to a full mmq_y tile,
                # so the recovered N shows the row-padding, which is real.)
                elif "mul_mat_q" in kname:
                    n *= mmq_y
            except (KeyError, ValueError, TypeError):
                n = 0
            # grid.y block count. For mul_mat_id the experts (or channels) tile along
            # grid.y, so gy = n_experts_used (e.g. 8); dense matvecs have gy = 1.
            try:
                gyd = int(r["Grid_Size_Y"]); wgy = int(r["Workgroup_Size_Y"])
                gy = (gyd // wgy) if wgy else 1
            except (KeyError, ValueError, TypeError):
                gy = 1
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
                 kname, n, nblk, gy))
    for evs in by_stream.values():
        evs.sort()
    return by_stream


_HIP_NAME_COLS = ("Function", "Api_Name", "Name", "Operation")


def graph_launch_starts(path):
    """Return the sorted start timestamps of every hipGraphLaunch in a HIP-API trace.
    In graph mode each measured forward pass IS one hipGraphLaunch, so consecutive launch
    timestamps bracket exactly one pass -- the clean way to isolate a single prefill pass
    (far more robust than guessing from GPU inter-dispatch gaps). [] if none / no file."""
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        namecol = next((c for c in _HIP_NAME_COLS if c in (reader.fieldnames or [])), None)
        if not namecol:
            return []
        for r in reader:
            if "GraphLaunch" in (r.get(namecol) or ""):
                try:
                    out.append(int(r["Start_Timestamp"]))
                except (KeyError, ValueError, TypeError):
                    pass
    return sorted(out)


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
                    # true N = grid.x * WvPrGrp. gs//ws = grid.x; WvPrGrp is the block Y
                    # dim, not present as a column here, but the total block size ws =
                    # warp_size(32) * WvPrGrp, so WvPrGrp = ws/32. A hardcoded *16 assumed
                    # the Q4_K kernel's WvPrGrp=16 and mis-tagged Q8_0 (WvPrGrp=8) N at 2x.
                    # (Must match the N-recovery in load_kernel_slices.)
                    wvpg = (ws // 32) if ws else 16
                    n *= wvpg if wvpg else 16
                    if "k512_fast" in r["Kernel_Name"]:   # YTILE=2 (see load_kernel_slices)
                        n *= 2
            except (KeyError, ValueError, TypeError):
                n = 0
            if n:
                agg_n[(fam, n)].append(v)
    by_fam = {fam: statistics.mean(v) * 1024.0 for fam, v in agg.items() if v}
    by_fam_n = {k: statistics.mean(v) * 1024.0 for k, v in agg_n.items() if v}
    # Per-(fam,N) fetch-size CLUSTERS. Two different-K weights can share an N (e.g.
    # ffn_down_shexp K=512 and attn_output K=4096 both dispatch N=2048), so the
    # (fam,N) mean blends their very different DRAM reads (1.1MB vs 8.9MB) -> a
    # physically-impossible over-fetch (4.5x on the small weight). But each weight's
    # own fetch is tightly clustered, so 1-D clustering the per-dispatch fetch values
    # recovers the distinct K reads without needing a complete order-map. The consumer
    # then picks the cluster whose center is nearest a weight's packed footprint.
    by_fam_n_clusters = {}
    for (fam, n), vals in agg_n.items():
        vb = sorted(v * 1024.0 for v in vals)     # bytes
        if not vb:
            continue
        # split where a gap exceeds max(64KB, 25% of the running center) -> tight modes
        clusters = [[vb[0]]]
        for x in vb[1:]:
            c = clusters[-1]
            center = c[len(c) // 2]
            if x - c[-1] > max(65536.0, 0.25 * center):
                clusters.append([x])
            else:
                c.append(x)
        by_fam_n_clusters[(fam, n)] = [statistics.mean(c) for c in clusters]
    return by_fam, by_fam_n, by_fam_n_clusters


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
                    # true N = grid.x * WvPrGrp. gs//ws = grid.x; WvPrGrp is the block Y
                    # dim, not present as a column here, but the total block size ws =
                    # warp_size(32) * WvPrGrp, so WvPrGrp = ws/32. A hardcoded *16 assumed
                    # the Q4_K kernel's WvPrGrp=16 and mis-tagged Q8_0 (WvPrGrp=8) N at 2x.
                    # (Must match the N-recovery in load_kernel_slices.)
                    wvpg = (ws // 32) if ws else 16
                    n *= wvpg if wvpg else 16
                    if "k512_fast" in r["Kernel_Name"]:   # YTILE=2 (see load_kernel_slices)
                        n *= 2
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


def parse_clean_tps(path, kind="tg"):
    """Parse throughput from collect.sh's clean_tps.txt (the untraced llama-bench
    run). Returns {"test": "tg64"/"pp128", "tps": float, "sd": float or None} for
    the last matching row, or None if missing/unparseable. kind selects the row
    family: "tg" (decode) or "pp" (prefill). This is the honest tok/s -- rocprofv3
    perturbs the traced runs.

    Prefers llama-bench JSON (-o json): takes the MEDIAN of samples_ts, exactly like
    the llamacpp regression harness (statistics.median(samples_ts)) -- more stable
    than a single -r 1 sample, whose first rep is often a cold-cache outlier (this is
    why an -r 1 clean run reported a pp ~25%% low vs the regression's -r 3 median).
    Falls back to the markdown table (avg +/- sd) for old text-format files."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    # llama-bench -o json: a list of per-(prompt,gen) rows carrying samples_ts.
    stripped = text.lstrip()
    if stripped[:1] in "[{":
        try:
            rows = json.loads(stripped)
        except ValueError:
            rows = None
        if rows:
            want_gen = kind == "tg"   # tg rows have n_gen>0; pp rows n_gen==0
            best = None
            for r in rows:
                if bool(int(r.get("n_gen", 0))) != want_gen:
                    continue
                ts = [float(x) for x in (r.get("samples_ts") or [])]
                if ts:
                    tps = statistics.median(ts)
                    sd = statistics.pstdev(ts) if len(ts) > 1 else 0.0
                elif r.get("avg_ts") is not None:
                    tps = float(r["avg_ts"])
                    sd = float(r.get("stddev_ts") or 0.0)
                else:
                    continue
                n = int(r.get("n_prompt", 0)) if kind == "pp" \
                    else int(r.get("n_gen", 0))
                best = {"test": "%s%d" % (kind, n), "tps": tps, "sd": sd}
            return best
    # Markdown table fallback (older clean_tps.txt without -o json).
    best = None
    row_re = re.compile(r"%s\d+" % kind)
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        test = next((c for c in cells if row_re.fullmatch(c)), None)
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


def parse_clean_model_name(path):
    """Best-effort model name from collect.sh's clean_tps.txt (llama-bench JSON).
    Prefers the gguf basename (minus .gguf) from `model_filename`; falls back to the
    `model_type` label llama-bench prints (e.g. "qwen35moe 35B.A3B Q4_K - Medium").
    Returns "" if unavailable. Lets the overlay title carry the model even when the
    caller did not pass --gguf (collect.sh always emits clean_tps.txt)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            stripped = fh.read().lstrip()
    except OSError:
        return ""
    if stripped[:1] not in "[{":
        return ""
    try:
        rows = json.loads(stripped)
    except ValueError:
        return ""
    row = (rows[0] if isinstance(rows, list) and rows else
           rows if isinstance(rows, dict) else None)
    if not row:
        return ""
    fn = row.get("model_filename") or ""
    if fn:
        base = os.path.basename(fn)
        return base[:-5] if base.endswith(".gguf") else base
    return row.get("model_type") or ""


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


def shortcuts_help_html(uid, title, sections):
    """Return a self-contained mouse/key shortcut helper: a small round "?" button
    plus a modal listing the shortcuts, with inline styles + an IIFE that wires
    open/close/Esc/backdrop-click. No shared CSS needed, so the SAME markup drops
    into the main overlay AND the child debug window (separate documents).

    uid      -- unique element-id prefix (docs may coexist; keep ids distinct).
    title    -- modal heading.
    sections -- [(section_name, [(keys, description), ...]), ...]. `keys` is shown
                in a monospace chip column; both are plain text (HTML-escaped here).
    """
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
    rows = []
    for name, items in sections:
        rows.append(
            '<div style="margin:10px 0 4px;color:#7fd1ff;font-size:12px;'
            'text-transform:uppercase;letter-spacing:.5px">%s</div>' % esc(name))
        rows.append('<table style="width:100%;border-collapse:collapse">')
        for keys, desc in items:
            rows.append(
                '<tr>'
                '<td style="padding:3px 12px 3px 0;white-space:nowrap;'
                'vertical-align:top"><span style="font-family:ui-monospace,'
                'Menlo,Consolas,monospace;background:#1b2130;border:1px solid '
                '#2a3340;border-radius:3px;padding:1px 6px;color:#dbe6f5;'
                'font-size:12px">%s</span></td>'
                '<td style="padding:3px 0;color:#c8d0da;font-size:13px">%s</td>'
                '</tr>' % (esc(keys), esc(desc)))
        rows.append("</table>")
    body = "".join(rows)
    b, m = uid + "Btn", uid + "Modal"
    return (
        '<button id="%s" title="mouse & keyboard shortcuts" '
        'style="cursor:pointer;width:24px;height:24px;border-radius:50%%;'
        'background:#1f2733;color:#dbe6f5;border:1px solid #3a4553;'
        'font-size:14px;line-height:1;padding:0">?</button>'
        '<div id="%s" style="display:none;position:fixed;inset:0;z-index:10000;'
        'background:rgba(0,0,0,.72);align-items:center;justify-content:center;'
        'padding:24px">'
        '<div style="position:relative;max-width:640px;max-height:88vh;overflow:auto;'
        'background:#0d1017;border:1px solid #2a2f3a;border-radius:6px;padding:14px 16px">'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        '<b style="color:#dbe6f5;font-size:15px">%s</b>'
        '<button id="%sClose" style="cursor:pointer;background:#1f2733;color:#d7dde5;'
        'border:1px solid #3a4553;border-radius:3px;padding:2px 10px">Close &times;</button>'
        '</div>%s</div></div>'
        "<scr" + "ipt>(function(){var b=document.getElementById('%s'),"
        "m=document.getElementById('%s'),c=document.getElementById('%sClose');"
        "if(!b||!m)return;function op(){m.style.display='flex';}"
        "function cl(){m.style.display='none';}b.onclick=op;if(c)c.onclick=cl;"
        "m.addEventListener('click',function(e){if(e.target===m)cl();});"
        "window.addEventListener('keydown',function(e){"
        "if(e.key==='Escape'&&m.style.display!=='none'){e.stopPropagation();cl();}"
        "else if((e.key==='?'||(e.key==='/'&&e.shiftKey))&&m.style.display==='none'){"
        "var t=e.target;if(t&&(t.tagName==='INPUT'||t.tagName==='SELECT'||"
        "t.tagName==='TEXTAREA'))return;e.preventDefault();op();}},true);})();"
        "</scr" + "ipt>"
    ) % (b, m, esc(title), uid, body, b, m, uid)


def load_loadwidth(path):
    """Load the disassembly load-width JSON ({family: {vector_loads, scalar_loads,
    lds_loads, dominant_lane_bytes, ...}}) produced from the gfx1151 device code
    objects. Keyed by the same family_of() names, so it joins onto slices."""
    with open(path) as fh:
        return json.load(fh)


_LAYER_BLK_RE = re.compile(r"^blk\.(\d+)\.")
_LAYER_SUFFIX_RE = re.compile(r"[-_]l?(\d+)(?:\s|\(|$)")


def _layer_of_name(name):
    """Recover the decoder block index from a ggml node name. Two conventions appear:
    GGUF tensor names use the `blk.<N>.` prefix (e.g. blk.3.attn_q.weight); the runtime
    cgraph cb() labels use a `-<N>` / `_l<N>` suffix (e.g. Qcur-23, ffn_moe_down-23,
    cache_k_l23). Returns the block index, or -1 for non-block nodes (norm, output).
    NOTE: ggml names UNNAMED ops `node_<N>` where N is a global node id, NOT a layer --
    those must resolve to -1 (they inherit their layer from the graph, not the number)."""
    if not name:
        return -1
    if name.startswith("node_"):
        return -1
    m = _LAYER_BLK_RE.match(name)
    if m:
        return int(m.group(1))
    m = _LAYER_SUFFIX_RE.search(name)
    if m:
        return int(m.group(1))
    return -1


def topology_key_for_gguf(gguf_tensors, gguf_meta):
    """Structural topology key for a model, from its gguf -- NO name, NO size. The key is
    `<general.architecture>-<8hex>` where the hex hashes the SET of distinct per-layer
    tensor roles (blk.<N>.<role> -> role). Identical for every size of one architecture,
    different exactly when the block structure differs (dense vs MoE vs per-layer-embd).
    Must stay byte-identical to extract_topology._arch_key so the checked-in
    topologies/<key>.json is found."""
    import hashlib
    arch = gguf_meta.get("general.architecture", "unknown")
    roles = set()
    for t in (gguf_tensors or []):
        nm = t.get("name", "") if isinstance(t, dict) else str(t)
        m = re.match(r"^blk\.\d+\.(.+)$", nm)
        if m:
            roles.add(m.group(1))
    fp = hashlib.sha1((arch + "|" + "|".join(sorted(roles))).encode()).hexdigest()[:8]
    return "%s-%s" % (arch, fp)


def load_checked_in_topology(key):
    """Load the checked-in structural topology for a key from topologies/<key>.json, or
    None if that architecture has not been captured yet. The stored skeleton is one
    representative layer's faithful node->src[] structure (timing-free); the viewer
    replays it per layer and overlays live kernel timings from the run's trace."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "topologies", key + ".json")
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def expand_topology_to_layers(skel, n_layer):
    """Expand a one-block skeleton into a per-layer node list the frontend consumes. The
    skeleton's `src` are INDEX-based (producing node index within the block, from the dump's
    storage-id DAG), so instantiating a layer is a pure index OFFSET -- copy the block, add
    L*B to each intra-block edge. No name resolution (the old name-based path split disjoint
    subgraphs when a cb-name repeated). The inter-layer residual is threaded by wiring each
    layer's entry (a block root that reads nothing but should consume the running stream) to
    the previous layer's exit (the block's terminal). Returns {step, nodes:[{i,name,op,L,
    src:[idx],...}], by_layer, edges_inferred:False}."""
    sk = skel.get("nodes", [])
    B = len(sk)
    if B == 0:
        return {"step": "topology (checked-in)", "n_nodes": 0, "nodes": [],
                "by_layer": {}, "edges_inferred": False, "provenance": ""}
    # Block entry = the first root (no in-block src) that is a norm/residual consumer -- the
    # node that in a full trace reads the previous block's output. Block exit = the last node
    # feeding it (the residual sink). We thread entry[L] <- exit[L-1] to connect layers.
    roots = [i for i, nd in enumerate(sk) if not nd.get("src")]
    entry = next((i for i in roots if "norm" in sk[i].get("name", "").lower()),
                 roots[0] if roots else 0)
    # exit = the highest-index node that is NOT a cache/state write (those are KV sinks, not
    # the residual). Falls back to the last node.
    exit_ = B - 1
    for i in range(B - 1, -1, -1):
        nm = sk[i].get("name", "").lower()
        if "cache" not in nm and "state" not in nm and "result" not in nm:
            exit_ = i
            break
    nodes = []
    for L in range(max(n_layer, 1)):
        base = L * B
        for j, nd in enumerate(sk):
            srcs = [base + s for s in nd.get("src", []) if 0 <= s < B and s != j]
            # residual thread: this layer's entry reads the previous layer's exit
            if j == entry and L > 0:
                srcs.append((L - 1) * B + exit_)
            nodes.append({"i": base + j, "name": "%s-%d" % (nd.get("name", "?"), L),
                          "op": nd.get("op", "?"), "L": L, "ne": [], "type": "",
                          # fam/quant/role baked into the skeleton (stable per arch): role is
                          # the join key to this run's order-mapped dispatch -> live per-size
                          # KxN + actual kernel without a view-time dump.
                          "fam": nd.get("fam", ""), "quant": nd.get("quant", ""),
                          # role = best-effort single role; roles = all roles a shared cb-name
                          # node can be (e.g. "node" -> attn_qkv/ssm_alpha/ssm_beta); the
                          # frontend picks whichever has a live dispatch in this layer.
                          "role": nd.get("role", ""), "roles": nd.get("roles", []),
                          # folded sub-kernels (quantize_q8_1 prep) -> the node's +Nk badge.
                          "kernels": nd.get("kernels", []),
                          "us_in": None, "src": list(dict.fromkeys(srcs)), "chain": False})
    by_layer = defaultdict(list)
    for nd in nodes:
        by_layer[nd["L"]].append(nd["i"])
    return {"step": "topology (checked-in)", "n_nodes": len(nodes), "nodes": nodes,
            "by_layer": dict(by_layer), "edges_inferred": False,
            "provenance": "FAITHFUL topology from checked-in architecture skeleton "
                          "(ggml-roofline storage-id INDEX edges, arch-keyed); live timings "
                          "not attached in this view."}


def _first_token_rows(rows):
    """A roofline dump from an N-token decode run replays the WHOLE graph once per token,
    so every layer's block appears N times. For the graph view we want ONE forward pass.
    Each token ends at the final lm_head op (name 'result_output' / 'result_embd', or the
    last row's name); the next token restarts the graph. Return just the first pass: rows
    up to and including the first terminal op. Falls back to the full list if no terminal
    marker is found (single-pass dump)."""
    if not rows:
        return rows
    terminals = ("result_output", "result_embd", "result_norm")
    for i, r in enumerate(rows):
        nm = str(r.get("name", ""))
        if nm in terminals:
            return rows[:i + 1]
    # No named terminal: detect a restart -- the first row's name recurring after a gap.
    first = str(rows[0].get("name", ""))
    if first:
        for i in range(1, len(rows)):
            if str(rows[i].get("name", "")) == first and i > len(rows) // 2 - 1:
                return rows[:i]
    return rows


def _reconstruct_from_roofline(rows):
    """Reconstruct the TRUE per-layer dataflow graph from a ggml-roofline artifact that
    carries per-invocation storage-id topology (llama.cpp fork PR #66). Each row is one
    op launch with:

        {"invocation": <int>, "name": "blk.3.attn_q", "ggml_op": "MUL_MAT",
         "out_storage_id": <int>, "in_storage_ids": [<int>, ...],
         "kernels": [{"name": ..., "gpu_time_us": <float>}, ...]}

    Edges are EXACT (not inferred): the `*_storage_id`s are real view-root-resolved
    tensor identities from node->src[]. We recover producer->consumer edges by
    last-writer-wins over execution order -- sort by `invocation`, track the most recent
    launch that wrote each storage id, and link each `in_storage_id` to that writer.
    This is architecture-agnostic: no per-model topology template is used, so it covers
    every model (dense, MoE, GDN/SSM, VLM) identically. Returns the standard node list
    (i, name, op, L, src=[node indices], us_in, ...) the frontend expects."""
    # Keep rows that carry storage-id edges under EITHER schema (aspirational
    # in/out_storage_id, or the emitter's real src_/dst_storage_id).
    rows = [r for r in rows
            if r.get("in_storage_ids") is not None or r.get("out_storage_id") is not None
            or r.get("src_storage_ids") is not None or r.get("dst_storage_id") is not None]
    # Preserve execution order. The emitter writes rows in launch order and omits an
    # explicit "invocation" index, so a sort by a missing key would collapse to a no-op
    # anyway; only sort when the field is actually present (stable-sort keeps order else).
    if any("invocation" in r for r in rows):
        rows = sorted(rows, key=lambda r: r.get("invocation", 0))
    rows = _first_token_rows(rows)
    nodes = []
    writer = {}                 # storage_id -> node index that last wrote it
    # Layer inference: the emitter's geometry-dedup schema (dst_storage_id + shape, NO
    # per-op name -- see ggml-cuda-roofline.cpp, ops collapsed by geometry_id) carries no
    # blk.<N> tag, so _layer_of_name can't recover L. Rows are in execution order and each
    # decode layer repeats the same op sequence, so we assign L by counting how many times
    # an identical (op,N,K,quant) signature has been seen so far -- the Nth occurrence is
    # layer N. This gives every matvec its true layer without needing names.
    _sig_seen = {}
    for idx, r in enumerate(rows):
        # Accept BOTH schemas: the aspirational name-based one (in_storage_ids/out_storage_id/
        # name) AND the emitter's real geometry schema (src_storage_ids/dst_storage_id, no
        # name -> synthesize a shape-based identity). Real names win when present.
        in_sids = r.get("in_storage_ids")
        if in_sids is None:
            in_sids = r.get("src_storage_ids") or []
        out_sid = r.get("out_storage_id")
        if out_sid is None:
            out_sid = r.get("dst_storage_id")
        name = r.get("name", "")
        if not name:
            # no tensor name in the dump: synthesize op@NxK[quant] and layer-tag it below so
            # the matcher and layer swim-lane still have a stable, shape-unique identity.
            _op = r.get("ggml_op", "op"); _N = r.get("N"); _K = r.get("K")
            _q = r.get("quant", "")
            if _N and _K:
                sig = "%s.%dx%d.%s" % (_op, _N, _K, _q)
            else:
                sig = "%s" % _op
            lyr = _sig_seen.get(sig, 0); _sig_seen[sig] = lyr + 1
            name = "blk.%d.%s" % (lyr, sig)
        L = _layer_of_name(name)
        kernels = r.get("kernels", [])
        us = sum(k.get("gpu_time_us", 0.0) for k in kernels) or None
        # Derive the kernel FAMILY (same normalization as the swim lane) from this op's
        # dominant kernel -- the last one, which is the real compute (a matmul op emits a
        # quantize prep kernel then the mul_mat_vec; we want the mul_mat). This `fam` is
        # what the swim lane shows and is the join key for two-way selection sync, since
        # a roofline dump carries durations but no absolute start/end to match by time.
        fam = family_of(kernels[-1]["name"]) if kernels else (r.get("ggml_op", "?"))
        op = r.get("ggml_op", "?")
        # Display label: usually the kernel family (matches the swim lane), BUT for generic
        # multi-purpose kernels the family is uninformative -- k_bin_bcast implements ADD,
        # MUL, SUB, etc., so three different ops (two residual ADDs + an expert MUL) would
        # all read "k_bin_bcast". For those, prefer the ggml OP name so the node says what
        # it actually IS. `fam` is kept separately for the swim-lane highlight join.
        GENERIC = ("k_bin_bcast", "ggml_compute_forward", "op_")
        label = op if any(gk in fam for gk in GENERIC) else fam
        # One ggml op can launch several GPU kernels (e.g. a quantized MUL_MAT launches
        # quantize_q8_1 then mul_mat_vec_q). The graph is one-node-per-op, so those
        # sub-kernels are folded into this node; carry their family + time so the node
        # detail can list them (nothing is hidden, the DAG stays clean).
        subk = [{"fam": family_of(k.get("name", "")),
                 "us": round(k.get("gpu_time_us", 0.0), 2)} for k in kernels]
        srcs = []
        for sid in in_sids:
            w = writer.get(sid)
            if w is not None and w != idx:
                srcs.append(w)
        nodes.append({"i": idx, "name": name, "op": op, "label": label,
                      "fam": fam, "kernels": subk, "L": L,
                      # carry the emitter's exact shape so the weight->dispatch matcher can
                      # disambiguate same-N siblings by (N,K,quant) instead of N alone.
                      "ne": [r.get("K"), r.get("N")] if r.get("N") else [],
                      "N": r.get("N"), "K": r.get("K"), "quant": r.get("quant", ""),
                      "n_experts": r.get("n_experts", 0),
                      # experts_used = experts ROUTED this launch (top_k). A grouped MoE
                      # mul_mat_id reads experts_used experts in ONE dispatch, so its eff BW
                      # numerator is per_expert_bytes * experts_used, not one expert.
                      "experts_used": r.get("experts_used", 0),
                      "top_k": r.get("top_k", 0), "type": "",
                      "us_in": (us * 1000.0) if us is not None else None,  # us -> ns
                      "src": list(dict.fromkeys(srcs)), "chain": False})
        if out_sid is not None:
            writer[out_sid] = idx        # last writer wins (also handles in-place aliasing)
    return nodes


def load_graph_json(path):
    """Load a compute-graph artifact for one step and index it by layer. Two schemas
    are accepted:

    (A) ROOFLINE with topology (llama.cpp fork PR #66) -- the FAITHFUL path. Rows carry
        per-invocation storage-id edges; we reconstruct exact producer->consumer edges
        via last-writer-wins (see _reconstruct_from_roofline). Detected by the presence
        of `in_storage_ids`/`out_storage_id` on the rows (under a top-level "ops" or
        "nodes"/"invocations" list). edges_inferred = False.

    (B) NODE list (trace-derived stand-in) -- nodes already carry `src` as node indices
        (edges may be inferred). Detected by `src` holding small node-index ints and no
        storage ids. edges_inferred defaults from the file.

    Node shape passed to the frontend: {i, name, op, L, ne, type, us_in, src:[i,...],
    chain}. `L` is the GGUF block index from the `blk.<N>.` name prefix (-1 otherwise);
    `by_layer` maps L -> [node i, ...] (the join key to the overlay's layer swim-lane)."""
    with open(path) as fh:
        g = json.load(fh)
    # A roofline artifact keys its per-op list under "ops" (or reuses "nodes"); detect
    # the topology schema by whether any row has storage-id edges.
    rows = g.get("rows") or g.get("ops") or g.get("invocations") or g.get("nodes") or []
    # Detect the roofline/storage-id schema. Accept the aspirational field names
    # (in_storage_ids/out_storage_id) AND the emitter's actual geometry-dedup names
    # (src_storage_ids/dst_storage_id -- see ggml-cuda-roofline.cpp). Either means we
    # have exact per-op edges -> the FAITHFUL reconstruction path, not inferred edges.
    is_roofline = any(isinstance(r, dict) and
                      ("in_storage_ids" in r or "out_storage_id" in r or
                       "src_storage_ids" in r or "dst_storage_id" in r) for r in rows)
    if is_roofline:
        nodes = _reconstruct_from_roofline(rows)
        edges_inferred = False
        provenance = ("FAITHFUL: edges reconstructed from ggml-roofline per-invocation "
                      "storage-id topology (PR #66), last-writer-wins over execution "
                      "order -- exact node->src[] dependencies, architecture-agnostic.")
        step = g.get("step", "roofline")
    else:
        nodes = rows
        edges_inferred = bool(g.get("edges_inferred", False))
        provenance = g.get("provenance", "")
        step = g.get("step", "")

    by_layer = defaultdict(list)
    for nd in nodes:
        if nd.get("L") is not None:
            L = int(nd["L"])
        else:
            L = _layer_of_name(nd.get("name", "") or "")
        nd["L"] = L
        by_layer[L].append(nd.get("i"))
    return {"step": step, "n_nodes": g.get("n_nodes", len(nodes)),
            "nodes": nodes, "by_layer": dict(by_layer),
            "edges_inferred": edges_inferred, "provenance": provenance}


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
    # Aggregate wave-cycles per hardware state across ALL captured waves, from the raw
    # (un-downsampled) timelines -- the exact wave-occupancy breakdown. States:
    # 1=Idle, 2=Exec (issuing), 3=Wait (waitcnt / memory), 4=Stall (dependency/backpr).
    # This is the "what were the waves doing over the kernel" summary: Exec% is useful
    # issue, Stall% is where dependency/latency piles up (e.g. dequant-convert chains).
    st_cy = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in raw:
        for seg in r[7]:
            if isinstance(seg, list) and len(seg) >= 2 and seg[1] > 0:
                st_cy[seg[0]] = st_cy.get(seg[0], 0) + seg[1]
    st_tot = sum(st_cy.values()) or 1
    state_mix = {"Idle": st_cy.get(1, 0), "Exec": st_cy.get(2, 0),
                 "Wait": st_cy.get(3, 0), "Stall": st_cy.get(4, 0),
                 "total": st_tot,
                 "pct": {k: round(100.0 * st_cy.get(v, 0) / st_tot, 1)
                         for k, v in (("Idle", 1), ("Exec", 2), ("Wait", 3), ("Stall", 4))}}
    for se, sm, cu, sl, wv, begin, end, tl in sorted(
            raw, key=lambda r: (r[0], r[1], r[3], r[4])):
        waves.append({"lab": "se%d sm%d sl%d wv%d" % (se, sm, sl, wv),
                      "se": se, "cu": cu, "simd": sm, "slot": sl, "wv": wv,
                      "begin": begin, "end": end, "rle": _rle(begin, tl)})
    return {"t0": t0, "t1": t1, "nb": _WAVE_NB,
            "states": ["", "Idle", "Exec", "Wait", "Stall"],
            "state_mix": state_mix, "n_waves": len(waves), "waves": waves}


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


def _att_isolate_run(dispatch_dir):
    """Select the per-wave `se*_wv*.json` files for ONE representative run of the traced
    kernel, discarding contamination the raw ATT buffer captured around it.

    A shape-exact perf run invokes the kernel thousands of times and the ATT capture
    window catches: (a) waves from the target mmq kernel, (b) waves from unrelated
    neighbour kernels (very different instruction counts), and (c) SEVERAL back-to-back
    generations of the target kernel time-multiplexed onto the traced SIMD. For a clean
    per-run picture we keep only the DOMINANT-instruction-count kernel (its waves all
    share one num_insts -- the mmq body) and then only its FIRST time-generation (waves
    whose begin is within one wave-duration of the earliest such wave). Returns a list
    of parsed wave dicts (with 'begin','end','timeline','instructions'); [] if none.

    This is what makes the utilization / state-mix numbers reflect a single mmq launch
    instead of a blend of 3 generations + two foreign kernels."""
    import glob
    waves = []
    for p in glob.glob(os.path.join(dispatch_dir, "se*_sm*_sl*_wv*.json")):
        try:
            with open(p) as fh:
                w = (json.load(fh).get("wave") or {})
        except (OSError, ValueError):
            continue
        ins = w.get("instructions")
        if ins and w.get("begin") is not None and w.get("end") is not None:
            waves.append(w)
    if not waves:
        return []
    # (a) dominant kernel = the instruction-count bucket with the most TOTAL work
    # (sum of instructions), NOT the most waves -- a capture often has many tiny
    # neighbour-kernel waves (e.g. 186-instr helpers) that outnumber but are dwarfed
    # by the few heavy mmq-body waves (~18k instr each). Group by num_insts, pick the
    # bucket maximizing count*num_insts.
    work = defaultdict(int)
    for w in waves:
        work[len(w["instructions"])] += len(w["instructions"])
    dom_n = max(work, key=work.get)
    dom = [w for w in waves if len(w["instructions"]) == dom_n]
    if not dom:
        return []
    # (b) first time-generation only: a launch's waves on one SIMD all START within a
    # few hundred cycles of each other (co-scheduled), while the NEXT generation begins
    # ~one wave-duration later. Keep waves whose begin is within a small tolerance of
    # the earliest -- tight enough to exclude the next generation.
    t_start = min(w["begin"] for w in dom)
    durs = sorted(w["end"] - w["begin"] for w in dom)
    med = durs[len(durs) // 2] or 1
    tol = max(2000, int(med * 0.05))
    gen0 = [w for w in dom if w["begin"] - t_start <= tol]
    return gen0 or dom


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
    import glob
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
    # Aggregate wave-state cycles from the per-wave `se*_wv*.json` timelines in the SAME
    # dispatch dir (occupancy.json has slot residency, not per-cycle state). Each wave's
    # `timeline` is [[state,cycles],...] with 1=Idle 2=Exec 3=Wait 4=Stall. This gives
    # the "what were the waves doing" utilization mix shown beside the stall table.
    state_mix = None
    st_cy = {1: 0, 2: 0, 3: 0, 4: 0}
    nwv = 0
    for w in _att_isolate_run(dispatch_dir):   # ONE representative run, not all gens
        tl = w.get("timeline") or []
        if not tl:
            continue
        nwv += 1
        for seg in tl:
            if isinstance(seg, list) and len(seg) >= 2 and seg[1] > 0:
                st_cy[seg[0]] = st_cy.get(seg[0], 0) + seg[1]
    tot = sum(st_cy.values())
    if tot > 0:
        state_mix = {"total": tot, "n_waves": nwv,
                     "pct": {k: round(100.0 * st_cy.get(v, 0) / tot, 1)
                             for k, v in (("Idle", 1), ("Exec", 2),
                                          ("Wait", 3), ("Stall", 4))}}
    return {"t0": t0, "t1": t1, "kernels": kernels, "lanes": lanes,
            "state_mix": state_mix, "n_waves": nwv}


# Per-wave instruction stream (se*_wv*.json `wave.instructions`) tuple:
#   [issue_cycle, class, _, _, code_line_idx]
# `class` maps to a hardware execution unit (verified against the ISA opcodes each
# class carries). This is the RCV "Utilization view" grouping.
_ATT_UNIT_BY_CLASS = {
    1: "SMEM",    # s_load_*            scalar memory
    2: "SALU",    # s_mov/add/lshl/cmp  scalar ALU
    3: "VMEM",    # global/buffer       vector memory
    5: "LDS",     # ds_load/store       local data share
    6: "VALU",    # v_* (incl v_wmma)   vector ALU / matrix
    7: "BRANCH", 8: "BRANCH",           # s_cbranch
    9: "WAIT",    # s_waitcnt/s_clause  wait / decode
    11: "MSG",    # s_barrier/sendmsg/endpgm
}
# Lane order (top -> bottom). NOTE (RDNA3.5/gfx1151): WMMA is NOT a separate matrix
# engine -- v_wmma_* are VALU instructions executed on the vector ALU (same datapath +
# register file as all other v_* ops). So WMMA and dequant/convert VALU CONTEND for the
# one VALU. We still break WMMA out (adjacent to VALU) to show how much of the VALU
# budget is the actual matmul vs the surrounding dequant/convert -- but both are VALU.
_ATT_UNIT_ORDER = ["WMMA", "VALU", "LDS", "VMEM", "SMEM", "SALU", "WAIT", "BRANCH", "MSG"]
_ATT_UTIL_NB = 1200   # horizontal cycle-bucket budget for the utilization timeline


def load_att_util(dispatch_dir):
    """Build the RCV-style per-hardware-unit utilization timeline for one decoded ATT
    dispatch dir. Each per-wave `se*_wv*.json` carries `wave.instructions`, a list of
    [issue_cycle, class, _, _, code_line] tuples. We classify each instruction to a HW
    unit (VALU / WMMA / LDS / VMEM / SMEM / SALU / WAIT / BRANCH / MSG) and mark that
    unit BUSY at the instruction's issue cycle. Aggregated across all captured waves and
    downsampled onto a fixed grid of _ATT_UTIL_NB cycle-buckets: a lane's bucket is the
    fraction of that bucket's waves-instructions that hit the unit (0..1 intensity), so
    a dense v_wmma/v_fma region reads as a solid block like RCV's colored cells.

    Returns {t0, t1, nb, units:[name...], lanes:{unit:[intensity per bucket]},
             busy:{unit: pct of all instr-cycles}} or None. WMMA is detected by ISA
    (v_wmma*) and pulled out of the VALU class so the matrix-engine lane is separate."""
    import glob
    if not dispatch_dir:
        return None
    # code line -> is-WMMA (to split WMMA out of the VALU class), from code.json.
    is_wmma = {}
    cpath = os.path.join(dispatch_dir, "code.json")
    try:
        with open(cpath) as fh:
            code = (json.load(fh) or {}).get("code") or []
        for idx, row in enumerate(code):
            isa = (row[0] if row else "") or ""
            if isa.lstrip().startswith("v_wmma"):
                is_wmma[idx] = True
    except (OSError, ValueError):
        code = []
    # Per-wave event lists (so the util view can break the merged 4-up back into the
    # individual co-resident waves of the workgroup). wave_ev[i] = [(cycle, unit)...];
    # wave_id[i] = a human slot label like "SIMD3 slot0". The merged aggregate is just
    # the concatenation of all per-wave events over one shared time axis.
    wave_ev = []
    wave_lbl = []
    t0 = None; t1 = 0
    simds = set()
    for w in _att_isolate_run(dispatch_dir):   # ONE representative run, not all gens
        insns = w.get("instructions") or []
        if not insns:
            continue
        simds.add((w.get("cu", 0), w.get("simd", 0)))
        wev = []
        for it in insns:
            if not isinstance(it, list) or len(it) < 5:
                continue
            cyc, cl, ln = it[0], it[1], it[4]
            unit = _ATT_UNIT_BY_CLASS.get(cl)
            if unit is None:
                continue
            if unit == "VALU" and is_wmma.get(ln):
                unit = "WMMA"
            wev.append((cyc, unit))
            if t0 is None or cyc < t0:
                t0 = cyc
            if cyc > t1:
                t1 = cyc
        if wev:
            wave_ev.append(wev)
            sm, sl = w.get("simd", 0), w.get("slot", 0)
            wave_lbl.append("SIMD%d slot%d" % (sm, sl))
    if not wave_ev or t0 is None:
        return None
    span = max(1, t1 - t0)
    bw = span / float(_ATT_UTIL_NB)
    units = [u for u in _ATT_UNIT_ORDER]

    # Bucket ONE event list into {unit:[intensity per bucket]}, busy{unit:pct}, ntot.
    # Intensity = fraction of that bucket's instructions (within this scope) that hit
    # the unit -- identical definition whether scope is one wave or the merged set.
    def _bucketize(evs):
        lane_cnt = {u: [0] * _ATT_UTIL_NB for u in units}
        buck_tot = [0] * _ATT_UTIL_NB
        busy = {u: 0 for u in units}
        for cyc, unit in evs:
            b = int((cyc - t0) / bw)
            if b >= _ATT_UTIL_NB:
                b = _ATT_UTIL_NB - 1
            if unit in lane_cnt:
                lane_cnt[unit][b] += 1
                busy[unit] += 1
            buck_tot[b] += 1
        ntot = len(evs) or 1
        active = [u for u in units if busy[u] > 0]
        lanes = {u: [round(lane_cnt[u][b] / buck_tot[b], 3) if buck_tot[b] else 0
                     for b in range(_ATT_UTIL_NB)] for u in active}
        return {"units": active, "lanes": lanes,
                "busy": {u: round(100.0 * busy[u] / ntot, 1) for u in active},
                "n_instr": ntot}

    merged_evs = [e for wev in wave_ev for e in wev]
    agg = _bucketize(merged_evs)
    per_wave = []
    for i, wev in enumerate(wave_ev):
        d = _bucketize(wev)
        d["label"] = wave_lbl[i]
        per_wave.append(d)
    return {"t0": t0, "t1": t1, "nb": _ATT_UTIL_NB,
            "units": agg["units"], "lanes": agg["lanes"], "busy": agg["busy"],
            "n_instr": agg["n_instr"], "n_waves": len(wave_ev),
            "n_simd": len(simds), "waves": per_wave}


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
    _util_cache = {}   # dispatch dir -> per-unit utilization timeline
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
        if ddir not in _util_cache:
            _util_cache[ddir] = load_att_util(ddir)
        util = _util_cache[ddir]
        out[fam] = {"sym": d["sym"], "n_disp": ndisp[fam], "stall": d["stall"],
                    "lat": d["lat"], "idle": d["idle"], "rows": d["rows"],
                    "has_src": d.get("has_src", False), "src_files": src_files,
                    "exec": exec_stream, "waves": waves, "occ": occ, "util": util}
    return out


# --- GGUF weight-tensor table (stdlib; per-dispatch true-shape mapping) --------
# GGUF value-type enum (gguf spec) used to walk metadata KV pairs.
_GGUF_SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9

# ggml_type -> (block_elems, block_bytes): on-disk packed size of one block.
# K-quants pack 256 elems/block; legacy quants 32; F32/F16/BF16 are dense.


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
    # MoE (gpt-oss): the router (ffn_gate_inp, an F32 mul_mat_vec_f) picks experts,
    # then the fused expert gate+up and the expert down projection dispatch. These
    # sit where the dense ffn_gate/up/down would in a non-MoE model.
    "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
    # Shared expert (qwen3.6/qwen3next MoE): after the routed experts, a dense FFN
    # runs every token -- fused gate+up (Q8_0) then down -- plus its own 1D sigmoid
    # gate (ffn_gate_inp_shexp, excluded: 1D output, not a real matvec dispatch).
    "ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "ffn_gate", "ffn_up", "ffn_down",
]
# Roles whose weight is a router/gating projection dispatched as F32 mul_mat_vec_f
# (not a quantized mul_mat_vec_q), so they are matvec-eligible regardless of quant.
_ROUTER_ROLES = {"ffn_gate_inp", "ffn_gate_inp_shexp"}
# ne-dim quant/dense types dispatched as a matvec at decode (K-quants, legacy
# quants, Q8_0 which carries the ssm alpha/beta scale projections, and MXFP4 which
# carries gpt-oss MoE expert weights -> mul_mat_vec_q<(ggml_type)39>).
_MATVEC_TYPES = {2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 39}


# MoE expert roles: their weight is 3D (one matrix per expert) and runs as a single
# grouped GGML_OP_MUL_MAT_ID dispatch over grid.y = n_experts_used, NOT one matvec per
# expert. The order-map must expect ONE dispatch for these (flagged is_moe) and match it
# against a trace slice with gy>1, else the positional lockstep desyncs on every MoE layer.
_MOE_EXPERT_ROLES = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}


def _seq_entry(t, layer, role):
    ne = t["ne"]
    is_moe = role in _MOE_EXPERT_ROLES and len(ne) >= 3
    # bytes for effbw: the packed footprint the DISPATCH streams. For a dense weight
    # that is the whole tensor (t["bytes"]). For a 3D MoE expert weight t["bytes"] is
    # the FULL ne[0]*ne[1]*n_experts stack (all 256 experts), but a decode dispatch
    # reads only the routed experts -- using the full stack inflated effbw ~165x
    # (e.g. 184MB/33us = 5567 GB/s, 24x roofline, and overfetch < 1.0 which is
    # physically impossible). Use the PER-EXPERT 2D slice (ne[0]*ne[1]) as the packed
    # footprint so effbw = one-expert-bytes / dispatch-time is a sane per-expert
    # roofline number. (The grouped dispatch runs gy = active-experts of these in one
    # launch; per-expert BW is the honest useful-work metric and matches FETCH_SIZE
    # once multiplied by the active-expert count.)
    if is_moe:
        pbytes = _gguf_packed_bytes([ne[0], ne[1]], t["gt"])
    else:
        pbytes = t["bytes"]
    return {"nm": t["name"], "L": layer, "role": role,
            "N": ne[1], "K": ne[0], "gt": t["gt"], "bytes": pbytes,
            "q": _GGML_TYPES.get(t["gt"], "type%d" % t["gt"]),
            # is_moe: a 3D expert weight -> grouped mul_mat_id dispatch (gy = experts).
            "is_moe": is_moe}


def _kernel_quant_type(kname):
    """The ggml_type the KERNEL was templated on, from its mangled name
    (mul_mat_vec_q<(ggml_type)12,...> -> 12 -> "Q4_K"), or the specialized-kernel
    quant (mul_mat_vec_q_wvsplitk_q8_0 -> "Q8_0"). None if the name carries no quant
    (mul_mat_vec_f = float, generic ops). Used to reject an order-map that would attach
    a weight whose quant contradicts the kernel that actually ran (Level 1 guard)."""
    m = re.search(r"\(ggml_type\)(\d+)", kname)
    if m:
        return _GGML_TYPES.get(int(m.group(1)), "type%d" % int(m.group(1)))
    m = re.search(r"wvsplitk_(q\d+_[0k]|q\d+_k)", kname, re.I)
    if m:
        return m.group(1).upper()
    # The plain wvsplitk kernel (no type suffix) is Q4_K-specialized (static_assert).
    if "wvsplitk" in kname:
        return "Q4_K"
    return None


def build_expected_sequence_from_dump(layer_graph, tensors):
    """Build the per-token matvec expected sequence from a roofline DUMP instead of
    the GGUF role heuristic. The dump carries EVERY op the model actually ran, in exact
    execution order, with exact (N, K, quant) -- including the GDN/SSM F32 projections
    (ssm_alpha/beta N=32, per-token gates N=1, router N=256) that the GGUF role-order
    builder deliberately skips (they have no clean weight-role mapping), which is why
    GGUF-only coverage stalls at ~26% on hybrid models. Using the dump lifts coverage to
    ~100% because the sequence IS the ground-truth op stream.

    Each dump matvec node -> an entry {N, K, q, is_moe, nm, role, L, bytes} in the same
    shape the order-map match loop expects. We attach a real GGUF weight name/role/bytes
    when a tensor uniquely matches (N, K, quant); otherwise synthesize a shape-based name.
    Returns the ONE-token sequence (delimited by the output-head vocab-N op)."""
    if not (layer_graph and layer_graph.get("nodes")):
        return []
    mv = [n for n in layer_graph["nodes"] if "MUL_MAT" in str(n.get("op", "")) and n.get("N")]
    if not mv:
        return []
    # Delimit one token: the output head is the largest-N op; it fires once per token.
    vocab_n = max(n["N"] for n in mv)
    heads = [i for i, n in enumerate(mv) if n["N"] == vocab_n]
    seq_nodes = mv[heads[0]:heads[1]] if len(heads) >= 2 else mv
    # Index GGUF tensors by (N, K, quant-name) for name/role/bytes attach. A shape may map
    # to several weights (same N,K,quant across layers/roles); we round-robin per shape in
    # execution order so repeated shapes get distinct per-layer names where possible.
    from collections import defaultdict
    by_shape = defaultdict(list)
    for t in tensors:
        ne = t["ne"]
        if len(ne) < 2:
            continue
        qn = _GGML_TYPES.get(t["gt"], "type%d" % t["gt"]).lower()
        by_shape[(ne[1], ne[0], qn)].append(t)
    shape_cursor = defaultdict(int)
    seq = []
    for n in seq_nodes:
        N, K = n["N"], n["K"]
        qn = (n.get("quant") or "").lower()
        cands = by_shape.get((N, K, qn), [])
        t = None
        if cands:
            t = cands[shape_cursor[(N, K, qn)] % len(cands)]
            shape_cursor[(N, K, qn)] += 1
        if t is not None:
            ne = t["ne"]
            is_moe = len(ne) >= 3
            pbytes = (_gguf_packed_bytes([ne[0], ne[1]], t["gt"]) if is_moe else t["bytes"])
            nm = t["name"]
            role = _role_of_name(nm)
            L = _layer_of_name(nm)
        else:
            # No GGUF weight for this op (e.g. a fused/intermediate node) -> synthesize.
            is_moe = bool(n.get("n_experts"))
            pbytes = _gguf_packed_bytes([K, N], _quant_name_to_gt(qn)) if qn != "f32" else K * N * 4
            nm = n.get("name") or ("op.%dx%d.%s" % (N, K, qn))
            role = "gdn" if qn == "f32" else "matvec"
            L = _layer_of_name(nm)
        seq.append({"nm": nm, "L": L, "role": role, "N": N, "K": K,
                    "gt": _quant_name_to_gt(qn), "bytes": pbytes,
                    "q": qn.upper() if qn != "f32" else None, "is_moe": is_moe})
    return seq


def _quant_name_to_gt(qn):
    inv = {v.lower(): k for k, v in _GGML_TYPES.items()}
    return inv.get((qn or "").lower(), 0)


def _role_of_name(nm):
    m = re.search(r"blk\.\d+\.([a-z_]+?)(?:\.weight)?$", nm or "")
    return m.group(1) if m else (nm or "")


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
            # Fused SwiGLU: the single gate dispatch also streams the up weight, so
            # its up sibling is not a separate dispatch. This holds for the dense
            # ffn_up and the MoE ffn_up_exps alike.
            if role in ("ffn_up", "ffn_up_exps", "ffn_up_shexp") and drop_ffn_up:
                continue
            t = roles.get(role)
            if t is None or len(t["ne"]) < 2:
                continue
            # Router (ffn_gate_inp) is an F32 mul_mat_vec_f -- matvec-eligible despite
            # not being a quant type; all other roles must be a quant/dense matvec type.
            if role not in _ROUTER_ROLES and t["gt"] not in _MATVEC_TYPES:
                continue
            ent = _seq_entry(t, layer, role)
            # Fold the up weight's bytes into the fused gate dispatch (dense or MoE),
            # else the theoretical denominator is ~2x too small and the dispatch looks
            # like it over-fetches ~2x when it does not.
            if role in ("ffn_gate", "ffn_gate_exps", "ffn_gate_shexp") and drop_ffn_up:
                up = {"ffn_gate": "ffn_up", "ffn_gate_exps": "ffn_up_exps",
                      "ffn_gate_shexp": "ffn_up_shexp"}.get(role)
                up = roles.get(up)
                if up is not None and up["gt"] in _MATVEC_TYPES:
                    # up["bytes"] is the raw tensor footprint (full 3D expert stack for
                    # MoE); mirror _seq_entry and fold only the PER-EXPERT 2D slice so a
                    # fused MoE gate+up dispatch's packed stays per-expert-honest.
                    upne = up["ne"]
                    if role == "ffn_gate_exps" and len(upne) >= 3:
                        ent["bytes"] += _gguf_packed_bytes([upne[0], upne[1]], up["gt"])
                    else:
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

def detect_boundaries_by_head(evs):
    """Token boundaries from the OUTPUT-HEAD anchor instead of inter-dispatch gaps.
    The vocab projection (lm_head) is a mul_mat_vec with the largest N in the graph
    and fires exactly once per decode token, so successive heads bracket each token.
    This is robust whether or not HIP graph replay is on -- unlike the gap detector,
    which relies on the clean inter-token idle that a single hipGraphLaunch leaves and
    breaks in EAGER mode (graphs disabled, e.g. MoE mul_mat_id above the mmvq batch
    cap), where dispatches are packed with no distinguishing gap.

    Returns the index of the FIRST dispatch AFTER each head (= each token's start), so
    the returned list plays the same role as detect_boundaries() output. Empty if there
    is no clear periodic head anchor (falls back to the gap detector at the call site)."""
    mv = [(i, n) for i, (s, e, nm, n, _nb, _gy) in enumerate(evs)
          if n and "mul_mat" in nm]
    if len(mv) < 3:
        return []
    max_n = max(n for _, n in mv)
    # The head must be a clear outlier (vocab >> any weight N); require it to dwarf the
    # next-largest distinct N, else this isn't a reliable anchor (e.g. a prefill trace).
    other = [n for _, n in mv if n < max_n]
    if other and max_n < 2 * max(other):
        return []
    heads = [i for i, n in mv if n == max_n]
    if len(heads) < 3:
        return []
    # Boundary = first dispatch after each head. The head ENDS a token, so the next
    # dispatch STARTS the following token. Drop the trailing head (no token follows it).
    bounds = [h + 1 for h in heads[:-1] if h + 1 < len(evs)]
    # De-noise: keep only boundaries spaced ~ the dominant period apart (guards against
    # a stray duplicate-N dispatch masquerading as a head).
    if len(bounds) < 3:
        return bounds
    deltas = [bounds[i] - bounds[i - 1] for i in range(1, len(bounds))]
    period = statistics.median(deltas) or 1
    kept = [bounds[0]]
    for b in bounds[1:]:
        if b - kept[-1] >= period * 0.5:
            kept.append(b)
    return kept


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
    ap.add_argument("--graph-json",
                    help="JSON dump of the ggml compute graph for one step "
                         "(optional): nodes = ops/tensors, edges = tensor deps "
                         "(node.src). Enables the per-layer graph popup -- click a "
                         "layer segment to see its actual dataflow DAG. Produced by "
                         "a llama.cpp cgraph dump; see load_graph_json for schema.")
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
    ap.add_argument("--peak-tops", type=float,
                    help="override peak fp16/int8 compute in TOPS shown in the "
                         "title (else from --arch; omitted if unknown)")
    ap.add_argument("--clean-tps-file",
                    help="path to collect.sh's clean_tps.txt (the untraced "
                         "llama-bench run): parses the decode (tg) row's t/s and "
                         "shows it in the header as the honest throughput, since "
                         "rocprofv3 perturbs the traced runs' timing. Silently "
                         "ignored if the file is missing or unparseable.")
    ap.add_argument("--out", help="output HTML path (required for the generator)")
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode",
                    help="trace regime (default decode). 'decode' segments the "
                         "timeline by the periodic per-token dispatch stream (mmvq). "
                         "'prefill' bakes the single prompt-processing forward pass "
                         "(MMQ) as one span -- no per-token periodicity; the layer/"
                         "role lanes (decode order-map) are omitted, CPU+GPU+family "
                         "lanes render.")
    # Second (alternate-regime) trace: when supplied, the overlay embeds BOTH this
    # regime and the --mode one, and shows a prefill/decode dropdown that switches
    # between them in-page. Each regime is its own clean sys-trace (collected with
    # -p N -n 0 for prefill and -p 0 -n M for decode), so they stay measured in
    # isolation -- more correct than splitting one mixed trace. --alt-mode is the
    # OTHER regime; if omitted it is the opposite of --mode.
    ap.add_argument("--alt-kernel-csv",
                    help="second regime's *_kernel_trace.csv; enables the in-page "
                         "prefill/decode dropdown (embeds both traces)")
    ap.add_argument("--alt-hip-csv", help="second regime's *_hip_api_trace.csv")
    ap.add_argument("--alt-pmc-csv", help="second regime's stall *_counter_collection.csv")
    ap.add_argument("--alt-fetch-csv", help="second regime's FETCH_SIZE *_counter_collection.csv")
    ap.add_argument("--alt-clean-tps-file",
                    help="second regime's clean_tps.txt (its untraced pp/tg row)")
    ap.add_argument("--alt-mode", choices=["decode", "prefill"],
                    help="the second regime's mode (default: opposite of --mode)")
    ap.add_argument("--tokens", type=int, default=2,
                    help="decode tokens to show in the viewport (default 2)")
    ap.add_argument("--skip-tokens", type=int, default=30,
                    help="tokens to skip before the window, to land in steady "
                         "state past warmup/prefill (default 30)")
    ap.add_argument("--context-tokens", type=int, default=0,
                    help="extra tokens baked on each side for the stepper (default 0)")
    ap.add_argument("--gap-threshold-us", type=float, default=None,
                    help="inter-dispatch gap (us) that marks a token boundary "
                         "(default 150; a per-arch BOUNDARY_PROFILES value may apply "
                         "instead unless this is set explicitly)")
    ap.add_argument("--boundary-method", choices=("head", "gap", "auto"), default=None,
                    help="override token-boundary detection method for this run "
                         "(default: per-arch BOUNDARY_PROFILES, else auto). head = "
                         "output-head anchor; gap = inter-dispatch gap; auto = head "
                         "then gap fallback")
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
    # Normalize the gap-threshold sentinel: None default means "not set explicitly", so a
    # per-arch BOUNDARY_PROFILES gap_us may apply. Resolve to 150us for the paths that use
    # it unconditionally (prefill span, regen-command echo). Runs for both the CLI generator
    # and serve.py, since both call build_payload.
    args._gap_threshold_set = args.gap_threshold_us is not None
    if args.gap_threshold_us is None:
        args.gap_threshold_us = 150.0
    peak_bw = peak_bw_for(args.arch, args.peak_bw)
    peak_tops = peak_tops_for(args.arch, args.peak_tops)
    # Surface the roofline peaks next to the arch string in the title (e.g.
    # "... (gfx1151)" -> "... (gfx1151, 230 GB/s, 43 TOPS f16/int8)"). The TOPS
    # clause is only shown when a peak is known for the arch.
    title = args.title
    tag = "%s, %g GB/s" % (args.arch, peak_bw)
    if peak_tops:
        tag += ", %g TOPS f16/int8" % peak_tops
    if args.arch in title:
        title = title.replace(args.arch, tag, 1)
    else:
        title = "%s (%s)" % (title, tag)

    by_stream = load_kernel_slices(args.kernel_csv, mmq_y_for(args.arch))
    if not by_stream:
        sys.exit(f"error: no kernel rows in {args.kernel_csv}")
    # Compute stream = the one with the most dispatches (stream 1 is model load).
    sid = max(by_stream, key=lambda s: len(by_stream[s]))
    evs = by_stream[sid]

    # Load the GGUF (tensor table + meta) up front so the model architecture is known
    # BEFORE token-boundary detection -- the boundary profile is arch-specific. Cheap
    # (mmap, no weight bytes read). Empty meta for trace-only runs -> _default profile.
    gguf_tensors, gguf_meta = ([], {})
    if args.gguf:
        gguf_tensors, gguf_meta = load_gguf_tensors(args.gguf)
    model_arch = gguf_meta.get("general.architecture", "")

    # Regime = the ONE place decode vs prefill diverge on how to slice + interpret the
    # trace. It owns window selection, the order-map reference, and metric labels, so the
    # shared payload path below has NO `if mode ==` for those -- a decode change can't
    # touch prefill (see regimes/). The Window carries everything downstream needs.
    regime = regime_for(args.mode)
    win = regime.select_window(evs, args, model_arch)
    baked = win.baked
    t0, t1 = win.t0, win.t1
    tok_starts = win.tok_starts
    lo_tok = win.lo_tok
    view_i0, view_i1 = win.view_i0, win.view_i1

    # PMC families -> color/stall lookup.
    fams = load_pmc_families(args.pmc_csv) if args.pmc_csv else {}

    # Measured roofline: DRAM read bytes/dispatch per family from the PMC
    # FETCH_SIZE run (post-L2 actual VRAM traffic). Replaces the old GGUF analytic
    # estimate -- measured attributes bytes to the exact kernel that moved them
    # (so it also fixes the Q5_K shared-quant-type over-attribution the analytic
    # method had), and covers every family, not just mul_mat_vec.
    fetch_bytes, fetch_bytes_n, fetch_bytes_nk = (load_fetch_bytes(args.fetch_csv)
                                                  if args.fetch_csv else ({}, {}, {}))
    loadwidth = load_loadwidth(args.loadwidth_json) if args.loadwidth_json else {}
    layer_graph = load_graph_json(args.graph_json) if args.graph_json else None
    att_by_fam = load_att_stats(args.att_dir) if args.att_dir else {}
    att_code_by_fam = load_att_code(args.att_dir) if args.att_dir else {}
    # Occupancy is dispatch-wide: many families share the SAME occ object (identical
    # 640-lane table). Pool distinct occ objects into att_occ_pool and replace each
    # family's inline occ with an index (occ_ref) so the HTML embeds it ONCE, not 18x.
    att_occ_pool = []
    _occ_seen = {}
    att_util_pool = []      # per-unit utilization timelines, pooled the same way
    _util_seen = {}
    for _fam, _c in att_code_by_fam.items():
        _o = _c.pop("occ", None)
        if _o is None:
            _c["occ_ref"] = -1
        else:
            _key = id(_o)
            if _key not in _occ_seen:
                _occ_seen[_key] = len(att_occ_pool)
                att_occ_pool.append(_o)
            _c["occ_ref"] = _occ_seen[_key]
        _u = _c.pop("util", None)
        if _u is None:
            _c["util_ref"] = -1
        else:
            _uk = id(_u)
            if _uk not in _util_seen:
                _util_seen[_uk] = len(att_util_pool)
                att_util_pool.append(_u)
            _c["util_ref"] = _util_seen[_uk]
    clean_tps = (parse_clean_tps(args.clean_tps_file,
                                 "pp" if args.mode == "prefill" else "tg")
                 if args.clean_tps_file else None)

    # Model name for the title/sub-header. Prefer the explicit --gguf basename (minus the
    # .gguf suffix, to match the clean_tps-derived name); else recover it from
    # clean_tps.txt (llama-bench records model_filename), so the model is always identified
    # even without --gguf. collect.sh always emits clean_tps.txt.
    model_name = ""
    if args.gguf:
        _base = os.path.basename(args.gguf)
        model_name = _base[:-5] if _base.endswith(".gguf") else _base
    if not model_name and args.clean_tps_file:
        model_name = parse_clean_model_name(args.clean_tps_file)

    # Prefill is COMPUTE-bound: it processes a batch of B prompt tokens per matmul
    # (a GEMM), so the roofline denominator is peak TOPS, not peak DRAM BW. B is the
    # prompt length (pp<B> from clean_tps, e.g. pp128 -> 128); each mapped matmul does
    # 2*N*K*B MACs, so achieved TOPS = 2*N*K*B/kernel_time. Decode is B=1 (a matvec)
    # and BW-bound, so it keeps the DRAM-BW roofline. compute_batch is 0 when the
    # prompt length is unknown (no clean_tps) -> per-slice TOPS is omitted, not wrong.
    is_prefill = args.mode == "prefill"
    compute_batch = 0
    if is_prefill and clean_tps:
        _bm = re.search(r"(\d+)", clean_tps.get("test", "") or "")
        compute_batch = int(_bm.group(1)) if _bm else 0

    # Optional GGUF order-map: build the expected matmul tensor sequence in execution
    # order and lockstep-match it onto the trace's matmul dispatches. llama.cpp fuses
    # the SwiGLU gate+up into one dispatch at DECODE (mmvq) but keeps them separate at
    # PREFILL (MMQ), so try both dropping and keeping ffn_up and pick whichever
    # candidate's N-sequence best matches the actual matmul dispatches.
    #  - decode:  matmul kernel = mul_mat_vec; reference = ONE steady-state token
    #             (between two token boundaries); the whole sequence repeats/token.
    #  - prefill: matmul kernel = mul_mat_q; reference = the WHOLE baked forward pass
    #             (one pass, no per-token repeat). N recovered as grid.x*mmq_y (above).
    expected_seq = []
    # gguf_tensors/gguf_meta already loaded up front (before boundary detection).
    # mm_key + the order-map reference come from the regime (decode: one steady-state token;
    # prefill: the whole baked pass) -- no mode branch here.
    mm_key = regime.mm_key
    if args.gguf:
        ref = win.ref
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

        # NOTE: build_expected_sequence_from_dump() constructs the GROUND-TRUTH per-token op
        # sequence from a roofline dump (every op incl. GDN F32 projections, exact N/K/quant)
        # -- the path to lift order-map coverage past the ~26% GGUF-heuristic ceiling on
        # hybrid GDN/SSM models. But the dump's graph-BUILD order and the trace's EXECUTION
        # order diverge (GDN ops interleave differently), so a naive positional/rotational
        # match only recovers ~33%. Reconciling them needs proper sequence alignment
        # (edit-distance / storage-id join), which is follow-up work; the helper is kept
        # for that. Until then we use the GGUF-heuristic sequence above.

        # Auto-attach the checked-in structural topology for this architecture when the
        # user did not pass an explicit --graph-json. The key is derived from the gguf's
        # tensor-role fingerprint (no model name/size), so every size of an architecture
        # resolves to the same checked-in skeleton; we expand it across the model's real
        # block count. This gives a faithful per-layer graph for any covered arch with
        # ZERO per-model data -- topologies/ is ~kilobytes, keyed by architecture.
        if layer_graph is None and gguf_tensors:
            _key = topology_key_for_gguf(gguf_tensors, gguf_meta)
            _skel = load_checked_in_topology(_key)
            if _skel is not None:
                _arch = gguf_meta.get("general.architecture", "")
                _nl = gguf_meta.get("%s.block_count" % _arch, 0) or 0
                if not _nl and expected_seq:
                    _nl = 1 + max((e["L"] for e in expected_seq), default=-1)
                layer_graph = expand_topology_to_layers(_skel, _nl)

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

    # Baked-relative indices where a new unit starts (reset the order-map pointer here so
    # each unit re-aligns to the expected sequence head). Regime-computed: decode = per-token
    # boundaries, prefill = a single span {0}.
    tok_boundary_idx = win.tok_boundary_idx

    # GPU slices in the baked span (relative ns from t0). If a GGUF sequence was
    # built, order-map each mul_mat_vec dispatch to its expected weight tensor
    # (lockstep by execution order within a token), guarded on launched N == the
    # weight's true ne[1] so a shape mismatch is reported, not silently attached.
    gpu_slices = []
    busy_ns = 0
    fam_busy = defaultdict(float)
    fam_count = defaultdict(int)
    # Dump shape-lookup: (kernel_family, launched-N) -> (K, quant). Built from the roofline
    # dump, which carries exact (N,K,quant) for EVERY op. This lets us label a dispatch's
    # KxN shape WITHOUT the order-map (which only covers ~26% on hybrid GDN models) -- the
    # (family, N) key is unique to K in practice (0 ambiguous combos on qwen3.6-35b), so a
    # dispatch's launched N + kernel family pins its K. Used to fill the detail table's
    # shape column for UNMAPPED dispatches (needed for kernel-shape-level optimization).
    dump_shape = {}
    if layer_graph and layer_graph.get("nodes"):
        _ds_tmp = {}
        for nd in layer_graph["nodes"]:
            if "MUL_MAT" not in str(nd.get("op", "")):
                continue
            N, K = nd.get("N"), nd.get("K")
            if not (N and K):
                continue
            ks = nd.get("kernels") or []
            fam_nd = family_of(ks[-1]["fam"]) if ks and ks[-1].get("fam") else nd.get("fam", "")
            fam_nd = fam_nd or nd.get("fam", "")
            key = (fam_nd, N)
            # only keep unambiguous (family,N)->K (drop the rare collision rather than guess).
            # None marks a known-ambiguous key; once marked it stays ambiguous.
            if key not in _ds_tmp:
                _ds_tmp[key] = (K, (nd.get("quant") or "").upper() or None)
            elif _ds_tmp[key] is not None and _ds_tmp[key][0] != K:
                _ds_tmp[key] = None
        dump_shape = {k: v for k, v in _ds_tmp.items() if v is not None}
    fam_macs = defaultdict(float)   # prefill: total 2*N*K*B MACs of a family's mapped slices
    # Launched-N values that a GGUF weight actually has -> a dispatch whose N is not in
    # this set cannot be a weight matvec (e.g. the ssm scalar F32 projections), so it is
    # excluded from the map denominator instead of counting as an unmapped miss.
    _expected_Nset = {ent["N"] for ent in expected_seq}
    ei = 0
    ti_ctr = 0
    mv_total = mv_mapped = 0
    for idx, (s, e, name, ncol, nblk, gy) in enumerate(baked):
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
              "blocks": nblk, "ti": ti_ctr,
              # launched output-row count + whether this is a matvec, so layer
              # segmentation can find the output head (largest-N matvec) structurally.
              "_ncol": ncol, "_mv": (mm_key in name)}
        # Attach the dump-derived KxN shape to EVERY matvec dispatch (order-map-independent),
        # so the detail table can show shape even for unmapped kernels.
        if (mm_key in name) and ncol:
            sk = dump_shape.get((fam, ncol))
            if sk:
                sl["shapeK"] = sk[0]
                sl["shapeQ"] = sk[1]
        ti_ctr += 1
        if expected_seq and mm_key in name and ncol:
            # Only count dispatches that COULD map to a weight in the denominator. A
            # hybrid (GDN/SSM) model fires small F32 mul_mat_vec_f scalar projections
            # (ssm_alpha/beta and per-token gates, launched N=1/32) that are matmuls but
            # not GGUF weight matvecs -- they have no expected entry by construction, so
            # counting them deflates the map %. Keep a dispatch in the denominator only
            # if its launched N appears among the expected weight Ns.
            if ncol not in _expected_Nset:
                gpu_slices.append(sl)
                continue
            mv_total += 1
            # Resync order-map (Level 1 type guard + Level 2 MoE guard). Instead of a
            # strict positional ei++ that cascades a desync forever, match this dispatch
            # to the FIRST expected entry at-or-after ei that is CONSISTENT with the
            # kernel that actually ran, then advance past it. Consistency:
            #  - type: a mul_mat_vec_q<(ggml_type)8> (Q8_0) kernel can't compute a Q5_K
            #    weight; mul_mat_vec_f (float router) carries no quant -> skip the check.
            #  - MoE: an expert weight (is_moe, 3D) runs as ONE grouped mul_mat_id with
            #    gy>1; a dense dispatch (gy==1) must map to a dense entry, and vice versa.
            #  - N: the launched output-row count must equal the weight's true N (the
            #    original join key), so same-type siblings don't false-match.
            # Scanning a bounded window self-corrects after an unmatched dispatch (a
            # kernel we did not model) without derailing the rest of the token.
            ktype = _kernel_quant_type(name)
            dispatch_is_moe = gy > 1
            # This dispatch's ACTUAL K, from the dump's unambiguous (family,N)->K lookup
            # (attached as shapeK at slice construction). Two different-K weights can share
            # an N (ffn_down_shexp K=512 and attn_output K=4096 both N=2048), and once K=512
            # moved to the k512_fast kernel the base kernel's N=2048 dispatches are ALL
            # K=4096 -- but a pure-N match would still slot ffn_down_shexp (K=512) onto them,
            # producing a nonsense row (42us K=4096 dispatch labelled with the 1.1MB K=512
            # weight -> over-fetch 8x). Require the candidate weight's K to equal the
            # dispatch's real K whenever we know it.
            dispatch_k = sl.get("shapeK")
            ent = None
            scan_max = min(len(expected_seq), ei + 8)
            j = ei
            while j < scan_max:
                cand = expected_seq[j]
                type_ok = (ktype is None or cand["q"] is None or ktype == cand["q"])
                moe_ok = (dispatch_is_moe == bool(cand.get("is_moe")))
                k_ok = (dispatch_k is None or cand.get("K") is None
                        or cand["K"] == dispatch_k)
                if type_ok and moe_ok and k_ok and cand["N"] == ncol:
                    ent = cand
                    ei = j + 1
                    break
                j += 1
            if ent is None:
                # no consistent entry in the window: advance one so we don't stall, but
                # leave this dispatch unmapped rather than attach a wrong-type weight.
                ei = min(ei + 1, len(expected_seq))
            if ent is not None:
                true_n = ent["N"]
                k = ent["K"]
                packed = ent["bytes"]
                # Measured DRAM bytes, best source first:
                #  1. per-weight order-mapped bytes (exact, when the order-map segmented);
                #  2. NEAREST fetch-size CLUSTER for (fam, launched-N): two different-K
                #     weights that share an N have distinct, tightly-clustered fetch sizes,
                #     so pick the cluster whose center is closest to THIS weight's packed
                #     footprint -- this separates e.g. ffn_down_shexp (K=512, ~1.1MB) from
                #     attn_output (K=4096, ~8.9MB) that both launch N=2048, instead of the
                #     (fam,N) mean that blended them into a bogus 4.5x over-fetch;
                #  3. the (fam,N) blend; 4. the family mean.
                mexact = ent["nm"] in fetch_by_name
                measured = fetch_by_name.get(ent["nm"])
                mnearest = False
                if not measured:
                    clusters = fetch_bytes_nk.get((fam, ncol))
                    if clusters and packed:
                        measured = min(clusters, key=lambda c: abs(c - packed))
                        mnearest = True
                if not measured:
                    measured = fetch_bytes_n.get((fam, ncol)) or fetch_bytes.get(fam, 0)
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
                    "mexact": mexact or mnearest,
                    # Over-fetch: measured DRAM bytes / theoretical packed footprint.
                    "overfetch": (round(measured / packed, 2)
                                  if (measured and packed) else 0),
                    # Effective (useful-work) bandwidth: the THEORETICAL bytes this
                    # matvec must move / its kernel time. Immune to over-fetch by
                    # construction (numerator is the algorithmic minimum, not measured
                    # traffic). 1 byte/ns == 1 GB/s. NOTE: these raw per-dispatch values
                    # divide by THIS one dispatch's duration, which on gfx1151 is subject
                    # to the sub-100us HIP-graph timestamp smear (a fraction of dispatches
                    # read artificially short/long -- see project memory), so a single
                    # dispatch can report an impossible >roofline BW. The displayed effbw
                    # is REPLACED below by a robust (fam,N,K)-median-duration recompute; we
                    # do NOT clamp to roofline (a real >roofline median would be a signal
                    # worth seeing, not something to hide). Raw kept for reference/debug.
                    "effbw_raw": round(packed / dur, 1) if dur else 0,
                    "effbw": round(packed / dur, 1) if dur else 0,
                    "effbw_pct": (round(packed / dur / peak_bw * 100, 1)
                                  if dur else 0),
                    # this dispatch's own duration (ns), so the post-pass can group
                    # same-(fam,N,K) slices and take a robust median duration.
                    "_dur": dur,
                    # Prefill compute roofline: this matmul does 2*N*K*B MACs over B
                    # prompt tokens; achieved TOPS = 2*N*K*B / kernel_time (1 MAC/ns ==
                    # 1e-3 TOPS -> /1e3). tops_pct rooflines against peak TOPS. Uses
                    # true N,K (algorithmic work, over-fetch/padding-immune). 0 for
                    # decode (B=1, BW-bound) or when batch/peak unknown.
                    "tops": (round(2.0 * true_n * k * compute_batch / dur / 1e3, 2)
                             if (dur and compute_batch) else 0),
                    "tops_pct": (round(2.0 * true_n * k * compute_batch / dur / 1e3
                                       / peak_tops * 100, 1)
                                 if (dur and compute_batch and peak_tops) else 0),
                    "nmatch": (true_n == ncol),
                }
                if true_n == ncol:
                    mv_mapped += 1
                if compute_batch:
                    fam_macs[fam] += 2.0 * true_n * k * compute_batch
        gpu_slices.append(sl)

    # --- Robust effbw: recompute per-slice effective BW from the MEDIAN duration of
    # all dispatches sharing the same (family, trueN, K) shape, not this one dispatch's
    # raw duration. Why: the kernel-trace grid encodes only N (output rows), so several
    # distinct-K weights land under one kernel label, AND gfx1151's sub-100us HIP-graph
    # timestamp smear makes a fraction of dispatches read artificially short/long. Both
    # effects let a single dispatch divide a correct byte count by a corrupted duration
    # and report an impossible >roofline BW (e.g. 8.9MB / 8us ~= 1079 GB/s vs the ~40us
    # true cost). Grouping by (fam,N,K) separates the different-K populations, and the
    # median is robust to the timestamp outliers. NO roofline clamp: a genuine
    # >roofline median is left visible as a signal, not hidden.
    # Group durations by the slice's ACTUAL launched N (_ncol, from the trace grid) +
    # family, NOT the order-mapped weight's trueN. Critical: when the order-map mis-
    # attaches a weight to a dispatch (the N-only-ambiguity / 25%-accuracy problem),
    # the weight's trueN != the dispatch's real launch N. Keying the median on trueN
    # then averages durations of DIFFERENTLY-launched dispatches -- e.g. an attn_gate
    # weight (trueN=4096, packed 8.9MB) mis-attached to N=2048 (~15us) dispatches gave
    # 8.9MB/15us = 583 GB/s (2.5x roofline). Keying on _ncol clusters like-launched
    # dispatches so the median duration is physically consistent with the launch.
    _eb_durs = defaultdict(list)
    for sl in gpu_slices:
        m = sl.get("map")
        if m and m.get("packed") and m.get("_dur"):
            _eb_durs[(sl["fam"], sl.get("_ncol"))].append(m["_dur"])
    _eb_med = {kk: sorted(v)[len(v) // 2] for kk, v in _eb_durs.items()}

    # AUTHORITATIVE effbw source: the roofline dump (--graph-json). Each dump op carries
    # EXACT (N,K,quant) + its own mul_mat kernel gpu_time -- no positional order-map, no
    # trace N-ambiguity. When present, this is the trustworthy per-shape kernel time to
    # roofline against (e.g. attn_gate 4096x2048 q8_0 -> 41us -> 215 GB/s, vs the trace
    # order-map's bogus 15us/583 GB/s). Build {(N,K,quant): median mul_mat us} from the
    # reconstructed graph nodes; the effbw loop below prefers it over the trace median.
    _dump_us = {}
    _dump_experts = {}
    if layer_graph and layer_graph.get("nodes"):
        _dump_durs = defaultdict(list)
        for nd in layer_graph["nodes"]:
            if "MUL_MAT" not in str(nd.get("op", "")):
                continue
            N, K = nd.get("N"), nd.get("K")
            if not (N and K):
                continue
            # the op's mul_mat kernel time (skip the quantize-prep subkernel): prefer a
            # subkernel whose family names a matvec/gemm, else the op's total us_in.
            mm = [sk["us"] for sk in (nd.get("kernels") or [])
                  if "mul_mat" in sk.get("fam", "") or "wvsplitk" in sk.get("fam", "")
                  or "gemm" in sk.get("fam", "").lower()]
            us = sum(mm) if mm else ((nd.get("us_in") or 0) / 1000.0)
            if us > 0:
                _dump_durs[(N, K, nd.get("quant", ""))].append(us)
        _dump_us = {kk: sorted(v)[len(v) // 2] for kk, v in _dump_durs.items()}
        # (N,K,quant) -> experts routed per grouped MoE dispatch (top_k). A grouped
        # mul_mat_id reads experts_used experts in one dispatch, so its effective packed
        # footprint is per_expert_bytes * experts_used, NOT one expert (which under-reported
        # MoE eff BW ~top_k-fold: Q4_K ffn_gate_exps 2048x512 read 5% roofline instead of ~42%).
        for nd in layer_graph["nodes"]:
            eu = nd.get("experts_used") or 0
            N, K = nd.get("N"), nd.get("K")
            if eu > 0 and N and K:
                _dump_experts[(N, K, (nd.get("quant") or "").lower())] = eu

    def _quant_norm(q):
        # trace/gguf quant label ("Q8_0") vs dump label ("q8_0") -> compare case-insensitively
        return (q or "").lower()

    def _eff_packed(m):
        # Effective bytes the DISPATCH streams = per-expert packed x experts_used for a
        # grouped MoE op, else packed for dense. (m["packed"] for MoE is already the
        # per-ONE-expert 2D footprint, see _seq_entry.)
        p = m.get("packed") or 0
        eu = _dump_experts.get((m.get("trueN"), m.get("K"), _quant_norm(m.get("q"))))
        if eu and eu > 1:
            return p * eu
        return p

    # Attach the DUMP's true per-op kernel time to every matvec slice. The trace Start->End
    # duration is WRONG for small-grid kernels (e.g. longk, grid.x=nrows_x=512 workgroups,
    # 77% occupancy): the scheduler co-runs them with neighbouring kernels, so their
    # timestamp span brackets ~100us of overlapping OTHER work while the kernel itself is
    # active only ~11us (confirmed by GRBM_GUI_ACTIVE). The dump runs graphs-disabled ->
    # each op's kernel time is its OWN gpu_time, overlap-free. Use it as the displayed
    # kernel time when available; the raw trace span is kept as _dur_raw for reference.
    for sl in gpu_slices:
        if not sl.get("_mv"):
            continue
        K = sl.get("shapeK"); N = sl.get("_ncol"); qn = _quant_norm(sl.get("shapeQ"))
        du = _dump_us.get((N, K, qn)) if (K and N and qn) else None
        if du:
            sl["dispatch_us"] = round(du, 2)

    for sl in gpu_slices:
        m = sl.get("map")
        if not (m and m.get("packed") and m.get("_dur")):
            continue
        # effective packed = per-expert packed x experts_used for grouped MoE ops, else packed.
        _ep = _eff_packed(m)
        m["eff_packed"] = _ep       # JS per-row eff BW divides THIS by the row's kernel time
        _eu = _dump_experts.get((m.get("trueN"), m.get("K"), _quant_norm(m.get("q"))))
        if _eu and _eu > 1:
            m["experts_used"] = _eu   # routed experts this grouped dispatch computed (top_k)
        # Prefer the authoritative dump time (exact shape, no order-map) when available.
        dump_med_us = _dump_us.get((m.get("trueN"), m.get("K"), _quant_norm(m.get("q"))))
        if dump_med_us:
            m["effbw"] = round(_ep / (dump_med_us * 1000.0), 1)
            m["effbw_pct"] = round(_ep / (dump_med_us * 1000.0) / peak_bw * 100, 1) if peak_bw else 0
            m["effbw_med_us"] = round(dump_med_us, 2)
            m["effbw_src"] = "roofline-dump (exact shape)"
            continue
        # Fallback (no dump): trace-median keyed on the actual launched N, gated on nmatch.
        # A weight-N vs launched-N mismatch means packed and duration belong to different
        # ops -> meaningless; leave unset and flag rather than fabricate.
        if not m.get("nmatch"):
            m["effbw"] = 0
            m["effbw_pct"] = 0
            m["effbw_mismatch"] = True
            continue
        med = _eb_med.get((sl["fam"], sl.get("_ncol")))
        if not med:
            continue
        m["effbw"] = round(_ep / med, 1)
        m["effbw_pct"] = round(_ep / med / peak_bw * 100, 1) if peak_bw else 0
        m["effbw_n"] = len(_eb_durs[(sl["fam"], sl.get("_ncol"))])
        m["effbw_med_us"] = round(med / 1000.0, 2)
        m["effbw_src"] = "trace order-map (median dur)"

    # UNMAPPED matvec dispatches: packed size and eff BW are pure functions of (N, K, quant)
    # + the dump's per-shape kernel time -- NONE of which need the order-map (which only
    # attaches the weight NAME). shapeK/_ncol/shapeQ were attached at slice construction
    # from the dump's unambiguous (family,N)->K lookup. So compute packed + eff BW for every
    # unmapped matvec too (F32 GDN projections have no packed-weight roofline -> skip those).
    for sl in gpu_slices:
        if sl.get("map") or not sl.get("_mv"):
            continue
        K = sl.get("shapeK"); N = sl.get("_ncol"); qs = sl.get("shapeQ")
        if not (K and N and qs) or qs.upper() == "F32":
            continue
        gt = _quant_name_to_gt(qs)
        packed = _gguf_packed_bytes([K, N], gt)
        dump_med_us = _dump_us.get((N, K, _quant_norm(qs)))
        med_us = dump_med_us if dump_med_us else (
            (_eb_med.get((sl["fam"], N)) or 0) / 1000.0)
        sh = {"K": K, "trueN": N, "launchN": N, "q": qs.upper(), "packed": packed,
              "role": "", "nm": "", "L": -1, "shape_only": True}
        if med_us:
            sh["effbw"] = round(packed / (med_us * 1000.0), 1)
            sh["effbw_pct"] = round(packed / (med_us * 1000.0) / peak_bw * 100, 1) if peak_bw else 0
            sh["effbw_med_us"] = round(med_us, 2)
            sh["effbw_src"] = ("roofline-dump (exact shape)" if dump_med_us
                               else "trace median (shape-keyed)")
        sl["map"] = sh

    map_stats = ({"total": mv_total, "mapped": mv_mapped,
                  "pct": round(100.0 * mv_mapped / mv_total, 1) if mv_total else 0,
                  "seq_len": len(expected_seq)}
                 if expected_seq else None)

    # GPU-busy = wall time with >=1 kernel running (merged-interval UNION), NOT the
    # sum of per-dispatch durations. Summing double-counts overlapping dispatches --
    # in EAGER mode (graphs disabled, e.g. MoE mul_mat_id over the mmvq batch cap) the
    # trace has back-to-back/overlapping kernels, so the sum can exceed the window span
    # (>100% "busy"). The union is overlap-correct in both graph and eager mode.
    busy_ns = 0
    _iv = sorted(((sl["s"], sl["e"]) for sl in gpu_slices))
    _cs = _ce = None
    for _s, _e in _iv:
        if _cs is None:
            _cs, _ce = _s, _e
        elif _s <= _ce:
            _ce = max(_ce, _e)
        else:
            busy_ns += _ce - _cs
            _cs, _ce = _s, _e
    if _cs is not None:
        busy_ns += _ce - _cs

    # Per-kernel steady-state stats: decode tokens are structurally identical, so
    # the Nth kernel of every token is the same dispatch. Aggregate each within-token
    # position (ti) across ALL post-warmup tokens in the full stream (not just the
    # baked view) so the selected-kernel panel can show a stable mean +/- spread
    # instead of one noisy single-dispatch duration (per-token jitter includes the
    # once-per-token host-serialized GDN edge, launch bubbles, interrupt latency).
    # Keyed "ti|family" so a rare token with a different kernel count self-segregates
    # rather than blending mismatched positions. Durations are ns (JS renders us).
    # Per-position kernel-duration stats: regime-computed (decode aggregates over the
    # periodic per-token repeats; prefill's single pass has none -> empty).
    kstats = win.kstats
    kstats_ntok = win.kstats_ntok

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
        n_layers = 1 + max(_roles) if _roles else 0
        # Window edges: each token boundary starts a window that runs to the next
        # boundary (decode), and the last window runs to the end of the baked slices.
        # Prefill has a single boundary {0}, so this is one window spanning the whole
        # forward pass -- append n as the closing edge so that window is processed.
        n = len(gpu_slices)
        starts = sorted(tok_boundary_idx) + [n]

        # STRUCTURAL layer anchor: each decode block runs an identical kernel sequence,
        # so a kernel family that fires exactly once per block (e.g. topk_moe_cuda for a
        # MoE model, or ssm_conv/rope for others) yields n_layers clean per-layer starts.
        # This is INDEPENDENT of the order-map, so layers segment correctly even when the
        # map is sparse (MoE mul_mat_id maps poorly) -- avoiding the failure where all the
        # unmapped slices collapse into one giant "layer". Pick the anchor per-window as a
        # family whose count in that window == n_layers; fall back to the map-fill below
        # when no such family exists (dense models where the map is dense anyway).
        def _anchor_starts(st, en):
            if n_layers < 2:
                return None
            fam_pos = defaultdict(list)
            for i in range(st, en):
                fam_pos[gpu_slices[i]["fam"]].append(i)
            # candidate anchors: exactly n_layers occurrences, and the LAST op family
            # (output head region) excluded. Prefer the one whose positions are most
            # evenly spaced (a true per-layer marker), tie-break by name stability.
            cands = [p for f, p in fam_pos.items() if len(p) == n_layers]
            if not cands:
                return None
            def spread(p):
                gaps = [p[i] - p[i - 1] for i in range(1, len(p))]
                mean = sum(gaps) / len(gaps)
                return sum((g - mean) ** 2 for g in gaps)  # variance*len; lower=evener
            return min(cands, key=spread)

        lay_L = [None] * n
        used_anchor = False
        for wi in range(len(starts) - 1):
            st = starts[wi]
            en = min(starts[wi + 1], n)
            anchor = _anchor_starts(st, en)
            if anchor is not None:
                used_anchor = True
                # Layer L owns [anchor[L] .. anchor[L+1]); slices before the first anchor
                # (leading norms/get_rows of block 0) belong to L0; the tail after the
                # last anchor's block (output head + final norm) is the head (L=-1).
                # Assign each anchor a block index by its ORDER (0..n_layers-1), matching
                # the GGUF block order that block_kind is keyed on.
                for i in range(st, en):
                    lay_L[i] = None
                # first anchor's block starts at st (its leading norms)
                for li in range(len(anchor)):
                    seg_lo = st if li == 0 else anchor[li]
                    seg_hi = anchor[li + 1] if li + 1 < len(anchor) else en
                    for i in range(seg_lo, seg_hi):
                        lay_L[i] = li
                # Split off the OUTPUT HEAD: the final norm + lm_head (vocab projection)
                # are not part of the last transformer block. The lm_head is the matvec
                # with the largest launched N in the window (>> any weight N); mark it and
                # everything after it (plus the final norm right before it) as head (L=-1)
                # so the huge lm_head kernel doesn't inflate the last layer's width.
                head_i = None
                head_n = 0
                for i in range(anchor[-1], en):
                    if gpu_slices[i].get("_mv") and gpu_slices[i].get("_ncol", 0) > head_n:
                        head_n = gpu_slices[i]["_ncol"]; head_i = i
                if head_i is not None:
                    # include the final norm(s) immediately preceding the head
                    edge = head_i
                    for k in range(head_i - 1, anchor[-1] - 1, -1):
                        if "norm" in gpu_slices[k].get("fam", "").lower():
                            edge = k
                        elif gpu_slices[k].get("_mv"):
                            break
                    for i in range(edge, en):
                        lay_L[i] = -1
            else:
              # --- map-driven fallback (dense models: order-map is dense) ---
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
              # Boundary correction to match the ggml compute graph (tensor-name layer
              # index): pure backward-fill pulls a layer's TRAILING residual binops (the
              # expert weighted-sum + residual adds, e.g. ffn_moe_weighted/l_out) into the
              # NEXT layer, because they physically precede that layer's first matvec. But
              # by the graph those belong to the layer that just finished. So at each
              # inter-layer gap, keep the trailing non-matvec slices with the PREVIOUS
              # layer; the new layer starts at its attention-input norm (the rms_norm that
              # feeds the next matvec). Scan each boundary where the mapped layer increases.
              for i in range(st + 1, en):
                prevL, curL = lay_L[i - 1], lay_L[i]
                if prevL is None or curL is None or curL <= prevL:
                    continue
                # find the first mapped matvec of the new layer curL
                fm = None
                for k in range(i, en):
                    mp = gpu_slices[k].get("map")
                    if mp is not None and mp["L"] == curL:
                        fm = k
                        break
                if fm is None:
                    continue
                # the new layer starts at the last norm before that matvec; everything
                # before it in this gap stays with the previous layer.
                norm_at = None
                for k in range(fm - 1, i - 1, -1):
                    if "norm" in gpu_slices[k].get("fam", "").lower():
                        norm_at = k
                        break
                edge = norm_at if norm_at is not None else fm
                for k in range(i, edge):
                    lay_L[k] = prevL
            # coalesce consecutive equal-layer runs into one segment (both paths).
            j = st
            while j < en:
                L = lay_L[j]
                k = j
                while k < en and lay_L[k] == L:
                    k += 1
                kind = "head" if L == -1 else block_kind.get(L, "?")
                name = "head" if L == -1 else ("L%d %s" % (L, kind))
                layers.append({"s": gpu_slices[j]["s"], "e": gpu_slices[k - 1]["e"],
                               "kind": kind, "name": name, "L": L})
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
    # Prefill bakes one forward pass (a single span); decode bakes (hi-lo) tokens.
    ntok_baked = win.ntok_baked   # decode tokens baked; 1 for prefill (regime)

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
            # Prefill compute roofline (family-level): total algorithmic MACs of this
            # family's mapped matmuls / total family kernel time -> achieved TOPS,
            # rooflined vs peak TOPS. 0 for decode or unmapped families.
            "tops": (round(fam_macs[fam] / fam_busy[fam] / 1e3, 1)
                     if fam_macs.get(fam) and fam_busy[fam] else 0),
            "tops_pct": (round(fam_macs[fam] / fam_busy[fam] / 1e3 / peak_tops * 100, 1)
                         if fam_macs.get(fam) and fam_busy[fam] and peak_tops else 0),
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
                          "tops": summary_i["tops"], "tops_pct": summary_i["tops_pct"],
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
    if args.graph_json:
        regen_parts.append("--graph-json " + os.path.abspath(args.graph_json))
    if args.gguf:
        regen_parts.append("--gguf " + os.path.abspath(args.gguf))
    if args.arch != DEFAULT_ARCH:
        regen_parts.append("--arch " + args.arch)
    if args.peak_bw:
        regen_parts.append("--peak-bw %g" % args.peak_bw)
    if args.peak_tops:
        regen_parts.append("--peak-tops %g" % args.peak_tops)
    if args.mode != "decode":
        regen_parts.append("--mode " + args.mode)
    if args.tokens != 2:
        regen_parts.append("--tokens %d" % args.tokens)
    if args.skip_tokens != 30:
        regen_parts.append("--skip-tokens %d" % args.skip_tokens)
    if args.context_tokens:
        regen_parts.append("--context-tokens %d" % args.context_tokens)
    if args._gap_threshold_set and args.gap_threshold_us != 150.0:
        regen_parts.append("--gap-threshold-us %g" % args.gap_threshold_us)
    if args.boundary_method:
        regen_parts.append("--boundary-method " + args.boundary_method)
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

    # Prefill TTFT estimate == the GPU prompt-eval floor, computed the SAME way the
    # llamacpp regression harness reports ttft_ms_estimate: n_prompt / prefill_tps.
    # n_prompt comes from the clean_tps test label (e.g. "pp128" -> 128); prefill_tps
    # is the untraced clean pp tok/s. Excludes tokenize/first-sample/scheduling, so
    # it is a floor, not the server-measured end-to-end TTFT. None if unavailable.
    ttft_est_ms = None
    if args.mode == "prefill" and clean_tps and clean_tps.get("tps", 0) > 0:
        _m = re.search(r"(\d+)", clean_tps.get("test", "") or "")
        _np = int(_m.group(1)) if _m else 0
        if _np > 0:
            ttft_est_ms = _np / clean_tps["tps"] * 1000.0

    payload = {
        "title": title,
        "model_name": model_name,
        "provenance": _provenance(),
        "mode": args.mode,
        "ttft_est_ms": ttft_est_ms,
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
        "att_util_pool": att_util_pool,
        "dbg_shortcuts": _DEBUG_SHORTCUTS_HTML,
        "graph_shortcuts": _GRAPH_SHORTCUTS_HTML,
        "has_att_code": bool(att_code_by_fam),
        # RDNA3.5 ISA one-line opcode glossary (mnemonic -> description), embedded
        # only when the debug view exists, so the view can show a hover tooltip
        # explaining each raw instruction. Keyed on the lowercased first token.
        "isa_gloss": ISA_GLOSSARY if att_code_by_fam else {},
        # Special-register / wait-counter glossary (operand token -> description),
        # so hovering vmcnt/lgkmcnt/SCC/EXEC/VCC/M0 etc. in an ISA line explains it.
        "reg_gloss": REG_GLOSSARY if att_code_by_fam else {},
        # Concept glossary (superblock, etc.) -- embedded unconditionally since the
        # tiling schematic (which uses it) only needs a GGUF-mapped shape, not ATT.
        "concept_gloss": CONCEPT_GLOSSARY,
        "att_cmd": att_cmd,
        # Live-tracing mode: false for the static export; serve.py flips this true
        # so the client shows a "Run Trace" button alongside the copy command.
        "att_server": False,
        "has_map": bool(expected_seq),
        "map_stats": map_stats,
        "kv_bytes_per_tok": kv_bytes_per_tok,
        "kv_ctx": kv_ctx,
        "layers": layers,
        "has_layers": bool(layers),
        "phases": phases,
        "has_phases": bool(phases),
        "layer_graph": layer_graph,
        "has_layer_graph": bool(layer_graph),
        "kstats": kstats,
        "has_kstats": bool(kstats),
        "kstats_ntok": kstats_ntok,
        "clean_tps": clean_tps,
        "peak_bw_gbs": peak_bw,
        "peak_tops": peak_tops or 0,
        # Prefill batch (prompt length B) behind the compute roofline; 0 in decode.
        "compute_batch": compute_batch,
        # MMQ output-row tile height (mmq_y) for the prefill GEMM tiling schematic.
        "mmq_y": mmq_y_for(args.arch),
        # Prefill leads with the compute roofline (peak TOPS); decode with DRAM BW.
        "compute_bound": is_prefill,
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


def _print_payload_summary(payload):
    if payload.get("mode") == "prefill":
        print(f"  prefill: 1 forward pass ({len(payload['gpu'])} GPU slices, "
              f"{len(payload['cpu'])} CPU calls).", file=sys.stderr)
    else:
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


def _alt_args(args):
    """A shallow copy of args with the --alt-* regime fields promoted to the primary
    input fields, so the same build_payload() produces the second regime's payload."""
    a = copy.copy(args)
    a.mode = args.alt_mode or ("decode" if args.mode == "prefill" else "prefill")
    a.kernel_csv = args.alt_kernel_csv
    a.hip_csv = args.alt_hip_csv
    a.pmc_csv = args.alt_pmc_csv
    a.fetch_csv = args.alt_fetch_csv
    a.clean_tps_file = args.alt_clean_tps_file
    return a


def build_bundle(args):
    """Return the render bundle. With no --alt-kernel-csv this is a single payload
    (backward-compatible). With one, it is {"payloads": {mode: payload, ...},
    "default_mode": args.mode} so the client can switch regimes via the dropdown."""
    primary = build_payload(args)
    if not getattr(args, "alt_kernel_csv", None):
        return primary
    alt = build_payload(_alt_args(args))
    return {"multi": True,
            "provenance": primary.get("provenance"),
            "default_mode": primary["mode"],
            "payloads": {primary["mode"]: primary, alt["mode"]: alt}}


def write_overlay(args):
    bundle = build_bundle(args)
    html = render_html(bundle)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}", file=sys.stderr)
    if isinstance(bundle, dict) and bundle.get("multi"):
        for m, p in bundle["payloads"].items():
            print(f" [{m}]", file=sys.stderr)
            _print_payload_summary(p)
    else:
        _print_payload_summary(bundle)


def render_html(bundle):
    # Escape "<" so any embedded markup in the payload (e.g. the child debug window's
    # help HTML, which contains a </script> close tag) cannot terminate the parent
    # <script> block early. JSON "<" decodes back to "<" in JS, so data is intact.
    data = json.dumps(bundle, separators=(",", ":")).replace("<", "\\u003c")
    # Inline the frontend JS from js/overlay.js (a real, lintable .js file -- see
    # tests/test_js_syntax.py) so the emitted HTML stays a single self-contained file.
    # The JS itself contains `const RAW = __DATA__;`, so inline it FIRST, then fill __DATA__.
    return (_HTML_TEMPLATE.replace("__OVERLAY_JS__", _overlay_js())
            .replace("__DATA__", data)
            .replace("__SHORTCUTS__", _MAIN_SHORTCUTS_HTML))


def _overlay_js():
    """Read the frontend JS bundle (inlined into the overlay at build time)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "js", "overlay.js")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# Mouse/keyboard help for the MAIN timeline overlay (see shortcuts_help_html).
_MAIN_SHORTCUTS_HTML = shortcuts_help_html("mainKeys", "Shortcuts -- timeline view", [
    ("Select", [
        ("click", "select one kernel (show detail panel)"),
        ("ctrl/cmd + click", "add / remove one kernel from the selection"),
        ("drag", "select kernels in a time range (lasso)"),
        ("ctrl/cmd + drag", "add the dragged range to the selection"),
        ("Esc", "clear the selection"),
    ]),
    ("Zoom / pan", [
        ("ctrl + wheel", "zoom horizontally (time) at the cursor"),
        ("+ / -", "zoom in / out"),
        ("shift + drag", "pan the time window"),
        ("shift + wheel", "pan horizontally (time)"),
        ("left / right", "pan by a quarter window"),
        ("shift + left / right", "pan by a full window"),
    ]),
    ("Measure (A/B markers)", [
        ("drag A or B", "move a marker; snaps to kernel edges"),
        ("alt + drag", "move a marker freely (no snap)"),
        ("double-click", "bring both markers into view"),
    ]),
    ("Trace (live-server mode)", [
        ("Run trace", "run ATT for the selected kernel on a free board, discarding "
                      "any on-disk trace first (fresh; picks up DWARF source lines)"),
    ]),
    ("General", [
        ("?", "open this help"),
    ]),
])

# Mouse/keyboard help for the CHILD debug window (ISA table + Occupancy View). This
# HTML is carried in the payload and injected into the child document's header.
_DEBUG_SHORTCUTS_HTML = shortcuts_help_html("dbgKeys", "Shortcuts -- trace view", [
    ("ISA table / step mode", [
        ("hover instruction", "show its ISA / register description"),
        ("click a row", "jump to its source line (when line info present)"),
        ("left / right", "step one instruction (executed order)"),
        ("n / p  or  j / k", "step next / previous instruction"),
        ("H / L", "jump to previous / next source line"),
        ("(Utilization view open)", "stepping moves a red playhead on the timeline"),
    ]),
    ("Occupancy view", [
        ("wheel", "scroll rows up / down"),
        ("ctrl + wheel", "zoom horizontally (time) at the cursor"),
        ("alt + wheel", "zoom vertically (rows)"),
        ("shift + wheel", "pan horizontally (time)"),
        ("drag", "pan the time window"),
        ("hover a wave", "show WG / SIMD / wave + cycle"),
        ("Esc", "close the Occupancy view"),
    ]),
    ("General", [
        ("?", "open this help"),
    ]),
])

# Mouse/keyboard help for the per-layer compute-graph popup. Embedded into the graph
# window's bar so it matches the "?" affordance on the other windows.
_GRAPH_SHORTCUTS_HTML = shortcuts_help_html("graphKeys", "Shortcuts -- layer graph", [
    ("Fusion analysis", [
        ("click a node", "add / remove it from the fusion set (register + occupancy verdict)"),
        ("click empty space", "clear the fusion set"),
        ("(select in timeline)", "highlights + centers the matching node here"),
    ]),
    ("Zoom / pan", [
        ("shift + drag", "pan the graph"),
        ("ctrl/alt + wheel", "zoom at the cursor"),
        ("Zoom + / -", "zoom in / out"),
        ("Fit", "fit the whole layer graph in view"),
    ]),
    ("General", [
        ("Esc", "close this window"),
        ("?", "open this help"),
    ]),
])


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
  /* Cap the right pane so a pathological kernel name (e.g. a ~290-char Tensile
     family) can't blow the summary table out to ~2000px and squeeze the timeline
     to zero. min-width:0 lets the flex item shrink; the table + cell rules below
     wrap the long name instead of forcing intrinsic width. */
  .right{flex:0 0 430px;max-width:430px;min-width:0;}
  /* Narrow (Windows) viewports: stack the panes instead of side-by-side. */
  @media (max-width:900px){
    .wrap{flex-direction:column;}
    .right{flex:1 1 auto;max-width:none;width:100%;}
  }
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
  /* Summary (family) table -- DECOUPLED from the shared auto-layout `table{}` rules
     (which caused repeated width regressions). Fixed layout with explicit numeric
     column widths: the 4 value columns get a guaranteed comfortable width (values are
     <=5 chars: counts, percentages, a short stall abbr), and the family column (no
     width set) absorbs ALL remaining pane width -- so it is as wide as possible while
     the numeric columns never wrap. A pathological ~290-char name wraps inside the
     family column (overflow-wrap) instead of forcing the pane wide. */
  #tbl{table-layout:fixed;}
  #tbl td:first-child,#tbl th:first-child{overflow-wrap:anywhere;word-break:break-word;}
  #tbl th:nth-child(2),#tbl td:nth-child(2),
  #tbl th:nth-child(3),#tbl td:nth-child(3),
  #tbl th:nth-child(4),#tbl td:nth-child(4){width:52px;white-space:nowrap;}
  #tbl th:nth-child(5),#tbl td:nth-child(5){width:50px;white-space:nowrap;}
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
  <h1 id="title" style="display:flex;align-items:center;gap:10px">
    <span id="titletext"></span>
    <select id="modesel" style="display:none;background:#2a2e3a;color:var(--fg);
      border:1px solid #3a3f4d;border-radius:6px;padding:3px 8px;font-size:13px;"></select>
  </h1>
  <div class="sub" id="sub"></div>
  <div class="sub" id="provfoot" style="margin-top:2px;opacity:.65;font-size:10px"></div>
</header>
<div class="wrap">
  <div class="left">
    <div class="bar">
      <button id="prev">&larr; Prev</button>
      <button id="next">Next &rarr;</button>
      <button id="zin">Zoom +</button>
      <button id="zout">Zoom &minus;</button>
      <button id="reset">Reset</button>
      <button id="markhome">Markers &rarr; view</button>
      <select id="findWhat" title="what to find">
        <option value="maxgap">largest intra-token gap</option>
        <option value="mineffbw">lowest eff-BW matvec (mmvq/mmq)</option>
      </select>
      <button id="findGo" title="find (click again for next-largest)">Find next</button>
      <button id="findPrev" title="previous (larger) match">Find prev</button>
      <button id="hwbtn" title="RDNA 3.5 WGP hardware reference" style="display:none">RDNA 3.5 HW</button>
      __SHORTCUTS__
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
      <h2 id="tblh2">Per-kernel-family/token (baked span)</h2>
      <table id="tbl"><thead><tr>
        <th>family</th><th id="tblcnt">cnt/tok</th><th id="tblmet">time%</th>
        <th id="tblmet2"></th><th>stall</th>
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
      <button id="hwclose">Close &times;</button>
    </div>
    <img id="hwimg" alt="RDNA 3.5 WGP: VGPR file, LDS banks, wave slots, gfx1151/gfx1150 constants"
      style="display:block;max-width:100%;height:auto;background:#fff;border-radius:3px">
    <div class="sub" style="margin-top:6px">The fusion-analysis panel models occupancy
      from these constants: 96 VGPR/wave = full 16-wave occupancy, 256 = scratch spill.</div>
  </div>
</div>
<script>
__OVERLAY_JS__
</script>
</body></html>
"""


if __name__ == "__main__":
    main()

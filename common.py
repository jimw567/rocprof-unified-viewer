"""Shared primitives used across the viewer's modules (main, regimes, loaders): the
GGML type/block tables, per-arch roofline peaks, stall-classification thresholds, and the
two hot helpers family_of() / dominant_stall(). These are the dependency HUB -- everything
needs family_of, and family_of needs _GGML_TYPES -- so they live in one leaf module with
no intra-package imports, which is what lets the other modules import it without cycles.
"""
import re

# --- ggml type tables --------------------------------------------------------
# ggml_type enum -> quant name (ggml.h). Keeps quant kernels distinct in family_of.
_GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    29: "IQ1_M", 30: "BF16", 39: "MXFP4",
}

# ggml_type -> (block elems, block bytes) for packed on-disk footprint.
_GGML_BLOCK = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2),
    # MXFP4 (type 39, gpt-oss MoE experts): QK_MXFP4=32 elems/block,
    # sizeof(block_mxfp4)=1 scale byte + 32/2 packed nibbles = 17 bytes.
    39: (32, 17),
}


def gguf_packed_bytes(ne, gt):
    """Packed on-disk byte footprint of a tensor with dims `ne` and ggml_type `gt`."""
    be, bb = _GGML_BLOCK.get(gt, (1, 4))
    n = 1
    for d in ne:
        n *= d
    return (n // be) * bb if be > 1 else n * bb


# --- per-GPU-arch roofline peaks ---------------------------------------------
# Peak DRAM bandwidth (GB/s), the roofline denominator. gfx1151 (Strix Halo) is 256-bit
# LPDDR5X-8000 = 256 GB/s theoretical, ~230 achievable. 1 B/ns == 1 GB/s.
PEAK_BW_GBS_BY_ARCH = {
    "gfx1151": 230.0,
}
# Peak compute (TOPS) for the fp16/int8 matmul path. gfx1151 = ~43 TOPS via WMMA.
PEAK_TOPS_BY_ARCH = {
    "gfx1151": 43.0,
}
# MMQ prefill GEMM output-row tile height (mmq_y in ggml-cuda/mmq.cuh): launched N is
# recovered as (Grid_Size_X/Workgroup_Size_X) * mmq_y. RDNA3.5 (gfx115x) = 64.
MMQ_Y_BY_ARCH = {
    "gfx1151": 64, "gfx1150": 64, "gfx1152": 64, "gfx1153": 64,
}
DEFAULT_ARCH = "gfx1151"


def peak_bw_for(arch, override=None):
    if override:
        return float(override)
    return PEAK_BW_GBS_BY_ARCH.get(arch, PEAK_BW_GBS_BY_ARCH[DEFAULT_ARCH])


def mmq_y_for(arch):
    return MMQ_Y_BY_ARCH.get(arch, 64)


def peak_tops_for(arch, override=None):
    if override:
        return float(override)
    return PEAK_TOPS_BY_ARCH.get(arch)


# --- stall classification ----------------------------------------------------
# Derived from gfx1151 4B decode PMC: mul_mat_vec_q = MemBusy 77 / L2 8 (memory);
# elementwise kernels sit low on everything; LDS bank conflicts ~0 on this arch.
MEM_BUSY_HI = 25.0     # MemUnitBusy% >= this + low L2 hit => memory-bound
L2_HIT_LO = 30.0       # L2CacheHit% <= this => traffic misses to VRAM
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


def dominant_stall(counters):
    """Classify a family's dominant stall from its mean PMC counters."""
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


def family_of(kernel_name):
    """Normalize a mangled/templated kernel name to a family (same aggregation used when
    collecting PMC, so PMC families join onto trace slices). Quantized kernels whose first
    template arg is (ggml_type)N keep the quant type so Q4_K vs Q6_K are distinct families;
    generic elementwise kernels carrying op_ in the template keep the op so add/mul/sub
    stay distinct."""
    short = re.sub(r"<.*", "", kernel_name).split("(")[0]
    short = short.split("void ")[-1].strip()
    m = re.search(r"<\s*\(ggml_type\)(\d+)", kernel_name)
    if m:
        n = int(m.group(1))
        short += "[" + _GGML_TYPES.get(n, "type%d" % n) + "]"
    mo = re.search(r"op_([a-z]+)\s*\(", kernel_name)
    if mo:
        short += "[" + mo.group(1) + "]"
    return short

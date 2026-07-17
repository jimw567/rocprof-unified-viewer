#!/usr/bin/env python3
"""Tally per-lane memory-load instruction widths per kernel family from the
gfx1151 device code objects extracted from libggml-hip.so. Emits JSON keyed by
the same family_of() names the viewer uses, so it can be joined onto slices via
rocprof-unified-viewer --loadwidth-json.

Categories:
  vector global/buffer/flat loads -> per-lane bytes (b32=4, d16=2, u8=1, ...)
  scalar s_load_* -> uniform (per-wave), counted separately
  ds_* -> LDS (not DRAM), counted separately

The llvm-nm / llvm-objdump / llvm-cxxfilt tools default to /opt/rocm/llvm/bin but
are overridable via LLVM_NM / LLVM_OBJDUMP / LLVM_CXXFILT so this can point at a
local ROCm install.
"""
import glob, json, re, subprocess, sys, os, collections

NM = os.environ.get("LLVM_NM", "/opt/rocm/llvm/bin/llvm-nm")
OBJDUMP = os.environ.get("LLVM_OBJDUMP", "/opt/rocm/llvm/bin/llvm-objdump")
CXXFILT = os.environ.get("LLVM_CXXFILT", "/opt/rocm/llvm/bin/llvm-cxxfilt")

_GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    29: "IQ1_M", 30: "BF16",
}


def family_of(kernel_name):
    short = re.sub(r"<.*", "", kernel_name).split("(")[0]
    short = short.split("void ")[-1].strip()
    m = re.search(r"<\s*\(ggml_type\)(\d+)", kernel_name)
    if m:
        n = int(m.group(1))
        short += "[" + _GGML_TYPES.get(n, "type%d" % n) + "]"
    return short


# per-lane byte width from the load mnemonic suffix
def load_bytes(mn):
    # order matters: check d16/u8/i8/u16/i16 before generic bNN
    if re.search(r"_d16", mn):
        return 2
    if re.search(r"_(u8|i8|sbyte|ubyte)", mn):
        return 1
    if re.search(r"_(u16|i16|short|ushort|sshort)", mn):
        return 2
    m = re.search(r"_b(\d+)$", mn) or re.search(r"_b(\d+)_", mn)
    if m:
        return int(m.group(1)) // 8
    # legacy dword forms
    if re.search(r"_dwordx4", mn):
        return 16
    if re.search(r"_dwordx3", mn):
        return 12
    if re.search(r"_dwordx2", mn):
        return 8
    if re.search(r"_dword", mn):
        return 4
    return 4  # default


def classify(mn):
    """Return (category, bytes_per_lane). category in vector/scalar/lds/other."""
    if mn.startswith("ds_"):
        return "lds", load_bytes(mn)
    if mn.startswith("s_load") or mn.startswith("s_buffer_load"):
        return "scalar", load_bytes(mn)
    if re.match(r"(global|buffer|flat)_load", mn):
        return "vector", load_bytes(mn)
    return None, 0


def disasm_symbol(obj, mangled):
    out = subprocess.run(
        [OBJDUMP, "-d", "--disassemble-symbols=" + mangled, obj],
        capture_output=True, text=True).stdout
    vec = collections.Counter()   # bytes_per_lane -> count
    scalar = collections.Counter()
    lds = collections.Counter()
    for line in out.splitlines():
        # instruction lines look like: "<addr>: <hex> <mnemonic> ..."
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        insn = parts[-1].strip()
        mn = insn.split()[0] if insn else ""
        cat, b = classify(mn)
        if cat == "vector":
            vec[b] += 1
        elif cat == "scalar":
            scalar[b] += 1
        elif cat == "lds":
            lds[b] += 1
    return vec, scalar, lds


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    objs = sorted(glob.glob(os.path.join(
        build_dir, "libggml-hip.so*.hipv4-amdgcn-amd-amdhsa--gfx1151")))
    fams = {}   # family -> record (first representative wins)
    for obj in objs:
        nm = subprocess.run([NM, "--defined-only", obj],
                            capture_output=True, text=True).stdout
        syms = []
        for line in nm.splitlines():
            p = line.split()
            if len(p) >= 3 and p[1] in ("t", "T"):
                syms.append(p[2])
        if not syms:
            continue
        # demangle in one batch
        dem = subprocess.run([CXXFILT], input="\n".join(syms),
                             capture_output=True, text=True).stdout.splitlines()
        for mangled, demangled in zip(syms, dem):
            fam = family_of(demangled)
            if fam in fams:
                continue
            # only kernels (skip helpers with no clear kernel shape): keep all,
            # but require at least the family look like a kernel name (alnum_)
            if not re.match(r"^[A-Za-z_][\w]*(\[[^\]]+\])?$", fam):
                continue
            vec, scalar, lds = disasm_symbol(obj, mangled)
            if not (vec or scalar or lds):
                continue
            total_vec = sum(vec.values())
            # dominant per-lane width by instruction count
            dom = max(vec.items(), key=lambda kv: kv[1])[0] if vec else 0
            fams[fam] = {
                "symbol": demangled[:200],
                "vector_loads": {str(k): v for k, v in sorted(vec.items())},
                "scalar_loads": {str(k): v for k, v in sorted(scalar.items())},
                "lds_loads": {str(k): v for k, v in sorted(lds.items())},
                "vector_load_count": total_vec,
                "dominant_lane_bytes": dom,
            }
    json.dump(fams, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()

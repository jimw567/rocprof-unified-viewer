#!/usr/bin/env bash
# collect.sh -- produce every input rocprof-unified-viewer needs, in one shot,
# on a gfx1151 board. Runs llama-bench under rocprofv3 three times:
#
#   1. --sys-trace   -> *_kernel_trace.csv + *_hip_api_trace.csv  (shared clock,
#                       so the CPU and GPU lanes overlay)
#   2. --pmc <stall> -> *_counter_collection.csv  (per-family stall coloring)
#   3. --pmc FETCH_SIZE -> *_counter_collection.csv  (achieved DRAM bytes -> BW%)
#
# then runs disasm_loadwidth.py against the build's device code objects to emit
# loadwidth.json. rocprofv3 must run where the GPU is (run this ON the board).
#
# PMC serializes/replays kernels once per pass, so timing there is distorted --
# that is why PMC comes from separate runs and is joined by kernel FAMILY, and
# why the sys-trace run (which gives real timing) is kept clean and separate.
#
# ROCm gotchas baked in (see README):
#   * Drive rocprofv3 from a pinned local ROCm, not a shared /opt/rocm that gets
#     repointed under you. Override with --rocm or $ROCM_DIR.
#   * --pmc counter config needs the SYSTEM ROCm runtime first on LD_LIBRARY_PATH
#     (llvm/lib subdir included for libamd_comgr's libLLVM), NOT the build dir --
#     the release bundles its own runtime and shadowing it crashes counter config.
#   * --sys-trace needs the build dir on LD_LIBRARY_PATH so llama-bench finds its
#     bundled ggml/hip libs; that is fine for tracing.
#   * FETCH_SIZE byte counters need a recent llama.cpp build (b20260715+); older
#     builds SIGSEGV during FETCH_SIZE collection.
set -euo pipefail

ROCM_DIR="${ROCM_DIR:-}"
BUILD_DIR=""
MODEL=""
OUT_DIR=""
NTOK=8
PMC_NTOK=2
KERNEL_REGEX=""
# Stall-classification counters + raw cycle counters for two derived ratios the
# viewer computes: EA busy% = GRBM_EA_BUSY/GRBM_GUI_ACTIVE (DRAM-interface busy,
# the true BW bottleneck meter) and ALU busy% = SQ_INST_CYCLES_VALU/SQ_BUSY_CYCLES
# (VALU-active fraction; can exceed 100%, VALU counts across 4 SIMDs/CU).
STALL_COUNTERS="MemUnitBusy L2CacheHit WriteUnitStalled OccupancyPercent Wavefronts LDSBankConflict GRBM_EA_BUSY GRBM_GUI_ACTIVE SQ_INST_CYCLES_VALU SQ_BUSY_CYCLES"
EXTRA=()

usage() {
  cat <<EOF
Usage: collect.sh --build-dir DIR --model M.gguf --out-dir DIR [opts] [-- llama-bench flags]

  --build-dir DIR   Directory containing llama-bench + libggml-hip.so* (required).
  --model PATH      GGUF model file (required).
  --out-dir DIR     Where inputs are written (required). Layout:
                      <out>/trace/<host>/*_kernel_trace.csv, *_hip_api_trace.csv
                      <out>/stall/<host>/*_counter_collection.csv
                      <out>/fetch/<host>/*_counter_collection.csv
                      <out>/loadwidth.json
  --rocm DIR        ROCm install to drive rocprofv3 (default: \$ROCM_DIR${ROCM_DIR:+ =
                    $ROCM_DIR}). REQUIRED (via flag or \$ROCM_DIR).
  -n N              Decode tokens for the sys-trace run (default: $NTOK).
  --pmc-n N         Decode tokens for the PMC runs (default: $PMC_NTOK; keep small,
                    PMC replays every kernel once per counter-set pass).
  --kernel REGEX    Restrict PMC collection to kernels matching REGEX
                    (rocprofv3 --kernel-include-regex). Default: all kernels.
  --counters "..."  Override the stall counter set.
                    Default: $STALL_COUNTERS
  Anything after -- is passed straight to llama-bench (e.g. -fa 1 -r 1).

After it finishes, feed the paths to the viewer:
  rocprof-unified-viewer \\
    --kernel-csv <out>/trace/<host>/*_kernel_trace.csv \\
    --hip-csv    <out>/trace/<host>/*_hip_api_trace.csv \\
    --pmc-csv    <out>/stall/<host>/*_counter_collection.csv \\
    --fetch-csv  <out>/fetch/<host>/*_counter_collection.csv \\
    --loadwidth-json <out>/loadwidth.json \\
    --out overlay.html --tokens 2
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --model)     MODEL="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    --rocm)      ROCM_DIR="$2"; shift 2 ;;
    -n)          NTOK="$2"; shift 2 ;;
    --pmc-n)     PMC_NTOK="$2"; shift 2 ;;
    --kernel)    KERNEL_REGEX="$2"; shift 2 ;;
    --counters)  STALL_COUNTERS="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    --)          shift; EXTRA=("$@"); break ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

[ -n "$BUILD_DIR" ] || { echo "ERROR: --build-dir required" >&2; usage >&2; exit 1; }
[ -n "$MODEL" ]     || { echo "ERROR: --model required" >&2; exit 1; }
[ -n "$OUT_DIR" ]   || { echo "ERROR: --out-dir required" >&2; exit 1; }
[ -n "$ROCM_DIR" ]  || { echo "ERROR: ROCm dir required: set --rocm DIR or \$ROCM_DIR" >&2; exit 1; }
[ -x "$BUILD_DIR/llama-bench" ] || { echo "ERROR: no llama-bench in $BUILD_DIR" >&2; exit 1; }
[ -f "$MODEL" ]     || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }

ROCPROFV3="$ROCM_DIR/bin/rocprofv3"
[ -x "$ROCPROFV3" ] || { echo "ERROR: rocprofv3 not found at $ROCPROFV3 (set --rocm)" >&2; exit 1; }

ROCM_LIBS="$ROCM_DIR/lib:$ROCM_DIR/lib/llvm/lib"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR/trace" "$OUT_DIR/stall" "$OUT_DIR/fetch"

echo "ROCm:      $ROCM_DIR"
echo "Build:     $BUILD_DIR"
echo "Model:     $MODEL"
echo "Out:       $OUT_DIR"
echo "Extra:     ${EXTRA[*]:-(none)}"
echo ""

# rocprofv3 can SIGSEGV in its rocpd/OMPT postprocess AFTER the csv is flushed;
# tolerate a nonzero exit and judge success by the CSV that we expect.
run_ok() { # $1=glob under dir  $2=dir  $3..=command
  local glob="$1" dir="$2"; shift 2
  set +e; "$@"; local rc=$?; set -e
  if ! find "$dir" -name "$glob" -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: no $glob under $dir (rocprofv3 rc=$rc)" >&2; exit 1
  fi
  [ "$rc" -eq 0 ] || echo "NOTE: rocprofv3 exited rc=$rc (csv written; harmless postprocess crash)"
}

# --- 1. sys-trace (real timing; build dir first so llama-bench finds its libs) --
echo "=== [1/4] sys-trace (-n $NTOK) ==="
( cd "$BUILD_DIR"
  PATH="$ROCM_DIR/bin:$PATH" \
  LD_LIBRARY_PATH="$BUILD_DIR:$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
  run_ok '*_kernel_trace.csv' "$OUT_DIR/trace" \
    "$ROCPROFV3" --sys-trace --output-format pftrace csv -d "$OUT_DIR/trace" -- \
    ./llama-bench -m "$MODEL" -p 0 -n "$NTOK" "${EXTRA[@]}" )

# --- PMC helper: system ROCm runtime FIRST (no build dir) -----------------------
KREGEX=()
[ -n "$KERNEL_REGEX" ] && KREGEX=(--kernel-include-regex "$KERNEL_REGEX")
pmc_run() { # $1=dir  $2..=counters
  local dir="$1"; shift
  ( cd "$BUILD_DIR"
    PATH="$ROCM_DIR/bin:$PATH" \
    LD_LIBRARY_PATH="$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
    run_ok '*_counter_collection.csv' "$dir" \
      "$ROCPROFV3" --pmc "$@" --output-format csv "${KREGEX[@]}" -d "$dir" -- \
      ./llama-bench -m "$MODEL" -p 0 -n "$PMC_NTOK" "${EXTRA[@]}" )
}

echo ""
echo "=== [2/4] PMC stall counters (-n $PMC_NTOK) ==="
pmc_run "$OUT_DIR/stall" $STALL_COUNTERS

echo ""
echo "=== [3/4] PMC FETCH_SIZE -> achieved DRAM bytes (-n $PMC_NTOK) ==="
pmc_run "$OUT_DIR/fetch" FETCH_SIZE

# --- 4. disasm load widths ------------------------------------------------------
echo ""
echo "=== [4/4] disasm load widths -> loadwidth.json ==="
LLVM_NM="$ROCM_DIR/lib/llvm/bin/llvm-nm" \
LLVM_OBJDUMP="$ROCM_DIR/lib/llvm/bin/llvm-objdump" \
LLVM_CXXFILT="$ROCM_DIR/lib/llvm/bin/llvm-cxxfilt" \
  python3 "$SCRIPT_DIR/disasm_loadwidth.py" "$BUILD_DIR" > "$OUT_DIR/loadwidth.json"
echo "Wrote $OUT_DIR/loadwidth.json"

echo ""
echo "Done. Generate the overlay with:"
HOST="$(hostname -s)"
echo "  rocprof-unified-viewer \\"
echo "    --kernel-csv $OUT_DIR/trace/$HOST/*_kernel_trace.csv \\"
echo "    --hip-csv    $OUT_DIR/trace/$HOST/*_hip_api_trace.csv \\"
echo "    --pmc-csv    $OUT_DIR/stall/$HOST/*_counter_collection.csv \\"
echo "    --fetch-csv  $OUT_DIR/fetch/$HOST/*_counter_collection.csv \\"
echo "    --loadwidth-json $OUT_DIR/loadwidth.json \\"
echo "    --out overlay.html --tokens 2"

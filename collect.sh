#!/usr/bin/env bash
# collect.sh -- produce every input rocprof-unified-viewer needs, in one shot,
# on a gfx1151 board. First runs llama-bench CLEAN (no rocprofv3) for the honest
# untraced tok/s, then runs it under rocprofv3 three times:
#
#   0. clean run     -> clean_tps.txt  (untraced baseline tok/s; rocprofv3 adds
#                       overhead -- sys-trace instruments every dispatch, PMC
#                       replays kernels -- so the traced runs' tok/s is NOT the
#                       real throughput. This bare run is the number to quote.)
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
NTOK=64
PMC_NTOK=2
PROMPT=0
BOTH=0
CLEAN_REPS=3
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
                      <out>/clean_tps.txt  (untraced baseline tok/s)
                      <out>/trace/<host>/*_kernel_trace.csv, *_hip_api_trace.csv
                      <out>/stall/<host>/*_counter_collection.csv
                      <out>/fetch/<host>/*_counter_collection.csv
                      <out>/loadwidth.json
  --rocm DIR        ROCm install to drive rocprofv3 (default: \$ROCM_DIR${ROCM_DIR:+ =
                    $ROCM_DIR}). REQUIRED (via flag or \$ROCM_DIR).
  -n N              Decode tokens for the sys-trace run (default: $NTOK).
  --pmc-n N         Decode tokens for the PMC runs (default: $PMC_NTOK; keep small,
                    PMC replays every kernel once per counter-set pass).
  --clean-reps N    Reps for the untraced clean run (default: $CLEAN_REPS). Emitted as
                    llama-bench JSON so the viewer takes the median of samples_ts
                    (matches the regression harness; -r 1 gives a cold first sample).
  --prompt N        Prefill mode: process an N-token prompt with 0 decode
                    (-p N -n 0) instead of the default decode workload
                    (-p 0 -n <-n>). N=0 (default) = decode; N>0 = prefill.
                    In prefill mode -n / --pmc-n are ignored (one forward pass),
                    and clean_tps.txt holds prompt-processing tp (pp) not decode
                    tg. Feed the resulting trace to the viewer with --mode prefill.
  --both            Collect BOTH regimes in one invocation: a decode run
                    (-p 0 -n <-n>) into <out>/decode/ and a prefill run
                    (-p <--prompt, default 128> -n 0) into <out>/prefill/. Prints
                    a single viewer command that embeds both, so the overlay shows
                    an in-page prefill/decode dropdown. Each regime is its own clean
                    sys-trace (measured in isolation).
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
    --clean-tps-file <out>/clean_tps.txt \\
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
    --prompt)    PROMPT="$2"; shift 2 ;;
    --both)      BOTH=1; shift ;;
    --clean-reps) CLEAN_REPS="$2"; shift 2 ;;
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

case "$PROMPT" in ''|*[!0-9]*) echo "ERROR: --prompt must be a non-negative integer" >&2; exit 1 ;; esac
# --both prefills default to 128 prompt tokens if --prompt was left at 0.
[ "$BOTH" -eq 1 ] && [ "$PROMPT" -eq 0 ] && PROMPT=128

HOST="$(hostname -s)"

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

KREGEX=()
[ -n "$KERNEL_REGEX" ] && KREGEX=(--kernel-include-regex "$KERNEL_REGEX")

# collect_regime MODE OUTDIR : run clean + sys-trace + both PMC passes for ONE
# regime into OUTDIR. Decode uses -p 0 -n NTOK (periodic per-token stream); prefill
# uses -p PROMPT -n 0 (one MMQ forward pass), where -n / --pmc-n do not apply.
collect_regime() {
  local mode="$1" out="$2"
  local trace_pn pmc_pn
  if [ "$mode" = "prefill" ]; then
    trace_pn=(-p "$PROMPT" -n 0); pmc_pn=(-p "$PROMPT" -n 0)
  else
    trace_pn=(-p 0 -n "$NTOK");  pmc_pn=(-p 0 -n "$PMC_NTOK")
  fi
  mkdir -p "$out/trace" "$out/stall" "$out/fetch"
  echo "=== [$mode] out=$out (trace: ${trace_pn[*]}; pmc: ${pmc_pn[*]}) ==="

  # 0. clean run: no rocprofv3, so the tok/s is the honest untraced throughput.
  # Emit JSON with CLEAN_REPS reps so the viewer takes the MEDIAN of samples_ts,
  # exactly like the llamacpp regression harness. A single -r 1 run reports a
  # cold-cache first sample (pp can read ~25% low), skewing the derived TTFT.
  echo "--- clean run (no rocprofv3, -r $CLEAN_REPS -o json) -> untraced tok/s (${trace_pn[*]}) ---"
  ( cd "$BUILD_DIR"
    LD_LIBRARY_PATH="$BUILD_DIR:$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
      ./llama-bench -o json -r "$CLEAN_REPS" -m "$MODEL" "${trace_pn[@]}" "${EXTRA[@]}" ) \
      | tee "$out/clean_tps.txt" | grep -E '"(n_prompt|n_gen|avg_ts)"' || true

  # 1. sys-trace (real timing; build dir first so llama-bench finds its libs).
  echo "--- sys-trace (${trace_pn[*]}) ---"
  ( cd "$BUILD_DIR"
    PATH="$ROCM_DIR/bin:$PATH" \
    LD_LIBRARY_PATH="$BUILD_DIR:$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
    run_ok '*_kernel_trace.csv' "$out/trace" \
      "$ROCPROFV3" --sys-trace --output-format pftrace csv -d "$out/trace" -- \
      ./llama-bench -m "$MODEL" "${trace_pn[@]}" "${EXTRA[@]}" )

  # 2/3. PMC (system ROCm runtime FIRST on LD_LIBRARY_PATH, no build dir).
  local pmc_run
  pmc_run() { local dir="$1"; shift
    ( cd "$BUILD_DIR"
      PATH="$ROCM_DIR/bin:$PATH" \
      LD_LIBRARY_PATH="$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
      run_ok '*_counter_collection.csv' "$dir" \
        "$ROCPROFV3" --pmc "$@" --output-format csv "${KREGEX[@]}" -d "$dir" -- \
        ./llama-bench -m "$MODEL" "${pmc_pn[@]}" "${EXTRA[@]}" )
  }
  echo "--- PMC stall counters (${pmc_pn[*]}) ---"
  pmc_run "$out/stall" $STALL_COUNTERS
  echo "--- PMC FETCH_SIZE (${pmc_pn[*]}) ---"
  pmc_run "$out/fetch" FETCH_SIZE
  echo ""
}

echo "ROCm:      $ROCM_DIR"
echo "Build:     $BUILD_DIR"
echo "Model:     $MODEL"
echo "Out:       $OUT_DIR"
echo "Extra:     ${EXTRA[*]:-(none)}"
echo ""

# viewer_flags MODE OUTDIR : echo the per-regime input flags (with a prefix so the
# same helper builds both the primary and the --alt-* set).
viewer_flags() { # $1=prefix ("" or "alt-")  $2=outdir
  local p="$1" d="$2"
  echo "    --${p}kernel-csv $d/trace/$HOST/*_kernel_trace.csv \\"
  echo "    --${p}hip-csv    $d/trace/$HOST/*_hip_api_trace.csv \\"
  echo "    --${p}pmc-csv    $d/stall/$HOST/*_counter_collection.csv \\"
  echo "    --${p}fetch-csv  $d/fetch/$HOST/*_counter_collection.csv \\"
  echo "    --${p}clean-tps-file $d/clean_tps.txt \\"
}

if [ "$BOTH" -eq 1 ]; then
  collect_regime decode  "$OUT_DIR/decode"
  collect_regime prefill "$OUT_DIR/prefill"
  # loadwidth is regime-independent (device disasm of the same build) -> once.
  echo "=== disasm load widths -> loadwidth.json ==="
  LLVM_NM="$ROCM_DIR/lib/llvm/bin/llvm-nm" \
  LLVM_OBJDUMP="$ROCM_DIR/lib/llvm/bin/llvm-objdump" \
  LLVM_CXXFILT="$ROCM_DIR/lib/llvm/bin/llvm-cxxfilt" \
    python3 "$SCRIPT_DIR/disasm_loadwidth.py" "$BUILD_DIR" > "$OUT_DIR/loadwidth.json"
  echo ""
  echo "Done. Generate the DUAL-REGIME overlay (in-page prefill/decode dropdown) with:"
  echo "  rocprof-unified-viewer \\"
  echo "    --mode decode \\"
  viewer_flags ""    "$OUT_DIR/decode"
  echo "    --alt-mode prefill \\"
  viewer_flags "alt-" "$OUT_DIR/prefill"
  echo "    --loadwidth-json $OUT_DIR/loadwidth.json \\"
  echo "    --out overlay.html --tokens 2"
else
  if [ "$PROMPT" -gt 0 ]; then MODE="prefill"; else MODE="decode"; fi
  collect_regime "$MODE" "$OUT_DIR"
  echo "=== disasm load widths -> loadwidth.json ==="
  LLVM_NM="$ROCM_DIR/lib/llvm/bin/llvm-nm" \
  LLVM_OBJDUMP="$ROCM_DIR/lib/llvm/bin/llvm-objdump" \
  LLVM_CXXFILT="$ROCM_DIR/lib/llvm/bin/llvm-cxxfilt" \
    python3 "$SCRIPT_DIR/disasm_loadwidth.py" "$BUILD_DIR" > "$OUT_DIR/loadwidth.json"
  echo ""
  echo "Done. Generate the overlay with:"
  echo "  rocprof-unified-viewer \\"
  viewer_flags "" "$OUT_DIR"
  echo "    --loadwidth-json $OUT_DIR/loadwidth.json \\"
  if [ "$MODE" = "prefill" ]; then
    echo "    --mode prefill --out overlay.html"
  else
    echo "    --out overlay.html --tokens 2"
  fi
fi

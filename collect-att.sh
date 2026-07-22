#!/usr/bin/env bash
# collect-att.sh -- capture + decode an ATT (Advanced Thread Trace) of ONE kernel
# on a gfx1151 board, producing the stats_ui_output_*_dispatch_*.csv files that
# rocprof-unified-viewer's --att-dir folds into the selected-kernel detail panel.
#
# ATT is a microscope, not a survey: rocprofv3 --att instruments ~1 CU/SIMD for a
# few dispatches and records a per-instruction cycle timeline (stall/latency/idle).
# So you point it at a SINGLE kernel symbol -- exactly the "select a kernel in the
# overlay, trace just that one" round-trip. The overlay prints the matching
# `--kernel REGEX` for whatever kernel you click.
#
# ROCm gotchas baked in (see README + collect.sh):
#   * Drive rocprofv3 from a pinned LOCAL ROCm 7.15 (the ATT decoder lib
#     librocprof-trace-decoder.so lives there); /opt/rocm gets repointed under you
#     and may lack the V2 decoder. Override with --rocm or $ROCM_DIR.
#   * --att-library-path must point at the ROCm lib dir that ships the decoder.
#   * The build dir goes on LD_LIBRARY_PATH so llama-bench finds its bundled
#     ggml/hip libs (tracing, unlike --pmc counter config, tolerates this).
#   * Keep the trace SMALL: default 256MB buffer, one target CU, one kernel. A
#     1GB buffer crawls/hangs. Some early dispatches still cut off ("Wave
#     incomplete: trace cutoff") and decode to empty dirs -- that is expected;
#     the viewer skips empty dispatches and uses the populated ones.
set -euo pipefail

ROCM_DIR="${ROCM_DIR:-}"
BUILD_DIR=""
MODEL=""
OUT_DIR=""
KERNEL_REGEX=""
RUNNER=""
RUNNER_ENV=()
NTOK=1
TARGET_CU=0
CONSEC=1
BUFFER_MB=256
SE_MASK="0x1"
SIMD_SELECT=""
EXTRA=()

usage() {
  cat <<EOF
Usage: collect-att.sh --kernel REGEX --build-dir DIR --model M.gguf --out-dir DIR [opts] [-- llama-bench flags]

  --kernel REGEX    rocprofv3 --kernel-include-regex: the kernel SYMBOL to trace
                    (e.g. mul_mat_vec_q_wvsplitk). REQUIRED. ATT filters by symbol,
                    so this captures every quant/shape variant of that kernel.
  --build-dir DIR   Directory containing the workload binary + libggml-hip.so*
                    (required). For the default runner this must hold llama-bench;
                    for --runner it must hold that binary (e.g. test-backend-ops).
  --model PATH      GGUF model file (required for the default llama-bench runner;
                    ignored when --runner is given).
  --out-dir DIR     Where the decoded ATT output is written (required). Feed this
                    same directory to the viewer's --att-dir.
  --runner "CMD"    Replace the default llama-bench decode workload with an
                    arbitrary command (run with cwd = --build-dir, so a bundled
                    binary is "./name"). Lets ATT target a single-kernel evaluator
                    instead of a full model graph -- far fewer cutoff dispatches.
                    Example (llama.cpp single-op perf harness):
                      --runner "./test-backend-ops perf -o MUL_MAT -p type_a=q4_K"
                    When set, --model is not required and -n is ignored; put any
                    workload flags inside the quoted string (not after --).
  --runner-env V=X  Set an environment variable for the --runner process (repeatable).
                    Used to inject a shape-exact test case, e.g. with a patched
                    test-backend-ops: --runner-env GGML_ATT_MULMAT=12,2560,8192
                    (ggml_type,K,N) so ATT captures the model's real matvec dims.
  --rocm DIR        ROCm install to drive rocprofv3 + supply the ATT decoder
                    (default: \$ROCM_DIR${ROCM_DIR:+ = $ROCM_DIR}). REQUIRED
                    (via flag or \$ROCM_DIR); no personal path is baked in.
  -n N              Decode tokens (default: $NTOK; keep tiny, ATT is heavy).
  --target-cu N     ATT target CU (default: $TARGET_CU).
  --consecutive N   --att-consecutive-kernels (default: $CONSEC).
  --buffer-mb N     --att-buffer-size in MB (default: $BUFFER_MB; raising this
                    reduces cutoffs but slows the run sharply -- 1024 can hang).
  --se-mask HEX     --att-shader-engine-mask (default: $SE_MASK; gfx1151 has one SE).
  --simd-select N   --att-simd-select: on gfx10+ this is the SIMD *ID* to detail
                    (0..3), NOT a bitmask -- ATT thread-traces ONE SIMD per run and
                    lumps the rest into other_simd_*.json. rocprofv3 default 0xF
                    resolves to SIMD 3. To cover every SIMD, run once per id (0,1,2,3)
                    and merge the outputs. Unset = leave rocprofv3 default.
  Anything after -- is passed straight to the default llama-bench runner (e.g.
  -fa 1). Ignored when --runner is set (put flags in the --runner string).

After it finishes, refold into the overlay:
  rocprof-unified-viewer <your existing flags> --att-dir <out-dir> --out overlay.html
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --kernel)      KERNEL_REGEX="$2"; shift 2 ;;
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --runner)      RUNNER="$2"; shift 2 ;;
    --runner-env)  RUNNER_ENV+=("$2"); shift 2 ;;
    --rocm)        ROCM_DIR="$2"; shift 2 ;;
    -n)            NTOK="$2"; shift 2 ;;
    --target-cu)   TARGET_CU="$2"; shift 2 ;;
    --consecutive) CONSEC="$2"; shift 2 ;;
    --buffer-mb)   BUFFER_MB="$2"; shift 2 ;;
    --se-mask)     SE_MASK="$2"; shift 2 ;;
    --simd-select) SIMD_SELECT="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    --)            shift; EXTRA=("$@"); break ;;
    *)             EXTRA+=("$1"); shift ;;
  esac
done

[ -n "$KERNEL_REGEX" ] || { echo "ERROR: --kernel REGEX required" >&2; usage >&2; exit 1; }
[ -n "$BUILD_DIR" ]    || { echo "ERROR: --build-dir required" >&2; exit 1; }
[ -n "$OUT_DIR" ]      || { echo "ERROR: --out-dir required" >&2; exit 1; }
[ -n "$ROCM_DIR" ]     || { echo "ERROR: ROCm dir required: set --rocm DIR or \$ROCM_DIR" >&2; exit 1; }
if [ -z "$RUNNER" ]; then
  [ -n "$MODEL" ]      || { echo "ERROR: --model required (or use --runner)" >&2; exit 1; }
  [ -x "$BUILD_DIR/llama-bench" ] || { echo "ERROR: no llama-bench in $BUILD_DIR" >&2; exit 1; }
  [ -f "$MODEL" ]      || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
fi

ROCPROFV3="$ROCM_DIR/bin/rocprofv3"
[ -x "$ROCPROFV3" ] || { echo "ERROR: rocprofv3 not found at $ROCPROFV3 (set --rocm)" >&2; exit 1; }
[ -d "$ROCM_DIR/lib" ] || { echo "ERROR: no lib dir at $ROCM_DIR/lib (ATT decoder)" >&2; exit 1; }

ROCM_LIBS="$ROCM_DIR/lib:$ROCM_DIR/lib/llvm/lib"
mkdir -p "$OUT_DIR"

# The workload run under ATT. Default = llama-bench decode; --runner overrides it
# with an arbitrary command (word-split; run with cwd = BUILD_DIR).
if [ -n "$RUNNER" ]; then
  # shellcheck disable=SC2206  # intentional word-split of the runner string
  WORKLOAD=($RUNNER)
else
  WORKLOAD=(./llama-bench -m "$MODEL" -p 0 -n "$NTOK" -r 1 "${EXTRA[@]}")
fi

echo "ROCm:      $ROCM_DIR"
echo "Build:     $BUILD_DIR"
echo "Model:     ${MODEL:-(n/a, --runner)}"
echo "Out:       $OUT_DIR"
echo "Kernel:    $KERNEL_REGEX"
echo "ATT:       target-cu=$TARGET_CU consecutive=$CONSEC buffer=${BUFFER_MB}MB se-mask=$SE_MASK simd-select=${SIMD_SELECT:-default}"
echo "Workload:  ${WORKLOAD[*]}"
[ ${#RUNNER_ENV[@]} -gt 0 ] && echo "RunnerEnv: ${RUNNER_ENV[*]}"
echo ""

# gfx10+: --att-simd-select is a SIMD *ID* (0..3); ATT details ONE SIMD per run.
# Only pass it when set so rocprofv3 keeps its own default (0xF -> SIMD 3) otherwise.
SIMD_ARGS=()
[ -n "$SIMD_SELECT" ] && SIMD_ARGS=(--att-simd-select "$SIMD_SELECT")

# rocprofv3 may exit nonzero in postprocess after decoding; judge success by the
# decoded stats CSVs actually present under the out dir.
set +e
( cd "$BUILD_DIR"
  # export any --runner-env vars so they reach the workload (rocprofv3's child).
  for kv in "${RUNNER_ENV[@]}"; do export "${kv?}"; done
  PATH="$ROCM_DIR/bin:$PATH" \
  LD_LIBRARY_PATH="$BUILD_DIR:$ROCM_LIBS:${LD_LIBRARY_PATH:-}" \
  "$ROCPROFV3" --att \
    --att-library-path "$ROCM_DIR/lib" \
    --att-target-cu "$TARGET_CU" \
    --att-consecutive-kernels "$CONSEC" \
    --att-buffer-size $((BUFFER_MB * 1024 * 1024)) \
    --att-shader-engine-mask "$SE_MASK" \
    "${SIMD_ARGS[@]}" \
    --kernel-include-regex "$KERNEL_REGEX" \
    -d "$OUT_DIR" -- \
    "${WORKLOAD[@]}" )
rc=$?
set -e

if ! find "$OUT_DIR" -name 'stats_ui_output_*_dispatch_*.csv' -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: no decoded stats_ui_output_*_dispatch_*.csv under $OUT_DIR (rocprofv3 rc=$rc)." >&2
  echo "       Check the kernel regex matched, and that the decoder lib is under $ROCM_DIR/lib." >&2
  exit 1
fi
[ "$rc" -eq 0 ] || echo "NOTE: rocprofv3 exited rc=$rc (stats decoded; harmless postprocess)"

# Count populated (non-cutoff) dispatches so the user knows there is usable data.
POP=0
while IFS= read -r f; do
  # a populated dispatch's stats CSV has instruction rows with nonzero hitcount
  if awk -F',' 'NR>1 && $4+0>0 {found=1; exit} END{exit !found}' "$f"; then
    POP=$((POP + 1))
  fi
done < <(find "$OUT_DIR" -name 'stats_ui_output_*_dispatch_*.csv')

echo ""
echo "Done. $POP populated dispatch(es) decoded under $OUT_DIR."
echo "Refold into the overlay with:"
echo "  rocprof-unified-viewer <your existing flags> --att-dir $OUT_DIR --out overlay.html"

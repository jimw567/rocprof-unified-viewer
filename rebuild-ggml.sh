#!/usr/bin/env bash
# rebuild-ggml.sh -- rebuild libggml-hip.so (and test-backend-ops) from a llama.cpp
# source tree WITH device DWARF line tables, so an ATT trace of the resulting binary
# can map ISA back to source in the viewer's Trace View.
#
# WHY THIS EXISTS: "Run trace" only RE-RUNS rocprofv3 against an existing
# libggml-hip.so; it cannot add debug info the binary lacks. ATT reads line info
# from the .debug_line section of the DEVICE code object embedded in that .so, and
# that section is only produced when the HIP compile is given -gline-tables-only
# (or -g). Stock/CI release builds omit it -> no source lines, no matter how many
# times you re-trace. This script produces a binary that HAS it.
#
# The build is driven from a user-pinned ROCm (default $ROCM_DIR), same toolchain
# that supplies the ATT decoder, so build + trace stay consistent. It does NOT use
# /opt/rocm (which is shared and gets repointed).
#
# ITERATION LOOP:
#   edit ggml/src/ggml-cuda/mmvq.cu   (the REAL kernel -- one source of truth)
#   ./rebuild-ggml.sh --src DIR --build-dir DIR --rocm DIR   # incremental (~30-60s)
#   # then trace the rebuilt binary against a single-op runner (few cutoffs):
#   collect-att.sh --build-dir <build>/bin \
#     --runner "./test-backend-ops perf -o MUL_MAT -p type_a=q4_K" \
#     --kernel mul_mat_vec_q --out-dir <out> --rocm DIR
# (the exact trace command is printed at the end.)
set -euo pipefail

ROCM_DIR="${ROCM_DIR:-}"
SRC_DIR=""
BUILD_DIR=""
GPU_TARGET="gfx1151"
JOBS=""
TARGETS="ggml-hip test-backend-ops"
RECONFIG=0

usage() {
  cat <<EOF
Usage: rebuild-ggml.sh --src DIR --build-dir DIR [--rocm DIR] [opts]

  --src DIR         llama.cpp source tree (contains CMakeLists.txt). REQUIRED.
  --build-dir DIR   Build directory to create/reuse. REQUIRED. Configured once
                    against the pinned ROCm with -gline-tables-only; later runs
                    are incremental (only changed .cu recompiles, .so relinks).
  --rocm DIR        ROCm install driving the HIP build (default \$ROCM_DIR${ROCM_DIR:+ = $ROCM_DIR}).
                    REQUIRED via flag or \$ROCM_DIR. NOT /opt/rocm.
  --arch ARCH       GPU offload target (default: $GPU_TARGET).
  --targets "T..."  Ninja targets to build (default: "$TARGETS"). ggml-hip yields
                    libggml-hip.so; test-backend-ops is the single-op trace runner.
  --jobs N          Parallel build jobs (default: nproc).
  --reconfigure     Force a fresh cmake configure even if the build dir exists
                    (use after changing ROCm/arch/flags). Deletes CMakeCache.txt.
  -h|--help         This help.

After it builds, it verifies .debug_line is present in the new libggml-hip.so and
prints the collect-att.sh command to trace it.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --src)         SRC_DIR="$2"; shift 2 ;;
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    --rocm)        ROCM_DIR="$2"; shift 2 ;;
    --arch)        GPU_TARGET="$2"; shift 2 ;;
    --targets)     TARGETS="$2"; shift 2 ;;
    --jobs)        JOBS="$2"; shift 2 ;;
    --reconfigure) RECONFIG=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "ERROR: unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -n "$SRC_DIR" ]   || { echo "ERROR: --src required" >&2; exit 1; }
[ -n "$BUILD_DIR" ] || { echo "ERROR: --build-dir required" >&2; exit 1; }
[ -n "$ROCM_DIR" ]  || { echo "ERROR: ROCm dir required: set --rocm DIR or \$ROCM_DIR" >&2; exit 1; }
[ -f "$SRC_DIR/CMakeLists.txt" ] || { echo "ERROR: no CMakeLists.txt in $SRC_DIR" >&2; exit 1; }
[ -x "$ROCM_DIR/bin/hipcc" ] || { echo "ERROR: no hipcc under $ROCM_DIR/bin (set --rocm)" >&2; exit 1; }

CLANG="$ROCM_DIR/llvm/bin/clang"
CLANGXX="$ROCM_DIR/llvm/bin/clang++"
[ -x "$CLANGXX" ] || CLANGXX="$ROCM_DIR/lib/llvm/bin/clang++"
[ -x "$CLANG" ]   || CLANG="$ROCM_DIR/lib/llvm/bin/clang"
: "${JOBS:=$(nproc 2>/dev/null || echo 8)}"

# Device DWARF line tables. -gline-tables-only keeps size/perf sane vs full -g while
# still giving the ATT decoder .debug_line for ISA->source mapping. Appended to the
# HIP flags so it augments the Release -O3 build rather than replacing it.
DEBUG_HIP_FLAGS="-gline-tables-only"

export ROCM_PATH="$ROCM_DIR"
export HIP_PATH="$ROCM_DIR"
export PATH="$ROCM_DIR/bin:$ROCM_DIR/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_DIR/lib:$ROCM_DIR/lib64:$ROCM_DIR/llvm/lib:${LD_LIBRARY_PATH:-}"

echo "ROCm:     $ROCM_DIR"
echo "Source:   $SRC_DIR"
echo "Build:    $BUILD_DIR"
echo "Arch:     $GPU_TARGET"
echo "Targets:  $TARGETS"
echo "HIP dbg:  $DEBUG_HIP_FLAGS"
echo ""

NEED_CONFIG=0
[ -f "$BUILD_DIR/build.ninja" ] || [ -f "$BUILD_DIR/Makefile" ] || NEED_CONFIG=1
[ "$RECONFIG" -eq 1 ] && NEED_CONFIG=1

if [ "$NEED_CONFIG" -eq 1 ]; then
  echo "== configuring (one-time; adds $DEBUG_HIP_FLAGS) =="
  [ "$RECONFIG" -eq 1 ] && rm -f "$BUILD_DIR/CMakeCache.txt"
  mkdir -p "$BUILD_DIR"
  # Flags mirror the known-good gfx11 build recipe, plus -gline-tables-only on the
  # HIP compile. GGML_NATIVE=OFF + CROSSCOMPILING so it builds off the target host.
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_C_COMPILER="$CLANG" \
    -DCMAKE_CXX_COMPILER="$CLANGXX" \
    -DCMAKE_CXX_FLAGS="-I$ROCM_DIR/include" \
    -DCMAKE_HIP_FLAGS="$DEBUG_HIP_FLAGS" \
    -DCMAKE_CROSSCOMPILING=ON -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="$GPU_TARGET" -DBUILD_SHARED_LIBS=ON \
    -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF \
    -DGGML_HIP_ROCWMMA_FATTN=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_SYSTEM_NAME=Linux
else
  echo "== incremental build (build dir already configured) =="
fi

echo ""
echo "== building: $TARGETS (jobs=$JOBS) =="
# shellcheck disable=SC2086  # intentional word-split of the target list
cmake --build "$BUILD_DIR" --target $TARGETS -j "$JOBS"

# Locate the freshly built libggml-hip.so (build layout varies: bin/ or lib/).
LIB="$(find "$BUILD_DIR" -name 'libggml-hip.so*' -type f 2>/dev/null | head -1)"
[ -n "$LIB" ] || { echo "ERROR: no libggml-hip.so produced under $BUILD_DIR" >&2; exit 1; }
LIBDIR="$(dirname "$LIB")"

echo ""
echo "== verifying device DWARF (.debug_line) in the new lib =="
# The HIP fat binary embeds the device ELF; llvm-dwarfdump reports its line program.
DWARFDUMP="$ROCM_DIR/llvm/bin/llvm-dwarfdump"
[ -x "$DWARFDUMP" ] || DWARFDUMP="$ROCM_DIR/lib/llvm/bin/llvm-dwarfdump"
HAS_DWARF=0
if [ -x "$DWARFDUMP" ]; then
  # count .cu source references in the line program; >0 means device line tables are
  # present (the decoder reads these). grep -F on ".cu" is robust to path formatting.
  NLINES="$("$DWARFDUMP" --debug-line "$LIB" 2>/dev/null | grep -cF '.cu' || true)"
  [ "${NLINES:-0}" -gt 0 ] && HAS_DWARF=1
fi
if [ "$HAS_DWARF" -eq 1 ]; then
  echo "OK: device .debug_line present ($NLINES .cu line entries) -- ISA->source"
  echo "    should populate in the Trace View once this binary is traced."
else
  echo "WARN: could not confirm .debug_line via llvm-dwarfdump (tool missing or no"
  echo "      entries). Verify by tracing and checking code.json sqtt_funcmap is non-empty."
fi

echo ""
echo "Built: $LIB"
echo "Now trace the rebuilt binary against a single-op runner (fewer cutoffs):"
echo "  collect-att.sh --build-dir $LIBDIR \\"
echo "    --runner \"./test-backend-ops perf -o MUL_MAT -p type_a=q4_K\" \\"
echo "    --kernel mul_mat_vec_q --out-dir <out-dir> --rocm $ROCM_DIR"
echo "(test-backend-ops binary should be in the same build; adjust its path if needed.)"

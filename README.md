# rocprof-unified-viewer

Fuse the profiling layers rocprofv3 gives you -- CPU (HIP-API) overhead, GPU
kernel timing, per-kernel hardware stall counters, and achieved DRAM bandwidth --
into ONE self-contained HTML timeline. No server, no dependencies, no network:
open the file in any browser.

v1 is specialized for **llama.cpp / ggml decode on gfx1151** (Strix Halo), but it
consumes generic rocprofv3 CSVs and has room to grow.

## Why this exists

No single existing tool overlays all of these layers at once:

- **Perfetto** shows the CPU and GPU tracks, but it can't tie a PMC counter back
  to the slice that produced it, chokes on large traces, and has no aggregate
  summary panel beside the timeline.
- **rocprof-compute** gives per-kernel counters but no timeline and (on the
  gfx1151 boards) resolves to a debug build that crashes on counter collection.

This tool renders a Canvas timeline with a **CPU (HIP-API) lane above** and a
**GPU (kernel) lane below** on a shared time axis, colors each GPU slice by its
**dominant stall reason**, and shows a **per-kernel-family summary** with achieved
bandwidth, stall breakdown, and load width -- all in one page, with hover detail
and a token stepper.

### Small window, not 128 tokens

Decode is **periodic**: every generated token replays the same kernel sequence.
128 tokens is ~99% redundant repetition -- exactly what makes Perfetto unusable.
The viewer defaults to a tiny **2-token** window (`--tokens 2`), landing in steady
state past warmup (`--skip-tokens 30`), so you see one clean period instead of a
wall of duplicates.

## Install

```bash
pip install -e .
```

This exposes two console scripts:

- `rocprof-unified-viewer` -- the HTML generator
- `rocprof-disasm-loadwidth` -- the load-width disassembly helper

Both are stdlib-only, so you can also just run them in place with no install:

```bash
python3 rocprof_unified_viewer.py --help
python3 disasm_loadwidth.py <build-dir>
```

## Producing the inputs

### One command (recommended)

Run [`collect.sh`](collect.sh) **on the board** (rocprofv3 needs the GPU). It runs
llama-bench under rocprofv3 three times (sys-trace, PMC stall counters, PMC
FETCH_SIZE) and then disassembles the device code for load widths:

```bash
./collect.sh \
    --build-dir /path/to/llamacpp-build \
    --model     /path/to/Model-Q4_K_M.gguf \
    --out-dir   ./run \
    -- -fa 1 -r 1
```

Outputs land under `./run/{trace,stall,fetch}/<host>/*.csv` and
`./run/loadwidth.json`. `collect.sh` prints the exact `rocprof-unified-viewer`
command to run next.

### By hand

The four rocprofv3 invocations `collect.sh` wraps are, roughly:

```bash
# 1. sys-trace: real timing, CPU + GPU lanes (shared clock)
rocprofv3 --sys-trace --output-format csv -d run/trace -- \
    ./llama-bench -m M.gguf -p 0 -n 8 -fa 1 -r 1

# 2. PMC counters: stall classification + raw cycles for the EA/ALU busy ratios
rocprofv3 --pmc MemUnitBusy L2CacheHit WriteUnitStalled OccupancyPercent \
    Wavefronts LDSBankConflict GRBM_EA_BUSY GRBM_GUI_ACTIVE \
    SQ_INST_CYCLES_VALU SQ_BUSY_CYCLES --output-format csv -d run/stall -- \
    ./llama-bench -m M.gguf -p 0 -n 2 -fa 1 -r 1

# 3. PMC FETCH_SIZE: measured DRAM read bytes -> achieved bandwidth
rocprofv3 --pmc FETCH_SIZE --output-format csv -d run/fetch -- \
    ./llama-bench -m M.gguf -p 0 -n 2 -fa 1 -r 1

# 4. per-family load widths from device disassembly
rocprof-disasm-loadwidth /path/to/llamacpp-build > run/loadwidth.json
```

The kernel + hip CSVs come from the SAME sys-trace run so they share a clock and
overlay correctly. PMC serializes and replays kernels once per counter-set pass,
which distorts timing -- so the PMC/FETCH CSVs come from SEPARATE runs and are
joined to slices by kernel-name **family** (per-family aggregate, never
per-dispatch).

## Generating the overlay

```bash
rocprof-unified-viewer \
    --kernel-csv     run/trace/<host>/*_kernel_trace.csv \
    --hip-csv        run/trace/<host>/*_hip_api_trace.csv \
    --pmc-csv        run/stall/<host>/*_counter_collection.csv \
    --fetch-csv      run/fetch/<host>/*_counter_collection.csv \
    --loadwidth-json run/loadwidth.json \
    --gguf           model.gguf \
    --out overlay.html --tokens 2
```

Only `--kernel-csv` and `--out` are required; every other input is optional and
adds a layer (`--hip-csv` = CPU lane, `--pmc-csv` = stall coloring, `--fetch-csv`
= achieved-bandwidth column, `--loadwidth-json` = per-lane load width in the
detail panel, `--gguf` = per-dispatch weight-tensor identity + padding/over-fetch).
Then just open `overlay.html` -- it is fully self-contained.

### GGUF weight mapping (`--gguf`)

Decode is strictly periodic: every token replays the same kernel sequence in the
same order. So each `mul_mat_vec` dispatch can be order-mapped to the exact GGUF
weight tensor it multiplies, giving each matvec slice a true identity in the detail
panel: weight name, `[K x N]` shape, quant type, packed on-disk footprint, launch-N
vs true-N padding, and a **measured over-fetch ratio** (per-family+N `FETCH_SIZE` /
packed bytes).

The join key is the launched output-row count `N = Grid_Size_X / Workgroup_Size_X`,
which equals the weight's true `ne[1]`. The kernel-name `(ggml_type)` template arg is
*not* a reliable weight-quant proxy (Q5_K weights dispatch under Q4_K/Q6_K kernels),
so the map joins on **shape (N), not type**. `ffn_gate`+`ffn_up` are fused into one
SwiGLU dispatch at decode; the tool tries both dropping and keeping `ffn_up` and picks
whichever candidate best matches the trace. On Qwen3.5-4B-Q4_K_M (gfx1151) the map is
100% (217 matvec dispatches/token), padding is ~0 (aligned shapes), and over-fetch is
~1.0x -- confirming the decode matvecs are clean read-once streams with no tiling waste.
The mapping % is shown in the header; the parser is stdlib-only (mmap, no `gguf` pip
dep) and only walks the tensor-info table (never reads the 2+GB of weights).

### CLI reference (viewer)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--kernel-csv` | (required) | rocprofv3 `*_kernel_trace.csv` (GPU slices) |
| `--hip-csv` | - | `*_hip_api_trace.csv` (CPU/host HIP-API lane) |
| `--pmc-csv` | - | `*_counter_collection.csv` stall counters (coloring) |
| `--fetch-csv` | - | `*_counter_collection.csv` FETCH_SIZE (achieved BW) |
| `--loadwidth-json` | - | per-family load widths from `disasm_loadwidth.py` |
| `--gguf` | - | GGUF model: order-map matvec dispatch -> weight (shape/pad/over-fetch) |
| `--arch` | `gfx1151` | selects peak DRAM BW for the roofline |
| `--peak-bw` | (from arch) | override peak DRAM bandwidth in GB/s |
| `--out` | (required) | output HTML path |
| `--tokens` | `2` | decode tokens shown in the viewport |
| `--skip-tokens` | `30` | tokens to skip before the window (past warmup) |
| `--context-tokens` | `0` | extra tokens baked on each side for the stepper |
| `--gap-threshold-us` | `150` | inter-dispatch gap marking a token boundary |
| `--title` | `llama.cpp decode overlay (gfx1151)` | HTML title |

## gfx1151 / ROCm gotchas

These are baked into `collect.sh`, but if you run rocprofv3 by hand:

- **Use a pinned local ROCm**, not a shared `/opt/rocm` that can be repointed and
  refreshed under you mid-run (silently breaking `--pmc` / comgr / counters).
  Point `collect.sh` at yours with `--rocm DIR` (or `$ROCM_DIR`).
- **`--pmc` counter config needs the SYSTEM ROCm runtime first** on
  `LD_LIBRARY_PATH` (including the `lib/llvm/lib` subdir for libamd_comgr's
  libLLVM), NOT the build dir. The llama.cpp release bundles its own runtime;
  letting it shadow the system one crashes counter config (SIGSEGV / error 38,
  no CSV). `--sys-trace` is the opposite: it needs the build dir on the path so
  llama-bench finds its bundled ggml/hip libs -- which is why the two run types
  set `LD_LIBRARY_PATH` differently.
- **FETCH_SIZE byte counters need a recent llama.cpp build** (b20260715+); older
  builds SIGSEGV during FETCH_SIZE collection.
- **rocprofv3 may exit nonzero** in its rocpd/OMPT postprocess AFTER the CSV is
  flushed -- judge success by the presence of the CSV, not the exit code
  (`collect.sh` does this for you).
- Timing is only trustworthy from the clean `--sys-trace` run; PMC/FETCH runs
  serialize kernels and are used only for per-family counter values.

## License

MIT. See [LICENSE](LICENSE).

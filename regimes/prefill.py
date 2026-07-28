"""Prefill regime: batch=N prompt tokens through MMQ GEMMs (mul_mat_q). One unit of work
is ONE full forward pass. llama-bench runs an eager warmup then REPEATS the measured pass
several times; in graph mode each measured pass is exactly one hipGraphLaunch, so the clean
way to isolate a single pass is to bake the GPU kernels between the last two launches.

This isolation is why prefill lives in its own file: the old shared "last big inter-dispatch
gap" heuristic baked ALL the repeated passes at once (they run back-to-back with no idle gap
under graph replay) -> ~Nx the weights -> the order-map collapsed. Nothing about decode
should ever touch this logic again.
"""
import sys


class PrefillRegime:
    name = "prefill"
    mm_key = "mul_mat_q"

    def select_window(self, evs, args, model_arch):
        from rocprof_unified_viewer import graph_launch_starts
        from regimes.base import Window

        # Prefer hipGraphLaunch brackets: the last two launches delimit one measured pass.
        gl = graph_launch_starts(args.hip_csv)
        baked = None
        if len(gl) >= 2:
            lo_t, hi_t = gl[-2], gl[-1]
            baked = [ev for ev in evs if lo_t <= ev[0] < hi_t]
        # Fallback (graphs-disabled capture, no launches): last contiguous run of dense GPU
        # work after the final big idle gap. ONLY used when there are no graph launches.
        if not baked:
            prologue = next((i for i, ev in enumerate(evs) if "mul_mat" in ev[2]), 0)
            GAP_NS = max(2_000_000, int(args.gap_threshold_us * 1000))
            a = prologue
            for i in range(len(evs) - 1, prologue, -1):
                if evs[i][0] - evs[i - 1][1] > GAP_NS:
                    a = i
                    break
            baked = evs[a:len(evs)]
        if not baked:
            sys.exit("error: no mul_mat* dispatches in prefill trace "
                     f"{args.kernel_csv}; is this a prefill (-p N -n 0) run?")

        t_first, t_last = baked[0][0], baked[-1][1]
        span = t_last - t_first
        head_pad = min(int(span * 0.10), 12_000_000)  # <= 12ms lead-in
        tail_pad = min(int(span * 0.05), 8_000_000)   # <= 8ms trailing room
        t0, t1 = t_first - head_pad, t_last + tail_pad

        # One span -> two tok_starts entries (pass start, pass end) so the initial viewport
        # spans the whole forward pass; the order-map reset boundary is just {0}.
        tok_starts = [t0, t1]

        # order-map reference = the WHOLE baked pass's matmul Ns (no per-token repeat).
        ref = [n for (s, e, nm, n, _nb, _gy) in baked if self.mm_key in nm and n]

        return Window(baked, t0, t1, tok_starts, lo_tok=0, view_i0=0, view_i1=1,
                      ntok_baked=1, tok_boundary_idx={0}, kstats_ntok=0, ref=ref)

    def metric_labels(self):
        return {"unit": "Forward", "cnt": "cnt/fwd", "primary": "TOPS%",
                "secondary": "time%", "compute_bound": True}

"""Decode regime: batch=1 per-token matvecs (mul_mat_vec). The trace is a periodic stream
of identical tokens; a unit of work is ONE token, bracketed by token boundaries found via
the per-arch BOUNDARY_PROFILES (head-anchor / gap detector). The order-map reference is one
steady-state token's matmul N-sequence, which repeats every token.
"""
import sys

from .base import Regime, Window


class DecodeRegime(Regime):
    name = "decode"
    mm_key = "mul_mat_vec"

    def select_window(self, evs, args, model_arch):
        # Deferred import: these boundary utilities live in the main module (they are the
        # arch-specific decode segmentation and are imported by tests at that path).
        from rocprof_unified_viewer import (
            boundary_profile_for, detect_boundaries, detect_boundaries_by_head)

        # Token segmentation strategy is chosen per model architecture (BOUNDARY_PROFILES).
        # The detectors stay generic; the profile only picks which one + threshold.
        # --boundary-method overrides the method; --gap-threshold-us (if set) the gap.
        prof = boundary_profile_for(model_arch)
        method = args.boundary_method or prof["method"]
        gap_us = args.gap_threshold_us
        if not args._gap_threshold_set and prof["gap_us"] is not None:
            gap_us = prof["gap_us"]

        if method == "head":
            bounds = detect_boundaries_by_head(evs)
        elif method == "gap":
            bounds = detect_boundaries(evs, gap_us * 1000)
        else:  # "auto": head first, gap fallback
            bounds = detect_boundaries_by_head(evs)
            if len(bounds) < args.skip_tokens + args.tokens + 2:
                gap_bounds = detect_boundaries(evs, gap_us * 1000)
                if len(gap_bounds) > len(bounds):
                    bounds = gap_bounds
        if len(bounds) < args.skip_tokens + args.tokens + 2:
            sys.exit(f"error: only {len(bounds)} token boundaries detected "
                     f"(arch={model_arch or 'unknown'}, method={method}); need > "
                     f"{args.skip_tokens + args.tokens}. Lower --skip-tokens, "
                     f"try --boundary-method, or lower --gap-threshold-us.")

        # Bake a wider span (context on each side) so the stepper can pan.
        lo_tok = max(0, args.skip_tokens - args.context_tokens)
        hi_tok = min(len(bounds) - 1, args.skip_tokens + args.tokens + args.context_tokens)
        a = bounds[lo_tok]
        b = bounds[hi_tok]
        baked = evs[a:b]
        t0, t1 = baked[0][0], baked[-1][1]

        tok_starts = [evs[bounds[k]][0] for k in range(lo_tok, hi_tok + 1)]
        view_i0 = args.context_tokens if lo_tok > 0 else 0
        view_i1 = min(len(tok_starts) - 1, view_i0 + args.tokens)

        # order-map reference = ONE steady-state token (between the skip-token boundary and
        # the next), matmul dispatch Ns in execution order; repeats every token.
        ref = [n for (s, e, nm, n, _nb, _gy)
               in evs[bounds[args.skip_tokens]:bounds[args.skip_tokens + 1]]
               if self.mm_key in nm and n]

        # baked-relative indices where each token starts (order-map pointer resets there).
        tok_boundary_idx = {bounds[k] - a for k in range(lo_tok, hi_tok + 1)}
        ntok_baked = hi_tok - lo_tok
        kstats_ntok = max(0, len(bounds) - 1 - args.skip_tokens)

        # Per-position kernel-duration stats: aggregate each op's (position-in-token, family)
        # duration across every token past skip. Needs the periodic per-token segmentation,
        # so it is decode-only (prefill's single pass has no repeats). Computed here where
        # `bounds` lives, rather than leaking bounds into the shared payload path.
        kstats = {}
        if kstats_ntok > 0:
            from rocprof_unified_viewer import family_of
            agg = {}
            for k in range(args.skip_tokens, len(bounds) - 1):
                for ti, (s, e, nm, _n, _nb, _gy) in enumerate(evs[bounds[k]:bounds[k + 1]]):
                    agg.setdefault((ti, family_of(nm)), []).append(e - s)
            for (ti, fam), durs in agg.items():
                cnt = len(durs)
                mean = sum(durs) / cnt
                std = (sum((d - mean) ** 2 for d in durs) / cnt) ** 0.5 if cnt > 1 else 0.0
                kstats["%d|%s" % (ti, fam)] = {
                    "n": cnt, "mean": round(mean, 1), "std": round(std, 1),
                    "min": round(min(durs), 1), "max": round(max(durs), 1),
                }

        return Window(baked, t0, t1, tok_starts, lo_tok, view_i0, view_i1,
                      ntok_baked, tok_boundary_idx, kstats_ntok, ref, kstats)

    def metric_labels(self):
        return {"unit": "Token", "cnt": "cnt/tok", "primary": "time%",
                "secondary": "BW%", "compute_bound": False}

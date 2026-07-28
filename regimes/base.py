"""Regime abstraction: decode vs prefill are two DIFFERENT ways to slice + interpret the
same trace, and they must not leak assumptions into each other. Each regime owns:

  - how to pick the baked window (which GPU slices are ONE unit of work: a decode token
    vs a prefill forward pass),
  - the order-map reference (the per-unit weight N-sequence to lockstep-match dispatches),
  - the matmul kernel key (mul_mat_vec vs mul_mat_q),
  - the frontend metric labels (BW% vs TOPS%, "Token" vs "Forward").

build_payload selects ONE regime up front and calls it; there are no `if mode ==`
branches for windowing/order-map in the shared path. That is the structural guarantee
that a decode-motivated change (e.g. graphs-off collection) cannot silently break prefill,
which is exactly the coupling that produced the 48-passes-baked bug.
"""


class Window:
    """The result of a regime's window selection -- everything the shared payload builder
    needs, so it never has to know which regime produced it.

    baked           : list of (start, end, name, N, nblk, gy) slices for the baked span
    t0, t1          : absolute ns bounds of the rendered window (with lead-in/context pad)
    tok_starts      : absolute ns of each unit boundary within the window (stepper snaps)
    lo_tok          : index of the first unit in tok_starts that is "real" (past context)
    view_i0, view_i1: initial viewport unit-index range [i0, i1)
    ntok_baked      : number of units baked (decode tokens; 1 for prefill)
    tok_boundary_idx: baked-relative slice indices where a unit starts (order-map resets)
    kstats_ntok     : units to average per-kernel stats over (0 disables per-token stats)
    ref             : the order-map reference N-sequence (one unit's matmul dispatch Ns)
    """

    __slots__ = ("baked", "t0", "t1", "tok_starts", "lo_tok", "view_i0", "view_i1",
                 "ntok_baked", "tok_boundary_idx", "kstats_ntok", "ref", "kstats")

    def __init__(self, baked, t0, t1, tok_starts, lo_tok, view_i0, view_i1,
                 ntok_baked, tok_boundary_idx, kstats_ntok, ref, kstats=None):
        self.baked = baked
        self.t0 = t0
        self.t1 = t1
        self.tok_starts = tok_starts
        self.lo_tok = lo_tok
        self.view_i0 = view_i0
        self.view_i1 = view_i1
        self.ntok_baked = ntok_baked
        self.tok_boundary_idx = tok_boundary_idx
        self.kstats_ntok = kstats_ntok
        self.ref = ref
        # kstats: {"<ti>|<fam>": {n,mean,std,min,max}} per-position kernel-duration stats;
        # decode-only (needs periodic per-token repeats). Empty {} for prefill.
        self.kstats = kstats or {}


class Regime:
    """Interface every regime implements. mm_key is the matmul kernel-name substring this
    regime's GEMMs use. select_window() does all the regime-specific slicing + reference
    building and returns a Window."""

    name = "?"
    mm_key = ""

    def select_window(self, evs, args, model_arch):
        raise NotImplementedError

    def metric_labels(self):
        """Frontend label set for this regime (baked into the payload so the JS reads a
        dict instead of scattered IS_PREFILL ternaries)."""
        raise NotImplementedError

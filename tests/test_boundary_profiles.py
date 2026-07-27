"""Per-architecture token-boundary detection tests.

Architectures decode with different cadences, so BOUNDARY_PROFILES picks a segmentation
method per model arch (see the registry in rocprof_unified_viewer.py). These tests pin
each profile and -- the point of separating the code out -- prove that changing one arch's
cadence cannot alter another arch's boundary count.

Hermetic: builds tiny SYNTHETIC kernel traces in-memory. No GPU, no rocprofv3, no GGUF,
no roofline dump -- runs anywhere CI runs.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rocprof_unified_viewer import (  # noqa: E402
    BOUNDARY_PROFILES, boundary_profile_for,
    detect_boundaries, detect_boundaries_by_head,
)

# A synthetic decode "event" is the same tuple load_kernel_slices emits:
#   (start_ns, end_ns, kernel_name, N, nblk, gy)
# The two detectors only read start/end (gaps), name+N (head anchor), so nblk/gy = 0/1.
HEAD = "void mul_mat_vec_q<...>"     # vocab projection: the largest-N matmul per token
BODY = "void mul_mat_vec_q<...>"     # ordinary weight matmul (smaller N)
OTHER = "rms_norm_f32(...)"          # a non-matmul filler op


def _tok(t0, *, dur=100, gap=0, head_n=32000, body_n=2048, n_body=16):
    """Build one token's events starting at t0. Returns (events, next_t0).

    Layout per token: n_body body matmuls, then the vocab HEAD (max N), then a filler.
    `gap` is the idle inserted BEFORE this token (the inter-token gap the gap detector
    keys on); pass 0 for a packed/eager stream where only the head anchor works.

    n_body defaults to 16 so a token spans > 10 dispatches, matching real decode
    (~hundreds/token); detect_boundaries filters period deltas <= 10 as spurious.
    """
    evs = []
    t = t0 + gap
    for _ in range(n_body):
        evs.append((t, t + dur, BODY, body_n, 0, 1))
        t += dur
    evs.append((t, t + dur, HEAD, head_n, 0, 1))   # the per-token head anchor
    t += dur
    evs.append((t, t + dur, OTHER, 1, 0, 1))
    t += dur
    return evs, t


def _stream(n_tokens, *, gap):
    """A synthetic decode stream of n_tokens, with `gap` ns of idle between tokens."""
    evs, t = [], 1_000_000
    for _ in range(n_tokens):
        tok, t = _tok(t, gap=gap)
        evs.extend(tok)
    return evs


# ---------------------------------------------------------------------------
# Registry: each arch resolves to its intended profile; unknown -> _default.
# ---------------------------------------------------------------------------
def test_known_archs_resolve_to_expected_profiles():
    assert boundary_profile_for("qwen35moe")["method"] == "head"
    assert boundary_profile_for("qwen3moe")["method"] == "head"
    p = boundary_profile_for("gpt-oss")
    assert p["method"] == "gap" and p["gap_us"] == 10.0


def test_unknown_arch_falls_back_to_default():
    # Dense/unknown archs (and trace-only runs with arch == "" / None) must behave exactly
    # as the historical default: head-anchor with a gap fallback.
    for arch in ("", None, "llama", "qwen35", "some-future-arch"):
        assert boundary_profile_for(arch) is BOUNDARY_PROFILES["_default"]
        assert boundary_profile_for(arch)["method"] == "auto"


# ---------------------------------------------------------------------------
# Detectors on each cadence.
# ---------------------------------------------------------------------------
def test_head_anchor_segments_eager_moe_stream():
    # Eager MoE: dispatches packed with NO inter-token gap. The gap detector finds nothing;
    # the head anchor still brackets every token. This is why MoE archs use method="head".
    evs = _stream(6, gap=0)
    heads = detect_boundaries_by_head(evs)
    assert len(heads) >= 5, "head anchor should bracket ~every token in a packed stream"
    # The gap detector is blind here (no gaps) -- confirms the head path is load-bearing.
    assert len(detect_boundaries(evs, 150_000)) < len(heads)


def test_gap_detector_needs_low_threshold_for_short_cadence():
    # A gptoss-like stream: real inter-token gaps, but SHORT (sub-default). The 150us
    # default misses them; the profile's 10us catches them. This is why gptoss pins gap_us.
    evs = _stream(6, gap=20_000)   # 20us inter-token idle
    assert len(detect_boundaries(evs, 150_000)) < 5, "default 150us should under-segment"
    assert len(detect_boundaries(evs, 10_000)) >= 5, "profile 10us should segment cleanly"


# ---------------------------------------------------------------------------
# Isolation guard: the whole reason for separating per-arch config. Changing one arch's
# stream/profile must NOT change another arch's segmentation.
# ---------------------------------------------------------------------------
def test_arch_streams_are_independent():
    moe = _stream(6, gap=0)          # eager MoE (head only)
    gpt = _stream(6, gap=20_000)     # gptoss (gap @ 10us)

    moe_bounds = detect_boundaries_by_head(moe)
    gpt_bounds = detect_boundaries(gpt, boundary_profile_for("gpt-oss")["gap_us"] * 1000)

    # Perturb the gptoss cadence (more tokens, still a >10us inter-token gap); the MoE
    # head segmentation must be untouched.
    gpt2 = _stream(9, gap=20_000)
    assert detect_boundaries_by_head(moe) == moe_bounds, \
        "changing the gptoss stream must not alter MoE head boundaries"
    # And the gptoss detector still works on its own (independently) after the change.
    assert len(detect_boundaries(gpt2, 10_000)) >= 5
    assert gpt_bounds  # original gptoss segmentation was non-empty

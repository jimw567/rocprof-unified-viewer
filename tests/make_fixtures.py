#!/usr/bin/env python3
"""Derive the tiny committed CSV fixtures in tests/fixtures/ from a real decode
collection. Run ONCE locally by a maintainer with access to a collect.sh output tree;
the trimmed outputs are committed (~300 KB total) so CI (no GPU / no rocprofv3) can
smoke-test the generator offline.

The real kernel CSV is ~60k rows for 64 decode tokens (~960 dispatches/token). Since
decode is periodic, we keep only the busiest stream and a HEAD_PER_TOKEN slice after
each of the first N_TOKENS_KEEP+1 token boundaries -- preserving the inter-token gaps
that detect_boundaries keys on, plus a diverse kernel mix, at ~24 (not 960) dispatches
per token. HIP-API rows are down-sampled (CPU lane only needs to render); PMC/FETCH keep
PMC_PER_FAM dispatches per (family, counter) since those are aggregated to a per-family
mean. The result clears the token gate for --skip-tokens 0 --tokens 2.

Usage:
  python3 tests/make_fixtures.py \
    --src /proj/gdba/jimwu/rocm/tmp/both-collect-test/decode --host xconucstrhalo41
"""
import argparse
import csv
import glob
import os
import sys

# Reuse the viewer's OWN slice loader + boundary detector so the fixture is trimmed
# on exactly the token edges the generator will later find (no approximation).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rocprof_unified_viewer import load_kernel_slices, detect_boundaries  # noqa: E402

N_TOKENS_KEEP = 5          # decode tokens to retain (gate needs > skip+tokens)
HEAD_PER_TOKEN = 24        # dispatches kept per token (after its boundary); the rest of
                           # the ~960/token are dropped -- decode is periodic so a head
                           # slice per token keeps the boundary gaps + a diverse kernel
                           # mix while shrinking the fixture ~40x.
GAP_NS = 150_000           # inter-dispatch gap that marks a token edge (viewer default)
HIP_SAMPLE = 20            # keep every Nth HIP-API row (CPU lane; fidelity not needed)
PMC_PER_FAM = 2            # dispatches per family per counter to keep (PMC is meaned)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _one(pattern):
    m = glob.glob(pattern)
    if not m:
        raise SystemExit("no file matching %s" % pattern)
    return m[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="collect.sh <out>/decode dir")
    ap.add_argument("--host", required=True, help="board hostname subdir")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kpath = _one("%s/trace/%s/*_kernel_trace.csv" % (a.src, a.host))
    hpath = _one("%s/trace/%s/*_hip_api_trace.csv" % (a.src, a.host))
    ppath = _one("%s/stall/%s/*_counter_collection.csv" % (a.src, a.host))
    fpath = _one("%s/fetch/%s/*_counter_collection.csv" % (a.src, a.host))

    # 1) kernel rows: keep only the busiest stream, and within it a HEAD_PER_TOKEN
    # slice after each of the first N_TOKENS_KEEP+1 token boundaries. Decode is
    # periodic, so a head slice per token preserves the boundary gaps + a diverse
    # kernel mix while shrinking ~960 dispatches/token down to ~24. The gap BEFORE
    # each kept head is intact (the dispatches we drop are all AFTER the head, before
    # the next boundary) so detect_boundaries still finds > N_TOKENS_KEEP edges.
    by_stream = load_kernel_slices(kpath)
    sid = max(by_stream, key=lambda s: len(by_stream[s]))
    evs = by_stream[sid]                       # (start, end, name, N, nblk) sorted
    bounds = detect_boundaries(evs, GAP_NS)
    if len(bounds) <= N_TOKENS_KEEP + 1:
        raise SystemExit("source has only %d boundaries; need > %d"
                         % (len(bounds), N_TOKENS_KEEP + 1))
    keep_idx = set()
    for b in bounds[:N_TOKENS_KEEP + 1]:
        for i in range(b, min(b + HEAD_PER_TOKEN, len(evs))):
            keep_idx.add(i)
    keep_ts = set(evs[i][0] for i in keep_idx)   # start timestamps of kept dispatches
    windows = [(evs[i][0], evs[i][1]) for i in sorted(keep_idx)]  # for hip filtering
    with open(kpath) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        hdr = rd.fieldnames
    kept = [r for r in rows
            if r["Stream_Id"] == sid and int(r["Start_Timestamp"]) in keep_ts]
    _write(os.path.join(OUT, "decode_kernel_trace.csv"), hdr, kept)

    # 2) hip rows overlapping any kept GPU dispatch window, down-sampled (the CPU lane
    # only needs to render, not be exhaustive).
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    with open(hpath) as fh:
        rd = csv.DictReader(fh)
        cand = [r for r in rd
                if int(r["End_Timestamp"]) >= lo and int(r["Start_Timestamp"]) <= hi]
        hhdr = rd.fieldnames
    hrows = cand[::HIP_SAMPLE]
    _write(os.path.join(OUT, "decode_hip_api_trace.csv"), hhdr, hrows)

    # 3) PMC/FETCH: PMC is aggregated to a per-family MEAN, so a few dispatches per
    # (family, counter) reproduce the same coloring/BW. Keep PMC_PER_FAM of each.
    fams = set(r["Kernel_Name"] for r in kept)
    for src, dst in ((ppath, "decode_stall_counter_collection.csv"),
                     (fpath, "decode_fetch_counter_collection.csv")):
        with open(src) as fh:
            rd = csv.DictReader(fh)
            chdr = rd.fieldnames
            seen = {}
            crows = []
            for r in rd:
                if r["Kernel_Name"] not in fams:
                    continue
                key = (r["Kernel_Name"], r.get("Counter_Name", ""))
                n = seen.get(key, 0)
                if n >= PMC_PER_FAM:
                    continue
                seen[key] = n + 1
                crows.append(r)
        _write(os.path.join(OUT, dst), chdr, crows)

    print("wrote fixtures to %s (kernel rows kept: %d)" % (OUT, len(kept)))


def _write(path, hdr, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

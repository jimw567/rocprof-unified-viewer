#!/usr/bin/env python3
"""Watch llama.cpp's per-architecture graph builders for changes.

The layer-graph feature reconstructs edges from a ggml-roofline storage-id topology dump
(architecture-agnostic -- no per-model template), so a build_graph change never silently
breaks reconstruction. But it still matters to KNOW when upstream:

  - adds a NEW architecture (a model we may start running, new block shapes), or
  - CHANGES an existing architecture's graph wiring (new ops, reordered dataflow),

because that can change what the overlay's swim-lane / roofline dump contains and is worth
a human glance. This script snapshots each `src/models/*.cpp` graph builder (plus the
shared helpers in `src/llama-graph.cpp`) and diffs against a saved baseline, reporting
ADDED / REMOVED / CHANGED architectures.

Usage:
    scan_build_graph.py --repo /path/to/llama.cpp [--baseline FILE] [--update]

  --repo      llama.cpp checkout to scan (required)
  --baseline  JSON baseline to compare against (default: ~/.cache/ruv/build_graph_baseline.json)
  --update    write the current snapshot as the new baseline (after reviewing the diff)

Exit code 0 = no changes, 1 = changes detected (so a cron/CI can alert). Read-only unless
--update is passed. No network, no build -- pure source hashing.
"""
import argparse
import hashlib
import json
import os
import re
import sys

# The graph builder is the ctor body `::graph::graph(...) { ... }` in each model file;
# fall back to hashing the whole file when the pattern is absent (a few builders differ).
_GRAPH_RE = re.compile(r"::graph::graph\([^{]*\{(.*)\}\s*$", re.S)


def _snapshot(repo):
    models_dir = os.path.join(repo, "src", "models")
    if not os.path.isdir(models_dir):
        sys.exit("error: %s has no src/models (not a current llama.cpp checkout?)" % repo)
    snap = {}
    for fn in sorted(os.listdir(models_dir)):
        if not fn.endswith(".cpp"):
            continue
        txt = open(os.path.join(models_dir, fn)).read()
        m = _GRAPH_RE.search(txt)
        body = m.group(1) if m else txt
        snap[fn] = hashlib.sha1(body.encode("utf-8", "replace")).hexdigest()[:16]
    # Shared graph helpers (build_attn / build_ffn / build_qkv, etc.) that every builder
    # expands through -- a change here affects many architectures at once.
    helpers = os.path.join(repo, "src", "llama-graph.cpp")
    if os.path.isfile(helpers):
        snap["__llama-graph.cpp"] = hashlib.sha1(
            open(helpers, "rb").read()).hexdigest()[:16]
    return snap


def _diff(old, new):
    old_k, new_k = set(old), set(new)
    added = sorted(new_k - old_k)
    removed = sorted(old_k - new_k)
    changed = sorted(k for k in (old_k & new_k) if old[k] != new[k])
    return added, removed, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="llama.cpp checkout to scan")
    ap.add_argument("--baseline",
                    default=os.path.expanduser("~/.cache/ruv/build_graph_baseline.json"))
    ap.add_argument("--update", action="store_true",
                    help="write the current snapshot as the new baseline")
    args = ap.parse_args()

    snap = _snapshot(args.repo)

    if not os.path.exists(args.baseline):
        os.makedirs(os.path.dirname(args.baseline), exist_ok=True)
        with open(args.baseline, "w") as fh:
            json.dump(snap, fh, indent=1, sort_keys=True)
        print("baseline created: %s (%d architectures)" % (args.baseline, len(snap) - 1))
        return 0

    old = json.load(open(args.baseline))
    added, removed, changed = _diff(old, snap)

    if not (added or removed or changed):
        print("build_graph: no changes (%d architectures)" % (len(snap) - 1))
        return 0

    print("build_graph CHANGES detected in %s:" % args.repo)
    for k in added:
        print("  ADDED    %s" % k)
    for k in removed:
        print("  REMOVED  %s" % k)
    for k in changed:
        tag = "shared helpers" if k.startswith("__") else "graph wiring"
        print("  CHANGED  %s  (%s)" % (k, tag))
    print("\nReview whether any of these affect models we run, then re-run with "
          "--update to accept the new baseline.")

    if args.update:
        with open(args.baseline, "w") as fh:
            json.dump(snap, fh, indent=1, sort_keys=True)
        print("baseline updated: %s" % args.baseline)
    return 1


if __name__ == "__main__":
    sys.exit(main())

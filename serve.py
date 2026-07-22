#!/usr/bin/env python3
"""Companion server for rocprof-unified-viewer: live ATT tracing from the browser.

The generated overlay is a static self-contained HTML. To fold per-instruction
ATT thread-trace stalls into a kernel's detail panel, the static flow is a
copy-paste round-trip (the panel prints a `collect-att.sh` command, you run it on
a board by hand, then regenerate with `--att-dir`). This server closes that loop:

    browser  ->  http://127.0.0.1:PORT  (this server, on YOUR workstation)
       | click a kernel -> "Trace now"
       v  POST /api/trace {kernel}
    server: pick a FREE-GPU host from $ROCPROF_ATT_HOSTS (ssh rocm-smi --showuse),
            pipe collect-att.sh over `ssh <host> bash -s` to run ATT on the board,
            read the decoded CSVs off shared NFS, load_att_stats() -> per-family JSON
       v
    browser: merge into the detail panel live.

Design constraints (deliberate):
  * The web server runs on THIS machine; all GPU work is dispatched to a board via
    ssh. No GPU is required to run the server (see --stub-att-dir).
  * NO hardcoded hostnames. The board list comes from an environment variable
    (default $ROCPROF_ATT_HOSTS, a colon-separated list); the server auto-selects
    a host whose GPU is idle.
  * Purely additive: the static export path (rocprof_unified_viewer.py) is
    unchanged. Only the served HTML flips `att_server` true to show "Trace now".

Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rocprof_unified_viewer import (ISA_GLOSSARY, REG_GLOSSARY, add_common_args,
                                    build_payload, load_att_code, load_att_stats,
                                    render_html)

# --- global server state (set in main) --------------------------------------
_ARGS = None
_PAYLOAD = None          # payload dict rendered into the served HTML
_HTML = b""              # cached rendered HTML bytes
_ALLOWED_SYMS = set()    # kernel symbols the client is allowed to ask ATT for
_ATT_CACHE = {}          # family -> stats dict (grows as traces complete)
_TRACE_LOCK = threading.Lock()   # single-flight: the board GPU + ATT is exclusive


def _ssh_base(host):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]


def _gpu_use_pct(host):
    """Return the GPU busy % for a host via `rocm-smi --showuse` over ssh, or None
    if unreachable / unparseable. rocm-smi is NOT on the default PATH on the boards,
    so we invoke it by full path under the board-side ROCm dir."""
    rocm_smi = os.path.join(_ARGS.rocm, "bin", "rocm-smi")
    try:
        out = subprocess.run(
            _ssh_base(host) + [shlex.quote(rocm_smi) + " --showuse"],
            capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 and not out.stdout:
        return None
    m = re.search(r"GPU use \(%\)\s*:\s*(\d+)", out.stdout)
    return int(m.group(1)) if m else None


def _host_status():
    """Probe every host in the list; return [{host, gpu_use_pct, reachable}]."""
    hosts = [h for h in os.environ.get(_ARGS.hosts_env, "").split(":") if h]
    rows = []
    for h in hosts:
        pct = _gpu_use_pct(h)
        rows.append({"host": h, "gpu_use_pct": pct, "reachable": pct is not None})
    return rows


def _pick_free_host():
    """Pick the first host at/under the busy threshold; else the least-busy
    reachable host. Returns (host, status_rows) or (None, status_rows)."""
    rows = _host_status()
    reachable = [r for r in rows if r["reachable"]]
    if not reachable:
        return None, rows
    for r in reachable:
        if r["gpu_use_pct"] <= _ARGS.busy_threshold:
            return r["host"], rows
    reachable.sort(key=lambda r: r["gpu_use_pct"])
    return reachable[0]["host"], rows


# ggml_type enum ints for the quant tags carried in a family name (e.g. "[Q4_K]").
# Used to build a shape-exact test-backend-ops MUL_MAT case for a selected matvec.
_GGML_TYPE_INT = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8,
                  "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}


def _run_att_on_host(host, sym, out_dir, shape=None):
    """Pipe the local collect-att.sh over `ssh host bash -s` so the board runs the
    repo's current script (no board deployment). Returns (ok, log).

    When `shape` (a {quant,K,N} dict from the selected matvec) is given AND the
    server is in test-backend-ops runner mode, trace the EXACT decode shape: inject
    it via GGML_ATT_MULMAT (needs the patched test-backend-ops) and filter the perf
    run with -p to that single case, so ATT captures the model's real dims."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "collect-att.sh")
    with open(script) as fh:
        script_src = fh.read()
    # Args after `--` become $1.. for `bash -s`. sym is allowlist-validated; quote
    # everything anyway. Never shell=True.
    sh_args = ["--", "--kernel", sym,
               "--build-dir", _ARGS.build_dir,
               "--out-dir", out_dir,
               "--rocm", _ARGS.rocm]
    if _ARGS.runner:
        # Single-kernel evaluator instead of the full llama-bench graph. The
        # runner string is passed as one collect-att.sh arg (it word-splits it);
        # --model / --bench-flags do not apply in this mode.
        runner = _ARGS.runner
        # Shape-exact matvec: if the selected family carries a quant + K + N and the
        # runner is test-backend-ops, target that exact MUL_MAT case.
        q = (shape or {}).get("q")
        K = (shape or {}).get("K")
        N = (shape or {}).get("N")
        ti = _GGML_TYPE_INT.get(str(q).upper()) if q else None
        if ti and K and N and "test-backend-ops" in runner and "-o MUL_MAT" in runner:
            # Override the configured runner with a shape-exact PERF run: the env var
            # injects the case; -p selects it by its unique dims (m=N,n=1,k=K). Perf
            # mode runs thousands of iterations so the ATT capture survives cutoff
            # (test mode's single dispatch gets truncated). Filter on dims only --
            # the quant is pinned by GGML_ATT_MULMAT, and vars() prints the quant tag
            # mixed-case which a regex would miss. NO quotes around the -p value:
            # collect-att.sh word-splits the runner string (WORKLOAD=($RUNNER)), so
            # quotes would become literal chars; the value has no spaces so bare works.
            base = runner.split(" -p ")[0]
            runner = '%s -p m=%d,n=1,k=%d' % (base, int(N), int(K))
            sh_args += ["--runner-env",
                        "GGML_ATT_MULMAT=%d,%d,%d" % (ti, int(K), int(N))]
            # DISABLE HIP graphs: test-backend-ops perf wraps the repeated launches in
            # a HIP graph, so ATT sees one opaque graphLaunch and can't thread-trace
            # the kernel (-> 0 populated dispatches). Stream mode makes each launch a
            # visible dispatch ATT can capture. Essential for the single-shape run.
            sh_args += ["--runner-env", "GGML_CUDA_DISABLE_GRAPHS=1"]
        sh_args += ["--runner", runner]
    else:
        sh_args += ["--model", _ARGS.model]
        bench = shlex.split(_ARGS.bench_flags) if _ARGS.bench_flags else []
        if bench:
            sh_args += ["--"] + bench
    argv = _ssh_base(host) + ["bash", "-s"] + [shlex.quote(a) for a in sh_args]
    try:
        proc = subprocess.run(argv, input=script_src, capture_output=True,
                              text=True, timeout=_ARGS.trace_timeout)
    except subprocess.TimeoutExpired:
        return False, "trace timed out after %ds" % _ARGS.trace_timeout
    except OSError as e:
        return False, "ssh failed: %s" % e
    log = (proc.stdout or "") + (proc.stderr or "")
    # collect-att.sh judges success by decoded CSVs, not exit code; verify on NFS.
    hit = glob.glob(os.path.join(out_dir, "**",
                                 "stats_ui_output_*_dispatch_*.csv"),
                    recursive=True)
    if not hit:
        return False, "no decoded ATT CSVs produced\n" + log[-2000:]
    return True, log


def _do_trace(sym, force=False, shape=None):
    """Run (or stub) a trace for one kernel symbol; fold results into the cache.
    Returns (http_status, response_dict).

    force=True wipes any existing att-<sym>/ output on disk BEFORE re-running, so a
    re-trace genuinely re-collects instead of accumulating alongside (and reloading)
    stale decoded data -- the usual reason to force is that the on-disk trace came
    from a build WITHOUT DWARF line tables and you want a fresh one that has them.

    shape={quant,K,N} (optional) targets a shape-exact matvec (see _run_att_on_host)."""
    if sym not in _ALLOWED_SYMS:
        return 400, {"ok": False, "error": "unknown kernel symbol"}
    if not _TRACE_LOCK.acquire(blocking=False):
        return 409, {"ok": False,
                     "error": "a trace is already running (the board GPU is "
                              "exclusive) -- try again shortly"}
    try:
        if _ARGS.stub_att_dir:
            src_dir = _ARGS.stub_att_dir
            host = "stub"
        else:
            host, rows = _pick_free_host()
            if not host:
                return 503, {"ok": False,
                             "error": "no reachable host in $%s (%s)"
                                      % (_ARGS.hosts_env,
                                         ":".join(r["host"] for r in rows) or "empty")}
            src_dir = os.path.join(_out_base(), "att-" + sym)
            if force:
                # only ever remove OUR own per-symbol scratch dir under the out-base
                shutil.rmtree(src_dir, ignore_errors=True)
            os.makedirs(src_dir, exist_ok=True)
            ok, log = _run_att_on_host(host, sym, src_dir, shape=shape)
            if not ok:
                return 500, {"ok": False, "error": log, "host": host}
        stats = load_att_stats(src_dir)
        code = load_att_code(src_dir)
        if not stats:
            return 500, {"ok": False, "host": host,
                         "error": "trace produced no populated dispatches "
                                  "(all cut off) -- retry"}
        _ATT_CACHE.update(stats)
        return 200, {"ok": True, "host": host, "fam_stats": stats,
                     "fam_code": code}
    finally:
        _TRACE_LOCK.release()


def _out_base():
    base = _ARGS.out_base or os.environ.get("ROCPROF_ATT_OUT_BASE")
    if not base:
        raise RuntimeError(
            "no NFS scratch base for trace output: set --out-base or "
            "$ROCPROF_ATT_OUT_BASE to a directory shared between this machine "
            "and the boards (so the server can read the decoded CSVs)")
    return base


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        sys.stderr.write("[serve] " + (fmt % a) + "\n")

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)
        elif path == "/api/hosts":
            if _ARGS.stub_att_dir:
                self._send_json(200, [{"host": "stub", "gpu_use_pct": 0,
                                       "reachable": True}])
            else:
                self._send_json(200, _host_status())
        elif path == "/api/att":
            q = parse_qs(urlparse(self.path).query)
            fam = q.get("fam", [""])[0]
            st = _ATT_CACHE.get(fam)
            self._send_json(200 if st else 404,
                            {"ok": bool(st), "fam_stats": {fam: st} if st else {}})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/trace":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, TypeError):
            self._send_json(400, {"ok": False, "error": "bad JSON body"})
            return
        sym = (req.get("kernel") or "").strip()
        force = bool(req.get("force"))
        # optional shape-exact matvec target {q, K, N} for the selected dispatch.
        shape = req.get("shape") if isinstance(req.get("shape"), dict) else None
        try:
            status, resp = _do_trace(sym, force=force, shape=shape)
        except RuntimeError as e:
            status, resp = 500, {"ok": False, "error": str(e)}
        self._send_json(status, resp)


def main():
    global _ARGS, _PAYLOAD, _HTML, _ALLOWED_SYMS
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)   # same input/render flags as the generator (--out unused)
    ap.add_argument("--port", type=int, default=8756,
                    help="localhost port to serve on (default 8756)")
    ap.add_argument("--hosts-env", default="ROCPROF_ATT_HOSTS",
                    help="env var holding a colon-separated list of GPU board "
                         "hostnames to dispatch ATT to (default ROCPROF_ATT_HOSTS). "
                         "No hostname is ever hardcoded.")
    ap.add_argument("--model",
                    help="GGUF model to run under ATT on the board (default: --gguf)")
    ap.add_argument("--rocm", default=os.environ.get("ROCM_DIR"),
                    help="board-side ROCm dir (drives rocprofv3 + rocm-smi; "
                         "default $ROCM_DIR). Must be a path valid ON the board "
                         "(shared NFS).")
    ap.add_argument("--out-base",
                    help="NFS scratch base for per-kernel trace dirs (att-<sym>/); "
                         "must be shared between this machine and the boards. "
                         "Default $ROCPROF_ATT_OUT_BASE.")
    ap.add_argument("--busy-threshold", type=int, default=10,
                    help="a host with GPU use <= this %% counts as free (default 10)")
    ap.add_argument("--bench-flags", default="-fa 1",
                    help="llama-bench flags passed through to collect-att.sh after "
                         "-- (default '-fa 1'; must match how the trace was collected)")
    ap.add_argument("--runner",
                    help="run a single-kernel evaluator under ATT instead of the "
                         "full llama-bench decode graph (passed to collect-att.sh "
                         "--runner; cwd = --build-dir). Fewer cutoff dispatches, "
                         "controlled shapes. Example: "
                         "'./test-backend-ops perf -o MUL_MAT -p type_a=q4_K'. "
                         "When set, --model / --bench-flags are not used.")
    ap.add_argument("--trace-timeout", type=int, default=300,
                    help="seconds to wait for a board ATT trace (default 300)")
    ap.add_argument("--stub-att-dir",
                    help="GPU-free testing: instead of ssh-ing to a board, return "
                         "load_att_stats(DIR) for every trace request. Verifies the "
                         "whole client->server->fold loop without a GPU.")
    args = ap.parse_args()
    if not args.model:
        args.model = args.gguf
    _ARGS = args

    if not args.stub_att_dir:
        if not args.rocm:
            ap.error("--rocm (or $ROCM_DIR) is required unless --stub-att-dir is set")
        if not args.build_dir:
            ap.error("--build-dir is required unless --stub-att-dir is set")
        if not args.model and not args.runner:
            ap.error("--model (or --gguf) is required unless --stub-att-dir or "
                     "--runner is set")

    _PAYLOAD = build_payload(args)
    _PAYLOAD["att_server"] = True
    # In live-server mode the debug view is produced on demand via "Trace now",
    # so embed the full ISA + register glossaries even when no --att-dir was
    # given at startup (build_payload only embeds them when att code is present).
    _PAYLOAD["isa_gloss"] = ISA_GLOSSARY
    _PAYLOAD["reg_gloss"] = REG_GLOSSARY
    _ALLOWED_SYMS = {fam.split("[", 1)[0] for fam in _PAYLOAD["fam_counters"]}
    _HTML = render_html(_PAYLOAD).encode()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    mode = ("STUB (%s)" % args.stub_att_dir if args.stub_att_dir
            else "$%s -> %s" % (args.hosts_env,
                                os.environ.get(args.hosts_env, "(unset!)")))
    print("rocprof-unified-viewer companion server", file=sys.stderr)
    print("  http://127.0.0.1:%d  (bind localhost only)" % args.port, file=sys.stderr)
    print("  ATT dispatch: %s" % mode, file=sys.stderr)
    print("  %d kernel symbols traceable" % len(_ALLOWED_SYMS), file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)


if __name__ == "__main__":
    main()

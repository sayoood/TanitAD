#!/usr/bin/env python3
"""DETACHED JOB WRAPPER for the u400 scene-cache build (and any other single GPU command): queue behind the shared dev-box GPU lock, run the command, write a DONE or FAILED marker
as the LAST action, always release the lock. Costs no model turns while it waits.

    python run_cache_job.py --marker-dir <dir> --tag cache_u400_4k --give-up-h 8 -- <cmd...>
Markers (in --marker-dir): <tag>.STARTED (json, written at once), then exactly one of <tag>.DONE / <tag>.FAILED (json: rc, seconds, log tail). DONE requires rc 0 AND the success token
`--ok-token` (default ZZCACHE_OK) in the child's output -- an exit code alone is not evidence the cache is right (proxy_cache prints the token only when C1 is bit-exact).
The child's stdout/stderr go to <marker-dir>/<tag>.child.log.  The lock is held for the whole child run; if the lock is not obtained within --give-up-h the marker is FAILED(lock).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock as GL


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        print("usage: run_cache_job.py --marker-dir D --tag T [--give-up-h 8] [--ok-token ZZCACHE_OK] [--no-lock] -- cmd ...")
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--give-up-h", type=float, default=8.0)
    ap.add_argument("--ok-token", default="ZZCACHE_OK")
    ap.add_argument("--no-lock", action="store_true", help="TEST ONLY: skip the GPU lock (a CPU command)")
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    md = Path(a.marker_dir)
    md.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    child_log = md / f"{a.tag}.child.log"
    (md / f"{a.tag}.STARTED").write_text(json.dumps({"pid": os.getpid(), "cmd": cmd, "t": t0, "local": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    rc, why, body = None, None, None
    try:
        if not a.no_lock:
            body = GL.acquire_blocking(a.give_up_h * 3600, log=lambda m: print(m, flush=True))
            print(f"[{time.strftime('%H:%M:%S')}] GPU lock taken: {body}", flush=True)
        with open(child_log, "wb") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        tail = child_log.read_bytes()[-3000:].decode("utf-8", "replace")
        ok = rc == 0 and a.ok_token in child_log.read_text(encoding="utf-8", errors="replace")
        why = None if ok else (f"rc={rc}" if rc != 0 else f"rc 0 but the token {a.ok_token!r} is absent from the child's output")
    except Exception:
        ok, tail, why = False, "", "wrapper exception: " + traceback.format_exc()[-1500:]
    finally:
        if body is not None:
            GL.release_gpu_lock(GL.LOCK, body)
    rec = {"rc": rc, "seconds": round(time.time() - t0, 1), "reason": why, "child_log": str(child_log), "tail": tail, "local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    final = md / (f"{a.tag}.DONE" if ok else f"{a.tag}.FAILED")
    final.write_text(json.dumps(rec, indent=1), encoding="utf-8")          # LAST ACTION
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

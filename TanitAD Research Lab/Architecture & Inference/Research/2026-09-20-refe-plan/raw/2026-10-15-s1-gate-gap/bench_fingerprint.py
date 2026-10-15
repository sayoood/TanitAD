#!/usr/bin/env python3
"""Timing + identity harness for the launch gate's fingerprints: the OLD gate_token.py (pre-2026-10-15) against the NEW one, on the same inputs.  CPU + file metadata only: no GPU, no
network, writes nothing but --out (and, on request, a synthetic tree under --synth-root).  Runnable ON THE POD (stat + read only) -- the dev-box figures are NOT the pod's.

  python bench_fingerprint.py --new <package>/refe/gate_token.py --token <PASS_*.json> --threads 1,16 --out token_identity.json     # the identity proof on the REAL data, no old code run (about 1 minute)
  python bench_fingerprint.py --old gate_token_OLD_snapshot.py --new <package>/refe/gate_token.py --out fp_timing.json \
        [--dir <a directory>]... [--file <a big file>]... [--threads 1,2,4,8,16,32] [--repeat 3]
        [--synth-root <dir> --synth-files 413184 --synth-shards 32]          # build (once) and time a synthetic tree of the pod's image-tree shape
        [--simulate-latency-ms 0.5 --simulate-files 20000]                    # SIMULATED per-stat latency (a sleep inside os.stat): mechanism only, never a pod figure
        [--store <label_store> --ckpt <checkpoint> --images <dir> --calib <file>]   # a realistic data_manifest() call, old vs new

EVIDENCE CLASSES in the JSON: every timing is MEASURED on the host named in `env` (that host only); `simulated` blocks are SIMULATED; nothing here is a pod measurement unless the JSON's
`env.node` is the pod.  Every old-vs-new comparison also records `identical` (dict equality of the fingerprints): a speed-up with `identical: false` is a defect, not a result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def cpu_load():
    """the box's CPU load at the start (Windows: WMI via PowerShell; POSIX: 1-minute loadavg per core) -- the figures of a saturated shared box are order-of-magnitude, say so"""
    try:
        if os.name == "nt":
            import subprocess
            r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"],
                               capture_output=True, text=True, timeout=30)
            return float(r.stdout.strip())
        return round(100 * os.getloadavg()[0] / (os.cpu_count() or 1), 1)
    except Exception:                                                # noqa: BLE001
        return None


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def timeit(fn, repeat: int):
    ts, val = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        val = fn()
        ts.append(time.perf_counter() - t0)
    return {"min_s": round(min(ts), 3), "median_s": round(statistics.median(ts), 3), "all_s": [round(t, 3) for t in ts]}, val


def build_synth(root: Path, n_files: int, shards: int) -> None:
    """a tree of the pod's image-tree shape: `shards` directories navtrain_current_NN of ~n_files/shards small files each (sizes 0..96 bytes; only the LISTING is ever read)."""
    marker = root.parent / f"{root.name}.done"                      # OUTSIDE the tree: a file inside it would be counted
    if marker.is_file() and marker.read_text() == f"{n_files}:{shards}":
        return
    per = n_files // shards
    extra = n_files - per * shards
    made = 0
    t0 = time.perf_counter()
    for s in range(shards):
        d = root / f"navtrain_current_{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per + (1 if s < extra else 0)):
            with open(d / f"{s:02d}_{i:06d}.jpg", "wb") as fh:
                fh.write(b"x" * (made % 97))
            made += 1
    marker.write_text(f"{n_files}:{shards}")
    print(f"built {made} files under {root} in {time.perf_counter() - t0:.0f}s", flush=True)


def bench_dir(old, new, path: str, threads: list, repeat: int, label: str) -> dict:
    out = {"path_label": label, "old": None, "new": {}}
    t, o = timeit(lambda: old.fingerprint(path), repeat)
    out["old"] = t
    out["n_files"], out["bytes"] = o.get("n_files"), o.get("bytes")
    for n in threads:
        os.environ[new.FP_THREADS_ENV] = str(n)
        t, nv = timeit(lambda: new.fingerprint(path), repeat)
        t["identical"] = nv == o
        t["speedup_vs_old_min"] = round(out["old"]["min_s"] / max(t["min_s"], 1e-9), 2)
        out["new"][str(n)] = t
    os.environ.pop(new.FP_THREADS_ENV, None)
    return out


def bench_hash(old, new, path: str, repeat: int) -> dict:
    import hashlib
    size = os.path.getsize(path)
    res = {"bytes": size, "variants": {}}
    want = None

    def readinto_fn(chunk):
        def f():
            h = hashlib.sha256()
            buf = bytearray(chunk)
            mv = memoryview(buf)
            with open(path, "rb", buffering=0) as fh:
                while True:
                    n = fh.readinto(mv)
                    if not n:
                        break
                    h.update(mv[:n])
            return h.hexdigest()
        return f
    variants = {"old gate_token.sha256_file (read 4 MiB)": lambda: old.sha256_file(path), "new gate_token.sha256_file (readinto 4 MiB)": lambda: new.sha256_file(path),
                "readinto 1 MiB": readinto_fn(1 << 20), "readinto 16 MiB": readinto_fn(1 << 24), "hashlib.file_digest (3.11+)": None}
    if hasattr(hashlib, "file_digest"):
        def fd():
            with open(path, "rb") as fh:
                return hashlib.file_digest(fh, "sha256").hexdigest()
        variants["hashlib.file_digest (3.11+)"] = fd
    else:
        variants.pop("hashlib.file_digest (3.11+)")
    for name, fn in variants.items():
        t, v = timeit(fn, repeat)
        want = want or v
        t["digest_equal_to_first"] = v == want
        t["MB_per_s_min"] = round(size / 1e6 / max(t["min_s"], 1e-9))
        res["variants"][name] = t
    return res


def cpu_only(old, new, path: str, threads: list, repeat: int) -> dict:
    """The Python overhead per file with the filesystem taken OUT: os.stat returns one pre-built result instantly (SIMULATED stat; the directory walk itself is the real os.scandir of `path`).
    This bounds how much of a per-file cost is interpreter work (Path objects, string formatting) rather than I/O: the part threads cannot overlap (the GIL) but string ops can shrink."""
    orig = os.stat
    fixed = orig(__file__)                                           # a regular file's stat result, returned for every entry INSIDE the tree
    root_s = os.fspath(Path(path))

    def fast(p, *a, **k):
        s = os.fspath(p)
        return orig(p, *a, **k) if (s == root_s or not s.startswith(root_s)) else fixed       # the root itself (is_file / is_dir) stays real
    os.stat = fast
    try:
        res = bench_dir(old, new, path, threads, repeat, path + " (os.stat replaced by a constant)")
    finally:
        os.stat = orig
    res["evidence_class"] = "SIMULATED stat (constant result); the walk and all Python per-file work are real.  Per-file microseconds = min_s / n_files."
    res["per_file_us"] = {"old": round(1e6 * res["old"]["min_s"] / max(res["n_files"], 1), 2),
                          "new": {k: round(1e6 * v["min_s"] / max(res["n_files"], 1), 2) for k, v in res["new"].items()}}
    return res


def simulate_latency(old, new, root: Path, files: int, ms: float, threads: list) -> dict:
    """SIMULATED: os.stat sleeps `ms` first (a network filesystem's per-file latency), on a tree of `files` files.  Shows the MECHANISM (sequential cost = files x latency; threads
    overlap it), not any pod figure."""
    build_synth(root, files, 8)
    orig = os.stat

    def slow(*a, **k):
        time.sleep(ms / 1000.0)
        return orig(*a, **k)
    os.stat = slow
    try:
        res = bench_dir(old, new, str(root), threads, 1, f"synthetic {files} files, +{ms} ms per stat (SIMULATED)")
    finally:
        os.stat = orig
    res["simulated_latency_ms"] = ms
    res["evidence_class"] = "SIMULATED (a sleep inside os.stat on this host); not a measurement of any network filesystem"
    return res


def check_token(new, token_path: str, threads: list) -> dict:
    """The identity proof on REAL data without re-running the old code: every input a PASS token binds was fingerprinted by the OLD code when the token was minted, and the token records the
    result.  Fingerprint the same inputs with the NEW code (at each thread count) and compare with the token's record -- per input, and for the whole data manifest (what `verify` compares)."""
    tok = json.loads(Path(token_path).read_text(encoding="utf-8"))
    b = tok["binding"]
    rows = []
    for key, rec in sorted(b["data"].items()):
        path = rec.get("local") or rec["path"]
        for n in threads:
            os.environ[new.FP_THREADS_ENV] = str(n)
            t0 = time.perf_counter()
            fp = new.fingerprint(path)
            rows.append({"input": key, "threads": n, "seconds": round(time.perf_counter() - t0, 2), "kind": fp.get("kind"), "n_files": fp.get("n_files"),
                         "equal_to_the_token_record": new.fp_key(fp) == new.fp_key(rec)})
    whole = []
    for n in threads:
        os.environ[new.FP_THREADS_ENV] = str(n)
        t0 = time.perf_counter()
        flags = sorted(k for k in b["data"] if k.startswith("--"))
        data = new.data_manifest(b["argv"], flags, b.get("gate_inputs") or {})
        whole.append({"threads": n, "seconds": round(time.perf_counter() - t0, 2), "data_manifest_equals_the_token": all(new.fp_key(data[k]) == new.fp_key(b["data"][k]) for k in b["data"]) and set(data) == set(b["data"])})
    os.environ.pop(new.FP_THREADS_ENV, None)
    return {"token": Path(token_path).name, "rows": rows, "whole_manifest": whole, "all_equal": all(r["equal_to_the_token_record"] for r in rows) and all(w["data_manifest_equals_the_token"] for w in whole)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", help="a PASS token (JSON): fingerprint every input it binds with the NEW code and compare with what the token recorded (the OLD code's digests) -- the identity proof on real data")
    ap.add_argument("--old", help="the pre-patch gate_token.py (needed by every mode except --token)")
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", action="append", default=[])
    ap.add_argument("--file", action="append", default=[])
    ap.add_argument("--threads", default="1,2,4,8,16,32")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--synth-root")
    ap.add_argument("--synth-files", type=int, default=413184)
    ap.add_argument("--synth-shards", type=int, default=32)
    ap.add_argument("--cpu-only", action="store_true", help="also time every --dir with os.stat replaced by a constant (the interpreter overhead per file)")
    ap.add_argument("--simulate-latency-ms", type=float, default=0.0)
    ap.add_argument("--simulate-files", type=int, default=20000)
    ap.add_argument("--simulate-root")
    ap.add_argument("--high-priority", action="store_true", help="raise this process's priority (Windows HIGH_PRIORITY_CLASS / POSIX nice -5, best effort): for a shared, saturated box")
    ap.add_argument("--store")
    ap.add_argument("--ckpt")
    ap.add_argument("--images")
    ap.add_argument("--calib")
    a = ap.parse_args()
    if a.high_priority:
        try:
            if os.name == "nt":
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
            else:
                os.nice(-5)
        except Exception as e:                                       # noqa: BLE001 -- best effort, recorded
            print("high priority not set:", e)
    if not a.token and not a.old:
        ap.error("--old is required unless --token is the only mode")
    old, new = (load(a.old, "gt_old") if a.old else None), load(a.new, "gt_new")
    threads = [int(x) for x in a.threads.split(",")]
    res = {"env": {"node": platform.node(), "python": sys.version.split()[0], "os": platform.platform(), "cpu_count": os.cpu_count(), "high_priority": bool(a.high_priority),
                   "cpu_load_pct_at_start": cpu_load()},
           "old": a.old, "new": a.new, "repeat": a.repeat, "dirs": [], "files": [], "data_manifest": None, "simulated": None}
    if a.token:
        res["token_check"] = check_token(new, a.token, threads)
        print(json.dumps({"token_check": {"all_equal": res["token_check"]["all_equal"], "seconds_by_threads": {str(w["threads"]): w["seconds"] for w in res["token_check"]["whole_manifest"]}}}), flush=True)
    if a.synth_root:
        build_synth(Path(a.synth_root), a.synth_files, a.synth_shards)
        a.dir.append(a.synth_root)
    for d in a.dir:
        res["dirs"].append(bench_dir(old, new, d, threads, a.repeat, d))
        print(json.dumps({"dir": d, "n_files": res["dirs"][-1]["n_files"], "old_min_s": res["dirs"][-1]["old"]["min_s"],
                          "new_min_s": {k: v["min_s"] for k, v in res["dirs"][-1]["new"].items()}, "identical": all(v["identical"] for v in res["dirs"][-1]["new"].values())}), flush=True)
    if a.cpu_only:
        res["cpu_only"] = []
        for d in a.dir:
            res["cpu_only"].append(cpu_only(old, new, d, threads, a.repeat))
            print(json.dumps({"cpu_only": d, "per_file_us": res["cpu_only"][-1]["per_file_us"]}), flush=True)
    for f in a.file:
        res["files"].append({"path": f, **bench_hash(old, new, f, a.repeat)})
        print(json.dumps({"file": f, "variants_MB_s": {k: v["MB_per_s_min"] for k, v in res["files"][-1]["variants"].items()}}), flush=True)
    if a.store and a.ckpt and a.images and a.calib:
        argv = ["--ckpt", a.ckpt, "--images", a.images, "--aux-drv-labels", a.store, "--calib", a.calib]
        flags = ("--ckpt", "--images", "--aux-drv-labels", "--calib")
        t_old, mo = timeit(lambda: old.data_manifest(argv, flags), a.repeat)
        dm = {"old": t_old, "new": {}}
        for n in threads:
            os.environ[new.FP_THREADS_ENV] = str(n)
            t, mn = timeit(lambda: new.data_manifest(argv, flags), a.repeat)
            t["identical"] = mn == mo and list(mn) == list(mo)
            t["speedup_vs_old_min"] = round(t_old["min_s"] / max(t["min_s"], 1e-9), 2)
            dm["new"][str(n)] = t
        os.environ.pop(new.FP_THREADS_ENV, None)
        res["data_manifest"] = dm
        print(json.dumps({"data_manifest": {"old_min_s": t_old["min_s"], "new_min_s": {k: v["min_s"] for k, v in dm["new"].items()},
                                              "identical": all(v["identical"] for v in dm["new"].values())}}), flush=True)
    if a.simulate_latency_ms > 0:
        res["simulated"] = simulate_latency(old, new, Path(a.simulate_root) if a.simulate_root else Path(f"sim_tree_{a.simulate_files}"), a.simulate_files, a.simulate_latency_ms, threads)
        print(json.dumps({"simulated": {"ms": a.simulate_latency_ms, "files": a.simulate_files, "old_s": res["simulated"]["old"]["min_s"],
                                        "new_s": {k: v["min_s"] for k, v in res["simulated"]["new"].items()}}}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

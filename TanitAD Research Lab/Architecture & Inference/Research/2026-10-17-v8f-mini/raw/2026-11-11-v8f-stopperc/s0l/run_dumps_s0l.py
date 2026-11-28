"""run_dumps_s0l.py -- T1-kin dumps for the STOPPERC S0-L read, tags BPRIME_s0 and L_s0 ONLY (CPU, zero training).  Usage: python run_dumps_s0l.py TAG [TAG ...]

A sibling of the pinned theory/stopbal/read/run_dumps_stopbal.py (not edited, not imported for its code: its constants are read from it so the baseline cannot drift).
The argv is v8f_t1_eval.py's of the DAG-OACC baseline, exactly as run_dumps_stopbal.py builds it, with ONLY these four flags replaced:
    --ckpt         the arm's own checkpoint            (BPRIME_s0: lorapod_eval/bprime/M2b_s0/ckpt_last.pt;  L_s0: env S0L_L_CKPT)
    --tag          <TAG>_m10_causal_heads
    --out-dir      env S0L_DUMPS (default F:/TanitAD-scratch/s0l_read/t1)
    --eval-cache   the arm's OWN eval cache root       (BPRIME_s0: lorapod_eval, the pod-encoded cache, pipeline ra1a-pooled-v1;  L_s0: env S0L_L_CACHE)
Everything else (tracks, eval grid, eval join, replan 10, deadreckon, logged scene, periodic, causal limit, dump-ticks/-heads, cpu) is the baseline's.
An arm is REFUSED (nothing launched) when: the tag is not BPRIME_s0 / L_s0; its checkpoint, its cache (<root>/eval/meta.json and clips) or any baseline input is
missing or unreadable; the checkpoint's sha256 differs from the expected one (BPRIME_s0: EXPECTED_SHA256 below, MEASURED 2026-11-13, equal to the INHERITED
b9aaf33e prefix of LOOP_STATE:544; L_s0: env S0L_L_SHA256, which must be set, and the cache must exist).  Same tree and 2 threads for both arms (OTBAND amendment-1 trap).
Writes its own done file after checking the header, 141 episode dumps, the ticks file and the unchanged checkpoint sha.  Do not run on a GPU: CUDA_VISIBLE_DEVICES=-1.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts"
TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
OUT = os.environ.get("S0L_DUMPS", "F:/TanitAD-scratch/s0l_read/t1")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MIN_FREE_GB = 4.0
MAX_WAIT_S = 3600
ARMS = ("BPRIME_s0", "L_s0")
BPRIME_CKPT = ART + "/lorapod_eval/bprime/M2b_s0/ckpt_last.pt"
BPRIME_CACHE = ART + "/lorapod_eval"
EXPECTED_SHA256 = {"BPRIME_s0": "b9aaf33e458c4da7a4e38becae905fbdd20b35b7dccf8858c973791e7e61c1a7"}
INHERITED_PREFIX = {"BPRIME_s0": "b9aaf33e"}
# the DAG-OACC baseline's inputs, copied from run_dumps_stopbal.py (its :34-36); the test asserts these literals are still in that file
BASE_TREE = ART + "/dagt1_tree_b43c7a9a"
RAW = BASE_TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
BASE_CACHE = ART + "/sfresh_inputs/cache"
INPUTS = {"--tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache": BASE_CACHE,
          "--eval-grid": RAW + "/eval_grid_windows.json", "--eval-join": RAW + "/join_eval141.npz"}
REPLACED = ("--ckpt", "--tag", "--out-dir", "--eval-cache")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M()
    m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def arm_paths(tag):
    """-> (ckpt, cache root, expected sha256 or None).  Reads env only; raises ValueError on an unknown tag."""
    if tag not in ARMS:
        raise ValueError(f"unknown tag {tag}: this driver runs {list(ARMS)} only")
    if tag == "BPRIME_s0":
        return BPRIME_CKPT, BPRIME_CACHE, EXPECTED_SHA256[tag]
    return os.environ.get("S0L_L_CKPT"), os.environ.get("S0L_L_CACHE"), os.environ.get("S0L_L_SHA256")


def build_argv(tag, ckpt, cache, out=None, tree=None):
    """The baseline argv of run_dumps_stopbal.py:run() with --ckpt, --tag, --out-dir, --eval-cache replaced.  Pure: no I/O."""
    out = OUT if out is None else out
    tree = TREE if tree is None else tree
    full = f"{tag}_m10_causal_heads"
    argv = [PY, tree + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ckpt, "--tag", full]
    for k, v in INPUTS.items():
        argv += [k, cache if k == "--eval-cache" else v]
    argv += ["--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic",
             "--limit", "causal", "--dump-ticks", out + "/ticks", "--dump-heads", "--device", "cpu"]
    return argv


def refusal(tag):
    """-> (reason or None, expected sha256 or None).  Reads files only; launches nothing.  Fails closed on anything unreadable."""
    try:
        ck, cache, want = arm_paths(tag)
    except ValueError as e:
        return f"REFUSED: {e}", None
    if tag == "L_s0":
        if not (ck and cache and want):
            return "REFUSED: L_s0 needs env S0L_L_CKPT, S0L_L_CACHE and S0L_L_SHA256 (L is not banked / its cache is not built)", None
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})", want
    if not os.path.isfile(cache + "/eval/meta.json") or not os.path.isdir(cache + "/eval/clips"):
        return f"REFUSED: eval cache missing or unreadable ({cache}/eval/{{meta.json,clips}})", want
    for k, v in INPUTS.items():
        if k != "--eval-cache" and not os.path.isfile(v):
            return f"REFUSED: input {k} missing ({v})", want
    if not os.path.isdir(TREE + "/stack/tanitad") or not os.path.isfile(TREE + "/stack/scripts/v8f_t1_eval.py"):
        return f"REFUSED: tree {TREE} is not exported", want
    try:
        got = sha(ck)
    except OSError as e:
        return f"REFUSED: checkpoint unreadable ({e})", want
    if tag in INHERITED_PREFIX and not got.startswith(INHERITED_PREFIX[tag]):
        return f"REFUSED: checkpoint sha256 {got[:8]} differs from the inherited prefix {INHERITED_PREFIX[tag]}", want
    if got != want:
        return f"REFUSED: checkpoint sha256 {got} differs from the expected {want}", want
    return None, want


def run(tag):
    full = f"{tag}_m10_causal_heads"
    done = f"{OUT}/done_{full}.txt"
    rec = {"tag": tag, "full_tag": full}
    if os.path.exists(done):
        rec["status"] = "skip (done file exists)"
        return rec
    why, want = refusal(tag)
    if why:
        rec["status"] = why
        return rec
    ck, cache, _ = arm_paths(tag)
    rec.update(ckpt=ck, ckpt_sha256=want, eval_cache=cache)
    t0 = time.time()
    while free_gb() < MIN_FREE_GB:
        if time.time() - t0 > MAX_WAIT_S:
            rec["status"] = f"REFUSED: free memory stayed below {MIN_FREE_GB} GB for {MAX_WAIT_S} s"
            return rec
        time.sleep(30)
    rec["free_gb_at_start"] = round(free_gb(), 2)
    rec["tree"] = TREE
    argv = build_argv(tag, ck, cache)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    os.makedirs(OUT + "/ticks", exist_ok=True)
    with open(f"{OUT}/argv_{full}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(argv) + "\n")
    t1 = time.time()
    with open(f"{OUT}/log_{full}.txt", "w", encoding="utf-8") as lf:
        rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    rec.update(rc=rc, wall_s=round(time.time() - t1, 1), free_gb_at_end=round(free_gb(), 2))
    rec["status"] = finalize(tag, rec) if rc == 0 else "FAILED (see log)"
    return rec


def finalize(tag, rec):
    """Done file only when the outputs are complete: header JSON, 141 episode dumps, the ticks file, the unchanged checkpoint."""
    full = f"{tag}_m10_causal_heads"
    hdr, dd, tk = f"{OUT}/t1v8f_{full}.json", f"{OUT}/dump_{full}", f"{OUT}/ticks/ticks_{full}.npz"
    n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
    if not (os.path.exists(hdr) and n == 141 and os.path.exists(tk)):
        return f"FAILED: outputs incomplete (header {os.path.exists(hdr)}, episode dumps {n}, ticks {os.path.exists(tk)})"
    why, want = refusal(tag)
    if why:
        return "FAILED: " + why
    d = {"tag": full, "limit": "causal", "ckpt_sha256": want, "eval_cache": rec.get("eval_cache"), "n_dumps": n, "ticks": tk, "header": hdr,
         "rc": rec.get("rc"), "wall_s": rec.get("wall_s"), "smoke": False, "written_by": "run_dumps_s0l.py finalize"}
    with open(f"{OUT}/done_{full}.txt", "w", encoding="utf-8") as f:
        json.dump(d, f)
    return "ok"


def main():
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    rep = {"tree": TREE, "out": OUT, "min_free_gb": MIN_FREE_GB, "threads": 2, "runs": []}
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, f"DUMPS_S0L_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

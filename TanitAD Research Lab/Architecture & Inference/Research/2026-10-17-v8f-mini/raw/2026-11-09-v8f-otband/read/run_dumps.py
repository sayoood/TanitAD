"""run_dumps.py -- T1-kin dumps for the OTBAND read (CPU, zero training).  Usage: python run_dumps.py TAG [TAG ...]

Each TAG is a key of ARMS.  The argv is the DAG-OACC baseline's (v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv) with only
--ckpt, --tag, --out-dir and --dump-ticks replaced, run from the exported tree of the code commit of record (make_tree.py).
Before each run: the checkpoint's sha256 must equal its bank manifest (or the baseline's done file), and free physical memory must be
at least MIN_FREE_GB (the planner-gate job and other sessions share this machine).  A TAG whose done file exists is skipped.
After PARITY_DAGOACC_s0 it runs parity_check.py.  Writes DUMPS_<stamp>.json next to this file.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "F:/TanitAD-scratch/otband_read/tree"
OUT = "F:/TanitAD-scratch/otband_read/t1"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
BASE_TREE = ART + "/dagt1_tree_b43c7a9a"
RAW = BASE_TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PULL = "C:/Users/Admin/t2v8f_pulls"
MIN_FREE_GB = 4.0
MAX_WAIT_S = 6 * 3600
INPUTS = {"--tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache": ART + "/sfresh_inputs/cache",
          "--eval-grid": RAW + "/eval_grid_windows.json", "--eval-join": RAW + "/join_eval141.npz"}
# tag: (checkpoint, expected sha256 source)
ARMS = {
    "PARITY_DAGOACC_s0": (ART + "/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", ("done", ART + "/v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt")),
    "OB-CTL_s0": (PULL + "/otband_OB-CTL_s0/ckpt_last.pt", ("bank", PULL + "/otband_OB-CTL_s0/BANK_MANIFEST.json")),
    "OB-BAND_s0": (PULL + "/otband_OB-BAND_s0/ckpt_last.pt", ("bank", PULL + "/otband_OB-BAND_s0/BANK_MANIFEST.json")),
    "OB-CTL_s1": (PULL + "/otband_OB-CTL_s1/ckpt_last.pt", ("bank", PULL + "/otband_OB-CTL_s1/BANK_MANIFEST.json")),
    "OB-BAND_s1": (PULL + "/otband_OB-BAND_s1/ckpt_last.pt", ("bank", PULL + "/otband_OB-BAND_s1/BANK_MANIFEST.json")),
    "OB-BAND_rep": (PULL + "/otband_OB-BAND_rep/ckpt_last.pt", ("bank", PULL + "/otband_OB-BAND_rep/BANK_MANIFEST.json")),
    # amendment 1: DAG-OACC s1 at this commit and these threads, so every beside row compares dumps made in one environment
    "PARITY_DAGOACC_s1": (ART + "/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt", ("done", ART + "/v8f-dagoacc-t1/done_DAGOACC_s1_m10_causal_heads.txt")),
    # amendment 2 (beside, post-hoc, decides nothing): the retrain's own init checkpoint, sha256 from OB-CTL_s0's summary.json levers.init_ckpt
    "INIT_M2b_s0": (ART + "/sfresh_inputs/inputs/M2b_s0/ckpt_last.pt", ("sha", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8")),
    # control P0b (README_READ.md, amendment 1): the BASELINE's own tool tree, same checkpoint, same threads as P0
    "PARITY_BASETREE_DAGOACC_s0": (ART + "/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", ("done", ART + "/v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt")),
}
TREE_OF = {"PARITY_BASETREE_DAGOACC_s0": BASE_TREE}


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


def expected_sha(src):
    kind, p = src
    if kind == "sha":
        return p
    d = json.load(open(p, encoding="utf-8"))
    return d["ckpt_sha256"] if kind == "done" else d["files"]["ckpt_last.pt"]["sha256"]


def run(tag):
    full = f"{tag}_m10_causal_heads"
    done = f"{OUT}/done_{full}.txt"
    rec = {"tag": tag, "full_tag": full}
    if os.path.exists(done):
        rec["status"] = "skip (done file exists)"
        return rec
    ck, src = ARMS[tag]
    want, got = expected_sha(src), sha(ck)
    rec.update(ckpt=ck, ckpt_sha256=got, expected_sha256=want)
    if got != want:
        rec["status"] = "REFUSED: checkpoint sha256 differs from its record"
        return rec
    t0 = time.time()
    while free_gb() < MIN_FREE_GB:
        if time.time() - t0 > MAX_WAIT_S:
            rec["status"] = f"REFUSED: free memory stayed below {MIN_FREE_GB} GB for {MAX_WAIT_S} s"
            return rec
        time.sleep(30)
    rec["waited_s"] = round(time.time() - t0, 1)
    rec["free_gb_at_start"] = round(free_gb(), 2)
    rec["tree"] = TREE_OF.get(tag, TREE)
    argv = [PY, rec["tree"] + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ck, "--tag", full]
    for k, v in INPUTS.items():
        argv += [k, v]
    argv += ["--out-dir", OUT, "--replan-steps", "10", "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic",
             "--limit", "causal", "--dump-ticks", OUT + "/ticks", "--dump-heads", "--device", "cpu"]
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
    """v8f_t1_eval.py writes t1v8f_<tag>.json, not a done file (the baseline's done file came from its wrapper chain).  The done file is
    written HERE, and only when the run's outputs are complete: the header JSON, 141 episode dumps and the ticks file."""
    full = f"{tag}_m10_causal_heads"
    hdr, dd, tk = f"{OUT}/t1v8f_{full}.json", f"{OUT}/dump_{full}", f"{OUT}/ticks/ticks_{full}.npz"
    n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
    if not (os.path.exists(hdr) and n == 141 and os.path.exists(tk)):
        return f"FAILED: outputs incomplete (header {os.path.exists(hdr)}, episode dumps {n}, ticks {os.path.exists(tk)})"
    ck, src = ARMS[tag]
    want, got = expected_sha(src), sha(ck)
    if got != want:
        return "FAILED: checkpoint sha256 changed during the run"
    d = {"tag": full, "limit": "causal", "ckpt_sha256": got, "n_dumps": n, "ticks": tk, "header": hdr, "rc": rec.get("rc"),
         "wall_s": rec.get("wall_s"), "smoke": False, "written_by": "run_dumps.py finalize"}
    with open(f"{OUT}/done_{full}.txt", "w", encoding="utf-8") as f:
        json.dump(d, f)
    return "ok"


def main():
    if sys.argv[1:2] == ["--finalize"]:
        # for runs that ended rc 0 under the first version of this driver (which looked for a done file the eval never writes)
        for t in sys.argv[2:]:
            assert t in ARMS, t
            full = f"{t}_m10_causal_heads"
            if os.path.exists(f"{OUT}/done_{full}.txt"):
                print(t, "done file exists")
                continue
            print(t, finalize(t, {"rc": 0, "wall_s": None}))
        if "PARITY_DAGOACC_s0" in sys.argv[2:]:
            pc = subprocess.run([PY, "-I", HERE + "/parity_check.py"], capture_output=True, encoding="utf-8")
            print(pc.stdout[-1500:], pc.stderr[-800:], flush=True)
        return
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    for k, v in INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"tree": TREE, "out": OUT, "inputs_sha256": {k: (sha(v) if os.path.isfile(v) else "directory") for k, v in INPUTS.items()},
           "min_free_gb": MIN_FREE_GB, "threads": 2, "runs": []}
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
        if t == "PARITY_DAGOACC_s0" and r["status"] in ("ok", "skip (done file exists)"):
            pc = subprocess.run([PY, "-I", HERE + "/parity_check.py"], capture_output=True, encoding="utf-8")
            print(pc.stdout[-1500:], pc.stderr[-800:], flush=True)
            rep["parity_rc"] = pc.returncode
    json.dump(rep, open(os.path.join(HERE, f"DUMPS_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""run_dumps_stopbal.py -- T1-kin dumps for the STOPBAL read (CPU, zero training).  Usage: python run_dumps_stopbal.py TAG [TAG ...]   (TAG in ARMS)

Adapted from theory/otband/read/run_dumps.py: the argv is v8f_t1_eval.py's of the DAG-OACC baseline with only --ckpt, --tag, --out-dir and --dump-ticks
replaced, run from the exported tree of the code of record (make_tree.py, commit 5ecdd7e6).  Before an arm runs it is REFUSED (nothing launched) when
  * its ckpt_last.pt is missing,
  * its BANK_MANIFEST.json is missing or its ckpt sha256 differs from the manifest,
  * its selfcheck.json is missing, or its levers (goal_pos_weight_stop / lon_class_weight_stop) or G-DVB do not match the card's values for the arm
    (control 1: BAL-NULL 1.0 / 1.0; BAL-G 4.5616 / 1.0; BAL-L 1.0 / 4.6991), or the manifest names another commit than the code of record.
A TAG whose done file exists is skipped.  The driver writes its own done file (v8f_t1_eval.py writes none) after checking the header, 141 episode dumps, the
ticks file and the checkpoint sha256.  Free physical memory must be at least MIN_FREE_GB (other sessions share this machine).  Dumps of every arm must be made
in ONE environment (README_READ.md, the OTBAND amendment-1 trap).  Writes DUMPS_<stamp>.json beside this file.  Do not run on a GPU: CUDA_VISIBLE_DEVICES=-1.
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stopbal_read as SR  # noqa: E402  (numpy only: the card's control-1 check)

TREE = os.environ.get("STOPBAL_TREE", "F:/TanitAD-scratch/stopbal_read/tree")
OUT = os.environ.get("STOPBAL_DUMPS", "F:/TanitAD-scratch/stopbal_read/t1")
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
ART = "E:/Projects/TanitAD-artifacts"
BASE_TREE = ART + "/dagt1_tree_b43c7a9a"
RAW = BASE_TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PULL = os.environ.get("STOPBAL_PULL", "C:/Users/Admin/t2v8f_pulls")
COMMIT = SR.CODE_OF_RECORD
MIN_FREE_GB = 4.0
MAX_WAIT_S = 6 * 3600
INPUTS = {"--tracks": ART + "/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache": ART + "/sfresh_inputs/cache",
          "--eval-grid": RAW + "/eval_grid_windows.json", "--eval-join": RAW + "/join_eval141.npz"}
ARMS = ("BAL-NULL_s0", "BAL-NULL_s1", "BAL-NULL_rep", "BAL-G_s0", "BAL-G_s1", "BAL-L_s0", "BAL-L_s1")


def bank_dir(tag):
    return f"{PULL}/stopbal_{tag}"


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


def refusal(tag):
    """-> (reason or None, expected ckpt sha256 or None).  Reads files only; launches nothing."""
    if tag not in ARMS:
        return f"unknown tag {tag}", None
    b = bank_dir(tag)
    ck = f"{b}/ckpt_last.pt"
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})", None
    man_p = f"{b}/BANK_MANIFEST.json"
    if not os.path.isfile(man_p):
        return f"REFUSED: BANK_MANIFEST.json missing ({man_p})", None
    man = json.load(open(man_p, encoding="utf-8"))
    want = man.get("files", {}).get("ckpt_last.pt", {}).get("sha256")
    if not want:
        return "REFUSED: BANK_MANIFEST.json holds no ckpt_last.pt sha256", None
    if man.get("commit") not in (None, COMMIT):
        return f"REFUSED: the bank was trained at commit {man.get('commit')}, the code of record is {COMMIT}", want
    if man.get("tag") not in (None, tag):
        return f"REFUSED: the bank manifest names tag {man.get('tag')}, not {tag}", want
    sc_p = f"{b}/selfcheck.json"
    if not os.path.isfile(sc_p):
        return f"REFUSED: selfcheck.json missing ({sc_p}): control 1 cannot be read", want
    bad = SR.check_levers(SR.kind_of(tag), json.load(open(sc_p, encoding="utf-8")))
    if bad:
        return "REFUSED: control 1 fails: " + " | ".join(bad), want
    if sha(ck) != want:
        return "REFUSED: checkpoint sha256 differs from its bank manifest", want
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
    ck = f"{bank_dir(tag)}/ckpt_last.pt"
    rec.update(ckpt=ck, ckpt_sha256=want)
    t0 = time.time()
    while free_gb() < MIN_FREE_GB:
        if time.time() - t0 > MAX_WAIT_S:
            rec["status"] = f"REFUSED: free memory stayed below {MIN_FREE_GB} GB for {MAX_WAIT_S} s"
            return rec
        time.sleep(30)
    rec["waited_s"] = round(time.time() - t0, 1)
    rec["free_gb_at_start"] = round(free_gb(), 2)
    rec["tree"] = TREE
    argv = [PY, TREE + "/stack/scripts/v8f_t1_eval.py", "run", "--ckpt", ck, "--tag", full]
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
    """The done file is written HERE, and only when the run's outputs are complete: the header JSON, 141 episode dumps, the ticks file, the unchanged checkpoint."""
    full = f"{tag}_m10_causal_heads"
    hdr, dd, tk = f"{OUT}/t1v8f_{full}.json", f"{OUT}/dump_{full}", f"{OUT}/ticks/ticks_{full}.npz"
    n = len([f for f in os.listdir(dd) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(dd) else 0
    if not (os.path.exists(hdr) and n == 141 and os.path.exists(tk)):
        return f"FAILED: outputs incomplete (header {os.path.exists(hdr)}, episode dumps {n}, ticks {os.path.exists(tk)})"
    why, want = refusal(tag)
    if why:
        return "FAILED: " + why
    d = {"tag": full, "limit": "causal", "ckpt_sha256": want, "n_dumps": n, "ticks": tk, "header": hdr, "rc": rec.get("rc"),
         "wall_s": rec.get("wall_s"), "smoke": False, "written_by": "run_dumps_stopbal.py finalize"}
    with open(f"{OUT}/done_{full}.txt", "w", encoding="utf-8") as f:
        json.dump(d, f)
    return "ok"


def main():
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    for k, v in INPUTS.items():
        assert os.path.exists(v), (k, v)
    assert os.path.exists(TREE + "/stack/scripts/v8f_t1_eval.py"), f"{TREE} is not exported: run make_tree.py first"
    rep = {"tree": TREE, "out": OUT, "commit": COMMIT, "inputs_sha256": {k: (sha(v) if os.path.isfile(v) else "directory") for k, v in INPUTS.items()},
           "min_free_gb": MIN_FREE_GB, "threads": 2, "runs": []}
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
    json.dump(rep, open(os.path.join(HERE, f"DUMPS_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

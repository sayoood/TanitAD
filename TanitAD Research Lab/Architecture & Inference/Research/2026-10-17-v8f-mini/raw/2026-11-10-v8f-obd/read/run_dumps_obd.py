"""run_dumps_obd.py -- T1-kin dumps for the OBD read (CPU, zero training).  Usage: python run_dumps_obd.py TAG [TAG ...]   (TAG in ARMS)

A thin wrapper over theory/otband/read/run_dumps.py (NOT edited): its tag table is extended with the OBD tags, its module paths are redirected (dump folder, tree, summary
file location), and its own `run()` does the work (argv = the DAG-OACC baseline's with only --ckpt, --tag, --out-dir, --dump-ticks replaced; CPU, 2 threads; the checkpoint sha256
is checked against the bank manifest before AND after the dump; waits for 4 GB of free memory; writes the done file itself).  Added here, before `run()` is reached:
  an arm is REFUSED (nothing launched, no exception) when its ckpt_last.pt or BANK_MANIFEST.json is missing, the manifest names another tag or another commit than the code of
  record cdbff971fa6dad4d20b937fe8abda6cb9f622adb, or its checkpoint sha256 differs from the manifest.  The export tree's pinned files are re-hashed against
  theory/otband/read/TREE_PINS.json at start (the tree must be the OTBAND export: stack/tanitad and taniteval at the code of record are byte-identical to it, README_READ.md).
Banks: C:/Users/Admin/t2v8f_pulls/obd_<tag>/ckpt_last.pt (+ BANK_MANIFEST.json {tag, commit, files{ckpt_last.pt{sha256}}}); SOURCE: theory/obd/box2/stages/obd_run.sh `BD="$PULLS/obd_$TAG"`
and its manifest writer.  UNVERIFIED on a real bank: none exists yet.
Dumps go to env OBD_DUMPS (default F:/TanitAD-scratch/otband_read/t1: the SAME folder and environment as the DAG-OACC reference dumps PARITY_DAGOACC_s0/s1 that the read needs).
All dumps of one read must be made in ONE environment (OTBAND trap: another thread count drifts the rolled track at the mm level).  Writes DUMPS_OBD_<stamp>.json beside this file.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.dirname(os.path.dirname(HERE))
OTB = os.path.join(THEORY, "otband", "read")
sys.path.insert(0, OTB)
import run_dumps as RD  # noqa: E402  (module level is definitions only; its main() is not called)

COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"
PULL = os.environ.get("OBD_PULLS", "C:/Users/Admin/t2v8f_pulls")
OUT = os.environ.get("OBD_DUMPS", "F:/TanitAD-scratch/otband_read/t1")
TREE = os.environ.get("OBD_TREE", "F:/TanitAD-scratch/otband_read/tree")
# the planned order of the card (chain order): NULL_s0, NULL_rep, CTL_s0, BAND_s0, CTL_s1, BAND_s1, BAND_rep; NULL_s1 is the conditional arm
OBD_TAGS = ("OBD-NULL_s0", "OBD-NULL_rep", "OBD-CTL_s0", "OBD-BAND_s0", "OBD-CTL_s1", "OBD-BAND_s1", "OBD-BAND_rep", "OBD-NULL_s1")
ARMS = {t: (f"{PULL}/obd_{t}/ckpt_last.pt", ("bank", f"{PULL}/obd_{t}/BANK_MANIFEST.json")) for t in OBD_TAGS}
for _t, _v in ARMS.items():
    RD.ARMS[_t] = _v
RD.OUT, RD.TREE, RD.PULL = OUT, TREE, PULL


def refusal(tag):
    """-> reason or None.  Files only; launches nothing."""
    if tag not in ARMS:
        return f"unknown tag {tag}"
    ck, (_, man_p) = ARMS[tag]
    if not os.path.isfile(ck):
        return f"REFUSED: checkpoint missing ({ck})"
    if not os.path.isfile(man_p):
        return f"REFUSED: BANK_MANIFEST.json missing ({man_p})"
    man = json.load(open(man_p, encoding="utf-8"))
    want = ((man.get("files") or {}).get("ckpt_last.pt") or {}).get("sha256")
    if not want:
        return "REFUSED: BANK_MANIFEST.json holds no ckpt_last.pt sha256"
    if man.get("tag") not in (None, tag):
        return f"REFUSED: the bank manifest names tag {man.get('tag')}, not {tag}"
    if man.get("commit") not in (None, COMMIT):
        return f"REFUSED: the bank was trained at commit {man.get('commit')}, the code of record is {COMMIT}"
    if RD.sha(ck) != want:
        return "REFUSED: checkpoint sha256 differs from its bank manifest"
    return None


def tree_problem():
    pins = json.load(open(os.path.join(OTB, "TREE_PINS.json"), encoding="utf-8"))["sha256"]
    for rel, want in pins.items():
        p = f"{TREE}/{rel}"
        if not os.path.isfile(p):
            return f"tree file missing: {p}"
        if RD.sha(p) != want:
            return f"tree file differs from TREE_PINS.json: {rel}"
    return None


def run(tag):
    why = refusal(tag)
    if why:
        return {"tag": tag, "status": why}
    return RD.run(tag)           # the OTBAND driver's own run(): memory wait, argv, sha before and after, done file


def main():
    tags = sys.argv[1:]
    assert tags and all(t in ARMS for t in tags), f"tags must be among {list(ARMS)}"
    tp = tree_problem()
    if tp:
        raise SystemExit("REFUSED: " + tp)
    for k, v in RD.INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"tree": TREE, "out": OUT, "commit": COMMIT, "inputs_sha256": {k: (RD.sha(v) if os.path.isfile(v) else "directory") for k, v in RD.INPUTS.items()},
           "min_free_gb": RD.MIN_FREE_GB, "threads": 2, "runs": []}
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
    json.dump(rep, open(os.path.join(HERE, f"DUMPS_OBD_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

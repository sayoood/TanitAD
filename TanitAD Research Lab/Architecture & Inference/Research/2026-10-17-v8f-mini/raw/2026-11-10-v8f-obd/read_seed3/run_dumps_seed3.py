"""run_dumps_seed3.py -- T1-kin dumps for the four seed-3 arms of AMEND_OBD_SEED3.md (CPU, zero training).
Usage: python run_dumps_seed3.py [TAG ...]      (TAG in OBD-CTL_s10 OBD-BAND_s10 OBD-CTL_s11 OBD-BAND_s11; no TAG = all four, in the amendment's order)

Reuses ../read/run_dumps_obd.py (imported, NOT edited) and through it the OTBAND driver's own run(): same tree (F:/TanitAD-scratch/otband_read/tree, pinned files re-hashed
against TREE_PINS.json), same dump folder (env OBD_DUMPS, default F:/TanitAD-scratch/otband_read/t1), same argv, 2 CPU threads, memory wait, checkpoint sha256 before and after.
Added here: the tag table is extended with the four banks C:/Users/Admin/t2v8f_pulls/obd_<tag>/ (BANK_MANIFEST.json {tag, commit, files{ckpt_last.pt{sha256}}});
  * a finished arm (done_<tag>_m10_causal_heads.txt exists) is SKIPPED first;
  * an arm whose ckpt / manifest is missing, whose manifest names another tag or commit, or whose checkpoint sha256 differs is REFUSED (run_dumps_obd.refusal, unchanged);
  * an arm whose argv.txt differs from its base arm's in more than --seed and --out (or carries another --seed) is REFUSED (seed3_common.argv_problem);
  * nothing is launched for a refused arm.
Writes DUMPS_SEED3_<stamp>.json beside this file.  UNVERIFIED on a real bank: none exists yet.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
READ = os.path.join(os.path.dirname(HERE), "read")
sys.path.insert(0, HERE)
sys.path.insert(0, READ)
import run_dumps_obd as D  # noqa: E402  (module level is definitions only)
import seed3_common as C   # noqa: E402

RD = D.RD
COMMIT, PULL, OUT, TREE = D.COMMIT, D.PULL, D.OUT, D.TREE
NEW_TAGS = C.NEW_TAGS


def _register():
    for t in NEW_TAGS:
        D.ARMS[t] = (f"{PULL}/obd_{t}/ckpt_last.pt", ("bank", f"{PULL}/obd_{t}/BANK_MANIFEST.json"))
        RD.ARMS[t] = D.ARMS[t]


_register()


def refusal(tag):
    """-> reason or None.  Files only; launches nothing."""
    if tag not in C.NEW:
        return f"unknown tag {tag}"
    why = D.refusal(tag)
    if why:
        return why
    why = C.argv_problem(PULL, tag)
    return None if why is None else "REFUSED: " + why


def run(tag):
    full = f"{tag}_m10_causal_heads"
    if os.path.exists(f"{RD.OUT}/done_{full}.txt"):
        return {"tag": tag, "full_tag": full, "status": "skip (done file exists)"}
    why = refusal(tag)
    if why:
        return {"tag": tag, "status": why}
    return RD.run(tag)


def main():
    tags = sys.argv[1:] or list(NEW_TAGS)
    assert all(t in C.NEW for t in tags), f"tags must be among {list(NEW_TAGS)}"
    tp = D.tree_problem()
    if tp:
        raise SystemExit("REFUSED: " + tp)
    for k, v in RD.INPUTS.items():
        assert os.path.exists(v), (k, v)
    rep = {"amendment_sha256": C.AMEND_SHA256, "tree": TREE, "out": OUT, "commit": COMMIT, "threads": 2, "min_free_gb": RD.MIN_FREE_GB,
           "inputs_sha256": {k: (RD.sha(v) if os.path.isfile(v) else "directory") for k, v in RD.INPUTS.items()}, "runs": []}
    for t in tags:
        r = run(t)
        print(time.strftime("%H:%M:%S"), json.dumps(r), flush=True)
        rep["runs"].append(r)
    json.dump(rep, open(os.path.join(HERE, f"DUMPS_SEED3_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-token post-DONE watcher (HANDOFF_2).  Waits for P2HO_DONE or P2HO_FAILED in the package, then (DONE only):
  1. featcheck.json        p2ho_featcheck.py  (logits == score_head(saved features), table/proposals/features row for row)
  2. recompute_check_333.json   the 333 tokens of the FIRST (partial) run were recomputed because their log's unit key changed; compare them with the
                                banked first-run table (p2heldout_work/final_333/): logits, proposals, per-cell PDMS/sub-scores, pick
  3. CODE_SHA256.json      p2ho_hashcode.py
  4. paths.json            make_paths.py
and writes FINALIZED.txt (the first-run FINALIZED.txt was moved to prev333/).  Reads/writes small files in the package only; the interpreter is the driverl-eval venv.
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe code/p2ho_post.py
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
WORK = "C:/Users/Admin/p2heldout_work"


def compare_333():
    old, new = np.load(f"{WORK}/final_333/table_p2_heldout.npz"), np.load(f"{WORK}/final/table_p2_heldout.npz")
    ot, nt = [str(t) for t in old["token"]], [str(t) for t in new["token"]]
    pos = {t: i for i, t in enumerate(nt)}
    missing = [t for t in ot if t not in pos]
    oi = np.array([i for i, t in enumerate(ot) if t in pos])
    ni = np.array([pos[ot[i]] for i in oi])
    r = {"n_old": len(ot), "n_new": len(nt), "n_old_tokens_missing_in_new": len(missing), "n_compared": int(len(oi))}
    if len(oi):
        d = lambda k: float(np.abs(old[k][oi].astype(np.float64) - new[k][ni].astype(np.float64)).max())  # noqa: E731
        r.update(logits_max_abs=d("logits"), proposals_max_abs=d("proposals"), pdms_max_abs=d("pdms"), sub_max_abs=d("sub"),
                 pick_equal=int((old["pick"][oi] == new["pick"][ni]).sum()), pdms_at_pick_max_abs=d("pdms_at_pick"),
                 log_equal=bool((old["log"][oi] == new["log"][ni]).all()))
        r["PASS"] = bool(not missing and r["pick_equal"] == len(oi) and r["pdms_max_abs"] <= 1e-4 and r["logits_max_abs"] <= 1e-4 and r["proposals_max_abs"] <= 1e-3)
    else:
        r["PASS"] = False
    return r


def main():
    py = sys.executable
    deadline = time.time() + 24 * 3600
    marker = None
    while time.time() < deadline:
        if os.path.exists(f"{ROOT}/P2HO_DONE"):
            marker = "P2HO_DONE"
            break
        if os.path.exists(f"{ROOT}/P2HO_FAILED"):
            marker = "P2HO_FAILED"
            break
        time.sleep(30)
    if marker is None:
        return 1
    time.sleep(20)
    notes = []
    if marker == "P2HO_DONE":
        rc = subprocess.call([py, f"{ROOT}/code/p2ho_featcheck.py", "--out", f"{ROOT}/featcheck.json"])
        notes.append(f"featcheck rc={rc}")
        try:
            r = compare_333()
            json.dump(r, open(f"{ROOT}/recompute_check_333.json", "w"), indent=1)
            notes.append(f"recompute_check_333 PASS={r['PASS']}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"recompute_check_333 EXCEPTION {e!r}")
        subprocess.call([py, f"{ROOT}/code/p2ho_hashcode.py"])
    subprocess.call([py, f"{ROOT}/code/make_paths.py"])
    open(f"{ROOT}/FINALIZED.txt", "w").write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} marker={marker}; {'; '.join(notes)}; paths.json rewritten\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

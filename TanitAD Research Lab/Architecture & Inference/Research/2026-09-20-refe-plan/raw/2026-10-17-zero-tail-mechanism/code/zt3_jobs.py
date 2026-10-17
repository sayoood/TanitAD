"""zt3_jobs.py -- build the geometry job list for PREREG_ZT3 (driverl-eval venv). Reads the banked tables only.

    python zt3_jobs.py --inroot <dir with c3/ p2/ navtest_full_tokens.json> --out ../data/jobs.npz

Gates G0 (counts) and the pick-reproduction identity run here; the job file carries, per token, up to 5
trajectories (A_P, A_S, B_P, B_S, H) as [8,3] ego-frame poses plus the table's DAC for each (H: -1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zero_tail as zt                       # the landed classifier, byte-identical (test_zt3 checks the sha256)
import zt3_lib as L

TRAJ = ("A_P", "A_S", "B_P", "B_S", "H")
HUMAN = "D:/Projects/TanitAD/data/refe_navtest/seams/refe_p4_human_navtest_all.npz"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm_view(T, rule):
    """per-row: pick, S, target flag, cause, sel."""
    pdms, sub, lg, pick = T["pdms"], T["sub"], T["logits"], T["pick"]
    score = L.selection_score(lg.astype(np.float64), rule)
    rep = (score.argmax(axis=1) == pick)
    nc, dac = sub[np.arange(len(pick)), pick, 0], sub[np.arange(len(pick)), pick, 1]
    zero = pdms[np.arange(len(pick)), pick] == 0
    cause = zt.cause_labels(nc, dac)
    sel = zt.sel_any(pdms)
    S = np.array([L.best_nonzero(score[i], pdms[i]) for i in range(len(pick))])
    target = zero & (cause == "DAC_only") & sel
    return {"score": score, "pick_reproduced": int(rep.sum()), "n": len(pick), "zero": zero, "cause": cause,
            "sel": sel, "S": S, "target": target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.inroot)
    files = {"c3_table": R / "c3/table.npz", "c3_tokens": R / "c3/tokens.json", "p2_table": R / "p2/p2_own_table/table.npz",
             "navtest_tokens": R / "navtest_full_tokens.json", "human_seam": Path(HUMAN)}
    A = np.load(files["c3_table"])
    B = np.load(files["p2_table"])
    c3 = json.load(open(files["c3_tokens"], encoding="utf-8"))
    nt = json.load(open(files["navtest_tokens"], encoding="utf-8"))
    H = np.load(files["human_seam"])
    va, vb = arm_view(A, "navsim_v1_lane"), arm_view(B, "navsim_v1")
    gates = {"G0_A_zero_529": int(va["zero"].sum()) == 529, "G0_B_zero_509": int(vb["zero"].sum()) == 509,
             "G0_B_sel_445": int((vb["zero"] & vb["sel"]).sum()) == 445,
             "pick_reproduced_A": f"{va['pick_reproduced']}/{va['n']}", "pick_reproduced_B": f"{vb['pick_reproduced']}/{vb['n']}"}
    print(json.dumps(gates))
    if not (gates["G0_A_zero_529"] and gates["G0_B_zero_509"] and gates["G0_B_sel_445"]
            and va["pick_reproduced"] == va["n"] and vb["pick_reproduced"] == vb["n"]):
        raise SystemExit("GATE FAIL (G0 or pick reproduction) -- stop, per PREREG section 1")
    ta = [str(t) for t in A["token"]]
    tb = [str(t) for t in B["token"]]
    ia, ib = {t: i for i, t in enumerate(ta)}, {t: i for i, t in enumerate(tb)}
    hi = {str(t): i for i, t in enumerate(H["token"])}
    toks = ta + [t for t in tb if t not in ia]
    n = len(toks)
    traj = np.full((n, len(TRAJ), 8, 3), np.nan, np.float32)
    have = np.zeros((n, len(TRAJ)), bool)
    exp_dac = np.full((n, len(TRAJ)), -1.0)
    for r, t in enumerate(toks):
        if t in ia:
            i = ia[t]
            for j, k in ((0, int(A["pick"][i])), (1, int(va["S"][i]))):
                if k >= 0:
                    traj[r, j], have[r, j], exp_dac[r, j] = A["proposals"][i, k], True, A["sub"][i, k, 1]
        if t in ib:
            i = ib[t]
            for j, k in ((2, int(B["pick"][i])), (3, int(vb["S"][i]))):
                if k >= 0:
                    traj[r, j], have[r, j], exp_dac[r, j] = B["proposals"][i, k], True, B["sub"][i, k, 1]
        traj[r, 4], have[r, 4] = H["poses"][hi[t]], True
    logs = np.array([nt["token_log"][t] for t in toks])
    meta = {"n_tokens": n, "n_c3": len(ta), "n_p2_only": n - len(ta), "n_target_A": int(va["target"].sum()),
            "n_target_B": int(vb["target"].sum()), "gates": gates,
            "inputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in files.items()}}
    np.savez(a.out, token=np.array(toks, dtype="U16"), log=logs, traj=traj, have=have, exp_dac=exp_dac,
             traj_names=np.array(TRAJ), meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

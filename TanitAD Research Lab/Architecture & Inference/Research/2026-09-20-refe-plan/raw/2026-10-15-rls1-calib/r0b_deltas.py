#!/usr/bin/env python3
"""Task 1 of the O2(i) package: R0b's RL-val numbers against C0 (kill_checks.jsonl stored only booleans for g5).

Arms (scores_<tag>_{A,B}.npz of the banked HF repos):  C0 = C0's BCE-only arm A (the registered K3 control, rls1_checks.RlValChecker reads
scores_<tag>_A.npz of the control dir);  R0b = R0b's EPO arm B (--epo-lam 0.25);  R0b-A = R0b's own same-pass BCE-only arm A (outcome O5's check);
R0 = R0's arm B at the same tag, for the side-by-side.  RL-val = SELECTION tier (3,000 sets, 75 logs), never an endpoint; one training seed per
arm; the CI answers "another draw of LOGS" only (H-ESTIM-SEED-1).  Self-check: the PDMS point delta must equal kill_checks.jsonl's
pdms_delta_x100 (0.5324359027777779 at u250, 0.8986615277777777 at u500).

    python r0b_deltas.py --out r0b_deltas.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as CC  # noqa: E402

KILL_CHECKS = {"update250": 0.5324359027777779, "update500": 0.8986615277777777}   # raw/2026-10-15-rls1b-r0b-kill/kill_checks.jsonl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=CC.DL_DEFAULT)
    ap.add_argument("--x", default=CC.X_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rels = ["data/data/rlfan_pdm_val.tar.gz", "data/data/rlfan_terms_val.tar.gz"]
    for t in ("update250", "update500"):
        rels += [f"c0/final/scores_{t}_A.npz", f"r0b/final/scores_{t}_A.npz", f"r0b/final/scores_{t}_B.npz", f"r0/final/scores_{t}_B.npz"]
    res = {"tool": "r0b_deltas/1", "tier": "RL-val (selection tier; never an endpoint)", "rule": "navsim_v1_lane",
           "estimator": "paired log-cluster bootstrap, proxy_eval.boot, 10,000, seed 20260927 -- another draw of LOGS only",
           "units": "points = x100 of the per-set sub-score at the pick", "inputs_sha256": CC.verify(a.dl, rels), "repos": CC.REPOS}
    keys, tg, ck, s, st = CC.load_val(a.x)
    res["rl_val_join"] = st
    res["n_sets"], res["n_logs"] = len(keys), len({k[0] for k in keys})
    out = {}
    for t in ("update250", "update500"):
        P = {nm: CC.picks_from_logits(CC.load_scores_aligned(os.path.join(a.dl, rel), keys, ck))
             for nm, rel in (("C0", f"c0/final/scores_{t}_A.npz"), ("R0b", f"r0b/final/scores_{t}_B.npz"),
                             ("R0bA", f"r0b/final/scores_{t}_A.npz"), ("R0", f"r0/final/scores_{t}_B.npz"))}
        row = {"R0b_minus_C0": CC.delta_table(s, P["R0b"], P["C0"], keys),
               "R0bA_minus_C0__O5_check": CC.delta_table(s, P["R0bA"], P["C0"], keys),
               "R0_minus_C0__side_by_side": CC.delta_table(s, P["R0"], P["C0"], keys),
               "R0b_minus_R0bA__same_pass": CC.delta_table(s, P["R0b"], P["R0bA"], keys)}
        row["g5_R0b_vs_C0"] = CC.g5(row["R0b_minus_C0"])
        o5 = row["R0bA_minus_C0__O5_check"]
        row["O5_confound"] = {"pdms_abs_gt_0.3": bool(abs(o5["PDMS"]["mean"]) > 0.3),
                              "any_separated_subscore": [q for q in CC.SUBS if o5[q]["separated"]],
                              "rule": "PREREG 10.5 O5: R0b's arm A differs from C0 by PDMS > 0.3 or any separated sub-score"}
        row["selfcheck_kill_checks_pdms_delta_x100"] = {"ours": row["R0b_minus_C0"]["PDMS"]["mean"], "kill_checks": KILL_CHECKS[t],
                                                       "abs_diff": abs(row["R0b_minus_C0"]["PDMS"]["mean"] - KILL_CHECKS[t])}
        if row["selfcheck_kill_checks_pdms_delta_x100"]["abs_diff"] > 1e-9:
            raise SystemExit(f"selfcheck failed at {t}: {row['selfcheck_kill_checks_pdms_delta_x100']}")
        out[t] = row
        print(t, "R0b-C0", {q: round(row["R0b_minus_C0"][q]["mean"], 3) for q in CC.SUBS}, flush=True)
    res["reads"] = out
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZR0B_DELTAS_OK", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

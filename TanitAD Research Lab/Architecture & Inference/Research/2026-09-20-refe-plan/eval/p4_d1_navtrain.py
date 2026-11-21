#!/usr/bin/env python3
"""P4 diagnostic D1 on HELD-OUT NAVTRAIN (EXPLORATORY; the prerequisite PI decision (b) names): are the teacher-target
DAC failures REVIEW_7 counted (4.97 % [1.30, 10.31], teacher_lane_check.py, NAVSIM's SIMULATED footprint) failures of
the teacher's own path, or of NAVSIM's tracker -- the same split D1 measured on navtest?

Inputs: REVIEW_7's per-sample verdicts (`raw/2026-10-04-lane-discipline/teacher_check_heldout.jsonl`, rank 0) and the
BANKED 10 Hz rank-0 target for the same (log, token) from the local bank `E:/Projects/TanitAD/data/refe_navtrain10`
(C3 showed this teacher reproduces such targets to 8e-7 m). Per sample, refe/navsim_dac.py on the trainval DB scenario:
raw20 (the target's own 20 poses), interp8, sim8 (NAVSIM's tracker). Control: sim8 must agree with REVIEW_7's recorded
`teacher.dac_viol` on every sample (two runs of one instrument -- a determinism check, reported as such).

    python eval/run_p4.py --script p4_d1_navtrain.py --log <log> -- --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
HELD = os.path.join(PKG, "raw", "2026-10-04-lane-discipline", "teacher_check_heldout.jsonl")
BANK = "E:/Projects/TanitAD/data/refe_navtrain10"
TRAINVAL = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import navsim_dac as ND
    import navtrain_scenarios as NS
    held = {}
    for ln in open(HELD, encoding="utf-8"):
        r = json.loads(ln)
        if int(r.get("rank", 0)) == 0 and int(r.get("step", 0)) == 0 and "teacher" in r:
            held[(r["log"], r["token"])] = r
    bank = {}
    for fp in glob.glob(f"{BANK}/**/targets_rank0.jsonl", recursive=True):
        for ln in open(fp, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            k = (r.get("log_name"), r.get("token"))
            if k in held and int(r.get("sim_hz", 0)) == 10 and int(r.get("step", 0)) == 0:
                bank[k] = r["traj"]
    by: dict = {}
    for (lg, tok) in bank:
        by.setdefault(lg, []).append(tok)
    # the navtrain DBs live in splits/trainval AND flat in splits/ (fetch_navtrain_dbs.py) -- index both
    dbs = NS.index_dbs(db_root=os.path.dirname(TRAINVAL))
    rows, t0 = [], time.time()
    for lg, toks in sorted(by.items()):
        db = dbs.get(lg, "")
        if not os.path.exists(db):
            rows += [{"log": lg, "token": t, "error": "no local DB"} for t in toks]
            continue
        for sc in NS.build_scenarios_for_log(db, toks):
            tok = sc._initial_lidar_token
            rec = {"log": lg, "token": tok, "review7_teacher_dac_viol": held[(lg, tok)]["teacher"]["dac_viol"],
                   "speed": held[(lg, tok)].get("speed")}
            try:
                ego = sc.get_ego_state_at_iteration(0)
                ra = ego.rear_axle
                polys = ND.drivable_polygons(sc.map_api, ego)
                t20 = np.asarray(bank[(lg, tok)], np.float64)
                p8 = ND.to_navsim(t20)
                rec["raw20"] = bool(ND.violations(polys, ND.corners_global(t20, ra.x, ra.y, ra.heading)[None])[0])
                rec["interp8"] = bool(ND.violations(polys, ND.corners_global(ND.at_10hz(p8), ra.x, ra.y, ra.heading)[None])[0])
                rec["sim8"] = bool(ND.violations(polys, ND.simulated_corners(p8[None], ego))[0])
            except Exception as e:                                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            rows.append(rec)
    ok = [r for r in rows if "error" not in r]
    F = [r for r in ok if r["sim8"]]

    def cls(r):
        return "A_teacher_path" if r["raw20"] else "B_subsampling" if r["interp8"] else "C_navsim_tracker"
    cnt = Counter(cls(r) for r in F)
    agree = [bool(r["sim8"]) == bool(r["review7_teacher_dac_viol"]) for r in ok]
    res = {"what": "D1 on held-out navtrain (rank 0, banked 10 Hz targets): anatomy of the teacher-target DAC failures",
           "n_heldout_rank0": len(held), "n_with_banked_10hz_target": len(bank), "n_checked": len(ok),
           "n_errors": len(rows) - len(ok), "sim8_fail": len(F),
           "sim8_fail_rate_pct": round(100.0 * len(F) / max(1, len(ok)), 3),
           "raw20_fail_rate_pct": round(100.0 * np.mean([r["raw20"] for r in ok]), 3) if ok else None,
           "classes": {k: [v, round(100.0 * v / max(1, len(F)), 2)] for k, v in sorted(cnt.items())},
           "control_agreement_with_review7_sim_verdict_pct": round(100.0 * np.mean(agree), 2) if ok else None,
           "control_note": "same instrument (navsim_dac simulated footprint) run twice on the same target: a determinism "
                           "check of the pipeline + the bank join, not an independent validation (that is navsim_dac's "
                           "own 100 % agreement with NAVSIM, and D1-navtest's 778/778)",
           "seconds": round(time.time() - t0, 1), "rows": rows}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("ZZD1NT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

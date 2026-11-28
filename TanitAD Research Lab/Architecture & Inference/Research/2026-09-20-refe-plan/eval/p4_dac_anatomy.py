#!/usr/bin/env python3
"""P4 diagnostic D1 (REPORT-ONLY; written after the registered verdict, so EXPLORATORY): WHERE does the teacher's
drivable-area failure on navtest come from? Re-query (P5, F1) can only fix a failure the TEACHER makes; it cannot fix
one that the plan FORMAT or NAVSIM's TRACKER makes. Three nested checks per token, same polygons, same footprint
(refe/navsim_dac.py -- 100.00 % agreement with NAVSIM, MEASURED there):
  raw20   the teacher's OWN rollout poses (20 at 0.2 s, rear axle, = the bank's target) -- no interpolation, no tracker
  interp8 the 8-pose plan at 0.5 s, linearly interpolated to 0.1 s (navsim_dac.at_10hz) -- no tracker
  sim8    NAVSIM's pipeline: the 8-pose plan tracked by the vendored LQR + kinematic bicycle (navsim_dac.simulated_states)
Classes for a harness-DAC-fail token: A = raw20 fails (the teacher itself leaves the drivable area);
B = raw20 ok, interp8 fails (0.5 s subsampling); C = interp8 ok, sim8 fails (the tracker); D = all three pass (the
instrument disagrees with the harness -- counted, never hidden).
Controls that must read KNOWN values: (1) the HUMAN plan (W3 export) on every checked token: harness DAC = 100 % ->
sim8 violations must be 0; (2) a teacher-PASS sample: sim8 must read compliant (agreement with the harness).

    python eval/run_p4.py --script p4_dac_anatomy.py --log <log> -- --teacher-csv <csv> --rows-dir <dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, HERE)

EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
C1_KNOWN = os.path.join(PKG, "raw", "2026-10-04-navtest-final", "frame_control_navtest_full.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-csv", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pass", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import csv
    import navsim_dac as ND
    import navtrain_scenarios as NS
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    known = set(json.load(open(C1_KNOWN, encoding="utf-8")).get("over_bar", {}))
    T = {r["token"]: r for r in csv.DictReader(open(a.teacher_csv, encoding="utf-8")) if r.get("token") not in (None, "average")}
    prim = sorted(t for t in T if t not in known)
    dac_fail = sorted(t for t in prim if float(T[t]["drivable_area_compliance"]) == 0.0)
    nc_only = sorted(t for t in prim if float(T[t]["drivable_area_compliance"]) > 0 and float(T[t]["no_at_fault_collisions"]) == 0.0)
    passes = sorted(t for t in prim if float(T[t]["drivable_area_compliance"]) > 0 and float(T[t]["no_at_fault_collisions"]) > 0)
    ctrl = random.Random(20261006).sample(passes, a.n_pass)
    rows = {}
    for p in glob.glob(os.path.join(a.rows_dir, "rows_s*of*.jsonl")):
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            if "plan8" in r:
                rows[r["token"]] = r
    want = {t: "dac_fail" for t in dac_fail}
    want.update({t: "pass_control" for t in ctrl})
    by: dict = {}
    for t in want:
        by.setdefault(exp[t]["log_name"], []).append(t)
    out, t0 = [], time.time()
    for li, (lg, lt) in enumerate(sorted(by.items())):
        for sc in NS.build_scenarios_for_log(os.path.join(TEST_DB, f"{lg}.db"), lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            rec = {"token": tok, "log": lg, "set": want[tok], "v0": rows[tok]["v0"]}
            try:
                ego = sc.get_ego_state_at_iteration(0)
                polys = ND.drivable_polygons(sc.map_api, ego)
                ra = ego.rear_axle
                t20 = np.asarray(rows[tok]["traj20"], np.float64)
                p8 = np.asarray(rows[tok]["plan8"], np.float64)
                h8 = np.asarray(exp[tok]["human_future_poses"], np.float64)
                raw20 = ND.violations(polys, ND.corners_global(t20, ra.x, ra.y, ra.heading)[None])[0]
                interp8 = ND.violations(polys, ND.corners_global(ND.at_10hz(p8), ra.x, ra.y, ra.heading)[None])[0]
                sim = ND.violations(polys, ND.simulated_corners(np.stack([p8, h8]), ego))
                rec.update({"raw20": bool(raw20), "interp8": bool(interp8), "sim8": bool(sim[0]),
                            "human_sim8": bool(sim[1]),
                            "harness_dac": float(T[tok]["drivable_area_compliance"]),
                            "harness_nc": float(T[tok]["no_at_fault_collisions"])})
            except Exception as e:                                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            out.append(rec)
        if (li + 1) % 10 == 0:
            print(f"  [{li + 1}/{len(by)}] {len(out)} tokens {time.time() - t0:.0f}s", flush=True)
    ok = [r for r in out if "error" not in r]
    F = [r for r in ok if r["set"] == "dac_fail"]
    Pc = [r for r in ok if r["set"] == "pass_control"]

    def cls(r):
        if r["raw20"]:
            return "A_teacher_rollout_leaves_drivable_area"
        if r["interp8"]:
            return "B_0p5s_plan_subsampling"
        if r["sim8"]:
            return "C_navsim_tracker"
        return "D_instrument_disagrees_with_harness"
    from collections import Counter
    cnt = Counter(cls(r) for r in F)
    res = {"what": "D1 (exploratory): anatomy of the registered teacher's NAVSIM drivable-area failures on navtest",
           "n_dac_fail": len(dac_fail), "n_nc_only_fail": len(nc_only), "n_checked_fail": len(F),
           "n_errors": len(out) - len(ok),
           "classes": {k: [v, round(100.0 * v / max(1, len(F)), 2)] for k, v in sorted(cnt.items())},
           "fail_by_v0": {lab: dict(Counter(cls(r) for r in F if lo <= r["v0"] < hi)) for lab, (lo, hi) in
                          (("<1", (0, 1)), ("1-5", (1, 5)), ("5-10", (5, 10)), (">=10", (10, 99)))},
           "control_human_sim8_violations": {"fail_set": sum(r["human_sim8"] for r in F),
                                             "pass_set": sum(r["human_sim8"] for r in Pc),
                                             "n": len(F) + len(Pc), "must_read": 0},
           "control_pass_set": {"n": len(Pc), "sim8_violations": sum(r["sim8"] for r in Pc),
                                "raw20_violations": sum(r["raw20"] for r in Pc),
                                "agreement_with_harness_pct": round(100.0 * np.mean([not r["sim8"] for r in Pc]), 2)},
           "agreement_on_fail_set_pct": round(100.0 * np.mean([r["sim8"] for r in F]), 2) if F else None,
           "seconds": round(time.time() - t0, 1), "rows": out}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    print("ZZD1_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

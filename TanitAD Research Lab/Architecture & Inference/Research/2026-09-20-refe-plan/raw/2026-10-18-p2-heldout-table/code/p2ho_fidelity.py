#!/usr/bin/env python3
"""P2 held-out table, gate 2: FIDELITY of the scoring instrument.  Runs in the navsim-crun interpreter (needs nuPlan DBs, maps, NAVSIM v1.1).

CONTROL A (gating)   The held-out sets' banked labels (label_version 5, u400 fan, plans 'executed (A7 repair)') are the labels of the instrument the table is scored with.
                     Re-score the SAME 64 u400 proposals of K sets per log through this job's scoring path (executed_plans8 -> metric_cache_parts -> score_plans) and require
                     |round(x, 5) - label| <= 1e-5 on nc, dac, ep, ttc, c, ddc, pdms for EVERY proposal of EVERY processed set (the ZT5 DG1 bar; the labels are 5-decimal rounded).
                     NEGATIVE CONTROL (must read FAIL): the same proposals in REVERSED order against the unreversed labels -- a comparison that cannot fail proves nothing.
CONTROL B (reported) P2's OWN landed navtest proposals (table_p2_full.npz, sha256 33441082...) re-scored through this scorer on K navtest tokens of one log, against the landed
                     W3 harness numbers of the same table.  The two instruments are known to differ a little (validate_pdm_targets.json: PDMS exact on 94.73 %, mean |d| 0.0008,
                     argmax 100 %); this re-measures that on P2's proposals.  Not gating: it says how far a held-out table sits from the navtest instrument.

    navsim-crun python p2ho_fidelity.py --out fidelity.json [--logs-per-run 3 --sets-per-log 4 --nav-tokens 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SUBS = ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")
SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"
LABELS = "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm"
NAVTEST_TABLE = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
NAVTEST_DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"


def index_dbs():
    out = {}
    for s in ("mini", "test", "trainval"):
        for p in glob.glob(os.path.join(DB_ROOT, s, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], p.replace("\\", "/"))
    for p in glob.glob(os.path.join(DB_ROOT, "*.db")):
        out.setdefault(os.path.basename(p)[:-3], p.replace("\\", "/"))
    return out


def load_sets(logs, per_log):
    """first `per_log` rank-0 sets (sorted by token) of each requested log: traj/yaw + the banked label vectors."""
    want = {lg: [] for lg in logs}
    for fp in sorted(glob.glob(f"{SETS}/onpolicy_*.jsonl")):
        lab = {}
        for ln in open(f"{LABELS}/pdm_{os.path.basename(fp)}", encoding="utf-8"):
            if any(lg in ln[:200] for lg in logs):
                d = json.loads(ln)
                lab[tuple(d["key"])] = d
        for ln in open(fp, encoding="utf-8"):
            i = ln.index('"log_name": "') + 13
            lg = ln[i:ln.index('"', i)]
            if lg not in want:
                continue
            d = json.loads(ln)
            key = (d["log_name"], d["token"], int(d["step"]), int(d["rank"]))
            if int(d["rank"]) != 0 or key not in lab:
                continue
            want[lg].append({"key": key, "traj": np.asarray(d["traj"], np.float64), "yaw": np.asarray(d["yaw"], np.float64), "lab": lab[key]})
    for lg in want:
        want[lg] = sorted(want[lg], key=lambda r: r["key"])[:per_log]
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default="C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
    ap.add_argument("--logs", nargs="*", default=None, help="held-out logs for control A (default: the first --n-logs held-out logs that have a DB)")
    ap.add_argument("--n-logs", type=int, default=3)
    ap.add_argument("--sets-per-log", type=int, default=4)
    ap.add_argument("--nav-tokens", type=int, default=8)
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.tree, "refe"))
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    PT._setup()
    dbs = index_dbs()
    ho = [l.strip() for l in open("D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/heldout_logs.txt") if l.strip()]
    logs = a.logs or [l for l in ho if l in dbs][:a.n_logs]
    t0 = time.time()
    R = {"what": "P2 held-out table gate 2: scoring-instrument fidelity", "logs_A": logs, "tolerance_rounded_abs": 1e-5}
    sets = load_sets(logs, a.sets_per_log)
    worst, worst_unr, n_sets, n_prop = {k: 0.0 for k in SUBS}, {k: 0.0 for k in SUBS}, 0, 0
    neg_worst, per_set = 0.0, []
    for lg in logs:
        rows = sets[lg]
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(dbs[lg], sorted({r["key"][1] for r in rows}), history_rows=40, future_rows=100)}
        for r in rows:
            sc = scs.get(r["key"][1])
            if sc is None:
                continue
            props = np.concatenate([r["traj"], r["yaw"][..., None]], -1)
            P8 = PT.executed_plans8(props, repair=True)
            parts = PT.metric_cache_parts(sc)
            res = PT.score_plans(parts, P8)
            neg = PT.score_plans(parts, P8[::-1].copy())
            dset = {}
            for k in SUBS:
                lab = np.asarray(r["lab"][k], np.float64)
                x = np.asarray(res[k], np.float64)
                dr = float(np.abs(np.round(x, 5) - lab).max())
                du = float(np.abs(x - lab).max())
                worst[k], worst_unr[k] = max(worst[k], dr), max(worst_unr[k], du)
                dset[k] = dr
                neg_worst = max(neg_worst, float(np.abs(np.round(np.asarray(neg[k], np.float64), 5) - lab).max()))
            per_set.append({"key": list(r["key"]), "max_abs_rounded": max(dset.values())})
            n_sets += 1
            n_prop += 64
    R["A"] = {"n_sets": n_sets, "n_proposals": n_prop, "max_abs_diff_rounded5_per_head": worst, "max_abs_diff_unrounded_per_head": worst_unr,
              "worst_set": max(per_set, key=lambda x: x["max_abs_rounded"]) if per_set else None,
              "negative_control_reversed_order_max_abs": neg_worst, "negative_control_reads_FAIL": bool(neg_worst > 1e-3)}
    R["A"]["PASS"] = bool(n_sets > 0 and max(worst.values()) <= 1e-5 and R["A"]["negative_control_reads_FAIL"])
    R["A"]["seconds"] = round(time.time() - t0, 1)
    print("A", json.dumps({k: R["A"][k] for k in ("n_sets", "PASS", "negative_control_reversed_order_max_abs")}), flush=True)
    if not a.skip_b:
        T = np.load(NAVTEST_TABLE, allow_pickle=False)
        tlog = np.array([str(x) for x in T["log"]])
        cand = [l for l in sorted(set(tlog.tolist())) if os.path.exists(f"{NAVTEST_DB}/{l}.db")]
        lg = cand[0]
        ii = np.flatnonzero(tlog == lg)[:a.nav_tokens]
        toks = [str(T["token"][i]) for i in ii]
        scs = {s.scenario_name: s for s in NS.build_scenarios_for_log(f"{NAVTEST_DB}/{lg}.db", sorted(toks), history_rows=40, future_rows=100)}
        dp, dsub, same_pick, n_ex, n_tot, arg_same = [], [], 0, 0, 0, 0
        for i, t in zip(ii, toks):
            sc = scs.get(t)
            if sc is None:
                continue
            res = PT.score_plans(PT.metric_cache_parts(sc), T["proposals"][i])
            x = np.asarray(res["pdms"], np.float64)
            y = T["pdms"][i].astype(np.float64)
            dp.append(np.abs(x - y))
            xs = np.stack([np.asarray(res[k], np.float64) for k in SUBS[:6]], -1)
            dsub.append(np.abs(xs - T["sub"][i].astype(np.float64)))
            n_ex += int((np.abs(x - y) <= 1e-4).sum())
            n_tot += 64
            arg_same += int(int(np.argmax(x)) == int(np.argmax(y)))
        dp = np.stack(dp) if dp else np.zeros((0, 64))
        R["B"] = {"navtest_log": lg, "n_tokens": int(dp.shape[0]), "n_plans": int(n_tot), "table_sha256_expected": "33441082b7044c951e3cef5fb898cac50797039aed93560b2712e3ff4c34906e",
                  "pdms_exact_within_1e-4_pct": round(100.0 * n_ex / max(n_tot, 1), 3), "pdms_mean_abs_diff": float(dp.mean()) if dp.size else None,
                  "pdms_max_abs_diff": float(dp.max()) if dp.size else None, "argmax_pdms_same_n": int(arg_same),
                  "sub_max_abs_diff_per_head": [float(np.stack(dsub)[..., j].max()) for j in range(6)] if dsub else None,
                  "banked_reference": "validate_pdm_targets.json: 300 tokens / 19,500 plans, PDMS exact 94.733 %, mean |d| 0.0008, argmax same 100 %",
                  "gating": False}
        print("B", json.dumps(R["B"]), flush=True)
    R["PASS"] = bool(R["A"]["PASS"])
    R["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"ZZP2HOFID_{'PASS' if R['PASS'] else 'FAIL'}")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

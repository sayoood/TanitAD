#!/usr/bin/env python3
"""SFT-2 labels: NAVSIM-faithful DRIVING DIRECTION and map LANE KEEPING for the EXISTING on-policy sets (pod, CPU only).

The sets keep their proposals and every existing label; this writes SIDE FILES keyed exactly as train.OnPolicyBank keys a
set (log_name, token, step, rank) plus its ckpt_step, so a label can only ever meet the proposals it was computed for:
    {"kind": "lane_labels", "key": [log, token, step, rank], "ckpt_step": c, "navsim_ddc": [M], "lane_keep": [M],
     "route_changed": bool, "ddc_teacher": [M]}
  * navsim_ddc   NAVSIM v1.1's DDC (1.0 / 0.5 / 0.0) on NAVSIM's own simulation of each proposal, on-route lanes = the
                 interior lanes of the scenario route AFTER NAVSIM's PDM route correction at the sample's ego state
                 (refe/navsim_lane.py; MEASURED 100.00 % agreement with NAVSIM's scorer on 13,000 navtest trajectories,
                 eval/validate_navsim_ddc.py)
  * lane_keep    1.0 = off the containing lane's centreline by > 1.0 m for >= 1.0 s outside junctions (LANE-1's lk10)
  * ddc_teacher  the label the scorer was trained on (DriveRL wrong-way, `ddc.violation` -> 1 - v), for the agreement table
Augmented (rank-1) rows keep their route ROADBLOCKS (lane_rank / goal_horizon_s change the lane or the goal distance,
route_lane_rank_patch.py), so the on-route lane set is the scenario's.
    python refe/onpolicy_relabel_lane.py --sets <dir> --out <dir> [--workers 8] [--limit N] [--files 'onpolicy_r0_*']
Exit markers: ZZRELABEL_DONE <sets> <failed>.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

_S: dict = {}


def _init():
    import navtrain_scenarios as NS
    _S["NS"] = NS
    _S["dbs"] = NS.index_dbs()


def relabel_log(job):
    """(log_name, [set rows]) -> [side-file lines], built from ONE scenario build for the log."""
    import navsim_dac as ND
    import navsim_lane as NL
    log, rows = job
    out, failed = [], 0
    try:
        if not _S:
            _init()
        scs = {sc.scenario_name: sc for sc in _S["NS"].build_scenarios_for_log(_S["dbs"][log], sorted({r["token"] for r in rows}))}
    except Exception as e:                                                   # noqa: BLE001
        return [], len(rows), f"{log}: scenario build {type(e).__name__}: {str(e)[:120]}"
    for r in rows:
        try:
            sc = scs[r["token"]]
            step = int(r["step"])
            ego = sc.get_ego_state_at_iteration(step)
            xy = np.asarray(r["traj"], np.float64)
            yw = np.asarray(r["yaw"], np.float64)
            props = np.concatenate([xy, yw[..., None]], -1)                       # [M, 20, 3], REFe grid (5 Hz)
            if r.get("repair") == "last_heading_hold":       # v5: the labels belong to the EXECUTED (repaired) plans
                from planner import repair_last_heading
                props = repair_last_heading(props)
            p8 = np.stack([ND.to_navsim(p) for p in props])
            sim = ND.simulated_states(p8, ego)
            ids0 = list(sc.get_route_roadblock_ids())
            ids, diag = NL.corrected_route_ids(sc.map_api, ego.rear_axle, ids0)
            ddc, lk = NL.navsim_ddc_and_lane(sim, sc.map_api, ego, ids)
            tdd = [1.0 - min(max(float(t.get("ddc.violation") or 0.0), 0.0), 1.0) for t in r["targets"]]
            out.append(json.dumps({"kind": "lane_labels", "key": [r["log_name"], r.get("token", ""), step, int(r.get("rank", 0))],
                                   "ckpt_step": int(r["ckpt_step"]), "label_version": int(r.get("label_version", 1)),
                                   "navsim_ddc": [float(x) for x in ddc], "lane_keep": [float(x) for x in lk],
                                   "route_changed": bool(diag.get("changed")), "repair": r.get("repair"), "ddc_teacher": tdd}) + "\n")
        except Exception:                                                    # noqa: BLE001
            failed += 1
            if failed <= 2:
                print(f"  {r.get('token')}: {traceback.format_exc()[-400:]}", flush=True)
    return out, failed, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default="onpolicy_*.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="sets per input file (smoke)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.sets, a.files)))
    n_sets = n_fail = 0
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for fpath in files:
            dst = os.path.join(a.out, "lane_" + os.path.basename(fpath))
            done = set()
            if os.path.exists(dst):
                for ln in open(dst, encoding="utf-8"):
                    try:
                        d = json.loads(ln)
                        done.add((tuple(d["key"]), d["ckpt_step"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
            CH = 4000                                         # stream: a set file is ~1.4 GB of JSON

            def slim(r):
                return {"log_name": r["log_name"], "token": r.get("token", ""), "step": r["step"], "rank": r.get("rank", 0),
                        "ckpt_step": r["ckpt_step"], "label_version": r.get("label_version", 1), "repair": r.get("repair"),
                        "traj": r["traj"], "yaw": r["yaw"], "targets": [{"ddc.violation": t.get("ddc.violation")} for t in r["targets"]]}

            def flush(by, f):
                nonlocal_counts = [0, 0]
                for lines, failed, err in pool.imap_unordered(relabel_log, list(by.items())):
                    if err:
                        print("  " + err, flush=True)
                    f.writelines(lines)
                    f.flush()
                    nonlocal_counts[0] += len(lines)
                    nonlocal_counts[1] += failed
                return nonlocal_counts

            by: dict = {}
            k = 0
            with open(dst, "a", encoding="utf-8") as f:
                for ln in open(fpath, encoding="utf-8"):
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if r.get("kind") != "onpolicy_set":
                        continue
                    key = ((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"]))
                    if key in done:
                        continue
                    by.setdefault(r["log_name"], []).append(slim(r))
                    k += 1
                    if k % CH == 0:
                        c = flush(by, f)
                        n_sets += c[0]
                        n_fail += c[1]
                        by = {}
                        print(f"  {os.path.basename(fpath)}: {k:,} read, {n_sets:,} written, {n_fail} failed, "
                              f"{(time.time() - t0) / max(n_sets, 1):.3f} s/set", flush=True)
                    if a.limit and k >= a.limit:
                        break
                if by:
                    c = flush(by, f)
                    n_sets += c[0]
                    n_fail += c[1]
            el = time.time() - t0
            print(f"{os.path.basename(fpath)}: total {n_sets:,} sets, {n_fail} failed, {el / max(n_sets, 1):.3f} s/set", flush=True)
    print(f"ZZRELABEL_DONE {n_sets} {n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

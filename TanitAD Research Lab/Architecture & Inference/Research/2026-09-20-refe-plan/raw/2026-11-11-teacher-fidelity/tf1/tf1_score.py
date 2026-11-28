#!/usr/bin/env python3
"""TF-1 (teacher fidelity, first measurement): FULL tracked NAVSIM v1.1 PDMS of the Tc20s tracker-aware targets.

Population: the 1,018 held-out rank-0 NAVTRAIN teacher targets of D1 / P5 / P5-S / E2 (11 logs). Not navtest.
Per (log, token), ONE NAVSIM PDMScorer call (vendored simulator + scorer, navsim @ 3e8291b) on
  [PDM-Closed,                                      index 0, EP normaliser (pairwise, as pdm_score.py scores [pdm, pred])
   T_tracked       to_navsim(t20)          -> LQR   identity control: must equal E2's teacher_tracked bit-for-bit
   Tc20s_tracked   to_navsim(Tc20s cubic)  -> LQR   PRIMARY arm (P5-S label)
   Tc20_tracked    to_navsim(Tc20 linear)  -> LQR   descriptive (P5 label)
   Tw_tracked      to_navsim(Tw cubic)     -> LQR   deliberate-regression arm: must score BELOW T
   H_tracked       logged 8 poses          -> LQR   identity control: must equal E2's human_tracked
   T_transparent   t20 as executed (E2's builder)   identity control: must equal E2's teacher_transparent
   Tc20s_transp    Tc20s as executed (E2's builder) descriptive: the DRAWN label as a path (the interp8 cost, in PDMS)]
Labels come from the banked P5-S npz (`labels20_t20_tc20_tw`: [t20, Tc20(s), Tw] per row): nothing is re-derived.
Per arm: nc, dac, ddc (weight 0 in v1.1, reported), ttc, ep, c, mult, pdms. PDMS = NC*DAC*(5 EP + 5 TTC + 2 C)/12.
Usage: python tf1_score.py --refe <dir> --cubic <p5s_cubic.npz> --linear <p5s_linear.npz> --out <jsonl> --shard i/N
       [--limit n] [--e2score <dir holding e2_score.py>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("REFE_SIM_HZ", "10")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

DB_ROOT = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
TRACKED = ("T_tracked", "Tc20s_tracked", "Tc20_tracked", "Tw_tracked", "H_tracked")
TRANSP = ("T_transparent", "Tc20s_transparent")


def score(parts, plans8, executed):
    """E2's score_states with DDC added: [PDM-Closed, *plans8 (LQR-simulated), *executed (given as executed)]."""
    import navsim_pdm_targets as PT
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    S = PT._setup()
    pdm_traj, ego, obs, centerline, lanes, dam = parts
    inp = [get_trajectory_as_array(pdm_traj, S["samp"], ego.time_point)]
    for p8 in plans8:
        it = transform_trajectory(Trajectory(np.asarray(p8, np.float32), S["samp8"]), ego)
        inp.append(get_trajectory_as_array(it, S["samp"], ego.time_point))
    sim = S["sim"].simulate_proposals(np.stack(inp, 0), ego)
    if executed:
        sim = np.concatenate([sim, np.stack(executed, 0)], 0)
    sc = S["scorer"]
    sc.score_proposals(sim, obs, centerline, lanes, dam)
    mult = sc._multi_metrics.prod(axis=0)
    raw = sc._progress_raw * mult
    ttc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.TTC])
    com = np.asarray(sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE])
    ddc = np.asarray(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION])
    out = []
    for j in range(1, sim.shape[0]):
        m = max(raw[0], raw[j])
        ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
        pdms = mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0
        out.append({"nc": float(sc._multi_metrics[MultiMetricIndex.NO_COLLISION][j]),
                    "dac": float(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA][j]),
                    "ddc": float(ddc[j]), "mult": float(mult[j]), "ep": float(ep), "ttc": float(ttc[j]),
                    "c": float(com[j]), "pdms": float(pdms)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--e2score", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       "p6base", "teacher_review", "e2e1"))
    ap.add_argument("--cubic", required=True)
    ap.add_argument("--linear", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.refe)
    sys.path.insert(0, a.e2score)
    import navsim_dac as ND
    import navtrain_scenarios as NS
    import navsim_pdm_targets as PT
    from navsim_pdm.pdm_array_representation import ego_state_to_state_array
    from e2_score import transparent_states, human_rel          # E2's builders, unchanged (identity control)
    zc, zl = np.load(a.cubic), np.load(a.linear)
    if not (np.array_equal(zc["logs"], zl["logs"]) and np.array_equal(zc["tokens"], zl["tokens"])):
        raise SystemExit("ZZTF1_FAIL npz row order differs between cubic and linear")
    if not np.array_equal(zc["labels20_t20_tc20_tw"][:, 0], zl["labels20_t20_tc20_tw"][:, 0]):
        raise SystemExit("ZZTF1_FAIL t20 differs between cubic and linear npz")
    lab = {}
    for i, (lg, tok) in enumerate(zip(zc["logs"].tolist(), zc["tokens"].tolist())):
        lab[(lg, tok)] = (zc["labels20_t20_tc20_tw"][i, 0], zc["labels20_t20_tc20_tw"][i, 1],
                          zl["labels20_t20_tc20_tw"][i, 1], zc["labels20_t20_tc20_tw"][i, 2])
    si, sn = (int(x) for x in a.shard.split("/"))
    by: dict = {}
    for lg, tok in sorted(lab)[si::sn]:                         # shard over SAMPLES (logs are uneven: 18..241)
        by.setdefault(lg, []).append(tok)
    logs = sorted(by)
    dbs = NS.index_dbs(db_root=DB_ROOT)
    n_done, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg in logs:
            toks = by[lg]
            if not os.path.exists(dbs.get(lg, "")):
                for t in toks:
                    fo.write(json.dumps({"log": lg, "token": t, "error": "no local DB"}) + "\n")
                continue
            for sc in NS.build_scenarios_for_log(dbs[lg], toks):
                tok = sc._initial_lidar_token
                rec = {"log": lg, "token": tok}
                try:
                    t20, tc20s, tc20, tw = (np.asarray(x, np.float64) for x in lab[(lg, tok)])
                    parts = PT.metric_cache_parts(sc)
                    ego = parts[1]
                    s0 = ego_state_to_state_array(ego)
                    h8 = human_rel(sc, ego)[4::5][:8]
                    res = score(parts, [ND.to_navsim(t20), ND.to_navsim(tc20s), ND.to_navsim(tc20), ND.to_navsim(tw), h8],
                                [transparent_states(t20, ego, s0), transparent_states(tc20s, ego, s0)])
                    for nm, r in zip(TRACKED + TRANSP, res):
                        rec[nm] = r
                except Exception as e:                                    # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_done += 1
                if a.limit and n_done >= a.limit:
                    print("ZZTF1_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1))
                    return 0
    print("ZZTF1_SHARD_DONE", a.shard, n_done, round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CPU only, NO model. Lane discipline of REFe's final model on navtest, per hypothesis, from NAVSIM's own geometry.

WHY (PI, 2026-10-04, after watching the video): "the planner fan is not always following the lane, and even if the fan is
within the target lane, sometimes the selected trajectory is leaving the lane" + "check the teacher itself: is it
considering drivable areas etc.". NAVSIM v1.1's PDMS cannot see either effect: `driving_direction_weight = 0.0`
(pdm_scorer.py:42), DAC counts ANY roadblock as drivable (including the oncoming one), and there is no lane-keeping term.

WHAT. For every navtest token, the 64 hypotheses of the final model (proptable/navtest_final/proposals.npz, the table the
video draws), the human's own 4 s future, and PDM-Closed (the metric cache's privileged reference) are simulated and scored
by NAVSIM v1.1's OWN PDMSimulator + PDMScorer on the token's metric cache -- one batch of 66 -- and these are read per
trajectory from the scorer's internals (41 steps, 0.1 s, t = 0 included):
  * nc, dac                 the multiplicative metrics
  * ddc                     NAVSIM's driving-direction compliance (oncoming progress in 1 s: <2 m 1.0, <6 m 0.5, else 0)
  * onc_n                   steps 1..40 whose CENTRE is in no on-route lane (EgoAreaIndex.ONCOMING_TRAFFIC)
  * strad_s                 the longest run (s) of steps whose footprint straddles lanes (EgoAreaIndex.MULTIPLE_LANES)
                            with the centre OUTSIDE every intersection polygon -- a lane-keeping proxy
  * dev_max                 max lateral distance (m) of the centre from the metric cache's route centerline, steps 1..40,
                            centre outside intersections (nan if every step is inside one)
  * pdms                    the harness score REPRODUCED pairwise: each trajectory scored as if submitted alone, i.e. EP
                            normalised against PDM-Closed only (pdm_score.py scores [pdm, pred]); NC, DAC, TTC, C are
                            per-trajectory in the scorer and do not depend on the batch.
CONTROLS (written into every row and summarised by lane_analyze.py):
  * pick_ok    argmax of the planner's navsim_v1 aggregate over the stored logits == the stored pick
  * csv_pdms   the harness's own score of the submitted pick (refe_navtest_final.csv) -- must equal pdms[pick]
Tokens are processed in a FIXED SHUFFLED ORDER (seed 20261004), so any prefix of the output is a uniform random sample.
    python lane_census.py [--workers 3] [--limit N]      (re-launches itself in the navsim v1.1 venv)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import lzma
import os
import pickle
import random
import subprocess
import sys
import time
import traceback

import numpy as np

DRV = os.environ.get("REFE_DRIVE", "E:")
HERE = os.path.dirname(os.path.abspath(__file__))
NV_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = f"{DRV}/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
CACHE = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DATA = f"{DRV}/Projects/TanitAD/data/refe_navtest"
PROPS = f"{DATA}/proptable/navtest_final/proposals.npz"
CSV = f"{DATA}/score/refe_navtest_final/refe_navtest_final.csv"
OUT = os.path.join(HERE, "lane_census.jsonl")
SEED = 20261004


def agg_v1(logits):
    p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12.0


def longest_run(mask_row):
    best = cur = 0
    for v in mask_row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def point_polyline_dist(pts, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    ap = pts[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - q, axis=-1).min(1)


_W = {}


def _init():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    samp = TrajectorySampling(num_poses=40, interval_length=0.1)
    _W["samp"] = samp
    _W["sim"] = PDMSimulator(samp)
    _W["scorer"] = PDMScorer(samp, PDMScorerConfig())
    P = np.load(PROPS)
    _W["P"] = {k: P[k] for k in ("token", "proposals", "logits", "pick")}
    _W["idx"] = {str(t): i for i, t in enumerate(_W["P"]["token"])}


def score_token(args):
    tok, log, hum = args
    try:
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            EgoAreaIndex, MultiMetricIndex, WeightedMetricIndex, BBCoordsIndex)
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        if not _W:
            _init()
        i = _W["idx"][tok]
        props = _W["P"]["proposals"][i].astype(np.float64)              # (64, 8, 3) NAVSIM frame, 2 Hz
        logits = _W["P"]["logits"][i]
        pick = int(_W["P"]["pick"][i])
        hum = np.asarray(hum, dtype=np.float64)
        assert props.shape == (64, 8, 3) and hum.shape == (8, 3), (props.shape, hum.shape)
        mc = pickle.load(lzma.open(f"{CACHE}/{log}/unknown/{tok}/metric_cache.pkl", "rb"))
        ego = mc.ego_state
        samp8 = TrajectorySampling(num_poses=8, interval_length=0.5)
        states = [get_trajectory_as_array(mc.trajectory, _W["samp"], ego.time_point)]
        for traj in list(props) + [hum]:
            it = transform_trajectory(Trajectory(traj.astype(np.float32), samp8), ego)
            states.append(get_trajectory_as_array(it, _W["samp"], ego.time_point))
        states = np.stack(states, 0)
        sim = _W["sim"].simulate_proposals(states, ego)
        sc = _W["scorer"]
        sc.score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        n = sim.shape[0]
        mult = sc._multi_metrics.prod(axis=0)
        raw = sc._progress_raw * mult
        ttc = sc._weighted_metrics[WeightedMetricIndex.TTC]
        com = sc._weighted_metrics[WeightedMetricIndex.COMFORTABLE]
        pdms = np.zeros(n)
        for j in range(n):
            m = max(raw[0], raw[j])
            ep = raw[j] / m if m > 5.0 else (1.0 if mult[j] != 0 else 0.0)
            pdms[j] = mult[j] * (5 * ep + 5 * ttc[j] + 2 * com[j]) / 12.0
        areas = sc._ego_areas                                            # (n, 41, 3)
        coords = sc._ego_coords                                          # (n, 41, 5, 2)
        dam = mc.drivable_area_map
        inter = dam.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
        if len(inter):
            inp = dam.points_in_polygons(coords)                         # (n_poly, n, 41, 5)
            c_int = inp[inter][..., BBCoordsIndex.CENTER].any(axis=0)    # (n, 41)
        else:
            c_int = np.zeros(areas.shape[:2], dtype=bool)
        strad = areas[..., EgoAreaIndex.MULTIPLE_LANES] & ~c_int
        onc = areas[..., EgoAreaIndex.ONCOMING_TRAFFIC]
        cl = mc.centerline._states_se2_array[:, :2]
        ctr = coords[:, :, BBCoordsIndex.CENTER]                         # (n, 41, 2)
        dev = point_polyline_dist(ctr.reshape(-1, 2), cl).reshape(n, -1)
        dev_m = np.where(c_int, np.nan, dev)[:, 1:]
        with np.errstate(all="ignore"):
            dev_max = np.where(np.isnan(dev_m).all(1), np.nan, np.nanmax(np.nan_to_num(dev_m, nan=-1.0), 1))
        agg = agg_v1(logits)
        r3 = lambda a: [None if not np.isfinite(x) else round(float(x), 4) for x in a]   # noqa: E731
        return {
            "token": tok, "log": log, "pick": pick, "pick_ok": int(np.argmax(agg)) == pick,
            "order": "0=pdm_closed,1..64=hypotheses,65=human",
            "nc": r3(sc._multi_metrics[MultiMetricIndex.NO_COLLISION]),
            "dac": r3(sc._multi_metrics[MultiMetricIndex.DRIVABLE_AREA]),
            "ddc": r3(sc._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION]),
            "ttc": r3(ttc), "c": r3(com), "pdms": r3(pdms),
            "onc_n": [int(x) for x in onc[:, 1:].sum(1)], "onc_t0": bool(onc[0, 0]),
            "strad_s": [round(longest_run(r) * 0.1, 1) for r in strad[:, 1:]], "strad_t0": bool(strad[0, 0]),
            "int_frac": r3(c_int[:, 1:].mean(1)),
            "dev_max": r3(dev_max), "dev_t0": round(float(dev[0, 0]), 3),
            "agg": r3(agg), "p_ddc": r3(1.0 / (1.0 + np.exp(-logits[:, 5].astype(np.float64)))),
        }
    except Exception as e:                                               # noqa: BLE001
        return {"token": tok, "log": log, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def main() -> int:
    if os.environ.get("REFE_LANE_CHILD") != "1":
        env = dict(os.environ, REFE_LANE_CHILD="1", PYTHONPATH=V11, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUPLAN_MAPS_ROOT=f"{DRV}/Archive/devbox-C/navsim/data/maps",
                   NUPLAN_MAP_VERSION="nuplan-maps-v1.0")
        return subprocess.call([NV_PY, os.path.abspath(__file__), *sys.argv[1:]], env=env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    a = ap.parse_args()
    E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    csv_s = {r["token"]: float(r["score"]) for r in csv.DictReader(open(CSV)) if r.get("valid") == "True"}
    toks = sorted(E)
    random.Random(SEED).shuffle(toks)
    done, keep = set(), []
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r and a.retry_errors:
                continue
            done.add(r["token"])
            keep.append(line)
        if a.retry_errors:
            with open(OUT, "w", encoding="utf-8") as f:
                f.writelines(keep)
    todo = [t for t in toks if t not in done]
    if a.limit:
        todo = todo[: max(a.limit - len(done), 0)]
    jobs = [(t, E[t]["log_name"], E[t]["human_future_poses"]) for t in todo]
    print(f"census: {len(done)} done, {len(jobs)} to do, {a.workers} workers", flush=True)
    t0 = time.time()
    import multiprocessing as mp
    with open(OUT, "a", encoding="utf-8") as f, mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap(score_token, jobs, chunksize=4), 1):
            if "error" not in r:
                r["csv_pdms"] = csv_s.get(r["token"])
            f.write(json.dumps(r) + "\n")
            if k % 200 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{k}/{len(jobs)}  {el / k:.2f} s/token  eta {(len(jobs) - k) * el / k / 60:.0f} min", flush=True)
    print("ZZLANE_CENSUS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
